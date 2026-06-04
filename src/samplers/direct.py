"""Direct posterior sampler for VP diffusion models."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from src.forward_model import ForwardModel
from src.sde import VPSDE, alpha_bar_from_times

__all__ = ["direct_posterior_sample"]


def _posterior_weights(
    atoms: Tensor,
    y: Tensor,
    forward_model: ForwardModel,
    noise_cov: float | Tensor,
) -> Tensor:
    r"""Normalised posterior weights :math:`\omega_n \propto p(y \mid A(\xi_n), \Sigma)`.

    Args:
        atoms: ``(N, D)`` prior atoms.
        y: ``(m,)`` observation.
        forward_model: measurement operator.
        noise_cov: observation noise covariance — scalar (isotropic),
            ``(m,)`` diagonal, or ``(m, m)`` full matrix.

    Returns:
        ``(N,)`` normalised weights.

    Raises:
        RuntimeError: if weights are non-finite after normalisation.
    """
    n = atoms.shape[0]
    meas = forward_model.fn(atoms).reshape(n, -1)  # (N, m)
    resid = y.reshape(1, -1) - meas  # (N, m)

    nc = torch.as_tensor(noise_cov, dtype=atoms.dtype, device=atoms.device)
    if nc.ndim == 2:
        log_w = -0.5 * (resid * torch.linalg.solve(nc, resid.T).T).sum(-1)
    elif nc.ndim == 1:
        log_w = -0.5 * (resid.pow(2) / nc.reshape(1, -1)).sum(-1)
    else:
        log_w = -0.5 * resid.pow(2).sum(-1) / nc

    omega = torch.softmax(log_w, dim=0)
    if not torch.isfinite(omega).all():
        raise RuntimeError(
            "Posterior weights are non-finite: the observation is incompatible "
            "with all atoms under the given noise covariance."
        )
    return omega


def _draw_samples(
    atoms: Tensor,
    omega: Tensor,
    times: Tensor,
    n_samples: int,
    sde: VPSDE,
    generator: torch.Generator | None,
) -> Tensor:
    r"""Draw i.i.d. marginal samples :math:`p(x_t \mid y)` at every time in ``times``.

    Samples component indices once from the posterior mixture; at each
    time step applies fresh Gaussian noise.

    Args:
        atoms: ``(N, D)`` prior atoms.
        omega: ``(N,)`` posterior mixture weights.
        times: ``(T,)`` decreasing time grid.
        n_samples: number of samples per time step.
        sde: VP-SDE providing the noise schedule.
        generator: optional RNG for reproducibility.

    Returns:
        ``(T, M, D)`` posterior samples.
    """
    t_steps = times.shape[0]
    d = atoms.shape[1]
    idx = torch.multinomial(omega, n_samples, replacement=True, generator=generator)
    x_i = atoms[idx]  # (M, D)
    alpha_bars = alpha_bar_from_times(times, sde.beta_min, sde.beta_max)  # (T,)
    s = alpha_bars.sqrt()[:, None, None]  # (T, 1, 1)
    sigma = (1.0 - alpha_bars).sqrt()[:, None, None]  # (T, 1, 1)
    z = torch.randn(t_steps, n_samples, d, dtype=atoms.dtype, device=atoms.device, generator=generator)
    return s * x_i[None] + sigma * z  # (T, M, D)


def direct_posterior_sample(
    y: Tensor,
    sde: VPSDE,
    forward_model: ForwardModel,
    noise_cov: float | Tensor,
    prior_sampler: Callable[[int], Tensor],
    dt: float,
    n_samples: int,
    *,
    n_atoms: int = 2000,
    t_hi: float = 0.99,
    t_lo: float = 0.01,
    seed: int | None = None,
) -> Tensor:
    r"""Draw exact marginal posterior samples :math:`p(x_t \mid y)` at a grid of times.

    Treats the finite atom set as an empirical prior and marginalises
    analytically:

    .. math::

        p(x_t \mid y) = \sum_{n=1}^{N} \omega_n
            \,\mathcal{N}(x_t;\,\sqrt{\bar\alpha(t)}\,\xi_n,\,(1-\bar\alpha(t))\,I)

    where :math:`\omega_n \propto p(y \mid A(\xi_n), \Sigma)` and the atoms
    :math:`\{\xi_n\}_{n=1}^{N}` are i.i.d. draws from ``prior_sampler``.
    Component indices are drawn once from :math:`\mathrm{Categorical}(\omega)`;
    fresh Gaussian noise is added independently at each :math:`t`.

    The time grid runs from ``t_hi`` to ``t_lo`` inclusive with
    ``T = max(1, round((t_hi - t_lo) / dt) + 1)`` steps.

    Note:
        Atoms must be 2-D: ``prior_sampler`` must return ``(n_atoms, D)`` even
        for :math:`D = 1`.  The output is always ``(T, M, D)``.

    Args:
        y: ``(m,)`` observation tensor.
        sde: VP-SDE that supplies the noise schedule via ``beta_min`` and
            ``beta_max``.
        forward_model: measurement operator :math:`A`.
        noise_cov: observation noise covariance :math:`\Sigma` — a scalar
            (isotropic), ``(m,)`` diagonal, or ``(m, m)`` full matrix.
        prior_sampler: callable ``(n: int) -> Tensor`` returning ``(n, D)``
            i.i.d. samples from the prior.
        dt: time-grid step size.
        n_samples: number of posterior samples :math:`M` per time step.
        n_atoms: number of atoms :math:`N` drawn from ``prior_sampler`` to
            approximate the prior.
        t_hi: upper end of the time grid (high-noise end).
        t_lo: lower end of the time grid (low-noise end).
        seed: optional RNG seed for reproducibility; ``None`` uses the global
            RNG.

    Returns:
        ``(T, M, D)`` posterior samples at each time in the grid.

    Raises:
        ValueError: if ``t_hi <= t_lo``, ``n_atoms < 1``, ``n_samples < 1``,
            or ``prior_sampler`` does not return a 2-D tensor.
        RuntimeError: if posterior weights are non-finite (the observation is
            incompatible with all atoms).
    """
    if t_hi <= t_lo:
        raise ValueError(f"t_hi must be greater than t_lo, got {t_hi} <= {t_lo}")
    if n_atoms < 1:
        raise ValueError(f"n_atoms must be at least 1, got {n_atoms}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}")

    atoms = prior_sampler(n_atoms)
    if atoms.ndim != 2:
        raise ValueError(
            f"prior_sampler must return a 2-D tensor (n_atoms, D), got shape {tuple(atoms.shape)}"
        )

    t_steps = max(1, round((t_hi - t_lo) / dt) + 1)
    times = torch.linspace(t_hi, t_lo, t_steps, dtype=atoms.dtype, device=atoms.device)

    y_t = torch.as_tensor(y, dtype=atoms.dtype, device=atoms.device)
    omega = _posterior_weights(atoms, y_t, forward_model, noise_cov)

    generator = torch.Generator(device=atoms.device).manual_seed(seed) if seed is not None else None
    return _draw_samples(atoms, omega, times, n_samples, sde, generator)
