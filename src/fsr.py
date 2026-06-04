"""Public ``run_fsr`` entry point: posterior sampling from an empirical prior.

The single function in this module wraps :class:`src.scores.FSR` (the
finite-sample reference likelihood approximation) and
:func:`src.samplers.em.em_step` (reverse-time Euler-Maruyama integration of
the VP-SDE).  Bring your own prior samples and a forward model; you get a
batch of posterior samples back.
"""

from __future__ import annotations

import numpy as np
import torch

from src.forward_model import ForwardModel
from src.samplers.em import em_step
from src.scores.fsr import FSR
from src.sde import VPSDE
from src.weights import prior_terms

__all__ = ["run_fsr"]


def run_fsr(
    atoms: torch.Tensor,
    y: torch.Tensor,
    forward_model: ForwardModel,
    noise_scale: float,
    *,
    sde: VPSDE | None = None,
    num_steps: int = 1000,
    num_particles: int = 256,
    t_min: float = 1e-3,
    t_max: float = 1.0,
    device: str | torch.device = "cpu",
    seed: int | None = None,
) -> torch.Tensor:
    r"""Draw posterior samples via the finite-sample reference (FSR) approximation.

    The empirical measure on ``atoms`` plays the role of both prior and
    denoiser, so the same ``(N, d)`` tensor drives the prior marginal score
    and the FSR likelihood score.  Reverse-time integration uses Euler-
    Maruyama from ``t_max`` down to ``t_min`` on a uniform grid of
    ``num_steps`` steps.

    Args:
        atoms (torch.Tensor): ``(N, d)`` empirical prior samples.
        y (torch.Tensor): ``(m,)`` observation.  ``forward_model.fn(atoms)``
            must have trailing dim ``m`` for the shape check to pass.
        forward_model (ForwardModel): measurement operator
            :math:`\mathcal A`.  Can be linear or nonlinear.
        noise_scale (float): measurement-noise standard deviation
            :math:`\sigma_n > 0`.
        sde (VPSDE | None): VP forward schedule.  Defaults to a fresh
            :class:`VPSDE` with the library defaults.
        num_steps (int): number of reverse-time EM steps.
        num_particles (int): number of independent posterior samples to
            return.
        t_min (float): smallest diffusion time reached by the reverse
            integration.  Must be ``> 0``.
        t_max (float): starting time of the reverse integration.  Must be
            ``> t_min``.
        device (str | torch.device): device for the returned samples.
        seed (int | None): optional torch RNG seed for reproducibility.

    Returns:
        torch.Tensor: ``(num_particles, d)`` posterior samples at ``t_min``.

    Raises:
        ValueError: on shape mismatches, non-positive ``noise_scale``, or
            inconsistent ``t_min`` / ``t_max``.
    """
    if atoms.ndim != 2:
        raise ValueError(
            f"atoms must have shape (N, d); got {tuple(atoms.shape)}.",
        )
    if y.ndim != 1:
        raise ValueError(f"y must be a 1-D tensor of shape (m,); got {tuple(y.shape)}.")
    if noise_scale <= 0:
        raise ValueError(f"noise_scale must be positive; got {noise_scale}.")
    if not (0 < t_min < t_max):
        raise ValueError(
            f"Require 0 < t_min < t_max; got t_min={t_min}, t_max={t_max}.",
        )
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1; got {num_steps}.")
    if num_particles < 1:
        raise ValueError(f"num_particles must be >= 1; got {num_particles}.")

    dev = torch.device(device)
    dtype = atoms.dtype
    atoms_d = atoms.to(device=dev, dtype=dtype)
    y_d = y.to(device=dev, dtype=dtype)

    # Validate forward-model output dim against y.
    with torch.no_grad():
        probe = forward_model.fn(atoms_d[:1]).reshape(1, -1)
    if probe.shape[-1] != y_d.shape[0]:
        raise ValueError(
            f"forward_model output dim {probe.shape[-1]} does not match y dim "
            f"{y_d.shape[0]}.",
        )

    if sde is None:
        sde = VPSDE()
    if seed is not None:
        torch.manual_seed(seed)

    beta_min = sde.beta_min
    beta_max = sde.beta_max

    def prior_score_fn(x_t: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        _, _, score = prior_terms(x_t, times, atoms_d, beta_min, beta_max)
        return score

    fsr = FSR(
        prior_score_fn=prior_score_fn,
        atoms=atoms_d,
        beta_min=beta_min,
        beta_max=beta_max,
        noise_variance=noise_scale * noise_scale,
        forward_model=forward_model,
    )

    def score_fn(x_t: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        return fsr.posterior_score(y_d, x_t, times, prior_score_fn)

    schedule = np.linspace(t_max, t_min, num_steps + 1)
    d = atoms_d.shape[1]
    state = torch.randn(num_particles, d, dtype=dtype, device=dev)
    for k in range(num_steps):
        t_now = float(schedule[k])
        t_next = float(schedule[k + 1])
        dt = t_now - t_next
        if dt <= 0.0:
            continue
        times = torch.full((num_particles,), t_now, dtype=dtype, device=dev)
        state = em_step(state, times, score_fn, dt, beta_min, beta_max)
    return state
