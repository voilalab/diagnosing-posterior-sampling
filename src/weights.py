"""Canonical finite-sample diffusion weight and score computation.

All functions that depend on the mixture-of-Gaussians structure of the
finite-sample forward process live here so that score modules share a
single implementation rather than duplicating it.
"""

from __future__ import annotations

import torch

from src.sde import alpha_bar_from_times

__all__ = [
    "compute_weights",
    "likelihood_prefactor",
    "prior_terms",
]


def compute_weights(
    x_t: torch.Tensor,
    alpha_bar_t: float | torch.Tensor,
    training_data: torch.Tensor,
    min_variance: float = 1e-8,
) -> torch.Tensor:
    r"""Compute normalized mixture weights :math:`w_i(\mathbf x_t, \bar\alpha(t))`.

    Args:
        x_t: ``(B, D)`` batch of states
        alpha_bar_t: scalar or ``(B,)`` tensor of :math:`\bar\alpha` values, all in ``(0, 1)``
        training_data: ``(N, D)`` training points
        min_variance: floor for the variance term (default: ``1e-8``)

    Returns:
        ``(B, N)`` normalized weights

    Raises:
        ValueError: if input shapes are incompatible or ``alpha_bar_t`` is out of range
    """
    if training_data.ndim != 2:
        raise ValueError(f"Expected training_data shape (N, D), got {tuple(training_data.shape)}")
    if x_t.ndim != 2:
        raise ValueError(f"Expected x_t shape (B, D), got {tuple(x_t.shape)}")
    if x_t.shape[1] != training_data.shape[1]:
        raise ValueError(
            "x_t and training_data must have the same D: "
            f"{x_t.shape[1]} vs {training_data.shape[1]}"
        )

    batch_size = x_t.shape[0]
    alpha = torch.as_tensor(alpha_bar_t, device=x_t.device, dtype=x_t.dtype)
    if alpha.ndim == 0:
        alpha = alpha.expand(batch_size)
    if alpha.ndim != 1 or alpha.shape[0] != batch_size:
        raise ValueError(
            f"Expected alpha_bar_t to be scalar or shape ({batch_size},), got {tuple(alpha.shape)}"
        )
    if torch.any(alpha <= 0) or torch.any(alpha >= 1):
        raise ValueError("All alpha_bar_t values must be in (0,1)")

    train = training_data.to(device=x_t.device, dtype=x_t.dtype)  # (N, D)
    x = x_t.unsqueeze(1)  # (B, 1, D)
    sqrt_alpha = torch.sqrt(alpha).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
    variance = (1.0 - alpha).unsqueeze(1).clamp_min(min_variance)  # (B, 1)

    means = sqrt_alpha * train.unsqueeze(0)  # (B, N, D)
    diff = x - means  # (B, N, D)
    log_probs = -0.5 * (diff * diff).sum(-1) / variance  # (B, N)
    return torch.softmax(log_probs, dim=1)


def prior_terms(
    x_t: torch.Tensor,
    times: torch.Tensor,
    training_data: torch.Tensor,
    beta_min: float,
    beta_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return weights, dlogp/dx, and prior score for the finite-sample diffusion prior.

    Args:
        x_t: ``(B, D)`` batch of states
        times: ``(B,)`` batch of times
        training_data: ``(N, D)`` training points
        beta_min: minimum beta of noise schedule
        beta_max: maximum beta of noise schedule

    Returns:
        Tuple of ``(weights, dlogp_dx, prior_score)``,
        shapes ``(B, N)``, ``(B, N, D)``, ``(B, D)``

    Raises:
        ValueError: if times and/or state shapes are not compatible
    """
    if x_t.ndim != 2:
        raise ValueError(f"Expected state shape (B, D), got {tuple(x_t.shape)}")
    if times.ndim != 1 or times.shape[0] != x_t.shape[0]:
        raise ValueError(
            f"Expected times shape ({x_t.shape[0]},), got {tuple(times.shape)}"
            f" for state {tuple(x_t.shape)}"
        )

    train = training_data.to(device=x_t.device, dtype=x_t.dtype)  # (N, D)
    x = x_t.unsqueeze(1)  # (B, 1, D)

    alpha = alpha_bar_from_times(
        times.to(dtype=x_t.dtype), beta_min, beta_max
    ).clamp(min=1e-10, max=1 - 1e-8)
    sqrt_alpha = torch.sqrt(alpha).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
    variance = (1.0 - alpha).clamp_min(1e-8)  # (B,)

    means = sqrt_alpha * train.unsqueeze(0)  # (B, N, D)
    diff = x - means  # (B, N, D)
    log_probs = -0.5 * (diff * diff).sum(-1) / variance.unsqueeze(1)  # (B, N)
    weights = torch.softmax(log_probs, dim=1)

    dlogp_dx = -diff / variance.unsqueeze(1).unsqueeze(-1)  # (B, N, D)
    prior_score = (weights.unsqueeze(-1) * dlogp_dx).sum(dim=1)  # (B, D)
    return weights, dlogp_dx, prior_score


def likelihood_prefactor(
    times: torch.Tensor,
    beta_min: float,
    beta_max: float,
    *,
    clamp_denominator: bool = False,
) -> torch.Tensor:
    r"""Compute the time-dependent likelihood score prefactor.

    Returns :math:`\sqrt{\bar\alpha(t)} / (1 - \bar\alpha(t))` as a ``(B,)`` tensor.
    Callers unsqueeze as needed for broadcasting.

    Args:
        times: ``(B,)`` diffusion times
        beta_min: minimum beta of noise schedule
        beta_max: maximum beta of noise schedule
        clamp_denominator: clamp ``(1 - alpha)`` to ``1e-8`` before dividing
            (guards against numerical blow-up near ``t=0``)

    Returns:
        ``(B,)`` prefactor values
    """
    alpha = alpha_bar_from_times(times, beta_min, beta_max).clamp(min=1e-10, max=1 - 1e-8)
    denominator = (1.0 - alpha).clamp_min(1e-8) if clamp_denominator else (1.0 - alpha)
    return torch.sqrt(alpha) / denominator
