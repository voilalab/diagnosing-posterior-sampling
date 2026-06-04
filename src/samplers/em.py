"""Euler-Maruyama sampler."""

from collections.abc import Callable

import torch

__all__ = ["em_step"]


def em_step(
    state: torch.Tensor,
    times: torch.Tensor,
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    step_size: float,
    beta_min: float = 0.01,
    beta_max: float = 20.0,
) -> torch.Tensor:
    """Implement a single reverse-time Euler-Maruyama step of the VP-SDE.

    Discretises the reverse-time VP-SDE (Anderson 1982):

    .. math::

        d\\mathbf{x} = \\bigl[\\tfrac{1}{2}\\beta(t)\\mathbf{x}
            + \\beta(t)\\nabla_{\\mathbf{x}}\\log p_t(\\mathbf{x})\\bigr]\\,d\\tau
            + \\sqrt{\\beta(t)}\\,d\\bar{W}

    where :math:`d\\tau = \\texttt{step\\_size} > 0` is elapsed *reverse* time
    and :math:`\\beta(t) = \\beta_{\\min} + (\\beta_{\\max} - \\beta_{\\min})\\,t`.
    The EM update is:

    .. math::

        \\mathbf{x}_{t-d\\tau} = \\mathbf{x}_t
            + \\bigl(\\tfrac{1}{2}\\beta_t\\mathbf{x}_t
              + \\beta_t\\,s_\\theta(\\mathbf{x}_t, t)\\bigr)\\,d\\tau
            + \\sqrt{\\beta_t\\,d\\tau}\\;\\varepsilon,
        \\quad \\varepsilon \\sim \\mathcal{N}(\\mathbf{0}, I).

    Args:
        state (torch.Tensor): :math:`(N, d)` batch of states to update.
        times (torch.Tensor): :math:`(N,)` batch of times of states.
        score_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Score function.
        step_size (float): Elapsed reverse time :math:`d\\tau > 0`.
        beta_min (float): Minimum value for linear beta schedule.
        beta_max (float): Maximum value for linear beta schedule.

    Returns:
        torch.Tensor: :math:`(N, d)` batch of updated states.
    """
    # Input validation
    if step_size <= 0:
        raise ValueError(f"Expected positive step size, got {step_size}")
    if beta_min < 0 or beta_max < 0:
        raise ValueError(f"Expected non-negative beta_min/beta_max, got {beta_min}, {beta_max}")
    if beta_max < beta_min:
        raise ValueError(f"Expected beta_max >= beta_min, got {beta_max} < {beta_min}")
    if state.ndim != 2:
        raise ValueError(f"Expected state batch to be 2D, got {state.ndim}")
    if times.ndim != 1:
        raise ValueError(f"Expected time batch to be 1D, got {times.ndim}")
    num_samples, _sample_dim = state.shape
    num_times, = times.shape
    if num_samples != num_times:
        raise ValueError(
            f"State batch and time batch are different sizes ({num_samples} != {num_times})"
        )

    if not torch.isfinite(state).all():
        raise ValueError("State contains non-finite values")
    if not torch.isfinite(times).all():
        raise ValueError("Times contain non-finite values")

    beta_t = beta_min + (beta_max - beta_min) * times
    beta_t = torch.clamp(beta_t, min=0.0)
    score = score_fn(state, times)
    if not torch.isfinite(score).all():
        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    drift = (0.5 * beta_t.unsqueeze(1) * state + beta_t.unsqueeze(1) * score) * step_size
    diffusion_scale = torch.sqrt(torch.clamp(beta_t.unsqueeze(1) * step_size, min=0.0))
    diffusion = diffusion_scale * torch.randn_like(state)
    return state + drift + diffusion
