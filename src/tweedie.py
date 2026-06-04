r"""Tweedie moment helpers for the VP-SDE posterior.

The Tweedie identity relates the posterior mean :math:`E[x_0 \mid x_t]` to the
score of the marginal :math:`\nabla \log p(x_t)`:

.. math::

    E[x_0 \mid x_t] = \frac{x_t + (1 - \bar\alpha_t)\,\nabla\log p(x_t)}{\sqrt{\bar\alpha_t}}

where :math:`\bar\alpha_t = \exp\!\bigl(-\beta_{\min} t - \tfrac{1}{2}(\beta_{\max} -
\beta_{\min}) t^2\bigr)` is the VP noise schedule.

Functions
---------
tweedie_mean
    Posterior mean given the prior-score *value* (not a callable).
tweedie_jacobian
    ``(B, D, D)`` Jacobian of the posterior mean via :func:`torch.func.jacrev`.
    Takes a callable so autograd traces through the score evaluation.
tweedie_cov_isotropic
    Song's isotropic covariance scalar :math:`(1 - \bar\alpha_t) / \bar\alpha_t`.
"""

from collections.abc import Callable

import torch
from torch import Tensor

from src.sde import alpha_bar_from_times

__all__ = [
    "tweedie_cov_isotropic",
    "tweedie_jacobian",
    "tweedie_mean",
]


def tweedie_mean(
    x_t: Tensor,
    times: Tensor,
    prior_score: Tensor,
    beta_min: float,
    beta_max: float,
) -> Tensor:
    r"""Posterior mean :math:`E[x_0 \mid x_t]` via the Tweedie identity.

    Computes :math:`(x_t + (1 - \bar\alpha_t) \cdot s) / \sqrt{\bar\alpha_t}` where
    ``prior_score`` is the already-evaluated marginal score :math:`\nabla \log p(x_t)`.

    Use :func:`tweedie_jacobian` when you need to differentiate through the score
    evaluation itself.

    Args:
        x_t: ``(B, D)`` batch of noisy states.
        times: ``(B,)`` diffusion times in ``[0, 1]``.
        prior_score: ``(B, D)`` prior score :math:`\nabla \log p(x_t)`, already evaluated.
        beta_min: Minimum noise-schedule rate :math:`\beta_{\min}`.
        beta_max: Maximum noise-schedule rate :math:`\beta_{\max}`.

    Returns:
        ``(B, D)`` posterior mean estimates :math:`E[x_0 \mid x_t]`.
    """
    ab = alpha_bar_from_times(times.to(dtype=x_t.dtype), beta_min, beta_max).unsqueeze(-1)
    return (x_t + (1.0 - ab) * prior_score) / ab.sqrt()


def tweedie_jacobian(
    x_t: Tensor,
    times: Tensor,
    prior_score_fn: Callable[[Tensor, Tensor], Tensor],
    beta_min: float,
    beta_max: float,
) -> Tensor:
    r"""Per-sample Jacobian of the Tweedie mean w.r.t. :math:`x_t`.

    Computes :math:`\partial E[x_0 \mid x_t] / \partial x_t` for each sample in the
    batch by tracing ``prior_score_fn`` through :func:`torch.func.jacrev`.  The
    callable form is required here so that autograd can differentiate through the
    score evaluation; use :func:`tweedie_mean` when you only need the value.

    Args:
        x_t: ``(B, D)`` batch of noisy states.
        times: ``(B,)`` diffusion times in ``[0, 1]``.
        prior_score_fn: Callable ``(x_t: (B, D), times: (B,)) -> (B, D)`` returning
            the prior score.  Must use standard differentiable PyTorch ops.
        beta_min: Minimum noise-schedule rate :math:`\beta_{\min}`.
        beta_max: Maximum noise-schedule rate :math:`\beta_{\max}`.

    Returns:
        ``(B, D, D)`` per-sample Jacobians.
    """

    def _mean_one(x_i: Tensor, t_i: Tensor) -> Tensor:
        s = prior_score_fn(x_i.unsqueeze(0), t_i.unsqueeze(0)).squeeze(0)
        ab = alpha_bar_from_times(t_i.unsqueeze(0), beta_min, beta_max).to(dtype=x_i.dtype).squeeze()
        return (x_i + (1.0 - ab) * s) / ab.sqrt()

    return torch.func.vmap(torch.func.jacrev(_mean_one, argnums=0))(x_t, times)


def tweedie_cov_isotropic(
    times: Tensor,
    beta_min: float,
    beta_max: float,
) -> Tensor:
    r"""Song's isotropic Tweedie covariance scalar :math:`(1 - \bar\alpha_t) / \bar\alpha_t`.

    This is the variance of the Tweedie posterior under a standard-normal prior,
    used by Song et al. as a time-dependent isotropic covariance approximation.

    Args:
        times: ``(B,)`` diffusion times in ``[0, 1]``.
        beta_min: Minimum noise-schedule rate :math:`\beta_{\min}`.
        beta_max: Maximum noise-schedule rate :math:`\beta_{\max}`.

    Returns:
        ``(B,)`` scalar variance per time step.
    """
    ab = alpha_bar_from_times(times, beta_min, beta_max)
    return (1.0 - ab) / ab
