r"""Diffusion Posterior Sampling (Chung et al. 2023) likelihood approximations.

The approximation replaces the intractable :math:`p(y \mid x_t)` with

.. math::

    p(y \mid x_t) \approx
    \mathcal{N}\!\bigl(y;\,
        \mathcal{A}(\hat{x}_0),\,
        \Sigma_n\bigr)

where :math:`\hat{x}_0 = E[x_0 \mid x_t]` is the Tweedie posterior mean and
:math:`\Sigma_n` is the observation noise covariance.  The covariance of the
backwards process is ignored; only the first moment is matched.

The likelihood score expands as

.. math::

    \nabla_{x_t} \log p(y \mid x_t) \;=\;
    J_{m_{0\mid t}}(x_t)^\top\,
    J_{\mathcal{A}}(m_{0\mid t}(x_t))^\top\,
    \Sigma_n^{-1}\,(y - \mathcal{A}(m_{0\mid t}(x_t))).

We evaluate the bracketed analytic factor first (using
:attr:`ForwardModel.derivative`), then propagate it through
:math:`J_{m_{0\mid t}}` with a single VJP.  Autograd is therefore responsible
only for the prior-Hessian contribution hidden inside
:math:`J_{m_{0\mid t}}`; it never traverses :math:`\mathcal{A}` or the
Mahalanobis form.

Two variants are exposed:

- :class:`SigmaDPS` is the textbook score above with the unmodified
  :math:`\Sigma_n^{-1}` prefactor.
- :class:`ZetaDPS` is the as-published practical variant from Chung et al.
  (2023): the score is rescaled by
  :math:`2\zeta / \lVert y - \mathcal{A}(m_{0\mid t})\rVert_2`, matching the
  state-dependent step size actually used in the DPS paper.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

from src.forward_model import ForwardModel
from src.scores.base import LikelihoodApproximation
from src.tweedie import tweedie_mean

__all__ = ["SigmaDPS", "ZetaDPS"]


class SigmaDPS(LikelihoodApproximation):
    r"""One-moment DPS likelihood approximation with :math:`\Sigma_n^{-1}` prefactor.

    Approximates :math:`p(y \mid x_t)` as
    :math:`\mathcal{N}(y;\, \mathcal{A}(\hat{x}_0),\, \Sigma_n)` where
    :math:`\hat{x}_0 = E[x_0 \mid x_t]` is computed via the Tweedie identity.
    The textbook DPS score (Chung et al. 2023).

    Args:
        prior_score_fn: callable ``(x_t, times) -> (B, D)`` returning
            :math:`\nabla \log p(x_t)`.  Must preserve the autograd graph
            back to ``x_t``; see :mod:`src.distributions` for graph-safe
            implementations.
        beta_min: minimum noise-schedule rate :math:`\beta_{\min}`.
        beta_max: maximum noise-schedule rate :math:`\beta_{\max}`.
        noise_variance: observation noise variance, scalar (isotropic) or
            ``(m,)`` tensor of diagonal entries.
        forward_model: measurement operator :math:`\mathcal{A}`.  May be
            nonlinear; :attr:`ForwardModel.derivative` supplies
            :math:`J_{\mathcal{A}}` at evaluation time.
    """

    def __init__(
        self,
        prior_score_fn: Callable[[Tensor, Tensor], Tensor],
        beta_min: float,
        beta_max: float,
        noise_variance: float | Tensor,
        forward_model: ForwardModel,
    ) -> None:
        self._prior_score_fn = prior_score_fn
        self._beta_min = beta_min
        self._beta_max = beta_max
        self._noise_variance = noise_variance
        self._forward_model = forward_model

    def _tweedie_mean(self, x_t: Tensor, times: Tensor) -> Tensor:
        r"""Posterior mean :math:`E[x_0 \mid x_t]` (value only)."""
        s = self._prior_score_fn(x_t, times)
        return tweedie_mean(x_t, times, s, self._beta_min, self._beta_max)

    def _log_lik(self, y: Tensor | float, x_t: Tensor, times: Tensor) -> Tensor:
        x_hat = self._tweedie_mean(x_t, times)
        a_xhat = self._forward_model.fn(x_hat).reshape(x_t.shape[0], -1)  # (B, m)
        y_t = torch.as_tensor(y, device=x_t.device, dtype=x_t.dtype).reshape(1, -1)
        noise_var = torch.as_tensor(
            self._noise_variance, device=x_t.device, dtype=x_t.dtype
        )
        log_const = -0.5 * torch.log(2.0 * torch.tensor(math.pi) * noise_var).sum()
        return log_const - 0.5 * ((y_t - a_xhat).pow(2) / noise_var).sum(-1)  # (B,)

    def likelihood(self, y: Tensor | float, x_t: Tensor, times: Tensor) -> Tensor:
        r"""Approximate density :math:`\mathcal{N}(y;\, \mathcal{A}(\hat{x}_0),\, \Sigma_n)`.

        Args:
            y: observed measurement, scalar or ``(m,)`` tensor.
            x_t: ``(B, D)`` batch of noisy states.
            times: ``(B,)`` diffusion times in ``[0, 1]``.

        Returns:
            ``(B,)`` density values.
        """
        return self._log_lik(y, x_t, times).exp()

    def likelihood_score(self, y: Tensor | float, x_t: Tensor, times: Tensor) -> Tensor:
        r"""Score :math:`\nabla_{x_t} \log p(y \mid x_t)` via option-B assembly.

        Args:
            y: observed measurement, scalar or ``(m,)`` tensor.
            x_t: ``(B, D)`` batch of noisy states.
            times: ``(B,)`` diffusion times in ``[0, 1]``.

        Returns:
            ``(B, D)`` likelihood score.
        """
        x_t_g = x_t.detach().requires_grad_(True)
        m_hat = self._tweedie_mean(x_t_g, times)                            # (B, D)

        m_hat_d = m_hat.detach()
        residual = self._residual(y, m_hat_d)                               # (B, m)
        noise_var = torch.as_tensor(
            self._noise_variance, device=x_t.device, dtype=x_t.dtype
        )
        a_jac = self._forward_model.derivative(m_hat_d)                     # (B, D) or (B, m, D)
        if a_jac.dim() == 2:
            a_jac = a_jac.unsqueeze(1)                                      # (B, 1, D)

        sigma_inv_r = residual / noise_var                                  # (B, m)
        v = torch.bmm(a_jac.transpose(1, 2), sigma_inv_r.unsqueeze(-1)).squeeze(-1)  # (B, D)

        (grad,) = torch.autograd.grad(m_hat, x_t_g, grad_outputs=v)
        return grad

    def _residual(self, y: Tensor | float, m_hat: Tensor) -> Tensor:
        r"""Compute :math:`y - \mathcal{A}(m_{0\mid t})` as ``(B, m)``."""
        a_m = self._forward_model.fn(m_hat).reshape(m_hat.shape[0], -1)     # (B, m)
        y_t = torch.as_tensor(y, device=m_hat.device, dtype=m_hat.dtype).reshape(1, -1)
        return y_t - a_m


class ZetaDPS(LikelihoodApproximation):
    r"""ζ-rescaled DPS score (Chung et al. 2023, "DPS-as-published").

    Wraps :class:`SigmaDPS` and rescales its likelihood score by
    :math:`2\zeta / \lVert y - \mathcal{A}(m_{0\mid t})\rVert_2`, matching the
    state-dependent step-size formulation actually used in the DPS paper.

    Args:
        prior_score_fn: callable ``(x_t, times) -> (B, D)``.
        beta_min: minimum noise-schedule rate :math:`\beta_{\min}`.
        beta_max: maximum noise-schedule rate :math:`\beta_{\max}`.
        noise_variance: observation noise variance.
        forward_model: measurement operator :math:`\mathcal{A}`.
        zeta: ζ hyperparameter from the DPS paper.  Defaults to ``1.0``.
    """

    def __init__(
        self,
        prior_score_fn: Callable[[Tensor, Tensor], Tensor],
        beta_min: float,
        beta_max: float,
        noise_variance: float | Tensor,
        forward_model: ForwardModel,
        zeta: float = 1.0,
    ) -> None:
        self._base = SigmaDPS(
            prior_score_fn=prior_score_fn,
            beta_min=beta_min,
            beta_max=beta_max,
            noise_variance=noise_variance,
            forward_model=forward_model,
        )
        self._zeta = float(zeta)

    @property
    def zeta(self) -> float:
        """ζ hyperparameter."""
        return self._zeta

    def likelihood(self, y: Tensor | float, x_t: Tensor, times: Tensor) -> Tensor:
        r"""Underlying SigmaDPS Gaussian density.

        The practical variant does not correspond to a likelihood whose
        log-gradient equals its score; the inherited Gaussian density is
        returned only for diagnostic purposes.
        """
        return self._base.likelihood(y, x_t, times)

    def likelihood_score(self, y: Tensor | float, x_t: Tensor, times: Tensor) -> Tensor:
        r"""ζ-rescaled DPS score.

        .. math::

            \text{score} \;=\;
            \frac{2\zeta}{\lVert y - \mathcal{A}(m_{0\mid t}(x_t))\rVert_2}
            \,\text{score}_{\text{SigmaDPS}}.

        Args:
            y: observed measurement, scalar or ``(m,)`` tensor.
            x_t: ``(B, D)`` batch of noisy states.
            times: ``(B,)`` diffusion times in ``[0, 1]``.

        Returns:
            ``(B, D)`` likelihood score.
        """
        g = self._base.likelihood_score(y, x_t, times)
        with torch.no_grad():
            m_hat = self._base._tweedie_mean(x_t, times)
            residual = self._base._residual(y, m_hat)                       # (B, m)
            norm = residual.norm(dim=-1, keepdim=True).clamp_min(
                torch.finfo(residual.dtype).tiny
            )
        return (2.0 * self._zeta / norm) * g
