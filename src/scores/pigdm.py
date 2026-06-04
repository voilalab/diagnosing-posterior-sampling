r"""Song et al. pseudoinverse-guided two-moment approximation (Pi-GDM).

For a *linear* forward operator :math:`\mathcal{A}(x) = Ax + b`, approximates
:math:`p(y \mid x_t)` as

.. math::

    p(y \mid x_t) \approx
    \mathcal{N}\!\bigl(y;\,
        A\,\hat{x}_0 + b,\,
        r_t^2\, A A^\top + \Sigma_n\bigr)

where :math:`\hat{x}_0 = E[x_0 \mid x_t]` is the Tweedie posterior mean and
:math:`r_t^2 = (1 - \bar\alpha_t) / \bar\alpha_t` is Song's isotropic
covariance scalar.  :math:`\Sigma_y := r_t^2 A A^\top + \Sigma_n` depends
only on time, so the score reduces to one VJP through :math:`m_{0\mid t}`
with an analytic seed.

Reference:
    Song, J., Vahdat, A., Mardani, M., & Kautz, J. (2023).
    *Pseudoinverse-Guided Diffusion Models for Inverse Problems.*
    ICLR. https://openreview.net/forum?id=9_gsMA8MRKQ
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

from src.forward_model import ForwardModel
from src.scores.base import LikelihoodApproximation
from src.tweedie import tweedie_cov_isotropic, tweedie_mean

__all__ = ["PiGDM"]


class PiGDM(LikelihoodApproximation):
    r"""Two-moment Pi-GDM likelihood approximation (Song et al. 2023).

    Restricted to linear forward models.  Raises :class:`ValueError` at
    construction when ``forward_model.is_linear`` is ``False``.

    Args:
        prior_score_fn: callable ``(x_t, times) -> (B, D)`` returning
            :math:`\nabla \log p(x_t)`.
        beta_min: minimum noise-schedule rate :math:`\beta_{\min}`.
        beta_max: maximum noise-schedule rate :math:`\beta_{\max}`.
        noise_variance: observation noise variance, scalar (isotropic) or
            ``(m,)`` tensor of diagonal entries.
        forward_model: measurement operator; must be linear
            (``forward_model.is_linear`` must be ``True``).

    Raises:
        ValueError: if ``forward_model.is_linear`` is ``False``.
    """

    def __init__(
        self,
        prior_score_fn: Callable[[Tensor, Tensor], Tensor],
        beta_min: float,
        beta_max: float,
        noise_variance: float | Tensor,
        forward_model: ForwardModel,
    ) -> None:
        if not forward_model.is_linear:
            raise ValueError(
                "PiGDM requires a linear forward model "
                f"(got forward_model.is_linear=False for '{forward_model.name}'). "
                "Set forward_model.is_linear=True when constructing a linear ForwardModel."
            )
        self._prior_score_fn = prior_score_fn
        self._beta_min = beta_min
        self._beta_max = beta_max
        self._noise_variance = noise_variance
        self._forward_model = forward_model

    def _sigma_y(self, x_hat: Tensor, times: Tensor) -> Tensor:
        r"""Compute :math:`\Sigma_y = \sigma_n^2 I_m + r_t^2 A A^\top`.

        Returns the per-sample variance ``(B, m, m)``.  ``A`` is constant
        for linear models, so the result depends only on time.
        """
        r2_t = tweedie_cov_isotropic(times, self._beta_min, self._beta_max)  # (B,)

        # A is constant for linear models; evaluate the analytic Jacobian once.
        a_jac = self._forward_model.derivative(x_hat.detach())               # (B, D) or (B, m, D)
        if a_jac.dim() == 2:
            a_jac = a_jac.unsqueeze(1)                                       # (B, 1, D)

        m = a_jac.shape[1]
        noise_var = torch.as_tensor(
            self._noise_variance, device=x_hat.device, dtype=x_hat.dtype
        )
        aat = torch.bmm(a_jac, a_jac.transpose(1, 2))                        # (B, m, m)
        eye = torch.eye(m, device=x_hat.device, dtype=x_hat.dtype).unsqueeze(0)
        return noise_var * eye + r2_t.unsqueeze(-1).unsqueeze(-1) * aat       # (B, m, m)

    def _tweedie_mean(self, x_t: Tensor, times: Tensor) -> Tensor:
        s = self._prior_score_fn(x_t, times)
        return tweedie_mean(x_t, times, s, self._beta_min, self._beta_max)

    def _residual(self, y: Tensor | float, m_hat: Tensor) -> Tensor:
        a_m = self._forward_model.fn(m_hat).reshape(m_hat.shape[0], -1)      # (B, m)
        y_t = torch.as_tensor(y, device=m_hat.device, dtype=m_hat.dtype).reshape(1, a_m.shape[-1])
        return y_t - a_m                                                     # (B, m)

    def _log_lik(self, y: Tensor | float, x_t: Tensor, times: Tensor) -> Tensor:
        x_hat = self._tweedie_mean(x_t, times)
        residual = self._residual(y, x_hat)

        sigma_y = self._sigma_y(x_hat, times)
        sigma_y_inv = torch.linalg.inv(sigma_y).detach()                     # (B, m, m)
        log_det = torch.logdet(
            2.0 * torch.tensor(math.pi, device=x_t.device, dtype=x_t.dtype) * sigma_y
        ).detach()                                                           # (B,)
        mahal = torch.einsum("bi,bij,bj->b", residual, sigma_y_inv, residual)
        return -0.5 * (mahal + log_det)                                      # (B,)

    def likelihood(self, y: Tensor | float, x_t: Tensor, times: Tensor) -> Tensor:
        r"""Approximate density under the Pi-GDM approximation.

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

        :math:`\Sigma_y(t)` is :math:`x_t`-independent, so the score is

        .. math::

            J_{m_{0\mid t}}(x_t)^\top\, A^\top\, \Sigma_y^{-1}\,(y - A m_{0\mid t} - b).

        Computed by forming the analytic seed
        :math:`v = A^\top \Sigma_y^{-1}(y - A m_{0\mid t} - b)` and a single
        VJP through :math:`m_{0\mid t}`.

        Args:
            y: observed measurement, scalar or ``(m,)`` tensor.
            x_t: ``(B, D)`` batch of noisy states.
            times: ``(B,)`` diffusion times in ``[0, 1]``.

        Returns:
            ``(B, D)`` likelihood score.
        """
        x_t_g = x_t.detach().requires_grad_(True)
        m_hat = self._tweedie_mean(x_t_g, times)                             # (B, D)

        m_hat_d = m_hat.detach()
        residual = self._residual(y, m_hat_d)                                # (B, m)
        sigma_y = self._sigma_y(m_hat_d, times)                              # (B, m, m)

        u = torch.linalg.solve(sigma_y, residual.unsqueeze(-1)).squeeze(-1)  # (B, m)

        a_jac = self._forward_model.derivative(m_hat_d)
        if a_jac.dim() == 2:
            a_jac = a_jac.unsqueeze(1)                                       # (B, 1, D)
        v = torch.bmm(a_jac.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1)    # (B, D)

        (grad,) = torch.autograd.grad(m_hat, x_t_g, grad_outputs=v)
        return grad
