r"""Tweedie Moment-Projected Diffusion (TMPD) likelihood approximation (Boys et al. 2024).

For a *linear* forward operator :math:`\mathcal{A}(x) = Ax + b`, approximates
:math:`p(y \mid x_t)` as

.. math::

    p(y \mid x_t) \approx
    \mathcal{N}\!\bigl(y;\,
        A\,\hat{x}_0 + b,\,
        A\,\operatorname{Cov}(x_0 \mid x_t)\,A^\top
        + \Sigma_n\bigr)

where :math:`\hat{x}_0 = E[x_0 \mid x_t]` is the Tweedie posterior mean and

.. math::

    \operatorname{Cov}(x_0 \mid x_t) \approx
    \frac{1 - \bar\alpha_t}{\sqrt{\bar\alpha_t}}\, J_{m_{0\mid t}},
    \qquad
    J_{m_{0\mid t}} = \frac{\partial \hat{x}_0}{\partial x_t}.

The score is assembled directly from the paper's formula

.. math::

    \nabla_{x_t} \log p(y \mid x_t) \;=\;
    J_{m_{0\mid t}}^\top A^\top
    \Sigma_{\mathrm{full}}^{-1}\,(y - A m_{0\mid t} - b),

without differentiating the full Gaussian log-density (which would drag in
the covariance-derivative terms — a different approximation than the one
proposed in the paper).  See the reference implementation at
https://github.com/bb515/tmpdtorch.

TMPD is restricted to linear forward operators; the Gaussian-Gaussian
conjugacy that underlies the formula does not hold for nonlinear
:math:`\mathcal{A}`.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

from src.forward_model import ForwardModel
from src.scores.base import LikelihoodApproximation
from src.sde import alpha_bar_from_times

__all__ = ["TMPD"]


class TMPD(LikelihoodApproximation):
    r"""TMPD two-moment likelihood approximation (Boys et al. 2024).

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
                "TMPD requires a linear forward model "
                f"(got forward_model.is_linear=False for '{forward_model.name}'). "
                "Set forward_model.is_linear=True when constructing a linear ForwardModel."
            )
        self._prior_score_fn = prior_score_fn
        self._beta_min = beta_min
        self._beta_max = beta_max
        self._noise_variance = noise_variance
        self._forward_model = forward_model

    def likelihood_score(self, y: Tensor | float, x_t: Tensor, times: Tensor) -> Tensor:
        r"""Score :math:`\nabla_{x_t} \log p(y \mid x_t)` via the paper formula.

        Let :math:`h(x_t) := A m_{0\mid t}(x_t) + b`.  We compute
        :math:`j_h` row-by-row with one VJP per output dimension (``m`` calls;
        ``m = 1`` for the project's 1-D-output testbeds), then assemble

        .. math::

            \Sigma_{\mathrm{full}} \;=\;
            \Sigma_n + \tfrac{1 - \bar\alpha}{\sqrt{\bar\alpha}}\, j_h\, A^\top

        and return :math:`j_h^\top \Sigma_{\mathrm{full}}^{-1}(y - h(x_t))`.

        Args:
            y: observed measurement, scalar or ``(m,)`` tensor.
            x_t: ``(B, D)`` batch of noisy states.
            times: ``(B,)`` diffusion times in ``[0, 1]``.

        Returns:
            ``(B, D)`` likelihood score.
        """
        ab = alpha_bar_from_times(
            times.to(dtype=x_t.dtype), self._beta_min, self._beta_max
        )                                                                    # (B,)
        sqrt_ab = ab.sqrt()
        v_t = 1.0 - ab

        x_t_g = x_t.detach().requires_grad_(True)
        s = self._prior_score_fn(x_t_g, times)
        m_hat = (x_t_g + v_t.unsqueeze(-1) * s) / sqrt_ab.unsqueeze(-1)      # (B, D)
        h_val = self._forward_model.fn(m_hat).reshape(x_t.shape[0], -1)      # (B, m)
        m_out = h_val.shape[-1]

        j_h_rows: list[Tensor] = []
        for j in range(m_out):
            (row,) = torch.autograd.grad(
                h_val[:, j].sum(),
                x_t_g,
                retain_graph=(j < m_out - 1),
                create_graph=False,
            )
            j_h_rows.append(row)
        j_h = torch.stack(j_h_rows, dim=1).detach()                          # (B, m, D)
        h_val_d = h_val.detach()

        a_jac = self._forward_model.derivative(m_hat.detach())               # (B, D) or (B, m, D)
        if a_jac.dim() == 2:
            a_jac = a_jac.unsqueeze(1)                                       # (B, 1, D)

        noise_var = torch.as_tensor(
            self._noise_variance, device=x_t.device, dtype=x_t.dtype
        )
        eye = torch.eye(m_out, device=x_t.device, dtype=x_t.dtype).unsqueeze(0)
        scale = (v_t / sqrt_ab).view(-1, 1, 1)
        sigma_full = noise_var * eye + scale * torch.bmm(j_h, a_jac.transpose(1, 2))

        y_t = torch.as_tensor(y, device=x_t.device, dtype=x_t.dtype).reshape(1, m_out)
        residual = (y_t - h_val_d).unsqueeze(-1)                             # (B, m, 1)
        u = torch.linalg.solve(sigma_full, residual)                         # (B, m, 1)

        return torch.bmm(j_h.transpose(1, 2), u).squeeze(-1)                 # (B, D)

    def likelihood(self, y: Tensor | float, x_t: Tensor, times: Tensor) -> Tensor:
        r"""Approximate density :math:`\mathcal{N}(y;\, h(x_t),\, \Sigma_{\mathrm{full}}(x_t))`.

        Provided for diagnostics only.  ``TMPD.likelihood_score`` does NOT
        equal the autograd of :math:`\log` of this density: the paper formula
        drops the covariance-derivative terms that the full log-density
        gradient would include.

        Args:
            y: observed measurement, scalar or ``(m,)`` tensor.
            x_t: ``(B, D)`` batch of noisy states.
            times: ``(B,)`` diffusion times in ``[0, 1]``.

        Returns:
            ``(B,)`` density values.
        """
        ab = alpha_bar_from_times(
            times.to(dtype=x_t.dtype), self._beta_min, self._beta_max
        )
        sqrt_ab = ab.sqrt()
        v_t = 1.0 - ab

        x_t_g = x_t.detach().requires_grad_(True)
        s = self._prior_score_fn(x_t_g, times)
        m_hat = (x_t_g + v_t.unsqueeze(-1) * s) / sqrt_ab.unsqueeze(-1)
        h_val = self._forward_model.fn(m_hat).reshape(x_t.shape[0], -1)
        m_out = h_val.shape[-1]

        j_h_rows: list[Tensor] = []
        for j in range(m_out):
            (row,) = torch.autograd.grad(
                h_val[:, j].sum(),
                x_t_g,
                retain_graph=True,
                create_graph=False,
            )
            j_h_rows.append(row)
        j_h = torch.stack(j_h_rows, dim=1).detach()

        a_jac = self._forward_model.derivative(m_hat.detach())
        if a_jac.dim() == 2:
            a_jac = a_jac.unsqueeze(1)

        noise_var = torch.as_tensor(
            self._noise_variance, device=x_t.device, dtype=x_t.dtype
        )
        eye = torch.eye(m_out, device=x_t.device, dtype=x_t.dtype).unsqueeze(0)
        scale = (v_t / sqrt_ab).view(-1, 1, 1)
        sigma_full = noise_var * eye + scale * torch.bmm(j_h, a_jac.transpose(1, 2))

        y_t = torch.as_tensor(y, device=x_t.device, dtype=x_t.dtype).reshape(1, m_out)
        residual = (y_t - h_val.detach()).unsqueeze(-1)
        sigma_inv = torch.linalg.inv(sigma_full)
        mahal = torch.bmm(
            residual.transpose(1, 2), torch.bmm(sigma_inv, residual),
        ).squeeze(-1).squeeze(-1)
        log_det = torch.logdet(
            2.0 * torch.tensor(math.pi, device=x_t.device, dtype=x_t.dtype) * sigma_full
        )
        return (-0.5 * (mahal + log_det)).exp()
