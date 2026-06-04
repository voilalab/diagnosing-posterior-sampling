"""Finite-sample reference (FSR) likelihood approximation.

Marginalises exactly over a finite set of atoms drawn from the prior,
treating the empirical measure as the prior.  Serves as the ground-truth
reference against which the moment-based approximations (SigmaDPS,
PiGDM, TMPD) are compared.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

from src.forward_model import ForwardModel
from src.scores.base import LikelihoodApproximation
from src.weights import likelihood_prefactor, prior_terms

__all__ = ["FSR"]


class FSR(LikelihoodApproximation):
    """Finite-sample reference (FSR) likelihood approximation.

    Marginalises over ``atoms`` with forward-process mixture weights for
    both the likelihood and the score.  The prior score contribution in
    :meth:`posterior_score` is supplied by ``prior_score_fn``, so the
    caller controls whether the prior comes from the same empirical
    measure or a closed-form model.

    Args:
        prior_score_fn: callable ``(x_t, times) -> (B, D)`` returning
            the prior marginal score.
        atoms: ``(N, D)`` finite sample from the prior.
        beta_min: minimum noise-schedule rate.
        beta_max: maximum noise-schedule rate.
        noise_variance: observation noise variance, scalar or ``(m,)``
            diagonal.
        forward_model: measurement operator.
        clamp_denominator: whether to clamp the ``(1 - alpha)`` term near
            ``t = 0``.
    """

    def __init__(
        self,
        prior_score_fn: Callable[[Tensor, Tensor], Tensor],
        atoms: Tensor,
        beta_min: float,
        beta_max: float,
        noise_variance: float | Tensor,
        forward_model: ForwardModel,
        *,
        clamp_denominator: bool = False,
    ) -> None:
        self._prior_score_fn = prior_score_fn
        self._atoms = atoms
        self._beta_min = beta_min
        self._beta_max = beta_max
        self._noise_variance = noise_variance
        self._forward_model = forward_model
        self._clamp_denominator = clamp_denominator

    def _log_lik_per_atom(self, y: Tensor | float, x_t: Tensor) -> Tensor:
        r"""Per-atom log :math:`\mathcal N(y; \mathcal A(\text{atom}_n), \Sigma_n)`.

        Args:
            y: observed measurement, scalar or ``(m,)`` tensor.
            x_t: ``(B, D)`` batch (used only for dtype/device).

        Returns:
            ``(N,)`` log-likelihoods per atom.
        """
        train = self._atoms.to(device=x_t.device, dtype=x_t.dtype)
        y_t = torch.as_tensor(y, device=x_t.device, dtype=x_t.dtype)
        noise_var = torch.as_tensor(
            self._noise_variance, device=x_t.device, dtype=x_t.dtype,
        )
        log_const = -0.5 * torch.log(2.0 * torch.tensor(math.pi) * noise_var).sum()
        train_meas = self._forward_model.fn(train).reshape(train.shape[0], -1)  # (N, m)
        diff_meas = y_t.reshape(1, -1) - train_meas                              # (N, m)
        return log_const - 0.5 * (diff_meas.pow(2) / noise_var).sum(-1)         # (N,)

    def likelihood(self, y: Tensor | float, x_t: Tensor, times: Tensor) -> Tensor:
        r"""Marginal likelihood :math:`p(y \mid x_t)` under the FSR model.

        Evaluates :math:`\sum_n w_n(x_t)\,p(y \mid x_0 = \text{atom}_n)` where
        :math:`w_n(x_t)` are the forward-process mixture weights.

        Args:
            y: observed measurement, scalar or ``(m,)`` tensor.
            x_t: ``(B, D)`` batch of noisy states.
            times: ``(B,)`` diffusion times in ``[0, 1]``.

        Returns:
            ``(B,)`` density values.
        """
        weights, _, _ = prior_terms(
            x_t, times, self._atoms, self._beta_min, self._beta_max,
        )                                                                       # (B, N)
        lik_modes = self._log_lik_per_atom(y, x_t).exp()                        # (N,)
        return (weights * lik_modes.unsqueeze(0)).sum(dim=1)                    # (B,)

    def likelihood_score(self, y: Tensor | float, x_t: Tensor, times: Tensor) -> Tensor:
        r"""Score :math:`\nabla_{x_t} \log p(y \mid x_t)` under the FSR model.

        Args:
            y: observed measurement, scalar or ``(m,)`` tensor.
            x_t: ``(B, D)`` batch of noisy states.
            times: ``(B,)`` diffusion times in ``[0, 1]``.

        Returns:
            ``(B, D)`` likelihood score.
        """
        weights, _, _ = prior_terms(
            x_t, times, self._atoms, self._beta_min, self._beta_max,
        )                                                                       # (B, N)
        train = self._atoms.to(device=x_t.device, dtype=x_t.dtype)              # (N, D)

        prefactor = likelihood_prefactor(
            times.to(dtype=x_t.dtype),
            self._beta_min,
            self._beta_max,
            clamp_denominator=self._clamp_denominator,
        ).unsqueeze(1)                                                          # (B, 1)

        log_lik_modes = self._log_lik_per_atom(y, x_t)                          # (N,)
        min_pos = torch.finfo(weights.dtype).tiny
        log_weights = torch.log(weights.clamp_min(min_pos))                     # (B, N)
        posterior_weights = torch.softmax(
            log_weights + log_lik_modes.unsqueeze(0), dim=1,
        )                                                                       # (B, N)

        prior_mean = (weights.unsqueeze(-1) * train.unsqueeze(0)).sum(dim=1)    # (B, D)
        posterior_mean = (
            posterior_weights.unsqueeze(-1) * train.unsqueeze(0)
        ).sum(dim=1)                                                            # (B, D)
        return prefactor * (posterior_mean - prior_mean)                        # (B, D)

    def posterior_score(
        self,
        y: Tensor | float,
        x_t: Tensor,
        times: Tensor,
        prior_score_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> Tensor:
        r"""Posterior score :math:`\nabla_{x_t} \log p(x_t \mid y)`.

        Args:
            y: observed measurement, scalar ``float`` or ``(m,)`` tensor.
            x_t: ``(B, D)`` batch of noisy states.
            times: ``(B,)`` diffusion times in ``[0, 1]``.
            prior_score_fn: callable ``(x_t, times) -> (B, D)`` for the
                prior score.

        Returns:
            ``(B, D)`` posterior score.
        """
        return prior_score_fn(x_t, times) + self.likelihood_score(y, x_t, times)
