"""Abstract base class for likelihood-score approximations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from torch import Tensor

__all__ = ["LikelihoodApproximation"]


class LikelihoodApproximation(ABC):
    """Shared interface for diffusion posterior likelihood approximations.

    Each subclass encapsulates a particular approximation of
    :math:`p(y \\mid x_t)` and exposes three methods with a fixed
    call-time signature ``(y, x_t, times)``.  Method-specific
    hyperparameters (noise schedule, forward model, etc.) are captured
    in the constructor so that call sites stay uniform.
    """

    @abstractmethod
    def likelihood(self, y: Tensor | float, x_t: Tensor, times: Tensor) -> Tensor:
        """Approximate density :math:`p(y \\mid x_t)`.

        Args:
            y: observed measurement, scalar or ``(m,)`` tensor.
            x_t: ``(B, D)`` batch of noisy states.
            times: ``(B,)`` diffusion times in ``[0, 1]``.

        Returns:
            ``(B,)`` density values.
        """

    @abstractmethod
    def likelihood_score(self, y: Tensor | float, x_t: Tensor, times: Tensor) -> Tensor:
        r"""Score :math:`\nabla_{x_t} \log p(y \mid x_t)`.

        Args:
            y: observed measurement, scalar or ``(m,)`` tensor.
            x_t: ``(B, D)`` batch of noisy states.
            times: ``(B,)`` diffusion times in ``[0, 1]``.

        Returns:
            ``(B, D)`` likelihood score.
        """

    def posterior_score(
        self,
        y: Tensor | float,
        x_t: Tensor,
        times: Tensor,
        prior_score_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> Tensor:
        r"""Posterior score :math:`\nabla_{x_t} \log p(x_t \mid y)`.

        Default implementation: ``prior_score_fn(x_t, times) + likelihood_score(y, x_t, times)``.

        Args:
            y: observed measurement, scalar ``float`` or ``(m,)`` tensor.
            x_t: ``(B, D)`` batch of noisy states.
            times: ``(B,)`` diffusion times in ``[0, 1]``.
            prior_score_fn: callable ``(x_t, times) -> (B, D)`` returning
                the prior marginal score.

        Returns:
            ``(B, D)`` posterior score.
        """
        return prior_score_fn(x_t, times) + self.likelihood_score(y, x_t, times)
