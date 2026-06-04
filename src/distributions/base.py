"""Abstract base class for analytic priors with closed-form VP marginals."""

import math
from abc import ABC, abstractmethod

import torch

from src.forward_model import ForwardModel
from src.sde import VPSDE


class Distribution(ABC):
    r"""Analytic prior with closed-form marginal and denoiser under VP forward.

    Subclasses must store a :class:`VPSDE` instance as ``self.sde`` and
    implement the abstract methods below; the marginal and denoiser
    closed forms follow from the prior via the VP transition kernel
    :math:`p_{t \mid 0}(x_t \mid x_0) = \mathcal N(\sqrt{\bar\alpha(t)} x_0,
    (1 - \bar\alpha(t)) I_d)`.

    Per ``overleaf/D-experiment-details.tex`` §D.1-D.3, each subclass exposes
    five unconditional objects (marginal density / sampler, marginal score,
    denoiser density / sampler, denoiser mean, denoiser covariance) and four
    conditional objects (likelihood density / sampler / score, posterior
    density / sampler / score).  When the closed form for a conditional
    object does not exist (e.g. Gaussian / GMM priors under a non-affine
    forward model), the subclass raises :class:`NotImplementedError`.
    """

    sde: VPSDE
    forward_model: ForwardModel | None
    noise_scale: float | None

    # ------------------------------------------------------------------
    # Unconditional objects
    # ------------------------------------------------------------------

    @abstractmethod
    def prior_density(self, x0: torch.Tensor):
        r"""Evaluate the prior density :math:`p_0(x_0)`."""
        raise NotImplementedError

    @abstractmethod
    def marginal_density(self, xt: torch.Tensor, t: float):
        r"""Evaluate the marginal density :math:`p_t(x_t)`.

        Under the VP-SDE forward process, the transition kernel is

        .. math::

            p_{t \mid 0}(x_t \mid x_0)
            = \mathcal{N}\!\left(x_t;\; \sqrt{\bar\alpha(t)}\, x_0,\;
                                  (1 - \bar\alpha(t))\, I\right),

        so the time-:math:`t` marginal is the convolution of the prior with
        this Gaussian kernel:

        .. math::

            p_t(x_t)
            = \int p_{t \mid 0}(x_t \mid x_0)\, p_0(x_0)\, \mathrm{d}x_0.
        """
        raise NotImplementedError

    @abstractmethod
    def denoiser_density(self, x0: torch.Tensor, xt: torch.Tensor, t: float):
        r"""Evaluate the denoiser density :math:`p_{0 \mid t}(x_0 \mid x_t)`.

        By Bayes' rule applied to the VP forward kernel and the prior,

        .. math::

            p_{0 \mid t}(x_0 \mid x_t)
            = \frac{p_{t \mid 0}(x_t \mid x_0)\, p_0(x_0)}{p_t(x_t)}.
        """
        raise NotImplementedError

    @abstractmethod
    def prior_sampler(self, num_samples: int):
        r"""Draw ``num_samples`` samples from the prior :math:`p_0(x_0)`."""
        raise NotImplementedError

    @abstractmethod
    def marginal_sampler(self, t: float, num_samples: int):
        r"""Draw ``num_samples`` samples from the marginal :math:`p_t(x_t)`."""
        raise NotImplementedError

    @abstractmethod
    def denoiser_sampler(self, xt: torch.Tensor, t: float, num_samples: int):
        r"""Draw ``num_samples`` samples from :math:`p_{0 \mid t}(x_0 \mid x_t)`."""
        raise NotImplementedError

    @abstractmethod
    def marginal_score(self, xt: torch.Tensor, t: float):
        r"""Compute the score function :math:`\nabla_{x_t} \log p_t(x_t)`."""
        raise NotImplementedError

    @abstractmethod
    def denoiser_mean(self, xt: torch.Tensor, t: float):
        r"""Denoiser mean :math:`m_{0 \mid t}(x_t) = E[x_0 \mid x_t]`.

        Args:
            xt (torch.Tensor): ``(B,)`` for 1D or ``(B, d)`` for ``d``-dim.
            t (float): diffusion time, must be positive.

        Returns:
            torch.Tensor: same shape as ``xt``.
        """
        raise NotImplementedError

    @abstractmethod
    def denoiser_cov(self, xt: torch.Tensor, t: float):
        r"""Denoiser covariance :math:`C_{0 \mid t}(x_t) = \mathrm{Cov}[x_0 \mid x_t]`.

        Args:
            xt (torch.Tensor): ``(B,)`` for 1D or ``(B, d)`` for ``d``-dim.
            t (float): diffusion time, must be positive.

        Returns:
            torch.Tensor: ``(B,)`` for 1D (scalar variance per row) or
            ``(B, d, d)`` for ``d``-dim (per-row covariance matrix).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Conditional objects (require forward_model + noise_scale to be set)
    # ------------------------------------------------------------------

    @abstractmethod
    def likelihood_density(
        self, y: torch.Tensor, xt: torch.Tensor, t: float,
    ):
        r"""Density :math:`p(y \mid x_t)` of the measurement marginal at ``t``.

        Marginalizes the measurement likelihood
        :math:`p(y \mid x_0) = \mathcal N(y;\, \mathcal A(x_0),\, \sigma^2 I_m)`
        against the denoiser:

        .. math::

            p(y \mid x_t) = \int p(y \mid x_0)\, p_{0 \mid t}(x_0 \mid x_t)\,
                             \mathrm{d}x_0.

        Requires ``self.forward_model`` and ``self.noise_scale`` to be set.
        Subclasses for which the integral does not admit a closed form
        (e.g. Gaussian / GMM priors with a non-affine forward model) raise
        :class:`NotImplementedError`.
        """
        raise NotImplementedError

    @abstractmethod
    def likelihood_sampler(
        self, xt: torch.Tensor, t: float, num_samples: int,
    ):
        r"""Per-row samples :math:`y \sim p(y \mid x_t)`.

        Requires ``self.forward_model`` and ``self.noise_scale`` to be set.
        """
        raise NotImplementedError

    @abstractmethod
    def likelihood_score(
        self, y: torch.Tensor, xt: torch.Tensor, t: float,
    ):
        r"""Score :math:`\nabla_{x_t} \log p(y \mid x_t)`.

        Closed-form for the discrete prior under any forward model, and for
        Gaussian / GMM priors under affine forward models.  Subclasses
        raise :class:`NotImplementedError` outside their closed-form regime.

        Requires ``self.forward_model`` and ``self.noise_scale`` to be set.
        """
        raise NotImplementedError

    @abstractmethod
    def posterior_density(
        self, xt: torch.Tensor, y: torch.Tensor, t: float,
    ):
        r"""Density :math:`p(x_t \mid y)` of the noisy posterior at ``t``.

        Requires ``self.forward_model`` and ``self.noise_scale`` to be set.
        """
        raise NotImplementedError

    @abstractmethod
    def posterior_sampler(
        self, y: torch.Tensor, t: float, num_samples: int,
    ):
        r"""Per-``y`` samples :math:`x_t \sim p(x_t \mid y)`.

        Requires ``self.forward_model`` and ``self.noise_scale`` to be set.
        """
        raise NotImplementedError

    @abstractmethod
    def posterior_score(
        self, xt: torch.Tensor, y: torch.Tensor, t: float,
    ):
        r"""Score :math:`\nabla_{x_t} \log p(x_t \mid y)`.

        By Bayes's rule the posterior score decomposes as
        :math:`\nabla_{x_t}\log p(y \mid x_t) + \nabla_{x_t}\log p(x_t)`.

        Requires ``self.forward_model`` and ``self.noise_scale`` to be set.
        """
        raise NotImplementedError

    def measurement_model(
        self, y: torch.Tensor, x0: torch.Tensor,
    ) -> torch.Tensor:
        r"""Density :math:`\mathcal N(y;\, \mathcal A(x_0),\, \sigma^2 I_m)`.

        Requires ``self.forward_model`` and ``self.noise_scale`` to be set.
        ``noise_scale`` is the noise standard deviation :math:`\sigma`; the
        covariance is :math:`\sigma^2 I_m` where ``m`` is the trailing
        dimension of :math:`\mathcal A(x_0)`.

        Args:
            y (torch.Tensor): observation; shape broadcasts against
                ``self.forward_model.fn(x0)``.
            x0 (torch.Tensor): ``(B,)`` or ``(B, d)``.

        Returns:
            torch.Tensor: ``(B,)`` likelihood values.

        Raises:
            NotImplementedError: if ``forward_model`` or ``noise_scale`` is None.
        """
        if self.forward_model is None or self.noise_scale is None:
            raise NotImplementedError(
                "measurement_model requires forward_model and noise_scale to be set.",
            )
        a_x0 = self.forward_model.fn(x0)
        diff = y - a_x0
        if diff.ndim <= 1:
            m = 1
            sq = diff.pow(2)
        else:
            m = diff.shape[-1]
            sq = diff.pow(2).sum(-1)
        variance = self.noise_scale * self.noise_scale
        log_norm = -0.5 * m * math.log(2 * math.pi * variance)
        return torch.exp(log_norm - 0.5 * sq / variance)
