r"""Discrete prior :math:`p_0(x_0) = \sum_i w_i \delta(x_0 - x_i)`.

Closed-form VP marginal, score, and denoiser are derived in the project
notes; this module implements them.
"""

import math

import torch

from src.distributions.base import Distribution
from src.forward_model import ForwardModel
from src.sde import VPSDE


class Discrete(Distribution):
    r"""Discrete prior :math:`p_0(x_0) = \sum_i w_i \delta(x_0 - x_i)`.

    The dimension ``d`` is inferred from ``atoms.ndim``: ``1`` if ``atoms``
    is 1-D, else ``atoms.shape[-1]``. ``weights`` are normalized in
    ``__init__``.

    Args:
        sde (VPSDE): VP forward schedule.
        atoms (torch.Tensor): support points; ``(K,)`` for 1D or ``(K, d)``.
        weights (torch.Tensor): ``(K,)`` non-negative weights; need not be normalized.
        forward_model (ForwardModel | None): optional measurement operator
            used by :meth:`measurement_model`. Defaults to ``None``.
        noise_scale (float | None): optional measurement-noise standard
            deviation :math:`\sigma`; covariance is :math:`\sigma^2 I_m`.
            Defaults to ``None``.

    Raises:
        ValueError: on shape mismatch between ``atoms`` and ``weights``,
            negative ``weights``, or zero/negative ``weights`` sum.
    """

    def __init__(
        self,
        sde: VPSDE,
        atoms: torch.Tensor,
        weights: torch.Tensor,
        forward_model: ForwardModel | None = None,
        noise_scale: float | None = None,
    ) -> None:
        if atoms.ndim not in (1, 2):
            raise ValueError(f"atoms must be 1-D or 2-D, got shape {tuple(atoms.shape)}.")
        if weights.ndim != 1 or weights.shape[0] != atoms.shape[0]:
            raise ValueError(
                f"weights shape {tuple(weights.shape)} incompatible with atoms shape "
                f"{tuple(atoms.shape)}.",
            )
        if torch.any(weights < 0):
            raise ValueError("weights must be non-negative.")
        total = weights.sum()
        if total <= 0:
            raise ValueError("weights must sum to a positive value.")
        self.sde = sde
        self.atoms = atoms
        self.weights = weights / total
        self.dim = 1 if atoms.ndim == 1 else atoms.shape[-1]
        self.forward_model = forward_model
        self.noise_scale = noise_scale

    def _vp_scalars(self, t: float) -> tuple[float, float]:
        r"""Return :math:`(\sqrt{\bar\alpha(t)}, 1 - \bar\alpha(t))` as Python floats."""
        if t <= 0:
            raise ValueError("Densities and scores are undefined at t = 0.")
        alpha_bar = self.sde.alpha_bar_fn(t)
        return math.sqrt(alpha_bar), 1.0 - alpha_bar

    def _log_atom_gauss(self, xt: torch.Tensor, t: float) -> torch.Tensor:
        r"""Per-atom log :math:`\mathcal N(x_t; \sqrt{\bar\alpha(t)} x_i, (1-\bar\alpha(t)) I_d)`.

        Args:
            xt (torch.Tensor): ``(B,)`` for 1D or ``(B, d)`` for ``d``-dim.
            t (float): diffusion time, must be positive.

        Returns:
            torch.Tensor: ``(B, K)`` log-density per atom.
        """
        mu_t, sigma2_t = self._vp_scalars(t)
        means = mu_t * self.atoms
        if self.dim == 1:
            sq = (xt.unsqueeze(-1) - means.unsqueeze(0)).pow(2)
        else:
            sq = (xt.unsqueeze(-2) - means.unsqueeze(0)).pow(2).sum(-1)
        log_norm = -0.5 * self.dim * math.log(2 * math.pi * sigma2_t)
        return log_norm - 0.5 * sq / sigma2_t

    def _log_responsibilities(self, xt: torch.Tensor, t: float) -> torch.Tensor:
        r"""Log of :math:`\tilde p_i(x_t)`. Shape ``(B, K)``."""
        return torch.log_softmax(torch.log(self.weights) + self._log_atom_gauss(xt, t), dim=-1)

    def prior_density(self, x0: torch.Tensor) -> torch.Tensor:
        r"""Mass at exact-match atoms; zero off-support.

        The prior is a sum of Dirac deltas, so the density is a measure rather
        than a function. We return the matching atom's weight (zero off the
        support) so spot checks are meaningful.

        Args:
            x0 (torch.Tensor): ``(B,)`` or ``(B, d)``.

        Returns:
            torch.Tensor: ``(B,)`` weight if ``x0`` matches an atom, else ``0``.
        """
        if self.dim == 1:
            match = torch.isclose(x0.unsqueeze(-1), self.atoms.unsqueeze(0))
        else:
            match = torch.isclose(x0.unsqueeze(-2), self.atoms.unsqueeze(0)).all(-1)
        return (match.to(self.weights.dtype) * self.weights.unsqueeze(0)).sum(-1)

    def marginal_density(self, xt: torch.Tensor, t: float) -> torch.Tensor:
        r"""Marginal :math:`\sum_i p_i \mathcal N(\sqrt{\bar\alpha} x_i,\, (1-\bar\alpha) I)`.

        Args:
            xt (torch.Tensor): ``(B,)`` or ``(B, d)``.
            t (float): diffusion time, must be positive.

        Returns:
            torch.Tensor: ``(B,)`` marginal density values.
        """
        return torch.logsumexp(
            torch.log(self.weights) + self._log_atom_gauss(xt, t), dim=-1,
        ).exp()

    def denoiser_density(
        self, x0: torch.Tensor, xt: torch.Tensor, t: float,
    ) -> torch.Tensor:
        r"""Denoiser :math:`p_{0 \mid t}(x_0 \mid x_t) = \sum_i \tilde p_i(x_t) \delta(x_0 - x_i)`.

        Like :meth:`prior_density`, returns the matching atom's responsibility
        (zero off the support).

        Args:
            x0 (torch.Tensor): ``(B,)`` or ``(B, d)``.
            xt (torch.Tensor): same leading shape as ``x0``.
            t (float): diffusion time, must be positive.

        Returns:
            torch.Tensor: ``(B,)`` posterior mass.
        """
        rho = self._log_responsibilities(xt, t).exp()
        if self.dim == 1:
            match = torch.isclose(x0.unsqueeze(-1), self.atoms.unsqueeze(0))
        else:
            match = torch.isclose(x0.unsqueeze(-2), self.atoms.unsqueeze(0)).all(-1)
        return (match.to(rho.dtype) * rho).sum(-1)

    def prior_sampler(self, num_samples: int) -> torch.Tensor:
        r"""Draw ``num_samples`` exact samples from the discrete prior.

        Args:
            num_samples (int): number of samples to draw.

        Returns:
            torch.Tensor: ``(num_samples,)`` for 1D or ``(num_samples, d)`` for ``d``-dim.
        """
        idx = torch.multinomial(self.weights, num_samples, replacement=True)
        return self.atoms[idx]

    def marginal_sampler(self, t: float, num_samples: int) -> torch.Tensor:
        r"""Draw samples by pushing prior samples through the VP kernel.

        Args:
            t (float): diffusion time, must be positive.
            num_samples (int): number of samples to draw.

        Returns:
            torch.Tensor: ``(num_samples,)`` or ``(num_samples, d)``.
        """
        mu_t, sigma2_t = self._vp_scalars(t)
        x0 = self.prior_sampler(num_samples)
        return mu_t * x0 + math.sqrt(sigma2_t) * torch.randn_like(x0)

    def denoiser_sampler(
        self, xt: torch.Tensor, t: float, num_samples: int,
    ) -> torch.Tensor:
        r"""Per-row categorical sampling over atoms with weights :math:`\tilde p_i(x_t)`.

        Args:
            xt (torch.Tensor): ``(B,)`` or ``(B, d)``.
            t (float): diffusion time, must be positive.
            num_samples (int): samples per row.

        Returns:
            torch.Tensor: ``(B, num_samples)`` for 1D or ``(B, num_samples, d)``.
        """
        rho = self._log_responsibilities(xt, t).exp()
        idx = torch.multinomial(rho, num_samples, replacement=True)
        return self.atoms[idx]

    def marginal_score(self, xt: torch.Tensor, t: float) -> torch.Tensor:
        r"""Marginal score :math:`\nabla_{x_t} \log p_t(x_t)`.

        Args:
            xt (torch.Tensor): ``(B,)`` or ``(B, d)``.
            t (float): diffusion time, must be positive.

        Returns:
            torch.Tensor: same shape as ``xt``.
        """
        mu_t, sigma2_t = self._vp_scalars(t)
        rho = self._log_responsibilities(xt, t).exp()
        means = mu_t * self.atoms
        if self.dim == 1:
            return (rho * (means.unsqueeze(0) - xt.unsqueeze(-1))).sum(-1) / sigma2_t
        return (
            rho.unsqueeze(-1) * (means.unsqueeze(0) - xt.unsqueeze(-2))
        ).sum(-2) / sigma2_t

    def _ensure_forward(self) -> None:
        r"""Raise if ``forward_model`` or ``noise_scale`` is unset."""
        if self.forward_model is None or self.noise_scale is None:
            raise NotImplementedError(
                "Forward-model density/sampler methods require forward_model "
                "and noise_scale to be set.",
            )

    def _log_lik_y_per_atom(self, y: torch.Tensor) -> torch.Tensor:
        r"""Per-atom log :math:`\mathcal N(y;\, \mathcal A(x_i),\, \sigma^2 I_m)`.

        Args:
            y (torch.Tensor): ``(B,)`` for scalar :math:`y`, else ``(B, m)``.

        Returns:
            torch.Tensor: ``(B, K)`` log-likelihoods.
        """
        self._ensure_forward()
        assert self.forward_model is not None
        assert self.noise_scale is not None
        a_atoms = self.forward_model.fn(self.atoms)
        sigma2_y = self.noise_scale * self.noise_scale
        if a_atoms.ndim == 1:
            diff = y.unsqueeze(-1) - a_atoms.unsqueeze(0)
            sq = diff.pow(2)
            m = 1
        else:
            m = a_atoms.shape[-1]
            diff = y.unsqueeze(-2) - a_atoms.unsqueeze(0)
            sq = diff.pow(2).sum(-1)
        log_norm = -0.5 * m * math.log(2 * math.pi * sigma2_y)
        return log_norm - 0.5 * sq / sigma2_y

    def likelihood_density(
        self, y: torch.Tensor, xt: torch.Tensor, t: float,
    ) -> torch.Tensor:
        r"""Density :math:`p(y \mid x_t)` of the measurement marginal.

        Closed form:

        .. math::

            p(y \mid x_t) = \sum_i \tilde p_i(x_t)\,
                              \mathcal N(y; \mathcal A(x_i), \sigma^2 I_m).

        Args:
            y (torch.Tensor): ``(B,)`` for scalar :math:`y`, else ``(B, m)``.
            xt (torch.Tensor): ``(B,)`` for 1-D or ``(B, d)`` for ``d``-dim.
            t (float): diffusion time, must be positive.

        Returns:
            torch.Tensor: ``(B,)`` density values.
        """
        log_rho = self._log_responsibilities(xt, t)
        log_lik = self._log_lik_y_per_atom(y)
        return torch.logsumexp(log_rho + log_lik, dim=-1).exp()

    def likelihood_sampler(
        self, xt: torch.Tensor, t: float, num_samples: int,
    ) -> torch.Tensor:
        r"""Per-row draws :math:`y \sim p(y \mid x_t)`.

        Sample atom :math:`i \sim \tilde p_i(x_t)` per row, then
        :math:`y = \mathcal A(x_i) + \sigma \epsilon` with
        :math:`\epsilon \sim \mathcal N(0, I_m)`.

        Args:
            xt (torch.Tensor): ``(B,)`` or ``(B, d)``.
            t (float): diffusion time, must be positive.
            num_samples (int): samples per row.

        Returns:
            torch.Tensor: ``(B, num_samples)`` for scalar :math:`y` or
            ``(B, num_samples, m)`` otherwise.
        """
        self._ensure_forward()
        assert self.forward_model is not None
        assert self.noise_scale is not None
        rho = self._log_responsibilities(xt, t).exp()
        idx = torch.multinomial(rho, num_samples, replacement=True)
        a_atoms = self.forward_model.fn(self.atoms)
        means = a_atoms[idx]
        noise = torch.randn_like(means)
        return means + self.noise_scale * noise

    def posterior_density(
        self, xt: torch.Tensor, y: torch.Tensor, t: float,
    ) -> torch.Tensor:
        r"""Density :math:`p(x_t \mid y)` of the noisy posterior.

        Closed form:

        .. math::

            p(x_t \mid y) = \sum_i \psi_i(y)\,
                \mathcal N\!\left(x_t;\, \sqrt{\bar\alpha(t)}\, x_i,\,
                                  (1-\bar\alpha(t)) I_d\right),

        with

        .. math::

            \psi_i(y) = \frac{p_i\, \mathcal N(y; \mathcal A(x_i), \sigma^2 I_m)}
                            {\sum_j p_j\, \mathcal N(y; \mathcal A(x_j), \sigma^2 I_m)}.

        Args:
            xt (torch.Tensor): ``(B,)`` or ``(B, d)``.
            y (torch.Tensor): same leading shape as ``xt``; trailing ``m``
                if multi-dim.
            t (float): diffusion time, must be positive.

        Returns:
            torch.Tensor: ``(B,)`` density values.
        """
        log_lik = self._log_lik_y_per_atom(y)
        log_psi = torch.log_softmax(torch.log(self.weights) + log_lik, dim=-1)
        log_g = self._log_atom_gauss(xt, t)
        return torch.logsumexp(log_psi + log_g, dim=-1).exp()

    def posterior_sampler(
        self, y: torch.Tensor, t: float, num_samples: int,
    ) -> torch.Tensor:
        r"""Per-``y`` draws :math:`x_t \sim p(x_t \mid y)`.

        Sample atom :math:`i \sim \psi_i(y)` per row, then
        :math:`x_t = \sqrt{\bar\alpha(t)} x_i + \sqrt{1-\bar\alpha(t)} \epsilon`.

        Args:
            y (torch.Tensor): ``(B,)`` or ``(B, m)``.
            t (float): diffusion time, must be positive.
            num_samples (int): samples per ``y`` row.

        Returns:
            torch.Tensor: ``(B, num_samples)`` for 1-D or
            ``(B, num_samples, d)`` for ``d``-dim.
        """
        self._ensure_forward()
        log_lik = self._log_lik_y_per_atom(y)
        psi = torch.softmax(torch.log(self.weights) + log_lik, dim=-1)
        idx = torch.multinomial(psi, num_samples, replacement=True)
        mu_t, sigma2_t = self._vp_scalars(t)
        centers = self.atoms[idx]
        noise = torch.randn_like(centers)
        return mu_t * centers + math.sqrt(sigma2_t) * noise

    # ------------------------------------------------------------------
    # New closed-form objects (per D-tex §D.1)
    # ------------------------------------------------------------------

    def denoiser_mean(self, xt: torch.Tensor, t: float) -> torch.Tensor:
        r"""Denoiser mean (D-tex eq. 49).

        :math:`m_{0\mid t}(x_t) = \sum_i \tilde p_i(x_t)\, x^{(i)}`.
        """
        rho = self._log_responsibilities(xt, t).exp()                       # (B, K)
        if self.dim == 1:
            return (rho * self.atoms.unsqueeze(0)).sum(-1)
        return (rho.unsqueeze(-1) * self.atoms.unsqueeze(0)).sum(-2)

    def denoiser_cov(self, xt: torch.Tensor, t: float) -> torch.Tensor:
        r"""Denoiser covariance (D-tex eq. 78).

        :math:`C_{0\mid t}(x_t) = \sum_i \tilde p_i\, x^{(i)} {x^{(i)}}^\top - m\, m^\top`.
        """
        rho = self._log_responsibilities(xt, t).exp()                       # (B, K)
        if self.dim == 1:
            mean = (rho * self.atoms.unsqueeze(0)).sum(-1)                  # (B,)
            second = (rho * self.atoms.unsqueeze(0).pow(2)).sum(-1)
            return second - mean.pow(2)
        outer_atoms = torch.einsum("ki,kj->kij", self.atoms, self.atoms)    # (K, d, d)
        second = (
            rho.unsqueeze(-1).unsqueeze(-1) * outer_atoms.unsqueeze(0)
        ).sum(-3)                                                           # (B, d, d)
        mean = (rho.unsqueeze(-1) * self.atoms.unsqueeze(0)).sum(-2)        # (B, d)
        return second - torch.einsum("bi,bj->bij", mean, mean)

    def _log_posterior_responsibilities(
        self, xt: torch.Tensor, y: torch.Tensor, t: float,
    ) -> torch.Tensor:
        r"""Posterior responsibilities, shape ``(B, K)``.

        :math:`\tilde\psi_i(x_t, y) \propto p_i\, N(y; A x_i)\, N(x_t; \sqrt{ab}\, x_i, v I)`.
        """
        log_w = torch.log(self.weights)
        log_g = self._log_atom_gauss(xt, t)                                 # (B, K)
        log_lik = self._log_lik_y_per_atom(y)                               # (B, K)
        return torch.log_softmax(log_w + log_g + log_lik, dim=-1)

    def posterior_score(
        self, xt: torch.Tensor, y: torch.Tensor, t: float,
    ) -> torch.Tensor:
        r"""Posterior score (D-tex eq. 115).

        :math:`-\sum_i \tilde\psi_i(x_t, y)\, (x_t - \sqrt{\bar\alpha}\, x^{(i)}) / v`.
        Works for any forward model (the per-atom likelihood is a finite sum).
        """
        mu_t, sigma2_t = self._vp_scalars(t)
        psi = self._log_posterior_responsibilities(xt, y, t).exp()          # (B, K)
        means = mu_t * self.atoms
        if self.dim == 1:
            return (psi * (means.unsqueeze(0) - xt.unsqueeze(-1))).sum(-1) / sigma2_t
        return (
            psi.unsqueeze(-1) * (means.unsqueeze(0) - xt.unsqueeze(-2))
        ).sum(-2) / sigma2_t

    def likelihood_score(
        self, y: torch.Tensor, xt: torch.Tensor, t: float,
    ) -> torch.Tensor:
        r"""Likelihood score: ``posterior_score - marginal_score`` (D-tex eq. 104)."""
        return self.posterior_score(xt, y, t) - self.marginal_score(xt, t)
