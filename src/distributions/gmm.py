r"""Gaussian-mixture prior with per-component full covariance.

The prior is :math:`p_0(x_0) = \sum_k w_k\, \mathcal N(x_0;\, m_{\mathrm{pr},k},\,
C_{\mathrm{pr},k})`.  Closed-form VP marginal, denoiser, and affine-likelihood
objects per ``overleaf/D-experiment-details.tex`` §D.3.  A 1-D scalar-variance
fast path is used when ``means`` is 1-D.
"""

import math

import torch

from src.distributions.base import Distribution
from src.forward_model import AffineForwardModel, ForwardModel
from src.sde import VPSDE


class GMM(Distribution):
    r"""Gaussian-mixture prior with per-component full covariance.

    :math:`p_0(x_0) = \sum_k w_k\, \mathcal N(x_0;\, m_{\mathrm{pr},k},\, C_{\mathrm{pr},k})`.

    Args:
        sde (VPSDE): VP forward schedule.
        means (torch.Tensor): component means; ``(K,)`` for 1D or ``(K, d)``.
        covs (torch.Tensor): per-component covariances; ``(K,)`` for 1D
            (scalar variances) or ``(K, d, d)`` positive-definite for ``d``-dim.
        weights (torch.Tensor): ``(K,)`` non-negative weights; normalized in
            ``__init__``.
        forward_model (ForwardModel | None): optional measurement operator.
            Defaults to ``None``.
        noise_scale (float | None): optional measurement-noise standard
            deviation :math:`\sigma_n`.  Defaults to ``None``.

    Raises:
        ValueError: on shape mismatch, non-positive ``covs``, negative
            ``weights``, or zero/negative ``weights`` sum.
    """

    def __init__(
        self,
        sde: VPSDE,
        means: torch.Tensor,
        covs: torch.Tensor,
        weights: torch.Tensor,
        forward_model: ForwardModel | None = None,
        noise_scale: float | None = None,
    ) -> None:
        if means.ndim not in (1, 2):
            raise ValueError(f"means must be 1-D or 2-D, got shape {tuple(means.shape)}.")
        k = means.shape[0]
        dim = 1 if means.ndim == 1 else means.shape[-1]
        if dim == 1:
            if covs.shape != (k,):
                raise ValueError(
                    f"covs must have shape ({k},) when means is 1-D, got "
                    f"{tuple(covs.shape)}.",
                )
            if torch.any(covs <= 0):
                raise ValueError("covs must be strictly positive.")
        else:
            if covs.shape != (k, dim, dim):
                raise ValueError(
                    f"covs must have shape ({k}, {dim}, {dim}), got "
                    f"{tuple(covs.shape)}.",
                )
            # Validate positive-definiteness per component.
            torch.linalg.cholesky(covs)
        if weights.shape != (k,):
            raise ValueError(
                f"weights must have shape ({k},), got {tuple(weights.shape)}.",
            )
        if torch.any(weights < 0):
            raise ValueError("weights must be non-negative.")
        total = weights.sum()
        if total <= 0:
            raise ValueError("weights must sum to a positive value.")
        self.sde = sde
        self.means = means
        self.covs = covs
        self.weights = weights / total
        self.dim = dim
        self.k = k
        self.forward_model = forward_model
        self.noise_scale = noise_scale
        # Cache Cholesky of per-component covariances for d-D solves; sentinel
        # for 1-D (the d-D code paths early-return before touching this).
        self._chol_covs: torch.Tensor = (
            torch.linalg.cholesky(covs) if dim > 1 else torch.empty(0)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _vp(self, t: float) -> tuple[float, float, float]:
        if t <= 0:
            raise ValueError("Densities and scores are undefined at t = 0.")
        ab = self.sde.alpha_bar_fn(t)
        return math.sqrt(ab), 1.0 - ab, ab

    def _component_marginal_cov(self, t: float) -> torch.Tensor:
        r"""Per-component marginal cov ``ab * C_k + (1-ab) I``."""
        _, v, ab = self._vp(t)
        if self.dim == 1:
            return ab * self.covs + v                                       # (K,)
        eye = torch.eye(self.dim, dtype=self.covs.dtype, device=self.covs.device)
        return ab * self.covs + v * eye.unsqueeze(0)                        # (K, d, d)

    def _component_denoiser_cov(self, t: float) -> torch.Tensor:
        r"""Per-component denoiser cov ``C_d,k = (C_k^-1 + ab/v * I)^-1``."""
        _, v, ab = self._vp(t)
        if self.dim == 1:
            return 1.0 / (1.0 / self.covs + ab / v)                         # (K,)
        eye = torch.eye(self.dim, dtype=self.covs.dtype, device=self.covs.device)
        prec_k = torch.cholesky_inverse(self._chol_covs)                    # (K, d, d)
        prec_d = prec_k + (ab / v) * eye                                    # (K, d, d)
        return torch.linalg.inv(prec_d)

    def _component_denoiser_means(
        self, xt: torch.Tensor, t: float,
    ) -> torch.Tensor:
        r"""Per-row, per-component denoiser mean ``m_d,k(x_t)``.

        Shape: ``(B, K)`` for 1D or ``(B, K, d)`` for ``d``-dim.
        """
        sqrt_ab, v, _ = self._vp(t)
        c_d_k = self._component_denoiser_cov(t)
        if self.dim == 1:
            # m_d,k = c_d,k * (m_k / s_k + sqrt_ab/v * x_t)
            return c_d_k.unsqueeze(0) * (
                (self.means / self.covs).unsqueeze(0) + (sqrt_ab / v) * xt.unsqueeze(-1)
            )                                                               # (B, K)
        prec_k_means = torch.cholesky_solve(
            self.means.unsqueeze(-1), self._chol_covs,
        ).squeeze(-1)                                                       # (K, d)
        rhs = prec_k_means.unsqueeze(0) + (sqrt_ab / v) * xt.unsqueeze(-2)  # (B, K, d)
        # m_d,k = c_d,k @ rhs_k; batch over K and B.
        # c_d_k: (K, d, d) -> need (1, K, d, d) for broadcasting with rhs (B, K, d)
        return torch.einsum("kij,bkj->bki", c_d_k, rhs)                     # (B, K, d)

    def _component_marginal_logpdf(
        self, xt: torch.Tensor, t: float,
    ) -> torch.Tensor:
        r"""Per-row, per-component log marginal ``log N(x_t; sqrt_ab m_k, ab C_k + v I)``.

        Shape: ``(B, K)``.
        """
        sqrt_ab, _, _ = self._vp(t)
        cov_m_k = self._component_marginal_cov(t)
        if self.dim == 1:
            means_xt = sqrt_ab * self.means                                 # (K,)
            diff = xt.unsqueeze(-1) - means_xt.unsqueeze(0)                 # (B, K)
            log_norm = -0.5 * torch.log(2 * math.pi * cov_m_k)              # (K,)
            return log_norm.unsqueeze(0) - 0.5 * diff.pow(2) / cov_m_k.unsqueeze(0)
        means_xt = sqrt_ab * self.means                                     # (K, d)
        chol = torch.linalg.cholesky(cov_m_k)                               # (K, d, d)
        diff = xt.unsqueeze(-2) - means_xt.unsqueeze(0)                     # (B, K, d)
        z = torch.linalg.solve_triangular(
            chol.unsqueeze(0), diff.unsqueeze(-1), upper=False,
        ).squeeze(-1)                                                       # (B, K, d)
        sq = z.pow(2).sum(-1)                                               # (B, K)
        log_det = 2.0 * torch.log(
            torch.diagonal(chol, dim1=-2, dim2=-1),
        ).sum(-1)                                                           # (K,)
        log_norm = -0.5 * self.dim * math.log(2 * math.pi) - 0.5 * log_det
        return log_norm.unsqueeze(0) - 0.5 * sq                             # (B, K)

    def _log_responsibilities(self, xt: torch.Tensor, t: float) -> torch.Tensor:
        r"""Log of :math:`\tilde p_k(x_t)`. Shape ``(B, K)``."""
        log_g = self._component_marginal_logpdf(xt, t)
        return torch.log_softmax(torch.log(self.weights) + log_g, dim=-1)

    def _log_iso_components(
        self,
        x: torch.Tensor,
        mean_kb: torch.Tensor,
        var_k: torch.Tensor,
    ) -> torch.Tensor:
        r"""Log per-component Gaussian density (1-D fast path)."""
        diff = x.unsqueeze(-1) - mean_kb                                    # (B, K)
        log_norm = -0.5 * torch.log(2 * math.pi * var_k)                    # (K,)
        return log_norm.unsqueeze(0) - 0.5 * diff.pow(2) / var_k.unsqueeze(0)

    # ------------------------------------------------------------------
    # Unconditional objects
    # ------------------------------------------------------------------

    def prior_density(self, x0: torch.Tensor) -> torch.Tensor:
        r"""Prior :math:`\sum_k w_k \mathcal N(x_0; m_k, C_k)`."""
        if self.dim == 1:
            log_g = self._log_iso_components(x0, self.means.unsqueeze(0), self.covs)
            return torch.logsumexp(torch.log(self.weights) + log_g, dim=-1).exp()
        # d-D
        chol = self._chol_covs                                              # (K, d, d)
        diff = x0.unsqueeze(-2) - self.means.unsqueeze(0)                   # (B, K, d)
        z = torch.linalg.solve_triangular(
            chol.unsqueeze(0), diff.unsqueeze(-1), upper=False,
        ).squeeze(-1)                                                       # (B, K, d)
        sq = z.pow(2).sum(-1)                                               # (B, K)
        log_det = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
        log_norm = -0.5 * self.dim * math.log(2 * math.pi) - 0.5 * log_det
        log_g = log_norm.unsqueeze(0) - 0.5 * sq
        return torch.logsumexp(torch.log(self.weights) + log_g, dim=-1).exp()

    def marginal_density(self, xt: torch.Tensor, t: float) -> torch.Tensor:
        r"""Marginal density: GMM at per-component VP-pushed means/covs."""
        log_g = self._component_marginal_logpdf(xt, t)
        return torch.logsumexp(torch.log(self.weights) + log_g, dim=-1).exp()

    def denoiser_density(
        self, x0: torch.Tensor, xt: torch.Tensor, t: float,
    ) -> torch.Tensor:
        r"""Denoiser :math:`\sum_k \tilde p_k(x_t) \mathcal N(x_0; m_{d,k}, C_{d,k})`."""
        log_rho = self._log_responsibilities(xt, t)                         # (B, K)
        mu_d_kb = self._component_denoiser_means(xt, t)                     # (B, K) or (B, K, d)
        c_d_k = self._component_denoiser_cov(t)                             # (K,) or (K, d, d)
        if self.dim == 1:
            log_g = self._log_iso_components(x0, mu_d_kb, c_d_k)            # (B, K)
            return torch.logsumexp(log_rho + log_g, dim=-1).exp()
        chol = torch.linalg.cholesky(c_d_k)                                 # (K, d, d)
        diff = x0.unsqueeze(-2) - mu_d_kb                                   # (B, K, d)
        z = torch.linalg.solve_triangular(
            chol.unsqueeze(0), diff.unsqueeze(-1), upper=False,
        ).squeeze(-1)
        sq = z.pow(2).sum(-1)                                               # (B, K)
        log_det = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
        log_norm = -0.5 * self.dim * math.log(2 * math.pi) - 0.5 * log_det
        log_g = log_norm.unsqueeze(0) - 0.5 * sq
        return torch.logsumexp(log_rho + log_g, dim=-1).exp()

    def prior_sampler(self, num_samples: int) -> torch.Tensor:
        idx = torch.multinomial(self.weights, num_samples, replacement=True)
        if self.dim == 1:
            centers = self.means[idx]
            scales = self.covs[idx].sqrt()
            noise = torch.randn(num_samples, dtype=centers.dtype, device=centers.device)
            return centers + scales * noise
        centers = self.means[idx]                                           # (N, d)
        chol = self._chol_covs[idx]                                         # (N, d, d)
        noise = torch.randn(
            num_samples, self.dim, dtype=centers.dtype, device=centers.device,
        )
        return centers + (chol @ noise.unsqueeze(-1)).squeeze(-1)

    def marginal_sampler(self, t: float, num_samples: int) -> torch.Tensor:
        sqrt_ab, v, _ = self._vp(t)
        x0 = self.prior_sampler(num_samples)
        return sqrt_ab * x0 + math.sqrt(v) * torch.randn_like(x0)

    def denoiser_sampler(
        self, xt: torch.Tensor, t: float, num_samples: int,
    ) -> torch.Tensor:
        r"""Per-row draws from :math:`\sum_k \tilde p_k \mathcal N(m_{d,k},\, C_{d,k})`."""
        rho = self._log_responsibilities(xt, t).exp()                       # (B, K)
        mu_d_kb = self._component_denoiser_means(xt, t)
        c_d_k = self._component_denoiser_cov(t)
        idx = torch.multinomial(rho, num_samples, replacement=True)         # (B, num_samples)
        b = idx.shape[0]
        if self.dim == 1:
            mu_chosen = torch.gather(mu_d_kb, 1, idx)                       # (B, num_samples)
            sigma_chosen = c_d_k[idx].sqrt()
            noise = torch.randn(b, num_samples, dtype=mu_chosen.dtype, device=mu_chosen.device)
            return mu_chosen + sigma_chosen * noise
        idx_d = idx.unsqueeze(-1).expand(-1, -1, self.dim)
        mu_chosen = torch.gather(mu_d_kb, 1, idx_d)                         # (B, num_samples, d)
        chol = torch.linalg.cholesky(c_d_k)                                 # (K, d, d)
        chol_chosen = chol[idx]                                             # (B, num_samples, d, d)
        noise = torch.randn(
            b, num_samples, self.dim, dtype=mu_chosen.dtype, device=mu_chosen.device,
        )
        return mu_chosen + (chol_chosen @ noise.unsqueeze(-1)).squeeze(-1)

    def marginal_score(self, xt: torch.Tensor, t: float) -> torch.Tensor:
        r"""Marginal score: weighted Gaussian-mixture score (D-tex eq. 198)."""
        sqrt_ab, _, _ = self._vp(t)
        cov_m_k = self._component_marginal_cov(t)
        rho = self._log_responsibilities(xt, t).exp()                       # (B, K)
        if self.dim == 1:
            means_xt = sqrt_ab * self.means                                 # (K,)
            diff = xt.unsqueeze(-1) - means_xt.unsqueeze(0)                 # (B, K)
            inv_cov = (1.0 / cov_m_k).unsqueeze(0)                          # (1, K)
            return -(rho * inv_cov * diff).sum(-1)
        means_xt = sqrt_ab * self.means                                     # (K, d)
        chol_m_k = torch.linalg.cholesky(cov_m_k)                           # (K, d, d)
        diff = xt.unsqueeze(-2) - means_xt.unsqueeze(0)                     # (B, K, d)
        sol = torch.cholesky_solve(
            diff.unsqueeze(-1), chol_m_k.unsqueeze(0),
        ).squeeze(-1)                                                       # (B, K, d)
        return -(rho.unsqueeze(-1) * sol).sum(-2)

    def denoiser_mean(self, xt: torch.Tensor, t: float) -> torch.Tensor:
        r"""Denoiser mean: :math:`\sum_k \tilde p_k(x_t) m_{d,k}(x_t)`."""
        rho = self._log_responsibilities(xt, t).exp()                       # (B, K)
        mu_d_kb = self._component_denoiser_means(xt, t)
        if self.dim == 1:
            return (rho * mu_d_kb).sum(-1)
        return (rho.unsqueeze(-1) * mu_d_kb).sum(-2)

    def denoiser_cov(self, xt: torch.Tensor, t: float) -> torch.Tensor:
        r"""Denoiser cov: within-component + pairwise spread of component means.

        Per D-tex eq. 214,

        .. math::

            C_{0\mid t}(x_t) = \sum_k \tilde p_k\, C_{d,k}
            + \tfrac{1}{2}\sum_{k,j} \tilde p_k\, \tilde p_j\,
              (m_{d,k} - m_{d,j})(m_{d,k} - m_{d,j})^\top.
        """
        rho = self._log_responsibilities(xt, t).exp()                       # (B, K)
        mu_d_kb = self._component_denoiser_means(xt, t)
        c_d_k = self._component_denoiser_cov(t)
        if self.dim == 1:
            within = (rho * c_d_k.unsqueeze(0)).sum(-1)                     # (B,)
            # Between: sum_k rho_k m_d,k^2 - (sum_k rho_k m_d,k)^2
            mean = (rho * mu_d_kb).sum(-1)
            second = (rho * mu_d_kb.pow(2)).sum(-1)
            between = second - mean.pow(2)
            return within + between
        within = (rho.unsqueeze(-1).unsqueeze(-1) * c_d_k.unsqueeze(0)).sum(-3)  # (B, d, d)
        # E[mm^T] - E[m]E[m]^T
        outer_per = torch.einsum("bki,bkj->bkij", mu_d_kb, mu_d_kb)         # (B, K, d, d)
        second = (rho.unsqueeze(-1).unsqueeze(-1) * outer_per).sum(-3)      # (B, d, d)
        mean = (rho.unsqueeze(-1) * mu_d_kb).sum(-2)                        # (B, d)
        between = second - torch.einsum("bi,bj->bij", mean, mean)
        return within + between

    # ------------------------------------------------------------------
    # Conditional objects (affine forward model only)
    # ------------------------------------------------------------------

    def _affine_coeffs(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(self.forward_model, AffineForwardModel):
            raise NotImplementedError(
                "GMM forward-model methods require an AffineForwardModel; "
                f"got {type(self.forward_model).__name__}.",
            )
        if self.noise_scale is None:
            raise NotImplementedError(
                "Forward-model methods require noise_scale to be set.",
            )
        return self.forward_model.matrix, self.forward_model.bias

    def likelihood_density(
        self, y: torch.Tensor, xt: torch.Tensor, t: float,
    ) -> torch.Tensor:
        r"""Likelihood density: mixture over per-component affine pushforwards."""
        a_mat, a_bias = self._affine_coeffs()
        assert self.noise_scale is not None
        log_rho = self._log_responsibilities(xt, t)                         # (B, K)
        mu_d_kb = self._component_denoiser_means(xt, t)
        c_d_k = self._component_denoiser_cov(t)
        sigma_n2 = self.noise_scale * self.noise_scale
        if self.dim == 1:
            a_val = a_mat
            mean_y_kb = a_val * mu_d_kb + a_bias                            # (B, K)
            var_y_k = sigma_n2 + c_d_k * (a_val * a_val)                    # (K,)
            diff = y.unsqueeze(-1) - mean_y_kb
            log_norm = -0.5 * torch.log(2 * math.pi * var_y_k)
            log_lik = log_norm.unsqueeze(0) - 0.5 * diff.pow(2) / var_y_k.unsqueeze(0)
            return torch.logsumexp(log_rho + log_lik, dim=-1).exp()
        m_dim = a_mat.shape[0]
        eye_m = torch.eye(m_dim, dtype=a_mat.dtype, device=a_mat.device)
        cov_y_k = sigma_n2 * eye_m + a_mat @ c_d_k @ a_mat.T                # (K, m, m)
        chol_y = torch.linalg.cholesky(cov_y_k)
        mean_y_kb = mu_d_kb @ a_mat.T + a_bias                              # (B, K, m)
        diff_y = y.unsqueeze(-2) - mean_y_kb                                # (B, K, m)
        z = torch.linalg.solve_triangular(
            chol_y.unsqueeze(0), diff_y.unsqueeze(-1), upper=False,
        ).squeeze(-1)
        sq = z.pow(2).sum(-1)
        log_det = 2.0 * torch.log(torch.diagonal(chol_y, dim1=-2, dim2=-1)).sum(-1)
        log_norm = -0.5 * m_dim * math.log(2 * math.pi) - 0.5 * log_det
        log_lik = log_norm.unsqueeze(0) - 0.5 * sq
        return torch.logsumexp(log_rho + log_lik, dim=-1).exp()

    def likelihood_sampler(
        self, xt: torch.Tensor, t: float, num_samples: int,
    ) -> torch.Tensor:
        r"""Per-row draws :math:`y \sim p(y \mid x_t)`."""
        a_mat, a_bias = self._affine_coeffs()
        assert self.noise_scale is not None
        rho = self._log_responsibilities(xt, t).exp()                       # (B, K)
        mu_d_kb = self._component_denoiser_means(xt, t)
        c_d_k = self._component_denoiser_cov(t)
        idx = torch.multinomial(rho, num_samples, replacement=True)         # (B, num_samples)
        b = idx.shape[0]
        sigma_n2 = self.noise_scale * self.noise_scale
        if self.dim == 1:
            a_val = a_mat
            mean_y_kb = a_val * mu_d_kb + a_bias                            # (B, K)
            var_y_k = sigma_n2 + c_d_k * (a_val * a_val)                    # (K,)
            mean_chosen = torch.gather(mean_y_kb, 1, idx)
            var_chosen = var_y_k[idx]
            noise = torch.randn(b, num_samples, dtype=mean_chosen.dtype, device=mean_chosen.device)
            return mean_chosen + var_chosen.sqrt() * noise
        m_dim = a_mat.shape[0]
        eye_m = torch.eye(m_dim, dtype=a_mat.dtype, device=a_mat.device)
        cov_y_k = sigma_n2 * eye_m + a_mat @ c_d_k @ a_mat.T                # (K, m, m)
        chol_y = torch.linalg.cholesky(cov_y_k)
        mean_y_kb = mu_d_kb @ a_mat.T + a_bias                              # (B, K, m)
        idx_m = idx.unsqueeze(-1).expand(-1, -1, m_dim)
        mean_chosen = torch.gather(mean_y_kb, 1, idx_m)
        chol_chosen = chol_y[idx]                                           # (B, num_samples, m, m)
        noise = torch.randn(
            b, num_samples, m_dim, dtype=mean_chosen.dtype, device=mean_chosen.device,
        )
        return mean_chosen + (chol_chosen @ noise.unsqueeze(-1)).squeeze(-1)

    def _xt_posterior_components(
        self, y: torch.Tensor, t: float,
    ):
        r"""Per-component posterior stats and mixing weights for :math:`p(x_t \mid y)`.

        Returns ``(mu_p_kb, sigma_p_k, log_pi_kb)``.
        """
        a_mat, a_bias = self._affine_coeffs()
        assert self.noise_scale is not None
        sqrt_ab, v, ab = self._vp(t)
        sigma_n2 = self.noise_scale * self.noise_scale
        if self.dim == 1:
            a_val = a_mat
            a_sq = a_val * a_val
            denom_k = self.covs * a_sq + sigma_n2                           # (K,)
            gain_k = self.covs * a_val / denom_k                            # (K,)
            mean_y_marg_k = a_val * self.means + a_bias                     # (K,)
            log_pi_kb = (
                torch.log(self.weights)
                - 0.5 * torch.log(2 * math.pi * denom_k)
                - 0.5 * (y.unsqueeze(-1) - mean_y_marg_k.unsqueeze(0)).pow(2)
                / denom_k.unsqueeze(0)
            )
            log_pi_kb = torch.log_softmax(log_pi_kb, dim=-1)                # (B, K)
            mu_p_kb = sqrt_ab * self.means.unsqueeze(0) + sqrt_ab * gain_k.unsqueeze(0) * (
                y.unsqueeze(-1) - a_val * self.means.unsqueeze(0) - a_bias
            )
            sigma_p2_k = (
                ab * self.covs + v
                - ab * self.covs * self.covs * a_sq / denom_k
            )                                                               # (K,)
            return mu_p_kb, sigma_p2_k, log_pi_kb
        m_dim = a_mat.shape[0]
        eye_m = torch.eye(m_dim, dtype=a_mat.dtype, device=a_mat.device)
        cov_y_marg_k = a_mat @ self.covs @ a_mat.T + sigma_n2 * eye_m       # (K, m, m)
        chol_y_marg = torch.linalg.cholesky(cov_y_marg_k)
        mean_y_marg_k = self.means @ a_mat.T + a_bias                       # (K, m)
        diff = y.unsqueeze(-2) - mean_y_marg_k.unsqueeze(0)                 # (B, K, m)
        z = torch.linalg.solve_triangular(
            chol_y_marg.unsqueeze(0), diff.unsqueeze(-1), upper=False,
        ).squeeze(-1)                                                       # (B, K, m)
        sq = z.pow(2).sum(-1)
        log_det = 2.0 * torch.log(
            torch.diagonal(chol_y_marg, dim1=-2, dim2=-1),
        ).sum(-1)
        log_norm_y = -0.5 * m_dim * math.log(2 * math.pi) - 0.5 * log_det
        log_pi_kb = torch.log_softmax(
            torch.log(self.weights).unsqueeze(0) + log_norm_y.unsqueeze(0) - 0.5 * sq,
            dim=-1,
        )                                                                   # (B, K)
        sol = torch.cholesky_solve(
            diff.unsqueeze(-1), chol_y_marg.unsqueeze(0),
        ).squeeze(-1)                                                       # (B, K, m)
        # Per D-tex: mu_post,k(y) = sqrt_ab m_k + sqrt_ab C_k A^T M_k^-1 (y - A m_k - b)
        # sol = M_k^-1 (y - A m_k - b); next factor is C_k A^T sol.
        c_k_at = torch.einsum("kij,mj->kim", self.covs, a_mat)              # (K, d, m)
        # mu_p,k = sqrt_ab m_k + sqrt_ab einsum("kim,bkm->bki", c_k_at, sol)
        mu_p_kb = sqrt_ab * self.means.unsqueeze(0) + sqrt_ab * torch.einsum(
            "kim,bkm->bki", c_k_at, sol,
        )                                                                   # (B, K, d)
        eye_d = torch.eye(self.dim, dtype=a_mat.dtype, device=a_mat.device)
        a_c_k = torch.einsum("mj,kjl->kml", a_mat, self.covs)               # (K, m, d) = A @ C_k
        # M_k^-1 A C_k, shape (K, m, d):
        m_inv_a_c_k = torch.cholesky_solve(a_c_k, chol_y_marg)              # (K, m, d)
        # C_k A^T M_k^-1 A C_k = c_k_at @ m_inv_a_c_k, shape (K, d, d)
        sigma_p_k = (
            ab * self.covs + v * eye_d
            - ab * torch.einsum("kim,kmj->kij", c_k_at, m_inv_a_c_k)
        )                                                                   # (K, d, d)
        return mu_p_kb, sigma_p_k, log_pi_kb

    def posterior_density(
        self, xt: torch.Tensor, y: torch.Tensor, t: float,
    ) -> torch.Tensor:
        r"""Posterior density: mixture over ``(m_post,k, C_post,k)`` with weights ``pi_k(y)``."""
        mu_p_kb, sigma_p_k, log_pi_kb = self._xt_posterior_components(y, t)
        if self.dim == 1:
            log_g = (
                -0.5 * torch.log(2 * math.pi * sigma_p_k).unsqueeze(0)
                - 0.5 * (xt.unsqueeze(-1) - mu_p_kb).pow(2) / sigma_p_k.unsqueeze(0)
            )
            return torch.logsumexp(log_pi_kb + log_g, dim=-1).exp()
        chol_p = torch.linalg.cholesky(sigma_p_k)                           # (K, d, d)
        diff = xt.unsqueeze(-2) - mu_p_kb                                   # (B, K, d)
        z = torch.linalg.solve_triangular(
            chol_p.unsqueeze(0), diff.unsqueeze(-1), upper=False,
        ).squeeze(-1)
        sq = z.pow(2).sum(-1)
        log_det = 2.0 * torch.log(torch.diagonal(chol_p, dim1=-2, dim2=-1)).sum(-1)
        log_norm = -0.5 * self.dim * math.log(2 * math.pi) - 0.5 * log_det
        log_g = log_norm.unsqueeze(0) - 0.5 * sq
        return torch.logsumexp(log_pi_kb + log_g, dim=-1).exp()

    def posterior_sampler(
        self, y: torch.Tensor, t: float, num_samples: int,
    ) -> torch.Tensor:
        r"""Per-``y`` draws :math:`x_t \sim p(x_t \mid y)`."""
        mu_p_kb, sigma_p_k, log_pi_kb = self._xt_posterior_components(y, t)
        pi_kb = log_pi_kb.exp()
        idx = torch.multinomial(pi_kb, num_samples, replacement=True)
        b = idx.shape[0]
        if self.dim == 1:
            mu_chosen = torch.gather(mu_p_kb, 1, idx)
            sigma_chosen = sigma_p_k[idx]
            noise = torch.randn(b, num_samples, dtype=mu_chosen.dtype, device=mu_chosen.device)
            return mu_chosen + sigma_chosen.sqrt() * noise
        idx_d = idx.unsqueeze(-1).expand(-1, -1, self.dim)
        mu_chosen = torch.gather(mu_p_kb, 1, idx_d)
        chol_p = torch.linalg.cholesky(sigma_p_k)
        chol_chosen = chol_p[idx]
        noise = torch.randn(
            b, num_samples, self.dim, dtype=mu_chosen.dtype, device=mu_chosen.device,
        )
        return mu_chosen + (chol_chosen @ noise.unsqueeze(-1)).squeeze(-1)

    def posterior_score(
        self, xt: torch.Tensor, y: torch.Tensor, t: float,
    ) -> torch.Tensor:
        r"""Posterior score: GMM score identity (D-tex eq. 246).

        .. math::

            \nabla_{x_t}\log p(x_t \mid y)
            = -\sum_k \pi'_k(x_t, y)\, C_{\mathrm{post},k}^{-1}
                (x_t - m_{\mathrm{post},k}(y)).
        """
        mu_p_kb, sigma_p_k, log_pi_kb = self._xt_posterior_components(y, t)
        # Per-component log N(x_t; m_p,k, C_p,k) for the responsibility update.
        if self.dim == 1:
            log_g = (
                -0.5 * torch.log(2 * math.pi * sigma_p_k).unsqueeze(0)
                - 0.5 * (xt.unsqueeze(-1) - mu_p_kb).pow(2) / sigma_p_k.unsqueeze(0)
            )
            log_pi_prime = torch.log_softmax(log_pi_kb + log_g, dim=-1)     # (B, K)
            pi_prime = log_pi_prime.exp()
            diff = xt.unsqueeze(-1) - mu_p_kb                               # (B, K)
            return -(pi_prime * diff / sigma_p_k.unsqueeze(0)).sum(-1)
        chol_p = torch.linalg.cholesky(sigma_p_k)                           # (K, d, d)
        diff = xt.unsqueeze(-2) - mu_p_kb                                   # (B, K, d)
        z = torch.linalg.solve_triangular(
            chol_p.unsqueeze(0), diff.unsqueeze(-1), upper=False,
        ).squeeze(-1)
        sq = z.pow(2).sum(-1)
        log_det = 2.0 * torch.log(torch.diagonal(chol_p, dim1=-2, dim2=-1)).sum(-1)
        log_norm = -0.5 * self.dim * math.log(2 * math.pi) - 0.5 * log_det
        log_g = log_norm.unsqueeze(0) - 0.5 * sq                            # (B, K)
        log_pi_prime = torch.log_softmax(log_pi_kb + log_g, dim=-1)
        pi_prime = log_pi_prime.exp()
        sol = torch.cholesky_solve(
            diff.unsqueeze(-1), chol_p.unsqueeze(0),
        ).squeeze(-1)                                                       # (B, K, d)
        return -(pi_prime.unsqueeze(-1) * sol).sum(-2)

    def likelihood_score(
        self, y: torch.Tensor, xt: torch.Tensor, t: float,
    ) -> torch.Tensor:
        r"""Likelihood score: ``posterior_score - marginal_score`` (D-tex eq. 251).

        Computed as a difference rather than directly to avoid duplicating
        the per-component bookkeeping.
        """
        return self.posterior_score(xt, y, t) - self.marginal_score(xt, t)
