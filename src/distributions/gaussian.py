r"""Gaussian prior :math:`p_0(x_0) = \mathcal N(x_0;\, m_{\mathrm{pr}},\, C_{\mathrm{pr}})`.

Closed-form VP marginal, denoiser, and affine-likelihood objects per
``overleaf/D-experiment-details.tex`` §D.2.  ``C_pr`` is a full positive-
definite covariance matrix; a 1-D scalar-variance fast path is used when
``mean`` is 0-D.
"""

import math

import torch

from src.distributions.base import Distribution
from src.forward_model import AffineForwardModel, ForwardModel
from src.sde import VPSDE


class Gaussian(Distribution):
    r"""Gaussian prior :math:`p_0(x_0) = \mathcal N(x_0;\, m_{\mathrm{pr}},\, C_{\mathrm{pr}})`.

    Args:
        sde (VPSDE): VP forward schedule.
        mean (torch.Tensor): prior mean; ``()`` for 1D or ``(d,)`` for ``d``-dim.
        cov (torch.Tensor): prior covariance; ``()`` for 1D (scalar variance)
            or ``(d, d)`` positive-definite for ``d``-dim.
        forward_model (ForwardModel | None): optional measurement operator.
            Defaults to ``None``.
        noise_scale (float | None): optional measurement-noise standard
            deviation :math:`\sigma_n`; covariance is :math:`\sigma_n^2 I_m`.
            Defaults to ``None``.

    Raises:
        ValueError: if ``mean`` has more than one axis, if ``cov`` has the
            wrong shape relative to ``mean``, or if ``cov`` is not positive
            definite (Cholesky fails).
    """

    def __init__(
        self,
        sde: VPSDE,
        mean: torch.Tensor,
        cov: torch.Tensor,
        forward_model: ForwardModel | None = None,
        noise_scale: float | None = None,
    ) -> None:
        if mean.ndim not in (0, 1):
            raise ValueError(f"mean must be 0-D or 1-D, got shape {tuple(mean.shape)}.")
        dim = 1 if mean.ndim == 0 else mean.shape[0]
        if dim == 1:
            if cov.ndim != 0:
                raise ValueError(
                    f"cov must be 0-D when mean is 0-D (1-D prior), got shape "
                    f"{tuple(cov.shape)}.",
                )
            if float(cov) <= 0:
                raise ValueError(f"cov must be positive, got {float(cov)}.")
        else:
            if cov.shape != (dim, dim):
                raise ValueError(
                    f"cov must have shape ({dim}, {dim}), got {tuple(cov.shape)}.",
                )
            # Validate positive-definiteness via Cholesky.
            torch.linalg.cholesky(cov)
        self.sde = sde
        self.mean = mean
        self.cov = cov
        self.dim = dim
        self.forward_model = forward_model
        self.noise_scale = noise_scale
        # Cache Cholesky of prior covariance for d-D solves; sentinel for 1-D
        # (the d-D code paths early-return before touching this).
        self._chol_cov: torch.Tensor = (
            torch.linalg.cholesky(cov) if dim > 1 else torch.empty(0)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _vp(self, t: float) -> tuple[float, float, float]:
        r"""Return ``(sqrt(ab), 1 - ab, ab)`` at time ``t``."""
        if t <= 0:
            raise ValueError("Densities and scores are undefined at t = 0.")
        ab = self.sde.alpha_bar_fn(t)
        return math.sqrt(ab), 1.0 - ab, ab

    def _marginal_cov(self, t: float) -> torch.Tensor:
        r"""Marginal covariance ``ab * C_pr + (1-ab) I``."""
        _, v, ab = self._vp(t)
        if self.dim == 1:
            return ab * self.cov + v
        eye = torch.eye(self.dim, dtype=self.cov.dtype, device=self.cov.device)
        return ab * self.cov + v * eye

    def _denoiser_cov_const(self, t: float) -> torch.Tensor:
        r"""Denoiser covariance ``C_d = (C_pr^-1 + ab/v * I)^-1`` — independent of x_t."""
        _, v, ab = self._vp(t)
        if self.dim == 1:
            return 1.0 / (1.0 / self.cov + ab / v)
        eye = torch.eye(self.dim, dtype=self.cov.dtype, device=self.cov.device)
        prec_pr = torch.cholesky_inverse(self._chol_cov)
        prec_d = prec_pr + (ab / v) * eye
        return torch.linalg.inv(prec_d)

    def _denoiser_mean_internal(self, xt: torch.Tensor, t: float) -> torch.Tensor:
        r"""Denoiser mean ``C_d (C_pr^-1 m_pr + sqrt(ab)/v * x_t)``."""
        sqrt_ab, v, _ = self._vp(t)
        c_d = self._denoiser_cov_const(t)
        if self.dim == 1:
            return c_d * (self.mean / self.cov + (sqrt_ab / v) * xt)
        prec_pr_mean = torch.cholesky_solve(
            self.mean.unsqueeze(-1), self._chol_cov,
        ).squeeze(-1)                                                       # (d,)
        rhs = prec_pr_mean.unsqueeze(0) + (sqrt_ab / v) * xt                # (B, d)
        return rhs @ c_d                                                    # c_d symmetric

    def _log_iso_gauss(
        self, x: torch.Tensor, mean: torch.Tensor, var: float | torch.Tensor,
    ) -> torch.Tensor:
        r"""Log-density of :math:`\mathcal N(\mathrm{mean}, \mathrm{var}\, I_d)` at ``x``.

        ``var`` is a scalar; used for the 1-D fast path.
        """
        diff = x - mean
        sq = diff.pow(2) if self.dim == 1 else diff.pow(2).sum(-1)
        var_t = float(var) if isinstance(var, torch.Tensor) else var
        log_norm = -0.5 * self.dim * math.log(2 * math.pi * var_t)
        return log_norm - 0.5 * sq / var_t

    def _log_mv_normal(
        self, x: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor,
    ) -> torch.Tensor:
        r"""Log-density of :math:`\mathcal N(\mathrm{mean},\, \mathrm{cov})` at ``x``.

        Args:
            x (torch.Tensor): ``(B, k)``.
            mean (torch.Tensor): broadcasts against ``x``.
            cov (torch.Tensor): ``(k, k)`` positive-definite.

        Returns:
            torch.Tensor: ``(B,)`` log-density.
        """
        chol = torch.linalg.cholesky(cov)
        diff = x - mean
        z = torch.linalg.solve_triangular(
            chol, diff.unsqueeze(-1), upper=False,
        ).squeeze(-1)
        log_det = 2.0 * torch.log(torch.diagonal(chol)).sum()
        k = x.shape[-1]
        return (
            -0.5 * k * math.log(2 * math.pi)
            - 0.5 * log_det
            - 0.5 * z.pow(2).sum(-1)
        )

    # ------------------------------------------------------------------
    # Unconditional objects
    # ------------------------------------------------------------------

    def prior_density(self, x0: torch.Tensor) -> torch.Tensor:
        r"""Prior density :math:`\mathcal N(x_0;\, m_{\mathrm{pr}},\, C_{\mathrm{pr}})`."""
        if self.dim == 1:
            return self._log_iso_gauss(x0, self.mean, self.cov).exp()
        return self._log_mv_normal(x0, self.mean, self.cov).exp()

    def marginal_density(self, xt: torch.Tensor, t: float) -> torch.Tensor:
        r"""Marginal density: Gaussian at mean ``sqrt(ab) m_pr`` cov ``ab C_pr + (1-ab) I``."""
        sqrt_ab, _, _ = self._vp(t)
        cov_m = self._marginal_cov(t)
        if self.dim == 1:
            return self._log_iso_gauss(xt, sqrt_ab * self.mean, cov_m).exp()
        return self._log_mv_normal(xt, sqrt_ab * self.mean, cov_m).exp()

    def denoiser_density(
        self, x0: torch.Tensor, xt: torch.Tensor, t: float,
    ) -> torch.Tensor:
        r"""Denoiser :math:`\mathcal N(x_0;\, m_{0\mid t}(x_t),\, C_{0\mid t})`."""
        mu_d = self._denoiser_mean_internal(xt, t)
        c_d = self._denoiser_cov_const(t)
        if self.dim == 1:
            return self._log_iso_gauss(x0, mu_d, c_d).exp()
        return self._log_mv_normal(x0, mu_d, c_d).exp()

    def prior_sampler(self, num_samples: int) -> torch.Tensor:
        r"""Draw ``num_samples`` samples from the prior."""
        if self.dim == 1:
            noise = torch.randn(num_samples, dtype=self.mean.dtype, device=self.mean.device)
            return self.mean + self.cov.sqrt() * noise
        chol = self._chol_cov
        noise = torch.randn(
            num_samples, self.dim, dtype=self.mean.dtype, device=self.mean.device,
        )
        return self.mean + noise @ chol.T

    def marginal_sampler(self, t: float, num_samples: int) -> torch.Tensor:
        r"""Push prior samples through the VP kernel."""
        sqrt_ab, v, _ = self._vp(t)
        x0 = self.prior_sampler(num_samples)
        return sqrt_ab * x0 + math.sqrt(v) * torch.randn_like(x0)

    def denoiser_sampler(
        self, xt: torch.Tensor, t: float, num_samples: int,
    ) -> torch.Tensor:
        r"""Draw ``num_samples`` denoiser samples per row of ``xt``."""
        mu_d = self._denoiser_mean_internal(xt, t)
        c_d = self._denoiser_cov_const(t)
        if self.dim == 1:
            shape = (mu_d.shape[0], num_samples)
            noise = torch.randn(shape, dtype=mu_d.dtype, device=mu_d.device)
            return mu_d.unsqueeze(-1) + c_d.sqrt() * noise
        chol = torch.linalg.cholesky(c_d)
        b = mu_d.shape[0]
        noise = torch.randn(
            b, num_samples, self.dim, dtype=mu_d.dtype, device=mu_d.device,
        )
        return mu_d.unsqueeze(1) + noise @ chol.T

    def marginal_score(self, xt: torch.Tensor, t: float) -> torch.Tensor:
        r"""Marginal score: ``-(ab C_pr + (1-ab) I)^{-1} (x_t - sqrt(ab) m_pr)``."""
        sqrt_ab, _, _ = self._vp(t)
        cov_m = self._marginal_cov(t)
        diff = xt - sqrt_ab * self.mean
        if self.dim == 1:
            return -diff / cov_m
        chol_m = torch.linalg.cholesky(cov_m)
        sol = torch.cholesky_solve(diff.unsqueeze(-1), chol_m).squeeze(-1)
        return -sol

    def denoiser_mean(self, xt: torch.Tensor, t: float) -> torch.Tensor:
        r"""Denoiser mean ``m_{0|t}(x_t)``."""
        return self._denoiser_mean_internal(xt, t)

    def denoiser_cov(self, xt: torch.Tensor, t: float) -> torch.Tensor:
        r"""Denoiser covariance ``C_{0|t}`` — independent of ``x_t``, broadcast across the batch."""
        c_d = self._denoiser_cov_const(t)
        b = xt.shape[0]
        if self.dim == 1:
            return torch.full((b,), float(c_d), dtype=xt.dtype, device=xt.device)
        return c_d.unsqueeze(0).expand(b, -1, -1)

    # ------------------------------------------------------------------
    # Conditional objects (affine forward model only)
    # ------------------------------------------------------------------

    def _affine_coeffs(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Return ``(matrix, bias)`` of the forward model or raise."""
        if not isinstance(self.forward_model, AffineForwardModel):
            raise NotImplementedError(
                "Gaussian forward-model methods require an AffineForwardModel; "
                f"got {type(self.forward_model).__name__}.",
            )
        if self.noise_scale is None:
            raise NotImplementedError(
                "Forward-model methods require noise_scale to be set.",
            )
        return self.forward_model.matrix, self.forward_model.bias

    def _y_marginal_stats(
        self, xt: torch.Tensor, t: float,
    ):
        r"""Per-row mean and global covariance of :math:`p(y \mid x_t)`."""
        a_mat, a_bias = self._affine_coeffs()
        assert self.noise_scale is not None
        sigma_n2 = self.noise_scale * self.noise_scale
        c_d = self._denoiser_cov_const(t)
        mu_d = self._denoiser_mean_internal(xt, t)
        if self.dim == 1:
            mean_y = a_mat * mu_d + a_bias
            var_y = sigma_n2 + float(c_d) * float(a_mat * a_mat)
            return mean_y, var_y
        mean_y = mu_d @ a_mat.T + a_bias                                    # (B, m)
        m_dim = a_mat.shape[0]
        eye_m = torch.eye(m_dim, dtype=a_mat.dtype, device=a_mat.device)
        cov_y = sigma_n2 * eye_m + a_mat @ c_d @ a_mat.T
        return mean_y, cov_y

    def _xt_posterior_stats(
        self, y: torch.Tensor, t: float,
    ):
        r"""Per-``y`` mean and global covariance of :math:`p(x_t \mid y)`."""
        a_mat, a_bias = self._affine_coeffs()
        assert self.noise_scale is not None
        sqrt_ab, v, ab = self._vp(t)
        sigma_n2 = self.noise_scale * self.noise_scale
        if self.dim == 1:
            denom = float(self.cov) * float(a_mat * a_mat) + sigma_n2
            gain = float(self.cov) * float(a_mat) / denom
            mu_p = sqrt_ab * self.mean + sqrt_ab * gain * (
                y - a_mat * self.mean - a_bias
            )
            cov_p = (
                ab * float(self.cov)
                + v
                - ab * float(self.cov) ** 2 * float(a_mat * a_mat) / denom
            )
            return mu_p, cov_p
        m_dim = a_mat.shape[0]
        eye_m = torch.eye(m_dim, dtype=a_mat.dtype, device=a_mat.device)
        cov_y_marg = a_mat @ self.cov @ a_mat.T + sigma_n2 * eye_m
        chol_y = torch.linalg.cholesky(cov_y_marg)
        diff = y - self.mean @ a_mat.T - a_bias                             # (B, m)
        sol = torch.cholesky_solve(diff.unsqueeze(-1), chol_y).squeeze(-1)  # (B, m)
        mu_p = sqrt_ab * self.mean + sqrt_ab * (sol @ a_mat) @ self.cov     # (B, d)
        eye_d = torch.eye(self.dim, dtype=a_mat.dtype, device=a_mat.device)
        a_c_pr = a_mat @ self.cov                                           # (m, d)
        m_inv_a_c_pr = torch.cholesky_solve(a_c_pr, chol_y)                 # (m, d)
        sigma_p_mat = (
            ab * self.cov + v * eye_d
            - ab * (self.cov @ a_mat.T) @ m_inv_a_c_pr
        )
        return mu_p, sigma_p_mat

    def likelihood_density(
        self, y: torch.Tensor, xt: torch.Tensor, t: float,
    ) -> torch.Tensor:
        r"""Likelihood density: Gaussian at ``A m_d + b``, cov ``sigma_n^2 I + A C_d A^T``."""
        mean_y, cov_y = self._y_marginal_stats(xt, t)
        if self.dim == 1:
            return self._log_iso_gauss(y, mean_y, cov_y).exp()
        return self._log_mv_normal(y, mean_y, cov_y).exp()

    def likelihood_sampler(
        self, xt: torch.Tensor, t: float, num_samples: int,
    ) -> torch.Tensor:
        r"""Per-row draws :math:`y \sim p(y \mid x_t)`."""
        mean_y, cov_y = self._y_marginal_stats(xt, t)
        if self.dim == 1:
            shape = (mean_y.shape[0], num_samples)
            noise = torch.randn(shape, dtype=mean_y.dtype, device=mean_y.device)
            return mean_y.unsqueeze(-1) + math.sqrt(cov_y) * noise
        b_size, m_dim = mean_y.shape
        chol_y = torch.linalg.cholesky(cov_y)
        noise = torch.randn(
            b_size, num_samples, m_dim, dtype=mean_y.dtype, device=mean_y.device,
        )
        return mean_y.unsqueeze(1) + noise @ chol_y.T

    def likelihood_score(
        self, y: torch.Tensor, xt: torch.Tensor, t: float,
    ) -> torch.Tensor:
        r"""Likelihood score (D-tex eq. 160).

        :math:`(\sqrt{\bar\alpha}/(1-\bar\alpha))\, C_{0\mid t} A^\top
        \Sigma_y^{-1} (y - A m_{0\mid t} - b)`,
        with :math:`\Sigma_y = \sigma_n^2 I_m + A C_{0\mid t} A^\top`.
        """
        a_mat, a_bias = self._affine_coeffs()
        assert self.noise_scale is not None
        sqrt_ab, v, _ = self._vp(t)
        c_d = self._denoiser_cov_const(t)
        mu_d = self._denoiser_mean_internal(xt, t)
        sigma_n2 = self.noise_scale * self.noise_scale
        if self.dim == 1:
            sigma_y = sigma_n2 + float(c_d) * float(a_mat * a_mat)
            residual = y - a_mat * mu_d - a_bias
            return (sqrt_ab / v) * float(c_d) * float(a_mat) * residual / sigma_y
        m_dim = a_mat.shape[0]
        eye_m = torch.eye(m_dim, dtype=a_mat.dtype, device=a_mat.device)
        sigma_y = sigma_n2 * eye_m + a_mat @ c_d @ a_mat.T                  # (m, m)
        chol_y = torch.linalg.cholesky(sigma_y)
        residual = y - mu_d @ a_mat.T - a_bias                              # (B, m)
        u = torch.cholesky_solve(residual.unsqueeze(-1), chol_y).squeeze(-1)  # (B, m)
        return (sqrt_ab / v) * (u @ a_mat) @ c_d                            # (B, d)

    def posterior_density(
        self, xt: torch.Tensor, y: torch.Tensor, t: float,
    ) -> torch.Tensor:
        r"""Posterior density: Gaussian at ``m_post(y)``, cov ``C_post``."""
        mu_p, cov_p = self._xt_posterior_stats(y, t)
        if self.dim == 1:
            return self._log_iso_gauss(xt, mu_p, cov_p).exp()
        return self._log_mv_normal(xt, mu_p, cov_p).exp()

    def posterior_sampler(
        self, y: torch.Tensor, t: float, num_samples: int,
    ) -> torch.Tensor:
        r"""Per-``y`` draws :math:`x_t \sim p(x_t \mid y)`."""
        mu_p, cov_p = self._xt_posterior_stats(y, t)
        if self.dim == 1:
            shape = (mu_p.shape[0], num_samples)
            noise = torch.randn(shape, dtype=mu_p.dtype, device=mu_p.device)
            return mu_p.unsqueeze(-1) + math.sqrt(cov_p) * noise
        b_size = mu_p.shape[0]
        chol_p = torch.linalg.cholesky(cov_p)
        noise = torch.randn(
            b_size, num_samples, self.dim, dtype=mu_p.dtype, device=mu_p.device,
        )
        return mu_p.unsqueeze(1) + noise @ chol_p.T

    def posterior_score(
        self, xt: torch.Tensor, y: torch.Tensor, t: float,
    ) -> torch.Tensor:
        r"""Posterior score: ``-C_post^{-1} (x_t - m_post(y))``."""
        mu_p, cov_p = self._xt_posterior_stats(y, t)
        diff = xt - mu_p
        if self.dim == 1:
            return -diff / cov_p
        chol_p = torch.linalg.cholesky(cov_p)
        sol = torch.cholesky_solve(diff.unsqueeze(-1), chol_p).squeeze(-1)
        return -sol
