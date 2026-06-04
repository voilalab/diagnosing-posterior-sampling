"""Accuracy tests for likelihood-score approximations.

For SigmaDPS, PiGDM, and FSR the test verifies that ``likelihood_score``
equals the autograd gradient of ``log(likelihood)``.  TMPD's score is the
paper's mean-gradient formula, which is NOT the gradient of the full
Gaussian log-density; it is checked against an independent closed-form
derivation on a Gaussian prior, and ZetaDPS against its analytic
rescaling of SigmaDPS.  TMPD and PiGDM reject non-linear forward models
at construction time.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from src.forward_model import ForwardModel
from src.scores import FSR, TMPD, PiGDM, SigmaDPS, ZetaDPS
from src.sde import alpha_bar_from_times
from src.weights import prior_terms

BETA_MIN = 1e-2
BETA_MAX = 2.0
NOISE_VAR = 0.1


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _linear_forward_1d(a: float = 1.0) -> ForwardModel:
    return ForwardModel(
        fn=lambda x: a * x,
        derivative=lambda x: torch.full_like(x, a),
        name=f"{a}x",
        is_linear=True,
    )


def _nonlinear_forward_1d() -> ForwardModel:
    return ForwardModel(
        fn=lambda x: x.pow(2),
        derivative=lambda x: 2.0 * x,
        name="x^2",
        is_linear=False,
    )


def _linear_forward_2d(a: float = 1.0) -> ForwardModel:
    """Projects first coordinate: ``A(x) = a * x[:, 0:1]``."""

    def fn(x: Tensor) -> Tensor:
        return a * x[:, 0:1]  # (B, 1)

    def deriv(x: Tensor) -> Tensor:
        b = x.shape[0]
        jac = torch.zeros(b, 1, 2, dtype=x.dtype, device=x.device)
        jac[:, 0, 0] = a
        return jac  # (B, 1, 2)

    return ForwardModel(fn=fn, derivative=deriv, name=f"{a}*x0", is_linear=True)


def _atoms_1d() -> Tensor:
    return torch.tensor([[-1.0], [1.0]], dtype=torch.float64)


def _atoms_2d() -> Tensor:
    return torch.tensor([[-1.0, 0.5], [1.0, -0.5]], dtype=torch.float64)


def _prior_score_fn_1d(atoms: Tensor) -> Callable[[Tensor, Tensor], Tensor]:
    def fn(x_t: Tensor, times: Tensor) -> Tensor:
        _, _, score = prior_terms(x_t, times, atoms, BETA_MIN, BETA_MAX)
        return score

    return fn


def _prior_score_fn_2d(atoms: Tensor) -> Callable[[Tensor, Tensor], Tensor]:
    def fn(x_t: Tensor, times: Tensor) -> Tensor:
        _, _, score = prior_terms(x_t, times, atoms, BETA_MIN, BETA_MAX)
        return score

    return fn


def _check_score_equals_autograd(
    obj: SigmaDPS | PiGDM | FSR,
    y: Tensor,
    x_t: Tensor,
    times: Tensor,
    *,
    atol: float = 1e-5,
) -> None:
    """Assert ``likelihood_score`` matches autograd of ``log(likelihood)``.

    Valid only for methods whose score *is* the gradient of their density
    approximation (SigmaDPS, PiGDM, FSR).  TMPD's score drops the
    covariance-derivative terms and is checked separately against the
    paper's closed-form formula.
    """
    x_t_g = x_t.detach().requires_grad_(True)
    log_lik = obj.likelihood(y, x_t_g, times).log()
    (autograd_score,) = torch.autograd.grad(log_lik.sum(), x_t_g)

    score = obj.likelihood_score(y, x_t, times)
    assert torch.allclose(score, autograd_score, atol=atol), (
        f"max |score - autograd| = {(score - autograd_score).abs().max():.2e}"
    )


def _make_1d_batch() -> tuple[Tensor, Tensor, Tensor]:
    x_t = torch.linspace(-2.0, 2.0, 5, dtype=torch.float64).unsqueeze(1)
    times = torch.full((5,), 0.5, dtype=torch.float64)
    y = torch.tensor(0.3, dtype=torch.float64)
    return y, x_t, times


def _make_2d_batch() -> tuple[Tensor, Tensor, Tensor]:
    x_t = torch.tensor(
        [[-1.0, 0.0], [0.0, 1.0], [1.0, -0.5]], dtype=torch.float64
    )
    times = torch.full((3,), 0.5, dtype=torch.float64)
    y = torch.tensor(0.3, dtype=torch.float64)
    return y, x_t, times


# ---------------------------------------------------------------------------
# SigmaDPS
# ---------------------------------------------------------------------------


def test_sigma_dps_score_equals_autograd_1d() -> None:
    atoms = _atoms_1d()
    obj = SigmaDPS(
        prior_score_fn=_prior_score_fn_1d(atoms),
        beta_min=BETA_MIN,
        beta_max=BETA_MAX,
        noise_variance=NOISE_VAR,
        forward_model=_linear_forward_1d(),
    )
    _check_score_equals_autograd(obj, *_make_1d_batch())


def test_sigma_dps_posterior_score_shape_and_finite_1d() -> None:
    atoms = _atoms_1d()
    prior_score_fn = _prior_score_fn_1d(atoms)
    obj = SigmaDPS(
        prior_score_fn=prior_score_fn,
        beta_min=BETA_MIN,
        beta_max=BETA_MAX,
        noise_variance=NOISE_VAR,
        forward_model=_linear_forward_1d(),
    )
    y, x_t, times = _make_1d_batch()
    out = obj.posterior_score(y, x_t, times, prior_score_fn)
    assert out.shape == x_t.shape
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# PiGDM
# ---------------------------------------------------------------------------


def test_pigdm_score_equals_autograd_1d() -> None:
    atoms = _atoms_1d()
    obj = PiGDM(
        prior_score_fn=_prior_score_fn_1d(atoms),
        beta_min=BETA_MIN,
        beta_max=BETA_MAX,
        noise_variance=NOISE_VAR,
        forward_model=_linear_forward_1d(),
    )
    _check_score_equals_autograd(obj, *_make_1d_batch())


def test_pigdm_score_equals_autograd_2d() -> None:
    atoms = _atoms_2d()
    obj = PiGDM(
        prior_score_fn=_prior_score_fn_2d(atoms),
        beta_min=BETA_MIN,
        beta_max=BETA_MAX,
        noise_variance=NOISE_VAR,
        forward_model=_linear_forward_2d(),
    )
    _check_score_equals_autograd(obj, *_make_2d_batch())


def test_pigdm_raises_for_nonlinear_at_construction() -> None:
    with pytest.raises(ValueError, match="linear"):
        PiGDM(
            prior_score_fn=lambda x, t: torch.zeros_like(x),
            beta_min=BETA_MIN,
            beta_max=BETA_MAX,
            noise_variance=NOISE_VAR,
            forward_model=_nonlinear_forward_1d(),
        )


# ---------------------------------------------------------------------------
# TMPD
# ---------------------------------------------------------------------------


def _gaussian_prior_score_fn(
    mu0: float, sigma0_sq: float,
) -> Callable[[Tensor, Tensor], Tensor]:
    r"""Closed-form marginal score for a 1-D Gaussian prior under VP-SDE.

    Marginal at time :math:`t`:
    :math:`\mathcal N(\sqrt{\bar\alpha}\,\mu_0,\, \bar\alpha\,\sigma_0^2 + (1-\bar\alpha))`.
    """

    def fn(x_t: Tensor, times: Tensor) -> Tensor:
        ab = alpha_bar_from_times(times.to(dtype=x_t.dtype), BETA_MIN, BETA_MAX)
        sqrt_ab = ab.sqrt().unsqueeze(-1)
        v = (ab * sigma0_sq + (1.0 - ab)).unsqueeze(-1)
        return -(x_t - sqrt_ab * mu0) / v

    return fn


def test_tmpd_matches_paper_formula_on_gaussian_prior() -> None:
    r"""Cross-check ``TMPD.likelihood_score`` against the closed-form formula.

    On a 1-D Gaussian prior :math:`\mathcal{N}(\mu_0, \sigma_0^2)` with affine
    :math:`\mathcal A(x) = a x + b`, the paper formula evaluates in closed form
    to :math:`J_{m} a (y - a m - b) / (\sigma_n^2 + a^2 C_{0\mid t})` with
    :math:`m = (\sqrt{\bar\alpha}\sigma_0^2 x_t + (1-\bar\alpha)\mu_0)/\gamma`,
    :math:`\gamma = \bar\alpha\sigma_0^2 + (1-\bar\alpha)`,
    :math:`J_m = \sqrt{\bar\alpha}\sigma_0^2/\gamma`, and
    :math:`C_{0\mid t} = (1-\bar\alpha)\sigma_0^2/\gamma`.
    """
    mu0, sigma0_sq = 0.3, 0.8
    a, b_off = 1.5, -0.2
    forward_model = ForwardModel(
        fn=lambda x: a * x + b_off,
        derivative=lambda x: torch.full_like(x, a),
        name=f"{a}x+{b_off}",
        is_linear=True,
    )
    obj = TMPD(
        prior_score_fn=_gaussian_prior_score_fn(mu0, sigma0_sq),
        beta_min=BETA_MIN,
        beta_max=BETA_MAX,
        noise_variance=NOISE_VAR,
        forward_model=forward_model,
    )
    y, x_t, times = _make_1d_batch()
    score = obj.likelihood_score(y, x_t, times)

    ab = alpha_bar_from_times(times.to(dtype=x_t.dtype), BETA_MIN, BETA_MAX)
    sqrt_ab = ab.sqrt().unsqueeze(-1)
    v_t = (1.0 - ab).unsqueeze(-1)
    gamma = ab.unsqueeze(-1) * sigma0_sq + v_t
    m_hat = (sqrt_ab * sigma0_sq * x_t + v_t * mu0) / gamma
    j_m = sqrt_ab * sigma0_sq / gamma
    c_0t = v_t * sigma0_sq / gamma
    sigma_full = NOISE_VAR + (a ** 2) * c_0t
    residual = y - (a * m_hat + b_off)
    expected = j_m * a * residual / sigma_full

    assert torch.allclose(score, expected, atol=1e-10), (
        f"max |score - closed-form| = {(score - expected).abs().max():.2e}"
    )


def test_tmpd_raises_for_nonlinear_at_construction() -> None:
    with pytest.raises(ValueError, match="linear"):
        TMPD(
            prior_score_fn=lambda x, t: torch.zeros_like(x),
            beta_min=BETA_MIN,
            beta_max=BETA_MAX,
            noise_variance=NOISE_VAR,
            forward_model=_nonlinear_forward_1d(),
        )


# ---------------------------------------------------------------------------
# ZetaDPS
# ---------------------------------------------------------------------------


def test_zeta_dps_rescales_sigma_dps() -> None:
    r"""ZetaDPS.likelihood_score = (2 zeta / ||y - A(m)||) * SigmaDPS.likelihood_score."""
    atoms = _atoms_1d()
    prior_score_fn = _prior_score_fn_1d(atoms)
    base = SigmaDPS(
        prior_score_fn=prior_score_fn,
        beta_min=BETA_MIN,
        beta_max=BETA_MAX,
        noise_variance=NOISE_VAR,
        forward_model=_linear_forward_1d(),
    )
    zeta = 0.7
    practical = ZetaDPS(
        prior_score_fn=prior_score_fn,
        beta_min=BETA_MIN,
        beta_max=BETA_MAX,
        noise_variance=NOISE_VAR,
        forward_model=_linear_forward_1d(),
        zeta=zeta,
    )
    y, x_t, times = _make_1d_batch()
    g = base.likelihood_score(y, x_t, times)
    p = practical.likelihood_score(y, x_t, times)

    # Residual norm computed independently from SigmaDPS internals.
    ab = alpha_bar_from_times(times.to(dtype=x_t.dtype), BETA_MIN, BETA_MAX)
    s = prior_score_fn(x_t, times)
    m_hat = (x_t + (1.0 - ab).unsqueeze(-1) * s) / ab.sqrt().unsqueeze(-1)
    norm = (y - m_hat).abs()  # ||y - A(m)||; here A = identity, m and y are 1-D.

    expected = (2.0 * zeta / norm) * g
    assert torch.allclose(p, expected, atol=1e-10), (
        f"max |practical - expected| = {(p - expected).abs().max():.2e}"
    )


# ---------------------------------------------------------------------------
# FSR
# ---------------------------------------------------------------------------


def test_fsr_score_equals_autograd_1d() -> None:
    atoms = _atoms_1d()
    prior_score_fn = _prior_score_fn_1d(atoms)
    obj = FSR(
        prior_score_fn=prior_score_fn,
        atoms=atoms,
        beta_min=BETA_MIN,
        beta_max=BETA_MAX,
        noise_variance=NOISE_VAR,
        forward_model=_linear_forward_1d(),
    )
    _check_score_equals_autograd(obj, *_make_1d_batch(), atol=1e-4)


def test_fsr_posterior_score_shape_and_finite_1d() -> None:
    atoms = _atoms_1d()
    prior_score_fn = _prior_score_fn_1d(atoms)
    obj = FSR(
        prior_score_fn=prior_score_fn,
        atoms=atoms,
        beta_min=BETA_MIN,
        beta_max=BETA_MAX,
        noise_variance=NOISE_VAR,
        forward_model=_linear_forward_1d(),
    )
    y, x_t, times = _make_1d_batch()
    out = obj.posterior_score(y, x_t, times, prior_score_fn)
    assert out.shape == x_t.shape
    assert torch.isfinite(out).all()
