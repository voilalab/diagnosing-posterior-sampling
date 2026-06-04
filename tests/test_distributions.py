r"""Tests for the closed-form objects added to :mod:`src.distributions`.

Covers the four new methods (``denoiser_mean``, ``denoiser_cov``,
``likelihood_score``, ``posterior_score``) on all three priors
(``Discrete``, ``Gaussian``, ``GMM``).  Checks include:

- the Bayes identity ``posterior_score = marginal_score + likelihood_score``;
- ``posterior_score`` matches the autograd of ``log posterior_density``;
- ``marginal_score`` matches the autograd of ``log marginal_density``;
- ``denoiser_mean`` matches the Tweedie identity
  :math:`m_{0\mid t}(x_t) = (x_t + (1-\bar\alpha)\, s(x_t,t)) / \sqrt{\bar\alpha}`;
- ``denoiser_cov`` is symmetric (for d-D) or non-negative (for 1-D).
"""

from __future__ import annotations

import math

import pytest
import torch

from src.distributions.discrete import Discrete
from src.distributions.gaussian import Gaussian
from src.distributions.gmm import GMM
from src.forward_model import AffineForwardModel
from src.sde import VPSDE

DTYPE = torch.float64
T = 0.5
TOL = 1e-6


def _vp_scalars(sde: VPSDE) -> tuple[float, float]:
    ab = sde.alpha_bar_fn(T)
    return math.sqrt(ab), 1.0 - ab


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sde() -> VPSDE:
    return VPSDE()


@pytest.fixture
def linear_1d() -> AffineForwardModel:
    return AffineForwardModel(
        matrix=torch.tensor(1.0, dtype=DTYPE),
        bias=torch.tensor(0.0, dtype=DTYPE),
        name="identity",
    )


@pytest.fixture
def linear_2d() -> AffineForwardModel:
    return AffineForwardModel(
        matrix=torch.tensor([[1.0, 0.5]], dtype=DTYPE),
        bias=torch.tensor([0.0], dtype=DTYPE),
        name="proj",
    )


# ---------------------------------------------------------------------------
# Discrete
# ---------------------------------------------------------------------------


def test_discrete_bayes_and_autograd_1d(sde, linear_1d) -> None:
    atoms = torch.tensor([-1.0, 0.5, 1.5], dtype=DTYPE)
    weights = torch.tensor([0.3, 0.4, 0.3], dtype=DTYPE)
    p = Discrete(sde, atoms, weights, forward_model=linear_1d, noise_scale=0.3)
    xt = torch.tensor([-1.0, 0.0, 0.5, 1.0], dtype=DTYPE)
    y = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=DTYPE)
    ms, ls, ps = p.marginal_score(xt, T), p.likelihood_score(y, xt, T), p.posterior_score(xt, y, T)
    assert torch.allclose(ps, ms + ls, atol=TOL)
    xt_g = xt.detach().requires_grad_(True)
    (grad,) = torch.autograd.grad(p.posterior_density(xt_g, y, T).log().sum(), xt_g)
    assert torch.allclose(grad, ps, atol=TOL)


def test_discrete_denoiser_mean_matches_tweedie_1d(sde, linear_1d) -> None:
    atoms = torch.tensor([-1.0, 0.5, 1.5], dtype=DTYPE)
    weights = torch.tensor([0.3, 0.4, 0.3], dtype=DTYPE)
    p = Discrete(sde, atoms, weights, forward_model=linear_1d, noise_scale=0.3)
    xt = torch.tensor([-1.0, 0.0, 0.5, 1.0], dtype=DTYPE)
    sqrt_ab, v = _vp_scalars(sde)
    tweedie = (xt + v * p.marginal_score(xt, T)) / sqrt_ab
    assert torch.allclose(p.denoiser_mean(xt, T), tweedie, atol=TOL)


def test_discrete_denoiser_cov_nonnegative_1d(sde, linear_1d) -> None:
    atoms = torch.tensor([-1.0, 0.5, 1.5], dtype=DTYPE)
    weights = torch.tensor([0.3, 0.4, 0.3], dtype=DTYPE)
    p = Discrete(sde, atoms, weights, forward_model=linear_1d, noise_scale=0.3)
    xt = torch.tensor([-1.0, 0.0, 0.5, 1.0], dtype=DTYPE)
    dc = p.denoiser_cov(xt, T)
    assert (dc >= -TOL).all()


def test_discrete_works_under_nonlinear_forward_1d(sde) -> None:
    from src.forward_model import ForwardModel

    fm = ForwardModel(
        fn=lambda x: x.pow(2),
        derivative=lambda x: 2.0 * x,
        name="quad",
        is_linear=False,
    )
    atoms = torch.tensor([-1.0, 0.5, 1.5], dtype=DTYPE)
    weights = torch.tensor([0.3, 0.4, 0.3], dtype=DTYPE)
    p = Discrete(sde, atoms, weights, forward_model=fm, noise_scale=0.3)
    xt = torch.tensor([-1.0, 0.0, 1.0], dtype=DTYPE)
    y = torch.tensor([0.5, 0.5, 0.5], dtype=DTYPE)
    ms, ls, ps = p.marginal_score(xt, T), p.likelihood_score(y, xt, T), p.posterior_score(xt, y, T)
    assert torch.allclose(ps, ms + ls, atol=TOL)


def test_discrete_bayes_and_autograd_2d(sde, linear_2d) -> None:
    atoms = torch.tensor(
        [[-1.0, 0.5], [1.0, -0.5], [0.0, 1.0]], dtype=DTYPE,
    )
    weights = torch.tensor([0.3, 0.4, 0.3], dtype=DTYPE)
    p = Discrete(sde, atoms, weights, forward_model=linear_2d, noise_scale=0.3)
    xt = torch.tensor([[-1.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
    y = torch.tensor([[0.5], [-0.2]], dtype=DTYPE)
    ms, ls, ps = p.marginal_score(xt, T), p.likelihood_score(y, xt, T), p.posterior_score(xt, y, T)
    assert torch.allclose(ps, ms + ls, atol=TOL)
    xt_g = xt.detach().requires_grad_(True)
    (grad,) = torch.autograd.grad(p.posterior_density(xt_g, y, T).log().sum(), xt_g)
    assert torch.allclose(grad, ps, atol=TOL)


# ---------------------------------------------------------------------------
# Gaussian
# ---------------------------------------------------------------------------


def test_gaussian_bayes_and_autograd_1d(sde, linear_1d) -> None:
    p = Gaussian(
        sde, torch.tensor(0.5, dtype=DTYPE), torch.tensor(0.8, dtype=DTYPE),
        forward_model=linear_1d, noise_scale=0.3,
    )
    xt = torch.tensor([-1.0, 0.0, 1.0], dtype=DTYPE)
    y = torch.tensor([0.5, 0.5, 0.5], dtype=DTYPE)
    ms, ls, ps = p.marginal_score(xt, T), p.likelihood_score(y, xt, T), p.posterior_score(xt, y, T)
    assert torch.allclose(ps, ms + ls, atol=TOL)
    xt_g = xt.detach().requires_grad_(True)
    (grad,) = torch.autograd.grad(p.posterior_density(xt_g, y, T).log().sum(), xt_g)
    assert torch.allclose(grad, ps, atol=TOL)


def test_gaussian_denoiser_mean_matches_tweedie_1d(sde, linear_1d) -> None:
    p = Gaussian(
        sde, torch.tensor(0.5, dtype=DTYPE), torch.tensor(0.8, dtype=DTYPE),
        forward_model=linear_1d, noise_scale=0.3,
    )
    xt = torch.tensor([-1.0, 0.0, 1.0], dtype=DTYPE)
    sqrt_ab, v = _vp_scalars(sde)
    tweedie = (xt + v * p.marginal_score(xt, T)) / sqrt_ab
    assert torch.allclose(p.denoiser_mean(xt, T), tweedie, atol=TOL)


def test_gaussian_full_cov_bayes_and_autograd_2d(sde, linear_2d) -> None:
    p = Gaussian(
        sde,
        torch.tensor([0.5, -0.3], dtype=DTYPE),
        torch.tensor([[1.0, 0.2], [0.2, 0.5]], dtype=DTYPE),
        forward_model=linear_2d, noise_scale=0.3,
    )
    xt = torch.tensor([[-1.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
    y = torch.tensor([[0.5], [-0.2]], dtype=DTYPE)
    ms, ls, ps = p.marginal_score(xt, T), p.likelihood_score(y, xt, T), p.posterior_score(xt, y, T)
    assert torch.allclose(ps, ms + ls, atol=TOL)
    xt_g = xt.detach().requires_grad_(True)
    (grad,) = torch.autograd.grad(p.posterior_density(xt_g, y, T).log().sum(), xt_g)
    assert torch.allclose(grad, ps, atol=TOL)


def test_gaussian_denoiser_cov_symmetric_2d(sde, linear_2d) -> None:
    p = Gaussian(
        sde,
        torch.tensor([0.5, -0.3], dtype=DTYPE),
        torch.tensor([[1.0, 0.2], [0.2, 0.5]], dtype=DTYPE),
    )
    xt = torch.tensor([[-1.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
    dc = p.denoiser_cov(xt, T)
    assert dc.shape == (2, 2, 2)
    assert torch.allclose(dc, dc.transpose(-1, -2), atol=TOL)


# ---------------------------------------------------------------------------
# GMM
# ---------------------------------------------------------------------------


def test_gmm_bayes_and_autograd_1d(sde, linear_1d) -> None:
    p = GMM(
        sde,
        torch.tensor([-1.0, 1.0], dtype=DTYPE),
        torch.tensor([0.25, 0.36], dtype=DTYPE),
        torch.tensor([0.4, 0.6], dtype=DTYPE),
        forward_model=linear_1d, noise_scale=0.3,
    )
    xt = torch.tensor([-1.0, 0.0, 1.0], dtype=DTYPE)
    y = torch.tensor([0.5, 0.5, 0.5], dtype=DTYPE)
    ms, ls, ps = p.marginal_score(xt, T), p.likelihood_score(y, xt, T), p.posterior_score(xt, y, T)
    assert torch.allclose(ps, ms + ls, atol=TOL)
    xt_g = xt.detach().requires_grad_(True)
    (grad,) = torch.autograd.grad(p.posterior_density(xt_g, y, T).log().sum(), xt_g)
    assert torch.allclose(grad, ps, atol=TOL)


def test_gmm_marginal_score_matches_autograd_1d(sde) -> None:
    p = GMM(
        sde,
        torch.tensor([-1.0, 1.0], dtype=DTYPE),
        torch.tensor([0.25, 0.36], dtype=DTYPE),
        torch.tensor([0.4, 0.6], dtype=DTYPE),
    )
    xt = torch.tensor([-1.0, 0.0, 1.0], dtype=DTYPE)
    xt_g = xt.detach().requires_grad_(True)
    (grad,) = torch.autograd.grad(p.marginal_density(xt_g, T).log().sum(), xt_g)
    assert torch.allclose(grad, p.marginal_score(xt, T), atol=TOL)


def test_gmm_denoiser_mean_matches_tweedie_1d(sde) -> None:
    p = GMM(
        sde,
        torch.tensor([-1.0, 1.0], dtype=DTYPE),
        torch.tensor([0.25, 0.36], dtype=DTYPE),
        torch.tensor([0.4, 0.6], dtype=DTYPE),
    )
    xt = torch.tensor([-1.0, 0.0, 1.0], dtype=DTYPE)
    sqrt_ab, v = _vp_scalars(sde)
    tweedie = (xt + v * p.marginal_score(xt, T)) / sqrt_ab
    assert torch.allclose(p.denoiser_mean(xt, T), tweedie, atol=TOL)


def test_gmm_full_cov_bayes_and_autograd_2d(sde, linear_2d) -> None:
    p = GMM(
        sde,
        torch.tensor([[-1.0, 0.5], [1.0, -0.5]], dtype=DTYPE),
        torch.tensor([
            [[0.5, 0.1], [0.1, 0.3]],
            [[0.4, -0.05], [-0.05, 0.6]],
        ], dtype=DTYPE),
        torch.tensor([0.3, 0.7], dtype=DTYPE),
        forward_model=linear_2d, noise_scale=0.3,
    )
    xt = torch.tensor([[-1.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
    y = torch.tensor([[0.5], [-0.2]], dtype=DTYPE)
    ms, ls, ps = p.marginal_score(xt, T), p.likelihood_score(y, xt, T), p.posterior_score(xt, y, T)
    assert torch.allclose(ps, ms + ls, atol=TOL)
    xt_g = xt.detach().requires_grad_(True)
    (grad,) = torch.autograd.grad(p.posterior_density(xt_g, y, T).log().sum(), xt_g)
    assert torch.allclose(grad, ps, atol=TOL)
