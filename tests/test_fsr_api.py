"""Smoke tests for :func:`src.fsr.run_fsr`.

Validates shape-checking and that the empirical posterior on a closed-form
Gaussian / linear problem matches the analytic posterior's first two
moments to within sampling error.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.forward_model import AffineForwardModel
from src.fsr import run_fsr


def test_run_fsr_returns_correct_shape() -> None:
    torch.manual_seed(0)
    atoms = torch.randn(256, 1, dtype=torch.float64)
    fm = AffineForwardModel(
        matrix=torch.tensor(1.0, dtype=torch.float64),
        bias=torch.tensor(0.0, dtype=torch.float64),
        name="identity",
    )
    y = torch.tensor([0.0], dtype=torch.float64)
    out = run_fsr(atoms, y, fm, noise_scale=0.5, num_steps=50, num_particles=64)
    assert out.shape == (64, 1)
    assert torch.isfinite(out).all()


def test_run_fsr_rejects_bad_shapes() -> None:
    torch.manual_seed(0)
    atoms_1d = torch.randn(256, dtype=torch.float64)                        # 1-D
    atoms_ok = torch.randn(256, 1, dtype=torch.float64)
    fm = AffineForwardModel(
        matrix=torch.tensor(1.0, dtype=torch.float64),
        bias=torch.tensor(0.0, dtype=torch.float64),
        name="identity",
    )
    y_1d = torch.tensor([0.0], dtype=torch.float64)
    y_2d = torch.tensor([[0.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="atoms"):
        run_fsr(atoms_1d, y_1d, fm, noise_scale=0.5)
    with pytest.raises(ValueError, match="y must be"):
        run_fsr(atoms_ok, y_2d, fm, noise_scale=0.5)
    with pytest.raises(ValueError, match="noise_scale"):
        run_fsr(atoms_ok, y_1d, fm, noise_scale=-1.0)


def test_run_fsr_recovers_gaussian_linear_posterior() -> None:
    r"""Empirical posterior mean / std match the analytic Gaussian-linear posterior.

    Prior :math:`\mathcal N(\mu_0, \sigma_0^2)`, identity forward, observation
    :math:`y`, noise :math:`\sigma_n`.  The analytic posterior is
    :math:`\mathcal N(\mu_p, \sigma_p^2)` with
    :math:`\sigma_p^2 = 1/(1/\sigma_0^2 + 1/\sigma_n^2)` and
    :math:`\mu_p = \sigma_p^2 (\mu_0/\sigma_0^2 + y/\sigma_n^2)`.
    """
    mu0, var0 = 1.0, 0.25
    sigma_n = 0.3
    y_val = 0.7

    torch.manual_seed(0)
    atoms = torch.randn(4096, 1, dtype=torch.float64) * math.sqrt(var0) + mu0
    fm = AffineForwardModel(
        matrix=torch.tensor(1.0, dtype=torch.float64),
        bias=torch.tensor(0.0, dtype=torch.float64),
        name="identity",
    )
    y = torch.tensor([y_val], dtype=torch.float64)

    samples = run_fsr(
        atoms, y, fm, noise_scale=sigma_n,
        num_steps=500, num_particles=1024, seed=42,
    )

    var_p = 1.0 / (1.0 / var0 + 1.0 / (sigma_n * sigma_n))
    mu_p = var_p * (mu0 / var0 + y_val / (sigma_n * sigma_n))

    emp_mean = float(samples.mean().item())
    emp_std = float(samples.std().item())

    # Tolerances are generous because we're not at t = 0 and use finite
    # particles; this catches gross regressions, not micro-precision.
    assert abs(emp_mean - mu_p) < 0.05, f"mean: empirical {emp_mean:.3f} vs analytic {mu_p:.3f}"
    assert abs(emp_std - math.sqrt(var_p)) < 0.05, (
        f"std: empirical {emp_std:.3f} vs analytic {math.sqrt(var_p):.3f}"
    )
