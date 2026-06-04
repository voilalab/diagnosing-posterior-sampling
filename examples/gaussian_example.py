"""Gaussian prior + FSR posterior + analytic posterior overlay.

Defines a 1-D Gaussian prior, draws empirical samples from it, runs
:func:`src.fsr.run_fsr`, and overlays the empirical posterior histogram
against the closed-form ``Gaussian.posterior_density``.

Run from the repository root::

    uv run python -m examples.gaussian_example
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from examples._common.plotting import empirical_pdf, setup_style
from src.distributions.gaussian import Gaussian
from src.forward_model import AffineForwardModel
from src.fsr import run_fsr
from src.sde import VPSDE

setup_style()


def main() -> None:
    """Build a Gaussian prior, sample posterior via FSR, plot the overlay."""
    sde = VPSDE()
    dtype = torch.float64

    mean = torch.tensor(0.5, dtype=dtype)
    cov = torch.tensor(0.6, dtype=dtype)
    forward = AffineForwardModel(
        matrix=torch.tensor(1.2, dtype=dtype),
        bias=torch.tensor(-0.1, dtype=dtype),
        name="affine",
    )
    noise_scale = 0.25
    prior = Gaussian(sde, mean, cov, forward_model=forward, noise_scale=noise_scale)
    y_value = 0.8
    y = torch.tensor([y_value], dtype=dtype)

    torch.manual_seed(0)
    atoms = prior.prior_sampler(2048).unsqueeze(-1)                         # (N, 1)
    samples = run_fsr(
        atoms, y, forward, noise_scale=noise_scale,
        num_steps=500, num_particles=4096, seed=42,
    )

    t_eval = 1e-2
    x_grid = torch.linspace(-2.5, 3.5, 400, dtype=dtype)
    y_b = torch.full_like(x_grid, y_value)
    analytic = prior.posterior_density(x_grid, y_b, t_eval)
    empirical = empirical_pdf(samples, x_grid)

    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    ax.plot(x_grid.numpy(), analytic.numpy(), label=f"analytic at $t={t_eval}$")
    ax.plot(x_grid.numpy(), empirical.numpy(), label="FSR samples", alpha=0.85)
    ax.axvline(y_value, linestyle="--", color="gray", linewidth=0.6, label="$y$")
    ax.set_xlabel("$x_0$")
    ax.set_ylabel("density")
    ax.legend()

    out = Path(__file__).parent / "gaussian_example.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
