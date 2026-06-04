"""Five-line FSR quickstart: bring your own atoms and get posterior samples.

Run from the repository root::

    uv run python -m examples.quickstart
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from examples._common.plotting import empirical_pdf, setup_style
from src.forward_model import AffineForwardModel
from src.fsr import run_fsr

setup_style()


def main() -> None:
    """Build a toy empirical prior, run FSR, save a posterior-histogram PDF."""
    torch.manual_seed(0)

    # (N, d) empirical prior: a 1-D Gaussian-ish cloud around 1.0.
    atoms = torch.randn(2048, 1, dtype=torch.float64) * 0.5 + 1.0
    forward = AffineForwardModel(
        matrix=torch.tensor(1.0, dtype=torch.float64),
        bias=torch.tensor(0.0, dtype=torch.float64),
        name="identity",
    )
    y = torch.tensor([0.7], dtype=torch.float64)

    samples = run_fsr(
        atoms, y, forward, noise_scale=0.3,
        num_steps=500, num_particles=1024, seed=42,
    )

    x_grid = torch.linspace(-1.0, 3.0, 200, dtype=torch.float64)
    pdf = empirical_pdf(samples, x_grid)

    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    ax.plot(x_grid.numpy(), pdf.numpy(), label="FSR posterior")
    ax.axvline(float(y.item()), linestyle="--", color="gray", linewidth=0.6, label="$y$")
    ax.set_xlabel("$x_0$")
    ax.set_ylabel("density")
    ax.legend()

    out = Path(__file__).parent / "quickstart.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
