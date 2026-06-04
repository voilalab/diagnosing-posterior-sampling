r"""Paper-style :math:`(t, x)` posterior-density heatmap.

Reproduces the *kind* of figure that forms the columns of the paper's linear
multimodal density grid (Fig. 2): the analytic posterior :math:`p(x_t \mid y)`
of a multimodal prior, rendered as a 2-D map over diffusion time :math:`t` and
state :math:`x_t`.  This is the "True posterior" / FSR column; swapping in an
approximation's posterior-marginal density would build the remaining columns.

It uses :func:`examples._common.heatmap.heatmap_from_density` together with the
closed-form :meth:`src.distributions.discrete.Discrete.posterior_density`, so no
sampling is needed -- the density is evaluated directly on a Cartesian grid.

Run from the repository root::

    uv run python -m examples.heatmap_example
"""

from __future__ import annotations

from pathlib import Path

import torch

from examples._common.heatmap import heatmap_from_density
from src.distributions.discrete import Discrete
from src.forward_model import AffineForwardModel
from src.sde import VPSDE


def main() -> None:
    """Render the analytic posterior of a three-atom prior as a (t, x) heatmap."""
    sde = VPSDE()
    dtype = torch.float64

    atoms_support = torch.tensor([-1.8, 0.2, 2.2], dtype=dtype)
    weights = torch.tensor([1.0, 1.0, 1.0], dtype=dtype)
    forward = AffineForwardModel(
        matrix=torch.tensor(1.0, dtype=dtype),
        bias=torch.tensor(0.0, dtype=dtype),
        name="identity",
    )
    noise_scale = 0.3
    prior = Discrete(sde, atoms_support, weights, forward_model=forward, noise_scale=noise_scale)
    y_value = -1.8

    t_grid = torch.linspace(1e-2, 1.0, 200, dtype=dtype)
    x_grid = torch.linspace(-4.0, 4.0, 400, dtype=dtype)

    def density_fn(t_axis: torch.Tensor, x_axis: torch.Tensor) -> torch.Tensor:
        """Posterior density ``p(x_t | y)`` on the ``(t, x)`` grid -> (N_T, N_X)."""
        y_b = torch.full_like(x_axis, y_value)
        rows = [prior.posterior_density(x_axis, y_b, float(t)) for t in t_axis.tolist()]
        return torch.stack(rows, dim=0)

    out = Path(__file__).parent / "heatmap_example.pdf"
    heatmap_from_density(
        density_fn, t_grid, x_grid,
        out_path=out,
        cbar_label=r"$p(x_t \mid y)$",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
