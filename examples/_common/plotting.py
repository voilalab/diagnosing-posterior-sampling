"""Matplotlib rcParams setup and small plot helpers for the examples.

Style follows ``CLAUDE.md`` plotting guidance: thin lines, hairline spines,
serif font, no top/right spines.  ``setup_style()`` is idempotent and safe
to call from every example script.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import torch

__all__ = ["empirical_pdf", "setup_style"]


def setup_style() -> None:
    """Apply the project's matplotlib rcParams once at module load."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 0.9,
        "lines.markersize": 3,
        "legend.frameon": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def empirical_pdf(
    samples: torch.Tensor, x_grid: torch.Tensor,
) -> torch.Tensor:
    """Histogram ``samples`` onto cell-centred bins defined by ``x_grid``.

    Args:
        samples (torch.Tensor): ``(N,)`` or ``(N, 1)`` 1-D samples.
        x_grid (torch.Tensor): ``(K,)`` uniformly-spaced cell centres.

    Returns:
        torch.Tensor: ``(K,)`` density values on the same grid.
    """
    x_np = x_grid.detach().cpu().numpy()
    half = 0.5 * float(x_np[1] - x_np[0])
    edges = np.concatenate(
        [[x_np[0] - half], 0.5 * (x_np[:-1] + x_np[1:]), [x_np[-1] + half]],
    )
    bin_width = float(edges[1] - edges[0])
    counts, _ = np.histogram(
        samples.detach().cpu().numpy().reshape(-1), bins=edges,
    )
    n = samples.shape[0]
    return torch.from_numpy(counts.astype(np.float64) / (n * bin_width))
