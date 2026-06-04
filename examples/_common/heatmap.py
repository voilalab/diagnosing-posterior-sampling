r"""Reusable :math:`(t, x)` density heatmaps.

These are the building block for the paper's headline figures (the
``True posterior`` / FSR / method columns of the linear and nonlinear
density grids): each renders posterior or marginal density as a 2-D map of
diffusion time :math:`t` against state :math:`x_t`.

Two entry points:

* :func:`heatmap_from_density` evaluates an analytic density on a
  caller-supplied :math:`(t, x)` grid.
* :func:`heatmap_from_samples` bins a :math:`(N_T, N_P)` particle trajectory
  per time slice into an empirical density.

Both share a single rendering primitive (:func:`_render_heatmap`) so the
visual style is identical across analytic and empirical heatmaps.

Contract: in both variants the resulting heatmap row at time :math:`t` is a
density integrating to 1 in :math:`x`. Variant 1 trusts the caller and
optionally checks via ``assert_normalized=True``; variant 2 normalises by
construction. Pre-project :math:`d > 1` data to a scalar before calling
:func:`heatmap_from_samples` -- this module never sees vector states.

rcParams are scoped via :func:`matplotlib.pyplot.rc_context` inside each
render call, so importing this module does not pollute global plotting state.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize, PowerNorm

# Style applied via rc_context inside each render -- never via global rcParams.
_RCPARAMS: dict[str, Any] = {
    "font.family": "serif",
    "font.size": 8,
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "lines.linewidth": 0.8,
    "lines.markersize": 2.5,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "image.cmap": "magma",
}


OverlayCurve = tuple[np.ndarray, np.ndarray, dict[str, Any]]
"""``(t_array, x_array, plot_kwargs)`` triple drawn on top of the heatmap."""


def _edges(centers: np.ndarray) -> np.ndarray:
    """Return ``len(centers)+1`` cell edges by midpointing.

    For a uniformly-spaced ``centers`` array originally derived from
    ``0.5 * (edges[:-1] + edges[1:])``, this exactly recovers the source
    ``edges`` array.

    Args:
        centers: shape ``(n,)`` cell centers, ``n >= 2``.

    Returns:
        Shape ``(n+1,)`` cell edges.
    """
    inner = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate([
        [centers[0] - 0.5 * (centers[1] - centers[0])],
        inner,
        [centers[-1] + 0.5 * (centers[-1] - centers[-2])],
    ])


def _render_heatmap(
    heat: np.ndarray,
    t_centers: np.ndarray,
    x_centers: np.ndarray,
    *,
    out_path: Path,
    norm: Normalize | None = None,
    cmap: str | None = None,
    overlays: list[OverlayCurve] | None = None,
    figsize: tuple[float, float] = (3.6, 2.6),
    xlabel: str = r"$t$",
    ylabel: str = r"$x_t$",
    cbar_label: str = r"$p(x_t)$",
    extra_savefig_paths: list[Path] | None = None,
) -> None:
    r"""Render a :math:`(t, x)` density heatmap to disk.

    Style is applied via :func:`matplotlib.pyplot.rc_context` so global
    rcParams are not modified.

    Args:
        heat: shape ``(N_T, N_X)`` non-negative density grid.
        t_centers: shape ``(N_T,)`` time-axis cell centers.
        x_centers: shape ``(N_X,)`` x-axis cell centers.
        out_path: primary output file path; the extension picks the format.
        norm: matplotlib color normalisation. Defaults to
            ``PowerNorm(gamma=0.55, vmin=0, vmax=heat.max())``.
        cmap: matplotlib colormap name. Defaults to ``None``, which falls back
            to ``_RCPARAMS["image.cmap"]`` -- so editing that key changes the
            colormap globally for this module.
        overlays: optional list of :data:`OverlayCurve` triples drawn over
            the heatmap. Disabled (``None``) by default.
        figsize: figure size in inches.
        xlabel: x-axis label (RST math).
        ylabel: y-axis label (RST math).
        cbar_label: colorbar label (RST math).
        extra_savefig_paths: additional paths to write the same figure to,
            useful for emitting both ``.pdf`` and ``.png`` from one render.
    """
    if norm is None:
        norm = PowerNorm(gamma=0.55, vmin=0.0, vmax=float(heat.max()))

    t_edges = _edges(t_centers)
    x_edges = _edges(x_centers)

    with plt.rc_context(_RCPARAMS):
        fig, ax = plt.subplots(figsize=figsize, dpi=200)
        # cmap=None -> pcolormesh uses rcParams["image.cmap"], which is the
        # value set in _RCPARAMS above.
        mesh = ax.pcolormesh(
            t_edges, x_edges, heat.T,
            cmap=cmap, norm=norm, shading="flat", rasterized=True,
        )

        if overlays:
            for t_arr, x_arr, style in overlays:
                ax.plot(t_arr, x_arr, **style)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(float(t_edges[0]), float(t_edges[-1]))
        ax.set_ylim(float(x_edges[0]), float(x_edges[-1]))

        cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.045)
        cbar.set_label(cbar_label)
        # matplotlib stub quirk: ty mis-types Colorbar.outline; valid at runtime.
        cbar.outline.set_linewidth(0.5)  # ty: ignore[call-non-callable]
        cbar.ax.tick_params(width=0.4)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        if extra_savefig_paths:
            for p in extra_savefig_paths:
                p.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(p)
        plt.close(fig)


def heatmap_from_density(
    density_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    t_grid: torch.Tensor,
    x_grid: torch.Tensor,
    *,
    out_path: Path,
    assert_normalized: bool = False,
    **render_kwargs: Any,
) -> None:
    r"""Render a heatmap of an analytic density on a :math:`(t, x)` grid.

    Args:
        density_fn: callable mapping ``(t_grid, x_grid)`` to a
            ``(N_T, N_X)`` density tensor :math:`p(x_t)`. Must be normalised
            in :math:`x` at every :math:`t`; the caller owns this.
        t_grid: shape ``(N_T,)`` time-axis cell centers.
        x_grid: shape ``(N_X,)`` x-axis cell centers (uniform spacing).
        out_path: output file path.
        assert_normalized: if True, raises :class:`ValueError` when
            :math:`\int p(x_t)\, dx` deviates from 1 by more than ``1e-3``
            at any :math:`t`. Useful as a debug check; off by default.
        **render_kwargs: forwarded to :func:`_render_heatmap`
            (``norm``, ``cmap``, ``overlays``, ``figsize``,
            ``extra_savefig_paths``, axis labels).

    Raises:
        ValueError: when ``assert_normalized=True`` and ``density_fn`` is
            not a density in :math:`x`.
    """
    pdf = density_fn(t_grid, x_grid)
    heat = pdf.detach().cpu().numpy()
    t_np = t_grid.detach().cpu().numpy()
    x_np = x_grid.detach().cpu().numpy()

    if assert_normalized:
        dx = float(x_grid[1] - x_grid[0])
        mass_err = float(np.max(np.abs(heat.sum(axis=1) * dx - 1.0)))
        if mass_err > 1e-3:
            raise ValueError(
                f"density_fn output not normalised in x: "
                f"max |int p dx - 1| = {mass_err:.2e}",
            )

    _render_heatmap(heat, t_np, x_np, out_path=out_path, **render_kwargs)


def heatmap_from_samples(
    trajectory: torch.Tensor,
    times: torch.Tensor,
    *,
    out_path: Path,
    n_x: int | None = None,
    x_range: tuple[float, float] | None = None,
    quantile_clip: tuple[float, float] = (0.001, 0.999),
    **render_kwargs: Any,
) -> None:
    r"""Render a heatmap from a particle trajectory by per-slice histogramming.

    Each row of the output is the empirical density of particles at that time,
    normalised by ``N_P * bin_width`` so it integrates to 1 in :math:`x`.

    Args:
        trajectory: shape ``(N_T, N_P)`` scalar particle positions per time.
            Pre-project :math:`d > 1` data to a scalar before calling.
        times: shape ``(N_T,)`` time of each slice.
        out_path: output file path.
        n_x: number of x bins. Defaults to
            ``min(round(sqrt(N_P) * 4), 400)``.
        x_range: explicit ``(x_lo, x_hi)`` bin range. When omitted, defaults
            to robust quantiles ``quantile_clip`` over all samples to avoid
            outlier-induced range stretching.
        quantile_clip: ``(lower, upper)`` quantiles for auto-range. Used only
            when ``x_range`` is None. Default ``(0.001, 0.999)``.
        **render_kwargs: forwarded to :func:`_render_heatmap`
            (``norm``, ``cmap``, ``overlays``, ``figsize``,
            ``extra_savefig_paths``, axis labels).

    Raises:
        ValueError: when ``trajectory`` is not 2-D, or ``times`` does not
            match its first dimension.
    """
    if trajectory.ndim != 2:
        raise ValueError(
            f"trajectory must be (N_T, N_P); got shape {tuple(trajectory.shape)}. "
            f"Pre-project d>1 data to a scalar before calling.",
        )
    if times.ndim != 1 or times.shape[0] != trajectory.shape[0]:
        raise ValueError(
            f"times must have shape (N_T,) matching trajectory[0]; got "
            f"times {tuple(times.shape)} vs trajectory {tuple(trajectory.shape)}.",
        )

    traj_np = trajectory.detach().cpu().numpy()
    times_np = times.detach().cpu().numpy()
    n_t, n_p = traj_np.shape

    if n_x is None:
        n_x = int(min(round(math.sqrt(n_p) * 4), 400))

    if x_range is None:
        flat = traj_np.reshape(-1)
        lo_q, hi_q = np.quantile(flat, list(quantile_clip)).tolist()
        x_lo, x_hi = float(lo_q), float(hi_q)
    else:
        x_lo, x_hi = x_range

    edges = np.linspace(x_lo, x_hi, n_x + 1)
    bin_width = float(edges[1] - edges[0])
    centers = 0.5 * (edges[:-1] + edges[1:])

    heat = np.empty((n_t, n_x), dtype=np.float64)
    for k in range(n_t):
        counts, _ = np.histogram(traj_np[k], bins=edges)
        heat[k] = counts / (n_p * bin_width)

    _render_heatmap(heat, times_np, centers, out_path=out_path, **render_kwargs)
