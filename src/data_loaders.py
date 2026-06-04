"""Tensor-loading helpers for the ``(N, d)`` shape that :func:`src.fsr.run_fsr` expects.

Each helper coerces an external representation into a 2-D ``torch.Tensor``
with shape ``(N, d)``.  A 1-D input is interpreted as ``(N, 1)`` (i.e. ``N``
scalar samples).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

__all__ = ["from_csv", "from_numpy", "from_tensors", "from_torch_file"]


def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
    """Reshape a 1-D ``(N,)`` tensor to ``(N, 1)``; raise on higher rank."""
    if t.ndim == 1:
        return t.unsqueeze(-1)
    if t.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D tensor, got shape {tuple(t.shape)}.")
    return t


def from_numpy(
    array: np.ndarray,
    *,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Convert a NumPy array of shape ``(N,)`` or ``(N, d)`` to a torch tensor.

    Args:
        array (np.ndarray): input samples.
        dtype (torch.dtype): output dtype.  Defaults to ``torch.float64``.
        device (str | torch.device): output device.  Defaults to CPU.

    Returns:
        torch.Tensor: ``(N, d)`` tensor.
    """
    return _ensure_2d(torch.as_tensor(array, dtype=dtype, device=device))


def from_tensors(
    samples: torch.Tensor | Sequence[torch.Tensor],
    *,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Coerce a tensor or a collection of per-sample tensors into ``(N, d)``.

    Accepts either a single ``torch.Tensor`` (a ``(N,)`` or ``(N, d)`` batch) or
    a sequence of individual per-sample tensors (each ``(d,)`` or scalar), which
    are stacked along a new leading axis.  This is the "bring your own tensor(s)"
    entry point for :func:`src.fsr.run_fsr`.

    Args:
        samples (torch.Tensor | Sequence[torch.Tensor]): a batch tensor, or a
            non-empty sequence of per-sample tensors of identical shape.
        dtype (torch.dtype): output dtype.  Defaults to ``torch.float64``.
        device (str | torch.device): output device.  Defaults to CPU.

    Returns:
        torch.Tensor: ``(N, d)`` tensor.

    Raises:
        ValueError: if ``samples`` is an empty sequence.
    """
    if isinstance(samples, torch.Tensor):
        return _ensure_2d(samples.to(dtype=dtype, device=device))
    if len(samples) == 0:
        raise ValueError("samples sequence is empty; nothing to stack.")
    stacked = torch.stack([s.to(dtype=dtype, device=device) for s in samples])
    return _ensure_2d(stacked)


def from_csv(
    path: str | Path,
    *,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
    delimiter: str = ",",
) -> torch.Tensor:
    """Load samples from a CSV file with no header.

    Args:
        path (str | Path): CSV file path.
        dtype (torch.dtype): output dtype.  Defaults to ``torch.float64``.
        device (str | torch.device): output device.  Defaults to CPU.
        delimiter (str): column separator.  Defaults to ``","``.

    Returns:
        torch.Tensor: ``(N, d)`` tensor.
    """
    array = np.loadtxt(str(path), delimiter=delimiter)
    return from_numpy(array, dtype=dtype, device=device)


def from_torch_file(
    path: str | Path,
    *,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Load samples from a ``.pt`` / ``.pth`` file saved with :func:`torch.save`.

    Args:
        path (str | Path): tensor file path.
        dtype (torch.dtype): output dtype.
        device (str | torch.device): output device.

    Returns:
        torch.Tensor: ``(N, d)`` tensor.
    """
    obj = torch.load(str(path), map_location=device, weights_only=True)
    if not isinstance(obj, torch.Tensor):
        raise TypeError(
            f"Expected the torch file to contain a Tensor, got {type(obj).__name__}.",
        )
    return _ensure_2d(obj.to(dtype=dtype, device=device))
