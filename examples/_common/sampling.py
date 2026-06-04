"""Generic reverse-time Euler-Maruyama loop shared by example scripts.

``run_fsr`` in :mod:`src.fsr` wraps the FSR-specific case; for the
method-comparison example we need the same loop driven by an arbitrary
posterior-score function (SigmaDPS, PiGDM, TMPD, ZetaDPS, ...).  This
module provides that thin wrapper.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from src.samplers.em import em_step
from src.sde import VPSDE


def reverse_em(
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    sde: VPSDE,
    num_particles: int,
    dim: int,
    num_steps: int,
    t_min: float = 1e-3,
    t_max: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
    seed: int | None = None,
) -> torch.Tensor:
    r"""Run reverse-time Euler-Maruyama integration of the VP-SDE.

    Args:
        score_fn: callable ``(state, times) -> (B, d)`` returning the
            posterior score :math:`\nabla_{x_t} \log p(x_t \mid y)`.
        sde (VPSDE): VP forward schedule.
        num_particles (int): number of independent particles.
        dim (int): state dimension ``d``.
        num_steps (int): number of EM steps from ``t_max`` to ``t_min``.
        t_min (float): smallest reverse-integration time (must be ``> 0``).
        t_max (float): starting reverse-integration time (must be ``> t_min``).
        dtype (torch.dtype): tensor dtype.
        device (str | torch.device): tensor device.
        seed (int | None): optional torch RNG seed.

    Returns:
        torch.Tensor: ``(num_particles, dim)`` final particle cloud at ``t_min``.
    """
    if seed is not None:
        torch.manual_seed(seed)
    dev = torch.device(device)
    schedule = np.linspace(t_max, t_min, num_steps + 1)
    state = torch.randn(num_particles, dim, dtype=dtype, device=dev)
    for k in range(num_steps):
        t_now = float(schedule[k])
        t_next = float(schedule[k + 1])
        dt = t_now - t_next
        if dt <= 0.0:
            continue
        times = torch.full((num_particles,), t_now, dtype=dtype, device=dev)
        state = em_step(state, times, score_fn, dt, sde.beta_min, sde.beta_max)
    return state
