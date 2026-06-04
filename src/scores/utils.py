"""Score utility functions."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from src.forward_model import ForwardModel


def make_score_fn(
    kind: Literal["fsr", "sigma_dps", "tmpd", "pigdm"],
    prior_score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    beta_min: float,
    beta_max: float,
    measurement: float | torch.Tensor,
    noise_variance: float | torch.Tensor,
    forward_model: ForwardModel,
    *,
    atoms: torch.Tensor | None = None,
    clamp_denominator: bool = False,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return a posterior score function for the given approximation kind.

    Args:
        kind: one of ``"fsr"``, ``"sigma_dps"``, ``"tmpd"``, ``"pigdm"``.
        prior_score_fn: callable ``(x_t, times) -> (B, D)`` returning
            the prior marginal score.
        beta_min: minimum noise-schedule rate.
        beta_max: maximum noise-schedule rate.
        measurement: observed measurement ``y``.
        noise_variance: observation noise variance, scalar or ``(m,)`` diagonal.
        forward_model: measurement operator.
        atoms: ``(N, D)`` training atoms; required for ``"fsr"``, ignored
            otherwise.
        clamp_denominator: clamp ``(1 - alpha)`` near ``t = 0``; used by
            ``"fsr"``.

    Returns:
        Callable ``(x_t, times) -> posterior_score``.

    Raises:
        ValueError: if ``kind`` is unrecognised, if ``atoms`` is required
            but not provided, or if ``"tmpd"``/``"pigdm"`` receive a
            non-linear ``forward_model``.
    """
    from src.scores.dps import SigmaDPS
    from src.scores.fsr import FSR
    from src.scores.pigdm import PiGDM
    from src.scores.tmpd import TMPD

    if kind == "sigma_dps":
        obj = SigmaDPS(prior_score_fn, beta_min, beta_max, noise_variance, forward_model)
        return lambda x_t, times: obj.posterior_score(
            measurement, x_t, times, prior_score_fn
        )

    if kind == "tmpd":
        obj = TMPD(prior_score_fn, beta_min, beta_max, noise_variance, forward_model)
        return lambda x_t, times: obj.posterior_score(
            measurement, x_t, times, prior_score_fn
        )

    if kind == "pigdm":
        obj = PiGDM(prior_score_fn, beta_min, beta_max, noise_variance, forward_model)
        return lambda x_t, times: obj.posterior_score(
            measurement, x_t, times, prior_score_fn
        )

    if kind == "fsr":
        if atoms is None:
            raise ValueError("'fsr' requires atoms to be provided")
        obj = FSR(
            prior_score_fn,
            atoms,
            beta_min,
            beta_max,
            noise_variance,
            forward_model,
            clamp_denominator=clamp_denominator,
        )
        return lambda x_t, times: obj.posterior_score(
            measurement, x_t, times, prior_score_fn
        )

    raise ValueError(
        f"Unknown kind {kind!r}. Valid options: 'fsr', 'sigma_dps', 'tmpd', 'pigdm'."
    )
