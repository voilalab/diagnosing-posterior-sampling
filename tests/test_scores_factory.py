"""Guard tests for ``make_score_fn`` dispatch over all supported kinds."""

from __future__ import annotations

from typing import Literal, cast

import pytest
import torch

from src.forward_model import ForwardModel
from src.scores import make_score_fn
from src.weights import prior_terms

KindName = Literal["fsr", "sigma_dps", "tmpd", "pigdm"]
KINDS: tuple[KindName, ...] = ("fsr", "sigma_dps", "tmpd", "pigdm")


def _build_args() -> dict:
    atoms = torch.tensor([[-1.0], [1.0]], dtype=torch.float64)
    forward_model = ForwardModel(
        fn=lambda x: x,
        derivative=lambda x: torch.ones_like(x),
        name="identity",
        is_linear=True,
    )

    def prior_score_fn(x_t: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        _, _, score = prior_terms(x_t, times, atoms, 1e-2, 2.0)
        return score

    return {
        "prior_score_fn": prior_score_fn,
        "beta_min": 1e-2,
        "beta_max": 2.0,
        "measurement": 0.5,
        "noise_variance": 0.1,
        "forward_model": forward_model,
        "atoms": atoms,
    }


@pytest.mark.parametrize("kind", KINDS)
def test_make_score_fn_returns_callable(kind: KindName) -> None:
    """Every recognised ``kind`` yields a callable."""
    fn = make_score_fn(kind, **_build_args())
    assert callable(fn)


def test_make_score_fn_rejects_unknown_kind() -> None:
    """Unknown kinds raise ``ValueError`` at construction time."""
    with pytest.raises(ValueError, match="Unknown kind"):
        make_score_fn(cast(KindName, "does_not_exist"), **_build_args())


def test_make_score_fn_fsr_requires_atoms() -> None:
    """``"fsr"`` raises when ``atoms`` is not provided."""
    args = _build_args()
    args.pop("atoms")
    with pytest.raises(ValueError, match="atoms"):
        make_score_fn("fsr", **args)


def test_make_score_fn_tmpd_raises_for_nonlinear() -> None:
    """``"tmpd"`` raises at construction when forward model is not linear."""
    args = _build_args()
    args["forward_model"] = ForwardModel(
        fn=lambda x: x.pow(2),
        derivative=lambda x: 2.0 * x,
        name="x^2",
        is_linear=False,
    )
    with pytest.raises(ValueError, match="linear"):
        make_score_fn("tmpd", **args)


def test_make_score_fn_pigdm_raises_for_nonlinear() -> None:
    """``"pigdm"`` raises at construction when forward model is not linear."""
    args = _build_args()
    args["forward_model"] = ForwardModel(
        fn=lambda x: x.pow(2),
        derivative=lambda x: 2.0 * x,
        name="x^2",
        is_linear=False,
    )
    with pytest.raises(ValueError, match="linear"):
        make_score_fn("pigdm", **args)
