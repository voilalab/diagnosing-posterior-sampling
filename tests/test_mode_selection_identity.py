"""Smoke tests for identity-mode score functions."""

import torch

from src.forward_model import ForwardModel
from src.scores import FSR, SigmaDPS
from src.weights import prior_terms


def _make_args() -> dict:
    training_data = torch.tensor([[-2.0], [2.0]], dtype=torch.float64)
    x_t = torch.linspace(-3.0, 3.0, 31, dtype=torch.float64).unsqueeze(1)
    times = torch.full((x_t.shape[0],), 1.0, dtype=torch.float64)
    forward_model = ForwardModel(
        fn=lambda x: x,
        derivative=lambda x: torch.ones_like(x),
        name="identity",
        is_linear=True,
    )

    def prior_score_fn(xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _, _, score = prior_terms(xt, t, training_data, 1e-2, 2.0)
        return score

    return {
        "training_data": training_data,
        "x_t": x_t,
        "times": times,
        "forward_model": forward_model,
        "prior_score_fn": prior_score_fn,
    }


def test_identity_scores_are_finite() -> None:
    args = _make_args()
    x_t = args["x_t"]
    times = args["times"]
    training_data = args["training_data"]
    forward_model = args["forward_model"]
    prior_score_fn = args["prior_score_fn"]

    fsr_obj = FSR(
        prior_score_fn=prior_score_fn,
        atoms=training_data,
        beta_min=1e-2,
        beta_max=2.0,
        noise_variance=0.1,
        forward_model=forward_model,
    )
    fsr_score = fsr_obj.posterior_score(
        torch.tensor(1.0, dtype=torch.float64), x_t, times, prior_score_fn
    )

    sigma_dps_obj = SigmaDPS(
        prior_score_fn=prior_score_fn,
        beta_min=1e-2,
        beta_max=2.0,
        noise_variance=0.1,
        forward_model=forward_model,
    )
    sigma_dps_score = sigma_dps_obj.posterior_score(
        torch.tensor(1.0, dtype=torch.float64), x_t, times, prior_score_fn
    )

    for name, score in [
        ("fsr", fsr_score),
        ("sigma_dps", sigma_dps_score),
    ]:
        assert score.shape == x_t.shape, f"{name} shape mismatch"
        assert torch.isfinite(score).all(), f"{name} has non-finite values"
