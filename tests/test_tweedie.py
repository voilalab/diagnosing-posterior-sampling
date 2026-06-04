import unittest

import torch

from src.sde import alpha_bar_from_times
from src.tweedie import (
    tweedie_cov_isotropic,
    tweedie_jacobian,
    tweedie_mean,
)

_BETA_MIN = 0.01
_BETA_MAX = 20.0
_B = 4
_D = 5


def _linear_score(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Score proportional to -x (standard Gaussian marginal)."""
    return -x


class TestTweedieMean(unittest.TestCase):
    def test_gaussian_closed_form(self):
        """For unit-Gaussian prior under VP-SDE, E[x0|xt] = sqrt(alpha_bar) * xt."""
        torch.manual_seed(42)
        times = torch.rand(_B) * 0.85 + 0.05
        x_t = torch.randn(_B, _D)

        # Unit-Gaussian marginal => prior_score = -x_t, E[x0|xt] = sqrt(ab)*x_t
        prior_score = -x_t
        ab = alpha_bar_from_times(times, _BETA_MIN, _BETA_MAX).unsqueeze(-1)
        expected = ab.sqrt() * x_t

        result = tweedie_mean(x_t, times, prior_score, _BETA_MIN, _BETA_MAX)

        self.assertEqual(result.shape, (_B, _D))
        self.assertLess((result - expected).abs().max().item(), 1e-5)

    def test_output_shape(self):
        """Output shape matches (B, D)."""
        x_t = torch.randn(_B, _D)
        times = torch.rand(_B)
        prior_score = torch.randn(_B, _D)
        result = tweedie_mean(x_t, times, prior_score, _BETA_MIN, _BETA_MAX)
        self.assertEqual(result.shape, (_B, _D))

    def test_no_nan(self):
        """No NaN output for standard interior times."""
        torch.manual_seed(0)
        x_t = torch.randn(_B, _D)
        times = torch.rand(_B) * 0.9 + 0.05
        prior_score = torch.randn(_B, _D)
        result = tweedie_mean(x_t, times, prior_score, _BETA_MIN, _BETA_MAX)
        self.assertFalse(torch.isnan(result).any())


class TestTweedieJacobian(unittest.TestCase):
    def test_output_shape(self):
        """Output shape is (B, D, D)."""
        torch.manual_seed(0)
        x_t = torch.randn(_B, _D)
        times = torch.rand(_B) * 0.9 + 0.05
        result = tweedie_jacobian(x_t, times, _linear_score, _BETA_MIN, _BETA_MAX)
        self.assertEqual(result.shape, (_B, _D, _D))

    def test_matches_analytical_linear(self):
        """For linear score s(x,t) = x @ W.T, Jacobian = (I + (1-ab)*W) / sqrt(ab)."""
        torch.manual_seed(1)
        B, D = 3, 4
        times = torch.rand(B) * 0.85 + 0.05
        x_t = torch.randn(B, D)
        W = torch.randn(D, D)

        def linear_score_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return x @ W.T

        result = tweedie_jacobian(x_t, times, linear_score_fn, _BETA_MIN, _BETA_MAX)

        ab = alpha_bar_from_times(times, _BETA_MIN, _BETA_MAX)  # (B,)
        for i in range(B):
            expected = (torch.eye(D) + (1.0 - ab[i]) * W) / ab[i].sqrt()
            self.assertLess((result[i] - expected).abs().max().item(), 1e-4)

    def test_matches_autograd_functional_jacobian(self):
        """tweedie_jacobian agrees with torch.autograd.functional.jacobian."""
        torch.manual_seed(2)
        B, D = 2, 3
        times = torch.tensor([0.3, 0.7])
        x_t = torch.randn(B, D)

        def nonlinear_score_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return torch.tanh(x) * t.unsqueeze(-1)

        result = tweedie_jacobian(x_t, times, nonlinear_score_fn, _BETA_MIN, _BETA_MAX)

        for i in range(B):
            t_i = times[i : i + 1]

            def _mean_i(xi: torch.Tensor) -> torch.Tensor:
                s = nonlinear_score_fn(xi.unsqueeze(0), t_i).squeeze(0)
                return tweedie_mean(xi.unsqueeze(0), t_i, s.unsqueeze(0), _BETA_MIN, _BETA_MAX).squeeze(0)

            ref_jac = torch.autograd.functional.jacobian(_mean_i, x_t[i])
            self.assertLess((result[i] - ref_jac).abs().max().item(), 1e-5)


class TestTweedieCovIsotropic(unittest.TestCase):
    def test_output_shape(self):
        """Output shape is (B,)."""
        times = torch.rand(_B) * 0.9 + 0.05
        result = tweedie_cov_isotropic(times, _BETA_MIN, _BETA_MAX)
        self.assertEqual(result.shape, (_B,))

    def test_positive(self):
        """Covariance is positive for t in (0, 1)."""
        times = torch.linspace(0.05, 0.95, 20)
        result = tweedie_cov_isotropic(times, _BETA_MIN, _BETA_MAX)
        self.assertTrue((result > 0).all())

    def test_monotone_increasing(self):
        """Covariance is monotonically increasing in t (alpha_bar shrinks)."""
        times = torch.linspace(0.05, 0.95, 50)
        result = tweedie_cov_isotropic(times, _BETA_MIN, _BETA_MAX)
        diffs = result[1:] - result[:-1]
        self.assertTrue((diffs > 0).all())


if __name__ == "__main__":
    unittest.main()
