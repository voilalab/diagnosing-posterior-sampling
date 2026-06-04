import unittest

import torch

from src.samplers import em_step
from src.sde import alpha_bar_from_times


def _zero_score(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x)


def _ones_score(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(x)


def _linear_score(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return -x


def _half_linear_score(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return -0.5 * x


def _time_dependent_score(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return -x * t.unsqueeze(1)


class TestEMStepValidation(unittest.TestCase):
    def test_negative_step_size_raises(self):
        state = torch.randn(10, 2)
        times = torch.rand(10)
        with self.assertRaises(ValueError):
            em_step(state, times, _zero_score, step_size=-0.1)

    def test_zero_step_size_raises(self):
        state = torch.randn(10, 2)
        times = torch.rand(10)
        with self.assertRaises(ValueError):
            em_step(state, times, _zero_score, step_size=0.0)

    def test_1d_state_raises(self):
        state = torch.randn(10)
        times = torch.rand(10)
        with self.assertRaises(ValueError):
            em_step(state, times, _zero_score, step_size=0.01)

    def test_3d_state_raises(self):
        state = torch.randn(10, 2, 3)
        times = torch.rand(10)
        with self.assertRaises(ValueError):
            em_step(state, times, _zero_score, step_size=0.01)

    def test_2d_times_raises(self):
        state = torch.randn(10, 2)
        times = torch.rand(10, 1)
        with self.assertRaises(ValueError):
            em_step(state, times, _zero_score, step_size=0.01)

    def test_mismatched_batch_sizes_raises(self):
        state = torch.randn(10, 2)
        times = torch.rand(5)
        with self.assertRaises(ValueError):
            em_step(state, times, _zero_score, step_size=0.01)


class TestEMStepShape(unittest.TestCase):
    def test_output_shape_preserved(self):
        state = torch.randn(10, 3)
        times = torch.rand(10)
        result = em_step(state, times, _zero_score, step_size=0.01)
        self.assertEqual(result.shape, state.shape)

    def test_output_shape_1d_samples(self):
        state = torch.randn(20, 1)
        times = torch.rand(20)
        result = em_step(state, times, _zero_score, step_size=0.01)
        self.assertEqual(result.shape, (20, 1))

    def test_output_shape_high_dim(self):
        state = torch.randn(5, 100)
        times = torch.rand(5)
        result = em_step(state, times, _zero_score, step_size=0.01)
        self.assertEqual(result.shape, (5, 100))


class TestEMStepToyScores(unittest.TestCase):
    """Tests with simple analytical score functions"""

    def test_zero_score_no_nan(self):
        """Zero score should produce no NaN (pure diffusion)"""
        state = torch.randn(100, 2)
        times = torch.rand(100)
        result = em_step(state, times, _zero_score, step_size=0.01)
        self.assertFalse(torch.isnan(result).any())

    def test_constant_score_no_nan(self):
        """Constant score should produce no NaN"""
        state = torch.randn(100, 2)
        times = torch.rand(100)
        result = em_step(state, times, _ones_score, step_size=0.01)
        self.assertFalse(torch.isnan(result).any())

    def test_linear_score_no_nan(self):
        """Linear score (Gaussian) should produce no NaN"""
        state = torch.randn(100, 2)
        times = torch.rand(100)
        result = em_step(state, times, _linear_score, step_size=0.01)
        self.assertFalse(torch.isnan(result).any())

    def test_scaled_linear_score_no_nan(self):
        """Scaled linear score should produce no NaN"""
        state = torch.randn(100, 3)
        times = torch.rand(100)
        result = em_step(state, times, _half_linear_score, step_size=0.01)
        self.assertFalse(torch.isnan(result).any())

    def test_time_dependent_score_no_nan(self):
        """Time-dependent score should produce no NaN"""
        state = torch.randn(100, 2)
        times = torch.rand(100)
        result = em_step(state, times, _time_dependent_score, step_size=0.01)
        self.assertFalse(torch.isnan(result).any())

    def test_various_step_sizes_no_nan(self):
        """Various step sizes should produce no NaN"""
        state = torch.randn(50, 2)
        times = torch.rand(50)
        for step_size in [1e-5, 1e-3, 1e-2, 0.1]:
            result = em_step(state, times, _linear_score, step_size=step_size)
            self.assertFalse(torch.isnan(result).any(), f"NaN with step_size={step_size}")

    def test_extreme_times_no_nan(self):
        """Edge case times (near 0 and 1) should produce no NaN"""
        state = torch.randn(50, 2)
        times = torch.tensor([0.0] * 25 + [1.0] * 25)
        result = em_step(state, times, _linear_score, step_size=0.01)
        self.assertFalse(torch.isnan(result).any())


class TestEMStepMultipleSteps(unittest.TestCase):
    """Tests that multiple sequential steps work without NaN"""

    def test_multiple_steps_toy_no_nan(self):
        """Multiple EM steps with toy score should not accumulate NaN"""
        state = torch.randn(50, 2)
        times = torch.ones(50) * 0.5
        for _ in range(100):
            state = em_step(state, times, _linear_score, step_size=0.01)
            self.assertFalse(torch.isnan(state).any())

    def test_multiple_steps_linear_score_no_nan(self):
        """Multiple EM steps with linear score should not accumulate NaN"""
        state = torch.randn(50, 2)
        times = torch.ones(50) * 0.5
        for i in range(100):
            state = em_step(state, times, _half_linear_score, step_size=0.01)
            self.assertFalse(torch.isnan(state).any(), f"NaN at step {i}")

    def test_reverse_diffusion_trajectory_no_nan(self):
        """Simulate reverse diffusion trajectory (decreasing times)"""
        state = torch.randn(50, 2)
        step_size = 0.01
        for i in range(50):
            t = 1.0 - i * step_size
            times = torch.ones(50) * t
            state = em_step(state, times, _linear_score, step_size=step_size)
            self.assertFalse(torch.isnan(state).any(), f"NaN at step {i}, t={t}")


class TestEMStepBetaParams(unittest.TestCase):
    """Tests with various beta scheduling parameters"""

    def test_custom_beta_min_max_no_nan(self):
        """Custom beta_min and beta_max should produce no NaN"""
        state = torch.randn(50, 2)
        times = torch.rand(50)
        result = em_step(state, times, _linear_score, step_size=0.01, beta_min=0.1, beta_max=10)
        self.assertFalse(torch.isnan(result).any())

    def test_small_beta_range_no_nan(self):
        """Small beta range should produce no NaN"""
        state = torch.randn(50, 2)
        times = torch.rand(50)
        result = em_step(state, times, _linear_score, step_size=0.01, beta_min=0.1, beta_max=0.5)
        self.assertFalse(torch.isnan(result).any())

    def test_large_beta_range_no_nan(self):
        """Large beta range should produce no NaN"""
        state = torch.randn(50, 2)
        times = torch.rand(50)
        result = em_step(state, times, _linear_score, step_size=0.001, beta_min=0.01, beta_max=50)
        self.assertFalse(torch.isnan(result).any())


class TestEMStepConvergence(unittest.TestCase):
    """Correctness tests: reverse-SDE trajectories converge to known distributions."""

    _BETA_MIN = 0.01
    _BETA_MAX = 20.0
    _N = 2000
    _STEP_SIZE = 0.01
    _T_START = 1.0
    _T_END = 0.02

    def _run_reverse(self, score_fn):
        torch.manual_seed(42)
        state = torch.randn(self._N, 2)
        t = self._T_START
        while t > self._T_END + 1e-9:
            times = torch.full((self._N,), t)
            state = em_step(state, times, score_fn, self._STEP_SIZE,
                            self._BETA_MIN, self._BETA_MAX)
            t -= self._STEP_SIZE
        return state

    def test_convergence_to_gaussian_prior(self):
        """Reverse-SDE with Gaussian prior score converges to N(mu, I)."""
        mu = torch.tensor([[2.0, -1.0]])

        def score_fn(x_t, times):
            alpha = alpha_bar_from_times(times, self._BETA_MIN, self._BETA_MAX)
            mean_t = alpha.sqrt().unsqueeze(1) * mu
            return -(x_t - mean_t)

        final = self._run_reverse(score_fn)
        emp_mean = final.mean(0)
        emp_std = final.std(0)
        for i, m in enumerate([2.0, -1.0]):
            self.assertAlmostEqual(emp_mean[i].item(), m, delta=0.15,
                                   msg=f"dim {i}: mean {emp_mean[i]:.3f} not close to {m}")
        for i in range(2):
            self.assertGreater(emp_std[i].item(), 0.7, f"dim {i}: std too small")
            self.assertLess(emp_std[i].item(), 1.4, f"dim {i}: std too large")

    def test_convergence_to_gaussian_posterior(self):
        """Reverse-SDE with exact posterior score converges to N(mu_post, Sigma_post).

        Prior: x ~ N(0, I).  Measurement: y = A x + eps, eps ~ N(0, sigma^2 I).
        A = [[1, 0]], sigma^2 = 0.01, y = [3.0].
        Posterior mean: mu_post ~ [2.97, 0].
        """
        sigma2 = 0.01
        # a_mat = [[1, 0]]: only first dim observed
        a_mat = torch.tensor([[1.0, 0.0]])   # (1, 2)
        y = torch.tensor([3.0])              # (1,)
        ata = a_mat.T @ a_mat                # (2, 2)

        # Analytical posterior for reference only (not used in score)
        sigma_post = torch.linalg.inv(torch.eye(2) + ata / sigma2)
        mu_post = sigma_post @ (a_mat.T @ y) / sigma2  # (2,)

        def score_fn(x_t, times):
            # Prior score for N(0,I): p_t(x) = N(0, I) => score = -x
            prior_score = -x_t
            # Exact likelihood score via Gaussian integral over x0 | x_t
            alpha = alpha_bar_from_times(times, self._BETA_MIN, self._BETA_MAX)  # (B,)
            sqrt_a = alpha.sqrt().unsqueeze(1)  # (B, 1)
            one_minus_a = (1.0 - alpha)  # (B,)
            # Sigma_t = (1-alpha) a_mat a_mat^T + sigma^2 I  -- scalar for 1D obs
            sigma_t = one_minus_a * (a_mat @ a_mat.T).squeeze() + sigma2  # (B,)
            # residual = y - a_mat sqrt_a x_t : (B, 1)
            residual = y.unsqueeze(0) - (x_t @ a_mat.T) * sqrt_a  # (B, 1)
            # grad = sqrt_a * a_mat^T * Sigma_t^{-1} * residual
            lik_score = sqrt_a * (residual / sigma_t.unsqueeze(1)) @ a_mat  # (B, 2)
            return prior_score + lik_score

        final = self._run_reverse(score_fn)
        emp_mean = final.mean(0)
        emp_std = final.std(0)

        self.assertAlmostEqual(emp_mean[0].item(), mu_post[0].item(), delta=0.2,
                               msg=f"dim 0: mean {emp_mean[0]:.3f} != {mu_post[0]:.3f}")
        self.assertAlmostEqual(emp_mean[1].item(), mu_post[1].item(), delta=0.15,
                               msg=f"dim 1: mean {emp_mean[1]:.3f} != {mu_post[1]:.3f}")
        self.assertGreater(emp_std[1].item(), 0.7, "dim 1 (unobserved): std too small")
        self.assertLess(emp_std[1].item(), 1.4, "dim 1 (unobserved): std too large")


if __name__ == "__main__":
    unittest.main(verbosity=2)
