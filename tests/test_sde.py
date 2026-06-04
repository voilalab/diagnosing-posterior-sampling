import math
import unittest

import torch

from src.sde import OU, VPSDE


class TestOUInit(unittest.TestCase):
    def test_default_params(self):
        ou = OU()
        self.assertEqual(ou.mean, 0)
        self.assertEqual(ou.sigma, math.sqrt(2))

    def test_custom_params(self):
        ou = OU(mean=1.0, sigma=0.5)
        self.assertEqual(ou.mean, 1.0)
        self.assertEqual(ou.sigma, 0.5)


class TestOUSemigroup(unittest.TestCase):
    def setUp(self):
        self.ou = OU()

    def test_semigroup_shape_k1(self):
        samples = torch.randn(10, 3)
        result = self.ou.semigroup(samples, t=0.1, k=1)
        self.assertEqual(result.shape, (10, 3))

    def test_semigroup_shape_k5(self):
        samples = torch.randn(10, 3)
        result = self.ou.semigroup(samples, t=0.1, k=5)
        self.assertEqual(result.shape, (50, 3))

    def test_semigroup_1d(self):
        samples = torch.randn(20, 1)
        result = self.ou.semigroup(samples, t=0.5, k=3)
        self.assertEqual(result.shape, (60, 1))


class TestVPSDEInit(unittest.TestCase):
    def test_default_params(self):
        vp = VPSDE()
        self.assertEqual(vp.beta_min, 1e-2)

    def test_custom_params(self):
        vp = VPSDE(beta_min=0.1, beta_max=10)
        self.assertEqual(vp.beta_min, 0.1)

    def test_alpha_bar_fn(self):
        vp = VPSDE()
        alpha_bar_0 = vp.alpha_bar_fn(0)
        self.assertAlmostEqual(alpha_bar_0, 1.0)


class TestVPSDESemigroup(unittest.TestCase):
    def setUp(self):
        self.vp = VPSDE()

    def test_semigroup_shape_k1(self):
        samples = torch.randn(10, 3)
        result = self.vp.semigroup(samples, t=0.1, k=1)
        self.assertEqual(result.shape, (10, 3))

    def test_semigroup_shape_k5(self):
        samples = torch.randn(10, 3)
        result = self.vp.semigroup(samples, t=0.1, k=5)
        self.assertEqual(result.shape, (50, 3))

    def test_semigroup_1d(self):
        samples = torch.randn(20, 1)
        result = self.vp.semigroup(samples, t=0.5, k=3)
        self.assertEqual(result.shape, (60, 1))


class TestSDESimulate(unittest.TestCase):
    def test_ou_simulate_shape(self):
        ou = OU()
        prior = torch.randn(5, 2)
        result = ou.simulate(prior, dt=0.1, final_t=1.0, n_samples=20)
        self.assertEqual(result.shape, (10, 20, 2))

    def test_vp_simulate_shape(self):
        vp = VPSDE()
        prior = torch.randn(5, 2)
        result = vp.simulate(prior, dt=0.1, final_t=1.0, n_samples=20)
        self.assertEqual(result.shape, (10, 20, 2))

    def test_simulate_maintains_n_samples(self):
        ou = OU()
        prior = torch.randn(3, 4)
        result = ou.simulate(prior, dt=0.1, final_t=0.5, n_samples=100)
        for step in range(result.shape[0]):
            self.assertEqual(result[step].shape[0], 100)


class TestSDEExpandingForwardProcess(unittest.TestCase):
    def test_ou_expanding_returns_list(self):
        ou = OU()
        prior = torch.randn(2, 3)
        result = ou.expanding_forward_process(prior, dt=0.1, final_t=0.3, k=2)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_ou_expanding_shapes(self):
        ou = OU()
        m, d, k = 2, 3, 2
        prior = torch.randn(m, d)
        result = ou.expanding_forward_process(prior, dt=0.1, final_t=0.3, k=k)
        for i, samples in enumerate(result):
            expected_n = m * (k ** (i + 1))
            self.assertEqual(samples.shape, (expected_n, d))

    def test_vp_expanding_shapes(self):
        vp = VPSDE()
        m, d, k = 3, 2, 2
        prior = torch.randn(m, d)
        result = vp.expanding_forward_process(prior, dt=0.1, final_t=0.2, k=k)
        self.assertEqual(result[0].shape, (m * k, d))
        self.assertEqual(result[1].shape, (m * k * k, d))


if __name__ == "__main__":
    unittest.main(verbosity=2)
