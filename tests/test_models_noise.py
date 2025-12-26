# tests/test_models_noise.py
import unittest

import numpy as np

from models.noise import (
    GaussianActionNoise,
    OrnsteinUhlenbeckActionNoise,
    VectorizedActionNoise,
)


class TestModelsNoise(unittest.TestCase):
    def setUp(self) -> None:
        self.mean = np.zeros(3)
        self.sigma = np.ones(3) * 0.1

    def test_gaussian_noise_basic(self):
        noise = GaussianActionNoise(self.mean, self.sigma)
        x = noise()
        self.assertEqual(x.shape, (3,))
        # sanity: variance not all zero over multiple calls
        samples = np.stack([noise() for _ in range(100)], axis=0)
        self.assertGreater(np.std(samples, axis=0).mean(), 0.0)

    def test_ou_noise_reset_and_shape(self):
        noise = OrnsteinUhlenbeckActionNoise(self.mean, self.sigma)
        x1 = noise()
        x2 = noise()
        self.assertEqual(x1.shape, (3,))
        self.assertEqual(x2.shape, (3,))

        noise.reset()
        x3 = noise()
        self.assertEqual(x3.shape, (3,))

    def test_vectorized_noise(self):
        n_envs = 4
        base_noise = GaussianActionNoise(self.mean, self.sigma)
        vec_noise = VectorizedActionNoise(base_noise, n_envs)

        noise_samples = vec_noise()
        self.assertEqual(noise_samples.shape, (n_envs, 3))

        # Test reset on a subset of envs
        vec_noise.reset(indices=[0, 2])


if __name__ == "__main__":
    unittest.main()
