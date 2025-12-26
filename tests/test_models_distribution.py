# tests/test_models_distribution.py
import unittest

import torch as th

from models.distribution import (
    DiagGaussianDistribution,
    SquashedDiagGaussianDistribution,
)


class TestModelsDistribution(unittest.TestCase):
    def test_diag_gaussian_shapes_and_entropy(self):
        dist = DiagGaussianDistribution(action_dim=3)
        mean = th.zeros(5, 3)
        log_std = th.zeros(5, 3)
        actions, log_prob = dist.sample(mean, log_std)

        self.assertEqual(actions.shape, (5, 3))
        self.assertEqual(log_prob.shape, (5, 1))

        ent = dist.entropy(log_std)
        self.assertEqual(ent.shape, (5, 1))

    def test_squashed_gaussian_range(self):
        dist = SquashedDiagGaussianDistribution(action_dim=2)
        mean = th.zeros(4, 2)
        log_std = th.zeros(4, 2)

        actions, log_prob = dist.sample(mean, log_std)
        self.assertEqual(actions.shape, (4, 2))
        self.assertEqual(log_prob.shape, (4, 1))
        self.assertTrue(th.all(actions <= 1.0 + 1e-6))
        self.assertTrue(th.all(actions >= -1.0 - 1e-6))

        # log_prob should be finite
        self.assertTrue(th.isfinite(log_prob).all())


if __name__ == "__main__":
    unittest.main()
