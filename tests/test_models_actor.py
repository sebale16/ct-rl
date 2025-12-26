# tests/test_models_actor.py
import unittest

import numpy as np
import torch as th
from gymnasium import spaces

from common.utils import get_device
from models.actor import StochasticActor, DeterministicActor


class TestModelsActor(unittest.TestCase):
    def setUp(self) -> None:
        self.device = get_device("auto")
        self.obs_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.act_space = spaces.Box(low=-2.0, high=2.0, shape=(2,), dtype=np.float32)
        self.obs_batch = th.zeros(8, 4).to(self.device)

    def test_stochastic_actor_shapes_and_range(self):
        actor = StochasticActor(
            observation_space=self.obs_space,
            action_space=self.act_space,
            net_arch=[32],
            device=self.device,
        )
        actions, log_prob, _ = actor(self.obs_batch, deterministic=False)
        self.assertEqual(actions.shape, (8, 2))
        self.assertEqual(log_prob.shape, (8, 1))
        # Rescaled to action_space bounds
        self.assertTrue(th.all(actions <= 2.0 + 1e-6))
        self.assertTrue(th.all(actions >= -2.0 - 1e-6))

    def test_deterministic_actor_shapes(self):
        actor = DeterministicActor(
            observation_space=self.obs_space,
            action_space=self.act_space,
            net_arch=[32],
            device=self.device,
        )
        actions = actor(self.obs_batch)
        self.assertEqual(actions.shape, (8, 2))
        self.assertTrue(th.all(actions <= 2.0 + 1e-6))
        self.assertTrue(th.all(actions >= -2.0 - 1e-6))


if __name__ == "__main__":
    unittest.main()
