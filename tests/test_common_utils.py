# tests/test_common_utils.py

import unittest

import torch as th
from gymnasium import spaces

from common.utils import (
    get_device,
    get_obs_shape,
    get_action_dim,
    get_flattened_obs_dim,
)


class TestUtils(unittest.TestCase):
    def test_get_device_and_flattened_dim(self):
        dev = get_device("cpu")
        self.assertIsInstance(dev, th.device)
        box = spaces.Box(low=0.0, high=1.0, shape=(3, 4), dtype=float)
        self.assertEqual(get_obs_shape(box), (3, 4))
        self.assertEqual(get_flattened_obs_dim(box), 12)

    def test_get_action_dim(self):
        box = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
        disc = spaces.Discrete(5)
        self.assertEqual(get_action_dim(box), 2)
        self.assertEqual(get_action_dim(disc), 1)


if __name__ == "__main__":
    unittest.main()
