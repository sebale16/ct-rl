"""Unit contract for common.sb3_demo's SB3 demonstration warm start.

Exercises DemoSeededOffPolicyMixin._sample_action against a bare stand-in
base class (mimicking OffPolicyAlgorithm._sample_action's signature) rather
than a real SB3 model + MuJoCo env, so the demonstration-vs-fallback
branching is checked without a training run. A real base class (not a
SimpleNamespace) is required for the mixin's zero-arg ``super()`` calls to
resolve.
"""

from __future__ import annotations

import unittest
import unittest.mock

import numpy as np
from gymnasium import spaces

from common.sb3_demo import DemoSeededOffPolicyMixin, demo_seeded_class


class _FakePolicy:
    """Identity scale/unscale so the returned action equals the demo action."""

    def scale_action(self, action):
        return action

    def unscale_action(self, action):
        return action


class _FakeBaseAlgorithm:
    """Stand-in for OffPolicyAlgorithm, tracking whether it was reached."""

    base_sample_action_calls = 0

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        _FakeBaseAlgorithm.base_sample_action_calls += 1
        return "base-action", "base-buffer-action"


class _DemoSeededFake(DemoSeededOffPolicyMixin, _FakeBaseAlgorithm):
    pass


def _make_fake_model(*, num_timesteps, demonstration_policy, demonstration_steps):
    model = _DemoSeededFake()
    model.num_timesteps = num_timesteps
    model.demonstration_policy = demonstration_policy
    model.demonstration_steps = demonstration_steps
    model.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    model.policy = _FakePolicy()
    model._last_obs = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    return model


class DemoSeededOffPolicyMixinTests(unittest.TestCase):
    def setUp(self):
        _FakeBaseAlgorithm.base_sample_action_calls = 0

    def test_defers_to_the_demonstration_policy_before_demonstration_steps(self):
        calls = []

        def demo_policy(obs):
            calls.append(np.array(obs, copy=True))
            return np.array([0.75], dtype=np.float32)

        model = _make_fake_model(
            num_timesteps=5, demonstration_policy=demo_policy, demonstration_steps=20000
        )
        action, buffer_action = model._sample_action(learning_starts=100)

        self.assertEqual(len(calls), 1)
        np.testing.assert_allclose(calls[0], [0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(action, [[0.75]])
        np.testing.assert_allclose(buffer_action, [[0.75]])
        self.assertEqual(_FakeBaseAlgorithm.base_sample_action_calls, 0)

    def test_clips_the_demonstration_action_to_the_action_space(self):
        model = _make_fake_model(
            num_timesteps=0,
            demonstration_policy=lambda obs: np.array([5.0], dtype=np.float32),
            demonstration_steps=100,
        )
        action, _ = model._sample_action(learning_starts=100)
        np.testing.assert_allclose(action, [[1.0]])  # clipped to action_space.high

    def test_falls_back_once_demonstration_steps_is_reached(self):
        model = _make_fake_model(
            num_timesteps=20000,
            demonstration_policy=lambda obs: np.array([0.9], dtype=np.float32),
            demonstration_steps=20000,
        )
        result = model._sample_action(learning_starts=100)
        self.assertEqual(result, ("base-action", "base-buffer-action"))
        self.assertEqual(_FakeBaseAlgorithm.base_sample_action_calls, 1)

    def test_falls_back_when_no_demonstration_policy_is_set(self):
        model = _make_fake_model(
            num_timesteps=0, demonstration_policy=None, demonstration_steps=100
        )
        result = model._sample_action(learning_starts=100)
        self.assertEqual(result, ("base-action", "base-buffer-action"))
        self.assertEqual(_FakeBaseAlgorithm.base_sample_action_calls, 1)

    def test_demo_seeded_class_is_cached_per_base_class(self):
        from stable_baselines3 import SAC, TD3

        sac_cls = demo_seeded_class(SAC)
        self.assertIs(demo_seeded_class(SAC), sac_cls)
        self.assertTrue(issubclass(sac_cls, DemoSeededOffPolicyMixin))
        self.assertTrue(issubclass(sac_cls, SAC))
        self.assertIsNot(demo_seeded_class(TD3), sac_cls)


if __name__ == "__main__":
    unittest.main()
