# tests/test_env_monitor.py
import unittest
import numpy as np
import gymnasium as gym

from environment.base import ContinuousEnv
from environment.monitor import Monitor


class DummyEnv(ContinuousEnv):
    def __init__(self, episode_length=5):
        super().__init__(max_steps=episode_length)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self._state = 0

    def _reset_physics(self, *, seed=None, options=None):
        self._state = 0
        return np.array([self._state], dtype=np.float32), {}

    def _step_physics(self, action, dt):
        self._state += 1
        obs = np.array([self._state], dtype=np.float32)
        reward = 1.0
        terminated = False
        truncated = self._state >= self.max_steps
        return obs, reward, terminated, truncated, {}, dt


class TestEnvMonitor(unittest.TestCase):
    def test_monitor_tracks_episode_stats(self):
        episode_len = 5
        env = DummyEnv(episode_length=episode_len)
        monitored_env = Monitor(env)

        obs, info = monitored_env.reset()
        self.assertEqual(obs.shape, (1,))
        self.assertNotIn("episode", info)

        for i in range(episode_len):
            action = monitored_env.action_space.sample()
            obs_t, t, _, reward, next_obs, next_t, terminated, truncated, info = (
                monitored_env.step_dt(action)
            )

            # Check shapes and types of returned values
            self.assertEqual(obs_t.shape, (1,))
            self.assertIsInstance(t, float)
            self.assertIsInstance(reward, float)
            self.assertEqual(next_obs.shape, (1,))
            self.assertIsInstance(next_t, float)
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)

            done = terminated or truncated
            if not done:
                self.assertNotIn("episode", info)

        # The last step should have episode stats
        self.assertTrue(done)
        self.assertIn("episode", info)
        ep_info = info["episode"]
        self.assertEqual(ep_info["l"], episode_len)
        self.assertAlmostEqual(ep_info["r"], float(episode_len))
