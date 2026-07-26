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


class DiagnosticEnv(DummyEnv):
    """Emits a per-step scalar that rises by 1 each step, plus a constant."""

    def _step_physics(self, action, dt):
        obs, reward, terminated, truncated, _, dt = super()._step_physics(action, dt)
        info = {"diag": float(self._state), "other": 7.0}
        if self._state == 2:
            info["sometimes"] = 5.0
            info["bad"] = float("nan")
        return obs, reward, terminated, truncated, info, dt


class TestMonitorInfoKeywords(unittest.TestCase):
    """``info_keywords`` logs behavior-policy diagnostics as running means."""

    def _run(self, keywords, steps=4):
        from common.logger import get_logger

        env = Monitor(DiagnosticEnv(episode_length=10), info_keywords=keywords)
        env.reset()
        for _ in range(steps):
            env.step_dt(env.action_space.sample())
        return get_logger().name_to_value

    def test_named_keys_are_logged_as_running_means(self):
        values = self._run(("diag",), steps=4)
        # diag takes 1, 2, 3, 4 -> mean 2.5, namespaced under rollout/.
        self.assertAlmostEqual(values["rollout/diag"], 2.5)

    def test_unnamed_info_keys_are_not_logged(self):
        values = self._run(("diag",), steps=4)
        self.assertNotIn("rollout/other", values)

    def test_default_is_no_diagnostic_logging(self):
        from common.logger import get_logger

        before = dict(get_logger().name_to_value)
        env = Monitor(DiagnosticEnv(episode_length=10))
        env.reset()
        for _ in range(3):
            env.step_dt(env.action_space.sample())
        after = get_logger().name_to_value
        self.assertEqual(
            [k for k in after if k.startswith("rollout/") and k not in before], []
        )

    def test_absent_and_non_finite_values_are_skipped(self):
        values = self._run(("sometimes", "bad", "missing"), steps=4)
        # Present on exactly one step, so the mean is that step's value.
        self.assertAlmostEqual(values["rollout/sometimes"], 5.0)
        # NaN must never enter a running mean, and absent keys create none.
        self.assertNotIn("rollout/bad", values)
        self.assertNotIn("rollout/missing", values)


class TestRolloutInfoKeys(unittest.TestCase):
    """The runner attaches the v6 exploration diagnostics and nothing else."""

    def test_v6_arms_share_the_diagnostic_set(self):
        from benchmarks.run_ct_rl import rollout_info_keys

        keys = rollout_info_keys("acrobot-swingup-v6")
        self.assertEqual(keys, rollout_info_keys("acrobot-swingup-v6-uniform"))
        self.assertIn("acrobot_energy_norm", keys)
        self.assertIn("acrobot_coordination_loss", keys)

    def test_other_tasks_get_no_diagnostics(self):
        from benchmarks.run_ct_rl import rollout_info_keys

        for env_id in ("cheetah-run", "acrobot-swingup-v4.2", "cartpole-swingup"):
            self.assertEqual(rollout_info_keys(env_id), ())
