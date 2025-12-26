# tests/test_env_dmc.py
import unittest
import numpy as np

# Try to import the wrapper; if it fails (e.g., dm_control not installed),
# we mark tests to be skipped.
try:
    from environment import DMCContinuousEnv  # noqa: F401

    HAVE_DMC = True
except Exception:
    HAVE_DMC = False


def _make_small_dmc_env():
    # Use the small dm_control task `cartpole` for testing
    return DMCContinuousEnv(
        domain_name="cartpole",
        task_name="swingup",
        time_sampling="uniform",
        dt=0.04,
        episode_duration=0.2,
    )


class TestDMCContinuousEnv(unittest.TestCase):
    @unittest.skipUnless(HAVE_DMC, "dm_control / DMCContinuousEnv not available")
    def test_reset_and_spaces(self):
        env = _make_small_dmc_env()
        obs, info = env.reset(seed=0)  # noqa: F841

        # obs matches observation_space
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertEqual(obs.dtype, np.float32)

        # Action sample shape
        a = env.action_space.sample()
        self.assertEqual(a.shape, env.action_space.shape)

        # Time reset & grid built
        self.assertTrue(np.isclose(env.cur_t, 0.0))
        times = env.time_points
        self.assertIsNotNone(times)
        num_steps = int(round(env.episode_duration / env.dt))
        self.assertEqual(times.shape, (num_steps + 1,))

        # Physics / control dt positive
        self.assertGreater(env.physics_dt, 0.0)
        self.assertGreater(env.control_dt, 0.0)

    @unittest.skipUnless(HAVE_DMC, "dm_control / DMCContinuousEnv not available")
    def test_step_dt_uses_time_grid_and_reports_dt(self):
        env = _make_small_dmc_env()
        env.reset(seed=1)
        times = env.time_points
        self.assertIsNotNone(times)

        for k in range(3):
            (
                obs,
                t,
                action,
                reward,
                next_obs,
                next_t,
                terminated,
                truncated,
                info,
            ) = env.step_dt(env.action_space.sample())

            # t ~ times[k]
            self.assertTrue(np.isclose(t, times[k], atol=1e-6))

            dt_expected = times[k + 1] - times[k]
            dt_req = info.get("dt_requested", None)
            dt_used = info.get("dt_used", None)

            self.assertIsNotNone(dt_req)
            self.assertIsNotNone(dt_used)

            self.assertTrue(np.isclose(dt_req, dt_expected, atol=1e-6))
            self.assertGreater(dt_used, 0.0)
            self.assertLessEqual(abs(dt_used - dt_req), env.physics_dt + 1e-8)

            # cur_t must match the returned next_t
            self.assertTrue(np.isclose(env.cur_t, next_t, atol=1e-9))

            self.assertFalse(terminated)
            self.assertFalse(truncated)

            self.assertEqual(obs.shape, env.observation_space.shape)
            self.assertEqual(next_obs.shape, env.observation_space.shape)

    @unittest.skipUnless(HAVE_DMC, "dm_control / DMCContinuousEnv not available")
    def test_step_wrapper_works_like_gym(self):
        env = _make_small_dmc_env()
        env.reset(seed=2)

        prev_t = env.cur_t
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        self.assertGreater(env.cur_t, prev_t)
        self.assertIn("dt_requested", info)
        self.assertIn("dt_used", info)
        self.assertGreater(info["dt_used"], 0.0)
        self.assertEqual(next_obs.shape, env.observation_space.shape)


if __name__ == "__main__":
    unittest.main()
