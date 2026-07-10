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


class TestRawStateObs(unittest.TestCase):
    """raw_state_obs=True: obs = [qpos; qvel] straight from the physics, for
    the structured dynamics model and the oracle drift on hinge/slide domains."""

    @unittest.skipUnless(HAVE_DMC, "dm_control / DMCContinuousEnv not available")
    def test_obs_matches_physics_state(self):
        for domain, task in (("cartpole", "swingup"), ("acrobot", "swingup")):
            env = DMCContinuousEnv(
                domain_name=domain, task_name=task, time_sampling="uniform",
                dt=0.01, episode_duration=0.2, seed=0, raw_state_obs=True,
            )
            nq = int(env._env.physics.model.nq)
            nv = int(env._env.physics.model.nv)
            self.assertEqual(env.observation_space.shape, (nq + nv,))
            obs, _ = env.reset(seed=0)
            data = env._env.physics.data
            np.testing.assert_allclose(
                obs, np.concatenate([data.qpos, data.qvel]).astype(np.float32),
                rtol=0, atol=1e-6, err_msg=domain,
            )
            next_obs, *_ = env.step(env.action_space.sample())
            np.testing.assert_allclose(
                next_obs,
                np.concatenate([data.qpos, data.qvel]).astype(np.float32),
                rtol=0, atol=1e-6, err_msg=domain,
            )

    @unittest.skipUnless(HAVE_DMC, "dm_control / DMCContinuousEnv not available")
    def test_string_flag_from_csv_is_coerced(self):
        env = DMCContinuousEnv(
            domain_name="cartpole", task_name="swingup", time_sampling="uniform",
            dt=0.01, episode_duration=0.2, raw_state_obs="True",
        )
        self.assertIs(env.raw_state_obs, True)

    @unittest.skipUnless(HAVE_DMC, "dm_control / DMCContinuousEnv not available")
    def test_rejects_quaternion_domains(self):
        # humanoid has a free root joint: nq = nv + 1, so d(qpos)/dt != qvel
        with self.assertRaises(ValueError):
            DMCContinuousEnv(
                domain_name="humanoid", task_name="stand",
                time_sampling="uniform", dt=0.025, episode_duration=0.2,
                raw_state_obs=True,
            )

    @unittest.skipUnless(HAVE_DMC, "dm_control / DMCContinuousEnv not available")
    def test_oracle_drift_matches_finite_difference(self):
        # raw obs makes dynamics_terms exact on any hinge/slide domain: at a
        # fine dt the analytic drift must match the realized increment.
        env = DMCContinuousEnv(
            domain_name="cartpole", task_name="swingup", time_sampling="uniform",
            dt=0.001, physics_dt=0.001, episode_duration=5.0, seed=0,
            raw_state_obs=True,
        )
        env.action_space.seed(0)
        O, A, NO, DT = [], [], [], []
        obs, _ = env.reset(seed=0)
        for _ in range(200):
            a = env.action_space.sample()
            o, t, _, r, no, nt, term, trunc, _ = env.step_dt(a)
            O.append(o); A.append(a); NO.append(no); DT.append(nt - t)
            obs = no if not (term or trunc) else env.reset()[0]
        O, A, NO, DT = map(lambda x: np.asarray(x, np.float32), (O, A, NO, DT))
        b = env.dynamics_terms(O, A)
        fd = (NO - O) / DT.reshape(-1, 1)
        corr = np.corrcoef(b.ravel(), fd.ravel())[0, 1]
        self.assertGreater(corr, 0.99)


if __name__ == "__main__":
    unittest.main()
