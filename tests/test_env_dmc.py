# tests/test_env_dmc.py
import copy
import multiprocessing
import os
import pickle
import unittest
from unittest import mock

import numpy as np

# Try to import the wrapper; if it fails (e.g., dm_control not installed),
# we mark tests to be skipped.
try:
    from environment import DMCContinuousEnv  # noqa: F401

    HAVE_DMC = True
except ImportError:
    HAVE_DMC = False


def _fork_drift_worker(env, obs, action, connection):
    """Exercise a lazily rebuilt rollout pool in a forked child process."""
    try:
        connection.send((os.getpid(), env.dynamics_terms(obs, action)))
    finally:
        env.close()
        connection.close()


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

    @unittest.skipUnless(HAVE_DMC, "dm_control / DMCContinuousEnv not available")
    def test_drift_backends_agree(self):
        # The threaded mujoco.rollout backend must reproduce the historical
        # per-sample forward() loop up to float32 rounding, on both supported
        # observation maps.
        for domain, task, kwargs in (
            ("cartpole", "swingup", dict(raw_state_obs=True, physics_dt=0.005)),
            ("cheetah", "run", {}),
        ):
            env = DMCContinuousEnv(
                domain_name=domain, task_name=task, time_sampling="uniform",
                dt=0.01, episode_duration=1.0, seed=0,
                drift_rollout_threads=4, **kwargs,
            )
            env.reset(seed=0)
            rng = np.random.default_rng(0)
            O = rng.normal(size=(64, env.observation_space.shape[0])) * 0.3
            A = rng.uniform(-1.0, 1.0, size=(64, env.action_space.shape[0]))
            env.drift_backend = "loop"
            b_loop = env.dynamics_terms(O, A)
            env.drift_backend = "rollout"
            b_roll = env.dynamics_terms(O, A)
            np.testing.assert_allclose(
                b_roll, b_loop, rtol=1e-4, atol=1e-3, err_msg=domain
            )
            env.close()

    @unittest.skipUnless(HAVE_DMC, "dm_control / DMCContinuousEnv not available")
    def test_drift_backend_validation(self):
        with self.assertRaises(ValueError):
            DMCContinuousEnv(
                domain_name="cartpole", task_name="swingup",
                time_sampling="uniform", dt=0.01, episode_duration=0.2,
                drift_backend="gpu",
            )
        with self.assertRaises(ValueError):
            DMCContinuousEnv(
                domain_name="cartpole", task_name="swingup",
                time_sampling="uniform", dt=0.01, episode_duration=0.2,
                drift_rollout_threads=0,
            )

    @unittest.skipUnless(HAVE_DMC, "dm_control / DMCContinuousEnv not available")
    def test_rollout_model_refreshes_after_randomizing_reset(self):
        env = DMCContinuousEnv(
            domain_name="point_mass", task_name="hard",
            time_sampling="uniform", dt=0.02, episode_duration=0.2,
            seed=0, raw_state_obs=True, drift_rollout_threads=2,
        )
        self.addCleanup(env.close)
        obs, _ = env.reset()
        actions = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [-1.0, 1.0]],
            dtype=np.float64,
        )
        states = np.repeat(obs[None, :], actions.shape[0], axis=0)
        env.drift_backend = "rollout"
        env.dynamics_terms(states, actions)
        old_wrap_prm = env._env.physics.model.wrap_prm.copy()
        self.assertIsNotNone(env._drift_rollout)

        obs, _ = env.reset()
        self.assertIsNone(env._drift_rollout)
        self.assertFalse(
            np.array_equal(old_wrap_prm, env._env.physics.model.wrap_prm)
        )
        states = np.repeat(obs[None, :], actions.shape[0], axis=0)
        env.drift_backend = "loop"
        expected = env.dynamics_terms(states, actions)
        env.drift_backend = "rollout"
        actual = env.dynamics_terms(states, actions)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    @unittest.skipUnless(HAVE_DMC, "dm_control / DMCContinuousEnv not available")
    def test_rollout_does_not_autoreset_unstable_rows(self):
        env = DMCContinuousEnv(
            domain_name="cheetah", task_name="run",
            time_sampling="uniform", dt=0.01, physics_dt=0.002,
            episode_duration=0.2, seed=0, drift_rollout_threads=2,
        )
        self.addCleanup(env.close)
        env.reset()
        states = np.zeros((2, env.observation_space.shape[0]), dtype=np.float64)
        states[:, 8:] = 1e5
        actions = np.zeros((2, env.action_space.shape[0]), dtype=np.float64)
        env.drift_backend = "loop"
        expected = env.dynamics_terms(states, actions)
        self.assertGreater(np.abs(expected[:, 8:]).max(), 1e10)
        env.drift_backend = "rollout"
        actual = env.dynamics_terms(states, actions)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-2)

    @unittest.skipUnless(HAVE_DMC, "dm_control / DMCContinuousEnv not available")
    def test_rollout_preserves_live_applied_force_context(self):
        env = DMCContinuousEnv(
            domain_name="cartpole", task_name="swingup",
            time_sampling="uniform", dt=0.01, episode_duration=0.2,
            seed=0, raw_state_obs=True, drift_rollout_threads=2,
        )
        self.addCleanup(env.close)
        obs, _ = env.reset()
        states = np.repeat(obs[None, :], 4, axis=0)
        actions = np.zeros((4, env.action_space.shape[0]), dtype=np.float64)

        # Instantiate the private model before changing data-side inputs. The
        # inputs must be snapshotted anew on every rollout call.
        env.drift_backend = "rollout"
        baseline = env.dynamics_terms(states, actions)
        data = env._env.physics.data
        data.qfrc_applied[:] = [3.0, -2.0]
        data.xfrc_applied[-1, :] = [2.0, -1.0, 0.0, 0.0, 0.0, 0.5]

        env.drift_backend = "loop"
        expected = env.dynamics_terms(states, actions)
        self.assertGreater(np.abs(expected - baseline).max(), 1.0)
        env.drift_backend = "rollout"
        actual = env.dynamics_terms(states, actions)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-5)

    @unittest.skipUnless(HAVE_DMC, "dm_control / DMCContinuousEnv not available")
    def test_rollout_pool_is_not_serialized(self):
        env = DMCContinuousEnv(
            domain_name="cartpole", task_name="swingup",
            time_sampling="uniform", dt=0.01, episode_duration=0.2,
            seed=0, raw_state_obs=True, drift_rollout_threads=2,
        )
        self.addCleanup(env.close)
        obs, _ = env.reset()
        states = np.repeat(obs[None, :], 3, axis=0)
        actions = np.zeros((3, env.action_space.shape[0]), dtype=np.float64)
        expected = env.dynamics_terms(states, actions)
        source_pool = env._drift_rollout[0]

        clones = [copy.deepcopy(env), pickle.loads(pickle.dumps(env))]
        for clone in clones:
            self.addCleanup(clone.close)
            self.assertIsNone(clone._drift_rollout)
            np.testing.assert_allclose(
                clone.dynamics_terms(states, actions), expected, rtol=0, atol=0
            )
            self.assertIsNotNone(clone._drift_rollout)

        self.assertIs(env._drift_rollout[0], source_pool)
        np.testing.assert_allclose(
            env.dynamics_terms(states, actions), expected, rtol=0, atol=0
        )

    @unittest.skipUnless(
        HAVE_DMC and "fork" in multiprocessing.get_all_start_methods(),
        "requires dm_control and the multiprocessing fork start method",
    )
    def test_rollout_pool_rebuilds_across_fork(self):
        env = DMCContinuousEnv(
            domain_name="cartpole", task_name="swingup",
            time_sampling="uniform", dt=0.01, episode_duration=0.2,
            seed=0, raw_state_obs=True, drift_rollout_threads=2,
        )
        self.addCleanup(env.close)
        obs, _ = env.reset()
        states = np.repeat(obs[None, :], 3, axis=0)
        actions = np.zeros((3, env.action_space.shape[0]), dtype=np.float64)
        expected = env.dynamics_terms(states, actions)
        self.assertIsNotNone(env._drift_rollout)

        context = multiprocessing.get_context("fork")
        receive, send = context.Pipe(duplex=False)
        process = context.Process(
            target=_fork_drift_worker,
            args=(env, states, actions, send),
        )
        try:
            process.start()
            send.close()
            process.join(5.0)
            if process.is_alive():
                process.terminate()
                process.join()
                self.fail("forked rollout child did not exit")
            self.assertEqual(process.exitcode, 0)
            self.assertTrue(receive.poll())
            child_pid, actual = receive.recv()
            self.assertNotEqual(child_pid, os.getpid())
            np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        finally:
            receive.close()
            if process.is_alive():
                process.terminate()
                process.join()

        # The pre-fork hook also cleared the parent's pool; it remains usable
        # and lazily rebuilds a process-local pool here.
        self.assertIsNone(env._drift_rollout)
        np.testing.assert_allclose(
            env.dynamics_terms(states, actions), expected, rtol=0, atol=0
        )

    @unittest.skipUnless(HAVE_DMC, "dm_control / DMCContinuousEnv not available")
    def test_default_rollout_threads_respect_cpu_allocation(self):
        import environment.dmc as dmc_module

        with mock.patch.object(
            dmc_module.os, "sched_getaffinity", return_value=set(range(32))
        ), mock.patch.dict(
            dmc_module.os.environ, {"SLURM_CPUS_PER_TASK": "4"}
        ):
            self.assertEqual(dmc_module._default_drift_rollout_threads(), 4)

        with mock.patch.object(
            dmc_module.os, "sched_getaffinity", return_value=set(range(32))
        ), mock.patch.dict(
            dmc_module.os.environ, {"SLURM_CPUS_PER_TASK": ""}
        ):
            self.assertEqual(dmc_module._default_drift_rollout_threads(), 8)

        env = DMCContinuousEnv(
            domain_name="cartpole", task_name="swingup",
            time_sampling="uniform", dt=0.01, episode_duration=0.2,
            drift_rollout_threads=3,
        )
        self.addCleanup(env.close)
        self.assertEqual(env.drift_rollout_threads, 3)


if __name__ == "__main__":
    unittest.main()
