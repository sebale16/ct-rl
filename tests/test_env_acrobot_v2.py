import unittest

import numpy as np

try:
    from dm_control.utils import rewards as dmc_rewards

    from environment import DMCContinuousEnv
    from environment.acrobot_v2 import BalanceV2

    HAVE_DMC = True
except ImportError:
    HAVE_DMC = False


@unittest.skipUnless(HAVE_DMC, "dm_control / Acrobot-v2 not available")
class TestAcrobotSwingupV2(unittest.TestCase):
    def _make_env(self, *, seed=0, raw_state_obs=True, **kwargs):
        defaults = dict(
            domain_name="acrobot",
            task_name="swingup-v2",
            seed=seed,
            raw_state_obs=raw_state_obs,
            time_sampling="uniform",
            dt=0.01,
            physics_dt=0.002,
            episode_duration=0.1,
        )
        defaults.update(kwargs)
        env = DMCContinuousEnv(**defaults)
        self.addCleanup(env.close)
        return env

    @staticmethod
    def _physics_state(env):
        data = env._env.physics.data
        return np.concatenate([data.qpos.copy(), data.qvel.copy()])

    @staticmethod
    def _set_physics_state(env, qpos, qvel=(0.0, 0.0)):
        physics = env._env.physics
        with physics.reset_context():
            physics.data.qpos[:] = np.asarray(qpos, dtype=np.float64)
            physics.data.qvel[:] = np.asarray(qvel, dtype=np.float64)
            physics.data.ctrl[:] = 0.0

    def test_public_alias_builds_expected_mechanism(self):
        env = self._make_env()

        self.assertEqual(env.domain_name, "acrobot")
        self.assertEqual(env.task_name, "swingup-v2")
        self.assertIsInstance(env._env.task, BalanceV2)
        self.assertEqual(
            (env._env.physics.model.nq, env._env.physics.model.nv,
             env._env.physics.model.nu),
            (2, 2, 1),
        )
        self.assertEqual(env.action_space.shape, (1,))
        np.testing.assert_allclose(env.action_space.low, [-1.0])
        np.testing.assert_allclose(env.action_space.high, [1.0])

    def test_constructor_seed_reproduces_reset_and_irregular_time_grid(self):
        common = dict(
            time_sampling="irregular",
            dt=0.01,
            min_dt=0.002,
            max_dt=0.03,
            max_steps=200,
            episode_duration=1.0,
            time_sampling_kwargs={"tail_p": 0.99, "tail_split": 0.9},
        )
        first = self._make_env(seed=17, **common)
        second = self._make_env(seed=17, **common)

        obs_first, _ = first.reset()
        obs_second, _ = second.reset()

        np.testing.assert_array_equal(obs_first, obs_second)
        np.testing.assert_array_equal(first.time_points, second.time_points)
        np.testing.assert_array_equal(
            self._physics_state(first), self._physics_state(second)
        )

    def test_explicit_reset_seed_is_repeatable_and_independent_of_time_schedule(self):
        irregular = self._make_env(
            seed=0,
            time_sampling="irregular",
            dt=0.01,
            min_dt=0.002,
            max_dt=0.03,
            max_steps=200,
            episode_duration=1.0,
        )
        obs_a, _ = irregular.reset(seed=91)
        state_a = self._physics_state(irregular)
        times_a = irregular.time_points.copy()

        irregular.step(np.zeros(1, dtype=np.float32))
        obs_b, _ = irregular.reset(seed=91)

        np.testing.assert_array_equal(obs_a, obs_b)
        np.testing.assert_array_equal(state_a, self._physics_state(irregular))
        np.testing.assert_array_equal(times_a, irregular.time_points)

        uniform = self._make_env(seed=999)
        uniform.reset(seed=91)
        np.testing.assert_array_equal(state_a, self._physics_state(uniform))

        irregular.reset(seed=92)
        self.assertFalse(
            np.array_equal(state_a, self._physics_state(irregular)),
            "different explicit reset seeds should change the reset noise",
        )

    def test_reset_stays_within_configured_down_pose_bounds(self):
        angle_noise = 0.03
        velocity_noise = 0.007
        env = self._make_env(
            task_kwargs={
                "angle_noise": angle_noise,
                "velocity_noise": velocity_noise,
            }
        )

        for seed in range(20):
            env.reset(seed=seed)
            qpos = np.asarray(env._env.physics.data.qpos)
            qvel = np.asarray(env._env.physics.data.qvel)
            error = qpos - np.asarray([np.pi, 0.0])
            self.assertTrue(np.all(np.abs(error) <= angle_noise))
            self.assertTrue(np.all(np.abs(qvel) <= velocity_noise))

    def test_native_and_raw_observations_keep_existing_contracts(self):
        raw_env = self._make_env(seed=23, raw_state_obs=True)
        native_env = self._make_env(seed=23, raw_state_obs=False)

        raw_obs, _ = raw_env.reset(seed=23)
        native_obs, _ = native_env.reset(seed=23)

        self.assertEqual(raw_env.observation_space.shape, (4,))
        self.assertEqual(native_env.observation_space.shape, (6,))
        self.assertEqual(raw_obs.dtype, np.float32)
        self.assertEqual(native_obs.dtype, np.float32)
        np.testing.assert_array_equal(
            self._physics_state(raw_env), self._physics_state(native_env)
        )
        np.testing.assert_allclose(
            raw_obs,
            self._physics_state(raw_env).astype(np.float32),
            rtol=0,
            atol=0,
        )

        physics = native_env._env.physics
        expected_native = np.concatenate(
            [physics.orientations(), physics.velocity()]
        ).astype(np.float32)
        np.testing.assert_allclose(native_obs, expected_native, rtol=0, atol=0)

    def test_reward_landmarks_formula_and_bounds(self):
        precision_weight = 0.2
        env = self._make_env(
            task_kwargs={
                "angle_noise": 0.0,
                "velocity_noise": 0.0,
                "precision_weight": precision_weight,
            }
        )
        env.reset(seed=0)
        task = env._env.task
        physics = env._env.physics

        landmarks = (
            ((0.0, 0.0), 0.0, 1.0),
            ((np.pi, 0.0), 4.0, 0.0),
            ((np.pi / 2.0, 0.0), np.sqrt(8.0), 0.2343145997339269),
        )
        for qpos, expected_distance, expected_reward in landmarks:
            with self.subTest(qpos=qpos):
                self._set_physics_state(env, qpos)
                terms = task.reward_terms(physics)
                radius = float(physics.named.model.site_size["target", 0])
                precise = float(
                    dmc_rewards.tolerance(
                        terms["tip_distance"], bounds=(0.0, radius), margin=1.0
                    )
                )
                progress = float(
                    np.clip(1.0 - terms["tip_distance"] / 4.0, 0.0, 1.0)
                )
                expected_formula = (
                    (1.0 - precision_weight) * progress
                    + precision_weight * precise
                )

                self.assertAlmostEqual(
                    terms["tip_distance"], expected_distance, places=12
                )
                self.assertAlmostEqual(terms["progress"], progress, places=12)
                self.assertAlmostEqual(terms["precision"], precise, places=12)
                self.assertAlmostEqual(terms["reward"], expected_formula, places=12)
                self.assertAlmostEqual(
                    terms["reward"], expected_reward, delta=1e-12
                )
                self.assertEqual(task.get_reward(physics), terms["reward"])

        rng = np.random.default_rng(4)
        for qpos in rng.uniform(-4.0 * np.pi, 4.0 * np.pi, size=(100, 2)):
            self._set_physics_state(env, qpos)
            terms = task.reward_terms(physics)
            self.assertTrue(np.isfinite(list(terms.values())).all())
            self.assertGreaterEqual(terms["reward"], 0.0)
            self.assertLessEqual(terms["reward"], 1.0)
            self.assertGreaterEqual(terms["progress"], 0.0)
            self.assertLessEqual(terms["progress"], 1.0)

    def test_reset_and_step_expose_episode_diagnostics(self):
        env = self._make_env(
            task_kwargs={"angle_noise": 0.0, "velocity_noise": 0.0}
        )
        _, reset_info = env.reset(seed=0)

        expected_keys = {
            "acrobot_tip_distance",
            "acrobot_tip_height",
            "acrobot_progress",
            "acrobot_precision",
            "acrobot_success",
            "acrobot_max_tip_height",
            "acrobot_success_fraction",
        }
        self.assertTrue(expected_keys.issubset(reset_info))
        self.assertAlmostEqual(reset_info["acrobot_tip_distance"], 4.0)
        self.assertAlmostEqual(reset_info["acrobot_tip_height"], 0.0)
        self.assertEqual(reset_info["acrobot_success"], 0.0)
        self.assertEqual(reset_info["acrobot_success_fraction"], 0.0)

        self._set_physics_state(env, (0.0, 0.0))
        _, reward_top, _, _, info_top = env.step(
            np.zeros(1, dtype=np.float32)
        )
        self.assertAlmostEqual(reward_top, 1.0, places=12)
        self.assertEqual(info_top["acrobot_success"], 1.0)
        self.assertEqual(info_top["acrobot_success_fraction"], 1.0)
        self.assertAlmostEqual(info_top["acrobot_max_tip_height"], 4.0)

        self._set_physics_state(env, (np.pi, 0.0))
        _, reward_down, _, _, info_down = env.step(
            np.zeros(1, dtype=np.float32)
        )
        self.assertLess(reward_down, 1e-12)
        self.assertEqual(info_down["acrobot_success"], 0.0)
        self.assertEqual(info_down["acrobot_success_fraction"], 0.5)
        self.assertAlmostEqual(info_down["acrobot_max_tip_height"], 4.0)

        _, reset_again = env.reset(seed=0)
        self.assertEqual(reset_again["acrobot_success_fraction"], 0.0)
        self.assertAlmostEqual(reset_again["acrobot_max_tip_height"], 0.0)

    def test_invalid_task_parameters_are_rejected(self):
        invalid = (
            {"angle_noise": -0.01},
            {"velocity_noise": np.inf},
            {"precision_weight": -0.1},
            {"precision_weight": 1.1},
        )
        for task_kwargs in invalid:
            with self.subTest(task_kwargs=task_kwargs), self.assertRaises(ValueError):
                DMCContinuousEnv(
                    domain_name="acrobot",
                    task_name="swingup-v2",
                    seed=0,
                    task_kwargs=task_kwargs,
                    time_sampling="uniform",
                    dt=0.01,
                    episode_duration=0.1,
                )

    def test_raw_oracle_loop_and_rollout_backends_agree(self):
        env = self._make_env(seed=5, drift_rollout_threads=2)
        env.reset(seed=5)
        if not env._drift_rollout_supported():
            self.skipTest("mujoco.rollout is unavailable")

        rng = np.random.default_rng(5)
        states = np.empty((64, 4), dtype=np.float64)
        states[:, :2] = rng.uniform(-np.pi, np.pi, size=(64, 2))
        states[:, 2:] = rng.uniform(-4.0, 4.0, size=(64, 2))
        actions = rng.uniform(-1.0, 1.0, size=(64, 1))

        env.drift_backend = "loop"
        expected = env.dynamics_terms(states, actions)
        env.drift_backend = "rollout"
        actual = env.dynamics_terms(states, actions)

        self.assertEqual(actual.shape, states.shape)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
