import unittest
from unittest import mock

import numpy as np

try:
    from dm_control.suite import acrobot as dmc_acrobot
    from dm_control.utils import rewards as dmc_rewards

    from environment import DMCContinuousEnv
    from environment.acrobot_v2 import (
        BalanceV2,
        BalanceV3,
        BalanceV4,
        BalanceV5,
        BalanceV6,
        STRICT_CAPTURE_DISTANCE,
        STRICT_CAPTURE_SPEED,
        V41_ENERGY_OVERSHOOT_MARGIN,
        V41_SPEED_BOUNDS,
        V41_SPEED_MARGIN,
        V6_ACTION_WEIGHT,
        V6_COST_SCALE,
        V6_STATE_WEIGHTS,
        swingup_v3,
        swingup_v4,
        swingup_v41,
        swingup_v42,
        swingup_v5,
        swingup_v6,
        swingup_v6_uniform,
    )

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


@unittest.skipUnless(HAVE_DMC, "dm_control / Acrobot-v3 not available")
class TestAcrobotSwingupV3Reward(unittest.TestCase):
    def setUp(self):
        self.physics = dmc_acrobot.Physics.from_xml_string(
            *dmc_acrobot.get_model_and_assets()
        )
        self.task = BalanceV3(
            random=0,
            angle_noise=0.0,
            velocity_noise=0.0,
            precision_weight=0.2,
        )

    def _set_physics_state(self, qpos, qvel=(0.0, 0.0)):
        with self.physics.reset_context():
            self.physics.data.qpos[:] = np.asarray(qpos, dtype=np.float64)
            self.physics.data.qvel[:] = np.asarray(qvel, dtype=np.float64)
            self.physics.data.ctrl[:] = 0.0

    def test_factory_builds_v3_with_an_exact_down_reset(self):
        env = swingup_v3(
            time_limit=0.1,
            random=19,
            environment_kwargs={"flat_observation": True},
            angle_noise=0.0,
            velocity_noise=0.0,
        )
        try:
            env.reset()
            self.assertIsInstance(env.task, BalanceV3)
            np.testing.assert_array_equal(env.physics.data.qpos, [np.pi, 0.0])
            np.testing.assert_array_equal(env.physics.data.qvel, [0.0, 0.0])
        finally:
            env.close()

    def test_continuous_wrapper_builds_v3_and_exposes_v3_diagnostics(self):
        env = DMCContinuousEnv(
            domain_name="acrobot",
            task_name="swingup-v3",
            seed=23,
            raw_state_obs=True,
            time_sampling="uniform",
            dt=0.01,
            physics_dt=0.002,
            episode_duration=0.1,
            task_kwargs={"angle_noise": 0.0, "velocity_noise": 0.0},
        )
        self.addCleanup(env.close)

        _, reset_info = env.reset(seed=23)
        self.assertIsInstance(env._env.task, BalanceV3)
        v3_keys = {
            "acrobot_upper_uprightness",
            "acrobot_lower_uprightness",
            "acrobot_extension",
            "acrobot_gym_height_success",
            "acrobot_exact_success",
        }
        self.assertTrue(v3_keys.issubset(reset_info))
        self.assertAlmostEqual(reset_info["acrobot_upper_uprightness"], 0.0)
        self.assertAlmostEqual(reset_info["acrobot_lower_uprightness"], 0.0)
        self.assertAlmostEqual(reset_info["acrobot_extension"], 1.0)
        self.assertEqual(reset_info["acrobot_gym_height_success"], 0.0)
        self.assertEqual(reset_info["acrobot_exact_success"], 0.0)

        _, _, _, _, step_info = env.step(np.zeros(1, dtype=np.float32))
        self.assertTrue(v3_keys.issubset(step_info))

    def test_v2_wrapper_info_schema_does_not_gain_v3_only_terms(self):
        env = DMCContinuousEnv(
            domain_name="acrobot",
            task_name="swingup-v2",
            seed=23,
            raw_state_obs=True,
            time_sampling="uniform",
            dt=0.01,
            physics_dt=0.002,
            episode_duration=0.1,
            task_kwargs={"angle_noise": 0.0, "velocity_noise": 0.0},
        )
        self.addCleanup(env.close)

        _, info = env.reset(seed=23)
        v3_only_keys = {
            "acrobot_upper_uprightness",
            "acrobot_lower_uprightness",
            "acrobot_extension",
            "acrobot_gym_height_success",
            "acrobot_exact_success",
        }
        self.assertTrue(v3_only_keys.isdisjoint(info))
        self.assertIn("acrobot_success", info)

    def test_reset_matches_v2_for_the_same_seed(self):
        v2_physics = dmc_acrobot.Physics.from_xml_string(
            *dmc_acrobot.get_model_and_assets()
        )
        v3_physics = dmc_acrobot.Physics.from_xml_string(
            *dmc_acrobot.get_model_and_assets()
        )
        kwargs = {
            "random": 37,
            "angle_noise": 0.03,
            "velocity_noise": 0.007,
            "precision_weight": 0.2,
        }
        BalanceV2(**kwargs).initialize_episode(v2_physics)
        BalanceV3(**kwargs).initialize_episode(v3_physics)

        np.testing.assert_array_equal(v3_physics.data.qpos, v2_physics.data.qpos)
        np.testing.assert_array_equal(v3_physics.data.qvel, v2_physics.data.qvel)

    def test_reward_landmarks_require_upright_extended_links(self):
        precision_weight = self.task.precision_weight
        landmarks = (
            {
                "name": "down",
                "qpos": (np.pi, 0.0),
                "upright": (0.0, 0.0),
                "extension": 1.0,
                "progress": 0.0,
                "tip_height": 0.0,
                "gym_success": 0.0,
                "exact_success": 0.0,
            },
            {
                "name": "upright",
                "qpos": (0.0, 0.0),
                "upright": (1.0, 1.0),
                "extension": 1.0,
                "progress": 1.0,
                "tip_height": 4.0,
                "gym_success": 1.0,
                "exact_success": 1.0,
            },
            {
                "name": "straight-horizontal",
                "qpos": (np.pi / 2.0, 0.0),
                "upright": (0.5, 0.5),
                "extension": 1.0,
                "progress": 0.5,
                "tip_height": 2.0,
                "gym_success": 0.0,
                "exact_success": 0.0,
            },
            {
                "name": "down-folded",
                "qpos": (np.pi, np.pi),
                "upright": (0.0, 1.0),
                "extension": 0.0,
                "progress": 0.0,
                "tip_height": 2.0,
                "gym_success": 0.0,
                "exact_success": 0.0,
            },
            {
                "name": "horizontal-folded",
                "qpos": (np.pi / 2.0, np.pi),
                "upright": (0.5, 0.5),
                "extension": 0.0,
                "progress": 0.0,
                "tip_height": 2.0,
                "gym_success": 0.0,
                "exact_success": 0.0,
            },
        )

        for landmark in landmarks:
            with self.subTest(landmark=landmark["name"]):
                self._set_physics_state(landmark["qpos"])
                terms = self.task.reward_terms(self.physics)
                precise = float(
                    dmc_rewards.tolerance(
                        terms["tip_distance"], bounds=(0.0, 0.2), margin=1.0
                    )
                )
                expected_reward = (
                    (1.0 - precision_weight) * landmark["progress"]
                    + precision_weight * precise
                )

                self.assertAlmostEqual(
                    terms["upper_uprightness"], landmark["upright"][0]
                )
                self.assertAlmostEqual(
                    terms["lower_uprightness"], landmark["upright"][1]
                )
                self.assertAlmostEqual(terms["extension"], landmark["extension"])
                self.assertAlmostEqual(terms["progress"], landmark["progress"])
                self.assertAlmostEqual(terms["tip_height"], landmark["tip_height"])
                self.assertEqual(
                    terms["gym_height_success"], landmark["gym_success"]
                )
                self.assertEqual(terms["exact_success"], landmark["exact_success"])
                self.assertEqual(terms["success"], terms["exact_success"])
                self.assertAlmostEqual(terms["precision"], precise)
                self.assertAlmostEqual(terms["reward"], expected_reward)
                self.assertAlmostEqual(
                    self.task.get_reward(self.physics), expected_reward
                )

    def test_every_exact_fold_earns_only_the_precision_tail(self):
        for elbow in (-np.pi, np.pi):
            for shoulder in np.linspace(-np.pi, np.pi, 9):
                with self.subTest(shoulder=shoulder, elbow=elbow):
                    self._set_physics_state((shoulder, elbow))
                    terms = self.task.reward_terms(self.physics)
                    self.assertAlmostEqual(terms["tip_distance"], 2.0)
                    self.assertAlmostEqual(terms["tip_height"], 2.0)
                    self.assertAlmostEqual(terms["extension"], 0.0)
                    self.assertAlmostEqual(terms["progress"], 0.0)
                    self.assertAlmostEqual(
                        terms["reward"],
                        self.task.precision_weight * terms["precision"],
                    )

    def test_progress_is_extension_times_mean_link_uprightness(self):
        rng = np.random.default_rng(11)
        for qpos in rng.uniform(-4.0 * np.pi, 4.0 * np.pi, size=(100, 2)):
            with self.subTest(qpos=qpos):
                self._set_physics_state(qpos)
                terms = self.task.reward_terms(self.physics)
                expected = terms["extension"] * 0.5 * (
                    terms["upper_uprightness"] + terms["lower_uprightness"]
                )
                self.assertAlmostEqual(terms["progress"], expected)

    def test_blended_reward_landscape_has_only_the_upright_periodic_local_maximum(self):
        # This analytic periodic grid guards against replacing the smooth
        # extension-weighted mean with a bottleneck/minimum.  The latter creates
        # spurious maxima near q1=q2=+/-2*pi/3 that can trap a policy.
        angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
        shoulder, elbow = np.meshgrid(angles, angles, indexing="ij")
        upper = (1.0 + np.cos(shoulder)) / 2.0
        lower = (1.0 + np.cos(shoulder + elbow)) / 2.0
        extension = (1.0 + np.cos(elbow)) / 2.0
        progress = extension * 0.5 * (upper + lower)
        tip_x = np.sin(shoulder) + np.sin(shoulder + elbow)
        tip_z = 2.0 + np.cos(shoulder) + np.cos(shoulder + elbow)
        distance = np.hypot(tip_x, tip_z - 4.0)
        precise = dmc_rewards.tolerance(
            distance, bounds=(0.0, 0.2), margin=1.0
        )
        reward = (
            (1.0 - self.task.precision_weight) * progress
            + self.task.precision_weight * precise
        )

        neighbors = [
            np.roll(np.roll(reward, di, axis=0), dj, axis=1)
            for di in (-1, 0, 1)
            for dj in (-1, 0, 1)
            if (di, dj) != (0, 0)
        ]
        local_maximum = np.logical_and.reduce(
            [reward >= neighbor for neighbor in neighbors]
        ) & np.logical_or.reduce([reward > neighbor for neighbor in neighbors])
        maxima = np.argwhere(local_maximum)

        self.assertEqual(maxima.shape, (1, 2))
        shoulder_index, elbow_index = maxima[0]
        self.assertEqual(angles[shoulder_index], 0.0)
        self.assertEqual(angles[elbow_index], 0.0)
        self.assertEqual(reward[shoulder_index, elbow_index], 1.0)

    def test_gym_height_threshold_is_strict_and_distinct_from_exact_success(self):
        # Straight links at shoulder pi/3 put the tip exactly at z=3.  Gym's
        # Acrobot terminal predicate uses height > 1 above the z=2 pivot.
        self._set_physics_state((np.pi / 3.0, 0.0))
        threshold = self.task.reward_terms(self.physics)
        self.assertAlmostEqual(threshold["tip_height"], 3.0)
        self.assertEqual(threshold["gym_height_success"], 0.0)
        self.assertEqual(threshold["exact_success"], 0.0)

        self._set_physics_state((np.pi / 3.0 - 1e-3, 0.0))
        above = self.task.reward_terms(self.physics)
        self.assertGreater(above["tip_height"], 3.0)
        self.assertEqual(above["gym_height_success"], 1.0)
        self.assertEqual(above["exact_success"], 0.0)


@unittest.skipUnless(HAVE_DMC, "dm_control / Acrobot-v2 not available")
class TestAcrobotSwingupV4Reward(unittest.TestCase):
    def setUp(self):
        self.physics = dmc_acrobot.Physics.from_xml_string(
            *dmc_acrobot.get_model_and_assets()
        )
        self.task = BalanceV4(
            random=0,
            angle_noise=0.0,
            velocity_noise=0.0,
            hold_weight=0.8,
        )
        # Calibrates the hanging/upright energy references.
        self.task.initialize_episode(self.physics)

    def _set_physics_state(self, qpos, qvel=(0.0, 0.0)):
        with self.physics.reset_context():
            self.physics.data.qpos[:] = np.asarray(qpos, dtype=np.float64)
            self.physics.data.qvel[:] = np.asarray(qvel, dtype=np.float64)
            self.physics.data.ctrl[:] = 0.0

    def test_factory_builds_v4_with_an_exact_down_reset(self):
        env = swingup_v4(
            time_limit=0.1,
            random=19,
            environment_kwargs={"flat_observation": True},
            angle_noise=0.0,
            velocity_noise=0.0,
        )
        try:
            env.reset()
            self.assertIsInstance(env.task, BalanceV4)
            self.assertEqual(env.task.speed_bounds, (0.0, 0.5))
            self.assertEqual(env.task.speed_margin, 2.0)
            np.testing.assert_array_equal(env.physics.data.qpos, [np.pi, 0.0])
            np.testing.assert_array_equal(env.physics.data.qvel, [0.0, 0.0])
        finally:
            env.close()

    def test_reward_before_calibration_raises(self):
        physics = dmc_acrobot.Physics.from_xml_string(
            *dmc_acrobot.get_model_and_assets()
        )
        task = BalanceV4(random=0)
        with self.assertRaises(RuntimeError):
            task.reward_terms(physics)

    def test_invalid_hold_weight_rejected(self):
        for hold_weight in (-0.1, 1.5, float("nan"), float("inf")):
            with self.subTest(hold_weight=hold_weight):
                with self.assertRaises(ValueError):
                    BalanceV4(random=0, hold_weight=hold_weight)

    def test_invalid_speed_gate_configuration_rejected(self):
        bad_bounds = (
            None,
            (),
            (0.0,),
            (0.0, 0.1, 0.2),
            (-0.1, 0.1),
            (0.2, 0.1),
            (0.0, float("nan")),
            (0.0, float("inf")),
        )
        for speed_bounds in bad_bounds:
            with self.subTest(speed_bounds=speed_bounds):
                with self.assertRaisesRegex(ValueError, "speed_bounds"):
                    BalanceV4(random=0, speed_bounds=speed_bounds)

        for speed_margin in (0.0, -0.5, float("nan"), float("inf")):
            with self.subTest(speed_margin=speed_margin):
                with self.assertRaisesRegex(ValueError, "speed_margin"):
                    BalanceV4(random=0, speed_margin=speed_margin)

    def test_default_speed_gate_preserves_published_v4_definition(self):
        self.assertEqual(self.task.speed_bounds, (0.0, 0.5))
        self.assertEqual(self.task.speed_margin, 2.0)

        self._set_physics_state((0.0, 0.0), qvel=(0.6, 0.0))
        terms = self.task.reward_terms(self.physics)
        expected = float(
            dmc_rewards.tolerance(
                terms["speed"],
                bounds=(0.0, 0.5),
                margin=2.0,
                value_at_margin=0.1,
                sigmoid="gaussian",
            )
        )
        self.assertAlmostEqual(terms["slow_gate"], expected)

    def test_strict_capture_uses_open_distance_and_speed_thresholds(self):
        self.assertEqual(STRICT_CAPTURE_DISTANCE, 0.2)
        self.assertEqual(STRICT_CAPTURE_SPEED, 0.2)
        inside_distance = np.nextafter(STRICT_CAPTURE_DISTANCE, 0.0)
        inside_speed = np.nextafter(STRICT_CAPTURE_SPEED, 0.0)

        self._set_physics_state((0.0, 0.0), qvel=(inside_speed, 0.0))
        with mock.patch.object(
            dmc_acrobot.Physics, "to_target", return_value=inside_distance
        ):
            self.assertEqual(
                self.task.reward_terms(self.physics)["strict_capture"], 1.0
            )

        # Both thresholds are strict: equality at either boundary is outside.
        with mock.patch.object(
            dmc_acrobot.Physics,
            "to_target",
            return_value=STRICT_CAPTURE_DISTANCE,
        ):
            self.assertEqual(
                self.task.reward_terms(self.physics)["strict_capture"], 0.0
            )

        self._set_physics_state(
            (0.0, 0.0), qvel=(STRICT_CAPTURE_SPEED, 0.0)
        )
        with mock.patch.object(
            dmc_acrobot.Physics, "to_target", return_value=inside_distance
        ):
            self.assertEqual(
                self.task.reward_terms(self.physics)["strict_capture"], 0.0
            )

    def test_reset_matches_v2_for_the_same_seed(self):
        v2_physics = dmc_acrobot.Physics.from_xml_string(
            *dmc_acrobot.get_model_and_assets()
        )
        v4_physics = dmc_acrobot.Physics.from_xml_string(
            *dmc_acrobot.get_model_and_assets()
        )
        kwargs = {"random": 37, "angle_noise": 0.03, "velocity_noise": 0.007}
        BalanceV2(**kwargs, precision_weight=0.2).initialize_episode(v2_physics)
        BalanceV4(**kwargs, hold_weight=0.8).initialize_episode(v4_physics)

        np.testing.assert_array_equal(v4_physics.data.qpos, v2_physics.data.qpos)
        np.testing.assert_array_equal(v4_physics.data.qvel, v2_physics.data.qvel)

    def test_energy_normalization_landmarks(self):
        self._set_physics_state((np.pi, 0.0))
        self.assertAlmostEqual(
            self.task.reward_terms(self.physics)["energy_norm"], 0.0, places=9
        )
        self._set_physics_state((0.0, 0.0))
        self.assertAlmostEqual(
            self.task.reward_terms(self.physics)["energy_norm"], 1.0, places=9
        )
        # Kinetic energy counts: a fast hanging swing carries positive Ẽ.
        self._set_physics_state((np.pi, 0.0), qvel=(4.0, 0.0))
        self.assertGreater(
            self.task.reward_terms(self.physics)["energy_norm"], 0.4
        )

    def test_terms_recompose_from_published_tolerances(self):
        rng = np.random.default_rng(5)
        qpos = rng.uniform(-2.0 * np.pi, 2.0 * np.pi, size=(25, 2))
        qvel = rng.uniform(-4.0, 4.0, size=(25, 2))
        for pose, velocity in zip(qpos, qvel):
            with self.subTest(qpos=pose, qvel=velocity):
                self._set_physics_state(pose, qvel=velocity)
                terms = self.task.reward_terms(self.physics)
                energy_close = float(
                    dmc_rewards.tolerance(
                        terms["energy_norm"],
                        bounds=(1.0, 1.0),
                        margin=1.0,
                        value_at_margin=0.1,
                        sigmoid="gaussian",
                    )
                )
                mean_upright = 0.5 * (
                    terms["upper_uprightness"] + terms["lower_uprightness"]
                )
                self.assertAlmostEqual(
                    terms["progress"], energy_close * 0.5 * (1.0 + mean_upright)
                )
                slow = float(
                    dmc_rewards.tolerance(
                        terms["speed"],
                        bounds=(0.0, 0.5),
                        margin=2.0,
                        value_at_margin=0.1,
                        sigmoid="gaussian",
                    )
                )
                self.assertAlmostEqual(terms["slow_gate"], slow)
                self.assertAlmostEqual(
                    terms["hold"], terms["precision"] * terms["slow_gate"]
                )
                expected = 0.2 * terms["progress"] + 0.8 * terms["hold"]
                self.assertAlmostEqual(
                    terms["reward"], float(np.clip(expected, 0.0, 1.0))
                )
                self.assertEqual(terms["success"], terms["exact_success"])

    def test_reward_landmarks_pay_only_the_slow_upright_capture(self):
        # Hanging rest: only the value-at-margin energy floor times the tilt.
        self._set_physics_state((np.pi, 0.0))
        down = self.task.reward_terms(self.physics)
        self.assertAlmostEqual(down["progress"], 0.05, places=6)
        self.assertAlmostEqual(down["reward"], 0.01, places=3)

        # Upright rest at the target: exact maximum.
        self._set_physics_state((0.0, 0.0))
        upright = self.task.reward_terms(self.physics)
        self.assertAlmostEqual(upright["reward"], 1.0, places=6)
        self.assertAlmostEqual(upright["hold"], 1.0, places=6)

        # v2's exploits stay dead: exact folds and the bent near-top hover.
        self._set_physics_state((0.0, np.pi))
        self.assertLess(self.task.reward_terms(self.physics)["reward"], 0.2)
        self._set_physics_state((0.18, 0.55), qvel=(1.8, -2.2))
        self.assertLess(self.task.reward_terms(self.physics)["reward"], 0.3)

        # Fast spin through the very top: energy overshoot plus speed gate.
        self._set_physics_state((0.0, 0.0), qvel=(3.5, 0.0))
        self.assertLess(self.task.reward_terms(self.physics)["reward"], 0.25)

        # Slow pass near the goal earns most of the hold payoff.
        self._set_physics_state((0.08, 0.05), qvel=(0.5, 0.4))
        self.assertGreater(self.task.reward_terms(self.physics)["reward"], 0.85)

    def test_static_reward_slice_has_only_the_upright_local_maximum(self):
        # Zero-velocity slice of the reward over the periodic joint grid.
        # Guards against a secondary energy/uprightness maximum a policy
        # could park on without capturing the target.
        n = 120
        angles = np.linspace(-np.pi, np.pi, n, endpoint=False)
        reward = np.empty((n, n))
        for i, shoulder in enumerate(angles):
            for j, elbow in enumerate(angles):
                self._set_physics_state((shoulder, elbow))
                reward[i, j] = self.task.reward_terms(self.physics)["reward"]

        neighbors = [
            np.roll(np.roll(reward, di, axis=0), dj, axis=1)
            for di in (-1, 0, 1)
            for dj in (-1, 0, 1)
            if (di, dj) != (0, 0)
        ]
        local_maximum = np.logical_and.reduce(
            [reward >= neighbor for neighbor in neighbors]
        ) & np.logical_or.reduce([reward > neighbor for neighbor in neighbors])
        maxima = np.argwhere(local_maximum)

        self.assertEqual(maxima.shape, (1, 2))
        shoulder_index, elbow_index = maxima[0]
        self.assertEqual(angles[shoulder_index], 0.0)
        self.assertEqual(angles[elbow_index], 0.0)

    def test_elbow_pumping_raises_reward_where_v3_does_not(self):
        # Scripted collocated pump: kick, then elbow torque against the
        # shoulder swing, backing off as Ẽ approaches 1.  The v4 reward must
        # track the injected energy; the v3 progress term must not.
        env = swingup_v4(
            time_limit=20.0,
            random=3,
            angle_noise=0.0,
            velocity_noise=0.0,
        )
        self.addCleanup(env.close)
        env.reset()
        physics = env.physics
        v3_task = BalanceV3(random=0, angle_noise=0.0, velocity_noise=0.0)

        v4_rewards, v3_rewards, energies = [], [], []
        for step in range(1200):
            terms = env.task.reward_terms(physics)
            energy_norm = terms["energy_norm"]
            if step < 100:
                action = 1.0
            else:
                gain = min(1.0, 4.0 * max(0.0, 1.0 - energy_norm))
                action = -np.sign(float(physics.data.qvel[0])) * gain
            env.step(np.asarray([action]))
            terms = env.task.reward_terms(physics)
            v4_rewards.append(terms["reward"])
            v3_rewards.append(v3_task.reward_terms(physics)["reward"])
            energies.append(terms["energy_norm"])

        v4_rewards = np.asarray(v4_rewards)
        v3_rewards = np.asarray(v3_rewards)
        energies = np.asarray(energies)

        self.assertGreater(energies.max(), 0.4)
        corr_v4 = np.corrcoef(energies, v4_rewards)[0, 1]
        corr_v3 = np.corrcoef(energies, v3_rewards)[0, 1]
        self.assertGreater(corr_v4, 0.75)
        self.assertGreater(corr_v4, corr_v3 + 0.2)

    def test_continuous_wrapper_builds_v4_and_exposes_v4_diagnostics(self):
        env = DMCContinuousEnv(
            domain_name="acrobot",
            task_name="swingup-v4",
            seed=23,
            raw_state_obs=True,
            time_sampling="uniform",
            dt=0.01,
            physics_dt=0.002,
            episode_duration=0.1,
            task_kwargs={"angle_noise": 0.0, "velocity_noise": 0.0},
        )
        self.addCleanup(env.close)

        _, reset_info = env.reset(seed=23)
        self.assertIsInstance(env._env.task, BalanceV4)
        v4_keys = {
            "acrobot_upper_uprightness",
            "acrobot_lower_uprightness",
            "acrobot_extension",
            "acrobot_gym_height_success",
            "acrobot_exact_success",
            "acrobot_energy_norm",
            "acrobot_speed",
            "acrobot_slow_gate",
            "acrobot_hold",
            "acrobot_strict_capture",
        }
        self.assertTrue(v4_keys.issubset(reset_info))
        self.assertAlmostEqual(reset_info["acrobot_energy_norm"], 0.0, places=6)
        self.assertAlmostEqual(reset_info["acrobot_slow_gate"], 1.0, places=6)
        self.assertAlmostEqual(reset_info["acrobot_hold"], 0.0, places=6)
        self.assertEqual(reset_info["acrobot_strict_capture"], 0.0)
        self.assertAlmostEqual(reset_info["acrobot_progress"], 0.05, places=6)

        _, _, _, _, step_info = env.step(np.zeros(1, dtype=np.float32))
        self.assertTrue(v4_keys.issubset(step_info))

    def test_v3_wrapper_info_schema_does_not_gain_v4_only_terms(self):
        env = DMCContinuousEnv(
            domain_name="acrobot",
            task_name="swingup-v3",
            seed=23,
            raw_state_obs=True,
            time_sampling="uniform",
            dt=0.01,
            physics_dt=0.002,
            episode_duration=0.1,
            task_kwargs={"angle_noise": 0.0, "velocity_noise": 0.0},
        )
        self.addCleanup(env.close)

        _, info = env.reset(seed=23)
        v4_only_keys = {
            "acrobot_energy_norm",
            "acrobot_speed",
            "acrobot_slow_gate",
            "acrobot_hold",
            "acrobot_strict_capture",
        }
        self.assertTrue(v4_only_keys.isdisjoint(info))
        self.assertIn("acrobot_success", info)


@unittest.skipUnless(HAVE_DMC, "dm_control / Acrobot-v2 not available")
class TestAcrobotSwingupV41OvershootMargin(unittest.TestCase):
    def setUp(self):
        self.physics = dmc_acrobot.Physics.from_xml_string(
            *dmc_acrobot.get_model_and_assets()
        )
        kwargs = {"random": 0, "angle_noise": 0.0, "velocity_noise": 0.0}
        self.v4 = BalanceV4(**kwargs)
        self.v41 = BalanceV4(
            **kwargs,
            energy_overshoot_margin=V41_ENERGY_OVERSHOOT_MARGIN,
            speed_bounds=V41_SPEED_BOUNDS,
            speed_margin=V41_SPEED_MARGIN,
        )
        self.v4.initialize_episode(self.physics)
        self.v41.initialize_episode(self.physics)

    def _set_physics_state(self, qpos, qvel=(0.0, 0.0)):
        with self.physics.reset_context():
            self.physics.data.qpos[:] = np.asarray(qpos, dtype=np.float64)
            self.physics.data.qvel[:] = np.asarray(qvel, dtype=np.float64)
            self.physics.data.ctrl[:] = 0.0

    def test_default_margin_keeps_v4_reward_identical_everywhere(self):
        default_task = BalanceV4(random=0, angle_noise=0.0, velocity_noise=0.0)
        default_task.initialize_episode(self.physics)
        self.assertEqual(default_task.energy_overshoot_margin, 1.0)
        rng = np.random.default_rng(7)
        for pose, velocity in zip(
            rng.uniform(-2.0 * np.pi, 2.0 * np.pi, size=(20, 2)),
            rng.uniform(-6.0, 6.0, size=(20, 2)),
        ):
            self._set_physics_state(pose, qvel=velocity)
            self.assertEqual(
                default_task.reward_terms(self.physics)["reward"],
                self.v4.reward_terms(self.physics)["reward"],
            )

    def test_energy_progress_identical_at_or_below_unity_energy(self):
        for qpos, qvel in (
            ((np.pi, 0.0), (0.0, 0.0)),
            ((0.0, np.pi), (0.0, 0.0)),
            ((2.2, 1.0), (2.0, -1.5)),
            ((0.0, 0.0), (0.0, 0.0)),
        ):
            with self.subTest(qpos=qpos, qvel=qvel):
                self._set_physics_state(qpos, qvel=qvel)
                t4 = self.v4.reward_terms(self.physics)
                t41 = self.v41.reward_terms(self.physics)
                self.assertLessEqual(t41["energy_norm"], 1.0)
                self.assertAlmostEqual(
                    t41["energy_norm"], t4["energy_norm"], places=12
                )
                self.assertAlmostEqual(
                    t41["progress"], t4["progress"], places=12
                )

    def test_overshoot_states_lose_their_ramp_income(self):
        # Fast spin through the top: the regime the v4 pilots converged to.
        self._set_physics_state((0.0, 0.0), qvel=(3.5, 0.0))
        t4 = self.v4.reward_terms(self.physics)
        t41 = self.v41.reward_terms(self.physics)
        self.assertGreater(t4["energy_norm"], 1.3)
        self.assertGreater(t4["reward"], 0.1)
        self.assertLess(t41["reward"], 0.05)

        # Large surplus energy at the bottom is discounted to the floor.
        self._set_physics_state((np.pi, 0.0), qvel=(7.0, 0.0))
        self.assertLess(self.v41.reward_terms(self.physics)["reward"], 0.02)

    def test_mild_overshoot_keeps_a_gradient_back_toward_unity(self):
        # Just above Ẽ=1 the discount must be partial, not a cliff, so the
        # policy sees a slope back toward the homoclinic energy.
        self._set_physics_state((np.pi, 0.0), qvel=(5.65, 0.0))
        terms = self.v41.reward_terms(self.physics)
        self.assertGreater(terms["energy_norm"], 1.0)
        self.assertLess(terms["energy_norm"], 1.15)
        self.assertGreater(terms["progress"], 0.2)

    def test_goal_unchanged_and_moving_pass_loses_hold_income(self):
        self._set_physics_state((0.0, 0.0))
        self.assertAlmostEqual(
            self.v41.reward_terms(self.physics)["reward"], 1.0, places=6
        )
        self._set_physics_state((0.08, 0.05), qvel=(0.5, 0.4))
        t4 = self.v4.reward_terms(self.physics)
        t41 = self.v41.reward_terms(self.physics)
        self.assertGreater(t4["slow_gate"], 0.9)
        self.assertLess(t41["slow_gate"], 0.1)
        self.assertLess(t41["hold"], 0.1 * t4["hold"])
        self.assertGreater(t4["reward"] - t41["reward"], 0.5)

    def test_v41_speed_gate_defaults_and_landmarks(self):
        self.assertEqual(V41_SPEED_BOUNDS, (0.0, 0.1))
        self.assertEqual(V41_SPEED_MARGIN, 0.5)
        self.assertEqual(self.v41.speed_bounds, V41_SPEED_BOUNDS)
        self.assertEqual(self.v41.speed_margin, V41_SPEED_MARGIN)

        self._set_physics_state((0.0, 0.0), qvel=(0.1, 0.0))
        self.assertAlmostEqual(
            self.v41.reward_terms(self.physics)["slow_gate"], 1.0
        )

        # 0.6 is exactly upper_bound + margin, so the Gaussian tolerance
        # reaches its published value_at_margin.
        self._set_physics_state((0.0, 0.0), qvel=(0.6, 0.0))
        self.assertAlmostEqual(
            self.v41.reward_terms(self.physics)["slow_gate"], 0.1
        )
        self.assertGreater(
            self.v4.reward_terms(self.physics)["slow_gate"], 0.9
        )

    def test_invalid_overshoot_margin_rejected(self):
        for margin in (0.0, -0.25, float("nan"), float("inf")):
            with self.subTest(margin=margin):
                with self.assertRaises(ValueError):
                    BalanceV4(random=0, energy_overshoot_margin=margin)

    def test_factory_defaults_to_capture_pressure_and_uniform_start(self):
        env = swingup_v41(
            time_limit=0.1,
            random=19,
            environment_kwargs={"flat_observation": True},
            velocity_noise=0.0,
        )
        try:
            self.assertIsInstance(env.task, BalanceV4)
            self.assertEqual(
                env.task.energy_overshoot_margin, V41_ENERGY_OVERSHOOT_MARGIN
            )
            self.assertEqual(env.task.speed_bounds, V41_SPEED_BOUNDS)
            self.assertEqual(env.task.speed_margin, V41_SPEED_MARGIN)
            self.assertTrue(env.task.uniform_start)
            starts = []
            for _ in range(40):
                env.reset()
                starts.append(np.array(env.physics.data.qpos))
            starts = np.stack(starts)
            # Not the near-hanging reset: angles cover the circle.
            self.assertGreater(np.ptp(starts[:, 0]), np.pi)
            self.assertGreater(np.ptp(starts[:, 1]), np.pi)
        finally:
            env.close()

    def test_uniform_start_puts_hold_income_in_the_start_distribution(self):
        # The reason uniform starts are needed: from hanging the capture
        # region is unreachable without the penalized overshoot, so the hold
        # term is never observed. Under uniform starts a meaningful share of
        # resets begin above the height already earning hold reward.
        env = swingup_v41(time_limit=0.1, random=7, velocity_noise=0.0)
        self.addCleanup(env.close)
        physics = env.physics
        above = 0
        hold_income = 0.0
        n = 400
        for _ in range(n):
            env.reset()
            terms = env.task.reward_terms(physics)
            if terms["tip_height"] > 3.0:
                above += 1
                hold_income += terms["hold"]
        self.assertGreater(above, n // 10)
        self.assertLess(above, n // 2)
        # Average hold over the whole start stream clears the 0.05 gate that
        # left the hanging-start v4.1 best_model empty.
        self.assertGreater(hold_income / n, 0.05)

    def test_uniform_start_false_restores_hanging_reset(self):
        env = swingup_v41(
            time_limit=0.1,
            random=19,
            environment_kwargs={"flat_observation": True},
            angle_noise=0.0,
            velocity_noise=0.0,
            uniform_start=False,
        )
        try:
            self.assertFalse(env.task.uniform_start)
            env.reset()
            np.testing.assert_array_equal(env.physics.data.qpos, [np.pi, 0.0])
        finally:
            env.close()

    def test_energy_calibration_survives_uniform_start(self):
        env = swingup_v41(time_limit=0.1, random=3, velocity_noise=0.0)
        self.addCleanup(env.close)
        env.reset()
        physics = env.physics
        for qpos, expected in (((np.pi, 0.0), 0.0), ((0.0, 0.0), 1.0)):
            with physics.reset_context():
                physics.data.qpos[:] = qpos
                physics.data.qvel[:] = 0.0
            self.assertAlmostEqual(
                env.task.reward_terms(physics)["energy_norm"],
                expected,
                places=9,
            )

    def test_plain_v4_factory_keeps_hanging_start(self):
        env = swingup_v4(
            time_limit=0.1, random=19, angle_noise=0.0, velocity_noise=0.0
        )
        try:
            self.assertFalse(env.task.uniform_start)
            env.reset()
            np.testing.assert_array_equal(env.physics.data.qpos, [np.pi, 0.0])
        finally:
            env.close()

    def test_wrapper_registration_and_uniform_start_override(self):
        wrapped = DMCContinuousEnv(
            domain_name="acrobot",
            task_name="swingup-v4.1",
            seed=23,
            raw_state_obs=True,
            time_sampling="uniform",
            dt=0.01,
            physics_dt=0.002,
            episode_duration=0.1,
            task_kwargs={"angle_noise": 0.0, "velocity_noise": 0.0},
        )
        self.addCleanup(wrapped.close)
        _, info = wrapped.reset(seed=23)
        self.assertEqual(
            wrapped._env.task.energy_overshoot_margin,
            V41_ENERGY_OVERSHOOT_MARGIN,
        )
        self.assertEqual(wrapped._env.task.speed_bounds, V41_SPEED_BOUNDS)
        self.assertEqual(wrapped._env.task.speed_margin, V41_SPEED_MARGIN)
        self.assertTrue(wrapped._env.task.uniform_start)
        self.assertIn("acrobot_energy_norm", info)
        self.assertIn("acrobot_strict_capture", info)

        down = DMCContinuousEnv(
            domain_name="acrobot",
            task_name="swingup-v4.1",
            seed=23,
            raw_state_obs=True,
            time_sampling="uniform",
            dt=0.01,
            physics_dt=0.002,
            episode_duration=0.1,
            task_kwargs={
                "angle_noise": 0.0,
                "velocity_noise": 0.0,
                "uniform_start": False,
            },
        )
        self.addCleanup(down.close)
        obs, _ = down.reset(seed=23)
        self.assertFalse(down._env.task.uniform_start)
        np.testing.assert_array_equal(
            down._env.physics.data.qpos, [np.pi, 0.0]
        )


@unittest.skipUnless(HAVE_DMC, "dm_control / Acrobot-v2 not available")
class TestAcrobotSwingupV5GymObjective(unittest.TestCase):
    def setUp(self):
        self.physics = dmc_acrobot.Physics.from_xml_string(
            *dmc_acrobot.get_model_and_assets()
        )
        self.task = BalanceV5(random=0, angle_noise=0.0, velocity_noise=0.0)

    def _set_physics_state(self, qpos, qvel=(0.0, 0.0)):
        with self.physics.reset_context():
            self.physics.data.qpos[:] = np.asarray(qpos, dtype=np.float64)
            self.physics.data.qvel[:] = np.asarray(qvel, dtype=np.float64)
            self.physics.data.ctrl[:] = 0.0

    @staticmethod
    def _pump_action(physics, step):
        """Kick, then bang-bang elbow torque against the shoulder swing."""
        if step < 100:
            return 1.0
        return float(-np.sign(float(physics.data.qvel[0])))

    def test_factory_default_uses_uniform_random_starts(self):
        env = swingup_v5(
            time_limit=0.1,
            random=19,
            environment_kwargs={"flat_observation": True},
            velocity_noise=0.0,
        )
        try:
            self.assertTrue(env.task.uniform_start)
            starts, above = [], 0
            for _ in range(60):
                env.reset()
                qpos = np.array(env.physics.data.qpos)
                starts.append(qpos)
                tip = float(env.physics.named.data.site_xpos["tip", "z"])
                above += int(tip > 3.0)
            starts = np.stack(starts)
            # Angles cover the circle, not the near-hanging neighborhood.
            self.assertGreater(np.ptp(starts[:, 0]), np.pi)
            self.assertGreater(np.ptp(starts[:, 1]), np.pi)
            # ~18.5 % of uniform resets begin above the height criterion, so
            # the sparse income exists in the start distribution itself.
            self.assertGreater(above, 0)
            self.assertLess(above, 40)
        finally:
            env.close()

    def test_uniform_start_resets_are_reseed_repeatable(self):
        task_a = BalanceV5(random=11, velocity_noise=0.0)
        task_b = BalanceV5(random=999, velocity_noise=0.0)
        physics_a = dmc_acrobot.Physics.from_xml_string(
            *dmc_acrobot.get_model_and_assets()
        )
        physics_b = dmc_acrobot.Physics.from_xml_string(
            *dmc_acrobot.get_model_and_assets()
        )
        task_b.reseed(11)
        task_a.initialize_episode(physics_a)
        task_b.initialize_episode(physics_b)
        np.testing.assert_array_equal(
            physics_a.data.qpos, physics_b.data.qpos
        )

    def test_down_start_option_matches_v2_reset(self):
        env = swingup_v5(
            time_limit=0.1,
            random=19,
            environment_kwargs={"flat_observation": True},
            angle_noise=0.0,
            velocity_noise=0.0,
            uniform_start=False,
        )
        try:
            env.reset()
            self.assertIsInstance(env.task, BalanceV5)
            np.testing.assert_array_equal(env.physics.data.qpos, [np.pi, 0.0])
            np.testing.assert_array_equal(env.physics.data.qvel, [0.0, 0.0])
        finally:
            env.close()

    def test_height_occupancy_reward_landmarks(self):
        # Hanging: no income, no termination anywhere in this task.
        self._set_physics_state((np.pi, 0.0))
        terms = self.task.reward_terms(self.physics)
        self.assertEqual(terms["reward"], 0.0)
        self.assertEqual(terms["gym_height_success"], 0.0)
        self.assertEqual(terms["progress"], 0.0)
        self.assertIsNone(self.task.get_termination(self.physics))

        # Straight links at shoulder pi/3: tip exactly 3.0, strictly below
        # the height predicate.
        self._set_physics_state((np.pi / 3.0, 0.0))
        terms = self.task.reward_terms(self.physics)
        self.assertAlmostEqual(terms["tip_height"], 3.0)
        self.assertEqual(terms["reward"], 0.0)

        # Just above the threshold: full occupancy income, episode continues.
        self._set_physics_state((np.pi / 3.0 - 1e-3, 0.0))
        terms = self.task.reward_terms(self.physics)
        self.assertGreater(terms["tip_height"], 3.0)
        self.assertEqual(terms["reward"], 1.0)
        self.assertEqual(terms["gym_height_success"], 1.0)
        self.assertIsNone(self.task.get_termination(self.physics))

        # Upright at the target: same occupancy income; success stays the
        # exact target hit.
        self._set_physics_state((0.0, 0.0))
        terms = self.task.reward_terms(self.physics)
        self.assertEqual(terms["reward"], 1.0)
        self.assertIsNone(self.task.get_termination(self.physics))
        self.assertEqual(terms["success"], terms["exact_success"])
        self.assertEqual(terms["exact_success"], 1.0)

    def test_scripted_pump_accrues_occupancy_without_ending_the_episode(self):
        env = swingup_v5(
            time_limit=30.0,
            random=3,
            angle_noise=0.0,
            velocity_noise=0.0,
            uniform_start=False,
        )
        self.addCleanup(env.close)
        env.reset()
        physics = env.physics
        rewards_seen = []
        first_above = None
        for step in range(2900):
            action = self._pump_action(physics, step)
            ts = env.step(np.asarray([action]))
            rewards_seen.append(float(ts.reward))
            if first_above is None and rewards_seen[-1] > 0.0:
                first_above = step
            self.assertFalse(
                ts.last(), "height crossing must not end the episode"
            )
        self.assertIsNotNone(first_above, "pump never exceeded the height")
        self.assertLess(first_above, 2500)
        # Income continues to accrue after the first crossing.
        self.assertGreater(sum(rewards_seen[first_above:]), 1.0)
        self.assertTrue(set(rewards_seen) <= {0.0, 1.0})

    def test_continuous_wrapper_truncates_at_duration_with_occupancy_income(self):
        env = DMCContinuousEnv(
            domain_name="acrobot",
            task_name="swingup-v5",
            seed=23,
            raw_state_obs=True,
            time_sampling="uniform",
            dt=0.01,
            physics_dt=0.002,
            episode_duration=14.0,
            task_kwargs={
                "angle_noise": 0.0,
                "velocity_noise": 0.0,
                "uniform_start": False,
            },
        )
        self.addCleanup(env.close)
        env.reset(seed=23)
        physics = env._env.physics

        total = 0.0
        seen_above = False
        terminated = truncated = False
        for step in range(1500):
            action = np.asarray([self._pump_action(physics, step)], np.float32)
            _, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            seen_above = seen_above or info["acrobot_gym_height_success"] == 1.0
            self.assertFalse(terminated)
            if truncated:
                break
        self.assertTrue(truncated)
        self.assertTrue(seen_above)
        self.assertGreater(total, 1.0)

    def test_continuous_wrapper_time_limit_truncates_without_termination(self):
        env = DMCContinuousEnv(
            domain_name="acrobot",
            task_name="swingup-v5",
            seed=23,
            raw_state_obs=True,
            time_sampling="uniform",
            dt=0.01,
            physics_dt=0.002,
            episode_duration=0.05,
            task_kwargs={
                "angle_noise": 0.0,
                "velocity_noise": 0.0,
                "uniform_start": False,
            },
        )
        self.addCleanup(env.close)
        env.reset(seed=23)

        terminated = truncated = False
        for _ in range(10):
            _, reward, terminated, truncated, _ = env.step(
                np.zeros(1, dtype=np.float32)
            )
            if terminated or truncated:
                break
        self.assertTrue(truncated)
        self.assertFalse(terminated)
        self.assertEqual(reward, 0.0)

    def test_dmc_internal_step_limit_maps_to_truncation_not_termination(self):
        # dm_control's own step limit emits LAST with discount 1; the wrapper
        # must report that as truncation so bootstrapping continues.
        env = DMCContinuousEnv(
            domain_name="acrobot",
            task_name="swingup-v2",
            seed=23,
            raw_state_obs=True,
            time_sampling="uniform",
            dt=0.01,
            physics_dt=0.002,
            max_steps=3,
            episode_duration=10.0,
            task_kwargs={"angle_noise": 0.0, "velocity_noise": 0.0},
        )
        self.addCleanup(env.close)
        env.reset(seed=23)

        terminated = truncated = False
        for _ in range(3):
            _, _, terminated, truncated, info = env.step(
                np.zeros(1, dtype=np.float32)
            )
            if terminated or truncated:
                break
        self.assertTrue(truncated)
        self.assertFalse(terminated)
        self.assertEqual(float(info["discount"]), 1.0)

    def test_v5_wrapper_info_schema_has_v3_family_terms_but_no_v4_terms(self):
        env = DMCContinuousEnv(
            domain_name="acrobot",
            task_name="swingup-v5",
            seed=23,
            raw_state_obs=True,
            time_sampling="uniform",
            dt=0.01,
            physics_dt=0.002,
            episode_duration=0.1,
            task_kwargs={
                "angle_noise": 0.0,
                "velocity_noise": 0.0,
                "uniform_start": False,
            },
        )
        self.addCleanup(env.close)

        _, info = env.reset(seed=23)
        v3_family_keys = {
            "acrobot_upper_uprightness",
            "acrobot_lower_uprightness",
            "acrobot_extension",
            "acrobot_gym_height_success",
            "acrobot_exact_success",
        }
        v4_only_keys = {
            "acrobot_energy_norm",
            "acrobot_speed",
            "acrobot_slow_gate",
            "acrobot_hold",
            "acrobot_strict_capture",
        }
        self.assertTrue(v3_family_keys.issubset(info))
        self.assertTrue(v4_only_keys.isdisjoint(info))
        self.assertEqual(info["acrobot_progress"], 0.0)
        self.assertEqual(info["acrobot_gym_height_success"], 0.0)


@unittest.skipUnless(HAVE_DMC, "dm_control / Acrobot-v2 not available")
class TestAcrobotSwingupV42Curriculum(unittest.TestCase):
    _MIN_SPREAD = 0.5

    def setUp(self):
        self.physics = dmc_acrobot.Physics.from_xml_string(
            *dmc_acrobot.get_model_and_assets()
        )
        self.task = BalanceV4(
            random=0,
            angle_noise=0.0,
            velocity_noise=0.0,
            curriculum=True,
            curriculum_min_spread=self._MIN_SPREAD,
        )

    def _draw_qpos(self, fraction, n=400):
        self.task.set_curriculum_fraction(fraction)
        draws = np.empty((n, 2))
        for i in range(n):
            self.task.initialize_episode(self.physics)
            draws[i] = np.asarray(self.physics.data.qpos, dtype=np.float64)
        return draws

    def test_fraction_zero_confines_starts_to_the_upright_band(self):
        # Upright rest is (shoulder, elbow) = (0, 0); at fraction 0 both joints
        # stay within +/- curriculum_min_spread of it.
        draws = self._draw_qpos(0.0)
        self.assertLessEqual(np.abs(draws).max(), self._MIN_SPREAD + 1e-9)
        # And the band is actually used, not collapsed to a point.
        self.assertGreater(np.abs(draws).max(), 0.4 * self._MIN_SPREAD)

    def test_band_half_width_grows_monotonically_with_fraction(self):
        widths = [np.abs(self._draw_qpos(f)).max() for f in (0.0, 0.25, 0.5, 1.0)]
        for lo, hi in zip(widths, widths[1:]):
            self.assertLess(lo, hi)

    def test_fraction_one_matches_the_uniform_reset_range(self):
        curriculum = self._draw_qpos(1.0)
        # Independent uniform-reset task with the same RNG seed / noise.
        uniform_task = BalanceV4(
            random=0, angle_noise=0.0, velocity_noise=0.0, uniform_start=True
        )
        uniform = np.empty_like(curriculum)
        for i in range(len(uniform)):
            uniform_task.initialize_episode(self.physics)
            uniform[i] = np.asarray(self.physics.data.qpos, dtype=np.float64)
        # Both span essentially the full [-pi, pi] circle on each joint.
        for arr in (curriculum, uniform):
            self.assertGreater(arr.max(), np.pi - 0.1)
            self.assertLess(arr.min(), -(np.pi - 0.1))
        # Distributions match in spread (same underlying uniform draw at f=1).
        np.testing.assert_allclose(
            curriculum.std(axis=0), uniform.std(axis=0), atol=0.15
        )

    def test_set_curriculum_fraction_clamps_to_unit_interval(self):
        self.task.set_curriculum_fraction(-0.5)
        self.assertEqual(self.task.curriculum_fraction, 0.0)
        self.task.set_curriculum_fraction(2.0)
        self.assertEqual(self.task.curriculum_fraction, 1.0)
        self.task.set_curriculum_fraction(0.3)
        self.assertAlmostEqual(self.task.curriculum_fraction, 0.3)
        with self.assertRaises(ValueError):
            self.task.set_curriculum_fraction(float("nan"))

    def test_invalid_min_spread_rejected(self):
        for bad in (0.0, -0.1, np.pi + 0.1, float("nan")):
            with self.assertRaises(ValueError):
                BalanceV4(random=0, curriculum=True, curriculum_min_spread=bad)

    def test_factory_v42_pairs_curriculum_reset_with_the_v41_reward(self):
        env = swingup_v42(
            time_limit=0.1,
            random=7,
            environment_kwargs={"flat_observation": True},
            angle_noise=0.0,
            velocity_noise=0.0,
        )
        try:
            env.reset()
            task = env.task
            self.assertIsInstance(task, BalanceV4)
            self.assertTrue(task.curriculum)
            # Reward identical to v4.1: tightened overshoot margin and slow gate.
            self.assertEqual(
                task.energy_overshoot_margin, V41_ENERGY_OVERSHOOT_MARGIN
            )
            self.assertEqual(task.speed_bounds, V41_SPEED_BOUNDS)
            self.assertEqual(task.speed_margin, V41_SPEED_MARGIN)
        finally:
            env.close()

    def test_curriculum_disabled_falls_back_to_the_uniform_reset(self):
        # Evaluation builds the task with curriculum off; it must then honor
        # uniform_start regardless of any fraction pushed onto it.
        task = BalanceV4(
            random=0,
            angle_noise=0.0,
            velocity_noise=0.0,
            curriculum=False,
            uniform_start=True,
        )
        task.set_curriculum_fraction(0.0)
        draws = np.empty((200, 2))
        for i in range(len(draws)):
            task.initialize_episode(self.physics)
            draws[i] = np.asarray(self.physics.data.qpos, dtype=np.float64)
        # Full uniform range, not the tight fraction-0 band.
        self.assertGreater(draws.max(), np.pi - 0.2)
        self.assertLess(draws.min(), -(np.pi - 0.2))


@unittest.skipUnless(HAVE_DMC, "dm_control / Acrobot-v2 not available")
class TestAcrobotSwingupV6QuadraticCost(unittest.TestCase):
    """The AR-EAPO quadratic cost (Choe et al., 2024, eq. 16) on v4.2's reset."""

    def setUp(self):
        self.physics = dmc_acrobot.Physics.from_xml_string(
            *dmc_acrobot.get_model_and_assets()
        )
        self.task = BalanceV6(
            random=0,
            angle_noise=0.0,
            velocity_noise=0.0,
            curriculum=False,
            uniform_start=False,
        )

    def _terms(self, qpos, qvel=(0.0, 0.0), ctrl=0.0):
        self.physics.named.data.qpos[["shoulder", "elbow"]] = qpos
        self.physics.data.qvel[:] = np.asarray(qvel, dtype=np.float64)
        self.physics.data.ctrl[:] = ctrl
        self.physics.forward()
        return self.task.reward_terms(self.physics)

    def test_defaults_are_the_published_ar_eapo_weights(self):
        self.assertEqual(V6_STATE_WEIGHTS, (50.0, 50.0, 4.0, 2.0))
        self.assertEqual(V6_ACTION_WEIGHT, 1.0)
        self.assertEqual(V6_COST_SCALE, 0.001)
        self.assertEqual(self.task.state_weights, V6_STATE_WEIGHTS)
        self.assertEqual(self.task.action_weight, V6_ACTION_WEIGHT)
        self.assertEqual(self.task.cost_scale, V6_COST_SCALE)
        self.assertEqual(self.task.reward_offset, 0.0)

    def test_reward_is_minus_the_weighted_square_of_state_and_command(self):
        qpos, qvel, ctrl = (0.4, -0.3), (1.5, -2.5), 0.6
        terms = self._terms(qpos, qvel, ctrl)
        w = V6_STATE_WEIGHTS
        expected = -V6_COST_SCALE * (
            w[0] * qpos[0] ** 2
            + w[1] * qpos[1] ** 2
            + w[2] * qvel[0] ** 2
            + w[3] * qvel[1] ** 2
            + V6_ACTION_WEIGHT * ctrl**2
        )
        self.assertAlmostEqual(terms["reward"], expected, places=12)
        # The three parts sum back to the cost.
        self.assertAlmostEqual(
            terms["angle_cost"] + terms["velocity_cost"] + terms["action_cost"],
            -expected,
            places=12,
        )

    def test_only_the_upright_rest_pose_at_zero_command_is_costless(self):
        self.assertEqual(self._terms((0.0, 0.0))["reward"], 0.0)
        for qpos, qvel, ctrl in (
            ((np.pi, 0.0), (0.0, 0.0), 0.0),  # hanging
            ((0.0, np.pi), (0.0, 0.0), 0.0),  # folded above the pivot
            ((0.0, 0.0), (0.0, 0.0), 1.0),  # upright but pushing
            ((0.0, 0.0), (3.0, 0.0), 0.0),  # upright but spinning through
        ):
            self.assertLess(self._terms(qpos, qvel, ctrl)["reward"], 0.0)

    def test_position_cost_is_monotone_along_the_swing_up(self):
        # Every shoulder angle between hanging and upright pays strictly less
        # than the one below it: the slope v4's energy shell does not provide.
        angles = np.linspace(np.pi, 0.0, 25)
        rewards_along = [self._terms((float(a), 0.0))["reward"] for a in angles]
        for lo, hi in zip(rewards_along, rewards_along[1:]):
            self.assertLess(lo, hi)
        self.assertAlmostEqual(rewards_along[-1], 0.0, places=12)

    def test_hanging_rest_pays_the_published_cost(self):
        # alpha * Q_1 * pi^2 with the published alpha = 0.001, Q_1 = 50.
        self.assertAlmostEqual(
            self._terms((np.pi, 0.0))["reward"],
            -V6_COST_SCALE * V6_STATE_WEIGHTS[0] * np.pi**2,
            places=12,
        )

    def test_angle_error_is_wrapped_across_the_branch_cut(self):
        below = self._terms((np.pi - 1e-3, 0.0))["reward"]
        at = self._terms((np.pi, 0.0))["reward"]
        above = self._terms((-np.pi + 1e-3, 0.0))["reward"]
        self.assertAlmostEqual(below, above, places=12)
        self.assertLess(at, below)
        # A full turn is the same pose, so it costs the same.
        self.assertAlmostEqual(
            self._terms((0.3, 0.0))["reward"],
            self._terms((0.3 + 2.0 * np.pi, 0.0))["reward"],
            places=10,
        )

    def test_reward_offset_shifts_uniformly_and_leaves_costs_untouched(self):
        offset = 0.5
        shifted = BalanceV6(
            random=0, curriculum=False, uniform_start=False, reward_offset=offset
        )
        for qpos, qvel in (((np.pi, 0.0), (0.0, 0.0)), ((0.2, -0.4), (2.0, 3.0))):
            base = self._terms(qpos, qvel)
            self.physics.named.data.qpos[["shoulder", "elbow"]] = qpos
            self.physics.data.qvel[:] = qvel
            self.physics.data.ctrl[:] = 0.0
            self.physics.forward()
            moved = shifted.reward_terms(self.physics)
            self.assertAlmostEqual(moved["reward"], base["reward"] + offset, 12)
            self.assertAlmostEqual(moved["angle_cost"], base["angle_cost"], 12)
            self.assertAlmostEqual(
                moved["velocity_cost"], base["velocity_cost"], 12
            )

    def test_weights_can_be_reweighted_per_component(self):
        task = BalanceV6(
            random=0,
            curriculum=False,
            uniform_start=False,
            state_weights=(1.0, 0.0, 0.0, 0.0),
            action_weight=0.0,
            cost_scale=1.0,
        )
        self.physics.named.data.qpos[["shoulder", "elbow"]] = (0.5, 2.0)
        self.physics.data.qvel[:] = (7.0, -9.0)
        self.physics.data.ctrl[:] = 1.0
        self.physics.forward()
        terms = task.reward_terms(self.physics)
        self.assertAlmostEqual(terms["reward"], -0.25, places=12)
        self.assertEqual(terms["velocity_cost"], 0.0)
        self.assertEqual(terms["action_cost"], 0.0)

    def test_progress_diagnostic_spans_the_unit_interval(self):
        self.assertAlmostEqual(self._terms((0.0, 0.0))["progress"], 1.0, 12)
        self.assertAlmostEqual(
            self._terms((np.pi, np.pi))["progress"], 0.0, places=12
        )
        self.assertAlmostEqual(
            self.task.max_angle_cost,
            V6_COST_SCALE
            * (V6_STATE_WEIGHTS[0] + V6_STATE_WEIGHTS[1])
            * np.pi**2,
            places=12,
        )

    def test_strict_capture_matches_the_shared_v4_family_thresholds(self):
        near = self._terms((0.0, 0.0), (0.5 * STRICT_CAPTURE_SPEED, 0.0))
        self.assertEqual(near["strict_capture"], 1.0)
        self.assertLess(near["tip_distance"], STRICT_CAPTURE_DISTANCE)
        fast = self._terms((0.0, 0.0), (2.0 * STRICT_CAPTURE_SPEED, 0.0))
        self.assertEqual(fast["strict_capture"], 0.0)
        self.assertEqual(self._terms((np.pi, 0.0))["strict_capture"], 0.0)

    def test_invalid_parameters_are_rejected(self):
        for bad in ((50.0, 50.0, 4.0), (50.0, 50.0, 4.0, -1.0), (np.nan,) * 4):
            with self.assertRaises(ValueError):
                BalanceV6(random=0, state_weights=bad)
        for bad in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                BalanceV6(random=0, cost_scale=bad)
        for bad in (-1.0, float("nan")):
            with self.assertRaises(ValueError):
                BalanceV6(random=0, action_weight=bad)
        with self.assertRaises(ValueError):
            BalanceV6(random=0, reward_offset=float("nan"))
        with self.assertRaises(ValueError):
            BalanceV6(random=0, curriculum=True, curriculum_min_spread=0.0)

    def test_factory_pairs_the_cost_with_the_v42_curriculum_reset(self):
        env = swingup_v6(
            time_limit=0.1,
            random=7,
            environment_kwargs={"flat_observation": True},
            angle_noise=0.0,
            velocity_noise=0.0,
        )
        try:
            env.reset()
            task = env.task
            self.assertIsInstance(task, BalanceV6)
            self.assertTrue(task.curriculum)
            self.assertTrue(task.uniform_start)
            self.assertEqual(task.state_weights, V6_STATE_WEIGHTS)
            # Curriculum band and its fraction plumbing behave exactly as v4.2's.
            task.set_curriculum_fraction(0.0)
            draws = np.empty((200, 2))
            for i in range(len(draws)):
                task.initialize_episode(env.physics)
                draws[i] = np.asarray(env.physics.data.qpos, dtype=np.float64)
            self.assertLessEqual(
                np.abs(draws).max(), task.curriculum_min_spread + 1e-9
            )
        finally:
            env.close()

    def test_episode_runs_to_the_time_limit_without_terminating(self):
        # A continuing task: the average-reward criterion needs the cost to be
        # unavoidable rather than escapable by ending the episode.
        env = swingup_v6(time_limit=0.2, random=1)
        try:
            time_step = env.reset()
            steps, total = 0, 0.0
            while not time_step.last():
                time_step = env.step(np.array([1.0], dtype=np.float32))
                total += float(time_step.reward)
                steps += 1
            self.assertGreater(steps, 1)
            self.assertLess(total, 0.0)
            # Truncation, not termination: dm_control keeps discount 1.
            self.assertEqual(float(time_step.discount), 1.0)
        finally:
            env.close()

    def test_wrapper_registration_exposes_cost_terms_and_not_v4_terms(self):
        env = DMCContinuousEnv(
            domain_name="acrobot",
            task_name="swingup-v6",
            seed=0,
            raw_state_obs=True,
            time_sampling="uniform",
            dt=0.01,
            physics_dt=0.002,
            episode_duration=0.1,
            task_kwargs={"curriculum": False, "uniform_start": False},
        )
        self.addCleanup(env.close)
        self.assertTrue(env.raw_state_obs)
        self.assertFalse(env.has_curriculum)
        obs, info = env.reset()
        self.assertEqual(obs.shape, (4,))
        _, reward, terminated, truncated, info = env.step(
            np.array([1.0], dtype=np.float32)
        )
        self.assertFalse(terminated)
        self.assertLess(reward, 0.0)
        for key in (
            "acrobot_angle_cost",
            "acrobot_velocity_cost",
            "acrobot_action_cost",
            "acrobot_energy_norm",
            "acrobot_kinetic_norm",
            "acrobot_velocity_cost_per_joule",
            "acrobot_coordination_loss",
            "acrobot_speed",
            "acrobot_strict_capture",
            "acrobot_gym_height_success",
        ):
            self.assertIn(key, info)
        # energy_norm is a shared physics diagnostic; slow_gate/hold are v4
        # reward terms and have no counterpart under the quadratic cost.
        for key in ("acrobot_slow_gate", "acrobot_hold"):
            self.assertNotIn(key, info)
        del truncated

    def test_curriculum_env_ids_and_capture_rule_cover_v6(self):
        from evaluations.sustained_capture import strict_capture_spec_for

        for env_id in ("acrobot-swingup-v6", "acrobot-swingup-v6-uniform"):
            self.assertIsNotNone(
                strict_capture_spec_for(algorithm="ct_sac", env_id=env_id)
            )
        # The trainer attaches the curriculum callback by env-id suffix: the
        # scheduled arm must match it and the uniform arm must not.
        suffixes = ("-v4.2", "-v6", "-curriculum")
        self.assertTrue("acrobot-swingup-v6".endswith(suffixes))
        self.assertFalse("acrobot-swingup-v6-uniform".endswith(suffixes))


@unittest.skipUnless(HAVE_DMC, "dm_control / Acrobot-v2 not available")
class TestAcrobotSwingupV6ExplorationDiagnostics(unittest.TestCase):
    """energy_norm and the per-joule ratio factor the velocity cost apart."""

    def setUp(self):
        self.physics = dmc_acrobot.Physics.from_xml_string(
            *dmc_acrobot.get_model_and_assets()
        )
        self.task = BalanceV6(
            random=0, curriculum=False, uniform_start=False,
            angle_noise=0.0, velocity_noise=0.0,
        )
        self.task._ensure_energy_calibrated(self.physics)
        self.span = self.task._energy_span
        self.M = self._mass_at([np.pi, 0.0])
        W = self.task.velocity_cost_matrix
        # Generalized modes of the pencil (W, M): cheapest and dearest per joule.
        self.cheap, self.dear = self._generalized_modes(W, self.M)

    @staticmethod
    def _generalized_modes(W, M):
        L = np.linalg.cholesky(M)
        S = np.linalg.solve(L, np.linalg.solve(L, W).T)
        _, vecs = np.linalg.eigh(0.5 * (S + S.T))
        back = np.linalg.solve(L.T, vecs)
        return back[:, 0], back[:, -1]

    def _mass_at(self, qpos):
        self.physics.data.qvel[:] = 0.0
        self.physics.named.data.qpos[["shoulder", "elbow"]] = qpos
        self.physics.forward()
        return self.task._mass_matrix(self.physics)

    def _terms(self, qpos, qvel):
        self.physics.named.data.qpos[["shoulder", "elbow"]] = qpos
        self.physics.data.qvel[:] = np.asarray(qvel, dtype=np.float64)
        self.physics.data.ctrl[:] = 0.0
        self.physics.forward()
        return self.task.reward_terms(self.physics)

    def _mode_state(self, mode, fraction):
        """q̇ at the hanging pose carrying `fraction` of the span in `mode`."""
        return mode * np.sqrt(2.0 * fraction * self.span / float(mode @ self.M @ mode))

    def test_span_is_the_hanging_to_upright_energy_barrier(self):
        self.physics.data.qvel[:] = 0.0
        self.physics.named.data.qpos[["shoulder", "elbow"]] = [0.0, 0.0]
        self.physics.forward()
        up = self.task._mechanical_energy(self.physics)
        self.physics.named.data.qpos[["shoulder", "elbow"]] = [np.pi, 0.0]
        self.physics.forward()
        hang = self.task._mechanical_energy(self.physics)
        self.assertAlmostEqual(self.span, up - hang, places=9)
        self.assertGreater(self.span, 0.0)

    def test_energy_norm_counts_potential_as_well_as_kinetic(self):
        # Hanging at rest holds none of the budget; upright at rest holds all of
        # it as potential, with no kinetic share at all.
        low = self._terms([np.pi, 0.0], [0.0, 0.0])
        self.assertAlmostEqual(low["energy_norm"], 0.0, places=9)
        self.assertAlmostEqual(low["kinetic_norm"], 0.0, places=9)
        top = self._terms([0.0, 0.0], [0.0, 0.0])
        self.assertAlmostEqual(top["energy_norm"], 1.0, places=9)
        self.assertAlmostEqual(top["kinetic_norm"], 0.0, places=9)
        # At the hanging pose the whole budget must be kinetic, so the two agree.
        moving = self._terms([np.pi, 0.0], self._mode_state(self.cheap, 0.5))
        self.assertAlmostEqual(moving["energy_norm"], 0.5, places=6)
        self.assertAlmostEqual(moving["kinetic_norm"], 0.5, places=6)

    def test_per_joule_ratio_is_flat_in_amplitude_within_a_mode(self):
        for mode in (self.cheap, self.dear):
            ratios = [
                self._terms([np.pi, 0.0], self._mode_state(mode, f))[
                    "velocity_cost_per_joule"
                ]
                for f in (0.1, 0.5, 1.0)
            ]
            for r in ratios[1:]:
                self.assertAlmostEqual(r, ratios[0], places=9)

    def test_per_joule_ratio_separates_coordination_from_flailing(self):
        cheap = self._terms([np.pi, 0.0], self._mode_state(self.cheap, 1.0))
        dear = self._terms([np.pi, 0.0], self._mode_state(self.dear, 1.0))
        self.assertGreater(dear["velocity_cost_per_joule"], 20.0 * cheap["velocity_cost_per_joule"])
        self.assertAlmostEqual(cheap["coordination_loss"], 0.0, places=6)
        self.assertAlmostEqual(dear["coordination_loss"], 1.0, places=6)

    def test_velocity_cost_alone_cannot_separate_rest_from_coordinated_pumping(self):
        # The property motivating the pair: an efficient pump at full swing-up
        # energy is cheaper than a fifth of the hanging position cost, so the
        # cost column looks like rest while energy_norm does not.
        rest = self._terms([np.pi, 0.0], [0.0, 0.0])
        pumped = self._terms([np.pi, 0.0], self._mode_state(self.cheap, 1.0))
        self.assertLess(pumped["velocity_cost"], 0.2 * rest["angle_cost"])
        self.assertAlmostEqual(rest["energy_norm"], 0.0, places=9)
        self.assertAlmostEqual(pumped["energy_norm"], 1.0, places=6)

    def test_ratio_is_nan_at_rest_so_it_never_enters_a_running_mean(self):
        rest = self._terms([np.pi, 0.0], [0.0, 0.0])
        self.assertTrue(np.isnan(rest["velocity_cost_per_joule"]))
        self.assertTrue(np.isnan(rest["coordination_loss"]))
        # The energy terms stay finite: they are what reports "not moving".
        self.assertEqual(rest["kinetic_norm"], 0.0)
        self.assertEqual(rest["energy_norm"], 0.0)
        # And the reward itself is never NaN.
        self.assertTrue(np.isfinite(rest["reward"]))

    def test_resting_steps_cannot_drag_the_coordination_mean_toward_zero(self):
        # A finite sentinel at rest would bias the logged mean toward
        # "coordinated" exactly on a collapsed policy, which rests the most.
        rest = self._terms([np.pi, 0.0], [0.0, 0.0])["coordination_loss"]
        flail = self._terms([np.pi, 0.0], self._mode_state(self.dear, 1.0))[
            "coordination_loss"
        ]
        self.assertAlmostEqual(flail, 1.0, places=6)
        # Mean over a 70 % resting / 30 % flailing rollout, NaN dropped as the
        # Monitor drops it, is the flailing value rather than 0.3.
        samples = np.array([rest] * 7 + [flail] * 3)
        self.assertAlmostEqual(float(np.nanmean(samples)), 1.0, places=6)
        self.assertEqual(int(np.count_nonzero(~np.isnan(samples))), 3)

    def test_coordination_loss_is_pose_normalized(self):
        # Bounds move with the elbow angle; the normalized reading does not.
        for qpos in ([np.pi, 0.0], [np.pi, 1.0], [1.6, 1.5], [0.0, 2.0]):
            M = self._mass_at(qpos)
            cheap, dear = self._generalized_modes(self.task.velocity_cost_matrix, M)
            lo = self._terms(qpos, cheap * 3.0)["coordination_loss"]
            hi = self._terms(qpos, dear * 3.0)["coordination_loss"]
            self.assertAlmostEqual(lo, 0.0, places=6, msg=f"at {qpos}")
            self.assertAlmostEqual(hi, 1.0, places=6, msg=f"at {qpos}")

    def test_bounds_bracket_every_direction_and_vary_with_pose(self):
        rng = np.random.RandomState(0)
        self._mass_at([np.pi, 0.0])
        lo, hi = self.task._cost_per_joule_bounds(self.physics)
        for _ in range(200):
            terms = self._terms([np.pi, 0.0], rng.normal(size=2) * 5.0)
            self.assertGreaterEqual(terms["velocity_cost_per_joule"], lo - 1e-12)
            self.assertLessEqual(terms["velocity_cost_per_joule"], hi + 1e-12)
        self._mass_at([np.pi, 2.0])
        folded = self.task._cost_per_joule_bounds(self.physics)
        self.assertNotAlmostEqual(folded[1], hi, places=3)

    def test_diagnostics_do_not_enter_the_reward(self):
        # reward must stay exactly the published cost, diagnostics aside.
        terms = self._terms([0.4, -0.3], [1.5, -2.5])
        w = V6_STATE_WEIGHTS
        expected = -V6_COST_SCALE * (
            w[0] * 0.4**2 + w[1] * 0.3**2 + w[2] * 1.5**2 + w[3] * 2.5**2
        )
        self.assertAlmostEqual(terms["reward"], expected, places=12)

    def test_calibration_restores_the_state_it_clobbers(self):
        task = BalanceV6(random=0, curriculum=False, uniform_start=False)
        self.physics.named.data.qpos[["shoulder", "elbow"]] = [0.7, -1.1]
        self.physics.data.qvel[:] = [2.0, -3.0]
        self.physics.data.ctrl[:] = 0.5
        self.physics.forward()
        task._ensure_energy_calibrated(self.physics)
        np.testing.assert_allclose(self.physics.data.qpos, [0.7, -1.1], atol=1e-12)
        np.testing.assert_allclose(self.physics.data.qvel, [2.0, -3.0], atol=1e-12)
        np.testing.assert_allclose(self.physics.data.ctrl, [0.5], atol=1e-12)


@unittest.skipUnless(HAVE_DMC, "dm_control / Acrobot-v2 not available")
class TestAcrobotSwingupV6UniformStart(unittest.TestCase):
    """``swingup-v6-uniform``: the v6 cost with no reset schedule at all."""

    def setUp(self):
        self.physics = dmc_acrobot.Physics.from_xml_string(
            *dmc_acrobot.get_model_and_assets()
        )

    def test_reward_is_identical_to_the_curriculum_arm(self):
        scheduled = swingup_v6(time_limit=0.1, random=0).task
        uniform = swingup_v6_uniform(time_limit=0.1, random=0).task
        for qpos, qvel, ctrl in (
            ((np.pi, 0.0), (0.0, 0.0), 0.0),
            ((0.0, 0.0), (0.0, 0.0), 0.0),
            ((0.7, -1.3), (4.0, -6.0), 0.8),
        ):
            self.physics.named.data.qpos[["shoulder", "elbow"]] = qpos
            self.physics.data.qvel[:] = qvel
            self.physics.data.ctrl[:] = ctrl
            self.physics.forward()
            left = scheduled.reward_terms(self.physics)
            right = uniform.reward_terms(self.physics)
            self.assertEqual(set(left), set(right))
            for key, value in left.items():
                # The per-joule terms are NaN at rest, so compare NaN-aware.
                if isinstance(value, float) and np.isnan(value):
                    self.assertTrue(np.isnan(right[key]), msg=key)
                else:
                    self.assertEqual(value, right[key], msg=key)

    def test_task_carries_no_band_schedule(self):
        task = swingup_v6_uniform(time_limit=0.1, random=0).task
        self.assertFalse(task.curriculum)
        self.assertTrue(task.uniform_start)
        # Pushing a fraction onto it must not narrow the start distribution.
        task.set_curriculum_fraction(0.0)
        draws = np.empty((300, 2))
        for i in range(len(draws)):
            task.initialize_episode(self.physics)
            draws[i] = np.asarray(self.physics.data.qpos, dtype=np.float64)
        self.assertGreater(draws.max(), np.pi - 0.2)
        self.assertLess(draws.min(), -(np.pi - 0.2))

    def test_uniform_start_false_restores_the_hanging_reset(self):
        env = swingup_v6_uniform(
            time_limit=0.1,
            random=0,
            angle_noise=0.0,
            velocity_noise=0.0,
            uniform_start=False,
        )
        try:
            env.reset()
            np.testing.assert_allclose(
                np.asarray(env.physics.data.qpos, dtype=np.float64),
                [np.pi, 0.0],
                atol=1e-9,
            )
        finally:
            env.close()

    def test_wrapper_registration_reports_no_curriculum_to_drive(self):
        env = DMCContinuousEnv(
            domain_name="acrobot",
            task_name="swingup-v6-uniform",
            seed=0,
            raw_state_obs=True,
            time_sampling="uniform",
            dt=0.01,
            physics_dt=0.002,
            episode_duration=0.1,
        )
        self.addCleanup(env.close)
        self.assertFalse(env.has_curriculum)
        obs, info = env.reset()
        self.assertEqual(obs.shape, (4,))
        self.assertIn("acrobot_angle_cost", info)
        # The no-op setter still has to be safe to call on every training env.
        env.set_curriculum_fraction(0.5)
        _, reward, terminated, _, _ = env.step(np.array([1.0], dtype=np.float32))
        self.assertFalse(terminated)
        self.assertLess(reward, 0.0)

    def test_both_v6_arms_are_configured_for_every_shared_mode(self):
        import csv

        with open("benchmarks/hyperparams/ct_sac.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        modes = {
            env_id: {
                r["mode"] for r in rows if r["env_id"] == env_id
            }
            for env_id in ("acrobot-swingup-v6", "acrobot-swingup-v6-uniform")
        }
        self.assertEqual(modes["acrobot-swingup-v6"], modes["acrobot-swingup-v6-uniform"])
        self.assertIn("final_mf", modes["acrobot-swingup-v6"])
        self.assertIn("final_oracle_rollout", modes["acrobot-swingup-v6"])


if __name__ == "__main__":
    unittest.main()
