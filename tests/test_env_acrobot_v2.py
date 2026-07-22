import unittest

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
        V41_ENERGY_OVERSHOOT_MARGIN,
        swingup_v3,
        swingup_v4,
        swingup_v41,
        swingup_v5,
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
        }
        self.assertTrue(v4_keys.issubset(reset_info))
        self.assertAlmostEqual(reset_info["acrobot_energy_norm"], 0.0, places=6)
        self.assertAlmostEqual(reset_info["acrobot_slow_gate"], 1.0, places=6)
        self.assertAlmostEqual(reset_info["acrobot_hold"], 0.0, places=6)
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
            **kwargs, energy_overshoot_margin=V41_ENERGY_OVERSHOOT_MARGIN
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

    def test_rewards_identical_at_or_below_unity_energy(self):
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
                self.assertAlmostEqual(t41["reward"], t4["reward"], places=12)

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

    def test_goal_and_slow_pass_unchanged(self):
        self._set_physics_state((0.0, 0.0))
        self.assertAlmostEqual(
            self.v41.reward_terms(self.physics)["reward"], 1.0, places=6
        )
        self._set_physics_state((0.08, 0.05), qvel=(0.5, 0.4))
        t4 = self.v4.reward_terms(self.physics)
        t41 = self.v41.reward_terms(self.physics)
        self.assertGreater(t41["reward"], 0.85)
        # The slow pass sits barely above Ẽ=1, so v4.1 trims it only mildly.
        self.assertLess(t4["reward"] - t41["reward"], 0.05)

    def test_invalid_overshoot_margin_rejected(self):
        for margin in (0.0, -0.25, float("nan"), float("inf")):
            with self.subTest(margin=margin):
                with self.assertRaises(ValueError):
                    BalanceV4(random=0, energy_overshoot_margin=margin)

    def test_factory_and_wrapper_registration(self):
        env = swingup_v41(
            time_limit=0.1,
            random=19,
            environment_kwargs={"flat_observation": True},
            angle_noise=0.0,
            velocity_noise=0.0,
        )
        try:
            env.reset()
            self.assertIsInstance(env.task, BalanceV4)
            self.assertEqual(
                env.task.energy_overshoot_margin, V41_ENERGY_OVERSHOOT_MARGIN
            )
            np.testing.assert_array_equal(env.physics.data.qpos, [np.pi, 0.0])
        finally:
            env.close()

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
        self.assertIn("acrobot_energy_norm", info)


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
        }
        self.assertTrue(v3_family_keys.issubset(info))
        self.assertTrue(v4_only_keys.isdisjoint(info))
        self.assertEqual(info["acrobot_progress"], 0.0)
        self.assertEqual(info["acrobot_gym_height_success"], 0.0)


if __name__ == "__main__":
    unittest.main()
