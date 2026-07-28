import unittest

import numpy as np

try:
    from dm_control.suite import cartpole

    from environment.double_cartpole_v2 import (
        CartpoleTwoPolesV2,
        DEFAULT_DESCENT_TIP_HEIGHTS,
        STABILIZATION_POINT,
        STRICT_CAPTURE_DISTANCE,
        STRICT_CAPTURE_SPEED,
        TIP_HEIGHT_BOUNDS,
        two_poles_v2,
    )
    from environment.tip_curriculum import INITIAL_TIP_HEIGHT_NORM

    HAVE_DMC = True
except Exception:  # pragma: no cover - exercised only without dm_control
    HAVE_DMC = False


@unittest.skipUnless(HAVE_DMC, "dm_control / double CartPole v2 unavailable")
class TestDoubleCartpoleTipCurriculum(unittest.TestCase):
    def _env(self, **kwargs):
        env = two_poles_v2(random=0, time_limit=0.1, **kwargs)
        self.addCleanup(env.close)
        return env

    @staticmethod
    def _physics():
        return cartpole.Physics.from_xml_string(
            *cartpole.get_model_and_assets(num_poles=2)
        )

    def test_factory_is_serial_double_cartpole(self):
        env = self._env()
        self.assertEqual(
            (env.physics.model.nq, env.physics.model.nv, env.physics.model.nu),
            (3, 3, 1),
        )
        self.assertEqual(np.asarray(env.physics.pole_angle_cosine()).shape, (2,))

    def test_default_ladder_is_near_upright_then_rest_descent(self):
        task = CartpoleTwoPolesV2(random=0)
        levels = task.curriculum_levels
        expected_initial_height = (
            TIP_HEIGHT_BOUNDS[0]
            + INITIAL_TIP_HEIGHT_NORM
            * (TIP_HEIGHT_BOUNDS[1] - TIP_HEIGHT_BOUNDS[0])
        )
        self.assertEqual(len(levels), 1 + len(DEFAULT_DESCENT_TIP_HEIGHTS))
        self.assertEqual(
            (levels[0].tip_height, levels[0].incoming_tip_speed),
            (expected_initial_height, 0.0),
        )
        self.assertEqual(
            tuple(level.tip_height for level in levels[1:]),
            DEFAULT_DESCENT_TIP_HEIGHTS,
        )
        self.assertTrue(all(level.incoming_tip_speed == 0.0 for level in levels))

    def test_diagnostics_track_selected_height_velocity_and_potential_level(self):
        task = CartpoleTwoPolesV2(random=0)

        for stage, level in enumerate(task.curriculum_levels):
            with self.subTest(stage=stage):
                task.set_curriculum_stage(stage)
                diagnostics = task.curriculum_diagnostics()
                height_norm = (level.tip_height + 1.0) / 4.0
                self.assertEqual(diagnostics["curriculum_stage"], float(stage))
                self.assertAlmostEqual(
                    diagnostics["curriculum_progress"],
                    stage / (task.num_curriculum_stages - 1),
                )
                self.assertEqual(
                    diagnostics["curriculum_start_tip_height"],
                    level.tip_height,
                )
                self.assertEqual(
                    diagnostics["curriculum_start_tip_speed"],
                    level.incoming_tip_speed,
                )
                self.assertAlmostEqual(
                    diagnostics["curriculum_start_tip_height_norm"],
                    height_norm,
                )
                self.assertAlmostEqual(
                    diagnostics[
                        "curriculum_start_potential_energy_norm"
                    ],
                    height_norm,
                )

    def test_first_reset_is_near_upright_at_rest(self):
        env = self._env()
        expected_height = (
            TIP_HEIGHT_BOUNDS[0]
            + INITIAL_TIP_HEIGHT_NORM
            * (TIP_HEIGHT_BOUNDS[1] - TIP_HEIGHT_BOUNDS[0])
        )
        for _ in range(64):
            env.reset()
            physics, task = env.physics, env.task
            terms = task.curriculum_terms(physics)
            np.testing.assert_array_equal(physics.data.qvel, np.zeros(3))
            self.assertEqual(float(physics.data.qpos[0]), 0.0)
            tip_x, tip_z, tip_vx, tip_vz = task._tip_kinematics(physics)
            self.assertAlmostEqual(tip_z, expected_height, places=12)
            self.assertEqual(tip_vx, 0.0)
            self.assertEqual(tip_vz, 0.0)
            expected_distance = float(
                np.hypot(
                    tip_x - STABILIZATION_POINT[0],
                    tip_z - STABILIZATION_POINT[1],
                )
            )
            # Folding shortens the chain toward the goal, so the level's
            # narrow fold is what keeps every draw a recovery.
            self.assertGreaterEqual(expected_distance, STRICT_CAPTURE_DISTANCE)
            self.assertAlmostEqual(
                terms["tip_height"], expected_height, places=12
            )
            self.assertAlmostEqual(
                terms["tip_distance"], expected_distance, places=12
            )
            self.assertEqual(terms["tip_speed"], 0.0)
            self.assertEqual(terms["strict_capture"], 0.0)
            self.assertEqual(terms["success"], 0.0)

    def test_resets_fold_the_second_hinge_within_the_level_spread(self):
        env = self._env()
        task = env.task

        for stage, level in enumerate(task.curriculum_levels):
            with self.subTest(stage=stage):
                task.set_curriculum_stage(stage)
                folds = []
                for _ in range(64):
                    env.reset()
                    self.assertEqual(float(env.physics.data.qpos[0]), 0.0)
                    folds.append(float(env.physics.data.qpos[2]))
                folds = np.asarray(folds)

                self.assertGreater(level.elbow_spread, 0.0)
                self.assertLessEqual(np.abs(folds).max(), level.elbow_spread)
                self.assertGreater(
                    np.abs(folds).max(), 0.5 * level.elbow_spread
                )
                self.assertTrue((folds > 0.0).any() and (folds < 0.0).any())

    def test_rest_descent_ends_at_hanging_at_rest(self):
        env = self._env()
        for stage, expected_height in enumerate(
            DEFAULT_DESCENT_TIP_HEIGHTS, start=1
        ):
            env.task.set_curriculum_stage(stage)
            env.reset()
            terms = env.task.curriculum_terms(env.physics)
            if np.isclose(expected_height, TIP_HEIGHT_BOUNDS[0]):
                # A folded chain cannot reach the lowest tip: the closest pose
                # in that fold splays symmetrically about vertical.
                self.assertGreaterEqual(terms["tip_height"], expected_height)
                self.assertLess(terms["tip_height"] - expected_height, 0.07)
            else:
                self.assertAlmostEqual(
                    terms["tip_height"], expected_height, places=12
                )
            np.testing.assert_array_equal(env.physics.data.qvel, np.zeros(3))
            self.assertEqual(float(env.physics.data.qpos[0]), 0.0)

        self.assertTrue(env.task.curriculum_complete)

    def test_unfolded_final_stage_is_the_exact_hanging_state(self):
        env = self._env(elbow_spread=0.0)
        env.task.set_curriculum_stage(env.task.num_curriculum_stages - 1)
        env.reset()

        np.testing.assert_allclose(env.physics.data.qpos, [0.0, np.pi, 0.0])
        np.testing.assert_array_equal(env.physics.data.qvel, np.zeros(3))

    def test_curriculum_disabled_is_exact_hanging_at_rest(self):
        env = self._env(curriculum=False)
        for _ in range(3):
            env.reset()
            np.testing.assert_allclose(env.physics.data.qpos, [0.0, np.pi, 0.0])
            np.testing.assert_array_equal(env.physics.data.qvel, np.zeros(3))
        self.assertTrue(env.task.curriculum_complete)
        terms = env.task.curriculum_terms(env.physics)
        self.assertEqual(terms["curriculum_enabled"], 0.0)
        self.assertEqual(terms["curriculum_start_tip_height"], -1.0)
        self.assertEqual(terms["curriculum_start_tip_speed"], 0.0)

    def test_tip_kinematics_uses_relative_second_hinge(self):
        physics = self._physics()
        task = CartpoleTwoPolesV2(random=0)
        qpos = np.asarray([0.3, 0.7, -0.2])
        qvel = np.asarray([0.4, -1.2, 0.5])
        physics.data.qpos[:] = qpos
        physics.data.qvel[:] = qvel
        physics.forward()

        phi = qpos[1] + qpos[2]
        phi_dot = qvel[1] + qvel[2]
        expected = (
            qpos[0] + np.sin(qpos[1]) + np.sin(phi),
            1.0 + np.cos(qpos[1]) + np.cos(phi),
            qvel[0] + np.cos(qpos[1]) * qvel[1] + np.cos(phi) * phi_dot,
            -np.sin(qpos[1]) * qvel[1] - np.sin(phi) * phi_dot,
        )
        np.testing.assert_allclose(task._tip_kinematics(physics), expected)

    def test_strict_capture_requires_distance_and_distal_speed(self):
        physics = self._physics()
        task = CartpoleTwoPolesV2(random=0)
        physics.data.qpos[:] = [0.0, 0.0, 0.0]

        physics.data.qvel[:] = [STRICT_CAPTURE_SPEED - 1e-6, 0.0, 0.0]
        physics.forward()
        self.assertEqual(task.curriculum_terms(physics)["strict_capture"], 1.0)

        physics.data.qvel[:] = [STRICT_CAPTURE_SPEED, 0.0, 0.0]
        physics.forward()
        self.assertEqual(task.curriculum_terms(physics)["strict_capture"], 0.0)

        physics.data.qpos[:] = [STRICT_CAPTURE_DISTANCE, 0.0, 0.0]
        physics.data.qvel[:] = 0.0
        physics.forward()
        self.assertEqual(task.curriculum_terms(physics)["strict_capture"], 0.0)

    def test_reseed_reproduces_both_mirror_sides(self):
        env = self._env()
        env.task.reseed(123)
        first = []
        for _ in range(12):
            env.reset()
            first.append(float(env.physics.data.qpos[1]))
        env.task.reseed(123)
        second = []
        for _ in range(12):
            env.reset()
            second.append(float(env.physics.data.qpos[1]))

        np.testing.assert_array_equal(first, second)
        self.assertTrue(any(theta < 0.0 for theta in first))
        self.assertTrue(any(theta > 0.0 for theta in first))

    def test_reward_is_exactly_stock_smooth_two_pole_reward(self):
        self.assertIs(CartpoleTwoPolesV2.get_reward, cartpole.Balance.get_reward)
        physics = self._physics()
        local = CartpoleTwoPolesV2(random=0)
        stock = cartpole.Balance(swing_up=True, sparse=False, random=0)

        for qpos, qvel, control_value in (
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0),
            ((0.2, 0.6, -0.4), (0.3, -1.7, 0.8), 0.4),
            ((-1.0, -2.5, 1.2), (-0.8, 4.0, -2.0), -1.0),
            ((0.0, np.pi, 0.0), (0.0, 0.0, 0.0), 0.0),
        ):
            physics.data.qpos[:] = qpos
            physics.data.qvel[:] = qvel
            physics.data.ctrl[:] = control_value
            physics.forward()
            self.assertEqual(
                float(local.get_reward(physics)),
                float(stock.get_reward(physics)),
            )


if __name__ == "__main__":
    unittest.main()
