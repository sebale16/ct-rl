import unittest

import numpy as np

try:
    from dm_control.suite import cartpole

    from environment.double_cartpole_v2 import (
        CartpoleTwoPolesV2,
        DEFAULT_BRAKE_TIP_HEIGHT,
        DEFAULT_BRAKE_TIP_SPEED,
        DEFAULT_DESCENT_TIP_HEIGHTS,
        STABILIZATION_POINT,
        STRICT_CAPTURE_DISTANCE,
        STRICT_CAPTURE_SPEED,
        two_poles_v2,
    )

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

    def test_default_ladder_is_braking_then_rest_descent(self):
        task = CartpoleTwoPolesV2(random=0)
        levels = task.curriculum_levels
        self.assertEqual(len(levels), 1 + len(DEFAULT_DESCENT_TIP_HEIGHTS))
        self.assertEqual(
            (levels[0].tip_height, levels[0].incoming_tip_speed),
            (DEFAULT_BRAKE_TIP_HEIGHT, DEFAULT_BRAKE_TIP_SPEED),
        )
        self.assertEqual(
            tuple(level.tip_height for level in levels[1:]),
            DEFAULT_DESCENT_TIP_HEIGHTS,
        )
        self.assertTrue(
            all(level.incoming_tip_speed == 0.0 for level in levels[1:])
        )

    def test_braking_reset_has_exact_tip_state_and_incoming_direction(self):
        env = self._env()
        env.reset()
        physics, task = env.physics, env.task
        terms = task.curriculum_terms(physics)
        tip_x, tip_z, tip_vx, tip_vz = task._tip_kinematics(physics)

        self.assertAlmostEqual(
            terms["tip_height"], DEFAULT_BRAKE_TIP_HEIGHT, places=12
        )
        self.assertAlmostEqual(
            terms["tip_speed"], DEFAULT_BRAKE_TIP_SPEED, places=12
        )
        target = np.asarray(STABILIZATION_POINT)
        tip = np.asarray([tip_x, tip_z])
        velocity = np.asarray([tip_vx, tip_vz])
        self.assertGreater(float(velocity @ (target - tip)), 0.0)
        self.assertEqual(float(physics.data.qpos[0]), 0.0)
        self.assertEqual(float(physics.data.qpos[2]), 0.0)
        self.assertEqual(float(physics.data.qvel[0]), 0.0)
        self.assertEqual(float(physics.data.qvel[2]), 0.0)

    def test_rest_descent_ends_at_exact_hanging_state(self):
        env = self._env()
        for stage, expected_height in enumerate(
            DEFAULT_DESCENT_TIP_HEIGHTS, start=1
        ):
            env.task.set_curriculum_stage(stage)
            env.reset()
            terms = env.task.curriculum_terms(env.physics)
            self.assertAlmostEqual(
                terms["tip_height"], expected_height, places=12
            )
            np.testing.assert_array_equal(env.physics.data.qvel, np.zeros(3))
            self.assertEqual(float(env.physics.data.qpos[0]), 0.0)
            self.assertEqual(float(env.physics.data.qpos[2]), 0.0)

        self.assertTrue(env.task.curriculum_complete)
        np.testing.assert_allclose(env.physics.data.qpos, [0.0, np.pi, 0.0])

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
