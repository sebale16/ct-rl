"""Tests for the Lai--She nonsmooth-Lyapunov Acrobot controller."""

import os
import unittest

os.environ.setdefault("MUJOCO_GL", "disable")

import mujoco
import numpy as np

from controllers import lai_she as ls
from environment.acrobot_xk import swingup_xk


class TestPublishedParameters(unittest.TestCase):
    def test_section_iv_plant_values(self):
        params = ls.PAPER_PARAMS
        self.assertEqual(params.m1, 1.0)
        self.assertEqual(params.m2, 1.0)
        self.assertEqual(params.i1, 0.083)
        self.assertEqual(params.i2, 0.33)
        self.assertEqual(params.l1, 1.0)
        self.assertEqual(params.l2, 2.0)
        self.assertEqual(params.lc1, 0.5)
        self.assertEqual(params.lc2, 1.0)
        self.assertAlmostEqual(params.a1, 1.333, places=12)
        self.assertAlmostEqual(params.a2, 1.33, places=12)
        self.assertAlmostEqual(params.a3, 1.0, places=12)
        self.assertAlmostEqual(params.b1, 14.7, places=12)
        self.assertAlmostEqual(params.b2, 9.8, places=12)
        self.assertAlmostEqual(params.energy_top, 24.5, places=12)

    def test_section_iv_controller_values(self):
        design = ls.Design()
        self.assertAlmostEqual(design.beta1, np.pi / 6.0)
        self.assertAlmostEqual(design.beta2, np.pi / 6.0)
        self.assertEqual(design.energy_tolerance, 1.2)
        self.assertEqual(design.kp1, 1.0)
        self.assertEqual(design.kd1, 1.0)
        self.assertEqual(design.ke1, 0.2)
        self.assertEqual(design.lambda1, 38.0)
        self.assertEqual(design.phi1, 10.0)
        self.assertEqual(design.zeta, -2.0)
        self.assertEqual(design.kp2, 1.0)
        self.assertEqual(design.kd2, 1.0)
        self.assertEqual(design.phi2, 5.0)
        self.assertEqual(design.lambda_alpha, 0.5)


class TestAngleTransformation(unittest.TestCase):
    def test_named_landmarks(self):
        # Horizontal-frame upright/hanging become x1=0/pi, respectively.
        upright = ls.xk_to_paper(np.array([np.pi / 2, 0.0, 0.0, 0.0]))
        hanging = ls.xk_to_paper(np.array([-np.pi / 2, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(upright, np.zeros(4), atol=1e-15)
        np.testing.assert_allclose(hanging, [np.pi, 0.0, 0.0, 0.0], atol=1e-15)

    def test_round_trip(self):
        rng = np.random.RandomState(0)
        for state in rng.normal(size=(100, 4)):
            np.testing.assert_allclose(ls.paper_to_xk(ls.xk_to_paper(state)), state)

    def test_transformed_dynamics_match_mujoco(self):
        env = swingup_xk(torque_limit=80.0)
        env.reset()
        physics = env.physics
        params = ls.AcrobotParams.from_physics(physics)
        rng = np.random.RandomState(1)
        worst = 0.0
        for _ in range(100):
            state = np.concatenate(
                [rng.uniform(-np.pi, np.pi, 2), rng.uniform(-6.0, 6.0, 2)]
            )
            torque_paper = rng.uniform(-40.0, 40.0)
            state_xk = ls.paper_to_xk(state)
            physics.data.qpos[:] = state_xk[:2]
            physics.data.qvel[:] = state_xk[2:]
            # The coordinate reflection requires tau_xk = -tau_paper.
            physics.data.ctrl[:] = -torque_paper / params.gear
            physics.forward()
            actual_acceleration = -np.asarray(physics.data.qacc).copy()
            drift, input_vector = params.drift_and_input(state)
            expected_acceleration = drift[2:] + input_vector[2:] * torque_paper
            worst = max(
                worst,
                float(np.max(np.abs(actual_acceleration - expected_acceleration))),
            )
        self.assertLess(worst, 2e-12)

    def test_paper_torque_is_sign_flipped_at_plant_boundary(self):
        params = ls.AcrobotParams(gear=100.0)
        controller = ls.LaiSheController(params, frame="xk")
        obs = ls.paper_to_xk(np.array([2.0, 0.4, -0.2, 0.3]))
        paper_torque = controller.torque_c1(ls.xk_to_paper(obs))
        action = controller(obs)
        self.assertAlmostEqual(float(action[0]) * params.gear, -paper_torque)


class TestControlLaws(unittest.TestCase):
    def setUp(self):
        self.params = ls.PAPER_PARAMS
        self.design = ls.Design()
        self.controller = ls.LaiSheController(
            self.params, self.design, frame="paper", torque_limit=80.0
        )

    def _derivative(self, state, torque):
        drift, input_vector = self.params.drift_and_input(state)
        return drift + input_vector * torque

    def test_c1_has_equation_21_lyapunov_derivative(self):
        rng = np.random.RandomState(2)
        for _ in range(100):
            state = rng.uniform([-2.5, -2.0, -5.0, -5.0],
                                [2.5, 2.0, 5.0, 5.0])
            f_eta, b_eta, energy_error = self.controller._terms(state)
            denominator = self.design.kd1 * b_eta + self.design.ke1 * energy_error
            if abs(denominator) < 0.1:
                continue
            torque = self.controller.torque_c1(state)
            derivative = self._derivative(state, torque)
            energy_rate = state[3] * torque
            actual = (
                self.design.kp1 * state[1] * state[3]
                + self.design.kd1 * state[3] * derivative[3]
                + self.design.ke1 * energy_error * energy_rate
            )
            expected = -self.design.lambda1 * state[3] * np.clip(
                state[3] / self.design.phi1, -1.0, 1.0
            )
            self.assertAlmostEqual(actual, expected, places=9)

    def test_c2_has_equation_28_lyapunov_derivative(self):
        rng = np.random.RandomState(3)
        for _ in range(100):
            state = rng.uniform([-2.5, -2.0, -5.0, -5.0],
                                [2.5, 2.0, 5.0, 5.0])
            torque = self.controller.torque_c2(state)
            derivative = self._derivative(state, torque)
            actual = state[3] * (
                self.design.kp2 * state[1] + self.design.kd2 * derivative[3]
            )
            _, b_eta, _ = self.controller._terms(state)
            lambda2 = self.design.lambda_alpha * (
                1.0 + self.controller.last_fuzzy_adjustment
            )
            expected = (
                -self.design.kd2 * lambda2 * b_eta * state[3]
                * np.clip(state[3] / self.design.phi2, -1.0, 1.0)
            )
            self.assertAlmostEqual(actual, expected, places=9)
            self.assertLessEqual(actual, 1e-12)

    def test_table_i_corner_rules(self):
        params = self.params
        design = self.design
        energy_scale = design.fuzzy_energy_scale or params.energy_span
        power_scale = design.fuzzy_power_scale
        self.assertAlmostEqual(
            ls.fuzzy_adjustment(-energy_scale, -power_scale, params, design),
            -1.0 + 1e-6,
        )
        self.assertAlmostEqual(
            ls.fuzzy_adjustment(0.0, 0.0, params, design), 0.0
        )
        self.assertAlmostEqual(
            ls.fuzzy_adjustment(energy_scale, power_scale, params, design),
            1.0 - 1e-6,
        )

    def test_lqr_closed_loop_is_asymptotically_stable(self):
        eigenvalues = np.linalg.eigvals(
            self.controller.a - self.controller.b @ self.controller.lqr_gain
        )
        self.assertTrue(np.all(eigenvalues.real < 0.0), eigenvalues)

    def test_attractive_area_uses_both_absolute_link_angles(self):
        controller = self.controller
        controller.design = ls.Design(energy_tolerance=1e6)
        self.assertTrue(controller.in_attractive_area(np.zeros(4)))
        self.assertFalse(
            controller.in_attractive_area(
                np.array([0.75 * controller.design.beta1, 2.0 * controller.design.beta2,
                          0.0, 0.0])
            )
        )

    def test_switches_are_one_way(self):
        controller = self.controller
        near_singularity = np.array([2.0, 0.2, 1.0, 0.1])
        # Directly exercise the state machine with a permissive threshold; the
        # control-law identities above test the equations independently.
        controller.design = ls.Design(zeta=-1e6, energy_tolerance=1e6)
        controller._maybe_switch(near_singularity)
        self.assertEqual(controller.stage, controller.STAGE_2)
        controller._maybe_switch(np.zeros(4))
        self.assertEqual(controller.stage, controller.STAGE_3)
        controller._maybe_switch(np.array([np.pi, 0.0, 0.0, 0.0]))
        self.assertEqual(controller.stage, controller.STAGE_3)


if __name__ == "__main__":
    unittest.main()
