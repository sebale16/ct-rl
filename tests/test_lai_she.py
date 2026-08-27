"""Tests for the Lai--She 2009 unified WCLF Acrobot controller."""

import os
import unittest

os.environ.setdefault("MUJOCO_GL", "disable")

import numpy as np

from controllers import lai_she as ls
from environment.acrobot_wclf import swingup_wclf
from environment.dmc import DMCContinuousEnv


class TestPublishedParameters(unittest.TestCase):
    def test_table_ii_acrobot(self):
        params = ls.PAPER_PARAMS
        self.assertEqual(params.m1, 1.0)
        self.assertEqual(params.m2, 1.0)
        self.assertEqual(params.i1, 8.33e-2)
        self.assertEqual(params.i2, 0.33)
        self.assertEqual(params.l1, 1.0)
        self.assertEqual(params.l2, 2.0)
        self.assertEqual(params.lc1, 0.5)
        self.assertEqual(params.lc2, 1.0)
        self.assertAlmostEqual(params.energy_top, 24.5, places=12)

    def test_published_acrobot_design(self):
        design = ls.Design()
        self.assertEqual(design.alpha1, 0.5)
        self.assertEqual(design.alpha2, 30.0)
        self.assertEqual(design.eta, 25.0)
        self.assertEqual(design.gamma0, 1.6)
        self.assertEqual(design.energy_epsilon, 0.5)
        self.assertEqual(design.energy_top, 24.5)
        self.assertAlmostEqual(design.angle1_tolerance, np.pi / 6.0)
        self.assertAlmostEqual(design.angle2_tolerance, np.pi / 6.0)
        self.assertEqual(design.velocity1_weight, 1e-3)
        self.assertEqual(design.velocity2_weight, 1e-3)
        self.assertEqual(design.velocity_tolerance, 1e3)
        self.assertEqual(design.energy_tolerance, 1.0)
        self.assertEqual(design.lqr_q, (1.0, 1.0, 1.0, 1.0))
        self.assertEqual(design.lqr_r, 0.5)

    def test_equation_75_gain_is_used_verbatim(self):
        controller = ls.LaiSheController()
        np.testing.assert_allclose(
            controller.lqr_gain,
            [[-260.559, -104.448, -112.604, -52.944]],
            atol=0.0,
        )
        # Rebuilding the CARE from rounded Table-II values remains very close.
        self.assertLess(
            float(np.max(np.abs(
                controller.recomputed_lqr_gain - controller.lqr_gain
            ))),
            0.21,
        )


class TestPaperCoordinatePlant(unittest.TestCase):
    def setUp(self):
        self.env = swingup_wclf(torque_interface=50.0)
        self.env.reset()
        self.physics = self.env.physics
        self.params = ls.AcrobotParams.from_physics(self.physics)

    def test_model_recovers_table_ii(self):
        expected = ls.PAPER_PARAMS
        for name in ("m1", "m2", "i1", "i2", "l1", "l2", "lc1", "lc2"):
            self.assertAlmostEqual(getattr(self.params, name), getattr(expected, name))

    def test_qpos_is_directly_in_paper_coordinates(self):
        for shoulder, height in ((0.0, 3.0), (np.pi / 2.0, 0.0), (np.pi, -3.0)):
            self.physics.data.qpos[:] = [shoulder, 0.0]
            self.physics.data.qvel[:] = 0.0
            self.physics.forward()
            tip = np.asarray(self.physics.named.data.site_xpos["tip"])
            self.assertAlmostEqual(float(tip[2]), height, places=10)

    def test_analytic_dynamics_match_mujoco(self):
        rng = np.random.RandomState(0)
        worst = 0.0
        for _ in range(100):
            state = np.concatenate(
                [rng.uniform(-np.pi, np.pi, 2), rng.uniform(-6.0, 6.0, 2)]
            )
            torque = rng.uniform(-40.0, 40.0)
            self.physics.data.qpos[:] = state[:2]
            self.physics.data.qvel[:] = state[2:]
            self.physics.data.ctrl[:] = torque / self.params.gear
            self.physics.forward()
            actual = np.asarray(self.physics.data.qacc).copy()
            drift, input_vector = self.params.drift_and_input(state)
            expected = drift[2:] + input_vector[2:] * torque
            worst = max(worst, float(np.max(np.abs(actual - expected))))
        self.assertLess(worst, 1e-12)

    def test_horizontal_frame_adapter_round_trips(self):
        rng = np.random.RandomState(1)
        for state in rng.normal(size=(100, 4)):
            np.testing.assert_allclose(ls.paper_to_xk(ls.xk_to_paper(state)), state)


class TestWCLFControlLaw(unittest.TestCase):
    def setUp(self):
        self.params = ls.PAPER_PARAMS
        self.controller = ls.LaiSheController(self.params)

    def test_beta_derivative_matches_finite_difference(self):
        for x2 in np.linspace(-np.pi, np.pi, 21):
            state = np.array([0.3, x2, -0.7, 1.2])
            step = 1e-6
            forward = state.copy()
            backward = state.copy()
            forward[1] += step * state[3]
            backward[1] -= step * state[3]
            numerical = (
                self.controller.beta(forward) - self.controller.beta(backward)
            ) / (2.0 * step)
            self.assertAlmostEqual(
                self.controller.beta_dot(state), numerical, places=7
            )

    def test_denominator_reduces_to_alpha1_ex_plus_eta(self):
        rng = np.random.RandomState(2)
        minimum = np.inf
        for _ in range(500):
            state = np.concatenate(
                [rng.uniform(-np.pi, np.pi, 2), rng.uniform(-8.0, 8.0, 2)]
            )
            energy_error = self.params.energy(state) - 24.5
            actual = (
                self.controller.design.alpha1 * energy_error
                + self.controller.beta(state)
                * self.params.elbow_input_gain(state[1])
            )
            expected = self.controller.design.alpha1 * energy_error + 25.0
            self.assertAlmostEqual(actual, expected, places=12)
            minimum = min(minimum, actual)
        self.assertGreater(minimum, 0.0)
        hanging = np.array([np.pi, 0.0, 0.0, 0.0])
        hanging_denominator = (
            0.5 * (self.params.energy(hanging) - 24.5) + 25.0
        )
        self.assertAlmostEqual(hanging_denominator, 0.5)

    def test_equation_26_wclf_derivative(self):
        rng = np.random.RandomState(3)
        for _ in range(100):
            state = np.concatenate(
                [rng.uniform(-2.5, 2.5, 2), rng.uniform(-5.0, 5.0, 2)]
            )
            torque = self.controller.swingup_torque(state)
            drift, input_vector = self.params.drift_and_input(state)
            closed_loop = drift + input_vector * torque
            step = 1e-6
            numerical = (
                self.controller.wclf(state + step * closed_loop)
                - self.controller.wclf(state - step * closed_loop)
            ) / (2.0 * step)
            expected = -self.controller.gamma(state) * state[3] ** 2
            self.assertAlmostEqual(numerical, expected, delta=2e-5)

    def test_attractive_area_matches_equations_17_18(self):
        controller = self.controller
        self.assertTrue(controller.in_attractive_area(np.zeros(4)))
        self.assertFalse(
            controller.in_attractive_area(np.array([0.6, 0.0, 0.0, 0.0]))
        )
        self.assertFalse(
            controller.in_attractive_area(np.array([0.3, 0.3, 0.0, 0.0]))
        )

    def test_lqr_linear_closed_loop_is_stable(self):
        eigenvalues = np.linalg.eigvals(
            self.controller.a - self.controller.b @ self.controller.lqr_gain
        )
        self.assertTrue(np.all(eigenvalues.real < 0.0), eigenvalues)


class TestPublishedStyleRollout(unittest.TestCase):
    def test_calibrated_release_switches_and_balances_without_saturation(self):
        env = DMCContinuousEnv(
            "acrobot",
            "swingup-wclf",
            raw_state_obs=True,
            dt=0.001,
            physics_dt=0.001,
            max_steps=12_001,
            episode_duration=12.0,
            task_kwargs=dict(
                initial_perturbation=0.2,
                torque_interface=50.0,
            ),
        )
        obs, _ = env.reset()
        controller = ls.LaiSheController(
            ls.AcrobotParams.from_physics(env._env.physics)
        )
        for _ in range(12_000):
            obs = env.step_dt(controller(obs))[4]
        self.assertIsNotNone(controller.switch_step)
        self.assertAlmostEqual(controller.switch_step * 0.001, 7.49, delta=0.15)
        self.assertEqual(controller.saturated_steps, 0)
        final = np.asarray(obs, dtype=np.float64)
        error = np.array([
            float(ls.wrap(final[0])),
            float(ls.wrap(final[1])),
            final[2],
            final[3],
        ])
        self.assertLess(float(np.linalg.norm(error)), 0.01)


if __name__ == "__main__":
    unittest.main()
