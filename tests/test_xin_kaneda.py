"""Tests for the Xin-Kaneda swing-up controller, its plant, and its metrics.

The strongest checks are fixtures: both papers publish enough numbers to pin the
gain-condition module down exactly, and the 2007 paper publishes the
characteristic equation of the closed loop at the hanging equilibrium, which
pins down the whole control law plus the linearization.
"""

import os
import unittest

os.environ.setdefault("MUJOCO_GL", "disable")

import mujoco
import numpy as np

from dm_control.suite import acrobot

from controllers import xin_kaneda as xk
from environment.acrobot_xk import DEFAULT_TORQUE_LIMIT, swingup_xk
from environment.dmc import DMCContinuousEnv
from evaluations import acrobot_homoclinic_metrics as metrics


def _repo_params():
    physics = acrobot.Physics.from_xml_string(*acrobot.get_model_and_assets())
    return xk.AcrobotParams.from_physics(physics)


class TestPublishedConstants(unittest.TestCase):
    """Fixtures quoted in the two papers, on their own simulation plant."""

    def setUp(self):
        self.params = xk.PAPER_PARAMS

    def test_grouped_parameters_match_the_papers(self):
        # m1 = m2 = 1, l1 = 1, l2 = 2, lc1 = 0.5, lc2 = 1, I1 = 0.083, I2 = 0.33.
        self.assertAlmostEqual(self.params.a1, 1.333, places=12)
        self.assertAlmostEqual(self.params.a2, 1.330, places=12)
        self.assertAlmostEqual(self.params.a3, 1.000, places=12)
        self.assertAlmostEqual(self.params.b1, 14.7, places=12)
        self.assertAlmostEqual(self.params.b2, 9.8, places=12)
        self.assertAlmostEqual(self.params.energy_top, 24.5, places=12)

    def test_rho_star_matches_2002_simulation_section(self):
        self.assertAlmostEqual(xk.rho_star(self.params), 0.8578, places=4)

    def test_eta_star_matches_2002_simulation_section(self):
        self.assertAlmostEqual(xk.eta_star(self.params), 0.2879, places=4)

    def test_xi_star_is_two_for_every_plant(self):
        # eq. (66) of 2007: the supremum is the q2 -> 0 limit, and it is 2
        # regardless of beta, which is why kp_boundary is always 2 b1 b2.
        for params in (self.params, _repo_params()):
            self.assertAlmostEqual(xk.xi_star(params), 2.0, places=6)

    def test_alpha_matches_2002_simulation_section(self):
        # 2002 reports alpha0 = 1.4443, alpha1 = 1.1620, alpha2 = 0.8280,
        # so alpha = alpha1^2 + alpha2^2 for alpha0 > 1.
        self.assertAlmostEqual(xk.alpha(self.params), 1.1620**2 + 0.8280**2, places=4)

    def test_2002_theorem3_thresholds_at_half_gain(self):
        # "we choose kE = 0.5 ... kD = 15 > kE E_top / rho* = 14.2804 and
        # kP = 22 > eta* kE theta4 theta5 g^2 = 20.7390".
        k_e = 0.5
        theorem3_kd = k_e * self.params.energy_top / xk.rho_star(self.params)
        theorem3_kp = (
            xk.eta_star(self.params) * k_e * self.params.b1 * self.params.b2
        )
        self.assertAlmostEqual(theorem3_kd, 14.2804, places=4)
        self.assertAlmostEqual(theorem3_kp, 20.7390, places=4)

    def test_2002_theorem4_thresholds_are_the_doubled_and_boundary_forms(self):
        # Theorem 4 needs kD > 2 kE E_top / rho* and kP > max(eta*, xi*) kE b1 b2.
        k_e = 0.5
        self.assertAlmostEqual(
            k_e * xk.kd_min_theorem4(self.params), 28.5608, places=4
        )
        self.assertAlmostEqual(
            k_e * xk.kp_boundary(self.params), 144.06, places=2
        )

    def test_2007_conditions_match_the_simulation_section(self):
        # "Conditions (25) and (43) are kD > 35.741, kP > 61.141".
        self.assertAlmostEqual(xk.kd_min(self.params), 35.741, places=3)
        self.assertAlmostEqual(xk.kp_min(self.params), 61.141, places=3)

    def test_2007_conditions_are_weaker_than_2002(self):
        # The exact no-singularity bound beats the sufficient one, and letting
        # the spurious equilibria exist beats removing them.
        self.assertLess(xk.kd_min(self.params), xk.kd_min_theorem4(self.params))
        self.assertLess(xk.kp_min(self.params), xk.kp_boundary(self.params))
        # The closed-form (43) is a valid bound on the exact requirement (51).
        self.assertLess(xk.kp_min_exact(self.params), xk.kp_min(self.params))

    def test_hanging_characteristic_equation_matches_2007_equation_71(self):
        # s^4 + 0.036 kV s^3 - 3.375 s^2 + 0.190 kV s - 43.076 = 0
        # at the paper's kD = 35.8, kP = 61.2.
        k_v = 66.3
        gains = xk.Gains(k_v=k_v, k_d=35.8, k_p=61.2)
        coefficients = np.poly(xk.hanging_jacobian(self.params, gains))
        self.assertAlmostEqual(coefficients[0], 1.0, places=9)
        self.assertAlmostEqual(coefficients[1] / k_v, 0.036, places=3)
        self.assertAlmostEqual(coefficients[2], -3.375, places=3)
        self.assertAlmostEqual(coefficients[3] / k_v, 0.190, places=3)
        self.assertAlmostEqual(coefficients[4], -43.076, places=3)

    def test_hanging_is_unstable_with_three_right_half_plane_roots(self):
        # Proposition 5(i): 0 < kP < 2 b1 b2 gives one left-half-plane root and
        # three in the right half plane.  The paper's gains sit there.
        regime = xk.hanging_regime(
            self.params, xk.Gains(k_v=66.3, k_d=35.8, k_p=61.2)
        )
        self.assertEqual(regime["n_unstable"], 3)
        self.assertEqual(regime["regime"], "prop4_fast")
        self.assertAlmostEqual(regime["max_real"], 1.9348, places=3)


class TestPaperFrameModelAgreement(unittest.TestCase):
    """The plant is built in the paper's coordinates, so no map is involved."""

    def setUp(self):
        env = swingup_xk()
        env.reset()
        self.physics = env.physics
        self.params = xk.AcrobotParams.from_physics(self.physics)
        self.rng = np.random.RandomState(0)

    def _random_states(self, count=200):
        angles = self.rng.uniform(-np.pi, np.pi, size=(count, 2))
        rates = self.rng.uniform(-6.0, 6.0, size=(count, 2))
        return np.concatenate([angles, rates], axis=1)

    def _set(self, state):
        self.physics.data.qpos[:] = state[:2]
        self.physics.data.qvel[:] = state[2:]
        self.physics.forward()

    def test_qpos_is_the_papers_coordinates(self):
        # eq. 10: upright is q1 = pi/2, and hanging is -pi/2.
        for shoulder, height in ((0.0, 0.0), (0.5 * np.pi, 3.0), (-0.5 * np.pi, -3.0)):
            with self.subTest(shoulder=shoulder):
                self._set(np.array([shoulder, 0.0, 0.0, 0.0]))
                tip = np.asarray(self.physics.named.data.site_xpos["tip"])
                self.assertAlmostEqual(float(tip[2]), height, places=9)

    def test_mass_matrix_matches_without_a_map(self):
        worst = 0.0
        for state in self._random_states():
            self._set(state)
            dense = np.zeros((2, 2))
            mujoco.mj_fullM(self.physics.model.ptr, dense, self.physics.data.qM)
            worst = max(
                worst, np.max(np.abs(dense - self.params.mass_matrix(state[1])))
            )
        self.assertLess(worst, 1e-12)

    def test_bias_matches_without_a_sign_flip(self):
        worst = 0.0
        for state in self._random_states():
            self._set(state)
            bias = np.asarray(self.physics.data.qfrc_bias, dtype=np.float64)
            analytic = self.params.bias(state[:2], state[2:])
            worst = max(worst, np.max(np.abs(bias - analytic)))
        self.assertLess(worst, 1e-11)

    def test_energy_matches_without_an_offset(self):
        worst = 0.0
        for state in self._random_states():
            self._set(state)
            dense = np.zeros((2, 2))
            mujoco.mj_fullM(self.physics.model.ptr, dense, self.physics.data.qM)
            qvel = np.asarray(self.physics.data.qvel, dtype=np.float64)
            mujoco_energy = 0.5 * float(qvel @ dense @ qvel) - float(
                np.asarray(self.physics.model.body_mass)
                @ (
                    np.asarray(self.physics.data.xipos)
                    @ np.asarray(self.physics.model.opt.gravity)
                )
            )
            analytic = self.params.energy(state[:2], state[2:])
            worst = max(worst, abs(mujoco_energy - analytic))
        self.assertLess(worst, 1e-11)


class TestMappedFrameModelAgreement(unittest.TestCase):
    """The stock dm_control Acrobot still agrees, through obs_to_paper."""

    def setUp(self):
        self.physics = acrobot.Physics.from_xml_string(
            *acrobot.get_model_and_assets()
        )
        self.params = xk.AcrobotParams.from_physics(self.physics)
        self.rng = np.random.RandomState(0)

    def _random_states(self, count=200):
        angles = self.rng.uniform(-np.pi, np.pi, size=(count, 2))
        rates = self.rng.uniform(-8.0, 8.0, size=(count, 2))
        return np.concatenate([angles, rates], axis=1)

    def _set(self, state):
        self.physics.data.qpos[:] = state[:2]
        self.physics.data.qvel[:] = state[2:]
        self.physics.forward()

    def test_recovered_parameters(self):
        self.assertAlmostEqual(self.params.a3, 0.5, places=12)
        self.assertAlmostEqual(self.params.b1, 1.5 * 9.81, places=12)
        self.assertAlmostEqual(self.params.b2, 0.5 * 9.81, places=12)
        self.assertAlmostEqual(self.params.gear, 2.0, places=12)

    def test_mass_matrix_matches_mj_full_m(self):
        worst = 0.0
        for state in self._random_states():
            self._set(state)
            dense = np.zeros((2, 2))
            mujoco.mj_fullM(self.physics.model.ptr, dense, self.physics.data.qM)
            paper = xk.obs_to_paper(state)
            worst = max(
                worst, np.max(np.abs(dense - self.params.mass_matrix(paper[1])))
            )
        self.assertLess(worst, 1e-12)

    def test_bias_matches_qfrc_bias_under_the_frame_map(self):
        # The map negates both generalized coordinates, so the bias forces
        # negate too: qfrc_bias = -(H + G) evaluated in paper coordinates.
        worst = 0.0
        for state in self._random_states():
            self._set(state)
            bias = np.asarray(self.physics.data.qfrc_bias, dtype=np.float64)
            paper = xk.obs_to_paper(state)
            analytic = self.params.bias(paper[:2], paper[2:])
            worst = max(worst, np.max(np.abs(bias + analytic)))
        self.assertLess(worst, 1e-11)

    def test_energy_differs_from_mujoco_by_the_height_reference(self):
        offset = 2.0 * (self.params.b1 + self.params.b2)
        worst = 0.0
        for state in self._random_states():
            self._set(state)
            dense = np.zeros((2, 2))
            mujoco.mj_fullM(self.physics.model.ptr, dense, self.physics.data.qM)
            qvel = np.asarray(self.physics.data.qvel, dtype=np.float64)
            mujoco_energy = 0.5 * float(qvel @ dense @ qvel) - float(
                np.asarray(self.physics.model.body_mass)
                @ (
                    np.asarray(self.physics.data.xipos)
                    @ np.asarray(self.physics.model.opt.gravity)
                )
            )
            paper = xk.obs_to_paper(state)
            analytic = self.params.energy(paper[:2], paper[2:])
            worst = max(worst, abs(mujoco_energy - analytic - offset))
        self.assertLess(worst, 1e-11)

    def test_metrics_energy_error_agrees_with_the_paper_frame(self):
        # evaluations.acrobot_homoclinic_metrics computes E - E_r straight from
        # repo coordinates; it must equal the mapped analytic value.
        states = self._random_states(64)
        mapped = np.array([xk.obs_to_paper(s) for s in states])
        direct = metrics.energy_error(mapped, self.params)
        expected = np.array(
            [self.params.energy(m[:2], m[2:]) - self.params.energy_top for m in mapped]
        )
        np.testing.assert_allclose(direct, expected, atol=1e-12)


class TestPlantConstants(unittest.TestCase):
    """The built plant must reproduce the paper's own mechanical parameters."""

    def setUp(self):
        env = swingup_xk()
        env.reset()
        self.params = xk.AcrobotParams.from_physics(env.physics)

    def test_recovered_parameters_are_the_papers(self):
        for name, expected in (
            ("a1", 1.333), ("a2", 1.330), ("a3", 1.000),
            ("b1", 14.7), ("b2", 9.8), ("gravity", 9.8),
        ):
            with self.subTest(name=name):
                self.assertAlmostEqual(
                    getattr(self.params, name), expected, places=9
                )
        self.assertAlmostEqual(self.params.energy_top, 24.5, places=9)

    def test_thresholds(self):
        self.assertAlmostEqual(xk.kd_min(self.params), 35.7406, places=4)
        self.assertAlmostEqual(xk.kp_min(self.params), 61.1410, places=4)
        self.assertAlmostEqual(xk.kp_boundary(self.params), 288.12, places=2)
        self.assertAlmostEqual(xk.alpha(self.params), 2.035828, places=6)
        self.assertNotEqual(xk.alpha(self.params), 0.0)

    def test_homoclinic_speed_and_asymptotic_torque(self):
        self.assertAlmostEqual(xk.homoclinic_speed(self.params), 4.584377, places=6)
        self.assertAlmostEqual(
            xk.asymptotic_torque_bound(self.params), 2.442119, places=6
        )

    def test_proposition5_regime_switches_at_the_boundary(self):
        boundary = xk.kp_boundary(self.params)
        below = xk.hanging_regime(
            self.params, xk.Gains(k_v=66.3, k_d=35.8, k_p=0.5 * boundary)
        )
        above = xk.hanging_regime(
            self.params, xk.Gains(k_v=66.3, k_d=35.8, k_p=2.0 * boundary)
        )
        self.assertEqual(below["n_unstable"], 3)
        self.assertEqual(above["n_unstable"], 2)
        # The correctness-versus-speed trade the branch is about: removing the
        # spurious equilibria costs two orders of magnitude of escape rate.
        self.assertGreater(below["max_real"], 20.0 * above["max_real"])

    def test_admissibility_rejects_gains_below_either_floor(self):
        with self.assertRaises(ValueError):
            xk.assert_admissible(
                self.params, xk.Gains(k_v=66.3, k_d=1.0, k_p=61.2)
            )
        with self.assertRaises(ValueError):
            xk.assert_admissible(
                self.params, xk.Gains(k_v=66.3, k_d=35.8, k_p=1.0)
            )
        xk.assert_admissible(self.params, xk.Gains(k_v=66.3, k_d=35.8, k_p=61.2))


class TestLyapunovDescent(unittest.TestCase):
    """``Vdot = -k_V qdot2^2 <= 0`` on the conservative, unclipped plant."""

    def setUp(self):
        self.params = xk.PAPER_PARAMS
        self.gains = xk.Gains(k_v=66.3, k_d=35.8, k_p=61.2)

    def _integrate(self, state, steps, step, **kwargs):
        def drift(x):
            return xk.closed_loop(self.params, self.gains, x, **kwargs)[0]

        trace = [state.copy()]
        for _ in range(steps):
            k1 = drift(state)
            k2 = drift(state + 0.5 * step * k1)
            k3 = drift(state + 0.5 * step * k2)
            k4 = drift(state + step * k3)
            state = state + (step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            trace.append(state.copy())
        return np.array(trace)

    def test_lyapunov_is_non_increasing(self):
        start = np.array([-0.5 * np.pi + 0.17, 0.0, 0.0, 0.0])
        trace = self._integrate(start, steps=20000, step=1e-3)
        values = np.array(
            [xk.lyapunov(self.params, self.gains, x) for x in trace]
        )
        # Round-off only; the initial value is O(400) here.
        self.assertLess(np.max(np.diff(values)), 1e-9)
        self.assertLess(values[-1], values[0])

    def test_damping_breaks_the_descent(self):
        # The obstruction the conservative plant exists to avoid: with joint
        # damping the -(E - E_r) d |qdot|^2 term is positive below the top
        # energy, so V rises.
        start = np.array([-0.5 * np.pi + 0.17, 0.0, 0.0, 0.0])
        trace = self._integrate(start, steps=20000, step=1e-3, damping=0.05)
        values = np.array(
            [xk.lyapunov(self.params, self.gains, x) for x in trace]
        )
        self.assertGreater(np.max(np.diff(values)), 1e-6)

    def test_reproduces_the_2007_lqr_switch_time(self):
        # Section 7 of 2007: on their plant, with kD = 35.8, kP = 61.2,
        # kV = 66.3 and the initial condition [-1.4, 0, 0, 0], "the switch was
        # taken about t = 8 s" under the eq. 74 test at zeta = 0.04.  This
        # exercises the control law, the dynamics, the gain floors, and the
        # metric-6 switching function together against a published number.
        params = xk.PAPER_PARAMS
        gains = xk.Gains(k_v=66.3, k_d=35.8, k_p=61.2)
        # Their gains clear the floors these functions compute, barely.
        self.assertGreater(gains.k_d, xk.kd_min(params))
        self.assertGreater(gains.k_p, xk.kp_min(params))

        state = np.array([-1.4, 0.0, 0.0, 0.0])
        step = 1e-4
        switch_time = float("inf")
        for index in range(int(12.0 / step)):
            residual = float(metrics.lqr_residual(state[None, :])[0])
            if residual < metrics.LQR_SWITCH_THRESHOLD:
                switch_time = index * step
                break
            k1, _ = xk.closed_loop(params, gains, state)
            k2, _ = xk.closed_loop(params, gains, state + 0.5 * step * k1)
            k3, _ = xk.closed_loop(params, gains, state + 0.5 * step * k2)
            k4, _ = xk.closed_loop(params, gains, state + step * k3)
            state = state + (step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.assertAlmostEqual(switch_time, 7.776, places=2)

    def test_denominator_stays_away_from_zero(self):
        start = np.array([-0.5 * np.pi + 0.17, 0.0, 0.0, 0.0])
        trace = self._integrate(start, steps=20000, step=1e-3)
        smallest = np.inf
        for x in trace:
            mass = self.params.mass_matrix(x[1])
            det = mass[0, 0] * mass[1, 1] - mass[0, 1] * mass[1, 0]
            energy_error = self.params.energy(x[:2], x[2:]) - self.params.energy_top
            smallest = min(
                smallest, abs(self.gains.k_d * mass[0, 0] + energy_error * det)
            )
        self.assertGreater(smallest, 1.0)


class TestEnvironment(unittest.TestCase):
    """The plant honours its two deviations from the stock model."""

    def test_damping_and_torque_limit_reach_the_model(self):
        for damping, limit in ((0.0, 2.0), (0.05, 2.0), (0.0, 64.0)):
            with self.subTest(damping=damping, torque_limit=limit):
                env = swingup_xk(damping=damping, torque_limit=limit)
                model = env.physics.model
                np.testing.assert_allclose(
                    np.asarray(model.dof_damping), damping
                )
                self.assertAlmostEqual(
                    float(np.asarray(model.actuator_gear)[0, 0]), limit, places=12
                )

    def test_defaults_are_the_conservative_plant(self):
        env = swingup_xk()
        np.testing.assert_allclose(np.asarray(env.physics.model.dof_damping), 0.0)
        self.assertAlmostEqual(
            float(np.asarray(env.physics.model.actuator_gear)[0, 0]),
            DEFAULT_TORQUE_LIMIT,
            places=12,
        )

    def test_control_period_finer_than_the_physics_step_is_a_no_op(self):
        # The wrapper realizes a control period as nsub = max(1, round(dt /
        # physics_dt)) physics steps, so requesting a period below the model's
        # own timestep silently degrades to that timestep.  The evaluation CLI
        # defaults physics_dt to min(dt, model timestep) because of this; the
        # behaviour is pinned here so the default cannot quietly stop mattering.
        from evaluations.eval_acrobot_xk import Arm, MODEL_TIMESTEP, build_env

        arm = Arm(
            k_v=66.3,
            k_d=35.8,
            k_p=61.2,
            torque_limit=DEFAULT_TORQUE_LIMIT,
            damping=0.0,
            t_max=1.0,
            start="paper",
        )
        fine = 1e-4
        # Without the default, the realized step stays at the model timestep.
        coarse_env = build_env(arm, 0, fine, physics_dt=MODEL_TIMESTEP)
        coarse_env.reset(seed=0)
        _, _, _, _, _, next_t = coarse_env.step_dt(np.zeros(1))[:6]
        self.assertAlmostEqual(next_t, MODEL_TIMESTEP, places=12)
        # With it, the requested period is realized.
        fine_env = build_env(arm, 0, fine)
        self.assertAlmostEqual(fine_env.physics_dt, fine, places=12)
        fine_env.reset(seed=0)
        _, _, _, _, _, next_t = fine_env.step_dt(np.zeros(1))[:6]
        self.assertAlmostEqual(next_t, fine, places=12)

    def test_raw_state_observation_is_the_generalized_state(self):
        env = DMCContinuousEnv(
            "acrobot", "swingup-xk", seed=0, raw_state_obs=True, dt=0.01
        )
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape, (4,))
        np.testing.assert_allclose(
            obs[:2], np.asarray(env._env.physics.data.qpos), atol=1e-12
        )
        np.testing.assert_allclose(
            obs[2:], np.asarray(env._env.physics.data.qvel), atol=1e-12
        )

    def test_release_reset_is_the_shared_evaluation_distribution(self):
        # The protocol recorded in docs/reward_shaping_for_acrobot_swingup.md:
        # straight chain, released from rest, shoulder displaced from hanging by
        # a magnitude in [0.05, 0.5] with a random sign.
        from environment.acrobot_xk import HANGING_SHOULDER, RELEASE_ANGLE_RANGE

        low, high = RELEASE_ANGLE_RANGE
        displacements = []
        for seed in range(40):
            env = DMCContinuousEnv(
                "acrobot",
                "swingup-xk",
                seed=seed,
                raw_state_obs=True,
                dt=0.002,
                task_kwargs=dict(release_start=True),
            )
            obs, _ = env.reset(seed=seed)
            qpos = np.asarray(env._env.physics.data.qpos, dtype=np.float64)
            qvel = np.asarray(env._env.physics.data.qvel, dtype=np.float64)
            # Straight chain, at rest.
            self.assertAlmostEqual(float(qpos[1]), 0.0, places=15)
            np.testing.assert_allclose(qvel, 0.0, atol=1e-15)
            displacements.append(float(qpos[0]) - HANGING_SHOULDER)
        magnitudes = np.abs(np.asarray(displacements))
        self.assertTrue(np.all(magnitudes >= low - 1e-12))
        self.assertTrue(np.all(magnitudes <= high + 1e-12))
        # Both signs occur, so the distribution is not one-sided.
        self.assertTrue(np.any(np.asarray(displacements) > 0))
        self.assertTrue(np.any(np.asarray(displacements) < 0))
        # Repeatable: the same seed gives the same start.
        env = DMCContinuousEnv(
            "acrobot", "swingup-xk", seed=7, raw_state_obs=True, dt=0.002,
            task_kwargs=dict(release_start=True),
        )
        first, _ = env.reset(seed=7)
        second, _ = env.reset(seed=7)
        np.testing.assert_allclose(first, second, atol=0.0)

    def test_release_reset_is_exclusive_with_the_others(self):
        from environment.acrobot_xk import swingup_xk as make

        with self.assertRaises(ValueError):
            make(release_start=True, paper_start=True)
        with self.assertRaises(ValueError):
            make(release_start=True, uniform_start=True)

    def test_paper_start_reproduces_the_2007_initial_condition(self):
        env = DMCContinuousEnv(
            "acrobot",
            "swingup-xk",
            seed=0,
            raw_state_obs=True,
            dt=0.01,
            task_kwargs=dict(paper_start=True),
        )
        obs, _ = env.reset(seed=0)
        # qpos is the paper's q1 outright, so no map is applied.  The
        # observation is float32, hence the dtype-level tolerance.
        self.assertAlmostEqual(float(obs[0]), -1.4, places=6)
        np.testing.assert_allclose(obs[1:], 0.0, atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(env._env.physics.data.qpos)[0], -1.4, atol=1e-15
        )

    def test_energy_references_match_the_analytic_span(self):
        env = swingup_xk()
        env.reset()
        params = xk.AcrobotParams.from_physics(env.physics)
        self.assertAlmostEqual(
            env.task.energy_span, params.energy_span, places=9
        )

    def test_no_dependency_on_the_old_reward_line(self):
        import controllers.xin_kaneda
        import environment.acrobot_xk
        import evaluations.acrobot_homoclinic_metrics

        banned = ("acrobot_v2", "sustained_capture")
        for module in (
            controllers.xin_kaneda,
            environment.acrobot_xk,
            evaluations.acrobot_homoclinic_metrics,
        ):
            with open(module.__file__) as handle:
                source = handle.read()
            for name in banned:
                self.assertNotIn(
                    f"import {name}", source, msg=f"{module.__name__} imports {name}"
                )
                self.assertNotIn(
                    f"from .{name}", source, msg=f"{module.__name__} imports {name}"
                )


class TestMetrics(unittest.TestCase):
    """Metric definitions, on trajectories whose answers are known."""

    def setUp(self):
        env = swingup_xk()
        env.reset()
        self.params = xk.AcrobotParams.from_physics(env.physics)
        self.scales = metrics.Scales.from_params(self.params)

    def _trajectory(self, time, state, torque=None):
        time = np.asarray(time, dtype=np.float64)
        state = np.asarray(state, dtype=np.float64)
        n = time.size - 1
        torque = np.zeros(n) if torque is None else np.asarray(torque, float)
        return metrics.Trajectory(
            time=time,
            state=state,
            torque=torque,
            commanded_torque=torque,
            torque_limit=2.0,
        )

    def test_scales_are_derived_from_the_orbit(self):
        self.assertAlmostEqual(self.scales.energy, 49.0, places=9)
        self.assertAlmostEqual(self.scales.angle, np.pi, places=12)
        self.assertAlmostEqual(
            self.scales.rate, xk.homoclinic_speed(self.params), places=12
        )

    def test_points_on_the_orbit_have_zero_distance_and_energy_error(self):
        # eq. 32: qdot1 = +- omega_s sqrt((1 - sin q1) / 2).
        for q1 in (-2.5, -1.0, 0.0, 1.0, 2.0, 0.5 * np.pi):
            for sign in (1.0, -1.0):
                rate = sign * self.scales.rate * np.sqrt(
                    max(0.5 * (1.0 - np.sin(q1)), 0.0)
                )
                state = np.array([[q1, 0.0, rate, 0.0]])
                with self.subTest(q1=q1, sign=sign):
                    self.assertLess(
                        metrics.orbit_distance(state, self.params)[0], 1e-3
                    )
                    self.assertLess(
                        abs(metrics.energy_error(state, self.params)[0]), 1e-9
                    )

    def test_upright_at_rest_is_on_the_orbit(self):
        state = np.array([[0.5 * np.pi, 0.0, 0.0, 0.0]])
        self.assertAlmostEqual(
            metrics.energy_error(state, self.params)[0], 0.0, places=9
        )
        self.assertAlmostEqual(
            metrics.orbit_distance(state, self.params)[0], 0.0, places=9
        )

    def test_hanging_at_rest_is_a_full_span_below_the_top(self):
        state = np.array([[-0.5 * np.pi, 0.0, 0.0, 0.0]])
        self.assertAlmostEqual(
            metrics.energy_error(state, self.params)[0],
            -self.params.energy_span,
            places=9,
        )

    def test_capture_requires_a_full_dwell(self):
        # Inside from t = 2 onward; with a 1 s dwell the capture time is 2.
        time = np.linspace(0.0, 5.0, 51)
        inside = time >= 2.0
        self.assertAlmostEqual(
            metrics.capture_time(time, inside, 1.0), 2.0, places=9
        )
        # A run shorter than the dwell does not count.
        brief = (time >= 2.0) & (time <= 2.5)
        self.assertEqual(metrics.capture_time(time, brief, 1.0), float("inf"))

    def test_capture_ignores_the_interval_straddling_first_entry(self):
        # Only the second endpoint of the interval [1, 2] qualifies, so the run
        # starts at t = 2, not t = 1.
        time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        inside = np.array([False, False, True, True, True])
        self.assertAlmostEqual(
            metrics.capture_time(time, inside, 1.0), 2.0, places=9
        )

    def test_rollout_drops_a_trailing_zero_length_step(self):
        # A pre-built time grid can run out before the duration check clears,
        # in which case the env reports a step with next_t == cur_t.  It carries
        # no physical time, and keeping it would break the trajectory invariant.
        class _Stalling:
            """An env whose last step advances the clock by nothing."""

            def __init__(self):
                self._env = swingup_xk()
                self._env.reset()
                self.cur_t = 0.0
                self._times = [0.1, 0.2, 0.2]
                self._index = 0

            def reset(self, seed=None):
                self.cur_t = 0.0
                self._index = 0
                return np.zeros(4, dtype=np.float64), {}

            def step_dt(self, action):
                next_t = self._times[self._index]
                self._index += 1
                done = self._index >= len(self._times)
                obs = np.zeros(4, dtype=np.float64)
                self.cur_t = next_t
                return (obs, 0.0, action, 0.0, obs, next_t, False, done, {})

        env = _Stalling()
        trajectory = metrics.rollout(env, lambda obs: np.zeros(1), seed=0)
        np.testing.assert_allclose(trajectory.time, [0.0, 0.1, 0.2])
        self.assertEqual(trajectory.torque.size, 2)

    def test_capture_handles_irregular_timesteps(self):
        time = np.array([0.0, 0.3, 0.35, 1.9, 2.05, 2.4, 3.9, 4.0])
        inside = np.array([False, True, True, True, True, True, True, True])
        # The first qualifying endpoint is t = 0.3; a 1 s dwell is reached at
        # the endpoint t = 1.9.
        self.assertAlmostEqual(
            metrics.capture_time(time, inside, 1.0), 0.3, places=9
        )

    def test_retention_is_the_post_capture_time_fraction(self):
        time = np.linspace(0.0, 4.0, 5)
        inside = np.array([True, True, True, False, True])
        # Capture at t = 0; intervals [0,1] and [1,2] qualify, [2,3] and [3,4]
        # do not, so the fraction is 0.5.
        self.assertAlmostEqual(
            metrics.retention_fraction(time, inside, 0.0), 0.5, places=9
        )

    def test_control_effort_integrates_to_the_capture_time(self):
        time = np.array([0.0, 1.0, 2.0, 3.0])
        torque = np.array([2.0, 2.0, 2.0])
        self.assertAlmostEqual(
            metrics.control_effort(time, torque, 2.0), 8.0, places=9
        )
        self.assertAlmostEqual(
            metrics.control_effort(time, torque, float("inf")), 12.0, places=9
        )

    def test_saturation_is_time_weighted(self):
        time = np.array([0.0, 1.0, 4.0])
        commanded = np.array([2.0, 0.5])
        self.assertAlmostEqual(
            metrics.saturation_fraction(time, commanded, 2.0), 0.25, places=9
        )

    def test_lqr_residual_uses_the_wrapped_upright_error(self):
        # Upright at rest is the zero of the switching function, and the
        # shoulder angle enters wrapped so a full turn does not accumulate.
        up = 0.5 * np.pi
        np.testing.assert_allclose(
            metrics.lqr_residual(np.array([[up, 0.0, 0.0, 0.0]])), 0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            metrics.lqr_residual(np.array([[up + 2.0 * np.pi, 0.0, 0.0, 0.0]])),
            0.0,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            metrics.lqr_residual(np.array([[up + 0.1, 0.02, 1.0, 2.0]])),
            0.1 + 0.02 + 0.1 + 0.2,
            atol=1e-12,
        )

    def test_squared_distance_is_minus_the_baseline_reward(self):
        env = swingup_xk()
        env.reset()
        physics = env.physics
        rng = np.random.RandomState(3)
        for _ in range(20):
            physics.data.qpos[:] = rng.uniform(-np.pi, np.pi, 2)
            physics.data.qvel[:] = rng.uniform(-4.0, 4.0, 2)
            physics.forward()
            state = np.concatenate(
                [np.asarray(physics.data.qpos), np.asarray(physics.data.qvel)]
            )
            reward = env.task.baseline_terms(physics)["reward"]
            self.assertAlmostEqual(
                metrics.squared_distance(state[None, :], self.params)[0],
                -reward,
                places=9,
            )


class TestClosedLoopOnThePlant(unittest.TestCase):
    """The controller, stepped through MuJoCo, reaches the orbit."""

    def test_paper_gains_reach_the_homoclinic_orbit_in_mujoco(self):
        # The same run as the analytic reproduction, stepped through MuJoCo.
        env = DMCContinuousEnv(
            "acrobot",
            "swingup-xk",
            seed=0,
            raw_state_obs=True,
            dt=0.001,
            physics_dt=0.001,
            max_steps=12001,
            episode_duration=12.0,
            task_kwargs=dict(paper_start=True),
        )
        params = xk.AcrobotParams.from_physics(env._env.physics)
        controller = xk.XinKanedaController(
            params, xk.Gains(k_v=66.3, k_d=35.8, k_p=61.2)
        )
        trajectory = metrics.rollout(env, controller, seed=0)
        result = metrics.evaluate_episode(trajectory, params)
        self.assertTrue(result.captured)
        self.assertLess(result.capture_time, 10.0)
        self.assertLess(result.min_abs_energy_error, 0.05)
        # The peak shoulder speed approaches the orbit's own peak.
        self.assertGreater(
            result.peak_shoulder_rate, 0.9 * xk.homoclinic_speed(params)
        )
        # The default gear is ample, so the command never clips.
        self.assertEqual(result.saturation, 0.0)
        self.assertLess(result.peak_commanded_torque, DEFAULT_TORQUE_LIMIT)

    def test_hanging_is_an_equilibrium_of_the_closed_loop(self):
        params = xk.PAPER_PARAMS
        gains = xk.Gains(k_v=66.3, k_d=35.8, k_p=61.2)
        drift, applied = xk.closed_loop(
            params, gains, np.array([-0.5 * np.pi, 0.0, 0.0, 0.0])
        )
        np.testing.assert_allclose(drift, 0.0, atol=1e-12)
        self.assertAlmostEqual(applied, 0.0, places=12)

    def test_controller_command_is_normalized_by_the_gear(self):
        env = swingup_xk()
        env.reset()
        params = xk.AcrobotParams.from_physics(env.physics)
        controller = xk.XinKanedaController(
            params, xk.Gains(k_v=66.3, k_d=35.8, k_p=61.2)
        )
        action = controller(np.array([-0.5 * np.pi + 0.2, 0.1, 0.3, -0.2]))
        self.assertEqual(action.shape, (1,))
        # No sign flip: the plant is already in the paper's frame.
        self.assertAlmostEqual(
            float(action[0]), controller.last_torque / params.gear, places=12
        )
        self.assertLessEqual(abs(float(action[0])), 1.0)

    def test_torque_limit_above_the_gear_is_rejected(self):
        params = _repo_params()
        with self.assertRaises(ValueError):
            xk.XinKanedaController(
                params,
                xk.Gains(k_v=45.0, k_d=12.0, k_p=60.0),
                torque_limit=1e3,
            )


if __name__ == "__main__":
    unittest.main()
