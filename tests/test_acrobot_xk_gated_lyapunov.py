"""Tests for the gated Xin--Kaneda/LQR Lyapunov construction."""

import unittest

import numpy as np

from controllers import xin_kaneda as xk
from controllers.acrobot_gated_lyapunov import (
    AttractiveRegion,
    GatedLyapunov,
    LQRDesign,
    NonsmoothLyapunov,
    UPRIGHT_STATE,
    XKLQRSwitchedController,
    lqr_scale_on_switch_region,
    lqr_solution,
    lqr_switch_residual,
    max_local_value_on_region,
    riccati_feedback,
    plant_drift_and_gain,
    upright_error,
    upright_linearization,
)


GAINS = xk.Gains(k_v=66.3, k_d=35.8, k_p=61.2)

# Lai, Wu, She and Yang (2009), equation (75): the state feedback they publish
# for this Acrobot at Q = I and R = 0.5.
LAI_PUBLISHED_FEEDBACK = np.array([-260.559, -104.448, -112.604, -52.944])
# Actuator of the evaluation protocol in docs/reward_shaping_for_acrobot_swingup.md.
TORQUE_LIMIT = 64.0


def _plant_drift(params, state, torque):
    state = np.asarray(state, dtype=np.float64)
    acceleration = np.linalg.solve(
        params.mass_matrix(state[1]),
        np.array([0.0, torque]) - params.bias(state[:2], state[2:]),
    )
    return np.concatenate([state[2:], acceleration])


def _region_states(params, region, rng, count, *, on_shell):
    """Draw states from the attractive region, optionally on its energy shell."""
    states = []
    tolerance = region.angle_tolerance
    while len(states) < count:
        shoulder = rng.uniform(-tolerance, tolerance)
        tip = rng.uniform(-tolerance, tolerance)
        q1 = 0.5 * np.pi + shoulder
        q2 = tip - shoulder
        potential = params.b1 * np.sin(q1) + params.b2 * np.sin(q1 + q2)
        kinetic = params.energy_top + region.energy_tolerance - potential
        if kinetic <= 0.0:
            continue
        factor = np.linalg.cholesky(params.mass_matrix(q2))
        direction = rng.normal(size=2)
        fraction = 1.0 if on_shell else rng.uniform(0.0, 1.0)
        direction *= np.sqrt(2.0 * kinetic * fraction) / np.linalg.norm(direction)
        velocity = np.linalg.solve(factor.T, direction)
        states.append([q1, q2, velocity[0], velocity[1]])
    return np.array(states)


class TestLocalLQRConstruction(unittest.TestCase):
    def setUp(self):
        self.params = xk.PAPER_PARAMS
        self.design = LQRDesign()

    def test_upright_error_wraps_both_angles(self):
        state = UPRIGHT_STATE + np.array([2.0 * np.pi, -2.0 * np.pi, 1.0, -2.0])
        np.testing.assert_allclose(
            upright_error(state), [0.0, 0.0, 1.0, -2.0], atol=1e-12
        )

    def test_switch_residual_is_equation_74(self):
        state = UPRIGHT_STATE + np.array([0.1, -0.02, 1.0, -2.0])
        self.assertAlmostEqual(
            lqr_switch_residual(state), 0.1 + 0.02 + 0.1 + 0.2, places=12
        )

    def test_smoothing_width_must_leave_a_full_lqr_core(self):
        with self.assertRaisesRegex(ValueError, "smooth_abs_epsilon"):
            LQRDesign(inner_switch_threshold=0.02, smooth_abs_epsilon=0.005)

    def test_analytic_upright_linearization_matches_nonlinear_plant(self):
        a, b = upright_linearization(self.params)
        step = 1e-6
        numerical_a = np.empty((4, 4))
        for column in range(4):
            offset = np.zeros(4)
            offset[column] = step
            numerical_a[:, column] = (
                _plant_drift(self.params, UPRIGHT_STATE + offset, 0.0)
                - _plant_drift(self.params, UPRIGHT_STATE - offset, 0.0)
            ) / (2.0 * step)
        numerical_b = (
            _plant_drift(self.params, UPRIGHT_STATE, step)
            - _plant_drift(self.params, UPRIGHT_STATE, -step)
        )[:, None] / (2.0 * step)
        np.testing.assert_allclose(a, numerical_a, atol=2e-9)
        np.testing.assert_allclose(b, numerical_b, atol=2e-10)

    def test_care_solution_and_feedback_are_stable(self):
        a, b, k, p = lqr_solution(self.params, self.design)
        q = np.diag(self.design.q)
        r = np.array([[self.design.r]])
        residual = a.T @ p + p @ a - p @ b @ np.linalg.solve(r, b.T @ p) + q
        np.testing.assert_allclose(residual, 0.0, atol=5e-8)
        self.assertGreater(np.min(np.linalg.eigvalsh(p)), 0.0)
        self.assertLess(np.max(np.real(np.linalg.eigvals(a - b @ k))), 0.0)

    def test_lqr_scale_bounds_the_whole_weighted_l1_region(self):
        _, _, _, p = lqr_solution(self.params, self.design)
        weights = np.asarray(self.design.switch_weights)
        scale = lqr_scale_on_switch_region(
            p, self.design.switch_threshold, weights
        )
        rng = np.random.RandomState(7)
        for _ in range(5000):
            magnitudes = rng.dirichlet(np.ones(4)) * self.design.switch_threshold
            signs = rng.choice((-1.0, 1.0), size=4)
            error = signs * magnitudes / weights
            self.assertLessEqual(float(error @ p @ error), scale * (1.0 + 1e-12))


class TestGatedLyapunov(unittest.TestCase):
    def setUp(self):
        self.candidate = GatedLyapunov(xk.PAPER_PARAMS, GAINS)

    def test_smooth_residual_is_a_conservative_switch_test(self):
        rng = np.random.RandomState(11)
        for _ in range(1000):
            state = UPRIGHT_STATE + rng.uniform(-0.5, 0.5, size=4)
            exact = lqr_switch_residual(state)
            smooth = self.candidate.smooth_switch_residual(state)
            self.assertGreaterEqual(smooth, exact)
            if self.candidate.gate(state) > 0.0:
                self.assertLess(exact, self.candidate.design.switch_threshold)

    def test_gate_has_the_expected_inner_and_outer_values(self):
        inner = UPRIGHT_STATE + np.array([0.01, 0.0, 0.0, 0.0])
        transition = UPRIGHT_STATE + np.array([0.03, 0.0, 0.0, 0.0])
        outer = UPRIGHT_STATE + np.array([0.05, 0.0, 0.0, 0.0])
        self.assertEqual(self.candidate.gate(inner), 1.0)
        self.assertGreater(self.candidate.gate(transition), 0.0)
        self.assertLess(self.candidate.gate(transition), 1.0)
        self.assertEqual(self.candidate.gate(outer), 0.0)

    def test_candidate_selects_the_expected_endpoint_values(self):
        inner = UPRIGHT_STATE + np.array([0.01, 0.0, 0.0, 0.0])
        outer = UPRIGHT_STATE + np.array([0.05, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(
            self.candidate.value(inner),
            self.candidate.lqr_value_normalized(inner),
            places=12,
        )
        self.assertAlmostEqual(
            self.candidate.value(outer), self.candidate.xk_value(outer), places=12
        )
        self.assertEqual(self.candidate.value(UPRIGHT_STATE), 0.0)

    def test_xk_component_matches_the_published_function(self):
        state = np.array([-0.7, 0.4, 1.2, -0.8])
        expected = xk.lyapunov(xk.PAPER_PARAMS, GAINS, state)
        scale = 0.5 * xk.PAPER_PARAMS.energy_span**2
        self.assertAlmostEqual(self.candidate.xk_value(state), expected / scale, 12)

    def test_exact_gradient_matches_finite_difference_in_all_gate_zones(self):
        states = (
            UPRIGHT_STATE + np.array([0.01, 0.002, 0.01, -0.01]),
            UPRIGHT_STATE + np.array([0.025, 0.002, 0.01, -0.01]),
            UPRIGHT_STATE + np.array([0.06, 0.01, 0.05, -0.02]),
        )
        step = 1e-7
        for state in states:
            with self.subTest(state=state):
                value, gradient = self.candidate.value_and_gradient(state)
                self.assertAlmostEqual(value, self.candidate.value(state), places=14)
                numerical = np.empty(4)
                for column in range(4):
                    offset = np.zeros(4)
                    offset[column] = step
                    numerical[column] = (
                        self.candidate.value(state + offset)
                        - self.candidate.value(state - offset)
                    ) / (2.0 * step)
                np.testing.assert_allclose(gradient, numerical, rtol=2e-6, atol=2e-8)

    def test_rate_includes_the_gate_derivative(self):
        state = UPRIGHT_STATE + np.array([0.025, 0.002, 0.01, -0.01])
        drift = np.array([0.01, -0.01, 0.3, -0.2])
        step = 1e-7
        numerical = (
            self.candidate.value(state + step * drift)
            - self.candidate.value(state - step * drift)
        ) / (2.0 * step)
        self.assertAlmostEqual(self.candidate.rate(state, drift), numerical, delta=2e-8)

    def test_local_lqr_value_decreases_under_its_nonlinear_feedback(self):
        rng = np.random.RandomState(13)
        for _ in range(100):
            error = rng.uniform(-1.0, 1.0, size=4)
            error *= 0.005 / max(lqr_switch_residual(UPRIGHT_STATE + error), 0.005)
            state = UPRIGHT_STATE + error
            self.assertEqual(self.candidate.gate(state), 1.0)
            torque = self.candidate.lqr_torque(state)
            drift = _plant_drift(xk.PAPER_PARAMS, state, torque)
            self.assertLess(self.candidate.rate(state, drift), 0.0)

    def test_candidate_retains_xk_zero_set_outside_the_gate(self):
        state = np.array(
            [-0.5 * np.pi, 0.0, xk.homoclinic_speed(xk.PAPER_PARAMS), 0.0]
        )
        self.assertEqual(self.candidate.gate(state), 0.0)
        self.assertAlmostEqual(self.candidate.xk_value(state), 0.0, places=12)
        self.assertAlmostEqual(self.candidate.value(state), 0.0, places=12)


class TestAttractiveRegion(unittest.TestCase):
    def setUp(self):
        self.params = xk.PAPER_PARAMS
        self.region = AttractiveRegion()

    def test_defaults_follow_lai_equation_17(self):
        self.assertAlmostEqual(self.region.energy_tolerance, 1.0, places=12)
        self.assertLess(self.region.angle_tolerance, np.pi / 6.0)

    def test_rejects_invalid_designs(self):
        with self.assertRaisesRegex(ValueError, "angle_tolerance"):
            AttractiveRegion(angle_tolerance=0.0)
        with self.assertRaisesRegex(ValueError, "energy_tolerance"):
            AttractiveRegion(energy_tolerance=-1.0)
        with self.assertRaisesRegex(ValueError, "transition_fraction"):
            AttractiveRegion(transition_fraction=1.0)
        with self.assertRaisesRegex(ValueError, "norm_order"):
            AttractiveRegion(norm_order=0.5)

    def test_residual_reads_each_of_the_three_conditions(self):
        tolerance = self.region.angle_tolerance
        shoulder = UPRIGHT_STATE + np.array([0.5 * tolerance, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(
            self.region.exact_residual(self.params, shoulder), 0.5, places=9
        )
        # An equal and opposite elbow angle holds q1 + q2 at upright, so only
        # the shoulder condition is active.
        tip = UPRIGHT_STATE + np.array([0.5 * tolerance, -0.5 * tolerance, 0.0, 0.0])
        self.assertAlmostEqual(
            self.region.exact_residual(self.params, tip), 0.5, places=9
        )
        # Rest at upright with a raised elbow: the energy condition binds.
        state = np.array([0.5 * np.pi, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(
            self.region.exact_residual(self.params, state), 0.0, places=9
        )

    def test_membership_agrees_with_the_printed_conditions(self):
        rng = np.random.RandomState(5)
        tolerance = self.region.angle_tolerance
        inside = 0
        for _ in range(2000):
            state = UPRIGHT_STATE + rng.uniform(-2.0 * tolerance, 2.0 * tolerance, 4)
            error = upright_error(state)
            shoulder = abs(error[0])
            tip = abs(
                np.arctan2(
                    np.sin(state[0] + state[1] - 0.5 * np.pi),
                    np.cos(state[0] + state[1] - 0.5 * np.pi),
                )
            )
            energy = abs(
                self.params.energy(state[:2], state[2:]) - self.params.energy_top
            )
            printed = (
                shoulder <= tolerance
                and tip <= tolerance
                and energy <= self.region.energy_tolerance
            )
            self.assertEqual(self.region.contains(self.params, state), printed)
            inside += printed
        self.assertGreater(inside, 0)

    def test_smooth_residual_never_understates_the_exact_one(self):
        rng = np.random.RandomState(17)
        tolerance = self.region.angle_tolerance
        for _ in range(2000):
            state = UPRIGHT_STATE + rng.uniform(-4.0 * tolerance, 4.0 * tolerance, 4)
            smooth, _ = self.region.smooth_residual_and_gradient(self.params, state)
            self.assertGreaterEqual(smooth, self.region.exact_residual(self.params, state))

    def test_smooth_residual_gradient_matches_finite_difference(self):
        rng = np.random.RandomState(19)
        step = 1e-7
        for _ in range(20):
            state = UPRIGHT_STATE + rng.uniform(-0.1, 0.1, 4)
            _, gradient = self.region.smooth_residual_and_gradient(self.params, state)
            numerical = np.empty(4)
            for column in range(4):
                offset = np.zeros(4)
                offset[column] = step
                numerical[column] = (
                    self.region.smooth_residual_and_gradient(
                        self.params, state + offset
                    )[0]
                    - self.region.smooth_residual_and_gradient(
                        self.params, state - offset
                    )[0]
                ) / (2.0 * step)
            np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=1e-7)


class TestRegionIsSharedAcrossFrames(unittest.TestCase):
    """The one home for equation (17) has to serve both coordinate frames."""

    def test_residual_is_even_so_both_frames_agree(self):
        """The 2009 frame is ``x = -e``; every condition is even in ``e``."""
        region = AttractiveRegion()
        rng = np.random.RandomState(43)
        for _ in range(2000):
            error = rng.uniform(-0.5, 0.5, 4)
            energy_error = rng.uniform(-2.0, 2.0)
            xk_frame = region.residual_of(
                error[0], error[0] + error[1], energy_error, error[2:]
            )
            paper_frame = region.residual_of(
                -error[0], -(error[0] + error[1]), energy_error, -error[2:]
            )
            self.assertAlmostEqual(xk_frame, paper_frame, places=14)

    def test_lai_she_design_builds_the_paper_region(self):
        from controllers import lai_she as ls

        design = ls.Design()
        region = design.attractive_region()
        self.assertAlmostEqual(region.angle_tolerance, np.pi / 6.0, places=12)
        self.assertAlmostEqual(region.effective_tip_tolerance, np.pi / 6.0, places=12)
        self.assertAlmostEqual(region.energy_tolerance, 1.0, places=12)
        self.assertAlmostEqual(region.velocity_tolerance, 1e3, places=12)

    def test_lai_she_controller_reaches_the_region_through_the_shared_object(self):
        from controllers import lai_she as ls

        controller = ls.LaiSheController()
        rng = np.random.RandomState(47)
        agreed = inside = 0
        for _ in range(5000):
            state = np.concatenate(
                [rng.uniform(-0.7, 0.7, 2), rng.uniform(-3.0, 3.0, 2)]
            )
            expected = (
                controller.region.residual_of(
                    float(ls.wrap(state[0])),
                    float(ls.wrap(state[0] + state[1])),
                    controller.params.energy(state) - controller.design.energy_top,
                    state[2:],
                )
                <= 1.0
            )
            self.assertEqual(controller.in_attractive_area(state), expected)
            agreed += 1
            inside += expected
        self.assertGreater(inside, 0)
        self.assertLess(inside, agreed)

    def test_velocity_condition_is_carried_even_though_it_is_vacuous(self):
        region = AttractiveRegion()
        # At the paper's weights the cap sits at ||qdot|| = 1e6, so it never
        # binds in practice, but it must still be enforced.
        self.assertLessEqual(region.residual_of(0.0, 0.0, 0.0, [1e5, 0.0]), 1.0)
        self.assertGreater(region.residual_of(0.0, 0.0, 0.0, [2e6, 0.0]), 1.0)

    def test_riccati_feedback_solves_the_care_it_is_given(self):
        params = xk.PAPER_PARAMS
        a, b = upright_linearization(params)
        weights = (1.0, 2.0, 3.0, 0.5)
        cost = 0.25
        k, p = riccati_feedback(a, b, weights, cost)
        residual = (
            a.T @ p + p @ a - p @ b @ (b.T @ p) / cost + np.diag(weights)
        )
        np.testing.assert_allclose(residual, 0.0, atol=5e-8)
        np.testing.assert_allclose(p, p.T, atol=0.0)
        np.testing.assert_allclose(k, b.T @ p / cost, rtol=1e-12)


class TestNonsmoothLyapunov(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = xk.PAPER_PARAMS
        cls.candidate = NonsmoothLyapunov(xk.PAPER_PARAMS, GAINS)
        cls.region = cls.candidate.region

    def test_local_design_reproduces_lai_published_feedback(self):
        np.testing.assert_allclose(
            self.candidate.k[0], LAI_PUBLISHED_FEEDBACK, rtol=2e-3
        )

    def test_delta_dominates_the_local_value_everywhere_on_the_region(self):
        """Lai equation (71): the offset must cover the whole attractive area."""
        rng = np.random.RandomState(23)
        states = _region_states(self.params, self.region, rng, 4000, on_shell=True)
        for state in states:
            error = upright_error(state)
            self.assertLessEqual(
                float(error @ self.candidate.p @ error),
                self.candidate.delta * (1.0 + 1e-6),
            )

    def test_delta_is_insensitive_to_the_search_grid(self):
        coarse = max_local_value_on_region(
            self.params,
            self.candidate.p,
            self.region,
            angle_samples=61,
            velocity_samples=241,
        )
        self.assertAlmostEqual(coarse / self.candidate.delta, 1.0, places=4)

    def test_gate_has_the_expected_inner_and_outer_values(self):
        tolerance = self.region.angle_tolerance
        fraction = self.region.transition_fraction
        inner = UPRIGHT_STATE + np.array([0.5 * fraction * tolerance, 0.0, 0.0, 0.0])
        transition = UPRIGHT_STATE + np.array(
            [0.5 * (1.0 + fraction) * tolerance, 0.0, 0.0, 0.0]
        )
        outer = UPRIGHT_STATE + np.array([2.0 * tolerance, 0.0, 0.0, 0.0])
        self.assertEqual(self.candidate.gate(inner), 1.0)
        self.assertGreater(self.candidate.gate(transition), 0.0)
        self.assertLess(self.candidate.gate(transition), 1.0)
        self.assertEqual(self.candidate.gate(outer), 0.0)

    def test_gate_is_active_only_inside_the_printed_region(self):
        rng = np.random.RandomState(29)
        tolerance = self.region.angle_tolerance
        for _ in range(2000):
            state = UPRIGHT_STATE + rng.uniform(-3.0 * tolerance, 3.0 * tolerance, 4)
            if self.candidate.gate(state) > 0.0:
                self.assertTrue(self.region.contains(self.params, state))

    def test_value_selects_the_expected_endpoint_pieces(self):
        tolerance = self.region.angle_tolerance
        inner = UPRIGHT_STATE + np.array([0.01 * tolerance, 0.0, 0.0, 0.0])
        outer = UPRIGHT_STATE + np.array([2.0 * tolerance, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(
            self.candidate.value(inner), self.candidate.local_value(inner), places=12
        )
        self.assertAlmostEqual(
            self.candidate.value(outer), self.candidate.swing_up_value(outer), places=12
        )
        self.assertEqual(self.candidate.value(UPRIGHT_STATE), 0.0)

    def test_offset_leaves_the_swing_up_shaping_untouched(self):
        """A constant shifts the value, never the gradient the reward uses."""
        bare = GatedLyapunov(self.params, GAINS)
        rng = np.random.RandomState(31)
        for _ in range(50):
            state = np.array([-0.5 * np.pi, 0.0, 0.0, 0.0]) + rng.uniform(-1.0, 1.0, 4)
            self.assertEqual(self.candidate.gate(state), 0.0)
            value, gradient = self.candidate.value_and_gradient(state)
            self.assertAlmostEqual(
                value - self.candidate.normalized_delta, bare.xk_value(state), places=10
            )
            np.testing.assert_allclose(
                gradient, bare.value_and_gradient(state)[1], rtol=1e-10, atol=1e-12
            )

    def test_exact_gradient_matches_finite_difference_in_all_gate_zones(self):
        tolerance = self.region.angle_tolerance
        # The middle state keeps q1 + q2 at upright so the shoulder condition
        # alone sets the residual, which places it squarely in the band.
        states = (
            UPRIGHT_STATE + np.array([0.1 * tolerance, 0.0, 0.05, -0.03]),
            UPRIGHT_STATE
            + np.array([0.75 * tolerance, -0.75 * tolerance, 0.05, -0.05]),
            UPRIGHT_STATE + np.array([3.0 * tolerance, 0.10, 0.50, -0.30]),
        )
        zones = set()
        step = 1e-7
        for state in states:
            with self.subTest(state=state):
                zones.add(np.clip(self.candidate.gate(state), 0.0, 1.0) in (0.0, 1.0))
                value, gradient = self.candidate.value_and_gradient(state)
                self.assertAlmostEqual(value, self.candidate.value(state), places=14)
                numerical = np.empty(4)
                for column in range(4):
                    offset = np.zeros(4)
                    offset[column] = step
                    numerical[column] = (
                        self.candidate.value(state + offset)
                        - self.candidate.value(state - offset)
                    ) / (2.0 * step)
                np.testing.assert_allclose(gradient, numerical, rtol=2e-6, atol=2e-8)
        self.assertGreater(
            sum(0.0 < self.candidate.gate(s) < 1.0 for s in states), 0
        )

    def test_rate_includes_the_gate_derivative(self):
        tolerance = self.region.angle_tolerance
        state = UPRIGHT_STATE + np.array(
            [0.75 * tolerance, -0.75 * tolerance, 0.05, -0.05]
        )
        self.assertGreater(self.candidate.gate(state), 0.0)
        self.assertLess(self.candidate.gate(state), 1.0)
        drift = np.array([0.05, -0.03, 0.3, -0.2])
        step = 1e-7
        numerical = (
            self.candidate.value(state + step * drift)
            - self.candidate.value(state - step * drift)
        ) / (2.0 * step)
        self.assertAlmostEqual(self.candidate.rate(state, drift), numerical, delta=2e-8)

    def test_value_never_rises_along_the_homoclinic_orbit(self):
        """The ridge that GatedLyapunov leaves on the orbit is gone."""
        speed = xk.homoclinic_speed(self.params)
        shoulders = np.linspace(0.5 * np.pi - 0.8, 0.5 * np.pi, 4001)
        orbit = [
            np.array([q1, 0.0, -speed * abs(np.sin(0.5 * (q1 - 0.5 * np.pi))), 0.0])
            for q1 in shoulders
        ]
        values = np.array([self.candidate.value(state) for state in orbit])
        self.assertLessEqual(float(np.max(np.diff(values))), 0.0)
        self.assertAlmostEqual(values[0], self.candidate.normalized_delta, places=12)
        self.assertAlmostEqual(values[-1], 0.0, places=12)

        ridge = np.array([GatedLyapunov(self.params, GAINS).value(s) for s in orbit])
        self.assertGreater(float(np.max(np.diff(ridge))), 0.0)

    def test_local_piece_is_a_clf_on_the_region_under_the_actuator(self):
        """Some admissible torque decreases e'Pe everywhere in the region."""
        rng = np.random.RandomState(37)
        states = np.vstack(
            [
                _region_states(self.params, self.region, rng, 3000, on_shell=True),
                _region_states(self.params, self.region, rng, 3000, on_shell=False),
            ]
        )
        margins = np.array(
            [self.candidate.clf_margin(state, TORQUE_LIMIT) for state in states]
        )
        self.assertLess(float(np.max(margins)), 0.0)

    def test_clf_margin_is_the_best_rate_an_admissible_torque_achieves(self):
        rng = np.random.RandomState(41)
        state = _region_states(self.params, self.region, rng, 1, on_shell=False)[0]
        margin = self.candidate.clf_margin(state, TORQUE_LIMIT)
        error = upright_error(state)
        gradient = 2.0 * (self.candidate.p @ error) / self.candidate.scale
        drift, gain = plant_drift_and_gain(self.params, state)
        rates = [
            gradient @ (drift + gain * torque)
            for torque in np.linspace(-TORQUE_LIMIT, TORQUE_LIMIT, 401)
        ]
        self.assertAlmostEqual(margin, min(rates), places=9)

    def test_delta_override_skips_the_search_and_is_validated(self):
        reused = NonsmoothLyapunov(
            self.params, GAINS, self.region, delta_override=self.candidate.delta
        )
        self.assertEqual(reused.delta, self.candidate.delta)
        state = UPRIGHT_STATE + np.array([0.05, 0.01, 0.2, -0.1])
        self.assertAlmostEqual(
            reused.value(state), self.candidate.value(state), places=12
        )
        with self.assertRaisesRegex(ValueError, "delta_override"):
            NonsmoothLyapunov(self.params, GAINS, self.region, delta_override=0.0)

    def test_clf_margin_rejects_a_nonpositive_bound(self):
        with self.assertRaisesRegex(ValueError, "torque_limit"):
            self.candidate.clf_margin(UPRIGHT_STATE, 0.0)


class TestXKLQRSwitchedController(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = xk.PAPER_PARAMS

    def _controller(self):
        return XKLQRSwitchedController(self.params, GAINS)

    def test_starts_and_stays_in_swing_up_away_from_the_region(self):
        c = self._controller()
        self.assertEqual(c.stage, c.SWING_UP)
        hanging = np.array([-0.5 * np.pi, 0.0, 0.0, 0.0])
        action = c(hanging)
        self.assertEqual(c.stage, c.SWING_UP)
        self.assertEqual(action.shape, (1,))
        self.assertTrue(np.all(np.isfinite(action)))
        self.assertLessEqual(float(np.abs(action[0])), 1.0 + 1e-9)

    def test_switches_to_balance_on_entering_the_region_and_latches(self):
        c = self._controller()
        self.assertTrue(c.lyapunov.region.contains(self.params, UPRIGHT_STATE))
        action = c(UPRIGHT_STATE)
        self.assertEqual(c.stage, c.BALANCE)
        self.assertEqual(c.switch_step, 0)
        expected = float(
            np.clip(
                c.lyapunov.lqr_torque(UPRIGHT_STATE),
                -c.torque_limit,
                c.torque_limit,
            )
            / self.params.gear
        )
        self.assertAlmostEqual(float(action[0]), expected, places=10)

        # One-way latch: back outside the region, stage does not revert.
        hanging = np.array([-0.5 * np.pi, 0.0, 0.0, 0.0])
        c(hanging)
        self.assertEqual(c.stage, c.BALANCE)

    def test_reset_returns_to_swing_up(self):
        c = self._controller()
        c(UPRIGHT_STATE)
        self.assertEqual(c.stage, c.BALANCE)
        c.reset()
        self.assertEqual(c.stage, c.SWING_UP)
        self.assertIsNone(c.switch_step)

    def test_rejects_a_nonpositive_torque_limit(self):
        with self.assertRaisesRegex(ValueError, "torque_limit"):
            XKLQRSwitchedController(self.params, GAINS, torque_limit=0.0)


if __name__ == "__main__":
    unittest.main()
