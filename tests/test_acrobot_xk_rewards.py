"""Reward-contract tests for the CT-SAC Acrobot Xin--Kaneda task."""

import os
import unittest

os.environ.setdefault("MUJOCO_GL", "disable")

import numpy as np

from controllers import xin_kaneda as xk
from environment.acrobot_xk import (
    DEFAULT_LYAPUNOV_K_D,
    DEFAULT_LYAPUNOV_K_P,
    DEFAULT_LYAPUNOV_K_V,
    ELBOW_ANGLE_LIMIT,
    ELBOW_RATE_LIMIT,
    HANGING_SHOULDER,
    HOMOCLINIC_ANGLE_TOLERANCE,
    LOWER_BOUND_TERMINATION_REWARD_SOURCE,
    SHOULDER_RATE_SCALE_LIMIT,
    UPRIGHT_SHOULDER,
    reward_rate_lower_bound,
    swingup_xk,
)
from environment.dmc import DMCContinuousEnv


class TestAcrobotXKRewards(unittest.TestCase):
    def _env(
        self,
        reward_kind="r0",
        *,
        eta=None,
        discount_rate=None,
        **task_kwargs,
    ):
        env = swingup_xk(
            reward_kind=reward_kind,
            eta=eta,
            discount_rate=discount_rate,
            angle_noise=0.0,
            velocity_noise=0.0,
            **task_kwargs,
        )
        env.reset()
        return env

    @staticmethod
    def _set_state_and_torque(env, state, torque):
        physics = env.physics
        physics.data.qpos[:] = np.asarray(state[:2], dtype=np.float64)
        physics.data.qvel[:] = np.asarray(state[2:], dtype=np.float64)
        gear = float(np.asarray(physics.model.actuator_gear)[0, 0])
        physics.data.ctrl[:] = float(torque) / gear
        physics.forward()

    def test_default_is_the_existing_r0_reward(self):
        env = self._env()
        state = np.array([-1.1, 0.4, 1.2, -0.7])
        self._set_state_and_torque(env, state, torque=8.0)
        terms = env.task.xk_reward_terms(env.physics)
        self.assertEqual(env.task.reward_kind, "r0")
        self.assertAlmostEqual(terms["reward"], terms["r0"], places=12)
        self.assertAlmostEqual(
            terms["reward"], env.task.baseline_terms(env.physics)["reward"], 12
        )

    def test_r1_is_published_lyapunov_function_scaled_by_hanging_rest(self):
        env = self._env("r1")
        params = xk.AcrobotParams.from_physics(env.physics)
        gains = xk.Gains(
            k_v=1.0, k_d=DEFAULT_LYAPUNOV_K_D, k_p=DEFAULT_LYAPUNOV_K_P
        )
        state = np.array([-0.7, 3.6, 1.4, -0.8])
        self._set_state_and_torque(env, state, torque=-5.0)
        terms = env.task.xk_reward_terms(env.physics)
        expected = xk.lyapunov(params, gains, state)
        self.assertAlmostEqual(terms["lyapunov"], expected, places=9)
        expected_scale = 0.5 * env.task.energy_span**2
        self.assertAlmostEqual(env.task.lyapunov_scale, expected_scale, places=12)
        self.assertAlmostEqual(terms["lyapunov_scale"], expected_scale, places=12)
        self.assertAlmostEqual(
            terms["lyapunov_normalized"], expected / expected_scale, places=12
        )
        self.assertAlmostEqual(terms["reward"], -expected / expected_scale, places=12)

    def test_hanging_rest_puts_r0_and_r1_on_the_same_minus_one_scale(self):
        env = self._env("r1")
        self._set_state_and_torque(
            env, [HANGING_SHOULDER, 0.0, 0.0, 0.0], torque=0.0
        )
        terms = env.task.xk_reward_terms(env.physics)
        self.assertAlmostEqual(terms["r0"], -1.0, places=12)
        self.assertAlmostEqual(terms["r1"], -1.0, places=12)
        self.assertAlmostEqual(terms["lyapunov_normalized"], 1.0, places=12)

    def test_lyapunov_rate_matches_the_state_directional_derivative(self):
        env = self._env("r2", eta=0.3)
        params = xk.AcrobotParams.from_physics(env.physics)
        gains = xk.Gains(
            k_v=1.0, k_d=DEFAULT_LYAPUNOV_K_D, k_p=DEFAULT_LYAPUNOV_K_P
        )
        state = np.array([-0.9, 0.6, 1.1, -0.75])
        torque = 7.5
        self._set_state_and_torque(env, state, torque)
        terms = env.task.xk_reward_terms(env.physics)

        q = state[:2]
        qdot = state[2:]
        qddot = np.linalg.solve(
            params.mass_matrix(q[1]),
            np.array([0.0, torque]) - params.bias(q, qdot),
        )
        drift = np.concatenate([qdot, qddot])
        eps = 1e-7
        finite_difference = (
            xk.lyapunov(params, gains, state + eps * drift)
            - xk.lyapunov(params, gains, state - eps * drift)
        ) / (2.0 * eps)
        self.assertAlmostEqual(
            terms["lyapunov_rate"], finite_difference, delta=2e-5
        )
        self.assertAlmostEqual(
            terms["lyapunov_rate_normalized"],
            terms["lyapunov_rate"] / terms["lyapunov_scale"],
            places=12,
        )
        self.assertAlmostEqual(
            terms["r2"],
            terms["r1"] - 0.3 * terms["lyapunov_rate_normalized"],
            places=10,
        )

    def test_eta_and_discount_rate_validation_matches_reward_definitions(self):
        for kind in ("r2", "r3"):
            with self.subTest(kind=kind, case="missing eta"):
                with self.assertRaises(ValueError):
                    swingup_xk(
                        reward_kind=kind,
                        **({"discount_rate": 0.1} if kind == "r3" else {}),
                    )
            with self.subTest(kind=kind, case="negative eta"):
                with self.assertRaises(ValueError):
                    swingup_xk(
                        reward_kind=kind,
                        eta=-0.1,
                        **({"discount_rate": 0.1} if kind == "r3" else {}),
                    )
        with self.assertRaises(ValueError):
            swingup_xk(reward_kind="r1", eta=0.1)
        with self.assertRaises(ValueError):
            swingup_xk(reward_kind="r3", eta=0.1)
        with self.assertRaises(ValueError):
            swingup_xk(reward_kind="r3", eta=0.1, discount_rate=-0.1)
        with self.assertRaises(ValueError):
            swingup_xk(reward_kind="r2", eta=0.1, discount_rate=0.1)

        env = self._env("r2", eta=0.0)
        self._set_state_and_torque(env, [-1.0, 0.2, 0.4, -0.3], torque=4.0)
        terms = env.task.xk_reward_terms(env.physics)
        self.assertAlmostEqual(terms["r2"], terms["r1"], places=12)

    def test_r3_is_discount_consistent_potential_shaping_rate(self):
        eta = 0.3
        discount_rate = 0.1
        env = self._env("r3", eta=eta, discount_rate=discount_rate)
        self._set_state_and_torque(env, [-0.9, 0.6, 1.1, -0.75], torque=7.5)
        terms = env.task.xk_reward_terms(env.physics)
        expected = (
            terms["r1"]
            - eta * terms["lyapunov_rate_normalized"]
            + discount_rate * eta * terms["lyapunov_normalized"]
        )
        self.assertAlmostEqual(terms["r3"], expected, places=12)
        self.assertAlmostEqual(terms["reward"], expected, places=12)

        eta_zero = self._env("r3", eta=0.0, discount_rate=discount_rate)
        self._set_state_and_torque(
            eta_zero, [-1.0, 0.2, 0.4, -0.3], torque=4.0
        )
        zero_terms = eta_zero.task.xk_reward_terms(eta_zero.physics)
        self.assertAlmostEqual(zero_terms["r3"], zero_terms["r1"], places=12)

    def test_r0_base_replaces_only_the_leading_lyapunov_reward(self):
        eta = 0.3
        discount_rate = 0.1
        env = self._env(
            "r3",
            eta=eta,
            discount_rate=discount_rate,
            reward_base="r0",
        )
        self._set_state_and_torque(env, [-0.9, 0.6, 1.1, -0.75], torque=7.5)
        terms = env.task.xk_reward_terms(env.physics)

        self.assertEqual(env.task.reward_base, "r0")
        self.assertAlmostEqual(terms["r1"], terms["r0"], places=12)
        self.assertAlmostEqual(
            terms["lyapunov_reward"], -terms["lyapunov_normalized"], places=12
        )
        self.assertAlmostEqual(
            terms["r2"],
            terms["r0"] - eta * terms["lyapunov_rate_normalized"],
            places=12,
        )
        self.assertAlmostEqual(
            terms["r3"],
            terms["r0"]
            - eta * terms["lyapunov_rate_normalized"]
            + discount_rate * eta * terms["lyapunov_normalized"],
            places=12,
        )
        self.assertAlmostEqual(terms["reward"], terms["r3"], places=12)

    def test_r0_base_xk_source_keeps_original_vdot_and_v_terms(self):
        eta = 0.3
        discount_rate = 0.5
        env = self._env(
            "r3",
            eta=eta,
            discount_rate=discount_rate,
            reward_base="r0",
            lyapunov_rate_source="xk_closed_loop",
        )
        state = [-0.9, 0.6, 1.1, -0.75]
        self._set_state_and_torque(env, state, torque=7.5)
        positive_torque = env.task.xk_reward_terms(env.physics)
        expected = (
            positive_torque["r0"]
            - eta
            * positive_torque["xk_closed_loop_lyapunov_rate_normalized"]
            + discount_rate
            * eta
            * positive_torque["lyapunov_normalized"]
        )
        self.assertAlmostEqual(positive_torque["r3"], expected, places=12)

        self._set_state_and_torque(env, state, torque=-7.5)
        negative_torque = env.task.xk_reward_terms(env.physics)
        self.assertNotAlmostEqual(
            positive_torque["lyapunov_rate"],
            negative_torque["lyapunov_rate"],
            places=6,
        )
        self.assertAlmostEqual(
            positive_torque["r3"], negative_torque["r3"], places=12
        )

    def test_xk_closed_loop_source_substitutes_the_analytical_identity(self):
        eta = 0.35
        env = self._env(
            "r2", eta=eta, lyapunov_rate_source="xk_closed_loop"
        )
        state = np.array([-0.9, 0.6, 1.1, -0.75])
        self._set_state_and_torque(env, state, torque=7.5)
        positive_torque = env.task.xk_reward_terms(env.physics)

        expected_rate = -DEFAULT_LYAPUNOV_K_V * state[3] ** 2
        self.assertAlmostEqual(
            positive_torque["xk_closed_loop_lyapunov_rate"], expected_rate
        )
        self.assertAlmostEqual(
            positive_torque["selected_lyapunov_rate"], expected_rate
        )
        self.assertAlmostEqual(
            positive_torque["r2"],
            positive_torque["r1"]
            - eta * positive_torque["selected_lyapunov_rate_normalized"],
            places=12,
        )

        self._set_state_and_torque(env, state, torque=-7.5)
        negative_torque = env.task.xk_reward_terms(env.physics)
        self.assertNotAlmostEqual(
            positive_torque["lyapunov_rate"],
            negative_torque["lyapunov_rate"],
            places=6,
        )
        self.assertAlmostEqual(
            positive_torque["selected_lyapunov_rate"],
            negative_torque["selected_lyapunov_rate"],
            places=12,
        )
        self.assertAlmostEqual(
            positive_torque["r2"], negative_torque["r2"], places=12
        )

    def test_xk_closed_loop_r3_uses_discount_consistent_substitution(self):
        eta = 0.35
        discount_rate = 0.1
        env = self._env(
            "r3",
            eta=eta,
            discount_rate=discount_rate,
            lyapunov_rate_source="xk_closed_loop",
        )
        self._set_state_and_torque(env, [-0.9, 0.6, 1.1, -0.75], torque=7.5)
        terms = env.task.xk_reward_terms(env.physics)
        expected = (
            -(1.0 - discount_rate * eta) * terms["lyapunov_normalized"]
            - eta * terms["xk_closed_loop_lyapunov_rate_normalized"]
        )
        self.assertAlmostEqual(terms["r3"], expected, places=12)
        self.assertAlmostEqual(terms["reward"], expected, places=12)

    def test_xk_closed_loop_r1_is_a_numerically_identical_control(self):
        actual = self._env("r1")
        surrogate = self._env(
            "r1", lyapunov_rate_source="xk_closed_loop"
        )
        state = [-0.7, 0.5, 1.2, -0.4]
        self._set_state_and_torque(actual, state, torque=4.0)
        self._set_state_and_torque(surrogate, state, torque=-4.0)
        actual_terms = actual.task.xk_reward_terms(actual.physics)
        surrogate_terms = surrogate.task.xk_reward_terms(surrogate.physics)
        self.assertAlmostEqual(actual_terms["r1"], surrogate_terms["r1"], 12)
        self.assertAlmostEqual(
            actual_terms["reward"], surrogate_terms["reward"], 12
        )

    def test_actual_rate_equals_xk_identity_under_the_exact_controller(self):
        env = self._env("r2", eta=0.3)
        params = xk.AcrobotParams.from_physics(env.physics)
        gains = xk.Gains(
            k_v=DEFAULT_LYAPUNOV_K_V,
            k_d=DEFAULT_LYAPUNOV_K_D,
            k_p=DEFAULT_LYAPUNOV_K_P,
        )
        state = np.array([-0.9, 0.2, 0.5, -0.3])
        commanded_torque = xk.torque(params, gains, state)
        self.assertLess(abs(commanded_torque), 20.0)
        self._set_state_and_torque(env, state, commanded_torque)
        terms = env.task.xk_reward_terms(env.physics)
        self.assertAlmostEqual(
            terms["lyapunov_rate"],
            terms["xk_closed_loop_lyapunov_rate"],
            places=9,
        )

    def test_xk_rate_source_and_gain_validation(self):
        with self.assertRaisesRegex(ValueError, "lyapunov_rate_source"):
            swingup_xk(reward_kind="r2", eta=0.1, lyapunov_rate_source="bad")
        with self.assertRaisesRegex(ValueError, "not meaningful"):
            swingup_xk(reward_kind="r0", lyapunov_rate_source="xk_closed_loop")
        for k_v in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(k_v=k_v):
                with self.assertRaisesRegex(ValueError, "k_v"):
                    swingup_xk(reward_kind="r1", k_v=k_v)
        with self.assertRaisesRegex(ValueError, r"discount_rate \* eta < 1"):
            swingup_xk(
                reward_kind="r3",
                eta=1.0,
                discount_rate=1.0,
                lyapunov_rate_source="xk_closed_loop",
            )

    def test_reward_base_validation_and_xk_r3_domain(self):
        with self.assertRaisesRegex(ValueError, "reward_base"):
            swingup_xk(reward_kind="r1", reward_base="bad")
        with self.assertRaisesRegex(ValueError, "only meaningful"):
            swingup_xk(reward_kind="r0", reward_base="r0")

        # The historical xk-dot r3 guard comes from the coefficient on -V.
        # With r0 as the state reward, both original-V shaping terms are
        # non-negative and lambda*eta no longer needs to stay below one.
        env = self._env(
            "r3",
            eta=1.0,
            discount_rate=1.0,
            reward_base="r0",
            lyapunov_rate_source="xk_closed_loop",
        )
        self.assertEqual(env.task.reward_base, "r0")

    def test_r0_base_lower_bounds_cover_both_derivatives_and_rate_caps(self):
        rng = np.random.default_rng(2468)
        eta = 0.3
        for cap in (ELBOW_RATE_LIMIT, ELBOW_RATE_LIMIT * np.sqrt(2.0)):
            baseline_bound = reward_rate_lower_bound(
                "r0", elbow_rate_limit=cap
            )
            for source in ("actual", "xk_closed_loop"):
                for kind, discount_rate in (("r2", None), ("r3", 0.5)):
                    with self.subTest(cap=cap, source=source, kind=kind):
                        kwargs = {
                            "reward_base": "r0",
                            "lyapunov_rate_source": source,
                            "elbow_rate_limit": cap,
                        }
                        env = self._env(
                            kind,
                            eta=eta,
                            discount_rate=discount_rate,
                            **kwargs,
                        )
                        bound = reward_rate_lower_bound(
                            kind,
                            reward_base="r0",
                            eta=eta,
                            discount_rate=discount_rate,
                            lyapunov_rate_source=source,
                            elbow_rate_limit=cap,
                        )
                        if source == "xk_closed_loop":
                            self.assertAlmostEqual(
                                bound,
                                baseline_bound,
                                places=12,
                            )
                        shoulder_limit = (
                            env.task.shoulder_rate_scale_limit
                            * env.task._rate_scale
                        )
                        for _ in range(32):
                            state = [
                                rng.uniform(-np.pi, np.pi),
                                rng.uniform(
                                    -env.task.elbow_angle_limit,
                                    env.task.elbow_angle_limit,
                                ),
                                rng.uniform(-shoulder_limit, shoulder_limit),
                                rng.uniform(-cap, cap),
                            ]
                            self._set_state_and_torque(
                                env, state, rng.uniform(-20.0, 20.0)
                            )
                            selected = env.task.xk_reward_terms(env.physics)[kind]
                            self.assertGreaterEqual(
                                selected + 1e-9, bound
                            )

    def test_xk_source_lower_bounds_follow_configured_elbow_rate_cap(self):
        caps = (ELBOW_RATE_LIMIT, ELBOW_RATE_LIMIT * np.sqrt(2.0))
        expected = (
            (-134.2241858507, -117.5330953822, -141.4469850858),
            (-251.9853299767, -220.6503666635, -265.8338233067),
        )
        for cap, values in zip(caps, expected):
            with self.subTest(cap=cap):
                self.assertAlmostEqual(
                    reward_rate_lower_bound(
                        "r3",
                        eta=0.35,
                        discount_rate=0.1,
                        lyapunov_rate_source="xk_closed_loop",
                        elbow_rate_limit=cap,
                    ),
                    values[0],
                    places=8,
                )
                self.assertAlmostEqual(
                    reward_rate_lower_bound(
                        "r3",
                        eta=0.31,
                        discount_rate=0.5,
                        lyapunov_rate_source="xk_closed_loop",
                        elbow_rate_limit=cap,
                    ),
                    values[1],
                    places=8,
                )
                self.assertAlmostEqual(
                    reward_rate_lower_bound(
                        "r1",
                        lyapunov_rate_source="xk_closed_loop",
                        elbow_rate_limit=cap,
                    ),
                    values[2],
                    places=8,
                )

    def test_xk_source_lower_bound_is_below_sampled_rewards_at_both_caps(self):
        rng = np.random.default_rng(4321)
        cases = (
            (0.35, 0.1, ELBOW_RATE_LIMIT),
            (0.35, 0.1, ELBOW_RATE_LIMIT * np.sqrt(2.0)),
            (0.31, 0.5, ELBOW_RATE_LIMIT),
            (0.31, 0.5, ELBOW_RATE_LIMIT * np.sqrt(2.0)),
        )
        for eta, discount_rate, cap in cases:
            with self.subTest(eta=eta, discount_rate=discount_rate, cap=cap):
                env = self._env(
                    "r3",
                    eta=eta,
                    discount_rate=discount_rate,
                    lyapunov_rate_source="xk_closed_loop",
                    elbow_rate_limit=cap,
                )
                bound = reward_rate_lower_bound(
                    "r3",
                    eta=eta,
                    discount_rate=discount_rate,
                    lyapunov_rate_source="xk_closed_loop",
                    elbow_rate_limit=cap,
                )
                shoulder_limit = (
                    env.task.shoulder_rate_scale_limit * env.task._rate_scale
                )
                for _ in range(128):
                    state = [
                        rng.uniform(-np.pi, np.pi),
                        rng.uniform(
                            -env.task.elbow_angle_limit,
                            env.task.elbow_angle_limit,
                        ),
                        rng.uniform(-shoulder_limit, shoulder_limit),
                        rng.uniform(-cap, cap),
                    ]
                    self._set_state_and_torque(
                        env, state, rng.uniform(-20.0, 20.0)
                    )
                    selected = env.task.xk_reward_terms(env.physics)["r3"]
                    self.assertGreaterEqual(selected + 1e-9, bound)

    def test_diagnostic_lower_bounds_match_each_reward(self):
        cases = (
            ("r0", None, None, -143.5810893656),
            ("r1", None, None, -141.4469850858),
            ("r2", 0.0, None, -141.4469850858),
            ("r2", 0.01, None, -145.4202218797),
            ("r2", 0.03, None, -153.3666954674),
            ("r2", 0.1, None, -181.1793530244),
            ("r2", 0.3, None, -260.6440889017),
            ("r2", 1.0, None, -538.7706644721),
            ("r3", 0.0, 0.1, -141.4469850858),
            ("r3", 0.01, 0.1, -145.2787748946),
            ("r3", 0.03, 0.1, -152.9423545121),
            ("r3", 0.1, 0.1, -179.7648831736),
            ("r3", 0.3, 0.1, -256.4006793491),
            ("r3", 1.0, 0.1, -524.6259659635),
            ("r3", 0.0, 0.5, -141.4469850858),
            ("r3", 0.01, 0.5, -144.7129869542),
            ("r3", 0.03, 0.5, -151.2449906911),
            ("r3", 0.1, 0.5, -174.1070037702),
            ("r3", 0.3, 0.5, -239.4270411388),
            ("r3", 1.0, 0.5, -468.0471719292),
        )
        for kind, eta, discount_rate, expected in cases:
            with self.subTest(kind=kind):
                self.assertAlmostEqual(
                    reward_rate_lower_bound(
                        kind, eta=eta, discount_rate=discount_rate
                    ),
                    expected,
                    places=8,
                )

    def test_task_always_resolves_the_lower_bound_as_its_failure_rate(self):
        cases = (
            {"reward_kind": "r0"},
            {"reward_kind": "r1"},
            {"reward_kind": "r1", "reward_base": "r0"},
            {"reward_kind": "r2", "eta": 0.1},
            {
                "reward_kind": "r2",
                "reward_base": "r0",
                "eta": 0.1,
                "lyapunov_rate_source": "xk_closed_loop",
            },
            {
                "reward_kind": "r3",
                "eta": 0.0,
                "discount_rate": 0.5,
                "reward_base": "r0",
                "lyapunov_rate_source": "actual",
            },
            {
                "reward_kind": "r3",
                "eta": 0.1,
                "discount_rate": 0.1,
                "reward_base": "lyapunov",
                "lyapunov_rate_source": "xk_closed_loop",
            },
            {
                "reward_kind": "r3",
                "eta": 0.01,
                "discount_rate": 0.1,
                "reward_base": "r0",
                "lyapunov_rate_source": "actual",
            },
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                reward_kind = kwargs.pop("reward_kind")
                env = self._env(reward_kind, **kwargs)
                expected = reward_rate_lower_bound(
                    reward_kind, **kwargs
                )
                self.assertAlmostEqual(
                    env.task.failure_reward_rate, expected, places=12
                )
                self.assertEqual(
                    env.task.failure_reward_rate_source,
                    LOWER_BOUND_TERMINATION_REWARD_SOURCE,
                )

    def test_diagnostic_lower_bound_is_below_sampled_admissible_rewards(self):
        rng = np.random.default_rng(1234)
        cases = (
            ("r0", None, None),
            ("r1", None, None),
            ("r2", 0.1, None),
            ("r2", 1.0, None),
            ("r3", 0.1, 0.1),
            ("r3", 1.0, 0.1),
        )
        for kind, eta, discount_rate in cases:
            with self.subTest(kind=kind, eta=eta):
                env = self._env(
                    kind, eta=eta, discount_rate=discount_rate
                )
                bound = reward_rate_lower_bound(
                    kind, eta=eta, discount_rate=discount_rate
                )
                shoulder_limit = (
                    SHOULDER_RATE_SCALE_LIMIT * env.task._rate_scale
                )
                for _ in range(128):
                    state = [
                        rng.uniform(-np.pi, np.pi),
                        rng.uniform(-ELBOW_ANGLE_LIMIT, ELBOW_ANGLE_LIMIT),
                        rng.uniform(-shoulder_limit, shoulder_limit),
                        rng.uniform(-ELBOW_RATE_LIMIT, ELBOW_RATE_LIMIT),
                    ]
                    torque = rng.uniform(-20.0, 20.0)
                    self._set_state_and_torque(env, state, torque)
                    selected = env.task.xk_reward_terms(env.physics)[kind]
                    self.assertGreaterEqual(selected + 1e-9, bound)

    def test_actual_rate_lower_bounds_cover_the_higher_elbow_rate_cap(self):
        rng = np.random.default_rng(9876)
        cap = ELBOW_RATE_LIMIT * np.sqrt(2.0)
        cases = (
            ("r0", None, None),
            ("r1", None, None),
            ("r2", 0.3, None),
            ("r3", 0.3, 0.1),
        )
        for kind, eta, discount_rate in cases:
            with self.subTest(kind=kind):
                env = self._env(
                    kind,
                    eta=eta,
                    discount_rate=discount_rate,
                    elbow_rate_limit=cap,
                )
                bound = reward_rate_lower_bound(
                    kind,
                    eta=eta,
                    discount_rate=discount_rate,
                    elbow_rate_limit=cap,
                )
                shoulder_limit = (
                    env.task.shoulder_rate_scale_limit * env.task._rate_scale
                )
                for _ in range(64):
                    state = [
                        rng.uniform(-np.pi, np.pi),
                        rng.uniform(
                            -env.task.elbow_angle_limit,
                            env.task.elbow_angle_limit,
                        ),
                        rng.uniform(-shoulder_limit, shoulder_limit),
                        rng.uniform(-cap, cap),
                    ]
                    self._set_state_and_torque(
                        env, state, rng.uniform(-20.0, 20.0)
                    )
                    selected = env.task.xk_reward_terms(env.physics)[kind]
                    self.assertGreaterEqual(selected + 1e-9, bound)

    def test_r0_wraps_elbow_but_r1_penalizes_winding(self):
        env = self._env("r1")
        left = np.array([-0.8, -np.pi + 0.2, 0.6, -0.1])
        right = left.copy()
        right[1] += 2.0 * np.pi
        self._set_state_and_torque(env, left, torque=0.0)
        left_terms = env.task.xk_reward_terms(env.physics)
        self._set_state_and_torque(env, right, torque=0.0)
        right_terms = env.task.xk_reward_terms(env.physics)
        self.assertAlmostEqual(left_terms["r0"], right_terms["r0"], places=10)
        self.assertNotAlmostEqual(left_terms["r1"], right_terms["r1"], places=3)

    def test_reward_kind_does_not_change_the_plant_transition(self):
        wrappers = [
            DMCContinuousEnv(
                "acrobot",
                "swingup-xk",
                seed=17,
                raw_state_obs=True,
                dt=0.002,
                physics_dt=0.002,
                task_kwargs={
                    "release_start": True,
                    "reward_kind": kind,
                    **({"eta": 0.2} if kind in ("r2", "r3") else {}),
                    **({"discount_rate": 0.1} if kind == "r3" else {}),
                },
            )
            for kind in ("r0", "r1", "r2", "r3")
        ]
        starts = [env.reset(seed=17)[0] for env in wrappers]
        for start in starts[1:]:
            np.testing.assert_allclose(start, starts[0], atol=0.0)
        transitions = [env.step_dt(np.array([0.25], dtype=np.float32)) for env in wrappers]
        for transition in transitions[1:]:
            np.testing.assert_allclose(transition[4], transitions[0][4], atol=0.0)
            self.assertEqual(transition[5], transitions[0][5])
        self.assertEqual(len({round(float(t[3]), 8) for t in transitions}), 4)

    def test_environment_emits_the_endpoint_reward(self):
        env = DMCContinuousEnv(
            "acrobot",
            "swingup-xk",
            seed=3,
            raw_state_obs=True,
            dt=0.002,
            physics_dt=0.002,
            task_kwargs={"reward_kind": "r2", "eta": 0.15},
        )
        env.reset(seed=3)
        transition = env.step_dt(np.array([-0.2], dtype=np.float32))
        endpoint_terms = env._env.task.xk_reward_terms(env._env.physics)
        self.assertAlmostEqual(transition[3], endpoint_terms["reward"], places=10)

    def test_upright_rest_is_inside_the_reward_independent_tube(self):
        env = self._env("r2", eta=0.1)
        self._set_state_and_torque(
            env, [UPRIGHT_SHOULDER, 0.0, 0.0, 0.0], torque=0.0
        )
        self.assertEqual(
            env.task.xk_diagnostic_terms(env.physics)["in_homoclinic_tube"], 1.0
        )
        self._set_state_and_torque(
            env,
            [
                UPRIGHT_SHOULDER,
                np.pi * (HOMOCLINIC_ANGLE_TOLERANCE + 1e-3),
                0.0,
                0.0,
            ],
            torque=0.0,
        )
        self.assertEqual(
            env.task.xk_diagnostic_terms(env.physics)["in_homoclinic_tube"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
