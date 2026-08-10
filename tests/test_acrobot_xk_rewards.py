"""Reward-contract tests for the CT-SAC Acrobot Xin--Kaneda task."""

import os
import unittest

os.environ.setdefault("MUJOCO_GL", "disable")

import numpy as np

from controllers import xin_kaneda as xk
from environment.acrobot_xk import (
    DEFAULT_LYAPUNOV_K_D,
    DEFAULT_LYAPUNOV_K_P,
    HOMOCLINIC_ANGLE_TOLERANCE,
    UPRIGHT_SHOULDER,
    swingup_xk,
)
from environment.dmc import DMCContinuousEnv


class TestAcrobotXKRewards(unittest.TestCase):
    def _env(self, reward_kind="r0", *, eta=None):
        env = swingup_xk(
            reward_kind=reward_kind,
            eta=eta,
            angle_noise=0.0,
            velocity_noise=0.0,
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

    def test_r1_is_exactly_minus_the_published_lyapunov_function(self):
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
        self.assertAlmostEqual(terms["reward"], -expected, places=9)

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
            terms["r2"],
            terms["r1"] - 0.3 * terms["lyapunov_rate"],
            places=10,
        )

    def test_eta_is_a_required_nonnegative_r2_parameter(self):
        with self.assertRaises(ValueError):
            swingup_xk(reward_kind="r2")
        with self.assertRaises(ValueError):
            swingup_xk(reward_kind="r2", eta=-0.1)
        with self.assertRaises(ValueError):
            swingup_xk(reward_kind="r1", eta=0.1)

        env = self._env("r2", eta=0.0)
        self._set_state_and_torque(env, [-1.0, 0.2, 0.4, -0.3], torque=4.0)
        terms = env.task.xk_reward_terms(env.physics)
        self.assertAlmostEqual(terms["r2"], terms["r1"], places=12)

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
                    **({"eta": 0.2} if kind == "r2" else {}),
                },
            )
            for kind in ("r0", "r1", "r2")
        ]
        starts = [env.reset(seed=17)[0] for env in wrappers]
        for start in starts[1:]:
            np.testing.assert_allclose(start, starts[0], atol=0.0)
        transitions = [env.step_dt(np.array([0.25], dtype=np.float32)) for env in wrappers]
        for transition in transitions[1:]:
            np.testing.assert_allclose(transition[4], transitions[0][4], atol=0.0)
            self.assertEqual(transition[5], transitions[0][5])
        self.assertEqual(len({round(float(t[3]), 8) for t in transitions}), 3)

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
