"""Runaway-termination contracts for the Xin--Kaneda Acrobot task."""

import os
import unittest

os.environ.setdefault("MUJOCO_GL", "disable")

import numpy as np

from environment.acrobot_xk import (
    DEFAULT_TORQUE_LIMIT,
    ELBOW_ANGLE_LIMIT,
    ELBOW_RATE_LIMIT,
    LOWER_BOUND_TERMINATION_REWARD_SOURCE,
    SHOULDER_RATE_SCALE_LIMIT,
    TERMINATION_ELBOW_ANGLE,
    TERMINATION_ELBOW_RATE,
    TERMINATION_REWARD_SOURCE,
    TERMINATION_SHOULDER_RATE,
    reward_rate_lower_bound,
    swingup_xk,
)
from environment.dmc import DMCContinuousEnv


class TestAcrobotXKTermination(unittest.TestCase):
    def setUp(self):
        self.env = swingup_xk(
            random=0,
            release_start=True,
            angle_noise=0.0,
            velocity_noise=0.0,
        )
        self.env.reset()

    def _set_state(self, q2=0.0, qd1=0.0, qd2=0.0):
        self.env.physics.data.qpos[:] = [-0.5 * np.pi, q2]
        self.env.physics.data.qvel[:] = [qd1, qd2]
        self.env.physics.forward()

    def _terminal_transition(self, task_kwargs, state, *, seed):
        env = DMCContinuousEnv(
            "acrobot",
            "swingup-xk",
            seed=seed,
            raw_state_obs=True,
            dt=0.001,
            physics_dt=0.001,
            episode_duration=1.0,
            task_kwargs=task_kwargs,
        )
        self.addCleanup(env.close)
        env.reset(seed=seed)
        env._env.physics.data.qpos[:] = state[:2]
        env._env.physics.data.qvel[:] = state[2:]
        env._env.physics.forward()
        return env.step_dt(np.zeros(1, dtype=np.float32))

    def test_default_gear_is_twenty_newton_metres(self):
        gear = float(np.asarray(self.env.physics.model.actuator_gear)[0, 0])
        self.assertEqual(DEFAULT_TORQUE_LIMIT, 20.0)
        self.assertEqual(gear, DEFAULT_TORQUE_LIMIT)

    def test_elbow_angle_limit_allows_two_full_turns(self):
        self.assertEqual(ELBOW_ANGLE_LIMIT, 4.0 * np.pi)
        for sign in (-1.0, 1.0):
            with self.subTest(sign=sign):
                self._set_state(q2=sign * (2.0 * np.pi + 0.2))
                self.assertIsNone(
                    self.env.task.get_termination(self.env.physics)
                )
                self.assertIsNone(self.env.task.last_termination_reason)

    def test_elbow_rate_limit_allows_speeds_beyond_two_pi(self):
        self.assertEqual(ELBOW_RATE_LIMIT, 4.0 * np.pi)
        for sign in (-1.0, 1.0):
            with self.subTest(sign=sign):
                self._set_state(qd2=sign * (2.0 * np.pi + 0.2))
                self.assertIsNone(
                    self.env.task.get_termination(self.env.physics)
                )
                self.assertIsNone(self.env.task.last_termination_reason)

    def test_each_limit_terminates_at_either_sign(self):
        omega_s = float(self.env.task._rate_scale)
        cases = (
            ("elbow angle", "q2", ELBOW_ANGLE_LIMIT, TERMINATION_ELBOW_ANGLE),
            ("elbow rate", "qd2", ELBOW_RATE_LIMIT, TERMINATION_ELBOW_RATE),
            (
                "shoulder rate",
                "qd1",
                SHOULDER_RATE_SCALE_LIMIT * omega_s,
                TERMINATION_SHOULDER_RATE,
            ),
        )
        for label, field, limit, reason in cases:
            for sign in (-1.0, 1.0):
                with self.subTest(limit=label, sign=sign):
                    values = {"q2": 0.0, "qd1": 0.0, "qd2": 0.0}
                    values[field] = sign * limit
                    self._set_state(**values)
                    self.assertEqual(
                        self.env.task.get_termination(self.env.physics), 0.0
                    )
                    self.assertEqual(self.env.task.last_termination_reason, reason)

    def test_values_just_inside_every_limit_continue(self):
        omega_s = float(self.env.task._rate_scale)
        self._set_state(
            q2=np.nextafter(ELBOW_ANGLE_LIMIT, 0.0),
            qd1=np.nextafter(SHOULDER_RATE_SCALE_LIMIT * omega_s, 0.0),
            qd2=np.nextafter(ELBOW_RATE_LIMIT, 0.0),
        )
        self.assertIsNone(self.env.task.get_termination(self.env.physics))
        self.assertIsNone(self.env.task.last_termination_reason)

    def test_custom_limits_are_instance_kwargs_at_exact_boundaries(self):
        limits = {
            "elbow_angle_limit": 1.5,
            "elbow_rate_limit": 2.5,
            "shoulder_rate_scale_limit": 0.75,
        }
        env = swingup_xk(random=1, release_start=True, **limits)
        env.reset()
        task = env.task
        self.assertEqual(task.elbow_angle_limit, 1.5)
        self.assertEqual(task.elbow_rate_limit, 2.5)
        self.assertEqual(task.shoulder_rate_scale_limit, 0.75)
        shoulder_limit = 0.75 * task._rate_scale

        cases = (
            ([0.0, 1.5, 0.0, 0.0], TERMINATION_ELBOW_ANGLE),
            ([0.0, 0.0, 0.0, -2.5], TERMINATION_ELBOW_RATE),
            ([0.0, 0.0, shoulder_limit, 0.0], TERMINATION_SHOULDER_RATE),
        )
        for state, reason in cases:
            with self.subTest(reason=reason):
                env.physics.data.qpos[:] = state[:2]
                env.physics.data.qvel[:] = state[2:]
                env.physics.forward()
                self.assertEqual(task.get_termination(env.physics), 0.0)
                self.assertEqual(task.last_termination_reason, reason)

        env.physics.data.qpos[:] = [
            0.0,
            np.nextafter(task.elbow_angle_limit, 0.0),
        ]
        env.physics.data.qvel[:] = [
            np.nextafter(shoulder_limit, 0.0),
            np.nextafter(task.elbow_rate_limit, 0.0),
        ]
        env.physics.forward()
        self.assertIsNone(task.get_termination(env.physics))

    def test_two_tasks_keep_independent_elbow_rate_limits(self):
        low = swingup_xk(random=2, elbow_rate_limit=2.0)
        high = swingup_xk(random=3, elbow_rate_limit=4.0)
        low.reset()
        high.reset()
        for env in (low, high):
            env.physics.data.qpos[:] = [0.0, 0.0]
            env.physics.data.qvel[:] = [0.0, 3.0]
            env.physics.forward()
        self.assertEqual(low.task.get_termination(low.physics), 0.0)
        self.assertIsNone(high.task.get_termination(high.physics))

    def test_termination_limit_kwargs_must_be_positive_and_finite(self):
        names = (
            "elbow_angle_limit",
            "elbow_rate_limit",
            "shoulder_rate_scale_limit",
        )
        for name in names:
            for value in (0.0, -1.0, float("nan"), float("inf")):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        swingup_xk(**{name: value})

    def test_elbow_angle_limit_uses_the_unwrapped_coordinate(self):
        self._set_state(q2=ELBOW_ANGLE_LIMIT + 0.2)
        self.assertEqual(self.env.task.get_termination(self.env.physics), 0.0)
        self.assertEqual(
            self.env.task.last_termination_reason, TERMINATION_ELBOW_ANGLE
        )

    def test_wrapper_emits_terminal_discount_and_reason_then_reset_clears_it(self):
        env = DMCContinuousEnv(
            "acrobot",
            "swingup-xk",
            seed=4,
            raw_state_obs=True,
            dt=0.001,
            physics_dt=0.001,
            episode_duration=1.0,
            task_kwargs={"release_start": True},
        )
        self.addCleanup(env.close)
        env.reset(seed=4)
        env._env.physics.data.qpos[1] = ELBOW_ANGLE_LIMIT + 0.2
        env._env.physics.data.qvel[:] = 0.0
        env._env.physics.forward()

        transition = env.step_dt(np.zeros(1, dtype=np.float32))
        reward, terminated, truncated, info = (
            transition[3],
            transition[6],
            transition[7],
            transition[8],
        )
        self.assertTrue(np.isfinite(reward))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(float(info["discount"]), 0.0)
        self.assertEqual(
            info["acrobot_xk_termination_reason"], TERMINATION_ELBOW_ANGLE
        )
        self.assertEqual(info["absorbing_failure"], 1.0)
        expected_failure_rate = info["acrobot_xk_reward"]
        self.assertAlmostEqual(
            info["absorbing_failure_reward_rate"], expected_failure_rate
        )
        self.assertAlmostEqual(reward, expected_failure_rate)
        self.assertEqual(
            info["absorbing_failure_reward_rate_source"],
            TERMINATION_REWARD_SOURCE,
        )
        self.assertNotIn("acrobot_xk_unpenalized_reward", info)
        self.assertAlmostEqual(
            info["absorbing_failure_remaining_seconds"], 0.999, places=12
        )

        _, reset_info = env.reset(seed=4)
        self.assertIsNone(env._env.task.last_termination_reason)
        self.assertNotIn("acrobot_xk_termination_reason", reset_info)

    def test_each_reward_uses_its_selected_cap_endpoint_rate(self):
        cases = (
            {"reward_kind": "r0"},
            {"reward_kind": "r1", "reward_base": "r0"},
            {
                "reward_kind": "r2",
                "reward_base": "r0",
                "eta": 0.3,
                "lyapunov_rate_source": "actual",
            },
            {
                "reward_kind": "r2",
                "reward_base": "r0",
                "eta": 0.3,
                "lyapunov_rate_source": "xk_closed_loop",
            },
            {
                "reward_kind": "r3",
                "reward_base": "r0",
                "eta": 0.0,
                "discount_rate": 0.5,
                "lyapunov_rate_source": "actual",
            },
            {
                "reward_kind": "r3",
                "reward_base": "lyapunov",
                "eta": 0.1,
                "discount_rate": 0.1,
                "lyapunov_rate_source": "xk_closed_loop",
            },
        )
        terminal_state = np.array([0.5 * np.pi, ELBOW_ANGLE_LIMIT, 0.0, 0.0])
        for seed, task_kwargs in enumerate(cases, start=20):
            with self.subTest(task_kwargs=task_kwargs):
                transition = self._terminal_transition(
                    task_kwargs, terminal_state, seed=seed
                )
                reward, terminated, truncated, info = (
                    transition[3],
                    transition[6],
                    transition[7],
                    transition[8],
                )
                self.assertTrue(terminated)
                self.assertFalse(truncated)
                self.assertAlmostEqual(reward, info["acrobot_xk_reward"])
                self.assertAlmostEqual(
                    reward, info["absorbing_failure_reward_rate"]
                )

    def test_unsafe_r3_replaces_positive_endpoint_with_lower_bound(self):
        upright = [0.5 * np.pi, ELBOW_ANGLE_LIMIT, 0.0, 0.0]
        hanging = [-0.5 * np.pi, ELBOW_ANGLE_LIMIT, 0.0, 0.0]
        upright_r0 = self._terminal_transition(
            {"reward_kind": "r0"}, upright, seed=30
        )
        hanging_r0 = self._terminal_transition(
            {"reward_kind": "r0"}, hanging, seed=31
        )
        self.assertNotAlmostEqual(upright_r0[3], hanging_r0[3], places=6)

        task_kwargs = {
            "reward_kind": "r3",
            "reward_base": "r0",
            "eta": 1.0,
            "discount_rate": 0.5,
            "lyapunov_rate_source": "xk_closed_loop",
        }
        bounded_r3 = self._terminal_transition(
            task_kwargs,
            upright,
            seed=32,
        )
        info = bounded_r3[8]
        self.assertGreater(info["acrobot_xk_reward"], 0.0)
        self.assertEqual(
            info["acrobot_xk_unpenalized_reward"], info["acrobot_xk_reward"]
        )
        expected = reward_rate_lower_bound(
            "r3",
            reward_base="r0",
            eta=1.0,
            discount_rate=0.5,
            lyapunov_rate_source="xk_closed_loop",
        )
        self.assertLess(expected, 0.0)
        self.assertAlmostEqual(
            bounded_r3[3], expected
        )
        self.assertAlmostEqual(
            info["absorbing_failure_reward_rate"], expected
        )
        self.assertEqual(
            info["absorbing_failure_reward_rate_source"],
            LOWER_BOUND_TERMINATION_REWARD_SOURCE,
        )
        bounded_hanging = self._terminal_transition(
            task_kwargs, hanging, seed=34
        )
        self.assertNotAlmostEqual(
            info["acrobot_xk_reward"],
            bounded_hanging[8]["acrobot_xk_reward"],
            places=6,
        )
        self.assertAlmostEqual(bounded_hanging[3], expected)

        actual_kwargs = {
            "reward_kind": "r3",
            "reward_base": "r0",
            "eta": 0.01,
            "discount_rate": 0.1,
            "lyapunov_rate_source": "actual",
        }
        actual_r3 = self._terminal_transition(
            actual_kwargs, upright, seed=33
        )
        actual_expected = reward_rate_lower_bound(
            "r3",
            reward_base="r0",
            eta=0.01,
            discount_rate=0.1,
            lyapunov_rate_source="actual",
        )
        self.assertGreater(actual_r3[8]["acrobot_xk_reward"], 0.0)
        self.assertAlmostEqual(actual_r3[3], actual_expected)
        self.assertAlmostEqual(
            actual_r3[8]["absorbing_failure_reward_rate"], actual_expected
        )

    def test_ordinary_horizon_truncation_has_no_failure_continuation(self):
        env = DMCContinuousEnv(
            "acrobot",
            "swingup-xk",
            seed=5,
            raw_state_obs=True,
            dt=0.001,
            physics_dt=0.001,
            episode_duration=0.001,
            task_kwargs={"release_start": True},
        )
        self.addCleanup(env.close)
        env.reset(seed=5)
        transition = env.step_dt(np.zeros(1, dtype=np.float32))
        self.assertFalse(transition[6])
        self.assertTrue(transition[7])
        self.assertNotIn("absorbing_failure", transition[8])
        self.assertNotIn("absorbing_failure_reward_rate", transition[8])
        self.assertNotIn("absorbing_failure_reward_rate_source", transition[8])
        self.assertNotIn("absorbing_failure_remaining_seconds", transition[8])

    def test_state_cap_takes_precedence_over_simultaneous_step_limit(self):
        env = DMCContinuousEnv(
            "acrobot",
            "swingup-xk",
            seed=6,
            raw_state_obs=True,
            dt=0.001,
            physics_dt=0.001,
            max_steps=1,
            episode_duration=0.001,
            task_kwargs={"release_start": True},
        )
        self.addCleanup(env.close)
        env.reset(seed=6)
        env._env.physics.data.qpos[1] = ELBOW_ANGLE_LIMIT + 0.2
        env._env.physics.data.qvel[:] = 0.0
        env._env.physics.forward()

        transition = env.step_dt(np.zeros(1, dtype=np.float32))
        self.assertTrue(transition[6])
        self.assertFalse(transition[7])
        self.assertEqual(float(transition[8]["discount"]), 0.0)
        self.assertEqual(transition[8]["absorbing_failure"], 1.0)
        self.assertEqual(
            transition[8]["absorbing_failure_remaining_seconds"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
