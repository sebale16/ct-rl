"""Runaway-termination contracts for the Xin--Kaneda Acrobot task."""

import os
import unittest

os.environ.setdefault("MUJOCO_GL", "disable")

import numpy as np

from environment.acrobot_xk import (
    DEFAULT_TORQUE_LIMIT,
    ELBOW_ANGLE_LIMIT,
    ELBOW_RATE_LIMIT,
    SHOULDER_RATE_SCALE_LIMIT,
    TERMINATION_ELBOW_ANGLE,
    TERMINATION_ELBOW_RATE,
    TERMINATION_SHOULDER_RATE,
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
        expected_failure_rate = env._env.task.failure_reward_rate
        self.assertAlmostEqual(
            info["absorbing_failure_reward_rate"], expected_failure_rate
        )
        self.assertAlmostEqual(reward, expected_failure_rate)
        self.assertIn("acrobot_xk_reward", info)
        self.assertEqual(
            info["acrobot_xk_unpenalized_reward"], info["acrobot_xk_reward"]
        )
        self.assertAlmostEqual(
            info["absorbing_failure_remaining_seconds"], 0.999, places=12
        )

        _, reset_info = env.reset(seed=4)
        self.assertIsNone(env._env.task.last_termination_reason)
        self.assertNotIn("acrobot_xk_termination_reason", reset_info)

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
