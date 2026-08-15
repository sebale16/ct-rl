"""Exact-seed contract for Acrobot-XK callback evaluations."""

import unittest

import numpy as np
import torch as th

from benchmarks.run_ct_rl import (
    ACROBOT_XK_ENV_ID,
    ACROBOT_XK_EVAL_SEEDS,
    _align_acrobot_xk_eval_termination_limits,
    _configure_acrobot_xk_start_distributions,
    _primary_eval_schedule,
    _resolved_eval_mode,
)
from evaluations.evaluation_helpers import evaluate_policy_per_episode


class _OneStepEnv:
    def __init__(self):
        self.reset_seeds = []

    def reset(self, *, seed=None):
        self.reset_seeds.append(seed)
        return np.zeros(1, dtype=np.float32), {}

    def step_dt(self, action):
        obs = np.zeros(1, dtype=np.float32)
        return obs, 0.0, action, 0.0, obs, 0.5, True, False, {}


class _ZeroModel:
    device = th.device("cpu")

    @staticmethod
    def act(obs, deterministic=True):
        del deterministic
        return th.zeros((obs.shape[0], 1), dtype=th.float32), None


class TestAcrobotXKEvalProtocol(unittest.TestCase):
    def test_runner_resolves_the_fixed_mode_and_seed_schedule(self):
        self.assertEqual(_resolved_eval_mode(ACROBOT_XK_ENV_ID, None), "xk_eval")
        self.assertEqual(
            _resolved_eval_mode(ACROBOT_XK_ENV_ID, "xk_eval"), "xk_eval"
        )
        with self.assertRaisesRegex(ValueError, "requires --eval_mode"):
            _resolved_eval_mode(ACROBOT_XK_ENV_ID, "xk_r0")

        reset_seed, count, episode_seeds = _primary_eval_schedule(
            ACROBOT_XK_ENV_ID, training_seed=7, requested_episodes=3
        )
        self.assertIsNone(reset_seed)
        self.assertEqual(count, 32)
        self.assertEqual(episode_seeds, ACROBOT_XK_EVAL_SEEDS)
        self.assertEqual(episode_seeds, tuple(range(20000, 20032)))

    def test_uniform_training_start_keeps_evaluation_near_hanging(self):
        train = {
            "dt": 0.002,
            "task_kwargs": {
                "reward_kind": "r2",
                "eta": 0.1,
                "uniform_start": False,
                "paper_start": False,
                "release_start": True,
            },
        }
        fixed_eval = {
            "dt": 0.001,
            "task_kwargs": {
                "reward_kind": "r0",
                "uniform_start": True,
                "paper_start": False,
                "release_start": False,
            },
        }
        original_train = {
            "dt": 0.002,
            "task_kwargs": {
                "reward_kind": "r2",
                "eta": 0.1,
                "uniform_start": False,
                "paper_start": False,
                "release_start": True,
            },
        }
        original_eval = {
            "dt": 0.001,
            "task_kwargs": {
                "reward_kind": "r0",
                "uniform_start": True,
                "paper_start": False,
                "release_start": False,
            },
        }

        configured_train, configured_eval = (
            _configure_acrobot_xk_start_distributions(
                ACROBOT_XK_ENV_ID,
                train,
                fixed_eval,
                uniform_training_start=True,
            )
        )

        self.assertEqual(train, original_train)
        self.assertEqual(fixed_eval, original_eval)
        self.assertIsNot(configured_train, train)
        self.assertIsNot(configured_train["task_kwargs"], train["task_kwargs"])
        self.assertIsNot(configured_eval, fixed_eval)
        self.assertIsNot(
            configured_eval["task_kwargs"], fixed_eval["task_kwargs"]
        )
        self.assertEqual(configured_train["task_kwargs"]["reward_kind"], "r2")
        self.assertEqual(configured_train["task_kwargs"]["eta"], 0.1)
        self.assertTrue(configured_train["task_kwargs"]["uniform_start"])
        self.assertFalse(configured_train["task_kwargs"]["paper_start"])
        self.assertFalse(configured_train["task_kwargs"]["release_start"])
        self.assertTrue(configured_eval["task_kwargs"]["release_start"])
        self.assertFalse(configured_eval["task_kwargs"]["uniform_start"])
        self.assertFalse(configured_eval["task_kwargs"]["paper_start"])

    def test_start_override_rejects_non_xk_uniform_training(self):
        train = {"task_kwargs": {"uniform_start": False}}
        fixed_eval = {"task_kwargs": {"uniform_start": True}}

        same_train, same_eval = _configure_acrobot_xk_start_distributions(
            "cartpole-swingup",
            train,
            fixed_eval,
            uniform_training_start=False,
        )
        self.assertIs(same_train, train)
        self.assertIs(same_eval, fixed_eval)
        with self.assertRaisesRegex(ValueError, "acrobot-swingup-xk"):
            _configure_acrobot_xk_start_distributions(
                "cartpole-swingup",
                train,
                fixed_eval,
                uniform_training_start=True,
            )

    def test_disabled_uniform_start_leaves_xk_training_reset_unchanged(self):
        train = {
            "task_kwargs": {
                "release_start": True,
                "release_angle_range": (0.1, 0.2),
            }
        }
        fixed_eval = {"task_kwargs": {"uniform_start": True}}

        configured_train, configured_eval = (
            _configure_acrobot_xk_start_distributions(
                ACROBOT_XK_ENV_ID,
                train,
                fixed_eval,
                uniform_training_start=False,
            )
        )

        self.assertEqual(configured_train, train)
        self.assertTrue(configured_train["task_kwargs"]["release_start"])
        self.assertEqual(
            configured_train["task_kwargs"]["release_angle_range"],
            (0.1, 0.2),
        )
        self.assertTrue(configured_eval["task_kwargs"]["release_start"])
        self.assertFalse(configured_eval["task_kwargs"]["uniform_start"])
        self.assertFalse(configured_eval["task_kwargs"]["paper_start"])

    def test_callback_eval_preserves_each_arms_termination_envelope(self):
        train = {
            "task_kwargs": {
                "reward_kind": "r3",
                "elbow_angle_limit": 4.0 * np.pi,
                "elbow_rate_limit": 4.0 * np.pi * np.sqrt(2.0),
                "shoulder_rate_scale_limit": 2.0,
            }
        }
        fixed_eval = {
            "dt": 0.001,
            "task_kwargs": {
                "reward_kind": "r0",
                "release_start": True,
            },
        }

        aligned = _align_acrobot_xk_eval_termination_limits(
            ACROBOT_XK_ENV_ID, train, fixed_eval
        )

        self.assertIsNot(aligned, fixed_eval)
        self.assertEqual(aligned["task_kwargs"]["reward_kind"], "r0")
        self.assertTrue(aligned["task_kwargs"]["release_start"])
        self.assertEqual(
            aligned["task_kwargs"]["elbow_rate_limit"],
            4.0 * np.pi * np.sqrt(2.0),
        )
        self.assertEqual(
            aligned["task_kwargs"]["elbow_angle_limit"], 4.0 * np.pi
        )
        self.assertEqual(
            aligned["task_kwargs"]["shoulder_rate_scale_limit"], 2.0
        )
        self.assertNotIn("elbow_rate_limit", fixed_eval["task_kwargs"])

    def test_callback_eval_does_not_change_other_environments(self):
        eval_kwargs = {"task_kwargs": {"reward_kind": "r0"}}
        result = _align_acrobot_xk_eval_termination_limits(
            "cartpole-swingup", {"task_kwargs": {}}, eval_kwargs
        )
        self.assertIs(result, eval_kwargs)

    def test_callback_eval_forces_release_start_without_configured_caps(self):
        train = {"task_kwargs": {"reward_kind": "r0"}}
        fixed_eval = {
            "dt": 0.001,
            "task_kwargs": {
                "reward_kind": "r0",
                "uniform_start": True,
                "paper_start": False,
                "release_start": False,
            },
        }

        aligned = _align_acrobot_xk_eval_termination_limits(
            ACROBOT_XK_ENV_ID, train, fixed_eval
        )

        self.assertIsNot(aligned, fixed_eval)
        self.assertTrue(aligned["task_kwargs"]["release_start"])
        self.assertFalse(aligned["task_kwargs"]["uniform_start"])
        self.assertFalse(aligned["task_kwargs"]["paper_start"])
        self.assertTrue(fixed_eval["task_kwargs"]["uniform_start"])
        self.assertFalse(fixed_eval["task_kwargs"]["release_start"])

    def test_evaluator_reseeds_each_episode_and_does_not_add_a_final_reset(self):
        env = _OneStepEnv()
        seeds = (20000, 20001, 20002)

        returns, lengths = evaluate_policy_per_episode(
            _ZeroModel(),
            env,
            n_eval_episodes=len(seeds),
            episode_seeds=seeds,
        )

        self.assertEqual(env.reset_seeds, list(seeds))
        self.assertEqual(returns, [0.0, 0.0, 0.0])
        self.assertEqual(lengths, [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
