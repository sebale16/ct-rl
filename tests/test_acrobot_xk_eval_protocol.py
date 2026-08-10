"""Exact-seed contract for Acrobot-XK callback evaluations."""

import unittest

import numpy as np
import torch as th

from benchmarks.run_ct_rl import (
    ACROBOT_XK_ENV_ID,
    ACROBOT_XK_EVAL_SEEDS,
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
