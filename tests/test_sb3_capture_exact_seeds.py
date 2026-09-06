"""Unit contract for common.sb3_callbacks.evaluate_sb3_policy_at_fixed_seeds.

Scripted single (non-vec) env + policy, mirroring
test_sb3_sustained_capture_callback.py's vectorized fixtures for
evaluate_sb3_policy_with_capture -- the episode data below reuses that
file's already-verified success/duration numbers (a 1-step, dt_used=1.0
episode succeeds; a 2-step, dt_used=0.6-each episode from a cold start
tops out at 0.6s and fails) so this test isn't the first thing computing
the tracker's expected output by hand.
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    import gymnasium as gym  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without SB3
    SB3_IMPORT_ERROR = exc
else:
    from common.sb3_callbacks import evaluate_sb3_policy_at_fixed_seeds
    from evaluations.sustained_capture import (
        STRICT_CAPTURE_INFO_KEY,
        SustainedCaptureSpec,
    )
    SB3_IMPORT_ERROR = None


class _ScriptedFixedSeedEnv:
    """Single gymnasium-style env driven by a per-seed episode script."""

    def __init__(self, episodes_by_seed: dict) -> None:
        self._episodes_by_seed = episodes_by_seed
        self.reset_seeds: list[int] = []
        self._current: dict | None = None
        self._step_index = 0

    def reset(self, *, seed=None):
        self.reset_seeds.append(seed)
        self._current = self._episodes_by_seed[seed]
        self._step_index = 0
        obs = np.zeros((1,), dtype=np.float32)
        info = {STRICT_CAPTURE_INFO_KEY: float(self._current["initial"])}
        return obs, info

    def step(self, action):
        del action
        inside, dt_used, reward = self._current["steps"][self._step_index]
        done = self._step_index + 1 == len(self._current["steps"])
        info = {STRICT_CAPTURE_INFO_KEY: float(inside), "dt_used": float(dt_used)}
        self._step_index += 1
        obs = np.zeros((1,), dtype=np.float32)
        if done:
            info["episode"] = {
                "r": float(self._current["monitor_reward"]),
                "l": len(self._current["steps"]),
            }
        return obs, reward, done, False, info


class _ScriptedSingleEnvPolicy:
    def predict(self, obs, deterministic=True):
        del obs, deterministic
        return np.zeros((1,), dtype=np.float32), None


@unittest.skipIf(
    SB3_IMPORT_ERROR is not None, f"SB3 unavailable: {SB3_IMPORT_ERROR}"
)
class EvaluateSb3PolicyAtFixedSeedsTests(unittest.TestCase):
    def test_resets_each_episode_at_its_exact_seed_in_order(self):
        env = _ScriptedFixedSeedEnv(
            {
                10: {
                    "initial": True,
                    "steps": [(True, 1.0, 1.0)],
                    "monitor_reward": 101.0,
                },
                20: {
                    "initial": False,
                    "steps": [(True, 0.6, 2.0), (True, 0.6, 3.0)],
                    "monitor_reward": 202.0,
                },
            }
        )
        result = evaluate_sb3_policy_at_fixed_seeds(
            _ScriptedSingleEnvPolicy(),
            env,
            [10, 20],
            deterministic=True,
            capture_spec=SustainedCaptureSpec(),
        )

        self.assertEqual(env.reset_seeds, [10, 20])
        self.assertEqual(result.rewards, [101.0, 202.0])
        self.assertEqual(result.lengths, [1, 2])
        # A 1-step episode holding the full duration_seconds succeeds; a
        # 2-step episode starting cold only accrues its second step's
        # duration (0.6s), short of the 1.0s default requirement.
        self.assertEqual(result.capture_successes, [True, False])
        np.testing.assert_allclose(result.capture_durations, [1.0, 0.6])

    def test_rejects_an_empty_seed_list(self):
        env = _ScriptedFixedSeedEnv({})
        with self.assertRaises(ValueError):
            evaluate_sb3_policy_at_fixed_seeds(
                _ScriptedSingleEnvPolicy(),
                env,
                [],
                deterministic=True,
                capture_spec=SustainedCaptureSpec(),
            )


if __name__ == "__main__":
    unittest.main()
