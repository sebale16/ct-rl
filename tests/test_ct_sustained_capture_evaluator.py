from __future__ import annotations

import unittest

import numpy as np

try:
    import torch as th

    from evaluations.evaluation_helpers import evaluate_policy_per_episode
    from evaluations.sustained_capture import SustainedCaptureSpec
except ImportError as exc:  # pragma: no cover - dependency-light environments
    CT_IMPORT_ERROR = exc
else:
    CT_IMPORT_ERROR = None


class _ZeroPolicy:
    device = "cpu"

    def act(self, observations, *, deterministic):
        del deterministic
        return th.zeros((observations.shape[0], 1)), None


class _TwoEpisodeContinuousEnv:
    """First episode enters late; second starts inside the strict region."""

    def __init__(self) -> None:
        self.observation_space = None
        self.action_space = None
        self._episode = -1
        self._step = 0
        self._time = 0.0
        self._episodes = [
            {
                "initial": False,
                "steps": [(True, 0.4), (True, 0.6)],
            },
            {
                "initial": True,
                "steps": [(True, 0.4), (True, 0.6)],
            },
        ]

    def reset(self, *, seed=None):
        del seed
        self._episode += 1
        self._step = 0
        self._time = 0.0
        episode = self._episodes[self._episode % len(self._episodes)]
        return np.zeros(1, dtype=np.float32), {
            "acrobot_strict_capture": float(episode["initial"])
        }

    def step_dt(self, action):
        del action
        episode = self._episodes[self._episode % len(self._episodes)]
        inside, duration = episode["steps"][self._step]
        old_time = self._time
        self._time += duration
        self._step += 1
        done = self._step == len(episode["steps"])
        info = {
            "acrobot_strict_capture": float(inside),
            "dt_used": float(duration),
        }
        observation = np.zeros(1, dtype=np.float32)
        return (
            observation,
            old_time,
            np.zeros(1, dtype=np.float32),
            1.0,
            observation,
            self._time,
            False,
            done,
            info,
        )


@unittest.skipIf(
    CT_IMPORT_ERROR is not None,
    f"continuous-time dependencies unavailable: {CT_IMPORT_ERROR}",
)
class CTSustainedCaptureEvaluatorTests(unittest.TestCase):
    def test_evaluator_uses_reset_endpoint_and_physical_step_durations(self):
        metrics = evaluate_policy_per_episode(
            model=_ZeroPolicy(),
            env=_TwoEpisodeContinuousEnv(),
            n_eval_episodes=2,
            capture_spec=SustainedCaptureSpec(),
            return_metrics=True,
            reset_seed=123,
        )

        self.assertEqual(metrics.returns, [2.0, 2.0])
        self.assertEqual(metrics.lengths, [2, 2])
        self.assertEqual(metrics.capture_successes, [False, True])
        np.testing.assert_allclose(metrics.capture_durations, [0.6, 1.0])


if __name__ == "__main__":
    unittest.main()
