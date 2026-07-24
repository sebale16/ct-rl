from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

try:
    import evaluations.eval_acrobot_v41_v5 as acrobot_eval
    from evaluations.sustained_capture import (
        CaptureEpisodeResult,
        SustainedCaptureSpec,
    )
except ImportError as exc:  # pragma: no cover - dependency-light environments
    EVALUATOR_IMPORT_ERROR = exc
else:
    EVALUATOR_IMPORT_ERROR = None


class _ZeroSB3Policy:
    def predict(self, obs, *, deterministic):
        del obs, deterministic
        return np.zeros(1, dtype=np.float32), None


class _ScriptedSingleEnv:
    def __init__(self, *, initial_inside, steps, publish_strict=True):
        self._initial_inside = bool(initial_inside)
        self._steps = list(steps)
        self._publish_strict = bool(publish_strict)
        self._index = 0
        self._clock = 0.0

    def reset(self, *, seed=None):
        del seed
        self._index = 0
        self._clock = 0.0
        info = {}
        if self._publish_strict:
            info["acrobot_strict_capture"] = float(self._initial_inside)
        return np.zeros(1, dtype=np.float32), info

    def step_dt(self, action):
        del action
        inside, dt_used = self._steps[self._index]
        old_clock = self._clock
        # Deliberately disagree with dt_used: strict residence must use the
        # physical duration published in info rather than this clock delta.
        self._clock += 0.05
        self._index += 1
        truncated = self._index == len(self._steps)
        obs = np.zeros(1, dtype=np.float32)
        info = {
            "acrobot_tip_height": 4.0,
            "acrobot_hold": 1.0,
            "dt_used": float(dt_used),
        }
        if self._publish_strict:
            info["acrobot_strict_capture"] = float(inside)
        return (
            obs,
            old_clock,
            np.zeros(1, dtype=np.float32),
            1.0,
            obs,
            self._clock,
            False,
            truncated,
            info,
        )


@unittest.skipIf(
    EVALUATOR_IMPORT_ERROR is not None,
    f"Acrobot evaluator dependencies unavailable: {EVALUATOR_IMPORT_ERROR}",
)
class AcrobotV41StrictMetricsTests(unittest.TestCase):
    def test_rollout_uses_reset_endpoint_and_dt_used(self):
        spec = SustainedCaptureSpec()
        policy = ("sb3", _ZeroSB3Policy())

        outside = acrobot_eval.rollout(
            _ScriptedSingleEnv(
                initial_inside=False,
                steps=[(True, 0.4), (True, 0.6)],
            ),
            policy,
            seed=123,
            capture_spec=spec,
        )
        self.assertIsNotNone(outside.capture_result)
        self.assertFalse(outside.capture_result.success)
        self.assertAlmostEqual(
            outside.capture_result.max_duration_seconds, 0.6
        )

        inside = acrobot_eval.rollout(
            _ScriptedSingleEnv(
                initial_inside=True,
                steps=[(True, 0.4), (True, 0.6)],
            ),
            policy,
            seed=123,
            capture_spec=spec,
        )
        self.assertIsNotNone(inside.capture_result)
        self.assertTrue(inside.capture_result.success)
        self.assertAlmostEqual(
            inside.capture_result.max_duration_seconds, 1.0
        )

        legacy = acrobot_eval.rollout(
            _ScriptedSingleEnv(
                initial_inside=False,
                steps=[(False, 0.5)],
                publish_strict=False,
            ),
            policy,
            seed=123,
            capture_spec=None,
        )
        self.assertIsNone(legacy.capture_result)

    def test_main_exports_callback_equivalent_rank_and_v5_nan(self):
        strict_spec = {
            "framework": "sb3",
            "algo": "ppo",
            "env_id": "acrobot-swingup-v4.1",
            "mode": "final_mf",
            "seed": 0,
            "kind": "best",
            "path": "strict.zip",
        }
        legacy_spec = {
            **strict_spec,
            "env_id": "acrobot-swingup-v5",
            "path": "legacy.zip",
        }
        rollout_results = [
            acrobot_eval.AcrobotRolloutMetrics(
                episode_return=10.0,
                max_tip_height=4.0,
                height_occupancy=0.2,
                hold_occupancy=0.1,
                capture_result=CaptureEpisodeResult(True, 1.2),
            ),
            acrobot_eval.AcrobotRolloutMetrics(
                episode_return=20.0,
                max_tip_height=2.0,
                height_occupancy=0.0,
                hold_occupancy=0.0,
                capture_result=CaptureEpisodeResult(False, 0.4),
            ),
            acrobot_eval.AcrobotRolloutMetrics(
                episode_return=1.0,
                max_tip_height=1.0,
                height_occupancy=0.0,
                hold_occupancy=0.0,
                capture_result=None,
            ),
            acrobot_eval.AcrobotRolloutMetrics(
                episode_return=2.0,
                max_tip_height=2.0,
                height_occupancy=0.0,
                hold_occupancy=0.0,
                capture_result=None,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "eval.csv"
            envs = []

            def make_env(**kwargs):
                del kwargs
                env = MagicMock()
                envs.append(env)
                return env

            with (
                patch.object(
                    acrobot_eval,
                    "discover",
                    return_value=[strict_spec, legacy_spec],
                ),
                patch.object(
                    acrobot_eval,
                    "env_kwargs_for",
                    return_value=({}, None),
                ),
                patch.object(acrobot_eval, "make_ct_env", side_effect=make_env),
                patch.object(acrobot_eval, "load_policy", return_value=object()),
                patch.object(
                    acrobot_eval,
                    "rollout",
                    side_effect=rollout_results,
                ),
                patch.object(acrobot_eval, "STARTS", [("uniform", True)]),
                patch.object(acrobot_eval, "N_EVAL", 2),
                patch.object(acrobot_eval, "OUT", str(out)),
            ):
                acrobot_eval.main()

            with out.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            rows[0]["strict_capture_success_rate"], "0.5"
        )
        self.assertEqual(
            rows[0]["strict_capture_mean_max_duration"], "0.8"
        )
        self.assertEqual(rows[0]["mean_return"], "15.0")
        self.assertEqual(rows[0]["mean_hold_occ"], "0.05")
        self.assertTrue(
            math.isnan(float(rows[1]["strict_capture_success_rate"]))
        )
        self.assertTrue(
            math.isnan(
                float(rows[1]["strict_capture_mean_max_duration"])
            )
        )
        self.assertEqual(len(envs), 4)
        for env in envs:
            env.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
