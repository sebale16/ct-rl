from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from common.callbacks import EvalCallback, MasteryCurriculumCallback
from common.mastery_curriculum import MasteryCurriculum
from evaluations.evaluation_helpers import EpisodeEvaluationResults
from evaluations.sustained_capture import SustainedCaptureSpec


class MasteryCurriculumTests(unittest.TestCase):
    def test_requires_consecutive_threshold_passes_and_clears_evidence(self):
        curriculum = MasteryCurriculum(
            num_stages=4, success_threshold=0.8, consecutive_evals=2
        )

        self.assertFalse(curriculum.observe(0.8))
        self.assertEqual(curriculum.consecutive_passes, 1)
        self.assertFalse(curriculum.observe(0.79))
        self.assertEqual(curriculum.consecutive_passes, 0)

        self.assertFalse(curriculum.observe(0.9))
        self.assertTrue(curriculum.observe(1.0))
        self.assertEqual(curriculum.stage, 1)
        self.assertEqual(curriculum.consecutive_passes, 0)

    def test_one_observation_advances_at_most_once_and_final_stage_saturates(self):
        curriculum = MasteryCurriculum(num_stages=3)

        self.assertTrue(curriculum.observe(1.0))
        self.assertEqual(curriculum.stage, 1)
        self.assertTrue(curriculum.observe(1.0))
        self.assertEqual(curriculum.stage, 2)
        for _ in range(5):
            self.assertFalse(curriculum.observe(1.0))
            self.assertEqual(curriculum.stage, 2)
            self.assertEqual(curriculum.consecutive_passes, 0)
        self.assertTrue(curriculum.at_final_stage)

    def test_rejects_invalid_configuration_and_probe_rates(self):
        for num_stages in (0, -1, 1.5, True):
            with self.subTest(num_stages=num_stages):
                with self.assertRaises(ValueError):
                    MasteryCurriculum(num_stages=num_stages)
        for threshold in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    MasteryCurriculum(2, success_threshold=threshold)
        with self.assertRaises(ValueError):
            MasteryCurriculum(2, consecutive_evals=0)

        curriculum = MasteryCurriculum(2)
        for rate in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(rate=rate):
                with self.assertRaises(ValueError):
                    curriculum.observe(rate)

    def test_state_roundtrip_preserves_stage_and_pending_evidence(self):
        source = MasteryCurriculum(
            num_stages=4, success_threshold=0.75, consecutive_evals=3
        )
        for rate in (0.8, 0.9, 1.0, 0.8):
            source.observe(rate)
        self.assertEqual(source.stage, 1)
        self.assertEqual(source.consecutive_passes, 1)

        restored = MasteryCurriculum(
            num_stages=4, success_threshold=0.75, consecutive_evals=3
        )
        restored.load_state_dict(source.state_dict())
        self.assertEqual(restored.state_dict(), source.state_dict())
        self.assertFalse(restored.observe(0.9))
        self.assertTrue(restored.observe(1.0))
        self.assertEqual(restored.stage, 2)

        incompatible = MasteryCurriculum(
            num_stages=5, success_threshold=0.75, consecutive_evals=3
        )
        with self.assertRaisesRegex(ValueError, "configuration"):
            incompatible.load_state_dict(source.state_dict())


class MasteryCurriculumCallbackTests(unittest.TestCase):
    @staticmethod
    def _callback(
        *, set_stage, success_threshold=0.8, consecutive_evals=1
    ):
        algorithm = MagicMock()
        algorithm.num_timesteps = 0
        algorithm.logger = MagicMock()
        callback = MasteryCurriculumCallback(
            set_stage=set_stage,
            num_stages=3,
            success_threshold=success_threshold,
            consecutive_evals=consecutive_evals,
        )
        callback.init_callback(algorithm)
        return callback, algorithm

    def test_reads_parent_capture_rate_advances_and_logs(self):
        stages: list[int] = []
        callback, algorithm = self._callback(
            set_stage=stages.append, consecutive_evals=2
        )
        callback.parent = SimpleNamespace(last_capture_success_rate=0.9)

        self.assertTrue(callback.on_step())
        self.assertEqual(stages, [])
        self.assertTrue(callback.on_step())
        self.assertEqual(stages, [1])
        algorithm.logger.record.assert_any_call("curriculum/stage", 1)
        algorithm.logger.record.assert_any_call(
            "curriculum/probe_success_rate", 0.9
        )

    def test_missing_capture_result_is_inert(self):
        stages: list[int] = []
        callback, algorithm = self._callback(set_stage=stages.append)
        callback.parent = SimpleNamespace(last_capture_success_rate=None)

        self.assertTrue(callback.on_step())
        self.assertEqual(callback.stage, 0)
        self.assertEqual(stages, [])
        algorithm.logger.record.assert_not_called()

    def test_callback_state_roundtrip_reapplies_restored_stage(self):
        source_stages: list[int] = []
        source, _ = self._callback(set_stage=source_stages.append)
        source.parent = SimpleNamespace(last_capture_success_rate=1.0)
        source.on_step()
        self.assertEqual(source.stage, 1)

        restored_stages: list[int] = []
        restored, _ = self._callback(set_stage=restored_stages.append)
        restored.load_state_dict(source.state_dict())
        self.assertEqual(restored.stage, 1)
        self.assertEqual(restored_stages, [1])

    @patch("common.callbacks.evaluate_policy_per_episode")
    def test_capture_eval_publishes_latest_result_before_adapter_runs(
        self, evaluate
    ):
        evaluate.return_value = EpisodeEvaluationResults(
            returns=[1.0, 2.0],
            lengths=[10, 10],
            capture_successes=[True, False],
            capture_durations=[1.2, 0.4],
        )
        stages: list[int] = []
        algorithm = MagicMock()
        algorithm.num_timesteps = 0
        algorithm.logger = MagicMock()
        algorithm.model = MagicMock()
        adapter = MasteryCurriculumCallback(
            set_stage=stages.append,
            num_stages=3,
            success_threshold=0.5,
        )
        evaluation = EvalCallback(
            eval_env=MagicMock(),
            eval_freq=1,
            n_eval_episodes=2,
            capture_spec=SustainedCaptureSpec(),
            callback_after_eval=adapter,
            verbose=0,
        )
        self.assertIsNone(evaluation.last_capture_success_rate)
        self.assertIsNone(evaluation.last_capture_duration)
        evaluation.init_callback(algorithm)

        algorithm.num_timesteps = 1
        log_events: list[tuple[str, str | None]] = []
        algorithm.logger.record.side_effect = (
            lambda key, *_args, **_kwargs: log_events.append(("record", key))
        )
        with patch(
            "common.callbacks.dump",
            side_effect=lambda **_kwargs: log_events.append(("dump", None)),
        ):
            self.assertTrue(evaluation.on_step())
        self.assertEqual(evaluation.last_capture_success_rate, 0.5)
        self.assertAlmostEqual(evaluation.last_capture_duration, 0.8)
        self.assertEqual(stages, [1])
        self.assertLess(
            log_events.index(("record", "curriculum/stage")),
            log_events.index(("dump", None)),
        )


try:
    from common.sb3_callbacks import (
        MasteryCurriculumCallback as SB3MasteryCurriculumCallback,
    )
except ImportError:  # pragma: no cover - dependency-light environments
    SB3MasteryCurriculumCallback = None


@unittest.skipIf(
    SB3MasteryCurriculumCallback is None, "stable-baselines3 is unavailable"
)
class SB3MasteryCurriculumCallbackTests(unittest.TestCase):
    def test_adapter_consumes_parent_rate_and_roundtrips(self):
        stages: list[int] = []
        callback = SB3MasteryCurriculumCallback(
            set_stage=stages.append,
            num_stages=2,
        )
        callback.parent = SimpleNamespace(last_capture_success_rate=0.8)
        callback.model = SimpleNamespace(logger=MagicMock())

        self.assertTrue(callback._on_step())
        self.assertEqual(stages, [1])
        callback.model.logger.record.assert_any_call("curriculum/stage", 1)

        restored_stages: list[int] = []
        restored = SB3MasteryCurriculumCallback(
            set_stage=restored_stages.append,
            num_stages=2,
        )
        restored.load_state_dict(callback.state_dict())
        self.assertEqual(restored_stages, [1])


if __name__ == "__main__":
    unittest.main()
