"""Tests for Acrobot XK metric 7: simulated seconds to capture rate."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluations.acrobot_training_metrics import (
    DEFAULT_CAPTURE_SUCCESS_TARGETS,
    capture_learning_curve,
    load_capture_learning_curve,
    training_simulated_seconds_to_capture_success,
)


class TestAcrobotXKTrainingMetric(unittest.TestCase):
    def test_default_targets_use_first_observed_checkpoint_crossing(self):
        seconds = [1.25, 2.75, 4.0, 5.5, 8.25]
        successes = np.array(
            [
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            ],
            dtype=bool,
        )

        curve = capture_learning_curve(
            seconds,
            successes,
            evaluation_timesteps=[100, 200, 300, 400, 500],
        )
        np.testing.assert_allclose(curve.success_rates, [0.2, 0.5, 0.8, 0.7, 0.9])
        np.testing.assert_array_equal(curve.evaluation_episode_counts, [10] * 5)
        self.assertEqual(
            curve.training_times(),
            {0.5: 2.75, 0.8: 4.0, 0.9: 8.25},
        )
        np.testing.assert_array_equal(
            curve.evaluation_timesteps, [100, 200, 300, 400, 500]
        )
        self.assertEqual(DEFAULT_CAPTURE_SUCCESS_TARGETS, (0.5, 0.8, 0.9))

    def test_crossing_is_inclusive_not_interpolated_and_can_be_unreached(self):
        result = training_simulated_seconds_to_capture_success(
            [37.2, 78.9, 121.4],
            [[False, False], [True, False], [False, False]],
            targets=(0.25, 0.5, 0.75),
        )
        self.assertEqual(result[0.25], 78.9)
        self.assertEqual(result[0.5], 78.9)
        self.assertTrue(np.isinf(result[0.75]))

    def test_ragged_callback_rows_are_supported(self):
        successes = np.empty(3, dtype=object)
        successes[0] = np.array([False, True])
        successes[1] = np.array([True, True, False, True])
        successes[2] = np.array([True])
        curve = capture_learning_curve([0.8, 2.1, 3.7], successes)
        np.testing.assert_allclose(curve.success_rates, [0.5, 0.75, 1.0])
        np.testing.assert_array_equal(curve.evaluation_episode_counts, [2, 4, 1])

    def test_npz_loader_prefers_capture_specific_physical_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluations.npz"
            rows = np.empty(2, dtype=object)
            rows[0] = np.array([False, False])
            rows[1] = np.array([True, False])
            np.savez(
                path,
                timesteps=np.array([25, 50, 75]),
                capture_timesteps=np.array([50, 75]),
                simulated_seconds=np.array([0.25, 0.5, 0.75]),
                capture_simulated_seconds=np.array([0.49, 0.74]),
                capture_successes=rows,
            )

            curve = load_capture_learning_curve(path)

        np.testing.assert_allclose(curve.simulated_seconds, [0.49, 0.74])
        np.testing.assert_array_equal(curve.evaluation_timesteps, [50, 75])
        np.testing.assert_allclose(curve.success_rates, [0.0, 0.5])

    def test_npz_loader_accepts_general_physical_time_axis(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluations.npz"
            np.savez(
                path,
                timesteps=np.array([100, 200]),
                simulated_seconds=np.array([1.7, 3.6]),
                capture_successes=np.array(
                    [[False, False], [True, True]], dtype=bool
                ),
            )

            curve = load_capture_learning_curve(path)

        np.testing.assert_allclose(curve.simulated_seconds, [1.7, 3.6])
        np.testing.assert_array_equal(curve.evaluation_timesteps, [100, 200])
        np.testing.assert_allclose(curve.success_rates, [0.0, 1.0])

    def test_timestep_only_artifact_requires_explicit_fixed_step_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluations.npz"
            np.savez(
                path,
                capture_timesteps=np.array([100, 250]),
                capture_successes=np.array(
                    [[False, False], [True, True]], dtype=bool
                ),
            )
            with self.assertRaisesRegex(ValueError, "explicit"):
                load_capture_learning_curve(path)
            curve = load_capture_learning_curve(
                path, legacy_seconds_per_timestep=0.002
            )

        np.testing.assert_allclose(curve.simulated_seconds, [0.2, 0.5])

    def test_explicit_legacy_conversion_can_recover_nan_resume_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluations.npz"
            np.savez(
                path,
                capture_simulated_seconds=np.array([np.nan, np.nan]),
                capture_timesteps=np.array([100, 250]),
                capture_successes=np.array(
                    [[False, False], [True, True]], dtype=bool
                ),
            )
            with self.assertRaisesRegex(ValueError, "invalid"):
                load_capture_learning_curve(path)
            curve = load_capture_learning_curve(
                path, legacy_seconds_per_timestep=0.002
            )

        np.testing.assert_allclose(curve.simulated_seconds, [0.2, 0.5])

    def test_invalid_artifacts_and_targets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            capture_learning_curve([1.0, 1.0], [[False], [True]])
        with self.assertRaisesRegex(ValueError, "row count"):
            capture_learning_curve([1.0, 2.0], [[False, True]])
        with self.assertRaisesRegex(ValueError, "binary"):
            capture_learning_curve([1.0], [[0.0, 0.5]])

        curve = capture_learning_curve([1.0], [[True]])
        with self.assertRaisesRegex(ValueError, r"in \[0, 1\]"):
            curve.training_times((1.1,))
        with self.assertRaisesRegex(ValueError, "unique"):
            curve.training_times((0.5, 0.5))
        with self.assertRaisesRegex(ValueError, "at least one"):
            curve.training_times(())


if __name__ == "__main__":
    unittest.main()
