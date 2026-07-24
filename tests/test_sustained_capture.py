import unittest

from evaluations.sustained_capture import (
    SustainedCaptureSpec,
    SustainedCaptureTracker,
    capture_selection_rank,
    strict_capture_spec_for,
)


def _capture_info(inside: bool, dt: float) -> dict[str, float]:
    return {
        "acrobot_strict_capture": float(inside),
        "dt_used": float(dt),
    }


class TestSustainedCaptureTracker(unittest.TestCase):
    def test_first_entry_does_not_claim_preceding_physical_time(self):
        tracker = SustainedCaptureTracker(
            1,
            SustainedCaptureSpec(),
            [{"acrobot_strict_capture": 0.0}],
        )

        self.assertIsNone(
            tracker.update_slot(0, _capture_info(True, 0.4), done=False)
        )
        result = tracker.update_slot(0, _capture_info(True, 0.6), done=True)

        self.assertIsNotNone(result)
        self.assertFalse(result.success)
        self.assertAlmostEqual(result.max_duration_seconds, 0.6)

    def test_failed_endpoint_resets_the_consecutive_run(self):
        tracker = SustainedCaptureTracker(
            1,
            SustainedCaptureSpec(),
            [{"acrobot_strict_capture": 1.0}],
        )

        tracker.update_slot(0, _capture_info(True, 0.4), done=False)
        tracker.update_slot(0, _capture_info(False, 0.2), done=False)
        tracker.update_slot(0, _capture_info(True, 0.6), done=False)
        result = tracker.update_slot(0, _capture_info(True, 0.6), done=True)

        self.assertIsNotNone(result)
        self.assertFalse(result.success)
        self.assertAlmostEqual(result.max_duration_seconds, 0.6)

    def test_one_second_threshold_accepts_float_roundoff_only_within_tolerance(self):
        spec = SustainedCaptureSpec(duration_seconds=1.0, duration_atol=1e-6)

        exact = SustainedCaptureTracker(
            1, spec, [{"acrobot_strict_capture": 1.0}]
        )
        exact.update_slot(0, _capture_info(True, 0.4), done=False)
        result = exact.update_slot(0, _capture_info(True, 0.6), done=True)
        self.assertIsNotNone(result)
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.max_duration_seconds, 1.0)

        within_tolerance = SustainedCaptureTracker(
            1, spec, [{"acrobot_strict_capture": 1.0}]
        )
        result = within_tolerance.update_slot(
            0, _capture_info(True, 1.0 - 5e-7), done=True
        )
        self.assertIsNotNone(result)
        self.assertTrue(result.success)

        outside_tolerance = SustainedCaptureTracker(
            1, spec, [{"acrobot_strict_capture": 1.0}]
        )
        result = outside_tolerance.update_slot(
            0, _capture_info(True, 1.0 - 2e-6), done=True
        )
        self.assertIsNotNone(result)
        self.assertFalse(result.success)

    def test_selection_rank_prioritizes_rate_then_mean_residence(self):
        no_success_long_residence = capture_selection_rank(
            [False, False], [20.0, 20.0]
        )
        one_success = capture_selection_rank([True, False], [1.0, 0.0])
        self.assertGreater(one_success, no_success_long_residence)

        shorter_tie = capture_selection_rank([True, False], [1.0, 0.2])
        longer_tie = capture_selection_rank([False, True], [0.8, 1.2])
        self.assertGreater(longer_tie, shorter_tie)

    def test_strict_checkpoint_rule_is_scoped_to_requested_pair(self):
        self.assertIsInstance(
            strict_capture_spec_for(
                algorithm="ct_sac", env_id="acrobot-swingup-v4.1"
            ),
            SustainedCaptureSpec,
        )
        self.assertIsInstance(
            strict_capture_spec_for(
                algorithm="ppo", env_id="acrobot-swingup-v4.1"
            ),
            SustainedCaptureSpec,
        )
        for algorithm, env_id in (
            ("sac", "acrobot-swingup-v4.1"),
            ("ct_td3", "acrobot-swingup-v4.1"),
            ("ppo", "acrobot-swingup-v4"),
            ("ct_sac", "acrobot-swingup-v5"),
        ):
            with self.subTest(algorithm=algorithm, env_id=env_id):
                self.assertIsNone(
                    strict_capture_spec_for(
                        algorithm=algorithm, env_id=env_id
                    )
                )


if __name__ == "__main__":
    unittest.main()
