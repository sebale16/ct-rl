import unittest

from evaluations.sustained_capture import (
    ACROBOT_XK_CAPTURE_INFO_KEY,
    SustainedCaptureSpec,
    SustainedCaptureTracker,
    capture_selection_rank,
    curriculum_mastery_capture_spec_for,
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

    def test_terminal_hold_rejects_capture_followed_by_a_fall(self):
        default = SustainedCaptureTracker(
            1,
            SustainedCaptureSpec(duration_seconds=2.0),
            [{"acrobot_strict_capture": 1.0}],
        )
        default.update_slot(0, _capture_info(True, 2.0), done=False)
        default_result = default.update_slot(
            0, _capture_info(False, 0.1), done=True
        )
        self.assertIsNotNone(default_result)
        self.assertTrue(default_result.success)
        self.assertEqual(default_result.max_duration_seconds, 2.0)

        terminal = SustainedCaptureTracker(
            1,
            SustainedCaptureSpec(
                duration_seconds=2.0,
                require_terminal_hold=True,
            ),
            [{"acrobot_strict_capture": 1.0}],
        )
        terminal.update_slot(0, _capture_info(True, 2.0), done=False)
        terminal_result = terminal.update_slot(
            0, _capture_info(False, 0.1), done=True
        )
        self.assertIsNotNone(terminal_result)
        self.assertFalse(terminal_result.success)
        self.assertEqual(terminal_result.max_duration_seconds, 2.0)

    def test_terminal_hold_must_be_long_enough_at_episode_end(self):
        successful = SustainedCaptureTracker(
            1,
            SustainedCaptureSpec(
                duration_seconds=2.0,
                require_terminal_hold=True,
            ),
            [{"acrobot_strict_capture": 0.0}],
        )
        successful.update_slot(0, _capture_info(True, 0.1), done=False)
        result = successful.update_slot(
            0, _capture_info(True, 2.0), done=True
        )
        self.assertIsNotNone(result)
        self.assertTrue(result.success)

        too_late = SustainedCaptureTracker(
            1,
            SustainedCaptureSpec(
                duration_seconds=2.0,
                require_terminal_hold=True,
            ),
            [{"acrobot_strict_capture": 1.0}],
        )
        too_late.update_slot(0, _capture_info(True, 2.5), done=False)
        too_late.update_slot(0, _capture_info(False, 0.1), done=False)
        too_late.update_slot(0, _capture_info(True, 0.1), done=False)
        result = too_late.update_slot(
            0, _capture_info(True, 1.0), done=True
        )
        self.assertIsNotNone(result)
        self.assertFalse(result.success)
        self.assertEqual(result.max_duration_seconds, 2.5)

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
        # All benchmarked algorithms share the rule on the two velocity-gated
        # capture tasks (v4.1 and the v4.2 curriculum), so every arm is ranked
        # by the same capture definition.
        for algorithm in ("ct_sac", "ct_td3", "ppo", "sac", "td3"):
            for env_id in ("acrobot-swingup-v4.1", "acrobot-swingup-v4.2"):
                with self.subTest(algorithm=algorithm, env_id=env_id):
                    self.assertIsInstance(
                        strict_capture_spec_for(
                            algorithm=algorithm, env_id=env_id
                        ),
                        SustainedCaptureSpec,
                    )
        # Out of scope: the unshaped v5 arm, the pre-capture v4 reward, and
        # any algorithm the benchmark does not run.
        for algorithm, env_id in (
            ("ct_sac", "acrobot-swingup-v5"),
            ("ppo", "acrobot-swingup-v4"),
            ("a2c", "acrobot-swingup-v4.1"),
        ):
            with self.subTest(algorithm=algorithm, env_id=env_id):
                self.assertIsNone(
                    strict_capture_spec_for(
                        algorithm=algorithm, env_id=env_id
                    )
                )

    def test_xk_uses_reward_independent_homoclinic_capture_signal(self):
        for algorithm in ("ct_sac", "ct_td3", "ppo", "sac", "td3"):
            with self.subTest(algorithm=algorithm):
                spec = strict_capture_spec_for(
                    algorithm=algorithm,
                    env_id="acrobot-swingup-xk",
                )
                self.assertIsNotNone(spec)
                self.assertEqual(spec.info_key, ACROBOT_XK_CAPTURE_INFO_KEY)
                self.assertEqual(spec.duration_seconds, 1.0)
                self.assertFalse(spec.require_terminal_hold)

        legacy = strict_capture_spec_for(
            algorithm="ct_sac",
            env_id="acrobot-swingup-v4.1",
        )
        self.assertEqual(legacy.info_key, "acrobot_strict_capture")

    def test_curriculum_mastery_requires_terminal_five_second_hold(self):
        for env_id in (
            "acrobot-swingup-v4.3",
            "acrobot-swingup-v6.1",
            "cartpole-two_poles-v2",
        ):
            with self.subTest(env_id=env_id):
                checkpoint = strict_capture_spec_for(
                    algorithm="ct_sac",
                    env_id=env_id,
                )
                mastery = curriculum_mastery_capture_spec_for(
                    algorithm="ct_sac",
                    env_id=env_id,
                )
                self.assertEqual(checkpoint.duration_seconds, 1.0)
                self.assertFalse(checkpoint.require_terminal_hold)
                self.assertEqual(mastery.duration_seconds, 5.0)
                self.assertTrue(mastery.require_terminal_hold)
                self.assertEqual(mastery.info_key, checkpoint.info_key)

    def test_non_curriculum_tasks_keep_the_checkpoint_capture_rule(self):
        for env_id in ("acrobot-swingup-v4.1", "acrobot-swingup-v6"):
            with self.subTest(env_id=env_id):
                mastery = curriculum_mastery_capture_spec_for(
                    algorithm="ct_sac",
                    env_id=env_id,
                )
                self.assertEqual(mastery.duration_seconds, 1.0)
                self.assertFalse(mastery.require_terminal_hold)


if __name__ == "__main__":
    unittest.main()
