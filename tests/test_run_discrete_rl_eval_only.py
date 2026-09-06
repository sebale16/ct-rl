"""End-to-end contract for run_discrete_rl's standalone --eval_only mode.

Trains a tiny real checkpoint (not mocked -- same rationale as
test_run_discrete_rl_resume.py) then evaluates it standalone, so the
checkpoint-path resolution and the capture-spec/plain-reward evaluator
branches are exercised against a real SB3 model.

acrobot-swingup-xk's fixed 32-episode protocol (see
common.demonstration.ACROBOT_XK_EVAL_SEEDS) makes every capture-spec-backed
eval run all 32 episodes regardless of --n_eval_episodes -- correct, but
each episode can run up to env_max_steps, so exercising it for real here
would make routine test runs take minutes instead of seconds. The tests
below that aren't specifically about that fixed-seed dispatch mock
common.sb3_callbacks.evaluate_sb3_policy_at_fixed_seeds; the dispatch
itself (right function, right seeds, n_eval_episodes overridden) is its own
test, and evaluate_sb3_policy_at_fixed_seeds itself is unit-tested in
tests/test_sb3_capture_exact_seeds.py without any of this real-checkpoint
machinery.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from benchmarks import run_discrete_rl as runner
    from common.demonstration import ACROBOT_XK_EVAL_SEEDS
    from common.sb3_callbacks import SB3CaptureEvaluation
except ImportError as exc:  # pragma: no cover - dependency-light environments
    RUNNER_IMPORT_ERROR = exc
else:
    RUNNER_IMPORT_ERROR = None

ENV_ID = "acrobot-swingup-xk"
MODE = "xk_r3_eta0p23_ctrl10ms_h2s_temp0p01_xkdot_q2dot4pi_logrecip_tau1p25e2_irregular1m"


@unittest.skipIf(
    RUNNER_IMPORT_ERROR is not None,
    f"SB3 benchmark dependencies unavailable: {RUNNER_IMPORT_ERROR}",
)
class EvalOnlyTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        runner.run_sb3_benchmark(
            algo="sac",
            env_id=ENV_ID,
            mode=MODE,
            eval_mode=None,
            seed=99,
            hyperparams_dir="benchmarks/hyperparams",
            log_root_dir=str(self.root / "logs"),
            save_root_dir=str(self.root / "models"),
            total_timesteps_override=200,
            desc="",
            increment_modeling=False,
            n_eval_episodes=1,
            run_id="evalonlytest",
        )
        self.save_dir = (
            self.root
            / "models"
            / "sac"
            / ENV_ID
            / MODE
            / "seed_99"
            / "dt_0_01_maxs_20000_evalonlytest"
        )

    def _evaluate(self, **kwargs):
        params = dict(
            algo="sac",
            env_id=ENV_ID,
            mode=MODE,
            eval_mode=None,
            seed=99,
            hyperparams_dir="benchmarks/hyperparams",
            save_root_dir=str(self.root / "models"),
            checkpoint=None,
            eval_which="final",
            n_eval_episodes=2,
            run_id="evalonlytest",
            output=None,
        )
        params.update(kwargs)
        return runner.evaluate_sb3_checkpoint(**params)

    def _evaluate_with_fixed_seeds_mocked(self, **kwargs):
        """Run _evaluate with the (slow, real-32-episode) fixed-seed
        evaluator replaced by a canned result -- for tests about dispatch,
        checkpoint resolution, or output plumbing, not about the fixed-seed
        evaluator's own correctness."""
        canned = SB3CaptureEvaluation(
            rewards=[1.0] * len(ACROBOT_XK_EVAL_SEEDS),
            lengths=[10] * len(ACROBOT_XK_EVAL_SEEDS),
            capture_successes=[True] * len(ACROBOT_XK_EVAL_SEEDS),
            capture_durations=[1.0] * len(ACROBOT_XK_EVAL_SEEDS),
        )
        with mock.patch.object(
            runner, "evaluate_sb3_policy_at_fixed_seeds", return_value=canned
        ) as mocked:
            summary = self._evaluate(**kwargs)
        return summary, mocked

    def test_dispatches_to_the_fixed_seed_evaluator_with_all_32_seeds(self):
        summary, mocked = self._evaluate_with_fixed_seeds_mocked()
        mocked.assert_called_once()
        called_seeds = mocked.call_args.args[2]
        self.assertEqual(tuple(called_seeds), ACROBOT_XK_EVAL_SEEDS)
        # Forced to the fixed protocol's count regardless of the requested
        # --n_eval_episodes (2, per _evaluate's default kwargs).
        self.assertEqual(summary["n_eval_episodes"], len(ACROBOT_XK_EVAL_SEEDS))
        self.assertEqual(summary["strict_capture_success_rate"], 1.0)

    def test_evaluates_the_final_checkpoint_with_strict_capture(self):
        summary, _ = self._evaluate_with_fixed_seeds_mocked()
        self.assertEqual(summary["checkpoint"], str(self.save_dir / "final_model.zip"))
        self.assertIsInstance(summary["mean_reward"], float)
        # acrobot-swingup-xk has a strict-capture spec configured.
        self.assertIsInstance(summary["strict_capture_success_rate"], float)
        self.assertIsInstance(summary["strict_capture_mean_max_duration"], float)

    def test_falls_back_to_plain_reward_when_no_capture_spec_is_configured(self):
        # No capture spec means no fixed-seed dispatch either (that path is
        # gated on capture_spec is not None); real VecEnv/evaluate_policy,
        # same as before, stays fast.
        with mock.patch.object(runner, "strict_capture_spec_for", return_value=None):
            summary = self._evaluate()
        self.assertIsNone(summary["strict_capture_success_rate"])
        self.assertIsNone(summary["strict_capture_mean_max_duration"])
        self.assertIsInstance(summary["mean_reward"], float)

    def test_eval_which_best_resolves_the_best_model_path(self):
        best_dir = self.save_dir / "best_model"
        best_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            str(self.save_dir / "final_model.zip"),
            str(best_dir / "best_model.zip"),
        )
        summary, _ = self._evaluate_with_fixed_seeds_mocked(eval_which="best")
        self.assertEqual(summary["checkpoint"], str(best_dir / "best_model.zip"))

    def test_explicit_checkpoint_overrides_eval_which(self):
        alt = self.save_dir / "final_model.zip"
        summary, _ = self._evaluate_with_fixed_seeds_mocked(
            checkpoint=str(alt)[: -len(".zip")]
        )
        self.assertEqual(summary["checkpoint"], str(alt))

    def test_missing_checkpoint_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            runner.evaluate_sb3_checkpoint(
                algo="sac",
                env_id=ENV_ID,
                mode=MODE,
                eval_mode=None,
                seed=12345,
                hyperparams_dir="benchmarks/hyperparams",
                save_root_dir=str(self.root / "models"),
                checkpoint=None,
                eval_which="final",
                n_eval_episodes=2,
                run_id="evalonlytest",
                output=None,
            )

    def test_writes_the_summary_to_the_requested_output_path(self):
        out = self.root / "summary.json"
        summary, _ = self._evaluate_with_fixed_seeds_mocked(output=str(out))
        self.assertEqual(json.loads(out.read_text()), summary)


if __name__ == "__main__":
    unittest.main()
