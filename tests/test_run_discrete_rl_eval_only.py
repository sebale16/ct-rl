"""End-to-end contract for run_discrete_rl's standalone --eval_only mode.

Trains a tiny real checkpoint (not mocked -- same rationale as
test_run_discrete_rl_resume.py) then evaluates it standalone, so the
checkpoint-path resolution and the capture-spec/plain-reward evaluator
branches are exercised against a real SB3 model.
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

    def test_evaluates_the_final_checkpoint_with_strict_capture(self):
        summary = self._evaluate()
        self.assertEqual(summary["checkpoint"], str(self.save_dir / "final_model.zip"))
        self.assertEqual(summary["n_eval_episodes"], 2)
        self.assertIsInstance(summary["mean_reward"], float)
        # acrobot-swingup-xk has a strict-capture spec configured.
        self.assertIsInstance(summary["strict_capture_success_rate"], float)
        self.assertIsInstance(summary["strict_capture_mean_max_duration"], float)

    def test_falls_back_to_plain_reward_when_no_capture_spec_is_configured(self):
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
        summary = self._evaluate(eval_which="best")
        self.assertEqual(summary["checkpoint"], str(best_dir / "best_model.zip"))

    def test_explicit_checkpoint_overrides_eval_which(self):
        alt = self.save_dir / "final_model.zip"
        summary = self._evaluate(checkpoint=str(alt)[: -len(".zip")])
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
        summary = runner.evaluate_sb3_checkpoint(
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
            output=str(out),
        )
        self.assertEqual(json.loads(out.read_text()), summary)


if __name__ == "__main__":
    unittest.main()
