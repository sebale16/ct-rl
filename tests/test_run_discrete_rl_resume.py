"""End-to-end contract for run_discrete_rl's SB3 resume/wall-clock support.

Real (tiny) training runs rather than mocks: the resume path threads a
checkpoint's ``model.zip``/``replay_buffer.pkl`` back through
``AlgoClass.load`` + ``load_replay_buffer`` + ``reset_num_timesteps=False``,
which is easy to get subtly wrong (e.g. losing the buffer, or restarting the
step counter) in a way pure mocks would not catch.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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
class ResumeAndWallClockStopTests(unittest.TestCase):
    def _run(self, root: Path, total_timesteps: int, resume: bool, max_seconds):
        runner.run_sb3_benchmark(
            algo="sac",
            env_id=ENV_ID,
            mode=MODE,
            eval_mode=None,
            seed=99,
            hyperparams_dir="benchmarks/hyperparams",
            log_root_dir=str(root / "logs"),
            save_root_dir=str(root / "models"),
            total_timesteps_override=total_timesteps,
            desc="",
            increment_modeling=False,
            n_eval_episodes=1,
            resume=resume,
            max_seconds=max_seconds,
            run_id="resumetest",
        )

    def _save_dir(self, root: Path) -> Path:
        return (
            root
            / "models"
            / "sac"
            / ENV_ID
            / MODE
            / "seed_99"
            / "dt_0_01_maxs_20000_resumetest"
        )

    def test_wall_clock_stop_checkpoints_without_a_final_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._run(root, total_timesteps=2000, resume=False, max_seconds=0.0)
            save_dir = self._save_dir(root)
            self.assertTrue((save_dir / "checkpoint.zip").exists())
            self.assertTrue((save_dir / "checkpoint.pkl").exists())
            self.assertFalse((save_dir / "final_model.zip").exists())

    def test_resume_continues_the_step_count_to_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._run(root, total_timesteps=2000, resume=False, max_seconds=0.0)
            save_dir = self._save_dir(root)
            from stable_baselines3 import SAC

            paused_steps = SAC.load(str(save_dir / "checkpoint")).num_timesteps
            self.assertGreater(paused_steps, 0)

            target = paused_steps + 20
            self._run(root, total_timesteps=target, resume=True, max_seconds=None)

            final_model = save_dir / "final_model.zip"
            self.assertTrue(final_model.exists())
            finished = SAC.load(str(final_model))
            self.assertGreaterEqual(finished.num_timesteps, target)


if __name__ == "__main__":
    unittest.main()
