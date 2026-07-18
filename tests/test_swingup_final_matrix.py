import contextlib
import io
import json
import os
import shlex
import unittest
from unittest import mock

from benchmarks.swingup_final_matrix import (
    ACROBOT_V2_ENV,
    BACKEND_MODES,
    CARTPOLE_ENV,
    COMBINED_MODE,
    FINAL_MODES,
    PHASE_COUNTS,
    acrobot_tuning_mode,
    build_jobs,
    command_for_job,
    expected_checkpoint_path,
    main,
)


class SwingupFinalMatrixTests(unittest.TestCase):
    def test_phase_counts_and_unique_zero_based_indices(self):
        for phase in (
            "backend-gate",
            "acrobot-tuning",
            "final-core",
            "final-combined",
            "all",
        ):
            jobs = build_jobs(phase)
            self.assertEqual(len(jobs), PHASE_COUNTS[phase])
            self.assertEqual([job.array_index for job in jobs], list(range(len(jobs))))
            identities = {
                (job.phase, job.env_id, job.mode, job.seed) for job in jobs
            }
            self.assertEqual(len(identities), len(jobs))

    def test_backend_gate_is_paired_and_resource_matched(self):
        jobs = build_jobs("backend-gate")
        cells = {(job.env_id, job.mode, job.seed) for job in jobs}
        expected = {
            (env_id, mode, seed)
            for env_id in (CARTPOLE_ENV, ACROBOT_V2_ENV)
            for mode in BACKEND_MODES
            for seed in (90, 91, 92)
        }
        self.assertEqual(cells, expected)
        self.assertTrue(all(job.cpus == 4 for job in jobs))
        self.assertTrue(all(job.total_timesteps == 100_000 for job in jobs))

    def test_tuning_factorial_and_mode_names(self):
        jobs = build_jobs("acrobot-tuning")
        cells = {
            (job.gamma, job.log_std_init, job.learning_rate, job.seed)
            for job in jobs
        }
        expected = {
            (gamma, log_std, learning_rate, seed)
            for gamma in (0.99, 0.995, 0.999)
            for log_std in (-0.5, -1.0)
            for learning_rate in (3e-4, 7.3e-4)
            for seed in (100, 101, 102, 103)
        }
        self.assertEqual(cells, expected)
        self.assertEqual(
            acrobot_tuning_mode(0.995, -1.0, 3e-4),
            "acrov2_tune_g0995_ls_m1p0_lr_3e4",
        )
        for job in jobs:
            self.assertEqual(
                job.mode,
                acrobot_tuning_mode(
                    job.gamma, job.log_std_init, job.learning_rate
                ),
            )

    def test_final_core_crosses_two_envs_six_modes_and_twelve_seeds(self):
        jobs = build_jobs("final-core")
        cells = {(job.env_id, job.mode, job.seed) for job in jobs}
        expected = {
            (env_id, mode, seed)
            for env_id in (CARTPOLE_ENV, ACROBOT_V2_ENV)
            for mode in FINAL_MODES
            for seed in range(12)
        }
        self.assertEqual(cells, expected)
        steps_by_env = {
            env_id: {job.total_timesteps for job in jobs if job.env_id == env_id}
            for env_id in (CARTPOLE_ENV, ACROBOT_V2_ENV)
        }
        self.assertEqual(steps_by_env[CARTPOLE_ENV], {500_000})
        self.assertEqual(steps_by_env[ACROBOT_V2_ENV], {1_000_000})

    def test_optional_combined_arm_has_both_envs_and_twelve_seeds(self):
        jobs = build_jobs("final-combined")
        self.assertEqual(
            {(job.env_id, job.mode, job.seed) for job in jobs},
            {
                (env_id, COMBINED_MODE, seed)
                for env_id in (CARTPOLE_ENV, ACROBOT_V2_ENV)
                for seed in range(12)
            },
        )

    def test_command_is_complete_and_shell_quoted(self):
        job = build_jobs("final-core")[0]
        command = command_for_job(
            job,
            python="/tmp/python with space",
            run_id="campaign one",
            desc="fresh run",
            max_seconds=3600,
            resume=True,
        )
        argv = shlex.split(command)
        self.assertEqual(argv[0], "/tmp/python with space")
        self.assertEqual(argv[1:3], ["-m", "benchmarks.run_ct_rl"])
        self.assertEqual(argv[argv.index("--env_id") + 1], CARTPOLE_ENV)
        self.assertEqual(argv[argv.index("--mode") + 1], "final_mf")
        self.assertEqual(argv[argv.index("--run_id") + 1], "campaign one")
        self.assertEqual(argv[argv.index("--desc") + 1], "fresh run")
        self.assertIn("--resume", argv)
        expected = expected_checkpoint_path(
            job,
            save_root="saved_models",
            run_id="campaign one",
            desc="fresh run",
        )
        self.assertEqual(expected.name, "final_model.pth")
        self.assertEqual(expected.parent.name, "dt_0_01_maxs_2000_fresh_run_campaign_one")

    def test_main_selects_one_slurm_row_and_emits_json(self):
        stdout = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"SLURM_ARRAY_TASK_ID": "11"}),
            contextlib.redirect_stdout(stdout),
        ):
            self.assertEqual(
                main(
                    [
                        "--phase",
                        "backend-gate",
                        "--slurm-index",
                        "--format",
                        "json",
                    ]
                ),
                0,
            )
        records = json.loads(stdout.getvalue())
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["array_index"], 11)
        self.assertEqual(records[0]["env_id"], ACROBOT_V2_ENV)
        self.assertEqual(records[0]["mode"], "final_oracle_rollout")
        self.assertEqual(records[0]["seed"], 92)


if __name__ == "__main__":
    unittest.main()
