from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    from benchmarks import run_discrete_rl as runner
    from evaluations.sustained_capture import SustainedCaptureSpec
except ImportError as exc:  # pragma: no cover - dependency-light environments
    RUNNER_IMPORT_ERROR = exc
else:
    RUNNER_IMPORT_ERROR = None


@unittest.skipIf(
    RUNNER_IMPORT_ERROR is not None,
    f"SB3 benchmark dependencies unavailable: {RUNNER_IMPORT_ERROR}",
)
class DiscreteHangingEvaluationTests(unittest.TestCase):
    def test_make_env_forwards_acrobot_task_kwargs(self):
        with tempfile.TemporaryDirectory() as directory:
            inner_env = MagicMock()
            with (
                patch.object(
                    runner, "DMCContinuousEnv", return_value=inner_env
                ) as dmc_env,
                patch.object(runner, "Monitor", return_value=inner_env),
            ):
                env = runner.make_env(
                    "acrobot-swingup-v4.1",
                    Path(directory),
                    seed=7,
                    env_meta={
                        "task_kwargs": {
                            "uniform_start": False,
                            "angle_noise": 0.02,
                        }
                    },
                )

        self.assertIs(env, inner_env)
        self.assertEqual(
            dmc_env.call_args.kwargs["task_kwargs"],
            {"uniform_start": False, "angle_noise": 0.02},
        )

    def test_runner_builds_independent_hanging_track(self):
        seed = 13
        env_meta = {
            "n_envs": 2,
            "task_kwargs": {"angle_noise": 0.05},
        }
        log_kwargs = {
            "save_freq": 100,
            "eval_freq": 20,
            "interval": 10,
        }
        capture_spec = SustainedCaptureSpec()
        train_env = MagicMock(name="train_env")
        primary_eval_env = MagicMock(name="primary_eval_env")
        hanging_eval_env = MagicMock(name="hanging_eval_env")
        model = MagicMock(name="model")
        checkpoint_callback = MagicMock(name="checkpoint_callback")
        primary_callback = MagicMock(name="primary_callback")
        hanging_callback = MagicMock(name="hanging_callback")
        log_callback = MagicMock(name="log_callback")
        callback_list = MagicMock(name="callback_list")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_dir = root / "logs"
            save_dir = root / "models"
            with (
                patch.object(
                    runner,
                    "load_sb3_hyperparams_from_table",
                    return_value=(1000, env_meta, {}, {}, log_kwargs),
                ),
                patch.object(
                    runner,
                    "build_save_path",
                    side_effect=[log_dir, save_dir],
                ),
                patch.object(
                    runner,
                    "make_vec_env",
                    side_effect=[
                        train_env,
                        primary_eval_env,
                        hanging_eval_env,
                    ],
                ) as make_vec_env,
                patch.object(runner, "configure", return_value=MagicMock()),
                patch.object(runner, "PPO", return_value=model),
                patch.object(
                    runner,
                    "strict_capture_spec_for",
                    return_value=capture_spec,
                ),
                patch.object(
                    runner,
                    "SustainedCaptureEvalCallback",
                    side_effect=[primary_callback, hanging_callback],
                ) as capture_callback,
                patch.object(
                    runner,
                    "RewardEvalCallback",
                ) as reward_callback,
                patch.object(
                    runner,
                    "CheckpointCallback",
                    return_value=checkpoint_callback,
                ),
                patch.object(
                    runner,
                    "LogEveryNTimesteps",
                    return_value=log_callback,
                ),
                patch.object(
                    runner,
                    "CallbackList",
                    return_value=callback_list,
                ) as callback_list_type,
            ):
                runner.run_sb3_benchmark(
                    algo="ppo",
                    env_id="acrobot-swingup-v4.1",
                    mode="final_mf",
                    eval_mode=None,
                    seed=seed,
                    hyperparams_dir="benchmarks/hyperparams",
                    log_root_dir=str(log_dir),
                    save_root_dir=str(save_dir),
                    total_timesteps_override=None,
                    desc="strict20s_test",
                    increment_modeling=False,
                    n_eval_episodes=20,
                    eval_hanging=True,
                )

        self.assertEqual(make_vec_env.call_count, 3)
        train_call, primary_call, hanging_call = make_vec_env.call_args_list
        self.assertEqual(train_call.kwargs["seed"], seed)
        self.assertEqual(primary_call.kwargs["seed"], seed + 1000)
        self.assertEqual(hanging_call.kwargs["seed"], seed + 2000)

        primary_kwargs = primary_call.kwargs["env_kwargs"]
        hanging_kwargs = hanging_call.kwargs["env_kwargs"]
        self.assertEqual(primary_kwargs["monitor_root"], log_dir / "eval")
        self.assertEqual(
            hanging_kwargs["monitor_root"], log_dir / "eval_hanging"
        )
        self.assertEqual(
            primary_kwargs["env_meta"]["task_kwargs"],
            {"angle_noise": 0.05},
        )
        self.assertEqual(
            hanging_kwargs["env_meta"]["task_kwargs"],
            {"angle_noise": 0.05, "uniform_start": False},
        )
        self.assertEqual(
            env_meta["task_kwargs"],
            {"angle_noise": 0.05},
            "the hanging override must not mutate train/primary metadata",
        )

        self.assertEqual(capture_callback.call_count, 2)
        primary_callback_call, hanging_callback_call = (
            capture_callback.call_args_list
        )
        self.assertIs(primary_callback_call.args[0], primary_eval_env)
        self.assertEqual(
            primary_callback_call.kwargs["reset_seed"], seed + 1000
        )
        self.assertIs(hanging_callback_call.args[0], hanging_eval_env)
        self.assertIs(
            hanging_callback_call.kwargs["capture_spec"], capture_spec
        )
        self.assertEqual(
            hanging_callback_call.kwargs["reset_seed"], seed + 2000
        )
        self.assertEqual(
            hanging_callback_call.kwargs["best_model_save_path"],
            str(save_dir / "best_model_hanging"),
        )
        self.assertEqual(
            hanging_callback_call.kwargs["log_path"],
            str(log_dir / "eval_hanging"),
        )
        self.assertEqual(
            hanging_callback_call.kwargs["log_prefix"], "eval_hanging"
        )
        self.assertEqual(
            hanging_callback_call.kwargs["n_eval_episodes"], 20
        )
        self.assertEqual(hanging_callback_call.kwargs["eval_freq"], 10)
        reward_callback.assert_not_called()

        callback_list_type.assert_called_once_with(
            [
                checkpoint_callback,
                primary_callback,
                hanging_callback,
                log_callback,
            ]
        )
        model.learn.assert_called_once_with(
            total_timesteps=1000,
            callback=callback_list,
            tb_log_name="ppo_acrobot-swingup-v4.1",
            log_interval=10**9,
        )
        hanging_eval_env.close.assert_called_once_with()
        primary_eval_env.close.assert_called_once_with()
        train_env.close.assert_called_once_with()

    def test_cli_forwards_eval_hanging(self):
        argv = [
            "run_discrete_rl.py",
            "--algos",
            "ppo",
            "--env_id",
            "acrobot-swingup-v4.1",
            "--mode",
            "final_mf",
            "--eval_hanging",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(runner, "run_sb3_benchmark") as launch,
        ):
            runner.main()

        launch.assert_called_once()
        self.assertTrue(launch.call_args.kwargs["eval_hanging"])


if __name__ == "__main__":
    unittest.main()
