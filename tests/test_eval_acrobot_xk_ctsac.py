"""Focused tests for the fixed-protocol learned Acrobot-XK evaluator."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch as th

from evaluations.acrobot_homoclinic_metrics import EpisodeMetrics
from evaluations.eval_acrobot_xk_ctsac import (
    DEFAULT_SEEDS,
    EPISODE_METRIC_FIELDS,
    OUTPUT_FIELDS,
    DeterministicCTPolicy,
    EvaluationProtocol,
    RewardMetadata,
    build_env,
    build_parser,
    build_task_kwargs,
    evaluate_checkpoint,
    load_metric7_summary,
    parse_seed_spec,
    resolve_reward_metadata,
    summarize_metrics,
    write_outputs,
)


def _episode(*, captured: bool, offset: float = 0.0) -> EpisodeMetrics:
    return EpisodeMetrics(
        duration=20.0,
        captured=captured,
        capture_time=5.0 + offset if captured else float("inf"),
        retention=0.9 - 0.1 * offset if captured else float("nan"),
        error_rms=0.04 + offset,
        error_rms_energy=0.03 + offset,
        error_rms_angle=0.01 + offset,
        error_rms_rate=0.02 + offset,
        orbit_distance_rms=0.05 + offset,
        orbit_distance_final=0.025 + offset,
        control_effort=200.0 + offset,
        saturation=0.0,
        lqr_time=12.0 + offset if captured else float("inf"),
        lqr_residual_min=0.02 + offset,
        min_abs_energy_error=0.01 + offset,
        final_abs_energy_error=0.015 + offset,
        peak_shoulder_rate=4.0 + offset,
        peak_commanded_torque=18.0 + offset,
    )


class TestAcrobotXKCTSACEvaluator(unittest.TestCase):
    def test_default_protocol_is_the_documented_fixed_protocol(self):
        protocol = EvaluationProtocol()
        self.assertEqual(protocol.seeds, DEFAULT_SEEDS)
        self.assertEqual(protocol.seeds[0], 20000)
        self.assertEqual(protocol.seeds[-1], 20031)
        self.assertEqual(len(protocol.seeds), 32)
        self.assertEqual(protocol.t_max, 20.0)
        self.assertEqual(protocol.dt, 0.001)
        self.assertEqual(protocol.physics_dt, 0.001)
        self.assertEqual(protocol.damping, 0.0)
        self.assertEqual(protocol.torque_limit, 20.0)
        self.assertEqual(protocol.max_steps, 20000)
        self.assertEqual(protocol.as_metadata()["start"], "release")

    def test_seed_parser_is_stop_exclusive_and_rejects_duplicates(self):
        self.assertEqual(parse_seed_spec("20000:20003"), (20000, 20001, 20002))
        self.assertEqual(parse_seed_spec("4,8,-2"), (4, 8, -2))
        with self.assertRaises(ValueError):
            parse_seed_spec("2,2")

    def test_reward_and_task_metadata_support_eta_sweeps(self):
        env_kwargs = {
            "task_kwargs": {
                "reward_kind": "r2",
                "eta": 0.05,
                "k_d": 35.8,
                "k_p": 61.2,
                "uniform_start": True,
                "damping": 0.3,
                "torque_limit": 7.0,
            }
        }
        reward = resolve_reward_metadata(env_kwargs, eta=0.2)
        self.assertEqual(reward, RewardMetadata("r2", 0.2))
        task = build_task_kwargs(env_kwargs, reward, EvaluationProtocol(seeds=(1,)))
        self.assertEqual(task["eta"], 0.2)
        self.assertEqual(task["k_d"], 35.8)
        self.assertEqual(task["k_p"], 61.2)
        self.assertTrue(task["release_start"])
        self.assertFalse(task["uniform_start"])
        self.assertFalse(task["paper_start"])
        self.assertEqual(task["damping"], 0.0)
        self.assertEqual(task["torque_limit"], 20.0)
        self.assertEqual(task["failure_reward_rate"], -1.0)
        with self.assertRaises(ValueError):
            RewardMetadata("r2", None)
        with self.assertRaises(ValueError):
            RewardMetadata("r1", 0.2)

    def test_r3_metadata_requires_and_emits_physical_discount_rate(self):
        env_kwargs = {
            "task_kwargs": {
                "reward_kind": "r3",
                "eta": 0.03,
                "discount_rate": 0.1,
                "failure_reward_rate": -2.5,
            }
        }
        configured = resolve_reward_metadata(env_kwargs)
        self.assertEqual(configured, RewardMetadata("r3", 0.03, 0.1))
        overridden = resolve_reward_metadata(
            env_kwargs, eta=0.3, discount_rate=0.2
        )
        self.assertEqual(overridden, RewardMetadata("r3", 0.3, 0.2))

        task = build_task_kwargs(
            env_kwargs, overridden, EvaluationProtocol(seeds=(1,))
        )
        self.assertEqual(task["reward_kind"], "r3")
        self.assertEqual(task["eta"], 0.3)
        self.assertEqual(task["discount_rate"], 0.2)
        self.assertEqual(task["failure_reward_rate"], -2.5)
        self.assertIn("discount_rate", OUTPUT_FIELDS)

        with self.assertRaisesRegex(ValueError, "requires.*discount"):
            RewardMetadata("r3", 0.1)
        with self.assertRaisesRegex(ValueError, "only meaningful"):
            RewardMetadata("r2", 0.1, 0.1)

    def test_cli_accepts_r3_discount_metadata(self):
        args = build_parser().parse_args(
            [
                "--checkpoint",
                "model.pth",
                "--mode",
                "xk_r3_eta0p1_fixed1ms_h10s",
                "--reward-kind",
                "r3",
                "--eta",
                "0.1",
                "--discount-rate",
                "0.1",
            ]
        )
        self.assertEqual(args.reward_kind, "r3")
        self.assertEqual(args.eta, 0.1)
        self.assertEqual(args.discount_rate, 0.1)

    def test_policy_always_requests_deterministic_action(self):
        class Model:
            device = th.device("cpu")

            def __init__(self):
                self.calls = []

            def act(self, obs, deterministic=False):
                self.calls.append((obs.detach().cpu().numpy(), deterministic))
                return th.tensor([[0.375]], dtype=th.float32), None

        model = Model()
        action = DeterministicCTPolicy(model)(np.array([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_allclose(action, [0.375])
        self.assertEqual(model.calls[0][0].shape, (1, 4))
        self.assertIs(model.calls[0][1], True)

    def test_env_builder_forces_raw_uniform_fixed_timing(self):
        protocol = EvaluationProtocol(seeds=(3,))
        task = build_task_kwargs({}, RewardMetadata("r0", None), protocol)
        sentinel = object()
        with patch(
            "environment.dmc.DMCContinuousEnv", return_value=sentinel
        ) as constructor:
            returned = build_env(seed=3, protocol=protocol, task_kwargs=task)
        self.assertIs(returned, sentinel)
        kwargs = constructor.call_args.kwargs
        self.assertEqual(kwargs["domain_name"], "acrobot")
        self.assertEqual(kwargs["task_name"], "swingup-xk")
        self.assertTrue(kwargs["raw_state_obs"])
        self.assertEqual(kwargs["time_sampling"], "uniform")
        self.assertEqual(kwargs["dt"], 0.001)
        self.assertEqual(kwargs["physics_dt"], 0.001)
        self.assertEqual(kwargs["episode_duration"], 20.0)
        self.assertEqual(kwargs["max_steps"], 20000)
        self.assertFalse(kwargs["return_reward_increment"])
        self.assertTrue(kwargs["task_kwargs"]["release_start"])

    def test_summary_covers_metrics_one_through_six(self):
        summary = summarize_metrics(
            [_episode(captured=True), _episode(captured=False, offset=0.1)]
        )
        self.assertEqual(
            summary["metric_1_successful_capture"]["success_rate"], 0.5
        )
        self.assertEqual(summary["metric_2_time_to_capture_seconds"]["median"], 5.0)
        self.assertEqual(summary["metric_3_set_retention"]["finite_count"], 1)
        self.assertIn("energy_rms", summary["metric_4a_homoclinic_set_error"])
        self.assertIn("final", summary["metric_4b_homoclinic_orbit_distance"])
        self.assertIn("effort", summary["metric_5_control"])
        self.assertEqual(summary["metric_6_lqr_region"]["entries"], 1)

    def test_evaluate_checkpoint_reports_metrics_but_no_return(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "seed_7" / "final_model.pth"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"checkpoint")
            config = SimpleNamespace(
                env_kwargs={
                    "task_kwargs": {
                        "reward_kind": "r3",
                        "eta": 0.03,
                        "discount_rate": 0.1,
                    }
                },
                model_kwargs={"pi_net_arch": [4]},
                config_sha256="config-hash",
            )
            env = Mock()
            env._env.physics = object()
            model = SimpleNamespace(device=th.device("cpu"))

            def fake_rollout(_env, _policy, seed, *, torque_limit):
                self.assertEqual(torque_limit, 20.0)
                return int(seed)

            def fake_reduce(trajectory, _params, _tube, *, lqr_threshold):
                self.assertEqual(lqr_threshold, 0.04)
                return _episode(captured=(trajectory == 11), offset=trajectory - 11)

            with (
                patch(
                    "evaluations.eval_acrobot_xk_ctsac.build_env",
                    return_value=env,
                ) as make_env,
                patch(
                    "evaluations.eval_acrobot_xk_ctsac.load_checkpoint_model",
                    return_value=model,
                ),
                patch(
                    "evaluations.eval_acrobot_xk_ctsac.AcrobotParams.from_physics",
                    return_value=object(),
                ),
                patch(
                    "evaluations.eval_acrobot_xk_ctsac.rollout",
                    side_effect=fake_rollout,
                ),
                patch(
                    "evaluations.eval_acrobot_xk_ctsac.evaluate_episode",
                    side_effect=fake_reduce,
                ),
            ):
                result = evaluate_checkpoint(
                    checkpoint=checkpoint,
                    mode="xk_r3_eta0p03_fixed1ms_h10s",
                    config=config,
                    protocol=EvaluationProtocol(seeds=(11, 12)),
                )

            self.assertEqual(len(result.rows), 2)
            self.assertEqual([row["seed"] for row in result.rows], [11, 12])
            self.assertEqual(result.rows[0]["reward_kind"], "r3")
            self.assertEqual(result.rows[0]["eta"], 0.03)
            self.assertEqual(result.rows[0]["discount_rate"], 0.1)
            self.assertEqual(
                result.summary["reward_metadata"]["discount_rate"], 0.1
            )
            self.assertEqual(
                result.summary["task_metadata"]["failure_reward_rate"], -1.0
            )
            self.assertEqual(result.rows[0]["train_seed"], 7)
            self.assertNotIn("episode_return", OUTPUT_FIELDS)
            self.assertNotIn("reward", OUTPUT_FIELDS)
            self.assertTrue(set(EPISODE_METRIC_FIELDS).issubset(result.rows[0]))
            self.assertEqual(
                result.summary["metrics"]["metric_1_successful_capture"][
                    "success_rate"
                ],
                0.5,
            )
            make_env.assert_called_once()
            env.close.assert_called_once()

    def test_csv_and_json_outputs_are_separate_and_json_is_finite(self):
        metrics = (_episode(captured=False),)
        rows = ({field: None for field in OUTPUT_FIELDS},)
        result = SimpleNamespace(
            rows=rows,
            metrics=metrics,
            summary={"metrics": summarize_metrics(metrics)},
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "episodes.csv"
            csv_path, json_path = write_outputs(result, output)
            with csv_path.open(newline="") as stream:
                written = list(csv.DictReader(stream))
            with json_path.open() as stream:
                aggregate_json = json.load(stream)
            self.assertEqual(len(written), 1)
            self.assertIsNone(
                aggregate_json["metrics"]["metric_2_time_to_capture_seconds"][
                    "median"
                ]
            )
            with self.assertRaises(FileExistsError):
                write_outputs(result, output)

    def test_metric7_uses_simulated_seconds_and_legacy_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            physical = Path(directory) / "physical.npz"
            np.savez(
                physical,
                capture_simulated_seconds=np.array([12.5, 31.25, 70.0]),
                capture_successes=np.array(
                    [
                        [False] * 10,
                        [True] * 5 + [False] * 5,
                        [True] * 9 + [False],
                    ],
                    dtype=bool,
                ),
            )
            summary = load_metric7_summary(physical)
            self.assertEqual(
                summary["axis"], "cumulative_simulated_physical_seconds"
            )
            self.assertEqual(
                summary["simulated_seconds_thresholds"],
                {"50%": 31.25, "80%": 70.0, "90%": 70.0},
            )

            legacy = Path(directory) / "legacy.npz"
            np.savez(
                legacy,
                capture_timesteps=np.array([100, 200]),
                capture_successes=np.array(
                    [[False, False], [True, True]], dtype=bool
                ),
            )
            with self.assertRaisesRegex(ValueError, "explicit"):
                load_metric7_summary(legacy)
            converted = load_metric7_summary(
                legacy, legacy_seconds_per_timestep=0.0005
            )
            self.assertEqual(
                converted["simulated_seconds_thresholds"]["50%"], 0.1
            )


if __name__ == "__main__":
    unittest.main()
