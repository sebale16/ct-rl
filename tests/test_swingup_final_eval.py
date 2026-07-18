from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from evaluations.evaluate_swingup_final import (
    REGIMES,
    StepSample,
    build_regime_env_kwargs,
    load_checkpoint_manifest,
    parse_seed_spec,
    summarize_episode,
    _validate_checkpoint_architecture,
    _checkpoint_train_seed,
)


def _sample(t0, dt, reward, obs, **info):
    return StepSample(
        t0=float(t0),
        t1=float(t0 + dt),
        dt_used=float(dt),
        reward=float(reward),
        next_obs=np.asarray(obs, dtype=np.float32),
        info=info,
    )


class SwingupFinalEvaluationTests(unittest.TestCase):
    def test_seed_spec_is_stop_exclusive_and_rejects_duplicates(self):
        self.assertEqual(parse_seed_spec("20000:20004"), (20000, 20001, 20002, 20003))
        self.assertEqual(parse_seed_spec("9:2:-3"), (9, 6, 3))
        self.assertEqual(parse_seed_spec("7,11,19"), (7, 11, 19))
        with self.assertRaises(ValueError):
            parse_seed_spec("1,1")
        with self.assertRaises(ValueError):
            parse_seed_spec("3:3")

    def test_regimes_preserve_train_timing_and_make_exact_uniform_timing(self):
        train = {
            "time_sampling": "irregular",
            "dt": 0.02,
            "min_dt": 0.002,
            "max_dt": 0.03,
            "max_steps": 2000,
            "episode_duration": 10,
            "time_sampling_kwargs": {"tail_p": 0.99},
            "return_reward_increment": True,
            "n_envs": 4,
        }
        irregular = build_regime_env_kwargs(train, REGIMES[0])
        uniform = build_regime_env_kwargs(train, REGIMES[1])

        self.assertEqual(irregular["time_sampling"], "irregular")
        self.assertEqual(irregular["dt"], 0.02)
        self.assertEqual(irregular["max_steps"], 5001)
        self.assertEqual(irregular["time_sampling_kwargs"], {"tail_p": 0.99})
        self.assertFalse(irregular["return_reward_increment"])
        self.assertNotIn("n_envs", irregular)

        self.assertEqual(uniform["time_sampling"], "uniform")
        self.assertEqual(uniform["dt"], 0.01)
        self.assertEqual(uniform["max_steps"], 1000)
        self.assertEqual(uniform["time_sampling_kwargs"], {"dist": "uniform"})
        self.assertEqual(train["return_reward_increment"], True)

    def test_manifest_paths_are_explicit_and_relative_to_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "checkpoints.json"
            manifest.write_text(
                json.dumps(
                    {
                        "checkpoints": [
                            {
                                "env_id": "cartpole-swingup",
                                "mode": "final_mf",
                                "checkpoint_path": "seed_0/final_model.pth",
                                "train_seed": 0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            specs = load_checkpoint_manifest(manifest)

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].checkpoint_path, root / "seed_0/final_model.pth")
        self.assertEqual(specs[0].metadata, {"train_seed": 0})
        self.assertEqual(_checkpoint_train_seed(specs[0]), 0)

    def test_acrobot_metrics_use_physical_time_and_last_second_overlap(self):
        samples = [
            _sample(
                0.0,
                0.4,
                0.0,
                [0, 0, 0, 0],
                acrobot_tip_distance=1.0,
                acrobot_progress=0.75,
                acrobot_tip_height=1.0,
            ),
            _sample(
                0.4,
                0.6,
                1.0,
                [0, 0, 0, 0],
                acrobot_tip_distance=0.1,
                acrobot_progress=0.975,
                acrobot_tip_height=3.9,
            ),
            _sample(
                1.0,
                0.5,
                1.0,
                [0, 0, 0, 0],
                acrobot_tip_distance=0.05,
                acrobot_progress=0.9875,
                acrobot_tip_height=4.0,
            ),
            _sample(
                1.5,
                0.5,
                0.0,
                [0, 0, 0, 0],
                acrobot_tip_distance=0.3,
                acrobot_progress=0.925,
                acrobot_tip_height=3.7,
            ),
        ]
        result = summarize_episode(
            env_id="acrobot-swingup-v2",
            initial_obs=np.zeros(4, dtype=np.float32),
            reset_info={
                "acrobot_tip_distance": 4.0,
                "acrobot_progress": 0.0,
                "acrobot_tip_height": 0.0,
            },
            samples=samples,
            terminated=False,
            truncated=True,
        )

        self.assertAlmostEqual(result["time_weighted_normalized_reward"], 0.55)
        self.assertAlmostEqual(result["uniform_0p01_equivalent_return"], 110.0)
        self.assertAlmostEqual(result["acrobot_time_weighted_hit_occupancy"], 0.55)
        self.assertAlmostEqual(result["acrobot_last_1s_hit_occupancy"], 0.5)
        self.assertFalse(result["acrobot_sustained_hit"])
        self.assertAlmostEqual(result["acrobot_max_sustained_hit_seconds"], 0.5)
        self.assertAlmostEqual(result["acrobot_time_to_first_hit"], 1.0)
        self.assertIsNone(result["acrobot_time_to_sustained_hit"])
        self.assertAlmostEqual(result["acrobot_min_tip_distance"], 0.05)
        self.assertAlmostEqual(result["acrobot_max_progress"], 0.9875)
        self.assertAlmostEqual(result["acrobot_max_tip_height"], 4.0)

    def test_cartpole_metrics_use_raw_angle_and_sparse_target_thresholds(self):
        upright_angle = math.acos(0.999)
        samples = [
            _sample(0.0, 0.25, 0.2, [0.4, math.pi, 0, 0]),
            _sample(0.25, 0.75, 0.8, [0.1, upright_angle, 0, 0]),
        ]
        result = summarize_episode(
            env_id="cartpole-swingup",
            initial_obs=np.asarray([0.0, math.pi, 0.0, 0.0]),
            reset_info={},
            samples=samples,
            terminated=False,
            truncated=True,
        )

        self.assertAlmostEqual(result["time_weighted_normalized_reward"], 0.65)
        self.assertAlmostEqual(
            result["cartpole_time_weighted_upright_occupancy"], 0.75
        )
        self.assertAlmostEqual(
            result["cartpole_time_weighted_centered_occupancy"], 0.75
        )
        self.assertAlmostEqual(
            result["cartpole_time_weighted_upright_centered_occupancy"], 0.75
        )
        self.assertEqual(result["cartpole_time_to_first_centered"], 0.0)
        self.assertAlmostEqual(result["cartpole_time_to_first_upright"], 1.0)
        self.assertAlmostEqual(
            result["cartpole_time_to_first_upright_centered"], 1.0
        )

    def test_checkpoint_validation_rejects_whole_network_mismatches(self):
        import torch as th

        state = {"weight": th.ones(1)}
        model = SimpleNamespace(
            q_nets=[object(), object()],
            q_target_nets=[object(), object()],
            actor_target=None,
            has_v_head=False,
        )
        payload = {
            "actor": state,
            "critics": [state, state],
            "critic_targets": [state, state],
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "final_model.pth"
            th.save(payload, checkpoint)
            _validate_checkpoint_architecture(model, checkpoint)

            th.save({**payload, "critics": [state]}, checkpoint)
            with self.assertRaisesRegex(ValueError, "configured model expects 2"):
                _validate_checkpoint_architecture(model, checkpoint)

            th.save({**payload, "v_net": state, "v_target_net": state}, checkpoint)
            with self.assertRaisesRegex(ValueError, "unexpected"):
                _validate_checkpoint_architecture(model, checkpoint)


if __name__ == "__main__":
    unittest.main()
