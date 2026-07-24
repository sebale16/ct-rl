from __future__ import annotations

import csv
import unittest
from pathlib import Path

import numpy as np

try:
    from environment.base import (
        generate_irregular_time_grid,
        generate_uniform_time_grid,
    )
except ImportError as exc:  # pragma: no cover - dependency-light environments
    ENV_IMPORT_ERROR = exc
else:
    ENV_IMPORT_ERROR = None


ROOT = Path(__file__).resolve().parents[1]
CT_TABLE = ROOT / "benchmarks" / "hyperparams" / "ct_sac.csv"
PPO_TABLE = ROOT / "benchmarks" / "hyperparams" / "ppo.csv"


def _matching_rows(path: Path, *, env_id: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            row for row in csv.DictReader(stream) if row["env_id"] == env_id
        ]


class AcrobotV41HorizonConfigTests(unittest.TestCase):
    def test_v41_rows_use_twenty_seconds_without_other_horizon_changes(self):
        ct_rows = _matching_rows(CT_TABLE, env_id="acrobot-swingup-v4.1")
        self.assertEqual(
            {row["mode"] for row in ct_rows}, {"final_mf", "fork_v41"}
        )
        for row in ct_rows:
            with self.subTest(framework="ct_sac", mode=row["mode"]):
                self.assertEqual(float(row["env_episode_duration"]), 20.0)
                self.assertEqual(int(row["env_max_steps"]), 5000)
                self.assertEqual(float(row["algo_gamma"]), 0.995)
                self.assertEqual(int(row["total_timesteps"]), 1_000_000)

        ppo_rows = _matching_rows(PPO_TABLE, env_id="acrobot-swingup-v4.1")
        self.assertEqual([row["mode"] for row in ppo_rows], ["final_mf"])
        row = ppo_rows[0]
        self.assertEqual(float(row["env_episode_duration"]), 20.0)
        self.assertEqual(int(row["env_max_steps"]), 2000)
        self.assertEqual(float(row["gamma"]), 0.995)
        self.assertEqual(int(row["total_timesteps"]), 1_000_000)
        self.assertEqual(int(row["n_steps"]), 2000)

    @unittest.skipIf(
        ENV_IMPORT_ERROR is not None,
        f"environment dependencies unavailable: {ENV_IMPORT_ERROR}",
    )
    def test_step_caps_reach_twenty_physical_seconds(self):
        uniform = generate_uniform_time_grid(20.0, 2000)
        self.assertEqual(len(uniform) - 1, 2000)
        self.assertAlmostEqual(float(uniform[-1]), 20.0)
        np.testing.assert_allclose(np.diff(uniform), 0.01)

        irregular_kwargs = {
            "min_dt": 0.002,
            "max_dt": 0.03,
            "mean_dt": 0.01,
            "physics_dt": 0.002,
            "time_sampling_kwargs": {
                "dist": "two_tail_uniform",
                "tail_p": 0.99,
                "tail_split": 0.9,
            },
        }
        for seed in range(32):
            with self.subTest(seed=seed):
                grid = generate_irregular_time_grid(
                    20.0,
                    5000,
                    rng=np.random.default_rng(seed),
                    **irregular_kwargs,
                )
                self.assertLessEqual(len(grid) - 1, 5000)
                self.assertAlmostEqual(float(grid[-1]), 20.0, places=9)


if __name__ == "__main__":
    unittest.main()
