import csv
import tempfile
import unittest
from pathlib import Path

from common.utils import (
    load_ct_hyperparams_from_table,
    load_sb3_hyperparams_from_table,
)


class TestTaskKwargsConfig(unittest.TestCase):
    @staticmethod
    def _write_table(
        directory: str,
        filename: str,
        row: dict[str, str],
    ) -> None:
        path = Path(directory) / filename
        with path.open("w", newline="") as table:
            writer = csv.DictWriter(table, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    def test_ct_loader_parses_task_kwargs_and_nested_booleans(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_table(
                directory,
                "ct_sac.csv",
                {
                    "mode": "xk_r2",
                    "env_id": "acrobot-swingup-xk",
                    "total_timesteps": "1000",
                    "env_task_kwargs": (
                        "reward_kind=r2;eta=0.125;release_start=true;"
                        "damping=0;label=eta_sweep"
                    ),
                    "env_time_sampling_kwargs": "randomize=false;tail_p=0.99",
                    "env_raw_state_obs": "true",
                },
            )

            _, env_kwargs, _, _, _ = load_ct_hyperparams_from_table(
                "ct_sac",
                "acrobot-swingup-xk",
                "xk_r2",
                hyperparams_dir=directory,
            )

        self.assertEqual(
            env_kwargs["task_kwargs"],
            {
                "reward_kind": "r2",
                "eta": 0.125,
                "release_start": True,
                "damping": 0,
                "label": "eta_sweep",
            },
        )
        self.assertEqual(
            env_kwargs["time_sampling_kwargs"],
            {"randomize": False, "tail_p": 0.99},
        )
        # Keep historical parsing for ordinary scalar columns: this change is
        # limited to dictionary-valued environment configuration.
        self.assertEqual(env_kwargs["raw_state_obs"], "true")

    def test_sb3_loader_parses_task_kwargs_the_same_way(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_table(
                directory,
                "sac.csv",
                {
                    "mode": "xk_r1",
                    "env_id": "acrobot-swingup-xk",
                    "total_timesteps": "1000",
                    "env_task_kwargs": (
                        "reward_kind=r1;release_start=TRUE;paper_start=False;"
                        "angle_noise=1e-3"
                    ),
                    "policy_activation_fn": "ReLU",
                },
            )

            _, env_meta, _, _, _ = load_sb3_hyperparams_from_table(
                "sac",
                "acrobot-swingup-xk",
                "xk_r1",
                hyperparams_dir=directory,
            )

        self.assertEqual(
            env_meta["task_kwargs"],
            {
                "reward_kind": "r1",
                "release_start": True,
                "paper_start": False,
                "angle_noise": 1e-3,
            },
        )


if __name__ == "__main__":
    unittest.main()
