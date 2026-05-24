# evaluations/evaluation_humanoid_stability.py
"""Report humanoid survival / fall metrics for each trained algorithm.

The DMC humanoid total return mixes "stayed upright" with "actually walked".
A policy that learns to stand but never steps forward can score similarly to
one that locomotes. This script rolls out each algorithm's best model and
reports physics-based stability metrics instead of (or alongside) the
aggregate return:

  - fall_rate           fraction of episodes where head height ever drops
                        below threshold (default 1.0 m; target stand height
                        in dm_control humanoid is 1.4 m).
  - survival_fraction   time-weighted fraction of each episode spent with
                        head height above threshold.
  - time_to_first_fall  mean time of the first below-threshold step over
                        episodes that fell.
  - min_head_height     mean of the per-episode minimum head height.

Usage:

  python -m evaluations.evaluation_humanoid_stability

Outputs CSVs to ``out/final_reports/humanoid_stability/``.
"""
from __future__ import annotations

import argparse
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from stable_baselines3.common.base_class import BaseAlgorithm

from environment.base import ContinuousEnv
from evaluations.evaluation_helpers import (
    ALGO_CLASS_MAP,
    create_evaluation_env_and_model,
    evaluate_policy_per_step,
    evaluate_sb3_policy_per_step,
)
from evaluations.evaluation_stats import load_best_models_for_eval
from evaluations.stability_metrics import (
    DEFAULT_HEAD_HEIGHT_THRESHOLD,
    aggregate_stability,
    compute_episode_stability,
    make_probe_fn,
)
from models.base import Model


warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
for name in ["gym", "gymnasium", "stable_baselines3", "matplotlib", "imageio", "PIL"]:
    logging.getLogger(name).setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"


DEFAULT_ENV_ID = "humanoid-walk"
DEFAULT_ALGOS: List[str] = [
    "ct_sac",
    "ct_td3",
    "sac",
    "td3",
    "ppo",
    "trpo",
    "cppo",
    "q_learning",
]


def _rollout_with_probe(
    model: Union[Model, BaseAlgorithm],
    env: ContinuousEnv,
    n_eval_episodes: int,
    probe_fn,
) -> Dict[str, Any]:
    if isinstance(model, BaseAlgorithm):
        return evaluate_sb3_policy_per_step(
            model,
            env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
            probe_fn=probe_fn,
        )
    return evaluate_policy_per_step(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        probe_fn=probe_fn,
    )


def _seed_model_dir_exists(*, algo: str, env_id: str, seed: int, root: Path) -> bool:
    base = root / algo / env_id / "top" / f"seed_{seed}" / "best_model"
    cls = ALGO_CLASS_MAP[algo]
    if hasattr(cls, "load") and "stable_baselines3" in str(cls):
        return (base / "best_model.zip").exists()
    return (base / "best_model.pth").exists()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default=DEFAULT_ENV_ID)
    parser.add_argument(
        "--mode",
        default="regular",
        choices=["regular", "top"],
        help="Eval timing mode (regular = dm_control default dt).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_HEAD_HEIGHT_THRESHOLD,
        help="Head-height (m) below which the agent is considered fallen.",
    )
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    parser.add_argument("--algos", nargs="+", default=DEFAULT_ALGOS)
    parser.add_argument("--saved_models_root", default="saved_models")
    parser.add_argument(
        "--out_dir", default="out/final_reports/humanoid_stability"
    )
    args = parser.parse_args()

    saved_root = Path(args.saved_models_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_fn = make_probe_fn(args.env_id)
    if probe_fn is None:
        raise SystemExit(
            f"No stability probe registered for env_id={args.env_id!r}. "
            "Currently only humanoid-* envs are supported."
        )

    rows: List[Dict[str, Any]] = []

    for seed in args.seeds:
        available = [
            a
            for a in args.algos
            if _seed_model_dir_exists(
                algo=a, env_id=args.env_id, seed=seed, root=saved_root
            )
        ]
        if not available:
            print(f"[seed_{seed}] skip (no models)")
            continue

        print(f"[seed_{seed}] loading {len(available)} models")
        models = load_best_models_for_eval(
            algos=available,
            env_id=args.env_id,
            mode="top",
            seed=seed,
            saved_models_root=saved_root,
            quarters=None,
        )

        env, _ = create_evaluation_env_and_model(
            env_id=args.env_id,
            model_class=ALGO_CLASS_MAP["ct_sac"],
            seed=seed,
            algo="ct_sac",
            mode=args.mode,
            quarters=None,
        )

        try:
            for algo, model in models.items():
                out = _rollout_with_probe(
                    model, env, args.n_eval_episodes, probe_fn
                )
                heights_per_ep = out.get("episode_step_probes", [])
                ts_per_ep = out["episode_timestamps"]
                returns_per_ep = out["episode_returns"]

                per_episode_metrics = [
                    compute_episode_stability(
                        heights=h, timestamps=ts, threshold=args.threshold
                    )
                    for h, ts in zip(heights_per_ep, ts_per_ep)
                ]
                agg = aggregate_stability(per_episode_metrics)

                row = {
                    "env_id": args.env_id,
                    "algo": algo,
                    "train_seed": seed,
                    "n_eval_episodes": args.n_eval_episodes,
                    "threshold_m": args.threshold,
                    "mean_return": float(np.mean(returns_per_ep)),
                    **agg,
                }
                rows.append(row)
                print(
                    f"  {algo:12s} "
                    f"return={row['mean_return']:7.2f}  "
                    f"fall_rate={row['fall_rate']:.2f}  "
                    f"surv={row['mean_survival_fraction']:.2f}  "
                    f"min_h={row['mean_min_head_height']:.2f}m"
                )
        finally:
            try:
                env.close()
            except Exception:
                pass

    if not rows:
        print("[done] No results collected.")
        return

    df = pd.DataFrame(rows)
    per_seed_csv = out_dir / "per_seed_stability.csv"
    df.to_csv(per_seed_csv, index=False)

    summary = (
        df.groupby(["env_id", "algo"])[
            [
                "mean_return",
                "fall_rate",
                "mean_survival_fraction",
                "mean_time_to_first_fall",
                "mean_min_head_height",
            ]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary_csv = out_dir / "summary_stability.csv"
    summary.to_csv(summary_csv, index=False)

    pivot_fall = df.pivot_table(
        index="algo", values="fall_rate", aggfunc=["mean", "std"]
    )
    pivot_surv = df.pivot_table(
        index="algo", values="mean_survival_fraction", aggfunc=["mean", "std"]
    )
    print("\n=== Fall rate (mean, std) over seeds ===")
    print(pivot_fall.round(3).to_string())
    print("\n=== Survival fraction (mean, std) over seeds ===")
    print(pivot_surv.round(3).to_string())
    print(f"\n[saved] {per_seed_csv}")
    print(f"[saved] {summary_csv}")


if __name__ == "__main__":
    main()
