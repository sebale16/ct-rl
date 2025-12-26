# benchmarks/finetune/make_manifest.py

import argparse
import json
from pathlib import Path

import pandas as pd

from benchmarks.finetune.stage_config import STAGES


def _parse_int_list(csv_str: str):
    return [int(x) for x in csv_str.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=STAGES.keys())
    parser.add_argument(
        "--algos", required=True, help="Comma-separated, e.g. 'ct_sac,ct_td3'"
    )
    parser.add_argument(
        "--env_ids",
        required=True,
        help="Comma-separated, e.g. 'cheetah-run,walker-walk'",
    )
    parser.add_argument("--hyperparams_dir", default="benchmarks/hyperparams")
    parser.add_argument(
        "--mode_prefix",
        default="",
        help="Optional prefix filter, e.g. 'irregular_dt_option_'",
    )
    parser.add_argument(
        "--runner_script",
        default="",
        help="Path to runner .py (relative to repo root). If empty, infer from algo prefix.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--log_root", default="logs/finetune")
    parser.add_argument("--save_root", default="saved_models/finetune")

    # Optional overrides (for smoke tests)
    parser.add_argument("--override_seeds", type=str, default="")
    parser.add_argument("--override_n_eval_episodes", type=int, default=-1)
    parser.add_argument("--override_timesteps_default", type=int, default=-1)
    parser.add_argument("--override_timesteps_humanoid", type=int, default=-1)

    args = parser.parse_args()

    stage_config = STAGES[args.stage]
    env_ids = [x.strip() for x in args.env_ids.split(",") if x.strip()]
    algos = [x.strip() for x in args.algos.split(",") if x.strip()]
    hyperparams_dir = Path(args.hyperparams_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seeds = stage_config["seeds"]
    if args.override_seeds:
        seeds = _parse_int_list(args.override_seeds)

    n_eval_episodes = stage_config["n_eval_episodes"]
    if args.override_n_eval_episodes >= 0:
        n_eval_episodes = int(args.override_n_eval_episodes)

    timesteps_default = stage_config["timesteps_default"]
    if args.override_timesteps_default >= 0:
        timesteps_default = int(args.override_timesteps_default)

    timesteps_override = stage_config["timesteps_overrides"].get(
        "humanoid-walk", timesteps_default
    )
    if args.override_timesteps_humanoid >= 0:
        timesteps_humanoid = int(args.override_timesteps_humanoid)

    jobs = []

    for algo in algos:
        csv_path = hyperparams_dir / f"{algo}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing hyperparams CSV: {csv_path}")

        df = pd.read_csv(csv_path)
        df = df[df["env_id"].isin(env_ids)]

        if args.mode_prefix:
            df = df[df["mode"].astype(str).str.startswith(args.mode_prefix)]

        # runner script inference if not provided
        if args.runner_script:
            runner_script = args.runner_script
        else:
            # heuristic: CT algos typically start with "ct_" or are "cpg/cppo/q_learning"
            is_ct = algo.startswith("ct_") or algo in {
                "cpg",
                "cppo",
                "q_learning",
            }
            runner_script = (
                "benchmarks/run_ct_rl.py" if is_ct else "benchmarks/run_discrete_rl.py"
            )

        modes_by_env = df.groupby("env_id")["mode"].unique().to_dict()

        for env_id in env_ids:
            modes = list(modes_by_env.get(env_id, []))
            if not modes:
                continue

            if env_id == "humanoid-walk":
                total_timesteps = timesteps_humanoid
            else:
                total_timesteps = timesteps_override.get(env_id, timesteps_default)

            for mode in modes:
                # ensure row exists for this (env_id, mode)
                if not ((df["env_id"] == env_id) & (df["mode"] == mode)).any():
                    continue

                for seed in seeds:
                    jobs.append(
                        dict(
                            runner_script=runner_script,
                            algo=algo,
                            env_id=env_id,
                            mode=mode,
                            seed=int(seed),
                            hyperparams_dir=str(hyperparams_dir),
                            total_timesteps=int(total_timesteps),
                            n_eval_episodes=int(n_eval_episodes),
                            log_root=str(Path(args.log_root) / args.stage),
                            save_root=str(Path(args.save_root) / args.stage),
                        )
                    )

    with out_path.open("w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")

    print(f"Wrote {len(jobs)} jobs -> {out_path}")


if __name__ == "__main__":
    main()
