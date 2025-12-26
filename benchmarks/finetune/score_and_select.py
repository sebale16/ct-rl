# benchmarks/finetune/score_and_select.py

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_eval_npz(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    ts = data["timesteps"].astype(int)
    results = data["results"]  # shape: [n_eval, n_eval_episodes]
    mean = results.mean(axis=1)
    return ts, mean


def auc(ts, y):
    return float(np.trapz(y, ts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage_log_root", required=True, help="e.g. logs/finetune/stage1"
    )
    parser.add_argument(
        "--hyperparams_dir_in", required=True, help="e.g. benchmarks/hyperparams"
    )
    parser.add_argument(
        "--hyperparams_dir_out",
        required=True,
        help="e.g. benchmarks/hyperparams_gen/stage2",
    )
    parser.add_argument("--keep_topk", type=int, required=True)
    parser.add_argument("--metric", choices=["auc", "final"], default="auc")
    args = parser.parse_args()

    log_root = Path(args.stage_log_root)
    hyperparams_in = Path(args.hyperparams_dir_in)
    hyperparams_out = Path(args.hyperparams_dir_out)
    hyperparams_out.mkdir(parents=True, exist_ok=True)

    rows = []
    for npz in log_root.rglob("eval/evaluations.npz"):
        # logs/<stage>/<algo>/<env_id>/<mode>/seed_<seed>/<run_name>/eval/evaluations.npz
        run_dir = npz.parents[1]
        seed_dir = run_dir.parent
        mode_dir = seed_dir.parent
        env_dir = mode_dir.parent
        algo_dir = env_dir.parent

        algo = algo_dir.name
        env_id = env_dir.name
        mode = mode_dir.name
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except Exception:
            continue

        ts, mean = load_eval_npz(npz)
        score = auc(ts, mean) if args.metric == "auc" else float(mean[-1])
        rows.append(dict(algo=algo, env_id=env_id, mode=mode, seed=seed, score=score))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No eval files found under: {log_root}")

    agg = df.groupby(["algo", "env_id", "mode"], as_index=False)["score"].mean()
    winners = (
        agg.sort_values(["algo", "env_id", "score"], ascending=[True, True, False])
        .groupby(["algo", "env_id"], as_index=False)
        .head(args.keep_topk)
    )

    winners.to_csv(hyperparams_out / "WINNERS.csv", index=False)
    print("Wrote winners summary ->", hyperparams_out / "WINNERS.csv")

    # Filter each <algo>.csv to only winning (env_id, mode) pairs
    for algo in winners["algo"].unique():
        csv_in = hyperparams_in / f"{algo}.csv"
        if not csv_in.exists():
            print(f"[WARN] missing {csv_in}, skipping")
            continue

        base = pd.read_csv(csv_in)
        keep_pairs = set(
            zip(
                winners[winners["algo"] == algo]["env_id"],
                winners[winners["algo"] == algo]["mode"],
            )
        )

        mask = base.apply(lambda r: (r["env_id"], r["mode"]) in keep_pairs, axis=1)
        out_csv = hyperparams_out / f"{algo}.csv"
        base[mask].to_csv(out_csv, index=False)
        print(f"Wrote filtered CSV -> {out_csv}")


if __name__ == "__main__":
    main()
