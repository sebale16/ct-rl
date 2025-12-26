import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def try_load_eval(npz_path: Path):
    # EvalCallback format: timesteps, results, ep_lengths
    try:
        data = np.load(npz_path, allow_pickle=True)
        for k in data.files:
            if k == "timesteps":
                ts = data[k].astype(int)
            elif k == "results":
                results = np.asarray(data[k])
        data.close()
        mean = results.mean(axis=1)
        return ts, mean
    except Exception:
        return None


def aggregate(eval_files):
    # timestep -> list of (seed-run mean reward at that timestep)
    bucket = defaultdict(list)

    for f in eval_files:
        out = try_load_eval(f)
        if out is None:
            continue
        ts, mean = out
        for t, m in zip(ts, mean):
            bucket[int(t)].append(float(m))

    if not bucket:
        return None

    xs = np.array(sorted(bucket.keys()), dtype=int)
    ys = np.array([np.mean(bucket[t]) for t in xs], dtype=float)
    ystd = np.array([np.std(bucket[t]) for t in xs], dtype=float)
    n = np.array([len(bucket[t]) for t in xs], dtype=int)
    return xs, ys, ystd, n


def run_live_compare(
    groups: dict[str, str],
    *,
    pattern: str = "eval/evaluations.npz",
    out_path: str = "",
):
    """
    groups: {label: root_dir}. Each root_dir contains multiple seed runs.
    pattern: relative path pattern to find eval files under root_dir.
    """
    if not groups:
        raise ValueError("groups must be a non-empty dict[label -> path]")

    plt.ion()
    fig, ax = plt.subplots()

    ax.clear()
    ax.set_xlabel("timesteps")
    ax.set_ylabel("mean eval reward")
    ax.set_title("Live eval comparison (mean ± std over seeds)")

    any_plotted = False

    for label, root_str in groups.items():
        root = Path(root_str)
        eval_files = list(root.rglob(pattern))
        agg = aggregate(eval_files)

        if agg is None:
            ax.plot([], [], label=f"{label} (no data; files={len(eval_files)})")
            continue

        xs, ys, ystd, n = agg
        any_plotted = True
        ax.plot(xs, ys, label=f"{label} (runs={len(eval_files)}, last n={n[-1]})")
        ax.fill_between(xs, ys - ystd, ys + ystd, alpha=0.2)

    if not any_plotted:
        ax.text(
            0.5,
            0.5,
            "No evaluations found yet",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.legend(loc="best")
    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="cheetah-run")
    parser.add_argument("--algos", type=str, default="ct_sac, sac")
    args = parser.parse_args()

    env_id = args.env_id
    algos = [algo.strip() for algo in args.algos.split(",") if algo.strip()]

    GROUPS = {}
    for algo in algos:
        if algo.lower() in ["sac", "td3", "ppo", "trpo"]:
            GROUPS[algo.upper()] = f"logs/discrete_benchmarks/{algo.lower()}/{env_id}"
        else:
            GROUPS[algo.upper()] = f"logs/{algo.lower()}/{env_id}"

    run_live_compare(
        GROUPS,
        pattern="eval/evaluations.npz",
        out_path=f"out/plots/live_eval_{env_id}.png",
    )
