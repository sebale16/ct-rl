# benchmarks/finetune/run_manifest_entry.py

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()

    lines = Path(args.manifest).read_text().splitlines()
    if args.index < 0 or args.index >= len(lines):
        raise IndexError(f"index={args.index} out of range (0..{len(lines)-1})")

    job = json.loads(lines[args.index])

    cmd = [
        "python",
        "-u",
        job["runner_script"],
        "--algos",
        job["algo"],
        "--env_id",
        job["env_id"],
        "--mode",
        job["mode"],
        "--seed",
        str(job["seed"]),
        "--hyperparams_dir",
        job["hyperparams_dir"],
        "--log_root",
        job["log_root"],
        "--save_root",
        job["save_root"],
        "--total_timesteps",
        str(job["total_timesteps"]),
        "--n_eval_episodes",
        str(job["n_eval_episodes"]),
        "--desc",
        "finetune",
    ]

    print("CMD:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
