"""Periodic target refresh for Stage A: freeze, fit, measure, refresh.

One round is:

1. Freeze the target network and compute HJB labels for the whole fixed
   dataset from it, training and held-out split alike.
2. Take ``--refresh-period`` optimizer steps against those frozen labels,
   minibatched, with no target motion at all in between.
3. Probe training and held-out label error and the change in eta_V, then hard
   copy the online network into the target and start the next round.

This sits between the two arms of ``check_acrobot_stage_a_targets``: that
diagnostic compared labels frozen for the whole run against labels moving every
single step. Here the refresh period is the knob, so the question is whether a
label that holds still for 100 or 500 steps is enough to keep the value gradient
bounded, or whether any refresh at all reinstates the divergence.

The target copy is hard, not Polyak: the point is a label that is exactly
constant within a round. ``--target-rate`` is therefore unused here, and the
per-round refresh period replaces it as the target's timescale.
"""
from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.check_acrobot_stage_a_targets import (
    batch_indices, collect_states, labels_for, probe, parser as checker_parser)
from benchmarks.run_acrobot_stage_a import _bank, build_agent, evaluate
from environment.acrobot_stage_a import AcrobotStageAEnv


def parser():
    p = checker_parser()
    p.description = __doc__
    p.add_argument("--refresh-period", type=int, default=100,
                   help="optimizer steps against one frozen label set before the target is copied")
    return p


def run(args):
    if args.refresh_period <= 0:
        raise ValueError("refresh_period must be positive")
    if args.updates % args.refresh_period:
        raise ValueError("updates must be a whole number of refresh periods")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {args.output}")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    agent, env_config = build_agent(args)
    train_bank, train_source = _bank(args.incoming_states)
    heldout_bank, heldout_source = _bank(args.incoming_eval_states)
    if (train_bank is None) != (heldout_bank is None):
        raise ValueError("incoming diagnostics require separate training and held-out banks")
    seeds = [int(child.generate_state(1)[0])
             for child in np.random.SeedSequence(args.dataset_seed).spawn(2)]
    datasets, collection = {}, {}
    for split, count, bank, seed in zip(("train", "heldout"),
                                        (args.train_states, args.heldout_states),
                                        (train_bank, heldout_bank), seeds):
        env = AcrobotStageAEnv(env_config, agent.oracle, agent.reward, incoming_states=bank)
        try:
            datasets[split], collection[split] = collect_states(
                agent, env, count=count, steps=args.collection_steps, seed=seed,
                exploration_std=args.exploration_std)
            failure_value = env.failure_value
        finally:
            env.close()
        datasets[split]["initial_labels"] = labels_for(
            agent, datasets[split], failure_value, args.probe_batch_size)
    if ({x.tobytes() for x in datasets["train"]["states"]}
            & {x.tobytes() for x in datasets["heldout"]["states"]}):
        raise ValueError("training and held-out state sets overlap; change --dataset-seed")

    args.output.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output / "dataset.npz"
    np.savez_compressed(dataset_path, failure_value=failure_value,
                        **{f"{s}_{k}": v for s, d in datasets.items() for k, v in d.items()})
    metadata = {
        "diagnostic": "periodic_target_refresh_v1",
        "refresh_period": args.refresh_period, "target_copy": "hard",
        "environment": asdict(env_config), "oracle": asdict(agent.oracle),
        "reward": asdict(agent.reward), "value_flow": asdict(agent.config),
        "reward_scale": agent.metadata.get("reward_scale", 1.),
        "auto_temperature": bool(agent.config.auto_temperature),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "hyperparams": args.hyperparams_source,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "collection": collection, "learned_dynamics": False,
        "torch_version": str(torch.__version__),
    }
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")

    eval_env = AcrobotStageAEnv(env_config, agent.oracle, agent.reward, incoming_states=heldout_bank)
    rng = np.random.default_rng(args.batch_seed)
    train = datasets["train"]
    rounds = args.updates // args.refresh_period
    failure = None
    try:
        with (args.output / "rounds.csv").open("w", newline="") as csv_file, \
                (args.output / "rounds.jsonl").open("w") as round_log, \
                (args.output / "evaluations.jsonl").open("w") as eval_log:
            writer = None
            previous_eta = None
            for index in range(rounds):
                update = index * args.refresh_period
                # 1. Freeze the target and label the whole dataset from it.
                labels = {s: labels_for(agent, d, failure_value, args.probe_batch_size)
                          for s, d in datasets.items()}
                before = {s: probe(agent, datasets[s], labels[s], failure_value,
                                   args.probe_batch_size)[0] for s in datasets}
                # 2. Fit those frozen labels; the target does not move at all.
                try:
                    for _ in range(args.refresh_period):
                        pick = batch_indices(train, args.batch_size, rng)
                        agent.fit_labels(train["states"][pick], labels["train"][pick],
                                         refresh_target=False)
                        if agent.config.auto_temperature:
                            # Keep stage_a_soft_auto faithful. Note this makes the
                            # frozen labels stale in temperature as well as in value.
                            agent.update_temperature(train["states"][pick],
                                                     terminal_mask=train["terminal"][pick])
                except (FloatingPointError, RuntimeError) as exc:
                    if isinstance(exc, RuntimeError) and "non-finite" not in str(exc):
                        raise
                    failure = {"round": index, "update": update, "error": str(exc)}
                    break
                # 3. Measure against the same frozen labels, then refresh.
                after = {s: probe(agent, datasets[s], labels[s], failure_value,
                                  args.probe_batch_size)[0] for s in datasets}
                done = update + args.refresh_period
                for split in datasets:
                    row = {"round": index, "update": done, "split": split,
                           "refresh_period": args.refresh_period,
                           "label_rmse_before": before[split]["label_rmse"],
                           "label_rmse_after": after[split]["label_rmse"],
                           "label_gap_closed": (1 - after[split]["label_rmse"]
                                                / before[split]["label_rmse"]
                                                if before[split]["label_rmse"] else None),
                           "label_drift_rms": before[split]["label_drift_rms"],
                           "eta_abs_mean_before": before[split]["eta_abs_mean"],
                           "eta_abs_mean_after": after[split]["eta_abs_mean"],
                           "eta_abs_mean_delta": after[split]["eta_abs_mean"] - before[split]["eta_abs_mean"],
                           "eta_abs_max_after": after[split]["eta_abs_max"],
                           "eta_abs_p50_after": after[split]["eta_abs_p50"],
                           "eta_abs_p90_after": after[split]["eta_abs_p90"],
                           "eta_abs_p99_after": after[split]["eta_abs_p99"],
                           "saturation_fraction_after": after[split]["saturation_fraction"],
                           "hjb_rms_after": after[split]["hjb_rms"],
                           "target_gap_rms_after": after[split]["target_gap_rms"],
                           "value_abs_max_after": after[split]["value_abs_max"],
                           "temperature": agent.temperature}
                    if writer is None:
                        writer = csv.DictWriter(csv_file, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    round_log.write(json.dumps(row, allow_nan=False) + "\n")
                csv_file.flush()
                round_log.flush()
                growth = (after["train"]["eta_abs_mean"] / previous_eta
                          if previous_eta else None)
                previous_eta = after["train"]["eta_abs_mean"]
                print(json.dumps({"round": index, "update": done,
                                  "eta_abs_mean": after["train"]["eta_abs_mean"],
                                  "eta_growth_per_round": growth,
                                  "train_label_rmse_after": after["train"]["label_rmse"],
                                  "heldout_label_rmse_after": after["heldout"]["label_rmse"],
                                  "saturation": after["train"]["saturation_fraction"]},
                                 allow_nan=False), flush=True)
                if not args.skip_rollouts and (done % args.eval_every == 0 or index == rounds - 1):
                    groups = [evaluate(agent, eval_env, seed=args.eval_seed, episodes=args.eval_episodes)]
                    eval_log.write(json.dumps({"round": index, "update": done, "groups": groups},
                                              allow_nan=False) + "\n")
                    eval_log.flush()
                # Hard copy: the next round's labels come from exactly this network.
                agent.target.load_state_dict(agent.value.state_dict())
                agent.save(args.output / "last.pt", metadata=metadata)
    finally:
        eval_env.close()
    summary = {"status": "nonfinite_failure" if failure else "completed",
               "failure": failure, "rounds": rounds,
               "refresh_period": args.refresh_period}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    if run(parser().parse_args())["failure"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
