"""Paired fixed-label / moving-HJB-label regression diagnostic for Stage A.

Collect two independent fixed state sets before fitting. Clone the same value,
target, and Adam state into both arms and feed them identical minibatches.
"""
from __future__ import annotations

from copy import deepcopy
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.run_acrobot_stage_a import _bank, build_agent, evaluate, parser as stage_a_parser
from environment.acrobot_stage_a import AcrobotStageAEnv


def parser():
    p = stage_a_parser()
    p.description = __doc__
    # Reuse CSV presets and explicit CLI overrides from the training runner.
    # Here a checkpoint initializes BOTH training arms, rather than eval only.
    for action in p._actions:
        if action.dest == "checkpoint":
            action.help = "initialize both arms, including Adam state, from this Stage A checkpoint"
        elif action.dest == "buffer_size":
            action.help = "unused by this fixed-dataset diagnostic"
    p.add_argument("--train-states", type=int, default=4096)
    p.add_argument("--heldout-states", type=int, default=2048)
    p.add_argument("--collection-steps", type=int, default=4096,
                   help="frozen-policy decisions per split before fitting; 0 uses only reset states")
    p.add_argument("--dataset-seed", type=int, default=10000)
    p.add_argument("--batch-seed", type=int, default=10001)
    p.add_argument("--probe-batch-size", type=int, default=512)
    p.add_argument("--skip-rollouts", action="store_true",
                   help="skip control evaluation during fitting; still collect the fixed dataset")
    return p


def collect_states(agent, env, *, count, steps, seed, exploration_std):
    """Half reset collocation, half states from independent frozen-policy rolls.

    Call separately for train and held-out splits: no trajectory is split into
    neighboring training/held-out rows. Truncations are not failure boundaries.
    """
    rng = np.random.default_rng(seed)
    local_count = count // 2 if steps else count
    local = env.canonical(env.sample_qv(rng, local_count))
    visited, terminal = [], []
    seconds = 0.
    episodes = 0
    if steps:
        z, _ = env.reset(seed=seed)
        episodes = 1
        for step in range(steps):
            action = agent.act(z, deterministic=agent.config.temperature == 0, rng=rng)
            if agent.config.temperature == 0:
                action = np.clip(action + rng.normal(0., exploration_std, 1), -1., 1.)
            next_z, _, terminated, truncated, info = env.step(action)
            visited.extend((z, next_z))
            terminal.extend((False, terminated))
            seconds += info["dt_used"]
            z = next_z
            if (terminated or truncated) and step + 1 < steps:
                z, _ = env.reset()
                episodes += 1
        # Consecutive transitions repeat their shared state. Remove these
        # duplicates before sampling a fixed dataset without replacement.
        visited, indices = np.unique(np.asarray(visited), axis=0, return_index=True)
        terminal = np.asarray(terminal)[indices]
        needed = count - local_count
        if len(visited) < needed:
            raise ValueError(f"only {len(visited)} distinct rollout states for {needed} requested; "
                             "increase --collection-steps or reduce dataset sizes")
        choice = rng.choice(len(visited), needed, replace=False)
        states = np.concatenate((local, visited[choice]))
        mask = np.r_[np.zeros(local_count, dtype=bool), terminal[choice]]
    else:
        states, mask = local, np.zeros(count, dtype=bool)
    return {"states": states, "terminal": mask,
            "rollout": np.arange(count) >= local_count}, {
                "seed": int(seed), "collection_steps": steps, "collection_physical_seconds": seconds,
                "trajectories": episodes, "reset_states": local_count,
                "rollout_states": count - local_count, "terminal_states": int(mask.sum())}


def labels_for(agent, dataset, failure_value, chunk_size):
    labels = []
    for start in range(0, len(dataset["states"]), chunk_size):
        end = start + chunk_size
        y, _, _ = agent.fitted_targets(dataset["states"][start:end],
                                       terminal_mask=dataset["terminal"][start:end], terminal_value=failure_value)
        labels.append(y.cpu().numpy())
    return np.concatenate(labels)


def probe(agent, dataset, labels, failure_value, chunk_size):
    """State derivatives and HJB residuals use the ONLINE deployed value."""
    arrays = {key: [] for key in ("value", "eta", "gradient_norm", "hjb", "target_value")}
    for start in range(0, len(dataset["states"]), chunk_size):
        z = dataset["states"][start:start + chunk_size]
        states, values, gradient = agent.value_gradient(z)
        eta = gradient[:, 3]
        hjb = (-agent.reward.state_cost(states, agent.oracle)
               + (gradient * agent.oracle.drift(states)).sum(-1)
               - agent.config.discount_rate * values + agent.soft_action_score(eta))
        with torch.no_grad():
            target_value = agent.target(states)
        for key, value in zip(arrays, (values, eta, gradient.norm(dim=-1), hjb, target_value)):
            arrays[key].append(value.cpu().numpy())
    arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    arrays["labels"] = labels
    arrays["action"] = np.clip(arrays["eta"] / (agent.reward.effort_weight * agent.oracle.torque_limit), -1., 1.)
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("non-finite fixed-state probe")

    def rms(values, mask=None):
        selected = np.asarray(values, dtype=np.float64) if mask is None else np.asarray(values, dtype=np.float64)[mask]
        return float(np.sqrt(np.mean(selected**2))) if selected.size else None

    error = arrays["value"] - labels
    terminal = dataset["terminal"]
    metrics = {
        "label_rmse": rms(error),
        "initial_label_rmse": rms(arrays["value"] - dataset["initial_labels"]),
        "label_drift_rms": rms(labels - dataset["initial_labels"]),
        "interior_label_rmse": rms(error, ~terminal),
        "boundary_label_rmse": rms(error, terminal),
        "reset_label_rmse": rms(error, ~dataset["rollout"]),
        "rollout_label_rmse": rms(error, dataset["rollout"]),
        "boundary_value_rmse": rms(arrays["value"] - failure_value, terminal),
        "hjb_rms": rms(arrays["hjb"], ~terminal),
        "target_gap_rms": rms(arrays["value"] - arrays["target_value"]),
        "value_abs_max": float(np.abs(arrays["value"]).max()),
        "eta_abs_mean": float(np.abs(arrays["eta"]).mean()),
        "eta_abs_max": float(np.abs(arrays["eta"]).max()),
        # The mean is dragged across the saturation threshold by a right tail
        # long before the bulk saturates, so carry the shape of |eta| too.
        "eta_abs_p50": float(np.percentile(np.abs(arrays["eta"]), 50)),
        "eta_abs_p90": float(np.percentile(np.abs(arrays["eta"]), 90)),
        "eta_abs_p99": float(np.percentile(np.abs(arrays["eta"]), 99)),
        "state_gradient_norm_mean": float(arrays["gradient_norm"].mean()),
        "state_gradient_norm_max": float(arrays["gradient_norm"].max()),
        "saturation_fraction": float(np.mean(np.abs(arrays["action"]) >= 1 - 1e-6)),
        "terminal_fraction": float(terminal.mean()),
    }
    return metrics, arrays


def batch_indices(dataset, size, rng):
    """Preserve Stage A's half-collocation / half-replay minibatch mix."""
    local = np.flatnonzero(~dataset["rollout"])
    replay = np.flatnonzero(dataset["rollout"])
    if not len(replay):
        return rng.choice(local, size, replace=True)
    return np.r_[rng.choice(local, size // 2, replace=True),
                 rng.choice(replay, size - size // 2, replace=True)]


def run(args):
    for name in ("updates", "batch_size", "eval_every", "eval_episodes", "threads",
                 "probe_batch_size", "train_states", "heldout_states"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if min(args.train_states, args.heldout_states, args.batch_size) < 2:
        raise ValueError("dataset sizes and batch_size must be at least two")
    if args.collection_steps < 0 or not np.isfinite(args.exploration_std) or args.exploration_std < 0:
        raise ValueError("collection_steps and exploration_std must be nonnegative and finite")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {args.output}")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    initial, env_config = build_agent(args)
    if initial.config.auto_temperature:
        raise ValueError("fixed/moving-label diagnostic requires fixed temperature to isolate value-target feedback; "
                         "use --no-auto-temperature with a fresh soft run or a fixed-temperature checkpoint")
    train_bank, train_source = _bank(args.incoming_states)
    heldout_bank, heldout_source = _bank(args.incoming_eval_states)
    if (train_bank is None) != (heldout_bank is None):
        raise ValueError("incoming diagnostics require separate training and held-out banks")
    if train_bank is not None and {x.tobytes() for x in train_bank} & {x.tobytes() for x in heldout_bank}:
        raise ValueError("training and held-out incoming banks overlap")
    seeds = [int(child.generate_state(1)[0]) for child in np.random.SeedSequence(args.dataset_seed).spawn(2)]
    if any(args.eval_seed <= seed < args.eval_seed + args.eval_episodes for seed in seeds):
        raise ValueError("collection and control-evaluation reset seeds overlap; change --dataset-seed")
    datasets, collection = {}, {}
    for split, count, bank, seed in zip(("train", "heldout"), (args.train_states, args.heldout_states),
                                       (train_bank, heldout_bank), seeds):
        env = AcrobotStageAEnv(env_config, initial.oracle, initial.reward, incoming_states=bank)
        try:
            datasets[split], collection[split] = collect_states(
                initial, env, count=count, steps=args.collection_steps, seed=seed,
                exploration_std=args.exploration_std)
            failure_value = env.failure_value
        finally:
            env.close()
        datasets[split]["initial_labels"] = labels_for(initial, datasets[split], failure_value, args.probe_batch_size)
    if {x.tobytes() for x in datasets["train"]["states"]} & {x.tobytes() for x in datasets["heldout"]["states"]}:
        raise ValueError("training and held-out state sets overlap; use nondegenerate resets or a different dataset seed")

    args.output.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output / "dataset.npz"
    np.savez_compressed(dataset_path, **{f"{split}_{key}": value for split, data in datasets.items() for key, value in data.items()},
                        frame="downward_vertical_canonical", failure_value=failure_value)
    metadata = {
        "diagnostic": "paired_fixed_and_moving_hjb_labels_v1",
        "environment": asdict(env_config), "oracle": asdict(initial.oracle), "reward": asdict(initial.reward),
        "value_flow": asdict(initial.config), "initial_updates": initial.updates,
        "reward_scale": initial.metadata.get("reward_scale", 1.),
        "unscaled_reward": initial.metadata.get("unscaled_reward", asdict(initial.reward)),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "hyperparams": initial.metadata.get("hyperparams") if args.checkpoint else args.hyperparams_source,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "collection": collection, "training_incoming": train_source, "heldout_incoming": heldout_source,
        "learned_dynamics": False, "torch_version": str(torch.__version__),
        "evaluation_policy": "deterministic_mode", "minibatches": "identical_indices_in_both_arms",
    }
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    initial.save(args.output / "initial.pt", metadata=metadata)
    agents = {arm: deepcopy(initial) for arm in ("fixed", "moving")}
    for arm in agents:
        (args.output / arm).mkdir()
    eval_env = AcrobotStageAEnv(env_config, initial.oracle, initial.reward, incoming_states=heldout_bank)
    failures = {}
    rng = np.random.default_rng(args.batch_seed)
    train = datasets["train"]
    try:
        with (args.output / "metrics.csv").open("w", newline="") as csv_file, \
                (args.output / "probes.jsonl").open("w") as probe_log, \
                (args.output / "training.jsonl").open("w") as training_log, \
                (args.output / "evaluations.jsonl").open("w") as eval_log:
            writer = None

            def assessment(arm, update):
                nonlocal writer
                agent = agents[arm]
                directory = args.output / arm
                snapshots, report = {}, {}
                for split, data in datasets.items():
                    labels = data["initial_labels"] if arm == "fixed" else labels_for(agent, data, failure_value, args.probe_batch_size)
                    metrics, arrays = probe(agent, data, labels, failure_value, args.probe_batch_size)
                    row = {"arm": arm, "update": update, "split": split, **metrics}
                    if writer is None:
                        writer = csv.DictWriter(csv_file, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    probe_log.write(json.dumps(row, allow_nan=False) + "\n")
                    report[split] = metrics
                    snapshots.update({f"{split}_{key}": value for key, value in arrays.items()})
                csv_file.flush()
                probe_log.flush()
                np.savez_compressed(directory / f"probe_{update:08d}.npz", **snapshots)
                agent.save(directory / f"checkpoint_{update:08d}.pt", metadata={**metadata, "arm": arm, "diagnostic_update": update})
                if not args.skip_rollouts:
                    groups = [evaluate(agent, eval_env, seed=args.eval_seed, episodes=args.eval_episodes)]
                    if heldout_bank is not None:
                        groups.append(evaluate(agent, eval_env, seed=args.eval_seed, episodes=args.eval_episodes, incoming=True))
                    eval_log.write(json.dumps({"arm": arm, "update": update, "groups": groups}, allow_nan=False) + "\n")
                    eval_log.flush()
                    report["control"] = {group["group"]: group["mean"] for group in groups}
                print(json.dumps({"arm": arm, "update": update, **report}, allow_nan=False), flush=True)

            for update in range(args.updates + 1):
                indices = batch_indices(train, args.batch_size, rng) if update else None
                for arm, agent in agents.items():
                    if arm in failures:
                        continue
                    try:
                        if update:
                            if arm == "fixed":
                                metrics = agent.fit_labels(train["states"][indices], train["initial_labels"][indices], refresh_target=False)
                            else:
                                metrics = agent.update(train["states"][indices], terminal_mask=train["terminal"][indices], terminal_value=failure_value)
                            if update == 1 or update % 100 == 0 or update == args.updates:
                                training_log.write(json.dumps({"arm": arm, "update": update, **metrics}, allow_nan=False) + "\n")
                                training_log.flush()
                        if update % args.eval_every == 0 or update == args.updates:
                            assessment(arm, update)
                    except (FloatingPointError, RuntimeError) as exc:
                        # Do not discard the other arm if one becomes nonfinite.
                        # Other runtime errors (e.g. OOM) must not be called divergence.
                        if isinstance(exc, RuntimeError) and "non-finite" not in str(exc):
                            raise
                        failures[arm] = {"update": update, "error": str(exc)}
                        (args.output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
                        print(json.dumps({"arm": arm, "failure": failures[arm]}), flush=True)
                if len(failures) == len(agents):
                    break
    finally:
        eval_env.close()
    summary = {"status": "nonfinite_failure" if failures else "completed", "failures": failures,
               "completed_updates": {arm: agent.updates - initial.updates for arm, agent in agents.items()}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    result = run(parser().parse_args())
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
