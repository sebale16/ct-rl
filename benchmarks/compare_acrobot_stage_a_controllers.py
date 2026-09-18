"""Compare frozen labels, moving labels, and online collection for Stage A.

Run with --no-finite-horizon to match the continuing-objective checkpoints.
All arms start from identical value, target, and optimizer states. The first
update is shared; subsequent online batches use fresh resets and growing replay.
"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm
import torch

from benchmarks.check_acrobot_stage_a_targets import (
    batch_indices, collect_states, labels_for, parser as diagnostic_parser, probe,
)
from benchmarks.run_acrobot_stage_a import _bank, build_agent, evaluate
from environment.acrobot_stage_a import AcrobotStageAEnv


ARMS = ("fixed_labels", "moving_labels", "online")


def parser():
    p = diagnostic_parser()
    p.description = __doc__
    # Keep the stationary experiment callable both before and after the
    # separate finite-horizon extension to the shared Stage A parser.
    if not any(action.dest == "finite_horizon" for action in p._actions):
        p.add_argument("--finite-horizon", action=argparse.BooleanOptionalAction, default=False,
                       help="this comparison requires --no-finite-horizon")
    for action in p._actions:
        if action.dest == "buffer_size":
            action.help = "capacity of the online arm's visited-state replay"
        elif action.dest == "checkpoint":
            action.help = "initialize all three arms, including optimizer state, from a checkpoint"
    p.add_argument("--probe-every", type=int, default=100,
                   help="curvature, fixed-state probes, and checkpoint interval; includes update zero")
    p.add_argument("--seeds", type=int, nargs="+",
                   help="run each seed in output/seed_N; otherwise use --seed directly in output")
    return p


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def neighborhood(agent, config):
    """Fixed axis slices, independent of training, in physical q/v coordinates."""
    origin = np.array([np.pi, 0., 0., 0.])
    points = [origin]
    for dim, span in enumerate((config.angle_radius, config.angle_radius,
                                config.velocity_radius, config.velocity_radius)):
        for offset in np.linspace(-span, span, 41):
            if offset:
                state = origin.copy()
                state[dim] += offset
                points.append(state)
    qv = np.asarray(points)
    states = agent.oracle.canonical(torch.tensor(qv, dtype=torch.float64)).numpy().astype(np.float32)
    return qv, states


def upright_probe(agent, states, dt):
    """Float64 geometric derivatives plus actual float32 deployment/target terms.

    The spectral tests concern the smooth mathematical controller. Float32
    torque at nominal upright is measured separately: steep networks can lose
    their exact equilibrium numerically. No training module is cast or modified.
    """
    net = deepcopy(agent.value).cpu().double()
    origin = torch.tensor([np.pi, 0., 0., 0.], dtype=torch.float64)
    gradient = torch.autograd.functional.jacobian(net, origin).detach().numpy()
    hessian = torch.autograd.functional.hessian(net, origin).detach().numpy()
    transform = np.eye(4)
    transform[2:, 2:] = agent.oracle.mass(origin[:2]).numpy()
    hessian_qv = transform.T @ hessian @ transform
    eigenvalues = np.linalg.eigvalsh(hessian_qv)
    drift = torch.autograd.functional.jacobian(agent.oracle.drift, origin).numpy()
    torque_jacobian = hessian[3] / agent.reward.effort_weight
    continuous = drift.copy()
    continuous[3] += torque_jacobian
    poles = np.linalg.eigvals(continuous)
    augmented = np.zeros((5, 5))
    augmented[:4, :4], augmented[3, 4] = drift, 1.
    held = expm(dt * augmented)
    held_poles = np.linalg.eigvals(held[:4, :4] + np.outer(held[:4, 4], torque_jacobian))
    metrics = {
        "value_upright": float(net(origin).detach()),
        "gradient_norm_upright": float(np.linalg.norm(gradient)),
        "hessian_min": float(eigenvalues[0]), "hessian_max": float(eigenvalues[-1]),
        "hessian_indefinite": bool(eigenvalues[0] < -1e-8 and eigenvalues[-1] > 1e-8),
        "continuous_max_real_pole": float(poles.real.max()),
        "held_spectral_radius": float(np.abs(held_poles).max()),
    }
    arrays = {"hessian_canonical": hessian, "hessian_qv": hessian_qv,
              "hessian_eigenvalues": eigenvalues, "continuous_poles": poles,
              "held_poles": held_poles, "torque_jacobian_qv": torque_jacobian @ transform}
    for name, target in (("online", False), ("target", True)):
        z, values, grad = agent.value_gradient(states, target=target)
        eta = grad[:, 3]
        terms = {"value": values, "reward": -agent.reward.state_cost(z, agent.oracle),
                 "drift": (grad * agent.oracle.drift(z)).sum(-1),
                 "discount": -agent.config.discount_rate * values,
                 "action_score": agent.soft_action_score(eta), "eta": eta}
        terms["hjb"] = sum(terms[key] for key in ("reward", "drift", "discount", "action_score"))
        terms["label"] = values + agent.config.value_step * terms["hjb"]
        terms["torque"] = (eta / agent.reward.effort_weight).clamp(
            -agent.oracle.torque_limit, agent.oracle.torque_limit)
        for key, tensor in terms.items():
            array = tensor.detach().cpu().numpy()
            arrays[f"{name}_{key}"] = array
            metrics[f"{name}_{key}_rms"] = float(np.sqrt(np.mean(array.astype(float)**2)))
        metrics[f"{name}_saturation_fraction"] = float(np.mean(
            np.abs(arrays[f"{name}_torque"]) >= agent.oracle.torque_limit * (1 - 1e-6)))
        metrics[f"{name}_positive_value_fraction"] = float(np.mean(arrays[f"{name}_value"] > 1e-7))
    metrics["upright_float32_torque"] = float(arrays["online_torque"][0])
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("non-finite upright probe")
    return metrics, arrays


class OnlineReplay:
    """Seeded with exactly the fixed dataset's visited states, then FIFO."""

    def __init__(self, data, capacity):
        visited = data["rollout"]
        if not visited.any() or capacity < int(visited.sum()):
            raise ValueError("buffer_size must fit all initial visited states, and collection_steps must be positive")
        self.states = np.empty((capacity, 4), dtype=np.float32)
        self.terminal = np.zeros(capacity, dtype=bool)
        self.count = 0
        for state, terminal in zip(data["states"][visited], data["terminal"][visited]):
            self.append(state, terminal)

    def append(self, state, terminal):
        slot = self.count % len(self.states)
        self.states[slot], self.terminal[slot] = state, terminal
        self.count += 1

    def batch(self, env, size, rng):
        local_count = size // 2
        fresh = env.canonical(env.sample_qv(rng, local_count))
        indices = rng.integers(min(self.count, len(self.states)), size=size - local_count)
        return (np.concatenate((fresh, self.states[indices])),
                np.r_[np.zeros(local_count, dtype=bool), self.terminal[indices]])


def comparison_step(agents, data, indices, failure_value, online_batch=None, *, first_update=False):
    """A/B share exact row indices; the first C update shares them as well."""
    results = {}
    for arm, agent in agents.items():
        if arm == "fixed_labels" or first_update:
            # Reuse identical cached labels on the first step, even if label
            # collection used a different GEMM batch shape and rounding.
            results[arm] = agent.fit_labels(data["states"][indices], data["initial_labels"][indices],
                                           refresh_target=arm != "fixed_labels")
        else:
            states, terminal = ((data["states"][indices], data["terminal"][indices])
                                if arm != "online" or online_batch is None else online_batch)
            results[arm] = agent.update(states, terminal_mask=terminal, terminal_value=failure_value)
    return results


def run(args):
    for name in ("updates", "batch_size", "eval_every", "eval_episodes", "threads", "buffer_size",
                 "probe_batch_size", "train_states", "heldout_states", "probe_every", "collection_steps"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if min(args.train_states, args.heldout_states, args.batch_size) < 2:
        raise ValueError("dataset sizes and batch_size must be at least two")
    if args.buffer_size < args.train_states - args.train_states // 2:
        raise ValueError("buffer_size must fit the fixed dataset's visited states")
    if not np.isfinite(args.exploration_std) or args.exploration_std < 0:
        raise ValueError("exploration_std must be finite and nonnegative")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {args.output}")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    initial, config = build_agent(args)
    if ((not args.checkpoint and args.finite_horizon) or getattr(config, "finite_horizon", False)
            or getattr(initial.config, "horizon_seconds", None) is not None):
        raise ValueError("this comparison isolates the stationary HJB update; use --no-finite-horizon or a continuing checkpoint")
    if initial.config.auto_temperature:
        raise ValueError("use fixed temperature (--no-auto-temperature) to isolate target and data feedback")
    train_bank, train_source = _bank(args.incoming_states)
    heldout_bank, heldout_source = _bank(args.incoming_eval_states)
    if (train_bank is None) != (heldout_bank is None):
        raise ValueError("incoming comparison requires separate training and held-out banks")
    if train_bank is not None and {x.tobytes() for x in train_bank} & {x.tobytes() for x in heldout_bank}:
        raise ValueError("training and held-out incoming banks overlap")
    seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(
        [args.dataset_seed, args.seed]).spawn(3)]
    if len(set(seeds)) != len(seeds) or any(args.eval_seed <= s < args.eval_seed + args.eval_episodes for s in seeds):
        raise ValueError("collection/evaluation seeds overlap; change --dataset-seed")
    datasets, collection = {}, {}
    with ExitStack() as stack:
        def make_env(bank):
            env = AcrobotStageAEnv(config, initial.oracle, initial.reward, incoming_states=bank)
            stack.callback(env.close)
            return env

        train_env, heldout_env, online_env, eval_env = (
            make_env(train_bank), make_env(heldout_bank), make_env(train_bank), make_env(heldout_bank))
        failure_value = train_env.failure_value
        for split, env, count, seed in zip(("train", "heldout"), (train_env, heldout_env),
                                           (args.train_states, args.heldout_states), seeds):
            data, info = collect_states(initial, env, count=count, steps=args.collection_steps,
                                        seed=seed, exploration_std=args.exploration_std)
            data["initial_labels"] = labels_for(initial, data, failure_value, args.probe_batch_size)
            datasets[split], collection[split] = data, info
        if {x.tobytes() for x in datasets["train"]["states"]} & {x.tobytes() for x in datasets["heldout"]["states"]}:
            raise ValueError("training and held-out states overlap; use nondegenerate resets")
        args.output.mkdir(parents=True, exist_ok=True)
        qv, near_states = neighborhood(initial, config)
        dataset_path = args.output / "dataset.npz"
        np.savez_compressed(dataset_path, **{f"{split}_{k}": v for split, data in datasets.items()
                                            for k, v in data.items()}, upright_qv=qv,
                            upright_states=near_states, failure_value=failure_value,
                            frame="downward_vertical_canonical")
        metadata = {
            "diagnostic": "three_arm_stage_a_controller_comparison_v1", "arms": list(ARMS),
            "environment": asdict(config), "oracle": asdict(initial.oracle), "reward": asdict(initial.reward),
            "value_flow": asdict(initial.config), "initial_updates": initial.updates,
            "reward_scale": initial.metadata.get("reward_scale", 1.),
            "unscaled_reward": initial.metadata.get("unscaled_reward", asdict(initial.reward)),
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "hyperparams": initial.metadata.get("hyperparams") if args.checkpoint else args.hyperparams_source,
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
            "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "collection": collection, "online_seed": seeds[2],
            "training_incoming": train_source, "heldout_incoming": heldout_source,
            "minibatches": "A/B identical indices; C identical first batch, then half fresh/half replay",
            "online_initial_replay": "fixed training dataset visited states; one decision per later update",
            "evaluation_policy": "deterministic_mode", "torch_version": str(torch.__version__),
        }
        write_json(args.output / "config.json", metadata)
        initial.save(args.output / "initial.pt", metadata=metadata)
        agents = {arm: deepcopy(initial) for arm in ARMS}
        for arm in ARMS:
            (args.output / arm).mkdir()
        train = datasets["train"]
        replay = OnlineReplay(train, args.buffer_size)
        batch_rng = np.random.default_rng(np.random.SeedSequence([args.batch_seed, args.seed]))
        online_rng = np.random.default_rng(seeds[2])
        state, _ = online_env.reset(seed=seeds[2])
        seconds = 0.
        files = {name: stack.enter_context((args.output / f"{name}.jsonl").open("w"))
                 for name in ("training", "upright", "evaluations")}
        csv_file = stack.enter_context((args.output / "metrics.csv").open("w", newline=""))
        writer = None
        first_indefinite = {arm: None for arm in ARMS}

        def assess(update, probes, rollouts):
            nonlocal writer
            for arm, agent in agents.items():
                directory = args.output / arm
                if probes:
                    metrics, arrays = upright_probe(agent, near_states, config.dt)
                    if metrics["hessian_indefinite"] and first_indefinite[arm] is None:
                        first_indefinite[arm] = update
                    files["upright"].write(json.dumps({"arm": arm, "update": update, **metrics}, allow_nan=False) + "\n")
                    np.savez_compressed(directory / f"upright_{update:08d}.npz", **arrays)
                    for split, data in datasets.items():
                        labels = (data["initial_labels"] if arm == "fixed_labels" else
                                  labels_for(agent, data, failure_value, args.probe_batch_size))
                        regression, _ = probe(agent, data, labels, failure_value, args.probe_batch_size)
                        row = {"arm": arm, "update": update, "split": split, **regression}
                        if writer is None:
                            writer = csv.DictWriter(csv_file, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                    agent.save(directory / f"checkpoint_{update:08d}.pt",
                               metadata={**metadata, "arm": arm, "diagnostic_update": update})
                    print(json.dumps({"arm": arm, "update": update,
                                      **{k: metrics[k] for k in ("hessian_min", "hessian_max", "held_spectral_radius")}}), flush=True)
                if rollouts and not args.skip_rollouts:
                    groups = [evaluate(agent, eval_env, seed=args.eval_seed, episodes=args.eval_episodes)]
                    if heldout_bank is not None:
                        groups.append(evaluate(agent, eval_env, seed=args.eval_seed,
                                               episodes=args.eval_episodes, incoming=True))
                    files["evaluations"].write(json.dumps({"arm": arm, "update": update, "groups": groups}, allow_nan=False) + "\n")
            for file in (*files.values(), csv_file):
                file.flush()

        completed = 0
        try:
            assess(0, True, True)
            for update in range(1, args.updates + 1):
                indices = batch_indices(train, args.batch_size, batch_rng)
                online_batch = None
                if update > 1:
                    agent = agents["online"]
                    action = agent.act(state, deterministic=agent.config.temperature == 0, rng=online_rng)
                    if agent.config.temperature == 0:
                        action = np.clip(action + online_rng.normal(0., args.exploration_std, 1), -1., 1.)
                    next_state, _, terminated, truncated, info = online_env.step(action)
                    replay.append(state, False)
                    replay.append(next_state, info["state_limit_failure"])
                    seconds += info["dt_used"]
                    state = online_env.reset()[0] if terminated or truncated else next_state
                    online_batch = replay.batch(online_env, args.batch_size, online_rng)
                metrics = comparison_step(agents, train, indices, failure_value, online_batch,
                                          first_update=update == 1)
                completed = update
                for arm in ARMS:
                    files["training"].write(json.dumps({"arm": arm, "update": update,
                        "online_collection_seconds": seconds if arm == "online" else 0.,
                        **metrics[arm]}, allow_nan=False) + "\n")
                assess(update, update % args.probe_every == 0 or update == args.updates,
                       update % args.eval_every == 0 or update == args.updates)
        except (FloatingPointError, RuntimeError, np.linalg.LinAlgError) as exc:
            # Preserve the last comparable step and do not label unrelated
            # runtime failures (e.g. OOM) as numerical divergence.
            write_json(args.output / "summary.json", {
                "status": "failed", "error": str(exc), "last_completed_update": completed,
                "completed_updates": {a: v.updates - initial.updates for a, v in agents.items()},
                "first_indefinite_probe": first_indefinite})
            raise
        np.savez_compressed(args.output / "online_replay_final.npz",
                            states=replay.states[:min(replay.count, len(replay.states))],
                            terminal=replay.terminal[:min(replay.count, len(replay.states))],
                            total_insertions=replay.count)
        summary = {"status": "completed", "completed_updates": {a: v.updates - initial.updates for a, v in agents.items()},
                   "first_indefinite_probe": first_indefinite, "online_collection_seconds": seconds,
                   "note": "First-indefinite times are probe times, not exact onset; zero means indefinite at initialization."}
        write_json(args.output / "summary.json", summary)
        return summary


def main():
    args = parser().parse_args()
    if args.seeds is None:
        run(args)
        return
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must be distinct")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {args.output}")
    summaries = {}
    for seed in args.seeds:
        child = deepcopy(args)
        child.seed, child.output = seed, args.output / f"seed_{seed}"
        summaries[str(seed)] = run(child)
    write_json(args.output / "summary.json", summaries)


if __name__ == "__main__":
    main()
