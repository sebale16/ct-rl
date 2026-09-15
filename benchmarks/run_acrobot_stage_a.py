"""Train/evaluate oracle-only PH value control for Acrobot capture and balance.

Example: python -m benchmarks.run_acrobot_stage_a --output out/stage_a --updates 10000
No learned PH model, actor network, LQR warm start, or controller switch is used.
"""
from __future__ import annotations

from dataclasses import asdict
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")

import numpy as np
import torch

from algorithms.acrobot_ph_value import AcrobotPHValue, ValueFlowConfig
from benchmarks.acrobot_stage_a_config import DEFAULT_HYPERPARAMS_DIR, DEFAULT_MODE, StageAArgumentParser
from environment.acrobot_stage_a import AcrobotStageAEnv, StageAConfig, load_incoming_states
from models.acrobot_oracle import AcrobotOracle, UprightReward


def evaluate(agent, env, *, seed=20000, episodes=16, incoming=False):
    """Fixed independent resets, deterministic mode, and physical-time metrics."""
    rows = []
    for index in range(episodes):
        z, info = env.reset(seed=seed + index, options={"incoming": incoming})
        discounted_return = effort = saturated_time = 0.
        elapsed = 0.
        while True:
            action = agent.act(z, deterministic=True)
            z, reward, terminated, truncated, info = env.step(action)
            duration = info["dt_used"]
            discounted_return += math.exp(-env.config.discount_rate * elapsed) * reward
            effort += info["torque"]**2 * duration
            saturated_time += float(abs(action[0]) >= 1 - 1e-6) * duration
            elapsed += duration
            if terminated or truncated:
                break
        rows.append({"seed": seed + index, "return": discounted_return,
                     "success": info["success"], "retained_success": info["retained_success"],
                     "max_hold_seconds": info["max_hold_seconds"],
                     "terminal_hold_seconds": info["terminal_hold_seconds"],
                     "first_capture_seconds": info["first_capture_seconds"],
                     "occupancy_fraction": info["occupancy_seconds"] / elapsed,
                     "torque_squared_integral": effort, "saturation_fraction": saturated_time / elapsed,
                     "state_limit_failure": info["state_limit_failure"], "physical_seconds": elapsed})
    averages = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]
                if key not in ("seed", "first_capture_seconds")}
    return {"group": "incoming" if incoming else "near_upright", "episodes": rows, "mean": averages}


def _bank(path):
    if path is None:
        return None, None
    states = load_incoming_states(path)
    return states, {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                    "count": len(states)}


def reward_scale_argument(value):
    if value == "auto":
        return value
    try:
        scale = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("reward scale must be 'auto' or a positive number") from exc
    if not math.isfinite(scale) or scale <= 0:
        raise argparse.ArgumentTypeError("reward scale must be finite and positive")
    return scale


def parser():
    p = StageAArgumentParser(description=__doc__)
    p.add_argument("--mode", default=DEFAULT_MODE, help="row name in acrobot_ph_value.csv")
    p.add_argument("--hyperparams-dir", "--hyperparams_dir", type=Path, default=DEFAULT_HYPERPARAMS_DIR)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, help="evaluate an existing Stage A checkpoint; no training")
    p.add_argument("--updates", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=100000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-episodes", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-seed", type=int, default=20000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--physics-dt", type=float, default=0.001)
    p.add_argument("--episode-seconds", type=float, default=5.)
    p.add_argument("--hold-seconds", type=float, default=1.)
    p.add_argument("--angle-radius", type=float, default=0.05)
    p.add_argument("--velocity-radius", type=float, default=0.1)
    p.add_argument("--capture-angle", type=float, default=0.1)
    p.add_argument("--capture-velocity", type=float, default=0.25)
    p.add_argument("--velocity-limit", type=float, default=2 * math.pi)
    p.add_argument("--elbow-limit", type=float, default=math.pi)
    p.add_argument("--shoulder-limit", type=float, default=math.pi / 2,
                   help="failure at this unwrapped shoulder deviation from upright pi (radians)")
    p.add_argument("--incoming-probability", type=float, default=0.5)
    p.add_argument("--torque-limit", type=float, default=20.)
    p.add_argument("--damping", type=float, default=0.)
    p.add_argument("--discount-rate", type=float, default=0.1)
    p.add_argument("--value-step", type=float, default=0.02)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--target-rate", type=float, default=0.01)
    p.add_argument("--hidden-width", type=int, default=64)
    p.add_argument("--momentum-scale", type=float, default=10.)
    p.add_argument("--grad-clip", type=float, default=10.)
    p.add_argument("--temperature", type=float, default=0., help="0: clipped analytic policy; >0: truncated-Gaussian policy")
    p.add_argument("--auto-temperature", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--temperature-learning-rate", type=float, default=3e-4)
    p.add_argument("--target-entropy", type=float, default=-1., help="differential entropy in normalized action a in [-1,1]")
    p.add_argument("--temperature-min", type=float, default=1e-4)
    p.add_argument("--temperature-max", type=float, default=10.)
    p.add_argument("--quadrature-points", type=int, default=128)
    p.add_argument("--exploration-std", type=float, default=0.02,
                   help="normalized action noise for deterministic training only; evaluation has none")
    p.add_argument("--angle1-weight", type=float, default=10.)
    p.add_argument("--angle2-weight", type=float, default=5.)
    p.add_argument("--velocity1-weight", type=float, default=1.)
    p.add_argument("--velocity2-weight", type=float, default=1.)
    p.add_argument("--velocity-scale", type=float, default=4.5844)
    p.add_argument("--effort-weight", type=float, default=0.01)
    p.add_argument("--reward-scale", type=reward_scale_argument, default="auto",
                   help="common multiplier for reward and temperature; auto normalizes the cost bound to one")
    p.add_argument("--state-cost-transform", type=str, choices=("identity", "log"), default="identity")
    p.add_argument("--log-reference-angle-deg", type=float, default=5.,
                   help="log epsilon is the base shoulder cost at this deviation, with zero velocity")
    p.add_argument("--incoming-states", type=Path, help="training NPZ: states[N,4] q/v and explicit frame")
    p.add_argument("--incoming-eval-states", type=Path, help="held-out NPZ from separate swing-up trajectories")
    return p


def build_agent(args):
    """Shared physical/value configuration for training and target diagnostics."""
    if args.checkpoint:
        agent = AcrobotPHValue.load(args.checkpoint, device=args.device)
        environment = dict(agent.metadata["environment"])
        environment.setdefault("shoulder_limit", None)  # Preserve historical evaluation domains.
        env_config = StageAConfig(**environment)
    else:
        env_config = StageAConfig(dt=args.dt, physics_dt=args.physics_dt,
                                 episode_seconds=args.episode_seconds, hold_seconds=args.hold_seconds,
                                 discount_rate=args.discount_rate, angle_radius=args.angle_radius,
                                 velocity_radius=args.velocity_radius, capture_angle=args.capture_angle,
                                 capture_velocity=args.capture_velocity, velocity_limit=args.velocity_limit,
                                 elbow_limit=args.elbow_limit, shoulder_limit=args.shoulder_limit,
                                 incoming_probability=args.incoming_probability)
        oracle = AcrobotOracle(damping=args.damping, torque_limit=args.torque_limit)
        raw_reward = UprightReward(**{key: getattr(args, key) for key in (
            "angle1_weight", "angle2_weight", "velocity1_weight", "velocity2_weight", "velocity_scale", "effort_weight")})
        scale = (1 / raw_reward.cost_bound(oracle, velocity_limit=env_config.velocity_limit,
                                          elbow_limit=env_config.elbow_limit, shoulder_limit=env_config.shoulder_limit)
                 if args.reward_scale == "auto" else float(args.reward_scale))
        reward = raw_reward.scaled(scale)
        if args.state_cost_transform == "log":
            bound = reward.base_state_cost_bound(velocity_limit=env_config.velocity_limit,
                                                 elbow_limit=env_config.elbow_limit,
                                                 shoulder_limit=env_config.shoulder_limit)
            reward = reward.with_log_state_cost(bound, args.log_reference_angle_deg)
        elif args.state_cost_transform != "identity":
            raise ValueError("state_cost_transform must be identity or log")
        flow_parameters = {key: getattr(args, key) for key in ValueFlowConfig.__dataclass_fields__}
        for key in ("temperature", "temperature_min", "temperature_max"):
            flow_parameters[key] *= scale
        flow = ValueFlowConfig(**flow_parameters)
        agent = AcrobotPHValue(oracle, reward, flow, device=args.device)
        # Keep the initial value-gradient policy identical across reward units.
        # This is initialization only, never a rescaling of loaded Adam state.
        with torch.no_grad():
            for network in (agent.value, agent.target):
                network.net[-1].weight.mul_(scale)
                network.net[-1].bias.mul_(scale)
        agent.metadata = {"reward_scale": scale, "unscaled_reward": asdict(raw_reward)}
    return agent, env_config


def run(args):
    for name in ("updates", "batch_size", "buffer_size", "eval_every", "eval_episodes", "threads"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.seed == args.eval_seed:
        raise ValueError("training and evaluation seeds must differ")
    if not np.isfinite(args.exploration_std) or args.exploration_std < 0:
        raise ValueError("exploration_std must be finite and nonnegative")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {args.output}")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train_bank, train_source = _bank(args.incoming_states)
    eval_bank, eval_source = _bank(args.incoming_eval_states)
    if train_bank is not None:
        if eval_bank is None:
            raise ValueError("incoming training requires --incoming-eval-states from held-out trajectories")
        if {row.tobytes() for row in train_bank} & {row.tobytes() for row in eval_bank}:
            raise ValueError("training and evaluation incoming banks overlap")
    agent, env_config = build_agent(args)
    train_env = AcrobotStageAEnv(env_config, agent.oracle, agent.reward, incoming_states=train_bank)
    eval_env = AcrobotStageAEnv(env_config, agent.oracle, agent.reward, incoming_states=eval_bank)
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        metadata = {"environment": asdict(env_config), "oracle": asdict(agent.oracle),
                    "reward": asdict(agent.reward), "value_flow": asdict(agent.config),
                    "reward_scale": agent.metadata.get("reward_scale", 1.),
                    "unscaled_reward": agent.metadata.get("unscaled_reward", asdict(agent.reward)),
                    "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "training_incoming": train_source, "evaluation_incoming": eval_source,
                    "hyperparams": agent.metadata.get("hyperparams") if args.checkpoint else args.hyperparams_source,
                    "learned_dynamics": False, "evaluation_policy": "deterministic_mode",
                    "torch_version": str(torch.__version__)}
        (args.output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")

        def assessment(update, simulated_seconds):
            groups = [evaluate(agent, eval_env, seed=args.eval_seed, episodes=args.eval_episodes)]
            if eval_bank is not None:
                groups.append(evaluate(agent, eval_env, seed=args.eval_seed, episodes=args.eval_episodes, incoming=True))
            result = {"update": update, "training_physical_seconds": simulated_seconds, "groups": groups}
            with (args.output / "evaluations.jsonl").open("a") as f:
                f.write(json.dumps(result, allow_nan=False) + "\n")
            print(json.dumps({"update": update, "training_physical_seconds": simulated_seconds,
                              "evaluation": {g["group"]: g["mean"] for g in groups}}), flush=True)
            # Prefer retention, then time held, then return; do not select only
            # by a shaped reward that can conceal failure to balance.
            return tuple(float(np.mean([g["mean"][key] for g in groups]))
                         for key in ("retained_success", "terminal_hold_seconds", "return"))

        best = assessment(agent.updates, 0.)
        if args.checkpoint:
            return
        agent.save(args.output / "best.pt", metadata=metadata)
        replay = np.empty((args.buffer_size, 4), dtype=np.float32)
        terminals = np.zeros(args.buffer_size, dtype=bool)
        count = 0
        seconds = 0.
        z, _ = train_env.reset(seed=args.seed)
        with (args.output / "training.jsonl").open("w") as log:
            for update in range(1, args.updates + 1):
                action = agent.act(z, deterministic=agent.config.temperature == 0, rng=rng)
                if agent.config.temperature == 0:
                    action = np.clip(action + rng.normal(0., args.exploration_std, size=1), -1., 1.)
                next_z, _, terminated, truncated, info = train_env.step(action)
                seconds += info["dt_used"]
                for sample, terminal in ((z, False), (next_z, terminated)):
                    slot = count % args.buffer_size
                    replay[slot], terminals[slot] = sample, terminal
                    count += 1
                z = next_z
                if terminated or truncated:
                    z, _ = train_env.reset()
                # Half fresh local/incoming collocation, half visited states.
                # Oracle access is explicit; no dynamics are fitted to replay.
                local_count = args.batch_size // 2
                fresh = train_env.canonical(train_env.sample_qv(rng, local_count))
                indices = rng.integers(min(count, args.buffer_size), size=args.batch_size - local_count)
                batch = np.concatenate((fresh, replay[indices]))
                mask = np.r_[np.zeros(local_count, dtype=bool), terminals[indices]]
                metrics = agent.update(batch, terminal_mask=mask, terminal_value=train_env.failure_value)
                if update == 1 or update % 100 == 0 or update == args.updates:
                    log.write(json.dumps({"update": update, "training_physical_seconds": seconds, **metrics}, allow_nan=False) + "\n")
                    log.flush()
                if update % args.eval_every == 0 or update == args.updates:
                    score = assessment(update, seconds)
                    if score > best:
                        best = score
                        agent.save(args.output / "best.pt", metadata=metadata)
                    agent.save(args.output / "last.pt", metadata=metadata)
    finally:
        train_env.close()
        eval_env.close()


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
