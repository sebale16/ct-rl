"""Prepare, train and evaluate the Acrobot Experiment 3 CT-SAC baselines.

Run ``python -m benchmarks.acrobot_experiment3 --help``. No command launches a
full sweep implicitly; the manifest fixes the protocol shared by every arm.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")

import numpy as np

from environment.acrobot_generalization import (
    ConfigurationDemonstrator, PARAMETER_ORDER, RandomizedAcrobotEnv,
    parameter_context,
)
from environment.acrobot_xk import PlantScales
from evaluations.eval_acrobot_xk_ctsac import (
    DeterministicCTPolicy, load_checkpoint_model, load_training_config,
    parse_seed_spec, summarize_metrics,
)


DEFAULT_MODE = (
    "xk_r3_eta0p23_ctrl10ms_h2s_temp0p01_xkdot_q2dot4pi_"
    "logrecip_xkdemo20k_tau1p25e2"
)
DEFAULT_MANIFEST = Path("benchmarks/configs/acrobot_experiment3.json")
HYPERPARAMS = Path(__file__).parent / "hyperparams"


def make_manifest(seed=31003, *, train_count=32, validation_count=8,
                  test_count=16, base_mode=DEFAULT_MODE):
    """Finite, disjoint configurations with separate interpolation/OOD tests."""
    if min(train_count, validation_count, test_count) < 1:
        raise ValueError("configuration counts must be positive")
    rng = np.random.default_rng(seed)

    def draw(split, count, extrapolate=False):
        rows = []
        for index in range(count):
            factors = rng.uniform(0.8, 1.2, 4)
            if extrapolate:
                # Exactly one factor lies outside the training box. Balance
                # the changed coordinate and direction over each eight rows.
                axis, high = index % 4, (index // 4) % 2
                factors[axis] = rng.uniform(1.25, 1.4) if high else rng.uniform(0.6, 0.75)
            rows.append({"id": f"{split}_{index:03d}",
                         "scales": dict(zip(PARAMETER_ORDER, factors.tolist()))})
        return rows

    bundle = load_training_config(base_mode, HYPERPARAMS)
    return {
        "schema_version": 1, "parameter_seed": seed,
        "parameter_order": list(PARAMETER_ORDER),
        "base_mode": base_mode, "base_config_sha256": bundle.config_sha256,
        "base_config": json.loads(bundle.config_json),
        "train_range": [0.8, 1.2], "context_encoding": "scale_minus_one",
        "evaluation_seeds": list(range(20000, 20032)),
        "selection": "final checkpoint; test splits are never used during training",
        "splits": {
            "nominal": [{"id": "nominal", "scales": asdict(PlantScales())}],
            "train": draw("train", train_count),
            "validation": draw("validation", validation_count),
            "interpolation": draw("interpolation", test_count),
            "extrapolation": draw("extrapolation", test_count, True),
        },
    }


def read_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported manifest schema")
    if manifest["parameter_order"] != list(PARAMETER_ORDER):
        raise ValueError("unexpected context parameter order")
    if manifest["context_encoding"] != "scale_minus_one":
        raise ValueError("unsupported context encoding")
    ids, factors = set(), set()
    for split in ("nominal", "train", "validation", "interpolation", "extrapolation"):
        rows = manifest["splits"][split]
        if not rows:
            raise ValueError(f"empty {split} split")
        for row in rows:
            scales = PlantScales.coerce(row["scales"])
            if row["id"] in ids or scales in factors:
                raise ValueError("configuration IDs and parameter tuples must be disjoint")
            ids.add(row["id"])
            factors.add(scales)
    bundle = load_training_config(manifest["base_mode"], HYPERPARAMS)
    if bundle.config_sha256 != manifest["base_config_sha256"]:
        raise ValueError("base hyperparameters changed; regenerate/review the manifest")
    if manifest["base_config"] != json.loads(bundle.config_json):
        raise ValueError("embedded base configuration does not match its source row")
    return manifest, bundle


def env_settings(bundle):
    settings = dict(bundle.env_kwargs)
    if int(settings.pop("n_envs", 1)) != 1:
        raise ValueError("Experiment 3 currently uses one training environment")
    if settings.get("time_sampling") != "uniform":
        raise ValueError("use fixed control timing for matched Experiment 3 budgets")
    if str(settings.get("raw_state_obs", "")).lower() not in ("true", "1"):
        raise ValueError("base mode must use raw state observations")
    if float(settings.get("task_kwargs", {}).get("damping", 0)) != 0:
        raise ValueError("Experiment 3 requires zero damping")
    if bundle.algo_kwargs.get("use_model_based_q", False):
        raise ValueError("Experiment 3 baselines must use model-free CT-SAC")
    if float(bundle.algo_kwargs.get("imitation_coef", 0)) != 0:
        raise ValueError("use demonstration seeding without replay imitation")
    return settings


def write_json(path, value):
    """Use strict JSON: undefined metrics are null, with success flags retained."""
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val) for val in item]
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            return None
        return item
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def manifest_digest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def train(args):
    from algorithms.ct_sac import CTSAC
    from common.callbacks import BaseCallback
    from common.logger import configure
    from common.utils import set_seed
    from environment.monitor import Monitor
    from models import ActorQCriticModel

    manifest, bundle = read_manifest(args.manifest)
    settings = env_settings(bundle)
    configurations = manifest["splits"]["train"]
    if args.arm in ("nominal", "specialist"):
        candidates = [row for rows in manifest["splits"].values() for row in rows]
        wanted = "nominal" if args.arm == "nominal" else args.configuration
        configurations = [row for row in candidates if row["id"] == wanted]
        if not configurations:
            raise ValueError("specialist requires a valid --configuration ID")
    args.output.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    configure(str(args.output / "logs"), output_formats=["csv", "json", "log"])
    env = RandomizedAcrobotEnv(
        configurations, conditioned=args.arm == "conditioned", seed=args.seed,
        episode_log=args.output / "episodes.jsonl", **settings)
    algo_kwargs = dict(bundle.algo_kwargs)
    controller = algo_kwargs.pop("demonstration_controller", None)
    if controller not in (None, "xin_kaneda"):
        raise ValueError("only Xin-Kaneda swing-up demonstrations are supported")
    if controller:
        algo_kwargs["demonstration_policy"] = ConfigurationDemonstrator(env)
    steps = args.steps if args.steps is not None else bundle.total_timesteps
    model_kwargs = dict(bundle.model_kwargs)
    if args.smoke:
        # Exercise resets, the demonstration/learning handoff, critic updates,
        # serialization and evaluation. Smoke artifacts cannot be full results.
        steps = 32
        model_kwargs.update(q_net_arch=[16, 16], pi_net_arch=[16, 16])
        algo_kwargs.update(buffer_size=64, batch_size=8, learning_starts=8,
                           demonstration_steps=8)
        env.episode_duration = 0.04
        env.max_steps = 4
    metadata = {
        "schema_version": 1, "arm": args.arm, "seed": args.seed,
        "configuration": args.configuration, "smoke": args.smoke,
        "manifest": manifest, "manifest_sha256": manifest_digest(manifest),
        "requested_steps": steps, "control_dt": env.dt,
        "episode_duration": env.episode_duration,
        "physics_dt": env.physics_dt,
        "effective_algo_kwargs": {key: value for key, value in algo_kwargs.items()
                                  if key != "demonstration_policy"},
        "model_arch_override": {"q_net_arch": [16, 16], "pi_net_arch": [16, 16]}
        if args.smoke else {},
        "demonstration_gain_rule": "max(base gain, 1.01 * plant floor) for k_D/k_P",
    }
    write_json(args.output / "run.json", metadata)

    class Progress(BaseCallback):
        def __init__(self):
            super().__init__()
            self.demonstration_seconds = 0.0
            self.previous_seconds = 0.0

        def _on_step(self):
            agent = self.algorithm
            seconds = agent.num_simulated_seconds
            if controller and agent.num_timesteps <= agent.demonstration_steps:
                self.demonstration_seconds += seconds - self.previous_seconds
            self.previous_seconds = seconds
            if agent.num_timesteps % int(bundle.log_kwargs["save_freq"]) == 0:
                agent.save(args.output / f"step_{agent.num_timesteps}.pth")
            return True

    progress = Progress()
    try:
        agent = CTSAC(env=Monitor(env), model=ActorQCriticModel,
                      model_kwargs=model_kwargs, seed=args.seed, device=args.device,
                      **algo_kwargs)
        agent.learn(total_timesteps=steps, callback=progress,
                    log_interval=int(bundle.log_kwargs["interval"]))
        agent.save(args.output / "final_model.pth")
        metadata.update(completed_steps=agent.num_timesteps,
                        gradient_updates=agent._n_updates,
                        simulated_seconds=agent.num_simulated_seconds,
                        demonstration_seconds=progress.demonstration_seconds)
        write_json(args.output / "run.json", metadata)
    finally:
        env.close()


def evaluate(args):
    from controllers.xin_kaneda import AcrobotParams
    from environment.dmc import DMCContinuousEnv
    from evaluations.acrobot_homoclinic_metrics import evaluate_episode, rollout

    if args.output.exists():
        raise FileExistsError(args.output)
    manifest, bundle = read_manifest(args.manifest)
    settings = env_settings(bundle)
    metadata = None
    if args.run:
        metadata = json.loads((args.run / "run.json").read_text())
        if metadata["manifest_sha256"] != manifest_digest(manifest):
            raise ValueError("training and evaluation manifests differ")
        conditioned = metadata["arm"] == "conditioned"
        checkpoint = args.run / "final_model.pth"
    else:
        conditioned = False
        checkpoint = args.checkpoint
    if checkpoint is None or not checkpoint.is_file():
        raise ValueError("provide --run or an existing frozen --checkpoint")
    model_kwargs = dict(bundle.model_kwargs)
    if metadata:
        model_kwargs.update(metadata["model_arch_override"])
    seeds = parse_seed_spec(args.seeds) if args.seeds else manifest["evaluation_seeds"]
    if args.horizon is not None:
        settings["episode_duration"] = args.horizon
        settings["max_steps"] = int(np.ceil(args.horizon / settings["dt"]))
    rows, by_split = [], {}
    model = None
    for split in args.splits:
        metrics = []
        per_configuration = {}
        for config in manifest["splits"][split]:
            kwargs = dict(settings)
            kwargs["task_kwargs"] = {**settings["task_kwargs"],
                                     "plant_scales": config["scales"]}
            env = DMCContinuousEnv("acrobot", "swingup-xk", **kwargs)
            try:
                if model is None:
                    if conditioned:
                        from gymnasium import spaces
                        env.observation_space = spaces.Box(-np.inf, np.inf, (8,), np.float32)
                    model = load_checkpoint_model(env, model_kwargs, checkpoint, args.device)
                policy = DeterministicCTPolicy(model)
                context = parameter_context(config["scales"])

                def act(obs):
                    return policy(np.concatenate((obs, context)) if conditioned else obs)

                params = AcrobotParams.from_physics(env._env.physics)
                config_metrics = []
                for seed in seeds:
                    trajectory = rollout(env, act, int(seed))
                    result = evaluate_episode(trajectory, params)
                    config_metrics.append(result)
                    metrics.append(result)
                    rows.append({"split": split, "configuration": config,
                                 "termination_reason": env._env.task.last_termination_reason,
                                 "plant_parameters": asdict(params),
                                 "seed": int(seed), **asdict(result)})
                per_configuration[config["id"]] = summarize_metrics(config_metrics)
            finally:
                env.close()
        by_split[split] = {
            "pooled": summarize_metrics(metrics),
            "per_configuration": per_configuration,
            "worst_configuration_capture_rate": min(
                np.mean([row["captured"] for row in rows
                         if row["configuration"]["id"] == config_id])
                for config_id in per_configuration),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, {
        "schema_version": 1, "arm": metadata["arm"] if metadata else "frozen",
        "training_seed": metadata["seed"] if metadata else args.training_seed,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "manifest": manifest, "manifest_sha256": manifest_digest(manifest),
        "training": metadata, "evaluation_settings": settings,
        "evaluation_seeds": list(seeds), "splits": by_split, "episodes": rows,
    })
    print(f"Wrote {len(rows)} episodes to {args.output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="write a reproducible parameter manifest")
    prepare.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    prepare.add_argument("--parameter-seed", type=int, default=31003)
    prepare.add_argument("--base-mode", default=DEFAULT_MODE)
    training = commands.add_parser("train", help="train one seed of a baseline")
    training.add_argument("--arm", choices=("randomized", "conditioned", "nominal", "specialist"), required=True)
    training.add_argument("--configuration", help="configuration ID for a specialist control")
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--steps", type=int)
    training.add_argument("--smoke", action="store_true")
    evaluation = commands.add_parser("evaluate", help="evaluate a frozen checkpoint on fixed splits")
    source = evaluation.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path, help="directory produced by train")
    source.add_argument("--checkpoint", type=Path, help="existing nominal state-only checkpoint")
    evaluation.add_argument("--training-seed", type=int, help="metadata for an existing checkpoint")
    evaluation.add_argument("--splits", nargs="+", default=["nominal", "interpolation", "extrapolation"],
                            choices=("nominal", "train", "validation", "interpolation", "extrapolation"))
    evaluation.add_argument("--seeds", help="override initial-state seeds, e.g. 20000:20002")
    evaluation.add_argument("--horizon", type=float, help="override physical evaluation horizon")
    for command in (training, evaluation):
        command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, make_manifest(args.parameter_seed, base_mode=args.base_mode))
    elif args.command == "train":
        if args.steps is not None and args.steps <= 0:
            parser.error("--steps must be positive")
        train(args)
    else:
        if args.horizon is not None and (not np.isfinite(args.horizon) or args.horizon <= 0):
            parser.error("--horizon must be finite and positive")
        evaluate(args)


if __name__ == "__main__":
    main()
