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
    from common.callbacks import BaseCallback, CallbackList, WallClockCheckpointCallback
    from common.checkpoint import _is_complete, load_checkpoint
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
    checkpoint_dir = args.output / "checkpoint"
    # The same wall-clock checkpoint the acrobot-xk sweeps use: a complete
    # checkpoint directory means this cell was paused near a queue wall and the
    # chain should continue it rather than start a new run from zero.
    resuming = bool(args.resume) and _is_complete(str(checkpoint_dir))
    previous = {}
    if resuming:
        previous = json.loads((args.output / "run.json").read_text())
    elif args.resume and args.output.exists():
        raise FileExistsError(
            f"{args.output} exists without a complete checkpoint; move it aside")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    configure(str(args.output / "logs"), output_formats=["csv", "json", "log"],
              append=resuming)
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
        "chunks": (previous.get("chunks", 1) + 1) if resuming else 1,
    }
    if resuming:
        # A chunk must continue the same run: refuse a checkpoint written under
        # a different arm, seed, manifest or budget rather than blending them.
        mismatch = [key for key in ("arm", "seed", "configuration", "smoke",
                                    "manifest_sha256", "requested_steps",
                                    "effective_algo_kwargs")
                    if previous.get(key) != metadata[key]]
        if mismatch:
            raise ValueError(f"cannot resume: {mismatch} differ from the paused run")
    write_json(args.output / "run.json", metadata)

    class Progress(BaseCallback):
        def __init__(self, demonstration_seconds=0.0):
            super().__init__()
            self.demonstration_seconds = float(demonstration_seconds)
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

    def sweep_state():
        """Experiment 3 state the algorithm checkpoint does not already carry.

        The configuration and reset streams live on the environment, so without
        this a resumed chunk would replay the same plant sequence from the top
        and bias the training distribution toward its first configurations.
        """
        return {
            "configuration_rng": env._configuration_rng.bit_generator.state,
            "reset_rng": env._reset_rng.bit_generator.state,
            "episode_index": env.episode_index,
            "demonstration_seconds": progress.demonstration_seconds,
        }

    progress = Progress()
    try:
        agent = CTSAC(env=Monitor(env), model=ActorQCriticModel,
                      model_kwargs=model_kwargs, seed=args.seed, device=args.device,
                      **algo_kwargs)
        if resuming:
            restored = load_checkpoint(agent, str(checkpoint_dir))
            env._configuration_rng.bit_generator.state = restored["configuration_rng"]
            env._reset_rng.bit_generator.state = restored["reset_rng"]
            env.episode_index = int(restored["episode_index"])
            progress.demonstration_seconds = float(restored["demonstration_seconds"])
            progress.previous_seconds = agent.num_simulated_seconds
            print(f"[resume] {checkpoint_dir}: {agent.num_timesteps}/{steps} steps, "
                  f"episode {env.episode_index}", flush=True)
        wall = None
        callbacks = [progress]
        if args.max_seconds:
            wall = WallClockCheckpointCallback(
                ckpt_dir=str(checkpoint_dir), max_seconds=args.max_seconds,
                extra_state_fn=sweep_state, verbose=1)
            callbacks.append(wall)
        agent.learn(total_timesteps=steps, callback=CallbackList(callbacks),
                    log_interval=int(bundle.log_kwargs["interval"]))
        paused = bool(wall is not None and wall.stopped)
        finished = agent.num_timesteps >= steps and not paused
        metadata.update(completed_steps=agent.num_timesteps,
                        gradient_updates=agent._n_updates,
                        simulated_seconds=agent.num_simulated_seconds,
                        demonstration_seconds=progress.demonstration_seconds,
                        finished=finished)
        write_json(args.output / "run.json", metadata)
        if args.max_seconds:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "STATUS").write_text(
                f"{'DONE' if finished else 'INCOMPLETE'} "
                f"num_timesteps={agent.num_timesteps} total_timesteps={steps}\n")
        if finished:
            # Written last: the chain treats final_model.pth as the cell's
            # completion marker, so it must never exist for a paused chunk.
            agent.save(args.output / "final_model.pth")
            print(f"Training finished: {agent.num_timesteps}/{steps} steps.", flush=True)
        else:
            print(f"Training paused at {agent.num_timesteps}/{steps} steps "
                  "(resume from checkpoint).", flush=True)
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


def merge(args):
    """Recombine seed-sharded evaluations of one checkpoint into one result.

    A full evaluation of one checkpoint is far longer than a development-queue
    wall, so it is run as disjoint seed ranges and merged here. Summaries are
    recomputed from the pooled episodes rather than averaged across shards.
    """
    from evaluations.acrobot_homoclinic_metrics import EpisodeMetrics

    if args.output.exists():
        raise FileExistsError(args.output)
    shards = [json.loads(Path(path).read_text()) for path in sorted(args.inputs)]
    if not shards:
        raise ValueError("no shard files given")
    fixed = ("arm", "training_seed", "checkpoint", "checkpoint_sha256",
             "manifest_sha256", "evaluation_settings")
    for shard in shards[1:]:
        differing = [key for key in fixed if shard[key] != shards[0][key]]
        if differing:
            raise ValueError(f"shards disagree on {differing}; not one evaluation")
    manifest = shards[0]["manifest"]
    fields = list(EpisodeMetrics.__dataclass_fields__)

    episodes, seen = [], set()
    for shard in shards:
        for row in shard["episodes"]:
            key = (row["split"], row["configuration"]["id"], row["seed"])
            if key in seen:
                raise ValueError(f"duplicate episode {key} across shards")
            seen.add(key)
            episodes.append(row)

    by_split = {}
    for split in sorted({row["split"] for row in episodes}):
        rows = [row for row in episodes if row["split"] == split]
        expected = {(config["id"], int(seed))
                    for config in manifest["splits"][split]
                    for seed in manifest["evaluation_seeds"]}
        covered = {(row["configuration"]["id"], int(row["seed"])) for row in rows}
        if covered != expected:
            raise ValueError(
                f"{split}: shards cover {len(covered)} of {len(expected)} episodes; "
                "merge only complete splits")
        def metrics_of(subset):
            # JSON writes undefined metrics as null; restore them as NaN so the
            # summaries treat them exactly as an unsharded evaluation would.
            return [EpisodeMetrics(**{name: (float("nan") if row[name] is None
                                             else row[name]) for name in fields})
                    for row in subset]
        per_configuration = {}
        for config in manifest["splits"][split]:
            subset = [row for row in rows if row["configuration"]["id"] == config["id"]]
            per_configuration[config["id"]] = summarize_metrics(metrics_of(subset))
        by_split[split] = {
            "pooled": summarize_metrics(metrics_of(rows)),
            "per_configuration": per_configuration,
            "worst_configuration_capture_rate": min(
                float(np.mean([row["captured"] for row in rows
                               if row["configuration"]["id"] == config_id]))
                for config_id in per_configuration),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {
        **{key: shards[0][key] for key in
           ("schema_version", "arm", "training_seed", "checkpoint",
            "checkpoint_sha256", "manifest", "manifest_sha256", "training",
            "evaluation_settings")},
        "evaluation_seeds": list(manifest["evaluation_seeds"]),
        "merged_from": [str(path) for path in sorted(args.inputs)],
        "splits": by_split, "episodes": episodes,
    })
    print(f"Merged {len(shards)} shards, {len(episodes)} episodes to {args.output}")


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
    training.add_argument("--resume", action="store_true",
                          help="continue a cell paused by --max-seconds")
    training.add_argument("--max-seconds", type=float,
                          help="wall-clock budget; write a resumable checkpoint "
                               "and stop cleanly near it (or on SIGTERM)")
    evaluation = commands.add_parser("evaluate", help="evaluate a frozen checkpoint on fixed splits")
    source = evaluation.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path, help="directory produced by train")
    source.add_argument("--checkpoint", type=Path, help="existing nominal state-only checkpoint")
    evaluation.add_argument("--training-seed", type=int, help="metadata for an existing checkpoint")
    evaluation.add_argument("--splits", nargs="+", default=["nominal", "interpolation", "extrapolation"],
                            choices=("nominal", "train", "validation", "interpolation", "extrapolation"))
    evaluation.add_argument("--seeds", help="override initial-state seeds, e.g. 20000:20002")
    evaluation.add_argument("--horizon", type=float, help="override physical evaluation horizon")
    merging = commands.add_parser(
        "merge", help="combine seed-sharded evaluations of one checkpoint")
    merging.add_argument("--inputs", type=Path, nargs="+", required=True)
    merging.add_argument("--output", type=Path, required=True)
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
    elif args.command == "merge":
        merge(args)
    elif args.command == "train":
        if args.steps is not None and args.steps <= 0:
            parser.error("--steps must be positive")
        if args.max_seconds is not None and args.max_seconds <= 0:
            parser.error("--max-seconds must be positive")
        train(args)
    else:
        if args.horizon is not None and (not np.isfinite(args.horizon) or args.horizon <= 0):
            parser.error("--horizon must be finite and positive")
        evaluate(args)


if __name__ == "__main__":
    main()
