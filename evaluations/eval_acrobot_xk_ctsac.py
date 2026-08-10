"""Reward-independent fixed-protocol evaluation of a CT-SAC Acrobot-XK policy.

The evaluator loads one ``ActorQCriticModel`` checkpoint using the model
architecture recorded in ``benchmarks/hyperparams/ct_sac.csv`` and acts with
the policy mean (``deterministic=True``).  It then scores the resulting state
and applied-torque trajectories with metrics 1--6 from
``docs/reward_shaping_for_acrobot_swingup.md``.  Episode reward and return are
deliberately neither accumulated nor written: ``r0``, ``r1`` and ``r2`` have
different numerical scales and are training-arm metadata, not evaluation
criteria.

By default every checkpoint is evaluated on the document's exact common
protocol: release-from-rest starts, seeds 20000--20031, a 20 second horizon,
0.5 ms control and physics periods, zero damping, and a 64 N m actuator gear.
The protocol arguments remain configurable for short smoke runs, but every
chosen value is recorded in both outputs.

Metric 7 can be added with ``--evaluations-npz``.  That artifact is read by
``evaluations.acrobot_training_metrics`` and reports the first observed
cumulative simulated physical time at 50%, 80%, and 90% strict-capture
success.  Decision counts are never treated as seconds.  A legacy uniform-step
artifact can be converted only with an explicit ``--legacy-seconds-per-step``.

Example
-------
::

    MUJOCO_GL=disable python -m evaluations.eval_acrobot_xk_ctsac \\
      --checkpoint /runs/xk_r2/seed_0/final_model.pth \\
      --mode xk_r2_eta0p1_fixed0p5ms --reward-kind r2 --eta 0.1 \\
      --evaluations-npz /runs/xk_r2/seed_0/eval/evaluations.npz \\
      --output results/acrobot_xk_r2_seed0.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

os.environ.setdefault("MUJOCO_GL", "disable")

import numpy as np

from controllers.xin_kaneda import AcrobotParams
from evaluations.acrobot_homoclinic_metrics import (
    LQR_SWITCH_THRESHOLD,
    EpisodeMetrics,
    TubeSpec,
    aggregate,
    evaluate_episode,
    rollout,
)


SCHEMA_VERSION = 1
ENV_ID = "acrobot-swingup-xk"
ALGORITHM = "ct_sac"
REWARD_KINDS = ("r0", "r1", "r2")

DEFAULT_SEEDS = tuple(range(20000, 20032))
DEFAULT_T_MAX = 20.0
DEFAULT_DT = 0.0005
DEFAULT_PHYSICS_DT = 0.0005
DEFAULT_DAMPING = 0.0
DEFAULT_TORQUE_LIMIT = 64.0
DEFAULT_RELEASE_ANGLE_RANGE = (0.05, 0.5)
DEFAULT_METRIC7_TARGETS = (0.5, 0.8, 0.9)

EPISODE_METRIC_FIELDS = tuple(EpisodeMetrics.__dataclass_fields__)
OUTPUT_FIELDS = (
    "schema_version",
    "algorithm",
    "env_id",
    "mode",
    "checkpoint_path",
    "checkpoint_sha256",
    "train_seed",
    "config_sha256",
    "reward_kind",
    "eta",
    "task_metadata_json",
    "seed",
    "start",
    "damping",
    "torque_limit",
    "t_max",
    "dt",
    "physics_dt",
    "tube_energy",
    "tube_angle",
    "tube_rate",
    "dwell_seconds",
    "lqr_threshold",
    *EPISODE_METRIC_FIELDS,
)


def parse_seed_spec(value: str) -> tuple[int, ...]:
    """Parse ``start:stop[:step]`` or a comma-separated seed list."""

    text = str(value).strip()
    if not text:
        raise ValueError("seed specification cannot be empty")
    if ":" in text:
        if "," in text:
            raise ValueError("use either a seed range or a comma-separated list")
        parts = text.split(":")
        if len(parts) not in (2, 3) or any(not part.strip() for part in parts):
            raise ValueError("seed range must be start:stop or start:stop:step")
        start, stop = int(parts[0]), int(parts[1])
        step = int(parts[2]) if len(parts) == 3 else 1
        if step == 0:
            raise ValueError("seed range step cannot be zero")
        seeds = tuple(range(start, stop, step))
    else:
        seeds = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not seeds:
        raise ValueError("seed specification selects no seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("evaluation seeds must be unique")
    return seeds


@dataclass(frozen=True)
class EvaluationProtocol:
    """Plant, reset, and timing settings used for every reward arm."""

    seeds: tuple[int, ...] = DEFAULT_SEEDS
    t_max: float = DEFAULT_T_MAX
    dt: float = DEFAULT_DT
    physics_dt: float = DEFAULT_PHYSICS_DT
    damping: float = DEFAULT_DAMPING
    torque_limit: float = DEFAULT_TORQUE_LIMIT
    release_angle_range: tuple[float, float] = DEFAULT_RELEASE_ANGLE_RANGE

    def __post_init__(self) -> None:
        seeds = tuple(int(seed) for seed in self.seeds)
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("protocol seeds must be non-empty and unique")
        object.__setattr__(self, "seeds", seeds)
        for name in ("t_max", "dt", "physics_dt", "torque_limit"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {value}")
            object.__setattr__(self, name, value)
        damping = float(self.damping)
        if not np.isfinite(damping) or damping < 0.0:
            raise ValueError(f"damping must be finite and >= 0, got {damping}")
        object.__setattr__(self, "damping", damping)
        low, high = (float(value) for value in self.release_angle_range)
        if not (np.isfinite(low) and np.isfinite(high) and 0.0 < low < high):
            raise ValueError(
                "release_angle_range must be finite and satisfy 0 < low < high"
            )
        object.__setattr__(self, "release_angle_range", (low, high))

    @property
    def max_steps(self) -> int:
        return int(math.ceil(self.t_max / self.dt - 1e-12))

    def as_metadata(self) -> dict[str, Any]:
        values = asdict(self)
        values["seeds"] = list(self.seeds)
        values["start"] = "release"
        values["max_steps"] = self.max_steps
        values["time_sampling"] = "uniform"
        values["raw_state_obs"] = True
        values["return_reward_increment"] = False
        return values


@dataclass(frozen=True)
class RewardMetadata:
    """Training-arm identity; reward values never enter the reported metrics."""

    reward_kind: str
    eta: Optional[float]

    def __post_init__(self) -> None:
        kind = str(self.reward_kind).strip().lower()
        if kind not in REWARD_KINDS:
            raise ValueError(
                f"reward_kind must be one of {REWARD_KINDS}, got {self.reward_kind!r}"
            )
        object.__setattr__(self, "reward_kind", kind)
        if self.eta is None:
            eta = None
        else:
            eta = float(self.eta)
            if not np.isfinite(eta) or eta < 0.0:
                raise ValueError(f"eta must be finite and >= 0, got {self.eta}")
        if kind == "r2" and eta is None:
            raise ValueError("reward_kind='r2' requires --eta or configured task eta")
        if kind != "r2" and eta is not None:
            raise ValueError("eta is only meaningful for reward_kind='r2'")
        object.__setattr__(self, "eta", eta)

    def task_values(self) -> dict[str, Any]:
        values: dict[str, Any] = {"reward_kind": self.reward_kind}
        if self.eta is not None:
            values["eta"] = self.eta
        return values


@dataclass(frozen=True)
class EvaluationResult:
    """In-memory per-episode and aggregate evaluator output."""

    rows: tuple[dict[str, Any], ...]
    metrics: tuple[EpisodeMetrics, ...]
    summary: dict[str, Any]


def resolve_reward_metadata(
    train_env_kwargs: Mapping[str, Any],
    *,
    reward_kind: Optional[str] = None,
    eta: Optional[float] = None,
) -> RewardMetadata:
    """Resolve explicit sweep metadata over values stored in ``task_kwargs``.

    Explicit arguments take precedence.  This supports checkpoints produced by
    a sweep launcher that kept a common model/config row while varying task
    parameters externally.  The full loaded config hash is also retained in
    the output, so the provenance of an override remains visible.
    """

    task_kwargs = dict(train_env_kwargs.get("task_kwargs", {}) or {})
    configured_kind = task_kwargs.get("reward_kind", "r0")
    kind = configured_kind if reward_kind is None else reward_kind
    if eta is not None:
        chosen_eta: Optional[float] = eta
    elif str(kind).strip().lower() == "r2":
        chosen_eta = task_kwargs.get("eta")
    else:
        chosen_eta = None
    return RewardMetadata(str(kind), chosen_eta)


def build_task_kwargs(
    train_env_kwargs: Mapping[str, Any],
    reward: RewardMetadata,
    protocol: EvaluationProtocol,
) -> dict[str, Any]:
    """Build the task portion of the fixed protocol.

    ``k_d`` and ``k_p`` are preserved when the training row explicitly set
    them because they define ``r1``/``r2`` metadata.  Every plant or reset value
    that affects the comparison is fixed by ``protocol``.
    """

    configured = dict(train_env_kwargs.get("task_kwargs", {}) or {})
    task = {
        key: configured[key]
        for key in ("k_d", "k_p")
        if key in configured
    }
    task.update(
        damping=protocol.damping,
        torque_limit=protocol.torque_limit,
        uniform_start=False,
        paper_start=False,
        release_start=True,
        release_angle_range=protocol.release_angle_range,
        **reward.task_values(),
    )
    return task


def build_env(
    *,
    seed: int,
    protocol: EvaluationProtocol,
    task_kwargs: Mapping[str, Any],
):
    """Construct one Acrobot-XK environment under the documented protocol."""

    # Lazy import keeps ``--help`` and pure aggregation tests usable without
    # initializing dm_control/MuJoCo.
    from environment.dmc import DMCContinuousEnv

    return DMCContinuousEnv(
        domain_name="acrobot",
        task_name="swingup-xk",
        seed=int(seed),
        raw_state_obs=True,
        time_sampling="uniform",
        dt=protocol.dt,
        physics_dt=protocol.physics_dt,
        max_steps=protocol.max_steps,
        episode_duration=protocol.t_max,
        return_reward_increment=False,
        task_kwargs=dict(task_kwargs),
    )


def load_training_config(mode: str, hyperparams_dir: Path):
    """Load the training row through the shared CT-SAC config helper."""

    from evaluations.evaluate_swingup_final import _load_config_bundle

    return _load_config_bundle(ENV_ID, mode, Path(hyperparams_dir))


def load_checkpoint_model(env, model_kwargs, checkpoint: Path, device: str):
    """Strictly reconstruct and load ``ActorQCriticModel`` via shared code."""

    from evaluations.evaluate_swingup_final import _load_model

    return _load_model(env, model_kwargs, Path(checkpoint), device)


class DeterministicCTPolicy:
    """Callable adapter for ``acrobot_homoclinic_metrics.rollout``."""

    def __init__(self, model) -> None:
        self.model = model

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        import torch as th

        obs = th.as_tensor(
            observation, dtype=th.float32, device=self.model.device
        ).reshape(1, -1)
        with th.no_grad():
            action, _ = self.model.act(obs, deterministic=True)
        return action.detach().cpu().numpy()[0]


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_train_seed(path: Path) -> Optional[int]:
    """Infer the conventional ``seed_N`` training seed from a checkpoint path."""

    for part in reversed(Path(path).parts):
        match = re.fullmatch(r"seed_(-?\d+)", part)
        if match:
            return int(match.group(1))
    return None


def _finite_stats(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "finite_count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p10": None,
            "p90": None,
        }
    return {
        "finite_count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "p10": float(np.quantile(finite, 0.10)),
        "p90": float(np.quantile(finite, 0.90)),
    }


def summarize_metrics(results: Sequence[EpisodeMetrics]) -> dict[str, Any]:
    """Aggregate every document metric without consulting episode rewards."""

    if not results:
        raise ValueError("at least one episode is required")
    primary = aggregate(results)
    capture_quantiles = {
        key: (float(value) if np.isfinite(value) else None)
        for key, value in primary.capture_quantiles.items()
    }
    lqr_hits = [np.isfinite(item.lqr_time) for item in results]
    return {
        "metric_1_successful_capture": {
            "episodes": len(results),
            "captures": int(sum(item.captured for item in results)),
            "success_rate": float(primary.success_rate),
        },
        "metric_2_time_to_capture_seconds": {
            **_finite_stats([item.capture_time for item in results]),
            "p10": capture_quantiles["p10"],
            "median": capture_quantiles["p50"],
            "p90": capture_quantiles["p90"],
        },
        "metric_3_set_retention": _finite_stats(
            [item.retention for item in results]
        ),
        "metric_4a_homoclinic_set_error": {
            "rms": _finite_stats([item.error_rms for item in results]),
            "energy_rms": _finite_stats(
                [item.error_rms_energy for item in results]
            ),
            "elbow_angle_rms": _finite_stats(
                [item.error_rms_angle for item in results]
            ),
            "elbow_rate_rms": _finite_stats(
                [item.error_rms_rate for item in results]
            ),
        },
        "metric_4b_homoclinic_orbit_distance": {
            "post_capture_rms": _finite_stats(
                [item.orbit_distance_rms for item in results]
            ),
            "final": _finite_stats(
                [item.orbit_distance_final for item in results]
            ),
        },
        "metric_5_control": {
            "effort": _finite_stats([item.control_effort for item in results]),
            "saturation_fraction": _finite_stats(
                [item.saturation for item in results]
            ),
        },
        "metric_6_lqr_region": {
            "entries": int(sum(lqr_hits)),
            "entry_rate": float(np.mean(lqr_hits)),
            "time_seconds": _finite_stats([item.lqr_time for item in results]),
            "minimum_residual": _finite_stats(
                [item.lqr_residual_min for item in results]
            ),
        },
    }


def load_metric7_summary(
    npz_path: Path,
    *,
    legacy_seconds_per_timestep: Optional[float] = None,
) -> dict[str, Any]:
    """Load and summarize metric 7 at the document's 50/80/90% targets."""

    try:
        from evaluations.acrobot_training_metrics import (
            load_capture_learning_curve,
        )
    except ImportError as exc:  # Keeps this evaluator usable on older branches.
        raise RuntimeError(
            "metric 7 requires evaluations/acrobot_training_metrics.py"
        ) from exc

    curve = load_capture_learning_curve(
        Path(npz_path),
        legacy_seconds_per_timestep=legacy_seconds_per_timestep,
    )
    crossings = curve.training_times(DEFAULT_METRIC7_TARGETS)
    thresholds = {
        f"{int(round(target * 100))}%": (
            float(value) if np.isfinite(value) else None
        )
        for target, value in crossings.items()
    }
    return {
        "source": str(Path(npz_path).expanduser().resolve()),
        "axis": "cumulative_simulated_physical_seconds",
        "evaluation_checkpoints": int(curve.simulated_seconds.size),
        "final_success_rate": float(curve.success_rates[-1]),
        "simulated_seconds_thresholds": thresholds,
        "legacy_seconds_per_timestep": legacy_seconds_per_timestep,
    }


def evaluate_checkpoint(
    *,
    checkpoint: Path,
    mode: str,
    config,
    protocol: EvaluationProtocol = EvaluationProtocol(),
    reward_kind: Optional[str] = None,
    eta: Optional[float] = None,
    tube_spec: TubeSpec = TubeSpec(),
    lqr_threshold: float = LQR_SWITCH_THRESHOLD,
    device: str = "cpu",
    evaluations_npz: Optional[Path] = None,
    legacy_seconds_per_timestep: Optional[float] = None,
) -> EvaluationResult:
    """Evaluate one checkpoint on all protocol seeds and return both outputs."""

    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    reward = resolve_reward_metadata(
        config.env_kwargs, reward_kind=reward_kind, eta=eta
    )
    task_kwargs = build_task_kwargs(config.env_kwargs, reward, protocol)
    task_metadata_json = json.dumps(
        task_kwargs, sort_keys=True, separators=(",", ":")
    )
    checkpoint_sha = _sha256_path(checkpoint)
    train_seed = infer_train_seed(checkpoint)
    config_sha = getattr(config, "config_sha256", None)

    env = build_env(
        seed=protocol.seeds[0], protocol=protocol, task_kwargs=task_kwargs
    )
    metrics: list[EpisodeMetrics] = []
    rows: list[dict[str, Any]] = []
    try:
        model = load_checkpoint_model(
            env, config.model_kwargs, checkpoint, device
        )
        policy = DeterministicCTPolicy(model)
        params = AcrobotParams.from_physics(env._env.physics)
        for seed in protocol.seeds:
            episode = evaluate_episode(
                rollout(env, policy, int(seed), torque_limit=protocol.torque_limit),
                params,
                tube_spec,
                lqr_threshold=lqr_threshold,
            )
            metrics.append(episode)
            row = {
                "schema_version": SCHEMA_VERSION,
                "algorithm": ALGORITHM,
                "env_id": ENV_ID,
                "mode": str(mode),
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "train_seed": train_seed,
                "config_sha256": config_sha,
                "reward_kind": reward.reward_kind,
                "eta": reward.eta,
                "task_metadata_json": task_metadata_json,
                "seed": int(seed),
                "start": "release",
                "damping": protocol.damping,
                "torque_limit": protocol.torque_limit,
                "t_max": protocol.t_max,
                "dt": protocol.dt,
                "physics_dt": protocol.physics_dt,
                "tube_energy": tube_spec.energy_tolerance,
                "tube_angle": tube_spec.angle_tolerance,
                "tube_rate": tube_spec.rate_tolerance,
                "dwell_seconds": tube_spec.dwell_seconds,
                "lqr_threshold": float(lqr_threshold),
                **episode.as_row(),
            }
            rows.append({field: row.get(field) for field in OUTPUT_FIELDS})
    finally:
        env.close()

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": ALGORITHM,
        "env_id": ENV_ID,
        "mode": str(mode),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "train_seed": train_seed,
        "config_sha256": config_sha,
        "reward_metadata": asdict(reward),
        "task_metadata": task_kwargs,
        "protocol": protocol.as_metadata(),
        "tube": {
            "energy_tolerance": tube_spec.energy_tolerance,
            "angle_tolerance": tube_spec.angle_tolerance,
            "rate_tolerance": tube_spec.rate_tolerance,
            "dwell_seconds": tube_spec.dwell_seconds,
        },
        "lqr_threshold": float(lqr_threshold),
        "metrics": summarize_metrics(metrics),
    }
    if evaluations_npz is not None:
        summary["metric_7_training_time"] = load_metric7_summary(
            evaluations_npz,
            legacy_seconds_per_timestep=legacy_seconds_per_timestep,
        )
    return EvaluationResult(tuple(rows), tuple(metrics), summary)


def write_outputs(
    result: EvaluationResult,
    output: Path,
    *,
    summary_output: Optional[Path] = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write per-episode CSV and aggregate JSON, refusing silent overwrite."""

    output = Path(output).expanduser().resolve()
    if summary_output is None:
        summary_output = output.with_suffix(".summary.json")
    else:
        summary_output = Path(summary_output).expanduser().resolve()
    for path in (output, summary_output):
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"output already exists: {path}; pass --overwrite to replace it"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(OUTPUT_FIELDS))
        writer.writeheader()
        writer.writerows(result.rows)
    with summary_output.open("w", encoding="utf-8") as stream:
        json.dump(result.summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return output, summary_output


def print_summary(summary: Mapping[str, Any]) -> None:
    """Print the common comparison headline plus optional metric 7."""

    metrics = summary["metrics"]
    capture = metrics["metric_1_successful_capture"]
    times = metrics["metric_2_time_to_capture_seconds"]
    retention = metrics["metric_3_set_retention"]
    print(
        f"capture: {capture['captures']}/{capture['episodes']} "
        f"({capture['success_rate']:.3f})"
    )
    if times["median"] is None:
        print("T_cap: no captures")
    else:
        print(
            "T_cap p10/p50/p90 [s]: "
            f"{times['p10']:.3f} / {times['median']:.3f} / {times['p90']:.3f}"
        )
    if retention["median"] is not None:
        print(f"retention median: {retention['median']:.3f}")
    metric7 = summary.get("metric_7_training_time")
    if metric7 is not None:
        text = []
        for label, seconds in metric7["simulated_seconds_thresholds"].items():
            text.append(
                f"{label}={'not reached' if seconds is None else seconds}"
            )
        print("cumulative simulated training seconds: " + ", ".join(text))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--mode",
        required=True,
        help="ct_sac.csv mode used to reconstruct the actor architecture",
    )
    parser.add_argument("--reward-kind", choices=REWARD_KINDS, default=None)
    parser.add_argument(
        "--eta",
        type=float,
        default=None,
        help="r2 shaping time scale; explicit so eta sweeps remain identifiable",
    )
    parser.add_argument(
        "--hyperparams-dir", type=Path, default=Path("benchmarks/hyperparams")
    )
    parser.add_argument("--evaluations-npz", type=Path, default=None)
    parser.add_argument(
        "--legacy-seconds-per-step",
        type=float,
        default=None,
        help=(
            "explicit conversion for legacy uniform-step NPZ files that have "
            "only timestep counts; invalid for irregular sampling"
        ),
    )
    parser.add_argument("--seeds", default="20000:20032")
    parser.add_argument("--t-max", type=float, default=DEFAULT_T_MAX)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--physics-dt", type=float, default=DEFAULT_PHYSICS_DT)
    parser.add_argument("--damping", type=float, default=DEFAULT_DAMPING)
    parser.add_argument("--torque-limit", type=float, default=DEFAULT_TORQUE_LIMIT)
    parser.add_argument("--tube-energy", type=float, default=0.05)
    parser.add_argument("--tube-angle", type=float, default=0.025)
    parser.add_argument("--tube-rate", type=float, default=0.05)
    parser.add_argument("--dwell-seconds", type=float, default=1.0)
    parser.add_argument("--lqr-threshold", type=float, default=LQR_SWITCH_THRESHOLD)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output", type=Path, default=Path("results/acrobot_xk_ctsac.csv")
    )
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if (
            args.legacy_seconds_per_step is not None
            and args.evaluations_npz is None
        ):
            raise ValueError(
                "--legacy-seconds-per-step requires --evaluations-npz"
            )
        protocol = EvaluationProtocol(
            seeds=parse_seed_spec(args.seeds),
            t_max=args.t_max,
            dt=args.dt,
            physics_dt=args.physics_dt,
            damping=args.damping,
            torque_limit=args.torque_limit,
        )
        tube = TubeSpec(
            energy_tolerance=args.tube_energy,
            angle_tolerance=args.tube_angle,
            rate_tolerance=args.tube_rate,
            dwell_seconds=args.dwell_seconds,
        )
        config = load_training_config(args.mode, args.hyperparams_dir)
        result = evaluate_checkpoint(
            checkpoint=args.checkpoint,
            mode=args.mode,
            config=config,
            protocol=protocol,
            reward_kind=args.reward_kind,
            eta=args.eta,
            tube_spec=tube,
            lqr_threshold=args.lqr_threshold,
            device=args.device,
            evaluations_npz=args.evaluations_npz,
            legacy_seconds_per_timestep=args.legacy_seconds_per_step,
        )
        csv_path, json_path = write_outputs(
            result,
            args.output,
            summary_output=args.summary_output,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print_summary(result.summary)
    print(f"per-episode CSV: {csv_path}")
    print(f"aggregate summary: {json_path}")


if __name__ == "__main__":
    main()
