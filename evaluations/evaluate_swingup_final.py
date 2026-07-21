"""Paired final-checkpoint evaluation for CartPole and Acrobot swing-up.

This evaluator intentionally does not search ``saved_models``.  Every
``final_model.pth`` is supplied explicitly, either with repeated
``--checkpoint ENV_ID MODE PATH`` arguments or in a CSV/JSON/JSONL manifest
containing ``env_id``, ``mode``, and ``checkpoint_path`` fields.  Relative
manifest paths are resolved relative to the manifest itself.

For each checkpoint and explicit evaluation seed, the deterministic policy is
run twice from the same reset state:

* ``irregular_train`` uses the timing configuration from the mode's CSV row;
* ``uniform_0p01`` uses a uniform 0.01-second control interval.

The primary reward statistic is the physical-time average
``sum(reward * dt_used) / sum(dt_used)``.  This avoids giving an irregular
schedule more or less weight merely because it contains a different number of
control decisions.  Target occupancies use the same endpoint-sampled,
``dt_used``-weighted convention.

Example::

    python -m evaluations.evaluate_swingup_final \
      --checkpoint cartpole-swingup final_mf /runs/cp/final_model.pth \
      --checkpoint acrobot-swingup-v2 final_mf /runs/ac/final_model.pth \
      --output results/swingup_final.csv

The default seed specification ``20000:20100`` is stop-exclusive and therefore
evaluates exactly 100 paired starts per checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SUPPORTED_ENVS = ("cartpole-swingup", "acrobot-swingup-v2", "acrobot-swingup-v3")
REGIMES = ("irregular_train", "uniform_0p01")
SCHEMA_VERSION = 2
UNIFORM_DT = 0.01


@dataclass(frozen=True)
class CheckpointSpec:
    env_id: str
    mode: str
    checkpoint_path: Path
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepSample:
    """One endpoint sample and the physical interval represented by it."""

    t0: float
    t1: float
    dt_used: float
    reward: float
    next_obs: np.ndarray
    info: Mapping[str, Any]


@dataclass(frozen=True)
class ConfigBundle:
    total_timesteps: int
    env_kwargs: Mapping[str, Any]
    model_kwargs: Mapping[str, Any]
    algo_kwargs: Mapping[str, Any]
    log_kwargs: Mapping[str, Any]
    config_json: str
    config_sha256: str
    hyperparams_path: Path
    hyperparams_sha256: str


OUTPUT_FIELDS = (
    "schema_version",
    "algorithm",
    "env_id",
    "mode",
    "checkpoint_index",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_metadata_json",
    "hyperparams_path",
    "hyperparams_sha256",
    "config_sha256",
    "config_json",
    "configured_total_timesteps",
    "train_seed",
    "eval_seed",
    "eval_regime",
    "time_sampling",
    "nominal_dt",
    "episode_steps",
    "episode_duration_used",
    "dt_used_min",
    "dt_used_max",
    "dt_used_mean",
    "dt_used_std",
    "terminated",
    "truncated",
    "raw_terminated",
    "raw_truncated",
    "time_limit_reached",
    "episode_return",
    "reward_time_integral",
    "time_weighted_normalized_reward",
    "uniform_0p01_equivalent_return",
    "acrobot_hit_distance",
    "acrobot_hit",
    "acrobot_time_weighted_hit_occupancy",
    "acrobot_last_1s_hit_occupancy",
    "acrobot_sustained_seconds",
    "acrobot_sustained_hit",
    "acrobot_max_sustained_hit_seconds",
    "acrobot_time_to_first_hit",
    "acrobot_time_to_sustained_hit",
    "acrobot_min_tip_distance",
    "acrobot_final_tip_distance",
    "acrobot_max_progress",
    "acrobot_max_tip_height",
    "cartpole_upright_cosine_threshold",
    "cartpole_center_radius",
    "cartpole_upright_hit",
    "cartpole_centered_hit",
    "cartpole_upright_centered_hit",
    "cartpole_time_weighted_upright_occupancy",
    "cartpole_time_weighted_centered_occupancy",
    "cartpole_time_weighted_upright_centered_occupancy",
    "cartpole_time_to_first_upright",
    "cartpole_time_to_first_centered",
    "cartpole_time_to_first_upright_centered",
    "cartpole_max_pole_cosine",
    "cartpole_min_abs_cart_position",
)


def parse_seed_spec(value: str) -> tuple[int, ...]:
    """Parse ``start:stop[:step]`` or a comma-separated list of integers."""

    text = str(value).strip()
    if not text:
        raise ValueError("evaluation seed specification cannot be empty")
    if ":" in text:
        if "," in text:
            raise ValueError("use either a seed range or a comma-separated list")
        parts = text.split(":")
        if len(parts) not in (2, 3) or any(part.strip() == "" for part in parts):
            raise ValueError(
                "seed range must have the form start:stop or start:stop:step"
            )
        start, stop = int(parts[0]), int(parts[1])
        step = int(parts[2]) if len(parts) == 3 else 1
        if step == 0:
            raise ValueError("seed range step cannot be zero")
        seeds = tuple(range(start, stop, step))
    else:
        seeds = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not seeds:
        raise ValueError("evaluation seed specification selects no seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("evaluation seeds must be unique")
    return seeds


def _canonicalize(value: Any) -> Any:
    """Convert loaded configuration values to stable, JSON-safe metadata."""

    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonicalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_to_spec(record: Mapping[str, Any], *, base_dir: Path) -> CheckpointSpec:
    missing = [key for key in ("env_id", "mode") if not str(record.get(key, "")).strip()]
    path_value = record.get("checkpoint_path", record.get("path"))
    if path_value is None or not str(path_value).strip():
        missing.append("checkpoint_path")
    if missing:
        raise ValueError(f"checkpoint manifest record is missing: {', '.join(missing)}")

    raw_path = Path(str(path_value)).expanduser()
    if not raw_path.is_absolute():
        raw_path = base_dir / raw_path
    metadata = {
        str(key): value
        for key, value in record.items()
        if key not in {"env_id", "mode", "checkpoint_path", "path"}
    }
    return CheckpointSpec(
        env_id=str(record["env_id"]).strip(),
        mode=str(record["mode"]).strip(),
        checkpoint_path=raw_path.resolve(),
        metadata=metadata,
    )


def load_checkpoint_manifest(path: Path) -> list[CheckpointSpec]:
    """Load checkpoint records from CSV, JSON, JSONL, or NDJSON."""

    manifest_path = path.expanduser().resolve()
    suffix = manifest_path.suffix.lower()
    if suffix == ".csv":
        with manifest_path.open("r", newline="", encoding="utf-8") as stream:
            records: Any = list(csv.DictReader(stream))
    elif suffix == ".json":
        with manifest_path.open("r", encoding="utf-8") as stream:
            records = json.load(stream)
        if isinstance(records, Mapping):
            records = records.get("checkpoints", records.get("records"))
    elif suffix in {".jsonl", ".ndjson"}:
        with manifest_path.open("r", encoding="utf-8") as stream:
            records = [
                json.loads(line)
                for line in stream
                if line.strip() and not line.lstrip().startswith("#")
            ]
    else:
        raise ValueError(
            f"unsupported manifest extension {suffix!r}; use .csv, .json, or .jsonl"
        )

    if not isinstance(records, list):
        raise ValueError("JSON manifest must be a list or contain a 'checkpoints' list")
    if not records:
        raise ValueError(f"checkpoint manifest is empty: {manifest_path}")
    if not all(isinstance(record, Mapping) for record in records):
        raise ValueError("every checkpoint manifest entry must be an object/row")
    return [
        _record_to_spec(record, base_dir=manifest_path.parent) for record in records
    ]


def _validate_specs(specs: Sequence[CheckpointSpec]) -> None:
    if not specs:
        raise ValueError("provide at least one --checkpoint or --manifest")
    identities: set[tuple[str, str, Path]] = set()
    for spec in specs:
        if spec.env_id not in SUPPORTED_ENVS:
            raise ValueError(
                f"unsupported final-evaluation environment {spec.env_id!r}; "
                f"choose one of {SUPPORTED_ENVS}"
            )
        if spec.checkpoint_path.name != "final_model.pth":
            raise ValueError(
                "final evaluation requires an explicit file named final_model.pth; "
                f"got {spec.checkpoint_path}"
            )
        if not spec.checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {spec.checkpoint_path}")
        identity = (spec.env_id, spec.mode, spec.checkpoint_path)
        if identity in identities:
            raise ValueError(f"duplicate checkpoint specification: {identity}")
        identities.add(identity)


def _checkpoint_train_seed(spec: CheckpointSpec) -> int | None:
    """Read the training seed from manifest metadata or the standard path."""

    for key in ("train_seed", "seed"):
        value = spec.metadata.get(key)
        if value is not None and str(value).strip():
            return int(value)
    for part in reversed(spec.checkpoint_path.parts):
        match = re.fullmatch(r"seed_(-?\d+)", part)
        if match:
            return int(match.group(1))
    return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _load_config_bundle(
    env_id: str, mode: str, hyperparams_dir: Path
) -> ConfigBundle:
    # Lazy import keeps ``python -m ... --help`` usable without the training
    # environment's optional MuJoCo dependencies being initialized.
    from common.utils import load_ct_hyperparams_from_table

    total, env, model, algo, log = load_ct_hyperparams_from_table(
        "ct_sac", env_id, mode, hyperparams_dir=hyperparams_dir
    )
    if not _as_bool(env.get("raw_state_obs", False)):
        raise ValueError(
            f"{env_id}/{mode} must set env_raw_state_obs=True for the final "
            "state-based swing-up metrics"
        )
    payload = {
        "algorithm": "ct_sac",
        "env_id": env_id,
        "mode": mode,
        "total_timesteps": total,
        "env_kwargs": env,
        "model_kwargs": model,
        "algo_kwargs": algo,
        "log_kwargs": log,
    }
    config_json = _canonical_json(payload)
    hyperparams_path = (hyperparams_dir / "ct_sac.csv").resolve()
    return ConfigBundle(
        total_timesteps=int(total),
        env_kwargs=dict(env),
        model_kwargs=dict(model),
        algo_kwargs=dict(algo),
        log_kwargs=dict(log),
        config_json=config_json,
        config_sha256=hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
        hyperparams_path=hyperparams_path,
        hyperparams_sha256=_sha256_path(hyperparams_path),
    )


def build_regime_env_kwargs(
    train_env_kwargs: Mapping[str, Any], regime: str
) -> dict[str, Any]:
    """Return the exact train-timing or uniform-0.01 evaluation environment."""

    if regime not in REGIMES:
        raise ValueError(f"unknown evaluation regime {regime!r}")
    kwargs = dict(train_env_kwargs)
    kwargs.pop("n_envs", None)
    # The evaluator integrates the already-normalized instantaneous task reward
    # itself.  Increment-mode would multiply by dt a second time.
    kwargs["return_reward_increment"] = False
    if regime == "irregular_train":
        if str(kwargs.get("time_sampling", "")).strip().lower() != "irregular":
            raise ValueError(
                "the final comparison's irregular_train regime requires an "
                "irregular training configuration"
            )
        duration = kwargs.get("episode_duration")
        if duration is not None:
            # ``max_steps`` is both the number of candidate intervals used to
            # build an irregular grid and a hard episode cap.  Training keeps
            # its configured cap, but fixed-horizon evaluation must give every
            # timing realization enough intervals to reach the same physical
            # endpoint as the uniform arm.  The worst case is one min_dt step
            # per interval.
            min_dt = kwargs.get("min_dt")
            if min_dt is None:
                min_dt = 0.5 * float(kwargs["dt"])
            duration = float(duration)
            min_dt = float(min_dt)
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("episode_duration must be finite and positive")
            if not math.isfinite(min_dt) or min_dt <= 0:
                raise ValueError("min_dt must be finite and positive")
            horizon_steps = int(math.ceil(duration / min_dt - 1e-12)) + 1
            kwargs["max_steps"] = max(
                int(kwargs.get("max_steps") or 0), horizon_steps
            )
    else:
        kwargs["time_sampling"] = "uniform"
        kwargs["dt"] = UNIFORM_DT
        kwargs["time_sampling_kwargs"] = {"dist": "uniform"}
        duration = kwargs.get("episode_duration")
        if duration is not None:
            uniform_steps = int(round(float(duration) / UNIFORM_DT))
            if uniform_steps < 1:
                raise ValueError("episode_duration must cover at least one uniform step")
            kwargs["max_steps"] = uniform_steps
    return kwargs


def _make_env(env_id: str, kwargs: Mapping[str, Any], seed: int):
    from environment.dmc import DMCContinuousEnv

    domain_name, task_name = env_id.split("-", 1)
    return DMCContinuousEnv(
        domain_name=domain_name,
        task_name=task_name,
        seed=int(seed),
        **dict(kwargs),
    )


def _load_model(env, model_kwargs: Mapping[str, Any], checkpoint: Path, device: str):
    from models.actor_q_critic import ActorQCriticModel

    kwargs = dict(model_kwargs)
    kwargs.pop("device", None)
    model = ActorQCriticModel(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        **kwargs,
    )
    try:
        _validate_checkpoint_architecture(model, checkpoint)
        model.load_state(str(checkpoint), strict=True)
    except Exception as exc:
        raise RuntimeError(
            f"checkpoint/config mismatch while loading {checkpoint}"
        ) from exc
    model.actor.eval()
    for critic in model.q_nets:
        critic.eval()
    if getattr(model, "has_v_head", False):
        model.v_net.eval()
    return model


def _validate_checkpoint_architecture(model, checkpoint: Path) -> None:
    """Reject checkpoint/model architecture mismatches before zipped loading.

    ``ActorQCriticModel.load_state(strict=True)`` delegates strictness to each
    individual PyTorch module.  Its critic loops use ``zip``, however, and the
    optional V/actor-target heads are conditional, so a missing or extra whole
    network would otherwise be accepted silently.
    """

    import torch as th

    try:
        payload = th.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions predating ``weights_only``.
        payload = th.load(checkpoint, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")

    expected_keys = {"actor", "critics", "critic_targets"}
    if getattr(model, "actor_target", None) is not None:
        expected_keys.add("actor_target")
    if bool(getattr(model, "has_v_head", False)):
        expected_keys.update(("v_net", "v_target_net"))
    actual_keys = set(payload)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(
            "checkpoint network keys do not match configured architecture: "
            f"missing={missing}, unexpected={unexpected}"
        )

    mapping_keys = expected_keys - {"critics", "critic_targets"}
    for key in sorted(mapping_keys):
        if not isinstance(payload[key], Mapping):
            raise ValueError(f"checkpoint entry {key!r} must be a state mapping")

    for key, networks in (
        ("critics", model.q_nets),
        ("critic_targets", model.q_target_nets),
    ):
        states = payload[key]
        if not isinstance(states, (list, tuple)):
            raise ValueError(f"checkpoint entry {key!r} must be a list")
        expected_count = len(networks)
        if len(states) != expected_count:
            raise ValueError(
                f"checkpoint entry {key!r} has {len(states)} networks; "
                f"configured model expects {expected_count}"
            )
        if not all(isinstance(state, Mapping) for state in states):
            raise ValueError(
                f"every network state in checkpoint entry {key!r} must be a mapping"
            )


def _policy_action(model, obs: np.ndarray) -> np.ndarray:
    import torch as th

    obs_tensor = th.as_tensor(
        np.asarray(obs, dtype=np.float32), dtype=th.float32, device=model.device
    ).unsqueeze(0)
    with th.no_grad():
        action_tensor, _ = model.act(obs_tensor, deterministic=True)
    action = action_tensor.detach().cpu().numpy()[0]
    if not np.all(np.isfinite(action)):
        raise FloatingPointError("deterministic policy emitted a non-finite action")
    return action


def _require_finite_scalar(mapping: Mapping[str, Any], key: str) -> float:
    if key not in mapping:
        raise KeyError(f"environment info is missing required metric {key!r}")
    value = float(mapping[key])
    if not math.isfinite(value):
        raise FloatingPointError(f"environment metric {key!r} is not finite")
    return value


def _cartpole_state(obs: np.ndarray) -> tuple[float, float]:
    raw = np.asarray(obs, dtype=np.float64).reshape(-1)
    if raw.size != 4:
        raise ValueError(
            "cartpole-swingup final metrics require raw [x, angle, xdot, angledot] "
            f"observations, got shape {raw.shape}"
        )
    return float(raw[0]), float(math.cos(float(raw[1])))


def _weighted_fraction(dt: np.ndarray, flags: np.ndarray) -> float:
    total = float(dt.sum())
    return float(np.dot(dt, flags.astype(np.float64)) / total) if total > 0 else 0.0


def _last_window_fraction(
    samples: Sequence[StepSample], flags: Sequence[bool], window_seconds: float
) -> float:
    final_t = float(samples[-1].t1)
    start = max(float(samples[0].t0), final_t - float(window_seconds))
    hit_time = 0.0
    covered = 0.0
    for sample, hit in zip(samples, flags):
        overlap = max(0.0, min(sample.t1, final_t) - max(sample.t0, start))
        covered += overlap
        if hit:
            hit_time += overlap
    return float(hit_time / covered) if covered > 0 else 0.0


def _first_hit_time(
    samples: Sequence[StepSample], flags: Sequence[bool], initial_hit: bool
) -> float | None:
    if initial_hit:
        return 0.0
    for sample, hit in zip(samples, flags):
        if hit:
            return float(sample.t1)
    return None


def _sustained_hit_summary(
    samples: Sequence[StepSample],
    flags: Sequence[bool],
    threshold: float,
    initial_hit: bool,
) -> tuple[bool, float, float | None]:
    # Hits are observed at transition endpoints.  When a trajectory first
    # enters the target at t1, we cannot claim it occupied the target over the
    # preceding [t0, t1] interval.  Start the sustained clock at that endpoint
    # and accrue time only between consecutive observed hits.
    run_start = float(samples[0].t0) if initial_hit else None
    previous_hit = bool(initial_hit)
    maximum = 0.0
    first_threshold_time: float | None = None
    for sample, hit in zip(samples, flags):
        if not hit:
            run_start = None
            previous_hit = False
            continue
        if not previous_hit:
            run_start = float(sample.t1)
        assert run_start is not None
        run = max(0.0, float(sample.t1) - run_start)
        maximum = max(maximum, run)
        if first_threshold_time is None and run + 1e-12 >= threshold:
            first_threshold_time = float(run_start + threshold)
        previous_hit = True
    return first_threshold_time is not None, float(maximum), first_threshold_time


def summarize_episode(
    *,
    env_id: str,
    initial_obs: np.ndarray,
    reset_info: Mapping[str, Any],
    samples: Sequence[StepSample],
    terminated: bool,
    truncated: bool,
    raw_terminated: bool | None = None,
    raw_truncated: bool | None = None,
    time_limit_reached: bool = False,
    acrobot_hit_distance: float = 0.2,
    sustained_seconds: float = 1.0,
    cartpole_upright_threshold: float = 0.995,
    cartpole_center_radius: float = 0.25,
) -> dict[str, Any]:
    """Summarize a completed episode using physical-time quadrature."""

    if not samples:
        raise ValueError("cannot summarize an episode with no transitions")
    dt = np.asarray([sample.dt_used for sample in samples], dtype=np.float64)
    rewards = np.asarray([sample.reward for sample in samples], dtype=np.float64)
    if not np.all(np.isfinite(dt)) or np.any(dt <= 0):
        raise ValueError("all dt_used values must be finite and positive")
    if not np.all(np.isfinite(rewards)):
        raise ValueError("all rewards must be finite")
    if np.any(rewards < -1e-7) or np.any(rewards > 1.0 + 1e-7):
        raise ValueError("swing-up task rewards must be normalized to [0, 1]")

    duration = float(dt.sum())
    reward_integral = float(np.dot(rewards, dt))
    result: dict[str, Any] = {key: None for key in OUTPUT_FIELDS}
    result.update(
        episode_steps=len(samples),
        episode_duration_used=duration,
        dt_used_min=float(dt.min()),
        dt_used_max=float(dt.max()),
        dt_used_mean=float(dt.mean()),
        dt_used_std=float(dt.std()),
        terminated=bool(terminated),
        truncated=bool(truncated),
        raw_terminated=bool(terminated if raw_terminated is None else raw_terminated),
        raw_truncated=bool(truncated if raw_truncated is None else raw_truncated),
        time_limit_reached=bool(time_limit_reached),
        episode_return=float(rewards.sum()),
        reward_time_integral=reward_integral,
        time_weighted_normalized_reward=reward_integral / duration,
        uniform_0p01_equivalent_return=reward_integral / UNIFORM_DT,
    )

    if env_id in ("acrobot-swingup-v2", "acrobot-swingup-v3"):
        initial_distance = _require_finite_scalar(reset_info, "acrobot_tip_distance")
        initial_progress = _require_finite_scalar(reset_info, "acrobot_progress")
        initial_height = _require_finite_scalar(reset_info, "acrobot_tip_height")
        distances = np.asarray(
            [
                _require_finite_scalar(sample.info, "acrobot_tip_distance")
                for sample in samples
            ],
            dtype=np.float64,
        )
        progresses = np.asarray(
            [
                _require_finite_scalar(sample.info, "acrobot_progress")
                for sample in samples
            ],
            dtype=np.float64,
        )
        heights = np.asarray(
            [
                _require_finite_scalar(sample.info, "acrobot_tip_height")
                for sample in samples
            ],
            dtype=np.float64,
        )
        hit_flags = distances <= float(acrobot_hit_distance)
        initial_hit = initial_distance <= float(acrobot_hit_distance)
        sustained, max_sustained, time_to_sustained = _sustained_hit_summary(
            samples, hit_flags, float(sustained_seconds), initial_hit
        )
        result.update(
            acrobot_hit_distance=float(acrobot_hit_distance),
            acrobot_hit=bool(initial_hit or bool(np.any(hit_flags))),
            acrobot_time_weighted_hit_occupancy=_weighted_fraction(dt, hit_flags),
            acrobot_last_1s_hit_occupancy=_last_window_fraction(
                samples, hit_flags, 1.0
            ),
            acrobot_sustained_seconds=float(sustained_seconds),
            acrobot_sustained_hit=bool(sustained),
            acrobot_max_sustained_hit_seconds=max_sustained,
            acrobot_time_to_first_hit=_first_hit_time(
                samples, hit_flags, initial_hit
            ),
            acrobot_time_to_sustained_hit=time_to_sustained,
            acrobot_min_tip_distance=float(
                min(initial_distance, float(distances.min()))
            ),
            acrobot_final_tip_distance=float(distances[-1]),
            acrobot_max_progress=float(
                max(initial_progress, float(progresses.max()))
            ),
            acrobot_max_tip_height=float(
                max(initial_height, float(heights.max()))
            ),
        )
    elif env_id == "cartpole-swingup":
        initial_x, initial_cosine = _cartpole_state(initial_obs)
        states = [_cartpole_state(sample.next_obs) for sample in samples]
        positions = np.asarray([state[0] for state in states], dtype=np.float64)
        cosines = np.asarray([state[1] for state in states], dtype=np.float64)
        upright_flags = cosines >= float(cartpole_upright_threshold)
        centered_flags = np.abs(positions) <= float(cartpole_center_radius)
        combined_flags = np.logical_and(upright_flags, centered_flags)
        initial_upright = initial_cosine >= float(cartpole_upright_threshold)
        initial_centered = abs(initial_x) <= float(cartpole_center_radius)
        initial_combined = initial_upright and initial_centered
        result.update(
            cartpole_upright_cosine_threshold=float(cartpole_upright_threshold),
            cartpole_center_radius=float(cartpole_center_radius),
            cartpole_upright_hit=bool(initial_upright or np.any(upright_flags)),
            cartpole_centered_hit=bool(initial_centered or np.any(centered_flags)),
            cartpole_upright_centered_hit=bool(
                initial_combined or np.any(combined_flags)
            ),
            cartpole_time_weighted_upright_occupancy=_weighted_fraction(
                dt, upright_flags
            ),
            cartpole_time_weighted_centered_occupancy=_weighted_fraction(
                dt, centered_flags
            ),
            cartpole_time_weighted_upright_centered_occupancy=_weighted_fraction(
                dt, combined_flags
            ),
            cartpole_time_to_first_upright=_first_hit_time(
                samples, upright_flags, initial_upright
            ),
            cartpole_time_to_first_centered=_first_hit_time(
                samples, centered_flags, initial_centered
            ),
            cartpole_time_to_first_upright_centered=_first_hit_time(
                samples, combined_flags, initial_combined
            ),
            cartpole_max_pole_cosine=float(
                max(initial_cosine, float(cosines.max()))
            ),
            cartpole_min_abs_cart_position=float(
                min(abs(initial_x), float(np.abs(positions).min()))
            ),
        )
    else:
        raise ValueError(f"unsupported environment {env_id!r}")
    return result


def _run_episode(
    *,
    model,
    env,
    env_id: str,
    eval_seed: int,
    expected_episode_duration: float,
    max_episode_steps: int,
    acrobot_hit_distance: float,
    sustained_seconds: float,
    cartpole_upright_threshold: float,
    cartpole_center_radius: float,
) -> dict[str, Any]:
    initial_obs, reset_info = env.reset(seed=int(eval_seed))
    obs = np.asarray(initial_obs, dtype=np.float32)
    samples: list[StepSample] = []
    terminated = truncated = False

    while not (terminated or truncated):
        if len(samples) >= int(max_episode_steps):
            raise RuntimeError(
                f"evaluation seed {eval_seed} exceeded --max-episode-steps="
                f"{max_episode_steps}"
            )
        action = _policy_action(model, obs)
        _, t0, _, reward, next_obs, t1, terminated, truncated, info = env.step_dt(
            action
        )
        if "dt_used" not in info:
            raise KeyError(
                "environment step info lacks dt_used; final metrics must use the "
                "actual physics duration"
            )
        dt_used = float(info["dt_used"])
        if not math.isfinite(dt_used) or dt_used <= 0:
            raise ValueError(f"invalid dt_used={dt_used!r}")
        if not math.isclose(
            float(t1) - float(t0), dt_used, rel_tol=1e-9, abs_tol=1e-8
        ):
            raise ValueError(
                f"timestamp advance {float(t1) - float(t0)} disagrees with "
                f"dt_used {dt_used}"
            )
        next_obs_array = np.asarray(next_obs, dtype=np.float32)
        samples.append(
            StepSample(
                t0=float(t0),
                t1=float(t1),
                dt_used=dt_used,
                reward=float(reward),
                next_obs=next_obs_array.copy(),
                info=dict(info),
            )
        )
        obs = next_obs_array

    duration = float(sum(sample.dt_used for sample in samples))
    expected_duration = float(expected_episode_duration)
    tolerance = max(1e-7, expected_duration * 1e-7)
    if not math.isclose(
        duration, expected_duration, rel_tol=0.0, abs_tol=tolerance
    ):
        last_info = samples[-1].info if samples else {}
        raise RuntimeError(
            f"evaluation seed {eval_seed} ended after {duration:.9g}s; "
            f"the paired comparison requires {expected_duration:.9g}s "
            f"(terminated={terminated}, truncated={truncated}, "
            f"physics_error={bool(last_info.get('physics_error', False))})"
        )

    return summarize_episode(
        env_id=env_id,
        initial_obs=np.asarray(initial_obs, dtype=np.float32),
        reset_info=dict(reset_info),
        samples=samples,
        # dm_control reports its own step-limit as termination, whereas the
        # continuous wrapper reports its horizon as truncation.  Once the same
        # requested physical horizon is verified, normalize both representations
        # while retaining the raw flags for diagnostics.
        terminated=False,
        truncated=True,
        raw_terminated=bool(terminated),
        raw_truncated=bool(truncated),
        time_limit_reached=True,
        acrobot_hit_distance=acrobot_hit_distance,
        sustained_seconds=sustained_seconds,
        cartpole_upright_threshold=cartpole_upright_threshold,
        cartpole_center_radius=cartpole_center_radius,
    )


def evaluate_checkpoint(
    *,
    spec: CheckpointSpec,
    checkpoint_index: int,
    seeds: Sequence[int],
    config: ConfigBundle,
    device: str,
    max_episode_steps: int,
    acrobot_hit_distance: float,
    sustained_seconds: float,
    cartpole_upright_threshold: float,
    cartpole_center_radius: float,
) -> list[dict[str, Any]]:
    """Evaluate one checkpoint in paired regimes and return output rows."""

    regime_kwargs = {
        regime: build_regime_env_kwargs(config.env_kwargs, regime)
        for regime in REGIMES
    }
    envs = {}
    try:
        for regime, kwargs in regime_kwargs.items():
            envs[regime] = _make_env(spec.env_id, kwargs, int(seeds[0]))
        # Spaces are identical between timing regimes.  Loading once also
        # ensures the two arms use byte-identical policy parameters.
        model = _load_model(
            envs[REGIMES[0]], config.model_kwargs, spec.checkpoint_path, device
        )
        checkpoint_sha = _sha256_path(spec.checkpoint_path)
        metadata_json = _canonical_json(spec.metadata)
        common = {
            "schema_version": SCHEMA_VERSION,
            "algorithm": "ct_sac",
            "env_id": spec.env_id,
            "mode": spec.mode,
            "checkpoint_index": int(checkpoint_index),
            "checkpoint_path": str(spec.checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_metadata_json": metadata_json,
            "hyperparams_path": str(config.hyperparams_path),
            "hyperparams_sha256": config.hyperparams_sha256,
            "config_sha256": config.config_sha256,
            "config_json": config.config_json,
            "configured_total_timesteps": config.total_timesteps,
            "train_seed": _checkpoint_train_seed(spec),
        }

        rows: list[dict[str, Any]] = []
        for eval_seed in seeds:
            # Keep paired rows adjacent in every output format.
            for regime in REGIMES:
                summary = _run_episode(
                    model=model,
                    env=envs[regime],
                    env_id=spec.env_id,
                    eval_seed=int(eval_seed),
                    expected_episode_duration=float(
                        regime_kwargs[regime]["episode_duration"]
                    ),
                    max_episode_steps=max_episode_steps,
                    acrobot_hit_distance=acrobot_hit_distance,
                    sustained_seconds=sustained_seconds,
                    cartpole_upright_threshold=cartpole_upright_threshold,
                    cartpole_center_radius=cartpole_center_radius,
                )
                summary.update(common)
                summary.update(
                    eval_seed=int(eval_seed),
                    eval_regime=regime,
                    time_sampling=str(regime_kwargs[regime]["time_sampling"]),
                    nominal_dt=float(regime_kwargs[regime]["dt"]),
                )
                rows.append({key: summary.get(key) for key in OUTPUT_FIELDS})
        return rows
    finally:
        for env in envs.values():
            env.close()


def _write_rows(
    rows: Sequence[Mapping[str, Any]], output: Path, output_format: str, overwrite: bool
) -> None:
    destination = output.expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {destination}; pass --overwrite to replace it"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            if output_format == "csv":
                writer = csv.DictWriter(stream, fieldnames=list(OUTPUT_FIELDS))
                writer.writeheader()
                writer.writerows(rows)
            else:
                for row in rows:
                    stream.write(
                        json.dumps(
                            dict(row), sort_keys=True, separators=(",", ":"),
                            allow_nan=False,
                        )
                    )
                    stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _output_format(path: Path, explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".jsonl", ".ndjson"}:
        return "jsonl"
    raise ValueError("inferable output extensions are .csv, .jsonl, and .ndjson")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate explicit final CartPole/Acrobot checkpoints on paired "
            "irregular-train and uniform-0.01 timing."
        )
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        nargs=3,
        metavar=("ENV_ID", "MODE", "FINAL_MODEL"),
        default=[],
        help=(
            "Explicit checkpoint triple; repeat for every final_model.pth. "
            "Paths are resolved from the current directory."
        ),
    )
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        default=[],
        help=(
            "CSV/JSON/JSONL checkpoint manifest with env_id, mode, and "
            "checkpoint_path; may be repeated."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--format",
        choices=("csv", "jsonl"),
        default=None,
        help="Output format; inferred from --output when omitted.",
    )
    parser.add_argument(
        "--eval-seeds",
        default="20000:20100",
        help="Stop-exclusive range or comma list (default: 20000:20100).",
    )
    parser.add_argument(
        "--hyperparams-dir", type=Path, default=Path("benchmarks/hyperparams")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--checkpoint-index",
        type=int,
        default=None,
        help=(
            "Evaluate only this zero-based checkpoint from the combined input "
            "manifest. Use a distinct --output per index for job arrays."
        ),
    )
    parser.add_argument("--max-episode-steps", type=int, default=100_000)
    parser.add_argument("--acrobot-hit-distance", type=float, default=0.2)
    parser.add_argument("--sustained-seconds", type=float, default=1.0)
    parser.add_argument(
        "--cartpole-upright-threshold",
        type=float,
        default=0.995,
        help="Minimum cos(pole angle), matching dm_control's sparse target.",
    )
    parser.add_argument(
        "--cartpole-center-radius",
        type=float,
        default=0.25,
        help="Maximum |cart position|, matching dm_control's sparse target.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    seeds = parse_seed_spec(args.eval_seeds)
    if args.max_episode_steps < 1:
        raise ValueError("--max-episode-steps must be positive")
    if not math.isfinite(args.acrobot_hit_distance) or args.acrobot_hit_distance <= 0:
        raise ValueError("--acrobot-hit-distance must be finite and positive")
    if not math.isfinite(args.sustained_seconds) or args.sustained_seconds <= 0:
        raise ValueError("--sustained-seconds must be finite and positive")
    if not -1.0 <= args.cartpole_upright_threshold <= 1.0:
        raise ValueError("--cartpole-upright-threshold must be in [-1, 1]")
    if not math.isfinite(args.cartpole_center_radius) or args.cartpole_center_radius <= 0:
        raise ValueError("--cartpole-center-radius must be finite and positive")
    return seeds


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        seeds = _validate_args(args)
        specs = [
            _record_to_spec(
                {"env_id": env_id, "mode": mode, "checkpoint_path": path},
                base_dir=Path.cwd(),
            )
            for env_id, mode, path in args.checkpoint
        ]
        for manifest in args.manifest:
            specs.extend(load_checkpoint_manifest(manifest))
        if not specs:
            raise ValueError("provide at least one --checkpoint or --manifest")
        indexed_specs = list(enumerate(specs))
        if args.checkpoint_index is not None:
            if not 0 <= args.checkpoint_index < len(indexed_specs):
                raise ValueError(
                    f"--checkpoint-index must be in [0, {len(indexed_specs) - 1}]"
                )
            indexed_specs = [indexed_specs[args.checkpoint_index]]
        # In a job array, validate only the selected record.  Other manifest
        # checkpoints may still be training and need not exist yet.
        _validate_specs([spec for _, spec in indexed_specs])
        output_format = _output_format(args.output, args.format)
        output_path = args.output.expanduser().resolve()
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"output already exists: {output_path}; pass --overwrite to replace it"
            )

        config_cache: dict[tuple[str, str], ConfigBundle] = {}
        rows: list[dict[str, Any]] = []
        for progress_index, (checkpoint_index, spec) in enumerate(indexed_specs):
            key = (spec.env_id, spec.mode)
            if key not in config_cache:
                config_cache[key] = _load_config_bundle(
                    spec.env_id, spec.mode, args.hyperparams_dir
                )
            print(
                f"[{progress_index + 1}/{len(indexed_specs)}] {spec.env_id} "
                f"{spec.mode}: {spec.checkpoint_path}",
                file=sys.stderr,
                flush=True,
            )
            rows.extend(
                evaluate_checkpoint(
                    spec=spec,
                    checkpoint_index=checkpoint_index,
                    seeds=seeds,
                    config=config_cache[key],
                    device=args.device,
                    max_episode_steps=args.max_episode_steps,
                    acrobot_hit_distance=args.acrobot_hit_distance,
                    sustained_seconds=args.sustained_seconds,
                    cartpole_upright_threshold=args.cartpole_upright_threshold,
                    cartpole_center_radius=args.cartpole_center_radius,
                )
            )
        _write_rows(rows, args.output, output_format, args.overwrite)
        print(
            f"wrote {len(rows)} rows ({len(indexed_specs)} checkpoints x "
            f"{len(seeds)} seeds x {len(REGIMES)} regimes) to "
            f"{args.output.expanduser().resolve()}",
            file=sys.stderr,
        )
        return 0
    except (OSError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
