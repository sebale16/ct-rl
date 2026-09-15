"""CSV presets for the dedicated oracle Acrobot value-flow runner."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


ENV_ID = "acrobot-stage-a"
DEFAULT_MODE = "stage_a_hard"
DEFAULT_HYPERPARAMS_DIR = Path(__file__).resolve().parent / "hyperparams"

# Same env_/model_/algo_/log_ convention as the CT-SAC table, adapted to
# Stage A's actual parameters. A value-flow update is not a SAC timestep.
CSV_ARGUMENTS = {
    "total_updates": "updates",
    "env_dt": "dt", "env_physics_dt": "physics_dt",
    "env_episode_duration": "episode_seconds", "env_hold_seconds": "hold_seconds",
    "env_angle_radius": "angle_radius", "env_velocity_radius": "velocity_radius",
    "env_capture_angle": "capture_angle", "env_capture_velocity": "capture_velocity",
    "env_torque_limit": "torque_limit", "env_damping": "damping",
    "env_velocity_limit": "velocity_limit", "env_elbow_limit": "elbow_limit",
    "env_shoulder_limit": "shoulder_limit",
    "env_reward_scale": "reward_scale",
    "env_state_cost_transform": "state_cost_transform",
    "env_log_reference_angle_deg": "log_reference_angle_deg",
    "env_incoming_probability": "incoming_probability",
    "model_hidden_width": "hidden_width", "model_momentum_scale": "momentum_scale",
    "algo_discount_rate": "discount_rate", "algo_value_step": "value_step",
    "algo_learning_rate": "learning_rate", "algo_target_rate": "target_rate",
    "algo_target_interval": "target_interval",
    "algo_grad_clip": "grad_clip", "algo_temperature": "temperature",
    "algo_auto_temperature": "auto_temperature",
    "algo_temperature_learning_rate": "temperature_learning_rate",
    "algo_target_entropy": "target_entropy",
    "algo_temperature_min": "temperature_min", "algo_temperature_max": "temperature_max",
    "algo_quadrature_points": "quadrature_points", "algo_exploration_std": "exploration_std",
    "algo_batch_size": "batch_size", "algo_buffer_size": "buffer_size",
    "log_eval_freq": "eval_every", "log_eval_episodes": "eval_episodes",
    **{f"env_{key}": key for key in (
        "angle1_weight", "angle2_weight", "velocity1_weight", "velocity2_weight",
        "velocity_scale", "effort_weight")},
}


def parse_bool(value):
    if value.lower() in ("true", "1"):
        return True
    if value.lower() in ("false", "0"):
        return False
    raise ValueError("expected true/false or 1/0")


def load_preset(directory, mode, argument_types):
    path = Path(directory) / "acrobot_ph_value.csv"
    # Parse and hash the same bytes for reproducible provenance.
    contents = path.read_bytes()
    reader = csv.DictReader(contents.decode("utf-8-sig").splitlines())
    header = reader.fieldnames or []
    if len(header) != len(set(header)) or not {"mode", "env_id"} <= set(header):
        raise ValueError("CSV requires unique headers including mode and env_id")
    unknown = set(header) - set(CSV_ARGUMENTS) - {"mode", "env_id", "comment"}
    if unknown:
        raise ValueError(f"unsupported Stage A CSV columns: {sorted(unknown)}")
    rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError("CSV row length does not match its header")
    matches = [row for row in rows if row["mode"].strip() == mode and row["env_id"].strip() == ENV_ID]
    if len(matches) != 1:
        raise ValueError(f"expected one row for env_id={ENV_ID!r}, mode={mode!r} in {path}; found {len(matches)}")
    row = matches[0]
    defaults = {}
    for column, destination in CSV_ARGUMENTS.items():
        value = row.get(column, "").strip()
        if value:
            try:
                defaults[destination] = argument_types[destination](value)
            except (ValueError, TypeError, argparse.ArgumentTypeError) as exc:
                raise ValueError(f"invalid {column}={value!r} in mode {mode!r}") from exc
    source = {"path": str(path.resolve()), "sha256": hashlib.sha256(contents).hexdigest(),
              "env_id": ENV_ID, "mode": mode, "row": row}
    return defaults, source


class StageAArgumentParser(argparse.ArgumentParser):
    """Read a preset first; explicit CLI arguments override its values."""

    def parse_args(self, args=None, namespace=None):
        if not hasattr(self, "_base_defaults"):
            self._base_defaults = {action.dest: action.default for action in self._actions}
        self.set_defaults(**self._base_defaults)
        selector = argparse.ArgumentParser(add_help=False)
        selector.add_argument("--mode", default=DEFAULT_MODE)
        selector.add_argument("--hyperparams-dir", "--hyperparams_dir", type=Path, default=DEFAULT_HYPERPARAMS_DIR)
        selector.add_argument("--checkpoint")
        selected, _ = selector.parse_known_args(args)
        # Checkpoint evaluation uses the recorded physical/model configuration,
        # not whichever training CSV happens to be present today.
        if selected.checkpoint:
            self.set_defaults(hyperparams_source=None)
        else:
            types = {action.dest: parse_bool if isinstance(action, argparse.BooleanOptionalAction) else action.type
                     for action in self._actions}
            try:
                defaults, source = load_preset(selected.hyperparams_dir, selected.mode, types)
            except (OSError, ValueError) as exc:
                self.error(str(exc))
            self.set_defaults(**defaults, hyperparams_source=source)
        return super().parse_args(args, namespace)
