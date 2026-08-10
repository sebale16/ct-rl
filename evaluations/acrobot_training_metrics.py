"""Reward-independent training-time metric for Acrobot XK comparisons.

Metric 7 in ``docs/reward_shaping_for_acrobot_swingup.md`` is the cumulative
*simulated physical interaction time* at which a policy first achieves a
requested strict-capture success rate.  For irregularly sampled CT-SAC this is
the sum of the realized ``next_t - t`` intervals, not the number of policy
decisions.  The training callback persists the two inputs in
``evaluations.npz``:

* ``capture_simulated_seconds`` is the cumulative training interaction time at
  each strict-capture evaluation checkpoint; and
* ``capture_successes`` contains one strict-capture boolean per evaluation
  episode at that checkpoint.

This metric intentionally does not use reward, optimizer wall-clock time, or
evaluation interactions.  A crossing is credited at the first *observed*
checkpoint whose empirical capture rate reaches the target; no interpolation
is performed.  Legacy timestep-only artifacts require an explicit conversion
factor because a decision count cannot be converted for irregular sampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_CAPTURE_SUCCESS_TARGETS = (0.5, 0.8, 0.9)


def _validated_target(target: float) -> float:
    value = float(target)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(
            f"capture success target must be finite and in [0, 1], got {value}"
        )
    return value


def _validated_simulated_seconds(
    evaluation_simulated_seconds: Sequence[float],
) -> np.ndarray:
    raw = np.asarray(evaluation_simulated_seconds)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(
            "evaluation_simulated_seconds must be a non-empty 1D sequence"
        )
    try:
        values = raw.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluation_simulated_seconds must be numeric") from exc
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(
            "evaluation_simulated_seconds must be finite and non-negative"
        )
    if np.any(np.diff(values) <= 0.0):
        raise ValueError(
            "evaluation_simulated_seconds must be strictly increasing"
        )
    return values


def _validated_timesteps(evaluation_timesteps: Sequence[int]) -> np.ndarray:
    """Validate optional decision-count provenance (not metric 7's x-axis)."""
    raw = np.asarray(evaluation_timesteps)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("evaluation_timesteps must be a non-empty 1D sequence")
    try:
        values = raw.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluation_timesteps must be numeric") from exc
    if (
        not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values != np.floor(values))
    ):
        raise ValueError(
            "evaluation_timesteps must contain finite non-negative whole counts"
        )
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("evaluation_timesteps must be strictly increasing")
    return values.astype(np.int64)


def _capture_rows(capture_successes, evaluations: int) -> list[np.ndarray]:
    """Normalize rectangular and callback-style object arrays into rows."""
    try:
        raw = np.asarray(capture_successes)
    except ValueError:
        # Ragged rows are valid: a resumed run may have changed its evaluation
        # episode count.  NumPy 1.24+ requires object dtype for such input.
        raw = np.asarray(capture_successes, dtype=object)

    if raw.ndim == 0 or raw.ndim > 2:
        raise ValueError(
            "capture_successes must have one 1D episode-result row per evaluation"
        )
    if raw.ndim == 2:
        if raw.shape[0] != evaluations:
            raise ValueError(
                "capture_successes row count must match the evaluation time axis"
            )
        return [raw[index] for index in range(evaluations)]

    # A 1D object array from the custom callback holds one ndarray per
    # evaluation.  A plain 1D boolean array is also useful for the common
    # one-evaluation or one-episode-per-evaluation cases.
    nested = any(np.asarray(item).ndim > 0 for item in raw)
    if nested:
        if raw.size != evaluations:
            raise ValueError(
                "capture_successes row count must match the evaluation time axis"
            )
        return [np.asarray(item) for item in raw]
    if evaluations == 1:
        return [raw]
    if raw.size == evaluations:
        return [raw[index : index + 1] for index in range(evaluations)]
    raise ValueError(
        "capture_successes row count must match the evaluation time axis"
    )


def _validated_success_row(row: np.ndarray, index: int) -> np.ndarray:
    values = np.asarray(row)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            f"capture_successes[{index}] must be a non-empty 1D sequence"
        )
    if values.dtype.kind == "b":
        return values.astype(bool, copy=False)
    if values.dtype.kind in "iuf":
        numeric = values.astype(np.float64)
        if np.all(np.isfinite(numeric)) and np.all(
            (numeric == 0.0) | (numeric == 1.0)
        ):
            return numeric.astype(bool)
    elif values.dtype.kind == "O":
        items = values.tolist()
        if all(isinstance(item, (bool, np.bool_)) for item in items):
            return values.astype(bool)
        if all(
            isinstance(item, (int, float, np.integer, np.floating))
            and np.isfinite(float(item))
            and float(item) in (0.0, 1.0)
            for item in items
        ):
            return values.astype(bool)
    raise ValueError(
        f"capture_successes[{index}] must contain only booleans or binary values"
    )


@dataclass(frozen=True)
class CaptureLearningCurve:
    """Strict-capture success rate at each training evaluation checkpoint."""

    simulated_seconds: np.ndarray = field(repr=False)
    success_rates: np.ndarray = field(repr=False)
    evaluation_episode_counts: np.ndarray = field(repr=False)
    evaluation_timesteps: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        simulated_seconds = _validated_simulated_seconds(self.simulated_seconds)
        rates = np.asarray(self.success_rates, dtype=np.float64)
        counts = np.asarray(self.evaluation_episode_counts)
        if rates.ndim != 1 or rates.shape != simulated_seconds.shape:
            raise ValueError("success_rates must have one value per evaluation")
        if not np.all(np.isfinite(rates)) or np.any(
            (rates < 0.0) | (rates > 1.0)
        ):
            raise ValueError("success_rates must be finite and in [0, 1]")
        if counts.ndim != 1 or counts.shape != simulated_seconds.shape:
            raise ValueError(
                "evaluation_episode_counts must have one value per evaluation"
            )
        try:
            numeric_counts = counts.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("evaluation_episode_counts must be numeric") from exc
        if (
            not np.all(np.isfinite(numeric_counts))
            or np.any(numeric_counts <= 0.0)
            or np.any(numeric_counts != np.floor(numeric_counts))
        ):
            raise ValueError("evaluation_episode_counts must be positive integers")

        # Own immutable copies: a frozen dataclass should not change when a
        # caller mutates the arrays it passed in.
        simulated_seconds = simulated_seconds.copy()
        rates = rates.copy()
        counts = numeric_counts.astype(np.int64)
        if self.evaluation_timesteps is None:
            timesteps = None
        else:
            timesteps = _validated_timesteps(self.evaluation_timesteps)
            if timesteps.shape != simulated_seconds.shape:
                raise ValueError(
                    "evaluation_timesteps must have one value per evaluation"
                )
            timesteps = timesteps.copy()
            timesteps.setflags(write=False)
        simulated_seconds.setflags(write=False)
        rates.setflags(write=False)
        counts.setflags(write=False)
        object.__setattr__(self, "simulated_seconds", simulated_seconds)
        object.__setattr__(self, "success_rates", rates)
        object.__setattr__(self, "evaluation_episode_counts", counts)
        object.__setattr__(self, "evaluation_timesteps", timesteps)

    def first_time_at(self, target: float) -> float:
        """First observed simulated second with rate at least ``target``."""
        target = _validated_target(target)
        hits = np.flatnonzero(self.success_rates >= target)
        return (
            float(self.simulated_seconds[hits[0]])
            if hits.size
            else float("inf")
        )

    def training_times(
        self, targets: Sequence[float] = DEFAULT_CAPTURE_SUCCESS_TARGETS
    ) -> dict[float, float]:
        """Map each requested success target to its first observed crossing."""
        values = tuple(_validated_target(target) for target in targets)
        if not values:
            raise ValueError("at least one capture success target is required")
        if len(set(values)) != len(values):
            raise ValueError("capture success targets must be unique")
        return {target: self.first_time_at(target) for target in values}


def capture_learning_curve(
    evaluation_simulated_seconds: Sequence[float],
    capture_successes,
    *,
    evaluation_timesteps: Sequence[int] | None = None,
) -> CaptureLearningCurve:
    """Build metric 7's empirical success curve from callback artifacts."""
    simulated_seconds = _validated_simulated_seconds(
        evaluation_simulated_seconds
    )
    rows = _capture_rows(capture_successes, simulated_seconds.size)
    successes = [
        _validated_success_row(row, index) for index, row in enumerate(rows)
    ]
    return CaptureLearningCurve(
        simulated_seconds=simulated_seconds,
        success_rates=np.asarray([np.mean(row) for row in successes]),
        evaluation_episode_counts=np.asarray([row.size for row in successes]),
        evaluation_timesteps=evaluation_timesteps,
    )


def training_simulated_seconds_to_capture_success(
    evaluation_simulated_seconds: Sequence[float],
    capture_successes,
    targets: Sequence[float] = DEFAULT_CAPTURE_SUCCESS_TARGETS,
) -> dict[float, float]:
    """Compute metric 7 directly from in-memory callback history."""
    return capture_learning_curve(
        evaluation_simulated_seconds, capture_successes
    ).training_times(targets)


def load_capture_learning_curve(
    npz_path: str | Path,
    *,
    legacy_seconds_per_timestep: float | None = None,
) -> CaptureLearningCurve:
    """Load metric 7 inputs from an ``evaluations.npz`` file.

    New artifacts contain physical seconds directly.  A legacy fixed-step
    artifact can be converted only when the caller explicitly supplies
    ``legacy_seconds_per_timestep``.  This fallback must not be used for an
    irregular time sampler because no single count-to-seconds factor exists.
    """
    path = Path(npz_path)
    with np.load(path, allow_pickle=True) as data:
        if "capture_successes" not in data.files:
            raise ValueError(
                f"{path} has no capture_successes; available keys: {data.files}"
            )
        physical_key = None
        if "capture_simulated_seconds" in data.files:
            physical_key = "capture_simulated_seconds"
        elif "simulated_seconds" in data.files:
            physical_key = "simulated_seconds"

        physical_error = None
        if physical_key is not None:
            physical_candidate = np.array(data[physical_key], copy=True)
            try:
                simulated_seconds = _validated_simulated_seconds(
                    physical_candidate
                )
            except ValueError as exc:
                physical_error = exc
        else:
            simulated_seconds = None

        if physical_key is None or physical_error is not None:
            if legacy_seconds_per_timestep is None:
                if physical_error is None:
                    detail = (
                        "has no capture_simulated_seconds or simulated_seconds"
                    )
                else:
                    detail = f"has invalid {physical_key}: {physical_error}"
                raise ValueError(
                    f"{path} {detail}; timestep-only artifacts need an explicit "
                    "legacy_seconds_per_timestep (and cannot be converted for "
                    f"irregular sampling); available keys: {data.files}"
                )
            scale = float(legacy_seconds_per_timestep)
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(
                    "legacy_seconds_per_timestep must be finite and > 0"
                )
            if "capture_timesteps" in data.files:
                legacy_timesteps = np.array(data["capture_timesteps"], copy=True)
            elif "timesteps" in data.files:
                legacy_timesteps = np.array(data["timesteps"], copy=True)
            else:
                raise ValueError(
                    f"{path} has no simulated-time or timestep axis; "
                    f"available keys: {data.files}"
                )
            simulated_seconds = _validated_timesteps(legacy_timesteps) * scale

        if "capture_timesteps" in data.files:
            timesteps = np.array(data["capture_timesteps"], copy=True)
        elif "timesteps" in data.files:
            candidate = np.array(data["timesteps"], copy=True)
            timesteps = (
                candidate
                if candidate.shape == np.asarray(simulated_seconds).shape
                else None
            )
        else:
            timesteps = None
        successes = np.array(data["capture_successes"], dtype=object, copy=True)
    return capture_learning_curve(
        simulated_seconds,
        successes,
        evaluation_timesteps=timesteps,
    )
