"""Shared physical-time tracking for strict Acrobot capture.

The environment publishes an instantaneous endpoint predicate.  This module
turns that predicate into an episodic residence metric without attributing the
interval before first entry to the captured state: time accrues only between
two consecutive qualifying endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np


STRICT_CAPTURE_INFO_KEY = "acrobot_strict_capture"
STRICT_CAPTURE_DURATION_SECONDS = 1.0
STRICT_CAPTURE_ENV_ID = "acrobot-swingup-v4.1"


@dataclass(frozen=True)
class SustainedCaptureSpec:
    """Definition of the endpoint signal and required continuous residence."""

    info_key: str = STRICT_CAPTURE_INFO_KEY
    duration_seconds: float = STRICT_CAPTURE_DURATION_SECONDS
    duration_atol: float = 1e-6

    def __post_init__(self) -> None:
        if not self.info_key:
            raise ValueError("capture info_key must be non-empty")
        if (
            not np.isfinite(self.duration_seconds)
            or self.duration_seconds <= 0.0
        ):
            raise ValueError("capture duration_seconds must be finite and > 0")
        if not np.isfinite(self.duration_atol) or self.duration_atol < 0.0:
            raise ValueError("capture duration_atol must be finite and >= 0")


@dataclass(frozen=True)
class CaptureEpisodeResult:
    """Strict-capture outcome for one completed episode."""

    success: bool
    max_duration_seconds: float


class SustainedCaptureTracker:
    """Track consecutive qualifying endpoint duration for parallel env slots."""

    def __init__(
        self,
        n_envs: int,
        spec: SustainedCaptureSpec,
        initial_infos: list[Mapping[str, Any]],
    ) -> None:
        if int(n_envs) <= 0:
            raise ValueError("n_envs must be positive")
        if len(initial_infos) != int(n_envs):
            raise ValueError(
                f"expected {n_envs} initial infos, got {len(initial_infos)}"
            )
        self.n_envs = int(n_envs)
        self.spec = spec
        self._previous_inside = np.zeros(self.n_envs, dtype=bool)
        self._run_seconds = np.zeros(self.n_envs, dtype=np.float64)
        self._max_seconds = np.zeros(self.n_envs, dtype=np.float64)
        for i, info in enumerate(initial_infos):
            self.reset_slot(i, info)

    def _inside(self, info: Mapping[str, Any], *, required: bool) -> bool:
        if self.spec.info_key not in info:
            if required:
                raise KeyError(
                    f"capture evaluation requires info[{self.spec.info_key!r}]"
                )
            return False
        value = float(info[self.spec.info_key])
        if not np.isfinite(value):
            raise ValueError(
                f"info[{self.spec.info_key!r}] must be finite, got {value}"
            )
        return bool(value)

    @staticmethod
    def _duration(info: Mapping[str, Any]) -> float:
        if "dt_used" not in info:
            raise KeyError("capture evaluation requires info['dt_used']")
        duration = float(info["dt_used"])
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError(
                f"capture evaluation requires finite dt_used > 0, got {duration}"
            )
        return duration

    def reset_slot(self, slot: int, initial_info: Mapping[str, Any]) -> None:
        """Begin a new episode from its observed reset endpoint."""

        self._previous_inside[slot] = self._inside(initial_info, required=True)
        self._run_seconds[slot] = 0.0
        self._max_seconds[slot] = 0.0

    def update_slot(
        self,
        slot: int,
        info: Mapping[str, Any],
        *,
        done: bool,
        reset_info: Optional[Mapping[str, Any]] = None,
    ) -> Optional[CaptureEpisodeResult]:
        """Consume one transition endpoint and optionally finish the episode."""

        inside = self._inside(info, required=False)
        if inside and self._previous_inside[slot]:
            self._run_seconds[slot] += self._duration(info)
        elif inside:
            # First observed entry: do not claim the preceding interval.
            self._run_seconds[slot] = 0.0
        else:
            self._run_seconds[slot] = 0.0

        self._max_seconds[slot] = max(
            self._max_seconds[slot], self._run_seconds[slot]
        )
        self._previous_inside[slot] = inside

        if not done:
            return None

        maximum = float(self._max_seconds[slot])
        result = CaptureEpisodeResult(
            success=(
                maximum + self.spec.duration_atol
                >= self.spec.duration_seconds
            ),
            max_duration_seconds=maximum,
        )
        if reset_info is None:
            self._previous_inside[slot] = False
            self._run_seconds[slot] = 0.0
            self._max_seconds[slot] = 0.0
        else:
            self.reset_slot(slot, reset_info)
        return result


def capture_selection_rank(
    successes: list[bool], max_durations: list[float]
) -> tuple[float, float]:
    """Lexicographic checkpoint rank: success rate, then residence duration."""

    if not successes or len(successes) != len(max_durations):
        raise ValueError(
            "capture rank requires equally sized, non-empty episode results"
        )
    durations = np.asarray(max_durations, dtype=np.float64)
    if not np.all(np.isfinite(durations)) or np.any(durations < 0.0):
        raise ValueError("capture durations must be finite and non-negative")
    return (
        float(np.mean(np.asarray(successes, dtype=np.float64))),
        float(np.mean(durations)),
    )


def strict_capture_spec_for(
    *, algorithm: str, env_id: str
) -> Optional[SustainedCaptureSpec]:
    """Return the shared best-checkpoint rule for the requested benchmark.

    The rollout code paths use different callback APIs, so keeping this scope
    decision here prevents PPO and CT-SAC from silently selecting checkpoints
    under different definitions.
    """

    if env_id != STRICT_CAPTURE_ENV_ID:
        return None
    if algorithm.lower() not in {"ppo", "ct_sac"}:
        return None
    return SustainedCaptureSpec()
