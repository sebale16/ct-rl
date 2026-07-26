"""Performance-gated progression for discrete curriculum stages."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def _positive_integer(name: str, value: int) -> int:
    """Return a validated positive integer without silently truncating."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric < 1.0:
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def _nonnegative_integer(name: str, value: int) -> int:
    """Return a validated integer greater than or equal to zero."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric < 0.0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(numeric)


def _success_rate(name: str, value: float) -> float:
    """Return a finite success rate on the closed unit interval."""

    try:
        rate = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and in [0, 1]") from exc
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return rate


class MasteryCurriculum:
    """Advance a discrete curriculum only after probe evaluations demonstrate mastery.

    A probe passes when its success rate is at least ``success_threshold``.
    ``consecutive_evals`` passing probes are required to advance one stage.
    Failed probes clear the consecutive-pass evidence, and advancing clears it
    again so one observation can never skip multiple stages.
    """

    def __init__(
        self,
        num_stages: int,
        success_threshold: float = 0.8,
        consecutive_evals: int = 1,
    ) -> None:
        self.num_stages = _positive_integer("num_stages", num_stages)
        self.success_threshold = _success_rate(
            "success_threshold", success_threshold
        )
        self.consecutive_evals = _positive_integer(
            "consecutive_evals", consecutive_evals
        )
        self._stage = 0
        self._consecutive_passes = 0

    @property
    def stage(self) -> int:
        return self._stage

    @property
    def consecutive_passes(self) -> int:
        return self._consecutive_passes

    @property
    def at_final_stage(self) -> bool:
        return self.stage == self.num_stages - 1

    def observe(self, success_rate: float) -> bool:
        """Consume one probe result and return whether one stage was advanced."""

        rate = _success_rate("probe success rate", success_rate)

        if self.at_final_stage:
            # Evidence has no meaning after the final reset distribution is active.
            self._consecutive_passes = 0
            return False

        if rate >= self.success_threshold:
            self._consecutive_passes += 1
        else:
            self._consecutive_passes = 0

        if self._consecutive_passes < self.consecutive_evals:
            return False

        self._stage += 1
        self._consecutive_passes = 0
        return True

    def state_dict(self) -> dict[str, Any]:
        """Return all configuration and dynamic state needed for exact resume."""

        return {
            "num_stages": self.num_stages,
            "success_threshold": self.success_threshold,
            "consecutive_evals": self.consecutive_evals,
            "stage": self.stage,
            "consecutive_passes": self.consecutive_passes,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore dynamic state, rejecting incompatible or malformed checkpoints."""

        if not isinstance(state, Mapping):
            raise ValueError("mastery curriculum state must be a mapping")
        required = {
            "num_stages",
            "success_threshold",
            "consecutive_evals",
            "stage",
            "consecutive_passes",
        }
        missing = required.difference(state)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"mastery curriculum state is missing: {names}")

        saved_num_stages = _positive_integer("num_stages", state["num_stages"])
        saved_threshold = _success_rate(
            "saved success_threshold", state["success_threshold"]
        )
        saved_consecutive_evals = _positive_integer(
            "consecutive_evals", state["consecutive_evals"]
        )
        if (
            saved_num_stages != self.num_stages
            or saved_threshold != self.success_threshold
            or saved_consecutive_evals != self.consecutive_evals
        ):
            raise ValueError(
                "mastery curriculum state configuration does not match this "
                "controller"
            )

        stage = _nonnegative_integer("stage", state["stage"])
        if stage >= self.num_stages:
            raise ValueError("saved stage is outside the configured stage range")
        consecutive_passes = _nonnegative_integer(
            "consecutive_passes", state["consecutive_passes"]
        )
        if consecutive_passes >= self.consecutive_evals:
            raise ValueError(
                "saved consecutive_passes must be below consecutive_evals"
            )
        if stage == self.num_stages - 1 and consecutive_passes != 0:
            raise ValueError("the final stage cannot carry pending pass evidence")

        # Mutate only after every field has been validated.
        self._stage = stage
        self._consecutive_passes = consecutive_passes
