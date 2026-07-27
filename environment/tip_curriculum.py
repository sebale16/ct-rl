"""Shared tip-height/velocity curriculum state for swing-up tasks.

The curriculum is deliberately discrete.  Its first level is the exact
upright configuration at rest, so the first skill is maintaining balance.
Every later level also starts at rest and lowers the tip until the last level
is the exact hanging configuration.

This module owns only the level specification and synchronization protocol.
Each mechanism maps ``(tip_height, incoming_tip_speed, side)`` to its own
generalized coordinates, and trainer-side deterministic probe evaluations
decide when a level has been mastered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


FRACTION_CURRICULUM_ENV_IDS = frozenset(
    {
        "acrobot-swingup-v4.2",
        "acrobot-swingup-v6",
        "cartpole-two_poles-curriculum",
    }
)
PERFORMANCE_CURRICULUM_ENV_IDS = frozenset(
    {
        "acrobot-swingup-v4.3",
        "acrobot-swingup-v6.1",
        "cartpole-two_poles-v2",
    }
)


@dataclass(frozen=True)
class TipCurriculumLevel:
    """One reset level expressed only through tip height and tip speed."""

    tip_height: float
    incoming_tip_speed: float


class TipHeightVelocityCurriculum:
    """Mixin implementing a synchronized, performance-gated reset ladder.

    Hosts call :meth:`_configure_tip_curriculum` from ``__init__`` and use the
    current :class:`TipCurriculumLevel` from ``initialize_episode``.  The mixin
    intentionally has no timestep/fraction setter: a trainer advances it only
    after a deterministic probe demonstrates sustained stabilization.
    """

    curriculum_kind = "performance"

    def _configure_tip_curriculum(
        self,
        *,
        curriculum: bool,
        tip_height_bounds: tuple[float, float],
        descent_tip_heights: Iterable[float],
    ) -> None:
        self.curriculum = bool(curriculum)

        try:
            hanging_height, upright_height = (
                float(value) for value in tip_height_bounds
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "tip_height_bounds must contain hanging and upright heights"
            ) from exc
        if (
            not np.isfinite(hanging_height)
            or not np.isfinite(upright_height)
            or upright_height <= hanging_height
        ):
            raise ValueError(
                "tip_height_bounds must be finite and ordered from hanging "
                "to upright"
            )

        try:
            descent = tuple(float(height) for height in descent_tip_heights)
        except (TypeError, ValueError) as exc:
            raise ValueError("descent_tip_heights must be numeric") from exc
        if not descent:
            raise ValueError("descent_tip_heights must contain at least hanging")
        if not np.all(np.isfinite(descent)):
            raise ValueError("descent_tip_heights must be finite")
        if any(
            height < hanging_height or height >= upright_height
            for height in descent
        ):
            raise ValueError(
                "descent_tip_heights must lie from hanging (inclusive) to "
                "upright (exclusive)"
            )
        if any(
            next_height >= height
            for height, next_height in zip(descent, descent[1:])
        ):
            raise ValueError("descent_tip_heights must be strictly decreasing")
        if not np.isclose(descent[-1], hanging_height, rtol=0.0, atol=1e-12):
            raise ValueError(
                "the last descent_tip_height must be the exact hanging height"
            )

        self._tip_height_bounds = (hanging_height, upright_height)
        self._tip_curriculum_levels = (
            TipCurriculumLevel(upright_height, 0.0),
            *(TipCurriculumLevel(height, 0.0) for height in descent),
        )
        self._curriculum_stage = (
            0 if self.curriculum else len(self._tip_curriculum_levels) - 1
        )

    @property
    def curriculum_stage(self) -> int:
        return int(self._curriculum_stage)

    @property
    def num_curriculum_stages(self) -> int:
        return len(self._tip_curriculum_levels)

    @property
    def curriculum_complete(self) -> bool:
        return self.curriculum_stage == self.num_curriculum_stages - 1

    @property
    def curriculum_level(self) -> TipCurriculumLevel:
        return self._tip_curriculum_levels[self.curriculum_stage]

    @property
    def curriculum_levels(self) -> tuple[TipCurriculumLevel, ...]:
        return self._tip_curriculum_levels

    def set_curriculum_stage(self, stage: int) -> None:
        """Select a discrete level, clipping to the valid range."""

        numeric = float(stage)
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError("curriculum stage must be a finite integer")
        self._curriculum_stage = int(
            np.clip(int(numeric), 0, self.num_curriculum_stages - 1)
        )

    def curriculum_state_dict(self) -> dict[str, Any]:
        """Serializable state needed to resume the reset distribution."""

        return {"stage": self.curriculum_stage}

    def load_curriculum_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict) or "stage" not in state:
            raise ValueError("curriculum state must contain 'stage'")
        self.set_curriculum_stage(state["stage"])

    def _curriculum_side(self) -> float:
        """Sample one of the mirror-symmetric approaches reproducibly."""

        return -1.0 if int(self.random.randint(2)) == 0 else 1.0

    def curriculum_diagnostics(self) -> dict[str, float]:
        level = self.curriculum_level
        hanging_height, upright_height = self._tip_height_bounds
        height_norm = (level.tip_height - hanging_height) / (
            upright_height - hanging_height
        )
        stage_progress = (
            self.curriculum_stage / (self.num_curriculum_stages - 1)
            if self.num_curriculum_stages > 1
            else 1.0
        )
        return {
            "curriculum_stage": float(self.curriculum_stage),
            "curriculum_num_stages": float(self.num_curriculum_stages),
            "curriculum_progress": float(stage_progress),
            "curriculum_start_tip_height": float(level.tip_height),
            "curriculum_start_tip_height_norm": float(height_norm),
            # All current hosts reset a fully extended chain, so normalized
            # gravitational potential is exactly normalized distal-tip height.
            # Starting speed remains explicit rather than being folded into a
            # misleading one-dimensional "energy level".
            "curriculum_start_potential_energy_norm": float(height_norm),
            "curriculum_start_tip_speed": float(level.incoming_tip_speed),
            "curriculum_complete": float(self.curriculum_complete),
        }
