"""Shared tip-height/velocity curriculum state for swing-up tasks.

The curriculum is deliberately discrete.  Its first level is a small
near-upright displacement at rest, so the first skill is recovering into and
maintaining balance.  Every later level also starts at rest and lowers the tip
until the last level hangs.

A level fixes the distal tip, not generally the pose that reaches it: from
stage 2 onward the chain is folded by a random relative angle at every reset,
so the same tip height is presented through a family of shapes.  Stage 1 is
the deliberate exception—an unfolded chain with only a tiny mirrored tilt.
Both hosts are two equal links on a pivot, so this module owns that mapping as
well as the level specification and synchronization protocol.  Trainer-side
deterministic probe evaluations decide when a level has been mastered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

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
# Stage 1 is visually vertical but not an exact unstable equilibrium.  Its
# fixed-magnitude tilt is mirrored left/right; expressing the level through
# height preserves the mechanism-neutral curriculum representation.
INITIAL_TIP_ANGLE_OFFSET_RAD = 1e-3
INITIAL_TIP_HEIGHT_NORM = 0.5 * (
    1.0 + float(np.cos(INITIAL_TIP_ANGLE_OFFSET_RAD))
)
# Half-width of the uniform relative-angle draw between the two links.  Zero
# restores the extended chain, one pose per level and side.
DEFAULT_ELBOW_SPREAD = float(np.pi / 6.0)


@dataclass(frozen=True)
class TipCurriculumLevel:
    """One reset level: a distal-tip height and speed, plus its fold range.

    ``elbow_spread`` is the half-width of the relative-angle draw admitted at
    this height.  It is the configured ceiling except near the stabilization
    point.  Stage 1 deliberately disables the fold so its only disturbance is
    the tiny mirrored whole-chain tilt.
    """

    tip_height: float
    incoming_tip_speed: float
    elbow_spread: float = 0.0


@dataclass(frozen=True)
class CurriculumPose:
    """One sampled reset in the shared two-link coordinates.

    ``first_link_angle`` is measured from vertical upright and
    ``elbow_angle`` is the second link relative to the first.
    """

    first_link_angle: float
    elbow_angle: float
    first_link_rate: float
    elbow_rate: float


class TipHeightVelocityCurriculum:
    """Mixin implementing a synchronized, performance-gated reset ladder.

    Hosts call :meth:`_configure_tip_curriculum` from ``__init__`` and draw one
    reset pose with :meth:`sample_curriculum_pose` from ``initialize_episode``.
    The mixin intentionally has no timestep/fraction setter: a trainer advances
    it only after a deterministic probe demonstrates sustained stabilization.
    """

    curriculum_kind = "performance"

    def _configure_tip_curriculum(
        self,
        *,
        curriculum: bool,
        tip_height_bounds: tuple[float, float],
        descent_tip_heights: Iterable[float],
        elbow_spread: float = DEFAULT_ELBOW_SPREAD,
        min_start_distance: float = 0.0,
    ) -> None:
        self.curriculum = bool(curriculum)

        self._curriculum_elbow_spread = float(elbow_spread)
        if not np.isfinite(self._curriculum_elbow_spread) or not (
            0.0 <= self._curriculum_elbow_spread < np.pi
        ):
            raise ValueError("elbow_spread must be finite and in [0, pi)")

        self._min_start_distance = float(min_start_distance)
        if (
            not np.isfinite(self._min_start_distance)
            or self._min_start_distance < 0.0
        ):
            raise ValueError(
                "min_start_distance must be finite and non-negative"
            )

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

        initial_height = hanging_height + INITIAL_TIP_HEIGHT_NORM * (
            upright_height - hanging_height
        )
        if descent[0] >= initial_height:
            raise ValueError(
                "the first descent_tip_height must be below the initial "
                "near-upright height"
            )

        self._tip_height_bounds = (hanging_height, upright_height)
        self._tip_curriculum_levels = tuple(
            TipCurriculumLevel(height, 0.0, self._level_elbow_spread(height))
            for height in (initial_height, *descent)
        )
        self._curriculum_stage = (
            0 if self.curriculum else len(self._tip_curriculum_levels) - 1
        )

    def _level_elbow_spread(self, tip_height: float) -> float:
        """Largest fold allowed near the goal, or the configured full spread.

        Folding shortens the arm, so near the stabilization point it moves the
        tip toward the goal rather than around it.  Writing ``c`` for the
        height above the pivot, the folded tip sits at distance
        ``sqrt(reach^2 cos^2(e/2) - c^2 + (reach - c)^2)`` from the goal,
        which gives the fold at which that distance first reaches the requested
        minimum.  If even the unfolded reference lies inside that minimum, as
        the intentionally tiny stage-1 tilt does, the fold is disabled rather
        than adding a larger second disturbance.  The five-second terminal
        mastery hold ensures that this unstable start still requires control.
        """

        hanging_height, upright_height = self._tip_height_bounds
        pivot = 0.5 * (hanging_height + upright_height)
        reach = 0.5 * (upright_height - hanging_height)
        offset = float(tip_height) - pivot
        cosine_squared = (
            offset**2 + self._min_start_distance**2 - (reach - offset) ** 2
        ) / reach**2
        if cosine_squared > 1.0:
            return 0.0
        if cosine_squared <= 0.0:
            return self.curriculum_elbow_spread
        return min(
            self.curriculum_elbow_spread,
            2.0 * float(np.arccos(np.sqrt(cosine_squared))),
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
    def curriculum_elbow_spread(self) -> float:
        return float(self._curriculum_elbow_spread)

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

    def _curriculum_elbow(self, level: TipCurriculumLevel) -> float:
        """Sample the relative angle between the two links at ``level``."""

        spread = float(level.elbow_spread)
        if spread <= 0.0:
            return 0.0
        return float(self.random.uniform(-spread, spread))

    def sample_curriculum_pose(
        self, level: Optional[TipCurriculumLevel] = None
    ) -> CurriculumPose:
        """Draw one reset pose for ``level`` (default: the current level).

        The two links are equal, so folding them by a relative angle ``e``
        leaves the tip on the bisector at distance ``reach cos(e/2)`` from the
        pivot: the first link carries the fold's half-angle and the arm no
        longer spans the full height range.  Requiring the requested tip height
        gives

            theta = arccos(offset / (reach cos(e/2))) - e/2,

        with ``offset`` the height above the pivot.  Deep folds cannot reach the
        vertical extremes, and there the clip returns the closest pose in that
        fold — the chain splayed symmetrically about the vertical, its tip just
        inside the extreme.  Every other level keeps its tip height exactly.

        Rotating the folded arm rigidly about the pivot gives the requested
        incoming tip speed at ``reach cos(e/2)`` rather than at full extension.
        Mirroring negates both angles, which reflects the pose and leaves the
        tip height fixed.
        """

        level = self.curriculum_level if level is None else level
        hanging_height, upright_height = self._tip_height_bounds
        pivot = 0.5 * (hanging_height + upright_height)
        reach = 0.5 * (upright_height - hanging_height)

        elbow = self._curriculum_elbow(level)
        fold = float(np.cos(0.5 * elbow))
        radius = reach * fold
        cosine = float(
            np.clip((float(level.tip_height) - pivot) / radius, -1.0, 1.0)
        )
        # An unfolded chain at a vertical extreme is one exact generalized
        # state, so mirroring it would only consume the RNG and turn the
        # canonical hanging pose into its equivalent negative.
        symmetric = elbow == 0.0 and abs(cosine) == 1.0
        side = 1.0 if symmetric else self._curriculum_side()
        first_link = float(np.arccos(cosine)) - 0.5 * elbow
        first_link_rate = -float(level.incoming_tip_speed) / radius
        return CurriculumPose(
            first_link_angle=side * first_link,
            elbow_angle=side * elbow,
            first_link_rate=side * first_link_rate,
            elbow_rate=0.0,
        )

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
            # Normalized gravitational potential of the extended chain, which
            # is the level's reference pose: a fold raises the inner link above
            # this value at the same tip height, by an amount the spread below
            # bounds.  Starting speed remains explicit rather than being folded
            # into a misleading one-dimensional "energy level".
            "curriculum_start_potential_energy_norm": float(height_norm),
            "curriculum_start_tip_speed": float(level.incoming_tip_speed),
            "curriculum_start_elbow_spread": float(level.elbow_spread),
            "curriculum_complete": float(self.curriculum_complete),
        }
