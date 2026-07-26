"""Curriculum resets for the serial double-pole CartPole swing-up.

Both local tasks keep dm_control's stock ``two_poles`` smooth reward unchanged:

* ``cartpole-two_poles-curriculum`` is the historical fraction-scheduled
  angle-band curriculum.
* ``cartpole-two_poles-v2`` is the performance-gated tip-height/velocity
  curriculum.  It begins with a braking task near upright, then lowers the
  distal tip through zero-velocity starts until reaching exact hanging.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np

from dm_control.rl import control
from dm_control.suite import base as suite_base
from dm_control.suite import cartpole

from .tip_curriculum import TipHeightVelocityCurriculum


# Slider travel limit of the cart-pole model (``range="-1.8 1.8"``).
_CART_LIMIT = 1.8

# The cart pivot is at z=1 and both links have unit length.  With the chain
# extended, the distal tip therefore spans [-1, 3].
TIP_HEIGHT_BOUNDS = (-1.0, 3.0)
STABILIZATION_POINT = (0.0, 3.0)
DEFAULT_BRAKE_TIP_HEIGHT = 2.8
DEFAULT_BRAKE_TIP_SPEED = 1.0
DEFAULT_DESCENT_TIP_HEIGHTS = (2.5, 2.0, 1.0, 0.0, -1.0)
STRICT_CAPTURE_DISTANCE = 0.2
STRICT_CAPTURE_SPEED = 0.2


class CartpoleTwoPolesCurriculum(cartpole.Balance):
    """Stock two-pole swing-up reward with a reverse-curriculum reset.

    Mechanism, observations, and reward are identical to dm_control's
    ``two_poles`` task (``cartpole.Balance`` with ``swing_up=True,
    sparse=False``).  This subclass overrides only the reset:

    * ``curriculum=True`` (default): sample a band around the upright pose
      (cart centered, both poles up) whose half-width grows with
      ``curriculum_fraction`` from ``curriculum_min_spread`` to ``pi`` on the
      hinges and from 0 to the slider limit on the cart.  At fraction 1 the
      draw is uniform over the full range.
    * ``curriculum=False, uniform_start=True``: uniform over the full range,
      independent of progress (the fixed evaluation reset).
    * ``curriculum=False, uniform_start=False``: the stock near-hanging
      swing-up reset, for the canonical swing-up-from-down probe.
    """

    def __init__(
        self,
        *,
        random=None,
        curriculum: bool = True,
        curriculum_min_spread: float = 0.5,
        uniform_start: bool = True,
        velocity_noise: float = 0.01,
    ) -> None:
        super().__init__(swing_up=True, sparse=False, random=random)
        self.curriculum = bool(curriculum)
        self.curriculum_min_spread = float(curriculum_min_spread)
        if not np.isfinite(self.curriculum_min_spread) or not (
            0.0 < self.curriculum_min_spread <= np.pi
        ):
            raise ValueError(
                "curriculum_min_spread must be finite and in (0, pi]"
            )
        self.uniform_start = bool(uniform_start)
        self.velocity_noise = float(velocity_noise)
        if not np.isfinite(self.velocity_noise) or self.velocity_noise < 0.0:
            raise ValueError("velocity_noise must be finite and non-negative")
        # Curriculum progress in [0, 1]; 0 = tightest near-upright band.
        self._curriculum_fraction = 0.0

    def reseed(self, seed: int) -> None:
        """Reset the RNG used for reset-state sampling (repeatable evals)."""
        self._random = np.random.RandomState(int(seed) % (2**32))

    def set_curriculum_fraction(self, fraction: float) -> None:
        """Set curriculum progress in [0, 1] (0 = tightest near-upright band)."""
        frac = float(fraction)
        if not np.isfinite(frac):
            raise ValueError("curriculum fraction must be finite")
        self._curriculum_fraction = float(np.clip(frac, 0.0, 1.0))

    @property
    def curriculum_fraction(self) -> float:
        return self._curriculum_fraction

    def _set_velocity_noise(self, physics) -> None:
        physics.named.data.qvel[:] = self.random.uniform(
            -self.velocity_noise, self.velocity_noise, physics.model.nv
        )

    def _initialize_uniform_episode(self, physics) -> None:
        """Uniform over the slider range and both hinge angles."""
        physics.named.data.qpos["slider"] = self.random.uniform(
            -_CART_LIMIT, _CART_LIMIT
        )
        physics.named.data.qpos[["hinge_1", "hinge_2"]] = self.random.uniform(
            -np.pi, np.pi, 2
        )
        self._set_velocity_noise(physics)
        suite_base.Task.initialize_episode(self, physics)

    def _initialize_curriculum_episode(self, physics) -> None:
        """Band around the upright whose width grows with progress.

        The hinge half-width grows from ``curriculum_min_spread`` to pi and the
        cart half-width from 0 to the slider limit, so early episodes start
        near the balanced pose and later episodes span the full range,
        including the hanging pose.  At fraction 1 the draw is uniform, matching
        ``_initialize_uniform_episode``.
        """
        frac = self._curriculum_fraction
        angle_spread = self.curriculum_min_spread + frac * (
            np.pi - self.curriculum_min_spread
        )
        offsets = self.random.uniform(-angle_spread, angle_spread, 2)
        # Wrap into (-pi, pi]; identity for spread <= pi, kept for safety.
        hinges = np.arctan2(np.sin(offsets), np.cos(offsets))
        cart = self.random.uniform(-frac * _CART_LIMIT, frac * _CART_LIMIT)

        physics.named.data.qpos["slider"] = cart
        physics.named.data.qpos[["hinge_1", "hinge_2"]] = hinges
        self._set_velocity_noise(physics)
        suite_base.Task.initialize_episode(self, physics)

    def initialize_episode(self, physics) -> None:
        if self.curriculum:
            self._initialize_curriculum_episode(physics)
        elif self.uniform_start:
            self._initialize_uniform_episode(physics)
        else:
            # Stock near-hanging swing-up reset (cartpole.Balance).
            super().initialize_episode(physics)


class CartpoleTwoPolesV2(TipHeightVelocityCurriculum, cartpole.Balance):
    """Stock two-pole reward with a distal-tip mastery curriculum.

    The reset keeps the two links fully extended (relative second hinge zero),
    leaving only one mirrored chain angle.  For requested distal-tip height
    ``h`` and incoming Cartesian speed ``v``:

    ``q1 = side * arccos((h - 1) / 2)``, ``q2 = 0``,
    ``q1_dot = -side * v / 2``, and ``q2_dot = 0``.

    Thus the distal-tip height and speed are exactly ``h`` and ``v``, and the
    nonzero braking velocity follows the short arc toward the upright target.
    ``get_reward`` is deliberately inherited unchanged from
    :class:`dm_control.suite.cartpole.Balance`.
    """

    def __init__(
        self,
        *,
        random=None,
        curriculum: bool = True,
        brake_tip_height: float = DEFAULT_BRAKE_TIP_HEIGHT,
        brake_tip_speed: float = DEFAULT_BRAKE_TIP_SPEED,
        descent_tip_heights: Iterable[float] = DEFAULT_DESCENT_TIP_HEIGHTS,
    ) -> None:
        super().__init__(swing_up=True, sparse=False, random=random)
        self._configure_tip_curriculum(
            curriculum=curriculum,
            tip_height_bounds=TIP_HEIGHT_BOUNDS,
            brake_tip_height=brake_tip_height,
            brake_tip_speed=brake_tip_speed,
            descent_tip_heights=descent_tip_heights,
        )
        if not self.curriculum:
            self.set_curriculum_stage(self.num_curriculum_stages - 1)

    def reseed(self, seed: int) -> None:
        """Reset the RNG used for mirror-symmetric reset sampling."""

        self._random = np.random.RandomState(int(seed) % (2**32))

    @staticmethod
    def _set_state_from_tip_level(
        physics, *, tip_height: float, incoming_tip_speed: float, side: float
    ) -> None:
        """Map one distal-tip level to the extended chain coordinates."""

        cosine = float(np.clip((float(tip_height) - 1.0) / 2.0, -1.0, 1.0))
        theta = float(side) * float(np.arccos(cosine))
        theta_dot = -float(side) * float(incoming_tip_speed) / 2.0

        physics.named.data.qpos["slider"] = 0.0
        physics.named.data.qpos[["hinge_1", "hinge_2"]] = (theta, 0.0)
        physics.named.data.qvel["slider"] = 0.0
        physics.named.data.qvel[["hinge_1", "hinge_2"]] = (
            theta_dot,
            0.0,
        )

    def initialize_episode(self, physics) -> None:
        if self.curriculum:
            level = self.curriculum_level
            # Make the terminal distribution one exact generalized state, not
            # the equivalent +/-pi representations.
            side = (
                1.0
                if np.isclose(
                    level.tip_height,
                    TIP_HEIGHT_BOUNDS[0],
                    rtol=0.0,
                    atol=1e-12,
                )
                else self._curriculum_side()
            )
            self._set_state_from_tip_level(
                physics,
                tip_height=level.tip_height,
                incoming_tip_speed=level.incoming_tip_speed,
                side=side,
            )
        else:
            physics.named.data.qpos["slider"] = 0.0
            physics.named.data.qpos[["hinge_1", "hinge_2"]] = (np.pi, 0.0)
            physics.named.data.qvel[:] = 0.0
        suite_base.Task.initialize_episode(self, physics)

    @staticmethod
    def _tip_kinematics(physics) -> tuple[float, float, float, float]:
        """Return distal ``(x, z, vx, vz)`` with hinge 2 treated as relative."""

        cart_x, theta_1, theta_2_relative = (
            float(value) for value in np.asarray(physics.data.qpos)
        )
        cart_velocity, theta_1_dot, theta_2_relative_dot = (
            float(value) for value in np.asarray(physics.data.qvel)
        )
        theta_2 = theta_1 + theta_2_relative
        theta_2_dot = theta_1_dot + theta_2_relative_dot

        sin_1, cos_1 = float(np.sin(theta_1)), float(np.cos(theta_1))
        sin_2, cos_2 = float(np.sin(theta_2)), float(np.cos(theta_2))
        tip_x = cart_x + sin_1 + sin_2
        tip_z = 1.0 + cos_1 + cos_2
        tip_vx = (
            cart_velocity + cos_1 * theta_1_dot + cos_2 * theta_2_dot
        )
        tip_vz = -sin_1 * theta_1_dot - sin_2 * theta_2_dot
        return tip_x, tip_z, tip_vx, tip_vz

    def curriculum_terms(self, physics) -> Dict[str, float]:
        """Distal-tip stabilization and current-stage diagnostics."""

        tip_x, tip_height, tip_vx, tip_vz = self._tip_kinematics(physics)
        target_x, target_z = STABILIZATION_POINT
        tip_distance = float(
            np.hypot(tip_x - target_x, tip_height - target_z)
        )
        tip_speed = float(np.hypot(tip_vx, tip_vz))
        strict_capture = float(
            tip_distance < STRICT_CAPTURE_DISTANCE
            and tip_speed < STRICT_CAPTURE_SPEED
        )
        return {
            "tip_height": float(tip_height),
            "tip_distance": tip_distance,
            "tip_speed": tip_speed,
            "strict_capture": strict_capture,
            "success": strict_capture,
            "curriculum_enabled": float(self.curriculum),
            **self.curriculum_diagnostics(),
        }

    diagnostic_terms = curriculum_terms


def two_poles_curriculum(
    *,
    time_limit: float = 10.0,
    random=None,
    environment_kwargs: Optional[Dict[str, Any]] = None,
    curriculum: bool = True,
    curriculum_min_spread: float = 0.5,
    uniform_start: bool = True,
    velocity_noise: float = 0.01,
):
    """Construct the ``cartpole-two_poles-curriculum`` dm_control environment.

    The reward is the stock two-pole swing-up reward; only the reset carries the
    reverse curriculum.  ``curriculum=False`` restores a fixed start for
    evaluation (uniform when ``uniform_start`` else near-hanging).
    """
    physics = cartpole.Physics.from_xml_string(
        *cartpole.get_model_and_assets(num_poles=2)
    )
    task = CartpoleTwoPolesCurriculum(
        random=random,
        curriculum=curriculum,
        curriculum_min_spread=curriculum_min_spread,
        uniform_start=uniform_start,
        velocity_noise=velocity_noise,
    )
    return control.Environment(
        physics,
        task,
        time_limit=float(time_limit),
        **dict(environment_kwargs or {}),
    )


def two_poles_v2(
    *,
    time_limit: float = 10.0,
    random=None,
    environment_kwargs: Optional[Dict[str, Any]] = None,
    curriculum: bool = True,
    brake_tip_height: float = DEFAULT_BRAKE_TIP_HEIGHT,
    brake_tip_speed: float = DEFAULT_BRAKE_TIP_SPEED,
    descent_tip_heights: Iterable[float] = DEFAULT_DESCENT_TIP_HEIGHTS,
):
    """Construct ``cartpole-two_poles-v2`` with the stock smooth reward."""

    physics = cartpole.Physics.from_xml_string(
        *cartpole.get_model_and_assets(num_poles=2)
    )
    task = CartpoleTwoPolesV2(
        random=random,
        curriculum=curriculum,
        brake_tip_height=brake_tip_height,
        brake_tip_speed=brake_tip_speed,
        descent_tip_heights=descent_tip_heights,
    )
    return control.Environment(
        physics,
        task,
        time_limit=float(time_limit),
        **dict(environment_kwargs or {}),
    )
