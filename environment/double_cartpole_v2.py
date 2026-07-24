"""Reverse-curriculum reset for the serial double-pole CartPole swing-up.

The reward is dm_control's stock ``two_poles`` smooth reward, unchanged.  Only
the episode reset changes: instead of always starting from the hanging pose,
training episodes start in a band around the upright whose width grows with
training progress, from a narrow near-upright cap up to the full state range.
Early episodes then begin already near the balanced pose, where the stock
reward is dense and the balance is directly learnable; as the band widens the
start reaches down toward the hanging pose, carrying the learned balance onto
progressively longer swing-ups.

At full width the curriculum reset coincides with a uniform draw over the
slider range and both hinge angles.  ``curriculum=False`` disables the schedule
and falls back to ``uniform_start`` (used for evaluation, where the start
distribution must be fixed).  The trainer drives progress by calling
``set_curriculum_fraction`` on the task each step.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from dm_control.rl import control
from dm_control.suite import base as suite_base
from dm_control.suite import cartpole


# Slider travel limit of the cart-pole model (``range="-1.8 1.8"``).
_CART_LIMIT = 1.8


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
