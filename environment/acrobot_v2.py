"""Reproducible, genuinely from-down Acrobot swing-up tasks.

The stock dm_control Acrobot task initializes both joints uniformly on
``[-pi, pi]`` and uses a narrow Gaussian target-distance reward.  That makes
evaluation dominated by reset luck and leaves almost no reward signal near the
hanging configuration.  This local variant keeps the same MuJoCo mechanism and
observations while changing only the task definition.  Both local versions:

* episodes start close to the fully hanging pose;
* explicit reseeding makes fixed evaluation starts repeatable.

``swingup-v2`` combines tip-distance progress with the precise stock reward.
That historical definition is preserved verbatim for checkpoint provenance.
``swingup-v3`` replaces its folded-link reward ridge with smooth progress equal
to elbow extension times mean absolute-link uprightness, while retaining a
small precise-target term near the exact goal.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from dm_control.rl import control
from dm_control.suite import acrobot
from dm_control.suite import base as suite_base


class BalanceV2(acrobot.Balance):
    """Acrobot swing-up with a near-down reset and bounded dense reward."""

    _MAX_TARGET_DISTANCE = 4.0

    def __init__(
        self,
        *,
        random=None,
        angle_noise: float = 0.05,
        velocity_noise: float = 0.01,
        precision_weight: float = 0.2,
    ) -> None:
        super().__init__(sparse=False, random=random)
        self.angle_noise = self._finite_nonnegative("angle_noise", angle_noise)
        self.velocity_noise = self._finite_nonnegative(
            "velocity_noise", velocity_noise
        )
        self.precision_weight = float(precision_weight)
        if not np.isfinite(self.precision_weight) or not (
            0.0 <= self.precision_weight <= 1.0
        ):
            raise ValueError("precision_weight must be finite and in [0, 1]")

    @staticmethod
    def _finite_nonnegative(name: str, value: float) -> float:
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return value

    def reseed(self, seed: int) -> None:
        """Reset the task RNG used for reset-state sampling."""
        self._random = np.random.RandomState(int(seed) % (2**32))

    def initialize_episode(self, physics) -> None:
        """Start near ``[shoulder=pi, elbow=0]`` with small velocity noise."""
        qpos_noise = self.random.uniform(-self.angle_noise, self.angle_noise, 2)
        qvel_noise = self.random.uniform(
            -self.velocity_noise, self.velocity_noise, physics.model.nv
        )
        physics.named.data.qpos[["shoulder", "elbow"]] = (
            np.asarray([np.pi, 0.0]) + qpos_noise
        )
        physics.named.data.qvel[:] = qvel_noise

        # Calling acrobot.Balance.initialize_episode would overwrite the pose
        # with the stock uniform [-pi, pi] reset.  Delegate directly to the task
        # base class for visualization bookkeeping instead.
        suite_base.Task.initialize_episode(self, physics)

    def reward_terms(self, physics) -> Dict[str, float]:
        """Return the bounded reward and its reward-independent diagnostics."""
        distance = float(physics.to_target())
        precise = float(super()._get_reward(physics, sparse=False))
        progress = float(
            np.clip(1.0 - distance / self._MAX_TARGET_DISTANCE, 0.0, 1.0)
        )
        reward = (
            (1.0 - self.precision_weight) * progress
            + self.precision_weight * precise
        )
        target_radius = float(physics.named.model.site_size["target", 0])
        tip_height = float(physics.named.data.site_xpos["tip", "z"])
        return {
            "reward": float(np.clip(reward, 0.0, 1.0)),
            "tip_distance": distance,
            "tip_height": tip_height,
            "progress": progress,
            "precision": precise,
            "success": float(distance <= target_radius),
        }

    def get_reward(self, physics) -> float:
        return self.reward_terms(physics)["reward"]


def swingup_v2(
    *,
    time_limit: float = 10.0,
    random=None,
    environment_kwargs: Optional[Dict[str, Any]] = None,
    angle_noise: float = 0.05,
    velocity_noise: float = 0.01,
    precision_weight: float = 0.2,
):
    """Construct the local ``acrobot-swingup-v2`` dm_control environment."""
    physics = acrobot.Physics.from_xml_string(*acrobot.get_model_and_assets())
    task = BalanceV2(
        random=random,
        angle_noise=angle_noise,
        velocity_noise=velocity_noise,
        precision_weight=precision_weight,
    )
    return control.Environment(
        physics,
        task,
        time_limit=float(time_limit),
        **dict(environment_kwargs or {}),
    )


class BalanceV3(BalanceV2):
    """From-down swing-up with smooth, fold-resistant dense progress."""

    _GYM_TARGET_HEIGHT = 3.0

    def reward_terms(self, physics) -> Dict[str, float]:
        """Return anti-fold reward terms and reward-independent diagnostics."""
        distance = float(physics.to_target())
        precise = float(acrobot.Balance._get_reward(self, physics, sparse=False))

        vertical = np.asarray(physics.vertical(), dtype=np.float64).reshape(-1)
        if vertical.shape != (2,):
            raise ValueError(
                "Acrobot vertical orientation must have shape (2,), got "
                f"{vertical.shape}"
            )
        upright = np.clip((vertical + 1.0) / 2.0, 0.0, 1.0)

        elbow = float(np.asarray(physics.named.data.qpos["elbow"]).item())
        extension = float(np.clip((1.0 + np.cos(elbow)) / 2.0, 0.0, 1.0))
        progress = float(extension * 0.5 * (upright[0] + upright[1]))
        reward = (
            (1.0 - self.precision_weight) * progress
            + self.precision_weight * precise
        )

        target_radius = float(physics.named.model.site_size["target", 0])
        tip_height = float(physics.named.data.site_xpos["tip", "z"])
        exact_success = float(distance <= target_radius)
        return {
            "reward": float(np.clip(reward, 0.0, 1.0)),
            "tip_distance": distance,
            "tip_height": tip_height,
            "progress": progress,
            "precision": precise,
            "upper_uprightness": float(upright[0]),
            "lower_uprightness": float(upright[1]),
            "extension": extension,
            "gym_height_success": float(tip_height > self._GYM_TARGET_HEIGHT),
            "exact_success": exact_success,
            # Preserve the established diagnostics contract: unqualified
            # ``success`` continues to mean the precise target-site hit.
            "success": exact_success,
        }


def swingup_v3(
    *,
    time_limit: float = 10.0,
    random=None,
    environment_kwargs: Optional[Dict[str, Any]] = None,
    angle_noise: float = 0.05,
    velocity_noise: float = 0.01,
    precision_weight: float = 0.2,
):
    """Construct the anti-fold ``acrobot-swingup-v3`` environment."""
    physics = acrobot.Physics.from_xml_string(*acrobot.get_model_and_assets())
    task = BalanceV3(
        random=random,
        angle_noise=angle_noise,
        velocity_noise=velocity_noise,
        precision_weight=precision_weight,
    )
    return control.Environment(
        physics,
        task,
        time_limit=float(time_limit),
        **dict(environment_kwargs or {}),
    )
