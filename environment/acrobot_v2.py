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
``swingup-v4`` replaces pose-purity shaping with energy regulation: the dense
term pays for holding total mechanical energy near the upright-rest level
(rewarding the elbow pumping that v3 penalized), and sustained income exists
only in the velocity-gated precise-hold term at the exact goal.
``swingup-v5`` is an unshaped height-occupancy control arm: reward 1 while
the tip exceeds the Gym height criterion (tip one link length above the
pivot), 0 otherwise, over a fixed-length episode — the return is the time
spent above the height.
``swingup-v6`` drops shaping altogether for the AR-EAPO quadratic cost
(Choe et al., 2024): reward is minus a weighted square of the state error to
the upright and of the command, so the position term slopes monotonically from
hanging to the goal.  It comes as a matched pair over that one reward, which
isolates the reset from the reward exactly as v4.1/v4.2 do:
``swingup-v6`` keeps the v4.2 reverse-curriculum reset, and
``swingup-v6-uniform`` replaces it with the fixed uniform-random draw.
``swingup-v4.3`` and ``swingup-v6.1`` are reward-preserving branches with a
performance-gated tip-height/velocity curriculum: near-upright at rest first,
then lower rest starts down to exact hanging.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import mujoco
import numpy as np

from dm_control.rl import control
from dm_control.suite import acrobot
from dm_control.suite import base as suite_base
from dm_control.utils import rewards

from .tip_curriculum import (
    DEFAULT_ELBOW_SPREAD,
    CurriculumPose,
    TipHeightVelocityCurriculum,
)


STRICT_CAPTURE_DISTANCE = 0.2
STRICT_CAPTURE_SPEED = 0.2


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

    def _initialize_uniform_episode(self, physics) -> None:
        """Reset to uniform random joint angles with small velocity noise.

        The stock-style exploring-starts reset used by v5 and the uniform
        v4.1 arms: about one draw in five begins above the Gym height, so a
        sparse or capture-pressured reward is observed from the start
        distribution rather than requiring a discovery path from hanging.
        """
        qpos = self.random.uniform(-np.pi, np.pi, 2)
        qvel_noise = self.random.uniform(
            -self.velocity_noise, self.velocity_noise, physics.model.nv
        )
        physics.named.data.qpos[["shoulder", "elbow"]] = qpos
        physics.named.data.qvel[:] = qvel_noise
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


class CurriculumReset:
    """Reverse-curriculum reset band shared by the v4.2 and v6 swing-up tasks.

    Mixed in ahead of the ``Balance`` bases so a task only has to call
    ``_configure_curriculum`` from its constructor and dispatch to
    ``_initialize_curriculum_episode`` from ``initialize_episode``.  The host
    task supplies ``random`` and ``velocity_noise``.
    """

    def _configure_curriculum(
        self, *, curriculum: bool, curriculum_min_spread: float
    ) -> None:
        self.curriculum = bool(curriculum)
        self.curriculum_min_spread = float(curriculum_min_spread)
        if not np.isfinite(self.curriculum_min_spread) or not (
            0.0 < self.curriculum_min_spread <= np.pi
        ):
            raise ValueError(
                "curriculum_min_spread must be finite and in (0, pi]"
            )
        # Curriculum progress in [0, 1]; 0 = tightest near-upright band.  The
        # trainer drives it from global training progress each step.
        self._curriculum_fraction = 0.0

    def set_curriculum_fraction(self, fraction: float) -> None:
        """Set curriculum progress in [0, 1] (0 = tightest near-upright band)."""
        frac = float(fraction)
        if not np.isfinite(frac):
            raise ValueError("curriculum fraction must be finite")
        self._curriculum_fraction = float(np.clip(frac, 0.0, 1.0))

    @property
    def curriculum_fraction(self) -> float:
        return self._curriculum_fraction

    def _initialize_curriculum_episode(self, physics) -> None:
        """Reset from a band around the upright whose width grows with progress.

        The half-width grows from ``curriculum_min_spread`` at fraction 0 to
        pi at fraction 1, so early episodes start near the upright rest pose
        (high energy, a short controlled fall to the goal) and later episodes
        span the full circle, including the near-hanging pose.  At fraction 1
        the draw is uniform on [-pi, pi] for both joints, matching the uniform
        reset, so the curriculum reset is a schedule over the same support that
        lowers the least-energy reachable start from near Ẽ = 1 toward Ẽ = 0.
        """
        spread = self.curriculum_min_spread + self._curriculum_fraction * (
            np.pi - self.curriculum_min_spread
        )
        offsets = self.random.uniform(-spread, spread, 2)
        # Wrap into (-pi, pi]; identity for spread <= pi, kept for safety.
        qpos = np.arctan2(np.sin(offsets), np.cos(offsets))
        qvel_noise = self.random.uniform(
            -self.velocity_noise, self.velocity_noise, physics.model.nv
        )
        physics.named.data.qpos[["shoulder", "elbow"]] = qpos
        physics.named.data.qvel[:] = qvel_noise
        suite_base.Task.initialize_episode(self, physics)


class MechanicalEnergy:
    """Mechanical-energy references shared by the v4 and v6 swing-up tasks.

    ``span = E_up − E_hang`` is the energy a swing-up must supply: the potential
    barrier between hanging at rest and upright at rest.  Both references come
    from the MuJoCo model, so they are pose-independent and constant for a task
    instance.  The host sets ``_energy_hang``/``_energy_span`` to None in its
    constructor and calls ``_calibrate_energy`` from ``initialize_episode``.
    """

    @staticmethod
    def _mass_matrix(physics) -> np.ndarray:
        """Dense joint-space inertia M(q) at the current configuration."""
        nv = int(physics.model.nv)
        mass_matrix = np.zeros((nv, nv), dtype=np.float64)
        mujoco.mj_fullM(physics.model.ptr, mass_matrix, physics.data.qM)
        return mass_matrix

    @classmethod
    def _mechanical_energy(cls, physics) -> float:
        """Total mechanical energy: ½q̇ᵀM(q)q̇ − Σᵢ mᵢ·g⃗·x⃗ᵢ."""
        qvel = np.asarray(physics.data.qvel, dtype=np.float64)
        kinetic = 0.5 * float(qvel @ cls._mass_matrix(physics) @ qvel)
        potential = -float(
            np.asarray(physics.model.body_mass)
            @ (np.asarray(physics.data.xipos) @ np.asarray(physics.model.opt.gravity))
        )
        return kinetic + potential

    def _calibrate_energy(self, physics) -> None:
        """Measure the hanging-rest and upright-rest energies from the model.

        Clobbers the physics state, so callers must set the episode pose after.
        """
        physics.data.qvel[:] = 0.0
        physics.named.data.qpos[["shoulder", "elbow"]] = [0.0, 0.0]
        physics.forward()
        energy_up = self._mechanical_energy(physics)
        physics.named.data.qpos[["shoulder", "elbow"]] = [np.pi, 0.0]
        physics.forward()
        energy_hang = self._mechanical_energy(physics)
        span = energy_up - energy_hang
        if not np.isfinite(span) or span <= 0.0:
            raise RuntimeError(
                "Acrobot energy calibration failed: upright-rest energy must "
                f"exceed hanging-rest energy, got span {span}"
            )
        self._energy_hang = energy_hang
        self._energy_span = span

    def _ensure_energy_calibrated(self, physics) -> None:
        """Calibrate on first use, restoring the state ``_calibrate_energy`` clobbers.

        Lets a task read the references from ``reward_terms`` without requiring
        an ``initialize_episode`` first, which matters where the energy is a
        diagnostic rather than a reward term.
        """
        if self._energy_hang is not None and self._energy_span is not None:
            return
        qpos = np.array(physics.data.qpos, dtype=np.float64, copy=True)
        qvel = np.array(physics.data.qvel, dtype=np.float64, copy=True)
        ctrl = np.array(physics.data.ctrl, dtype=np.float64, copy=True)
        try:
            self._calibrate_energy(physics)
        finally:
            physics.data.qpos[:] = qpos
            physics.data.qvel[:] = qvel
            physics.data.ctrl[:] = ctrl
            physics.forward()


class BalanceV4(CurriculumReset, MechanicalEnergy, BalanceV3):
    """Energy-regulated from-down swing-up with a velocity-gated hold reward.

    reward = (1 − hold_weight)·ramp + hold_weight·hold, both factors in [0, 1]:

    * ``ramp = energy_close · (1 + mean_uprightness)/2`` where ``energy_close``
      is a Gaussian tolerance around the normalized mechanical energy of the
      upright rest pose (Ẽ = 1; hanging rest is Ẽ = 0).  Any elbow motion that
      pumps energy toward Ẽ = 1 raises this term, so the transient swing-up
      behavior is rewarded rather than penalized; overshooting energy (fast
      spinning) is symmetrically discounted.  The uprightness tilt halves the
      value of parking on the Ẽ = 1 manifold away from the top.
    * ``hold = precise · slow``: the stock precise target reward gated by a
      Gaussian tolerance on ‖q̇‖.  Sustained near-maximal income therefore
      exists only while balancing at the exact goal; wobbling or slowly
      spinning through the target region earns transient fractions at most.

    Mechanism and observations are identical to v2/v3.  ``uniform_start``
    selects the reset: ``False`` (v4) starts near hanging; ``True`` (the
    v4.1 default) starts from uniform random joint angles so the hold region
    is present in the start distribution — see ``energy_overshoot_margin``.
    """

    _SPEED_BOUNDS = (0.0, 0.5)
    _SPEED_MARGIN = 2.0
    _ENERGY_MARGIN = 1.0

    def __init__(
        self,
        *,
        random=None,
        angle_noise: float = 0.05,
        velocity_noise: float = 0.01,
        hold_weight: float = 0.8,
        energy_overshoot_margin: float = 1.0,
        speed_bounds: tuple[float, float] = _SPEED_BOUNDS,
        speed_margin: float = _SPEED_MARGIN,
        uniform_start: bool = False,
        curriculum: bool = False,
        curriculum_min_spread: float = 0.5,
    ) -> None:
        super().__init__(
            random=random,
            angle_noise=angle_noise,
            velocity_noise=velocity_noise,
        )
        self.hold_weight = float(hold_weight)
        if not np.isfinite(self.hold_weight) or not (
            0.0 <= self.hold_weight <= 1.0
        ):
            raise ValueError("hold_weight must be finite and in [0, 1]")
        self.energy_overshoot_margin = float(energy_overshoot_margin)
        if (
            not np.isfinite(self.energy_overshoot_margin)
            or self.energy_overshoot_margin <= 0.0
        ):
            raise ValueError(
                "energy_overshoot_margin must be finite and positive"
            )
        try:
            speed_lo, speed_hi = (float(v) for v in speed_bounds)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "speed_bounds must contain exactly two numbers"
            ) from exc
        if (
            not np.isfinite(speed_lo)
            or not np.isfinite(speed_hi)
            or speed_lo < 0.0
            or speed_hi < speed_lo
        ):
            raise ValueError(
                "speed_bounds must be finite, non-negative, and ordered"
            )
        self.speed_bounds = (speed_lo, speed_hi)
        self.speed_margin = float(speed_margin)
        if not np.isfinite(self.speed_margin) or self.speed_margin <= 0.0:
            raise ValueError("speed_margin must be finite and positive")
        self.uniform_start = bool(uniform_start)
        self._configure_curriculum(
            curriculum=curriculum,
            curriculum_min_spread=curriculum_min_spread,
        )
        self._energy_hang: Optional[float] = None
        self._energy_span: Optional[float] = None

    def initialize_episode(self, physics) -> None:
        # Energy calibration is pose-independent, so the reset choice below
        # composes with it cleanly.
        self._calibrate_energy(physics)
        if self.curriculum:
            self._initialize_curriculum_episode(physics)
        elif self.uniform_start:
            self._initialize_uniform_episode(physics)
        else:
            super().initialize_episode(physics)

    def reward_terms(self, physics) -> Dict[str, float]:
        """Return energy-regulated reward terms and diagnostics."""
        if self._energy_hang is None or self._energy_span is None:
            raise RuntimeError(
                "BalanceV4 reward requested before initialize_episode "
                "calibrated the energy references"
            )
        distance = float(physics.to_target())
        precise = float(acrobot.Balance._get_reward(self, physics, sparse=False))

        vertical = np.asarray(physics.vertical(), dtype=np.float64).reshape(-1)
        if vertical.shape != (2,):
            raise ValueError(
                "Acrobot vertical orientation must have shape (2,), got "
                f"{vertical.shape}"
            )
        upright = np.clip((vertical + 1.0) / 2.0, 0.0, 1.0)
        mean_upright = 0.5 * (upright[0] + upright[1])

        energy_norm = (
            self._mechanical_energy(physics) - self._energy_hang
        ) / self._energy_span
        # Piecewise margin: the deficit side keeps the broad pumping ramp,
        # the overshoot side may be tightened (v4.1) so spinning past the
        # upright-rest energy is discounted hard and the policy regulates
        # toward slow top passes.  Both sides meet at 1 at the bound.
        energy_margin = (
            self._ENERGY_MARGIN
            if energy_norm <= 1.0
            else self.energy_overshoot_margin
        )
        energy_close = float(
            rewards.tolerance(
                energy_norm,
                bounds=(1.0, 1.0),
                margin=energy_margin,
                value_at_margin=0.1,
                sigmoid="gaussian",
            )
        )
        ramp = float(energy_close * 0.5 * (1.0 + mean_upright))

        speed = float(
            np.linalg.norm(np.asarray(physics.data.qvel, dtype=np.float64))
        )
        slow = float(
            rewards.tolerance(
                speed,
                bounds=self.speed_bounds,
                margin=self.speed_margin,
                value_at_margin=0.1,
                sigmoid="gaussian",
            )
        )
        hold = float(precise * slow)
        reward = (1.0 - self.hold_weight) * ramp + self.hold_weight * hold

        elbow = float(np.asarray(physics.named.data.qpos["elbow"]).item())
        extension = float(np.clip((1.0 + np.cos(elbow)) / 2.0, 0.0, 1.0))
        target_radius = float(physics.named.model.site_size["target", 0])
        tip_height = float(physics.named.data.site_xpos["tip", "z"])
        exact_success = float(distance <= target_radius)
        strict_capture = float(
            distance < STRICT_CAPTURE_DISTANCE and speed < STRICT_CAPTURE_SPEED
        )
        return {
            "reward": float(np.clip(reward, 0.0, 1.0)),
            "tip_distance": distance,
            "tip_height": tip_height,
            "progress": ramp,
            "precision": precise,
            "upper_uprightness": float(upright[0]),
            "lower_uprightness": float(upright[1]),
            "extension": extension,
            "energy_norm": float(energy_norm),
            "speed": speed,
            "slow_gate": slow,
            "hold": hold,
            "strict_capture": strict_capture,
            "gym_height_success": float(tip_height > self._GYM_TARGET_HEIGHT),
            "exact_success": exact_success,
            "success": exact_success,
        }


def swingup_v4(
    *,
    time_limit: float = 10.0,
    random=None,
    environment_kwargs: Optional[Dict[str, Any]] = None,
    angle_noise: float = 0.05,
    velocity_noise: float = 0.01,
    hold_weight: float = 0.8,
    energy_overshoot_margin: float = 1.0,
    speed_bounds: tuple[float, float] = BalanceV4._SPEED_BOUNDS,
    speed_margin: float = BalanceV4._SPEED_MARGIN,
    uniform_start: bool = False,
):
    """Construct the energy-regulated ``acrobot-swingup-v4`` environment."""
    physics = acrobot.Physics.from_xml_string(*acrobot.get_model_and_assets())
    task = BalanceV4(
        random=random,
        angle_noise=angle_noise,
        velocity_noise=velocity_noise,
        hold_weight=hold_weight,
        energy_overshoot_margin=energy_overshoot_margin,
        speed_bounds=speed_bounds,
        speed_margin=speed_margin,
        uniform_start=uniform_start,
    )
    return control.Environment(
        physics,
        task,
        time_limit=float(time_limit),
        **dict(environment_kwargs or {}),
    )


V41_ENERGY_OVERSHOOT_MARGIN = 0.25
V41_SPEED_BOUNDS = (0.0, 0.1)
V41_SPEED_MARGIN = 0.5


def swingup_v41(
    *,
    time_limit: float = 10.0,
    random=None,
    environment_kwargs: Optional[Dict[str, Any]] = None,
    angle_noise: float = 0.05,
    velocity_noise: float = 0.01,
    hold_weight: float = 0.8,
    speed_bounds: tuple[float, float] = V41_SPEED_BOUNDS,
    speed_margin: float = V41_SPEED_MARGIN,
    uniform_start: bool = True,
):
    """Construct ``acrobot-swingup-v4.1``: v4 capture pressure, uniform start.

    The pumping ramp remains identical to v4 for Ẽ ≤ 1. Above the
    upright-rest energy its margin drops from 1.0 to 0.25, so passing the top
    with surplus energy loses ramp income. The hold speed tolerance is also
    tightened from bounds [0, 0.5], margin 2.0 to bounds [0, 0.1], margin
    0.5, making appreciable hold income require an actually slow tip capture.

    Episodes start from uniform random joint angles (``uniform_start=True``,
    the default).  The capture-pressured reward has its maximum at the slow
    hold on the Ẽ = 1 manifold, but from hanging that region is reachable
    only through the overshoot the margin now penalizes — the hanging-start
    v4.1 pilots removed their own discovery path and never captured.  The
    uniform reset puts near-top, near-Ẽ = 1 states in the start distribution
    so the hold is learned directly and its value propagates outward.
    ``uniform_start=False`` restores the near-hanging reset.
    """
    physics = acrobot.Physics.from_xml_string(*acrobot.get_model_and_assets())
    task = BalanceV4(
        random=random,
        angle_noise=angle_noise,
        velocity_noise=velocity_noise,
        hold_weight=hold_weight,
        energy_overshoot_margin=V41_ENERGY_OVERSHOOT_MARGIN,
        speed_bounds=speed_bounds,
        speed_margin=speed_margin,
        uniform_start=uniform_start,
    )
    return control.Environment(
        physics,
        task,
        time_limit=float(time_limit),
        **dict(environment_kwargs or {}),
    )


def swingup_v42(
    *,
    time_limit: float = 10.0,
    random=None,
    environment_kwargs: Optional[Dict[str, Any]] = None,
    angle_noise: float = 0.05,
    velocity_noise: float = 0.01,
    hold_weight: float = 0.8,
    speed_bounds: tuple[float, float] = V41_SPEED_BOUNDS,
    speed_margin: float = V41_SPEED_MARGIN,
    curriculum: bool = True,
    curriculum_min_spread: float = 0.5,
    uniform_start: bool = True,
):
    """Construct ``acrobot-swingup-v4.2``: v4.1 reward with a reverse curriculum.

    The per-step reward is identical to v4.1 — the same pumping ramp with the
    tightened overshoot margin and the same velocity-gated hold.  Only the
    training reset changes: instead of a fixed uniform draw, episodes start in
    a band around the upright whose half-width grows with training progress,
    from ``curriculum_min_spread`` up to the full circle.  Early episodes then
    begin already near the top, where the slow capture v4.1 rewards is directly
    learnable; as the band widens the start energy reaches down toward hanging,
    so the capture value learned first propagates onto progressively longer
    swing-ups.

    The trainer drives progress by calling ``set_curriculum_fraction`` on the
    task each step.  ``curriculum=False`` disables the schedule and falls back
    to ``uniform_start`` (used for evaluation, where the start distribution
    must be fixed); at fraction 1 the curriculum reset already coincides with
    the uniform draw.
    """
    physics = acrobot.Physics.from_xml_string(*acrobot.get_model_and_assets())
    task = BalanceV4(
        random=random,
        angle_noise=angle_noise,
        velocity_noise=velocity_noise,
        hold_weight=hold_weight,
        energy_overshoot_margin=V41_ENERGY_OVERSHOOT_MARGIN,
        speed_bounds=speed_bounds,
        speed_margin=speed_margin,
        uniform_start=uniform_start,
        curriculum=curriculum,
        curriculum_min_spread=curriculum_min_spread,
    )
    return control.Environment(
        physics,
        task,
        time_limit=float(time_limit),
        **dict(environment_kwargs or {}),
    )


class BalanceV5(BalanceV3):
    """Unshaped height-occupancy objective with uniform random starts.

    Reward is 1 while the tip strictly exceeds the Gym height criterion and
    0 otherwise, with no termination: with the wrapper's reward-increment
    convention the return is the physical time spent above the height over
    the fixed-length episode.  There is no dense term below the height and
    therefore nothing to park on, and maximal income is sustained tip
    elevation — balancing near the top is the implicit optimum without any
    velocity gate or target-distance shaping.

    By default episodes start from uniform random joint angles at near-zero
    velocity (``uniform_start=True``) instead of the near-hanging pose the
    shaped versions use.  About one reset in five then begins above the
    height, so the sparse income is present in the replay data from the
    first episodes and its value can propagate outward to lower starts;
    from the hanging start alone the reward is never observed at all.
    Resets above the line are unstable inverted poses, so collecting their
    income immediately trains the balance skill.  ``uniform_start=False``
    restores the shared near-hanging reset.

    The Gym predicate −cos θ₁ − cos(θ₁+θ₂) > 1 is tip height strictly above
    one link length over the pivot, i.e. ``tip_z > 3`` on this scaled model —
    identical to the ``gym_height_success`` diagnostic of v3/v4.  Mechanism
    and observations are identical to v2–v4.
    """

    def __init__(
        self,
        *,
        random=None,
        angle_noise: float = 0.05,
        velocity_noise: float = 0.01,
        uniform_start: bool = True,
    ) -> None:
        super().__init__(
            random=random,
            angle_noise=angle_noise,
            velocity_noise=velocity_noise,
        )
        self.uniform_start = bool(uniform_start)

    def initialize_episode(self, physics) -> None:
        if self.uniform_start:
            self._initialize_uniform_episode(physics)
        else:
            super().initialize_episode(physics)

    def _gym_height_reached(self, physics) -> bool:
        tip_height = float(physics.named.data.site_xpos["tip", "z"])
        return tip_height > self._GYM_TARGET_HEIGHT

    def reward_terms(self, physics) -> Dict[str, float]:
        """Return the height-occupancy reward and reward-independent terms."""
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
        target_radius = float(physics.named.model.site_size["target", 0])
        tip_height = float(physics.named.data.site_xpos["tip", "z"])
        gym_height_success = float(tip_height > self._GYM_TARGET_HEIGHT)
        exact_success = float(distance <= target_radius)
        return {
            "reward": gym_height_success,
            "tip_distance": distance,
            "tip_height": tip_height,
            # No dense progress term exists below the height criterion.
            "progress": 0.0,
            "precision": precise,
            "upper_uprightness": float(upright[0]),
            "lower_uprightness": float(upright[1]),
            "extension": extension,
            "gym_height_success": gym_height_success,
            "exact_success": exact_success,
            "success": exact_success,
        }


def swingup_v5(
    *,
    time_limit: float = 30.0,
    random=None,
    environment_kwargs: Optional[Dict[str, Any]] = None,
    angle_noise: float = 0.05,
    velocity_noise: float = 0.01,
    uniform_start: bool = True,
):
    """Construct the height-occupancy ``acrobot-swingup-v5`` environment."""
    physics = acrobot.Physics.from_xml_string(*acrobot.get_model_and_assets())
    task = BalanceV5(
        random=random,
        angle_noise=angle_noise,
        velocity_noise=velocity_noise,
        uniform_start=uniform_start,
    )
    return control.Environment(
        physics,
        task,
        time_limit=float(time_limit),
        **dict(environment_kwargs or {}),
    )


# Published AR-EAPO weights for the IROS-2024 acrobot and pendubot:
# Q = diag(50, 50, 4, 2) over (θ₁, θ₂, θ̇₁, θ̇₂), R = 1, α = 0.001.
V6_STATE_WEIGHTS = (50.0, 50.0, 4.0, 2.0)
V6_ACTION_WEIGHT = 1.0
V6_COST_SCALE = 0.001
# Kinetic energy below which the per-joule ratio has no direction to report and
# ``velocity_cost_per_joule``/``coordination_loss`` are NaN rather than 0.
_COST_PER_JOULE_MIN_KINETIC = 1e-9


class BalanceV6(CurriculumReset, MechanicalEnergy, BalanceV3):
    """Quadratic-cost swing-up: the AR-EAPO reward on a continuing task.

        reward = −α[(s − g)ᵀQ(s − g) + aᵀRa],  s = [θ₁, θ₂, θ̇₁, θ̇₂],  g = 0,

    with Q = diag(``state_weights``), R = ``action_weight``, α =
    ``cost_scale``.  The defaults are the published values of Choe et al.
    (2024), eq. 16, whose only shaping is this one term — no energy shell, no
    velocity gate, no precise-target tail.  Every configuration is separated
    from the goal by a strictly monotone position cost, which is what v4's
    Gaussian energy ramp does not provide: at the hanging rest pose v4 pays a
    flat 0.01/step, so the swing-up has nothing to descend.

    Two deviations from the paper, both forced by this mechanism:

    * Angle errors are wrapped into (−π, π] before squaring.  The paper's raw
      difference is discontinuous at the branch cut, which the uniform and
      curriculum resets sample directly.
    * R multiplies the normalized actuator command a ∈ [−1, 1] rather than a
      torque in N·m, so ``action_weight`` carries the paper's R·τ_max².  At the
      default it contributes at most α = 0.001 per step.

    Unlike v2–v5 the reward is a cost: ≤ 0, equal to 0 only at the upright rest
    pose with zero command, and never clipped.  Nothing terminates, so this is
    the continuing MDP an average-reward criterion assumes.  ``reward_offset``
    adds a constant; it leaves the optimal policy unchanged under both the
    discounted-soft and the average-reward objective (no state is absorbing and
    the time limit truncates with bootstrapping) and exists only to put returns
    on the same scale as the [0, 1]-reward arms.

    The reset is v4.2's: ``curriculum`` starts episodes in a band around the
    upright whose half-width grows from ``curriculum_min_spread`` to pi with
    training progress.  With the curriculum off, ``uniform_start`` selects the
    uniform draw (evaluation) or the near-hanging pose.

    Mechanism and observations are identical to v2–v5.
    """

    def __init__(
        self,
        *,
        random=None,
        angle_noise: float = 0.05,
        velocity_noise: float = 0.01,
        state_weights: tuple[float, ...] = V6_STATE_WEIGHTS,
        action_weight: float = V6_ACTION_WEIGHT,
        cost_scale: float = V6_COST_SCALE,
        reward_offset: float = 0.0,
        uniform_start: bool = True,
        curriculum: bool = True,
        curriculum_min_spread: float = 0.5,
    ) -> None:
        super().__init__(
            random=random,
            angle_noise=angle_noise,
            velocity_noise=velocity_noise,
        )
        weights = np.asarray(state_weights, dtype=np.float64).reshape(-1)
        if weights.shape != (4,):
            raise ValueError(
                "state_weights must hold four values (θ₁, θ₂, θ̇₁, θ̇₂), got "
                f"shape {weights.shape}"
            )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("state_weights must be finite and non-negative")
        self.state_weights = tuple(float(w) for w in weights)
        self.action_weight = self._finite_nonnegative(
            "action_weight", action_weight
        )
        self.cost_scale = float(cost_scale)
        if not np.isfinite(self.cost_scale) or self.cost_scale <= 0.0:
            raise ValueError("cost_scale must be finite and positive")
        self.reward_offset = float(reward_offset)
        if not np.isfinite(self.reward_offset):
            raise ValueError("reward_offset must be finite")
        self.uniform_start = bool(uniform_start)
        self._configure_curriculum(
            curriculum=curriculum,
            curriculum_min_spread=curriculum_min_spread,
        )
        self._energy_hang: Optional[float] = None
        self._energy_span: Optional[float] = None

    @property
    def max_angle_cost(self) -> float:
        """Position cost at the antipode: α(w₁ + w₂)π², the worst pose."""
        w_angle = self.state_weights[:2]
        return float(self.cost_scale * (w_angle[0] + w_angle[1]) * np.pi**2)

    @property
    def velocity_cost_matrix(self) -> np.ndarray:
        """W with q̇ᵀWq̇ equal to the reward's velocity cost."""
        return self.cost_scale * np.diag(np.asarray(self.state_weights[2:]))

    def _cost_per_joule_bounds(self, physics) -> tuple[float, float]:
        """Cheapest and dearest velocity cost per joule of kinetic energy at q.

        The velocity cost is q̇ᵀWq̇ and the kinetic energy ½q̇ᵀM(q)q̇, so their
        ratio is a generalized Rayleigh quotient bounded by twice the extreme
        eigenvalues of the pencil (W, M).  M(q) is symmetric positive definite,
        so a Cholesky factor turns it into a symmetric eigenproblem.  The bounds
        move with the elbow angle, which is what makes the normalized reading
        below comparable across poses.
        """
        mass_matrix = self._mass_matrix(physics)
        factor = np.linalg.cholesky(mass_matrix)
        whitened = np.linalg.solve(
            factor, np.linalg.solve(factor, self.velocity_cost_matrix).T
        )
        eigenvalues = np.linalg.eigvalsh(0.5 * (whitened + whitened.T))
        return 2.0 * float(eigenvalues[0]), 2.0 * float(eigenvalues[-1])

    def initialize_episode(self, physics) -> None:
        # Pose-independent, and every branch below sets the pose afterwards.
        self._calibrate_energy(physics)
        if self.curriculum:
            self._initialize_curriculum_episode(physics)
        elif self.uniform_start:
            self._initialize_uniform_episode(physics)
        else:
            super().initialize_episode(physics)

    def reward_terms(self, physics) -> Dict[str, float]:
        """Return the quadratic cost, its three parts, and the shared diagnostics."""
        qpos = np.asarray(physics.data.qpos, dtype=np.float64).reshape(-1)
        qvel = np.asarray(physics.data.qvel, dtype=np.float64).reshape(-1)
        if qpos.shape != (2,) or qvel.shape != (2,):
            raise ValueError(
                "Acrobot state must be two positions and two velocities, got "
                f"qpos {qpos.shape}, qvel {qvel.shape}"
            )
        # Wrapped error to the upright rest pose g = 0 (shoulder = elbow = 0).
        angle_error = np.arctan2(np.sin(qpos), np.cos(qpos))
        w_angle = np.asarray(self.state_weights[:2], dtype=np.float64)
        w_velocity = np.asarray(self.state_weights[2:], dtype=np.float64)
        angle_cost = self.cost_scale * float(w_angle @ angle_error**2)
        velocity_cost = self.cost_scale * float(w_velocity @ qvel**2)
        # physics.control() is the command dm_control applied for this step.
        command = np.asarray(physics.control(), dtype=np.float64).reshape(-1)
        action_cost = (
            self.cost_scale * self.action_weight * float(command @ command)
        )
        reward = self.reward_offset - (angle_cost + velocity_cost + action_cost)

        distance = float(physics.to_target())
        precise = float(acrobot.Balance._get_reward(self, physics, sparse=False))
        vertical = np.asarray(physics.vertical(), dtype=np.float64).reshape(-1)
        if vertical.shape != (2,):
            raise ValueError(
                "Acrobot vertical orientation must have shape (2,), got "
                f"{vertical.shape}"
            )
        upright = np.clip((vertical + 1.0) / 2.0, 0.0, 1.0)
        # Exploration diagnostics.  The reward's velocity cost multiplies two
        # independent things — how much energy is in motion, and how wastefully
        # it is carried — so the pair below factors them apart.  ``energy_norm``
        # is the swing-up budget currently held (kinetic and potential
        # together), the slow variable pumping raises; the per-joule ratio is
        # scale-free in q̇ and therefore reads coordination alone.
        self._ensure_energy_calibrated(physics)
        energy_norm = (
            self._mechanical_energy(physics) - self._energy_hang
        ) / self._energy_span
        kinetic = 0.5 * float(qvel @ self._mass_matrix(physics) @ qvel)
        kinetic_norm = kinetic / self._energy_span
        cheap_bound, dear_bound = self._cost_per_joule_bounds(physics)
        if kinetic > _COST_PER_JOULE_MIN_KINETIC:
            cost_per_joule = velocity_cost / kinetic
            # 0 = the cheapest coordination at this pose, 1 = the dearest.
            coordination_loss = (cost_per_joule - cheap_bound) / max(
                dear_bound - cheap_bound, 1e-12
            )
            coordination_loss = float(np.clip(coordination_loss, 0.0, 1.0))
        else:
            # At rest there is no direction to read, and any finite sentinel
            # would enter the logged running mean: zero sits below the ratio's
            # lower bound, so resting steps would drag the mean toward
            # "perfectly coordinated" — the exact misreading these terms exist
            # to prevent, and worst on a collapsed policy, which rests most.
            # NaN is dropped by the Monitor instead, leaving the logged value
            # to mean coordination *while moving*.  ``energy_norm`` and
            # ``kinetic_norm`` stay finite and report whether it moves at all.
            cost_per_joule, coordination_loss = float("nan"), float("nan")

        elbow = float(qpos[1])
        extension = float(np.clip((1.0 + np.cos(elbow)) / 2.0, 0.0, 1.0))
        speed = float(np.linalg.norm(qvel))
        target_radius = float(physics.named.model.site_size["target", 0])
        tip_height = float(physics.named.data.site_xpos["tip", "z"])
        exact_success = float(distance <= target_radius)
        strict_capture = float(
            distance < STRICT_CAPTURE_DISTANCE and speed < STRICT_CAPTURE_SPEED
        )
        return {
            "reward": float(reward),
            "tip_distance": distance,
            "tip_height": tip_height,
            # The reward has no separate progress factor; report the position
            # cost re-expressed on [0, 1] (1 at the goal, 0 at the antipode).
            "progress": float(
                np.clip(1.0 - angle_cost / self.max_angle_cost, 0.0, 1.0)
            ),
            "precision": precise,
            "upper_uprightness": float(upright[0]),
            "lower_uprightness": float(upright[1]),
            "extension": extension,
            "angle_cost": angle_cost,
            "velocity_cost": velocity_cost,
            "action_cost": action_cost,
            "energy_norm": float(energy_norm),
            "kinetic_norm": float(kinetic_norm),
            "velocity_cost_per_joule": float(cost_per_joule),
            "coordination_loss": coordination_loss,
            "speed": speed,
            "strict_capture": strict_capture,
            "gym_height_success": float(tip_height > self._GYM_TARGET_HEIGHT),
            "exact_success": exact_success,
            "success": exact_success,
        }


def swingup_v6(
    *,
    time_limit: float = 20.0,
    random=None,
    environment_kwargs: Optional[Dict[str, Any]] = None,
    angle_noise: float = 0.05,
    velocity_noise: float = 0.01,
    state_weights: tuple[float, ...] = V6_STATE_WEIGHTS,
    action_weight: float = V6_ACTION_WEIGHT,
    cost_scale: float = V6_COST_SCALE,
    reward_offset: float = 0.0,
    curriculum: bool = True,
    curriculum_min_spread: float = 0.5,
    uniform_start: bool = True,
):
    """Construct ``acrobot-swingup-v6``: the AR-EAPO quadratic cost on v4.2's reset.

    The per-step reward is the published quadratic state-and-command cost; the
    reset is the v4.2 reverse curriculum, driven by the trainer through
    ``set_curriculum_fraction``.  ``curriculum=False`` disables the schedule and
    falls back to ``uniform_start``, which evaluation requires so the start
    distribution stays fixed.  For a training arm that never schedules the
    reset, use ``swingup_v6_uniform`` instead of passing ``curriculum=False``
    here: it registers under its own env id, so the trainer does not attach a
    curriculum callback and the two arms stay separately identifiable.
    """
    physics = acrobot.Physics.from_xml_string(*acrobot.get_model_and_assets())
    task = BalanceV6(
        random=random,
        angle_noise=angle_noise,
        velocity_noise=velocity_noise,
        state_weights=state_weights,
        action_weight=action_weight,
        cost_scale=cost_scale,
        reward_offset=reward_offset,
        uniform_start=uniform_start,
        curriculum=curriculum,
        curriculum_min_spread=curriculum_min_spread,
    )
    return control.Environment(
        physics,
        task,
        time_limit=float(time_limit),
        **dict(environment_kwargs or {}),
    )


def swingup_v6_uniform(
    *,
    time_limit: float = 20.0,
    random=None,
    environment_kwargs: Optional[Dict[str, Any]] = None,
    angle_noise: float = 0.05,
    velocity_noise: float = 0.01,
    state_weights: tuple[float, ...] = V6_STATE_WEIGHTS,
    action_weight: float = V6_ACTION_WEIGHT,
    cost_scale: float = V6_COST_SCALE,
    reward_offset: float = 0.0,
    uniform_start: bool = True,
    curriculum: bool = False,
):
    """Construct ``acrobot-swingup-v6-uniform``: the v6 cost on a uniform reset.

    Identical reward, identical 20 s runway, no reset schedule: every episode
    draws both joint angles uniformly on [-pi, pi] at near-zero velocity, the
    same fixed distribution the v4.1 arms train on and every v6 evaluation
    already uses.  Paired against ``swingup_v6`` this isolates the curriculum,
    since the two differ in nothing else.  ``uniform_start=False`` restores the
    near-hanging reset for the from-down eval track.

    The curriculum is not merely defaulted off but absent: the task carries no
    band schedule to drive, so ``has_curriculum`` is False and the trainer
    attaches no curriculum callback.  ``curriculum`` is accepted only as False,
    because the runner and the evaluation harness pass ``curriculum=False``
    generically to pin eval start distributions; asking this factory to enable
    one is a request for ``swingup_v6`` and is rejected rather than ignored.
    """
    if curriculum:
        raise ValueError(
            "swingup_v6_uniform has no reset schedule; use swingup_v6 "
            "(acrobot-swingup-v6) for the curriculum arm"
        )
    physics = acrobot.Physics.from_xml_string(*acrobot.get_model_and_assets())
    task = BalanceV6(
        random=random,
        angle_noise=angle_noise,
        velocity_noise=velocity_noise,
        state_weights=state_weights,
        action_weight=action_weight,
        cost_scale=cost_scale,
        reward_offset=reward_offset,
        uniform_start=uniform_start,
        curriculum=False,
    )
    return control.Environment(
        physics,
        task,
        time_limit=float(time_limit),
        **dict(environment_kwargs or {}),
    )


# The performance-gated curriculum variants below intentionally live beside,
# rather than replace, the historical v4.2/v6 angle-band curricula.  Their
# reset ladder is expressed only in absolute tip height and incoming Cartesian
# tip speed.  Keeping the reward classes as the second MRO parent makes the
# reward computation byte-for-byte the established v4.1/v6 implementation.
ACROBOT_TIP_HEIGHT_BOUNDS = (0.0, 4.0)
ACROBOT_DESCENT_TIP_HEIGHTS = (3.5, 3.0, 2.0, 1.0, 0.0)


class _AcrobotTipHeightVelocityCurriculum(TipHeightVelocityCurriculum):
    """Map the mechanism-neutral tip curriculum onto Acrobot coordinates."""

    _TIP_SITE = "tip"

    def _configure_acrobot_tip_curriculum(
        self,
        *,
        curriculum: bool,
        descent_tip_heights: tuple[float, ...],
        elbow_spread: float = DEFAULT_ELBOW_SPREAD,
    ) -> None:
        self._configure_tip_curriculum(
            curriculum=curriculum,
            tip_height_bounds=ACROBOT_TIP_HEIGHT_BOUNDS,
            descent_tip_heights=descent_tip_heights,
            elbow_spread=elbow_spread,
            min_start_distance=STRICT_CAPTURE_DISTANCE,
        )

    def _initialize_acrobot_tip_episode(self, physics) -> None:
        """Reset to one sampled pose at the current height/speed level.

        Both links are one length unit and the shoulder sits at ``z = 2``, so
        ``tip_z = 2 + cos(q1) + cos(q1 + q2)`` and the mixin's pose maps
        directly onto the two hinges.
        """

        if self.curriculum:
            pose = self.sample_curriculum_pose()
        else:
            # Fixed evaluation is the exact canonical hanging state, regardless
            # of the angle/velocity noise accepted by the historical bases and
            # of the fold the training resets draw.
            pose = CurriculumPose(np.pi, 0.0, 0.0, 0.0)

        physics.named.data.qpos[["shoulder", "elbow"]] = [
            pose.first_link_angle,
            pose.elbow_angle,
        ]
        physics.data.qvel[:] = 0.0
        physics.named.data.qvel[["shoulder", "elbow"]] = [
            pose.first_link_rate,
            pose.elbow_rate,
        ]
        suite_base.Task.initialize_episode(self, physics)

    @classmethod
    def _tip_cartesian_velocity(cls, physics) -> np.ndarray:
        """Return the world-frame linear velocity of the tip site."""

        nv = int(physics.model.nv)
        jacobian = np.zeros((3, nv), dtype=np.float64)
        rotational = np.zeros((3, nv), dtype=np.float64)
        tip_id = int(physics.model.name2id(cls._TIP_SITE, "site"))
        mujoco.mj_jacSite(
            physics.model.ptr,
            physics.data.ptr,
            jacobian,
            rotational,
            tip_id,
        )
        qvel = np.asarray(physics.data.qvel, dtype=np.float64).reshape(nv)
        return jacobian @ qvel

    @classmethod
    def _tip_cartesian_speed(cls, physics) -> float:
        return float(np.linalg.norm(cls._tip_cartesian_velocity(physics)))

    def initialize_episode(self, physics) -> None:
        # Both reward parents use the same model-derived energy references.
        self._calibrate_energy(physics)
        self._initialize_acrobot_tip_episode(physics)

    def curriculum_terms(self, physics) -> Dict[str, float]:
        """Mechanism-neutral reset state exposed by the Gym wrapper."""

        terms = self.curriculum_diagnostics()
        terms.update(
            {
                "curriculum_enabled": float(self.curriculum),
                "tip_speed": self._tip_cartesian_speed(physics),
            }
        )
        return terms

    def reward_terms(self, physics) -> Dict[str, float]:
        """Add tip/curriculum diagnostics without changing the parent reward."""

        terms = super().reward_terms(physics)
        tip_speed = self._tip_cartesian_speed(physics)
        terms["tip_speed"] = tip_speed
        terms.update(self.curriculum_diagnostics())
        terms["curriculum_enabled"] = float(self.curriculum)
        terms["strict_capture"] = float(
            float(terms["tip_distance"]) < STRICT_CAPTURE_DISTANCE
            and tip_speed < STRICT_CAPTURE_SPEED
        )
        return terms


class BalanceV43(_AcrobotTipHeightVelocityCurriculum, BalanceV4):
    """The exact v4.1 reward with a performance-gated tip reset ladder."""

    def __init__(
        self,
        *,
        random=None,
        angle_noise: float = 0.05,
        velocity_noise: float = 0.01,
        hold_weight: float = 0.8,
        curriculum: bool = True,
        descent_tip_heights: tuple[
            float, ...
        ] = ACROBOT_DESCENT_TIP_HEIGHTS,
        elbow_spread: float = DEFAULT_ELBOW_SPREAD,
    ) -> None:
        super().__init__(
            random=random,
            angle_noise=angle_noise,
            velocity_noise=velocity_noise,
            hold_weight=hold_weight,
            energy_overshoot_margin=V41_ENERGY_OVERSHOOT_MARGIN,
            speed_bounds=V41_SPEED_BOUNDS,
            speed_margin=V41_SPEED_MARGIN,
            uniform_start=False,
            curriculum=False,
        )
        self._configure_acrobot_tip_curriculum(
            curriculum=curriculum,
            descent_tip_heights=descent_tip_heights,
            elbow_spread=elbow_spread,
        )


def swingup_v43(
    *,
    time_limit: float = 20.0,
    random=None,
    environment_kwargs: Optional[Dict[str, Any]] = None,
    angle_noise: float = 0.05,
    velocity_noise: float = 0.01,
    hold_weight: float = 0.8,
    curriculum: bool = True,
    descent_tip_heights: tuple[float, ...] = ACROBOT_DESCENT_TIP_HEIGHTS,
    elbow_spread: float = DEFAULT_ELBOW_SPREAD,
):
    """Construct ``acrobot-swingup-v4.3``."""

    physics = acrobot.Physics.from_xml_string(*acrobot.get_model_and_assets())
    task = BalanceV43(
        random=random,
        angle_noise=angle_noise,
        velocity_noise=velocity_noise,
        hold_weight=hold_weight,
        curriculum=curriculum,
        descent_tip_heights=descent_tip_heights,
        elbow_spread=elbow_spread,
    )
    return control.Environment(
        physics,
        task,
        time_limit=float(time_limit),
        **dict(environment_kwargs or {}),
    )


class BalanceV61(_AcrobotTipHeightVelocityCurriculum, BalanceV6):
    """The exact v6 quadratic reward with the tip reset ladder."""

    def __init__(
        self,
        *,
        random=None,
        angle_noise: float = 0.05,
        velocity_noise: float = 0.01,
        state_weights: tuple[float, ...] = V6_STATE_WEIGHTS,
        action_weight: float = V6_ACTION_WEIGHT,
        cost_scale: float = V6_COST_SCALE,
        reward_offset: float = 0.0,
        curriculum: bool = True,
        descent_tip_heights: tuple[
            float, ...
        ] = ACROBOT_DESCENT_TIP_HEIGHTS,
        elbow_spread: float = DEFAULT_ELBOW_SPREAD,
    ) -> None:
        super().__init__(
            random=random,
            angle_noise=angle_noise,
            velocity_noise=velocity_noise,
            state_weights=state_weights,
            action_weight=action_weight,
            cost_scale=cost_scale,
            reward_offset=reward_offset,
            uniform_start=False,
            curriculum=False,
        )
        self._configure_acrobot_tip_curriculum(
            curriculum=curriculum,
            descent_tip_heights=descent_tip_heights,
            elbow_spread=elbow_spread,
        )


def swingup_v61(
    *,
    time_limit: float = 20.0,
    random=None,
    environment_kwargs: Optional[Dict[str, Any]] = None,
    angle_noise: float = 0.05,
    velocity_noise: float = 0.01,
    state_weights: tuple[float, ...] = V6_STATE_WEIGHTS,
    action_weight: float = V6_ACTION_WEIGHT,
    cost_scale: float = V6_COST_SCALE,
    reward_offset: float = 0.0,
    curriculum: bool = True,
    descent_tip_heights: tuple[float, ...] = ACROBOT_DESCENT_TIP_HEIGHTS,
    elbow_spread: float = DEFAULT_ELBOW_SPREAD,
):
    """Construct ``acrobot-swingup-v6.1``."""

    physics = acrobot.Physics.from_xml_string(*acrobot.get_model_and_assets())
    task = BalanceV61(
        random=random,
        angle_noise=angle_noise,
        velocity_noise=velocity_noise,
        state_weights=state_weights,
        action_weight=action_weight,
        cost_scale=cost_scale,
        reward_offset=reward_offset,
        curriculum=curriculum,
        descent_tip_heights=descent_tip_heights,
        elbow_spread=elbow_spread,
    )
    return control.Environment(
        physics,
        task,
        time_limit=float(time_limit),
        **dict(environment_kwargs or {}),
    )
