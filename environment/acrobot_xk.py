"""Acrobot plant with Xin & Kaneda's geometry and their analysis's assumptions.

The swing-up results of `Xin & Kaneda 2007
<https://doi.org/10.1002/rnc.1184>`_ are derived and simulated for a specific
two-link robot, and the plant here reproduces it exactly:

    m1 = m2 = 1 kg, l1 = 1 m, l2 = 2 m, lc1 = 0.5 m, lc2 = 1 m,
    I1 = 0.083 kg m^2, I2 = 0.33 kg m^2, g = 9.8 m/s^2

which gives the grouped parameters their equations are written in,
``a = (1.333, 1.330, 1.000)`` and ``b = (14.7, 9.8)``, and hence ``E_r = 24.5``.
The link inertias are set through explicit ``<inertial>`` elements rather than
inferred from geom shape, so ``I1`` and ``I2`` land on the published values
instead of on whatever a capsule of that length would have.

Two further properties of their setting are carried over:

* **No dissipation.**  ``Edot = qdot2 tau2`` is the engine of the entire
  derivation.  With joint damping ``d`` it becomes
  ``Edot = qdot2 tau2 - d (qdot1^2 + qdot2^2)``, which leaves
  ``Vdot = -k_V qdot2^2 - (E - E_r) d |qdot|^2`` whose second term is *positive*
  throughout a swing-up.  Worse, on the target set ``q2 = qdot2 = 0`` the
  injected power is identically zero while the shoulder keeps dissipating, so
  the homoclinic orbit is not an invariant set of the damped closed loop for any
  gains.  ``damping`` therefore defaults to 0; a positive value stays reachable
  so the obstruction can be measured rather than asserted.
* **Ample actuation.**  The law is derived without an input bound, and on this
  geometry it asks for around 20 N*m at the paper's own gains.  ``torque_limit``
  sets the actuator gear and defaults high enough never to bind.  It is a kwarg
  rather than a constant because a learned policy sees it as an action scaling:
  the plant applies ``tau2 = gear * ctrl`` with ``ctrl`` in ``[-1, 1]``, so the
  gear multiplies exploration noise and the meaning of the entropy target.

The task carries no shaping.  Its reward is the ``r0`` baseline of
``docs/reward_shaping_for_acrobot_swingup.md``,

    r0(x) = -[(Etil / E_s)^2 + (q2 / q_s)^2 + (qdot2 / omega_s)^2]

which exists so the environment is well formed; the analytical controller
ignores it, and the evaluation metrics are recomputed from raw state so they
stay reward-independent.
"""

from __future__ import annotations

import collections
from typing import Any, Dict, Optional

import numpy as np

from dm_control.rl import control
from dm_control import mujoco
from dm_control.suite import base as suite_base
from dm_control.suite import common


# Xin & Kaneda's published mechanical parameters (their Section 7).
LINK1_LENGTH = 1.0
LINK2_LENGTH = 2.0
LINK1_COM = 0.5
LINK2_COM = 1.0
LINK1_MASS = 1.0
LINK2_MASS = 1.0
LINK1_INERTIA = 0.083
LINK2_INERTIA = 0.33
GRAVITY = 9.8

# Full reach of the chain, used to place the target site, the light and the floor.
REACH = LINK1_LENGTH + LINK2_LENGTH

# High enough never to bind: the law's peak demand on this geometry is about
# 20 N*m at the paper's gains.
DEFAULT_TORQUE_LIMIT = 64.0

# Reset used by the swing-up arms: near hanging, matching the paper's own
# initial condition, which sits a small angle off the downward equilibrium.
DEFAULT_ANGLE_NOISE = 0.05
DEFAULT_VELOCITY_NOISE = 0.01

# The "release" reset: the chain held straight and released from rest with the
# shoulder displaced from hanging.  This is the family the paper's own initial
# condition belongs to (its displacement is 0.1708 rad), and it is the shared
# evaluation distribution for the reward experiments -- see
# ``docs/reward_shaping_for_acrobot_swingup.md``.  The displacement is bounded
# away from zero because hanging is an equilibrium of the closed loop, so a zero
# displacement would never start.
RELEASE_ANGLE_RANGE = (0.05, 0.5)

# The upright and hanging equilibria of eq. 10, and the paper's own initial
# condition, all directly in qpos because the model is built in its coordinates.
UPRIGHT_SHOULDER = 0.5 * np.pi
HANGING_SHOULDER = -0.5 * np.pi
PAPER_INITIAL_SHOULDER = -1.4

# The planar problem only constrains the inertia about the hinge axis; the
# out-of-plane principal moment just has to keep the tensor admissible.
_MINOR_INERTIA = 0.001

# The model is laid out so that ``qpos`` *is* the paper's ``(q1, q2)``, which
# removes the coordinate transform rather than hiding it:
#
#   * the links rest along +x, so ``q1 = 0`` is link 1 horizontal, as in eq. 10;
#   * the hinges turn about -y, so increasing ``q1`` lifts the link toward +z
#     and upright is ``q1 = +pi/2``, hanging ``-pi/2``;
#   * the shoulder sits at the world origin, so MuJoCo's gravitational potential
#     equals ``P(q) = b1 sin q1 + b2 sin(q1 + q2)`` of eq. 7 with no offset.
#
# Consequences, all verified in the tests: ``mj_fullM`` equals ``M(q)``,
# ``qfrc_bias`` equals ``+(H + G)`` with no sign flip, the MuJoCo mechanical
# energy equals ``E`` outright, and ``gear * ctrl`` is ``tau2`` directly.
# ``diaginertia`` is ordered ``(about the rod, about y, about z)`` so the
# published moment lands on the hinge axis.
_MODEL_XML = """
<mujoco model="acrobot-xk">
  <include file="./common/visual.xml"/>
  <include file="./common/skybox.xml"/>
  <include file="./common/materials.xml"/>

  <option timestep="0.01" integrator="RK4" gravity="0 0 -{gravity}">
    <flag constraint="disable" energy="enable"/>
  </option>

  <!-- Large enough offscreen buffer for the swing-up render. -->
  <visual>
    <global offwidth="1280" offheight="960"/>
  </visual>

  <default>
    <joint type="hinge" axis="0 -1 0" damping="{damping}" limited="false"/>
    <geom type="capsule" mass="0"/>
  </default>

  <worldbody>
    <light name="light" pos="0 0 {light_height}"/>
    <geom name="floor" pos="0 0 {floor_height}" size="6 6 .2" type="plane"
          material="grid"/>
    <site name="target" type="sphere" pos="0 0 {reach}" size="0.2"
          material="target" group="3"/>
    <camera name="fixed" pos="0 -8 0" zaxis="0 -1 0"/>
    <body name="upper_arm" pos="0 0 0">
      <joint name="shoulder"/>
      <inertial pos="{lc1} 0 0" mass="{m1}" diaginertia="{minor} {i1} {i1}"/>
      <geom name="upper_arm" fromto="0 0 0 {l1} 0 0" size="0.05" material="self"/>
      <body name="lower_arm" pos="{l1} 0 0">
        <joint name="elbow"/>
        <inertial pos="{lc2} 0 0" mass="{m2}" diaginertia="{minor} {i2} {i2}"/>
        <geom name="lower_arm" fromto="0 0 0 {l2} 0 0" size="0.049"
              material="self"/>
        <site name="tip" pos="{l2} 0 0" size="0.01"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="elbow" joint="elbow" gear="{gear}" ctrllimited="true"
           ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


def _model_xml(damping: float, torque_limit: float) -> bytes:
    """Xin-Kaneda's geometry as a MuJoCo model, at the given damping and gear."""
    if not np.isfinite(damping) or damping < 0.0:
        raise ValueError(f"damping must be finite and >= 0, got {damping}")
    if not np.isfinite(torque_limit) or torque_limit <= 0.0:
        raise ValueError(
            f"torque_limit must be finite and > 0, got {torque_limit}"
        )
    return _MODEL_XML.format(
        gravity=GRAVITY,
        damping=float(damping),
        gear=float(torque_limit),
        reach=REACH,
        light_height=REACH + 2.0,
        floor_height=-(REACH + 3.0),
        l1=LINK1_LENGTH,
        l2=LINK2_LENGTH,
        lc1=LINK1_COM,
        lc2=LINK2_COM,
        m1=LINK1_MASS,
        m2=LINK2_MASS,
        i1=LINK1_INERTIA,
        i2=LINK2_INERTIA,
        minor=_MINOR_INERTIA,
    ).encode("utf-8")


class BalanceXK(suite_base.Task):
    """Swing-up task on the conservative plant, with the ``r0`` baseline reward.

    Deliberately standalone: it shares no code with the ``v2 ... v6.1`` reward
    line, so its numbers cannot drift when those tasks are edited.
    """

    def __init__(
        self,
        *,
        random=None,
        angle_noise: float = DEFAULT_ANGLE_NOISE,
        velocity_noise: float = DEFAULT_VELOCITY_NOISE,
        uniform_start: bool = False,
        paper_start: bool = False,
        release_start: bool = False,
        release_angle_range: tuple = RELEASE_ANGLE_RANGE,
    ) -> None:
        super().__init__(random=random)
        self.angle_noise = float(angle_noise)
        self.velocity_noise = float(velocity_noise)
        if not np.isfinite(self.angle_noise) or self.angle_noise < 0.0:
            raise ValueError("angle_noise must be finite and >= 0")
        if not np.isfinite(self.velocity_noise) or self.velocity_noise < 0.0:
            raise ValueError("velocity_noise must be finite and >= 0")
        self.uniform_start = bool(uniform_start)
        self.paper_start = bool(paper_start)
        self.release_start = bool(release_start)
        chosen = sum(
            (self.uniform_start, self.paper_start, self.release_start)
        )
        if chosen > 1:
            raise ValueError(
                "uniform_start, paper_start and release_start are mutually "
                "exclusive resets"
            )
        low, high = (float(v) for v in release_angle_range)
        if not (0.0 < low < high):
            raise ValueError(
                f"release_angle_range must satisfy 0 < low < high, got {low}, {high}"
            )
        self.release_angle_range = (low, high)
        self._energy_hang: Optional[float] = None
        self._energy_span: Optional[float] = None
        self._rate_scale: Optional[float] = None

    # --- Mechanical energy ------------------------------------------------

    @staticmethod
    def _mass_matrix(physics) -> np.ndarray:
        import mujoco

        nv = int(physics.model.nv)
        mass_matrix = np.zeros((nv, nv), dtype=np.float64)
        mujoco.mj_fullM(physics.model.ptr, mass_matrix, physics.data.qM)
        return mass_matrix

    @classmethod
    def _mechanical_energy(cls, physics) -> float:
        """Kinetic plus gravitational potential, read from the model."""
        qvel = np.asarray(physics.data.qvel, dtype=np.float64)
        kinetic = 0.5 * float(qvel @ cls._mass_matrix(physics) @ qvel)
        potential = -float(
            np.asarray(physics.model.body_mass)
            @ (np.asarray(physics.data.xipos) @ np.asarray(physics.model.opt.gravity))
        )
        return kinetic + potential

    def _calibrate_energy(self, physics) -> None:
        """Measure hanging-rest and upright-rest energies. Clobbers the state."""
        physics.data.qvel[:] = 0.0
        physics.named.data.qpos[["shoulder", "elbow"]] = [UPRIGHT_SHOULDER, 0.0]
        physics.forward()
        energy_up = self._mechanical_energy(physics)
        # M11 with the elbow straight, which is where the homoclinic orbit lives;
        # taking it here keeps the reward's velocity scale a constant instead of
        # a pose-dependent one.
        extended_m11 = float(self._mass_matrix(physics)[0, 0])
        physics.named.data.qpos[["shoulder", "elbow"]] = [HANGING_SHOULDER, 0.0]
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
        # sqrt(2 E_s / M11(0)) is exactly the peak shoulder speed on the
        # homoclinic orbit, so r0 and the evaluation metrics share one scale.
        self._rate_scale = float(np.sqrt(2.0 * span / extended_m11))

    @property
    def energy_span(self) -> float:
        if self._energy_span is None:
            raise RuntimeError("energy references are not calibrated yet")
        return self._energy_span

    @property
    def energy_top(self) -> float:
        if self._energy_hang is None or self._energy_span is None:
            raise RuntimeError("energy references are not calibrated yet")
        return self._energy_hang + self._energy_span

    # --- Episode ----------------------------------------------------------

    def reseed(self, seed) -> None:
        """Make evaluation starts repeatable without rebuilding the task."""
        self._random = np.random.RandomState(seed)

    def initialize_episode(self, physics) -> None:
        self._calibrate_energy(physics)
        if self.uniform_start:
            physics.named.data.qpos[["shoulder", "elbow"]] = self.random.uniform(
                -np.pi, np.pi, 2
            )
            physics.named.data.qvel[["shoulder", "elbow"]] = 0.0
        elif self.release_start:
            # Straight chain, released from rest, shoulder displaced from
            # hanging by a magnitude drawn uniformly and a random sign.
            low, high = self.release_angle_range
            displacement = self.random.uniform(low, high)
            if self.random.uniform() < 0.5:
                displacement = -displacement
            physics.named.data.qpos[["shoulder", "elbow"]] = [
                HANGING_SHOULDER + displacement,
                0.0,
            ]
            physics.named.data.qvel[["shoulder", "elbow"]] = 0.0
        elif self.paper_start:
            physics.named.data.qpos[["shoulder", "elbow"]] = [
                PAPER_INITIAL_SHOULDER,
                0.0,
            ]
            physics.named.data.qvel[["shoulder", "elbow"]] = 0.0
        else:
            physics.named.data.qpos[["shoulder", "elbow"]] = [
                HANGING_SHOULDER,
                0.0,
            ] + self.random.uniform(-self.angle_noise, self.angle_noise, 2)
            physics.named.data.qvel[["shoulder", "elbow"]] = self.random.uniform(
                -self.velocity_noise, self.velocity_noise, 2
            )
        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Wrapped joint angles and rates.

        dm_control's stock Acrobot helpers read the body z-axes, which assume
        links along +z; this model lays them along +x, so the observation is
        built from the joint angles directly.  The swing-up arms use
        ``raw_state_obs`` anyway, which bypasses this entirely.
        """
        angles = np.asarray(physics.data.qpos, dtype=np.float64)
        obs = collections.OrderedDict()
        obs["orientations"] = np.concatenate([np.cos(angles), np.sin(angles)])
        obs["velocity"] = physics.velocity()
        return obs

    # --- Reward -----------------------------------------------------------

    def get_reward(self, physics) -> float:
        return float(self.baseline_terms(physics)["reward"])

    def baseline_terms(self, physics) -> Dict[str, float]:
        """The ``r0`` baseline and its three normalized parts.

        Scales follow the doc: ``E_s`` is the swing-up energy span, ``q_s = pi``,
        and ``omega_s`` is the peak shoulder speed on the homoclinic orbit,
        ``sqrt(2 E_s / M11(0))`` — the speed at which the whole span is carried
        as kinetic energy in the extended pose.
        """
        if self._energy_hang is None or self._energy_span is None:
            self._calibrate_energy(physics)
        qpos = np.asarray(physics.data.qpos, dtype=np.float64).reshape(-1)
        qvel = np.asarray(physics.data.qvel, dtype=np.float64).reshape(-1)
        energy_error = self._mechanical_energy(physics) - self.energy_top
        elbow = float(np.arctan2(np.sin(qpos[1]), np.cos(qpos[1])))
        omega_scale = self._rate_scale
        parts = (
            (energy_error / self.energy_span) ** 2,
            (elbow / np.pi) ** 2,
            (float(qvel[1]) / omega_scale) ** 2,
        )
        return {
            "reward": -float(sum(parts)),
            "energy_error": float(energy_error),
            "energy_error_norm": float(energy_error / self.energy_span),
            "elbow": elbow,
            "elbow_rate": float(qvel[1]),
            "shoulder_rate": float(qvel[0]),
        }


def swingup_xk(
    *,
    time_limit: float = 60.0,
    random=None,
    environment_kwargs: Optional[Dict[str, Any]] = None,
    damping: float = 0.0,
    torque_limit: float = DEFAULT_TORQUE_LIMIT,
    angle_noise: float = DEFAULT_ANGLE_NOISE,
    velocity_noise: float = DEFAULT_VELOCITY_NOISE,
    uniform_start: bool = False,
    paper_start: bool = False,
    release_start: bool = False,
    release_angle_range: tuple = RELEASE_ANGLE_RANGE,
):
    """Construct ``acrobot-swingup-xk``.

    ``damping = 0`` and a configurable ``torque_limit`` are the two deviations
    from the stock model; both are recoverable from the built model, so a caller
    can always confirm which plant it is holding.
    """
    physics = mujoco.Physics.from_xml_string(
        _model_xml(damping, torque_limit), common.ASSETS
    )
    task = BalanceXK(
        random=random,
        angle_noise=angle_noise,
        velocity_noise=velocity_noise,
        uniform_start=uniform_start,
        paper_start=paper_start,
        release_start=release_start,
        release_angle_range=release_angle_range,
    )
    return control.Environment(
        physics,
        task,
        time_limit=float(time_limit),
        **dict(environment_kwargs or {}),
    )
