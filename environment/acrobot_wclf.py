"""Table-II Acrobot plant for the Lai--She 2009 WCLF controller."""

from __future__ import annotations

import collections
from typing import Any, Dict, Optional

import numpy as np

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base as suite_base
from dm_control.suite import common


LINK1_LENGTH = 1.0
LINK2_LENGTH = 2.0
LINK1_COM = 0.5
LINK2_COM = 1.0
LINK1_MASS = 1.0
LINK2_MASS = 1.0
LINK1_INERTIA = 8.33e-2
LINK2_INERTIA = 0.33
GRAVITY = 9.8
ENERGY_TOP = 24.5
DEFAULT_TORQUE_INTERFACE = 50.0
REACH = LINK1_LENGTH + LINK2_LENGTH
_MINOR_INERTIA = 0.001


_MODEL_XML = """
<mujoco model="acrobot-wclf-2009">
  <include file="./common/visual.xml"/>
  <include file="./common/skybox.xml"/>
  <include file="./common/materials.xml"/>

  <option timestep="0.002" integrator="RK4" gravity="0 0 -{gravity}">
    <flag constraint="disable" energy="enable"/>
  </option>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <default>
    <joint type="hinge" axis="0 1 0" damping="0" limited="false"/>
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
      <inertial pos="0 0 {lc1}" mass="{m1}"
                diaginertia="{i1} {i1} {minor}"/>
      <geom name="upper_arm" fromto="0 0 0 0 0 {l1}" size="0.05"
            material="self"/>
      <body name="lower_arm" pos="0 0 {l1}">
        <joint name="elbow"/>
        <inertial pos="0 0 {lc2}" mass="{m2}"
                  diaginertia="{i2} {i2} {minor}"/>
        <geom name="lower_arm" fromto="0 0 0 0 0 {l2}" size="0.049"
              material="self"/>
        <site name="tip" pos="0 0 {l2}" size="0.01"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="elbow" joint="elbow" gear="{gear}" ctrllimited="true"
           ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


def _model_xml(torque_interface: float) -> bytes:
    if not np.isfinite(torque_interface) or torque_interface <= 0.0:
        raise ValueError("torque_interface must be finite and > 0")
    return _MODEL_XML.format(
        gravity=GRAVITY,
        light_height=REACH + 2.0,
        floor_height=-(REACH + 3.0),
        reach=REACH,
        l1=LINK1_LENGTH,
        l2=LINK2_LENGTH,
        lc1=LINK1_COM,
        lc2=LINK2_COM,
        m1=LINK1_MASS,
        m2=LINK2_MASS,
        i1=LINK1_INERTIA,
        i2=LINK2_INERTIA,
        minor=_MINOR_INERTIA,
        gear=float(torque_interface),
    ).encode("utf-8")


class SwingUpWCLF(suite_base.Task):
    """Minimal task exposing the paper-coordinate plant without reward shaping."""

    def __init__(self, *, initial_perturbation: float = 0.0, random=None) -> None:
        super().__init__(random=random)
        self.initial_perturbation = float(initial_perturbation)
        if not np.isfinite(self.initial_perturbation):
            raise ValueError("initial_perturbation must be finite")

    def initialize_episode(self, physics) -> None:
        physics.named.data.qpos[["shoulder", "elbow"]] = [
            np.pi + self.initial_perturbation,
            0.0,
        ]
        physics.named.data.qvel[["shoulder", "elbow"]] = 0.0
        super().initialize_episode(physics)

    def get_observation(self, physics):
        return collections.OrderedDict(
            position=np.asarray(physics.data.qpos, dtype=np.float64).copy(),
            velocity=np.asarray(physics.data.qvel, dtype=np.float64).copy(),
        )

    def get_reward(self, physics) -> float:
        qpos = np.asarray(physics.data.qpos, dtype=np.float64)
        qvel = np.asarray(physics.data.qvel, dtype=np.float64)
        error = np.array([np.arctan2(np.sin(qpos[0]), np.cos(qpos[0])),
                          np.arctan2(np.sin(qpos[1]), np.cos(qpos[1])),
                          qvel[0], qvel[1]])
        return float(-error @ error)


def swingup_wclf(
    *,
    time_limit: float = 20.0,
    random=None,
    environment_kwargs: Optional[Dict[str, Any]] = None,
    initial_perturbation: float = 0.0,
    torque_interface: float = DEFAULT_TORQUE_INTERFACE,
):
    """Construct the paper-coordinate 2009 WCLF Acrobot environment."""
    physics = mujoco.Physics.from_xml_string(
        _model_xml(torque_interface), common.ASSETS
    )
    task = SwingUpWCLF(
        initial_perturbation=initial_perturbation,
        random=random,
    )
    return control.Environment(
        physics,
        task,
        time_limit=float(time_limit),
        **dict(environment_kwargs or {}),
    )
