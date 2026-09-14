"""Stage A: fixed upright reward, local/incoming resets, MuJoCo XK plant.

Public observations are canonical, with shoulder measured from down. MuJoCo's
reused XK XML uses the horizontal shoulder frame; conversion occurs only here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import gymnasium as gym
import numpy as np
import torch
from dm_control import mujoco
from dm_control.suite import common

from environment.acrobot_xk import _model_xml
from models.acrobot_oracle import AcrobotOracle, UprightReward


def load_incoming_states(path):
    """NPZ: states[N,4] q/v, frame scalar; do not guess angle conventions."""
    with np.load(path, allow_pickle=False) as data:
        states = np.asarray(data["states"], dtype=np.float64).copy()
        frame = str(data["frame"].item())
    if states.ndim != 2 or states.shape[1] != 4 or not len(states) or not np.isfinite(states).all():
        raise ValueError("incoming states must be a nonempty finite [N,4] q/v array")
    if frame == "xin_kaneda_qv":
        states[:, 0] += np.pi / 2
    elif frame != "downward_vertical_qv":
        raise ValueError("frame must be 'xin_kaneda_qv' or 'downward_vertical_qv'")
    return states


@dataclass(frozen=True)
class StageAConfig:
    dt: float = 0.01
    physics_dt: float = 0.001
    episode_seconds: float = 5.0
    discount_rate: float = 0.1
    angle_radius: float = 0.05
    velocity_radius: float = 0.1
    capture_angle: float = 0.1
    capture_velocity: float = 0.25
    hold_seconds: float = 1.0
    velocity_limit: float = 2 * math.pi
    elbow_limit: float = math.pi
    shoulder_limit: float | None = math.pi / 2
    incoming_probability: float = 0.5

    def __post_init__(self):
        for name, value in vars(self).items():
            if name == "shoulder_limit" and value is None:
                continue  # Legacy checkpoints had no shoulder failure boundary.
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("dt", "physics_dt", "episode_seconds", "discount_rate", "capture_angle",
                     "capture_velocity", "hold_seconds", "velocity_limit", "elbow_limit"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.incoming_probability > 1:
            raise ValueError("incoming_probability must be at most one")
        if self.hold_seconds > self.episode_seconds:
            raise ValueError("hold_seconds cannot exceed episode_seconds")
        if self.velocity_radius >= self.velocity_limit:
            raise ValueError("reset velocity radius must be below the velocity limit")
        if self.angle_radius >= self.elbow_limit:
            raise ValueError("reset angle radius must be below the elbow limit")
        if self.shoulder_limit is not None and self.angle_radius >= self.shoulder_limit:
            raise ValueError("reset angle radius must be below the shoulder limit")
        for name in ("dt", "episode_seconds"):
            ratio = getattr(self, name) / self.physics_dt
            if ratio < 1 or not np.isclose(ratio, round(ratio), rtol=0, atol=1e-8):
                raise ValueError(f"{name} must be an integer multiple of physics_dt")


class AcrobotStageAEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(self, config=None, oracle=None, reward=None, *, incoming_states=None):
        self.config = config or StageAConfig()
        self.oracle = oracle or AcrobotOracle()
        self.reward_spec = reward or UprightReward()
        # XML geometry is fixed; reject a mismatched oracle rather than quietly
        # calling the wrong mechanical model an oracle.
        nominal = AcrobotOracle()
        for name in ("a1", "a2", "a3", "b1", "b2"):
            if not np.isclose(getattr(self.oracle, name), getattr(nominal, name)):
                raise ValueError("Stage A MuJoCo geometry requires nominal XK inertias/gravity")
        self.physics = mujoco.Physics.from_xml_string(
            _model_xml(self.oracle.damping, self.oracle.torque_limit), common.ASSETS)
        self.physics.model.opt.timestep = self.config.physics_dt
        self.action_space = gym.spaces.Box(-1., 1., shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
        self.incoming_states = None if incoming_states is None else np.asarray(incoming_states, dtype=np.float64).copy()
        if self.incoming_states is not None:
            if (self.incoming_states.ndim != 2 or self.incoming_states.shape[1] != 4
                    or not len(self.incoming_states) or not np.isfinite(self.incoming_states).all()):
                raise ValueError("incoming_states must be finite nonempty [N,4] q/v")
            if any(self._failed(x) for x in self.incoming_states):
                raise ValueError("incoming states must lie within declared state limits")
        self.failure_rate = -self.reward_spec.cost_bound(
            self.oracle, velocity_limit=self.config.velocity_limit,
            elbow_limit=self.config.elbow_limit, shoulder_limit=self.config.shoulder_limit)
        self.failure_value = self.failure_rate / self.config.discount_rate
        self._done = True

    def sample_qv(self, rng, count=1, *, incoming=None):
        states = np.empty((count, 4))
        states[:, :2] = np.array([np.pi, 0.]) + rng.uniform(-self.config.angle_radius, self.config.angle_radius, (count, 2))
        states[:, 2:] = rng.uniform(-self.config.velocity_radius, self.config.velocity_radius, (count, 2))
        if incoming is True and self.incoming_states is None:
            raise ValueError("incoming reset requested without an incoming dataset")
        if self.incoming_states is not None and incoming is not False:
            mask = np.ones(count, dtype=bool) if incoming is True else rng.random(count) < self.config.incoming_probability
            states[mask] = self.incoming_states[rng.integers(len(self.incoming_states), size=mask.sum())]
        return states

    def canonical(self, qv):
        with torch.no_grad():
            return self.oracle.canonical(torch.as_tensor(qv, dtype=torch.float64)).numpy().astype(np.float32)

    def qv(self):
        result = np.r_[self.physics.data.qpos.copy(), self.physics.data.qvel.copy()]
        result[0] += np.pi / 2
        return result

    def _inside(self, qv):
        errors = np.arctan2(np.sin(qv[:2] - [np.pi, 0.]), np.cos(qv[:2] - [np.pi, 0.]))
        return bool(np.all(np.abs(errors) <= self.config.capture_angle)
                    and np.all(np.abs(qv[2:]) <= self.config.capture_velocity))

    def _failed(self, qv):
        return bool(np.any(np.abs(qv[2:]) >= self.config.velocity_limit)
                    or abs(qv[1]) >= self.config.elbow_limit
                    or (self.config.shoulder_limit is not None
                        and abs(qv[0] - math.pi) >= self.config.shoulder_limit))

    def _rate(self, qv, torque):
        r = self.reward_spec
        return float(-(2 * r.angle1_weight * np.sin((qv[0] - np.pi) / 2)**2
                       + 2 * r.angle2_weight * np.sin(qv[1] / 2)**2
                       + r.velocity1_weight * (qv[2] / r.velocity_scale)**2
                       + r.velocity2_weight * (qv[3] / r.velocity_scale)**2
                       + 0.5 * r.effort_weight * torque**2))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "qv" in options:
            qv = np.asarray(options["qv"], dtype=np.float64)
        else:
            qv = self.sample_qv(self.np_random, incoming=options.get("incoming"))[0]
        if qv.shape != (4,) or not np.isfinite(qv).all() or self._failed(qv):
            raise ValueError("reset qv must be finite shape (4,) and within state limits")
        self.physics.reset()
        with self.physics.reset_context():
            self.physics.data.qpos[:] = qv[:2] - [np.pi / 2, 0.]
            self.physics.data.qvel[:] = qv[2:]
            self.physics.data.ctrl[:] = 0.
        self.elapsed = self.run_seconds = self.max_hold = self.occupancy_seconds = 0.
        self.first_capture = 0. if self._inside(qv) else None
        self._steps = 0
        self._done = False
        return self.canonical(qv), self._info(qv, 0., 0., False)

    def _info(self, qv, duration, torque, failed):
        return {"dt_used": duration, "physical_time": self.elapsed, "qv": qv.copy(),
                "torque": torque, "inside_capture": self._inside(qv),
                "max_hold_seconds": self.max_hold, "terminal_hold_seconds": self.run_seconds,
                "success": self.max_hold + 1e-9 >= self.config.hold_seconds,
                "retained_success": self.run_seconds + 1e-9 >= self.config.hold_seconds,
                "occupancy_seconds": self.occupancy_seconds, "first_capture_seconds": self.first_capture,
                "state_limit_failure": failed, "failure_value": self.failure_value}

    def step(self, action):
        if self._done:
            raise RuntimeError("reset is required before stepping a completed episode")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (1,) or not np.isfinite(action).all():
            raise ValueError("action must be a finite normalized vector of shape (1,)")
        torque = float(np.clip(action[0], -1., 1.) * self.oracle.torque_limit)
        self.physics.data.ctrl[:] = torque / self.oracle.torque_limit
        n = min(round(self.config.dt / self.config.physics_dt),
                round(self.config.episode_seconds / self.config.physics_dt) - self._steps)
        h = self.config.physics_dt
        beta = self.config.discount_rate
        interval_reward = duration = 0.
        qv = self.qv()
        failed = False
        for _ in range(n):
            inside_before = self._inside(qv)
            rate_before = self._rate(qv, torque)
            self.physics.step()
            new_qv = self.qv()
            if not np.isfinite(new_qv).all():
                raise FloatingPointError("non-finite MuJoCo Acrobot state")
            # Discounted trapezoidal quadrature, local to this decision interval.
            interval_reward += 0.5 * h * (math.exp(-beta * duration) * rate_before
                                          + math.exp(-beta * (duration + h)) * self._rate(new_qv, torque))
            duration += h
            self._steps += 1
            self.elapsed = self._steps * h
            inside = self._inside(new_qv)
            if inside and inside_before:
                self.run_seconds += h
                self.occupancy_seconds += h
            else:
                self.run_seconds = 0.
            self.max_hold = max(self.max_hold, self.run_seconds)
            if inside and self.first_capture is None:
                self.first_capture = self.elapsed
            qv = new_qv
            failed = self._failed(qv)
            if failed:
                # Absorbing failure has a pessimistic continuing reward, rather
                # than rewarding early exit from this negative-reward task.
                interval_reward += math.exp(-beta * duration) * self.failure_value
                break
        truncated = not failed and self._steps >= round(self.config.episode_seconds / h)
        self._done = bool(failed or truncated)
        return self.canonical(qv), float(interval_reward), failed, truncated, self._info(qv, duration, torque, failed)

    def render(self):
        return self.physics.render(height=480, width=640, camera_id=0)

    def close(self):
        self.physics.free()
