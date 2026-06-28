# ctrllib/env/dmc.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from dm_control import suite, rl
from dm_env import specs as dm_specs

from .base import ContinuousEnv


# --------------------------- Spec & Obs helpers ---------------------------


def _bounded_array_to_box(spec: dm_specs.BoundedArray) -> spaces.Box:
    low = np.full(spec.shape, spec.minimum, dtype=np.float32)
    high = np.full(spec.shape, spec.maximum, dtype=np.float32)
    return spaces.Box(low=low, high=high, dtype=np.float32)


def _array_to_box(spec: dm_specs.Array) -> spaces.Box:
    low = np.full(spec.shape, -np.inf, dtype=np.float32)
    high = np.full(spec.shape, np.inf, dtype=np.float32)
    return spaces.Box(low=low, high=high, dtype=np.float32)


def _spec_to_box(spec) -> spaces.Space:
    if isinstance(spec, dm_specs.BoundedArray):
        return _bounded_array_to_box(spec)
    if isinstance(spec, dm_specs.Array):
        return _array_to_box(spec)
    if isinstance(spec, dict):
        return spaces.Dict({k: _spec_to_box(v) for k, v in spec.items()})
    raise TypeError(f"Unsupported spec type: {type(spec)}")


def _flatten_obs(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """Flatten dict observations deterministically (sorted keys)."""
    if isinstance(obs, dict):
        parts = []
        for k in sorted(obs.keys()):
            v = np.asarray(obs[k], dtype=np.float32).ravel()
            parts.append(v)
        return (
            np.concatenate(parts, axis=0).astype(np.float32)
            if parts
            else np.zeros((0,), dtype=np.float32)
        )
    return np.asarray(obs, dtype=np.float32)


# --------------------------- Continuous dm_control Env ---------------------------


class DMCContinuousEnv(ContinuousEnv):
    """Continuous-time dm_control environment.

    - Wraps `dm_control.suite.load(domain_name, task_name, ...)`
    - Provides both:
        -) continuous-time API via `step_dt`
        -) Gym-compatible API via `step`
    - Uses MuJoCo's physics timestep and substeps to approximate requested `dt`.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        domain_name: str,
        task_name: str,
        *,
        # dm_control configuration
        seed: Optional[int] = None,
        flat_observation: bool = True,
        task_kwargs: Optional[Dict[str, Any]] = None,
        environment_kwargs: Optional[Dict[str, Any]] = None,
        # Continuous-time (superclass) configuration
        time_sampling: str = "uniform",  # "uniform" or "irregular"
        dt: float = 0.02,
        physics_dt: Optional[float] = None,  # override MuJoCo physics timestep
        min_dt: Optional[float] = None,
        max_dt: Optional[float] = None,
        max_steps: Optional[int] = None,
        episode_duration: Optional[float] = None,
        time_sampling_kwargs: Optional[
            Dict[str, Any]
        ] = None,  # Keyword arguments for time sampling ("irregular")
        return_reward_increment: bool = False,
    ) -> None:
        # Initialize ContinuousEnv (time grid, dt sampling, etc.)
        super().__init__(
            time_sampling=time_sampling,
            dt=dt,
            physics_dt=physics_dt,
            min_dt=min_dt,
            max_dt=max_dt,
            max_steps=max_steps,
            episode_duration=episode_duration,
            time_sampling_kwargs=time_sampling_kwargs,
            return_reward_increment=return_reward_increment,
        )

        # --- Build dm_control env ---
        task_kwargs = dict(task_kwargs or {})
        if seed is not None:
            task_kwargs["random"] = seed

        environment_kwargs = dict(environment_kwargs or {})
        environment_kwargs.setdefault("flat_observation", flat_observation)

        self._env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
        )

        if hasattr(self._env, "_step_limit"):
            if max_steps is not None:
                self._env._step_limit = int(max_steps)
            else:
                self._env._step_limit = int(1e9)

        # Domain/task identifiers (used by model-based dynamics helpers)
        self.domain_name = domain_name
        self.task_name = task_name

        # Keep default dt for algorithm's time conversion
        self.dt_default = self._env.control_timestep()

        if physics_dt is not None:
            if physics_dt <= 0.0:
                raise ValueError(f"physics_dt must be > 0, got {physics_dt}")
            self._env.physics.model.opt.timestep = float(physics_dt)

        # Keep track of elapsed time for diagnostics (separate from cur_t)
        self._elapsed_time: float = 0.0

        # Action / observation spaces
        self._action_spec = self._env.action_spec()
        self._obs_spec = self._env.observation_spec()

        # Last observation for physics error handling
        self._last_obs_dmc: Optional[np.ndarray] = None

        self.action_space = _spec_to_box(self._action_spec)

        # Build a flat observation Box by concatenating pieces deterministically
        if isinstance(self._obs_spec, dict):
            lows, highs = [], []
            for k in sorted(self._obs_spec.keys()):
                s = self._obs_spec[k]
                b = _spec_to_box(s)
                lows.append(b.low.ravel())
                highs.append(b.high.ravel())
            low = (
                np.concatenate(lows).astype(np.float32)
                if lows
                else np.zeros((0,), dtype=np.float32)
            )
            high = (
                np.concatenate(highs).astype(np.float32)
                if highs
                else np.zeros((0,), dtype=np.float32)
            )
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            self.observation_space = _spec_to_box(self._obs_spec)

    def render(
        self,
        mode: str = "human",
        height: int = 480,
        width: int = 640,
        camera_id: int = 0,
    ) -> Optional[np.ndarray]:
        """Renders the environment.

        By convention, if mode is:
        - human: render to the current display or terminal and
          return None.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image.
        """
        if mode == "rgb_array":
            return self._env.physics.render(
                height=height, width=width, camera_id=camera_id
            )
        elif mode == "human":
            # dm_control does not have a built-in 'human' render mode that opens a window.
            # For now, we'll just return the rgb_array. A custom viewer would be needed.
            return self._env.physics.render(
                height=height, width=width, camera_id=camera_id
            )
        return None

    # --------------------------- Physics timing helpers ---------------------------

    def _get_physics_dt(self) -> float:
        """MuJoCo physics substep (seconds)."""
        return float(self._env.physics.model.opt.timestep)

    def _get_control_dt(self) -> float:
        """Control timestep (seconds) used by the Task/Environment."""
        if hasattr(self._env, "control_timestep"):
            return float(self._env.control_timestep())
        if hasattr(self._env, "_control_timestep"):
            return float(self._env._control_timestep)
        if hasattr(self._env, "_n_sub_steps"):
            return float(self._get_physics_dt() * int(self._env._n_sub_steps))
        raise RuntimeError(
            "Cannot determine control timestep from dm_control env version."
        )

    def _set_control_dt_for_one_step(self, new_dt: float):
        """Temporarily set control timestep & substeps to approximate `new_dt`."""
        phys_dt = self._get_physics_dt()
        nsub = max(1, int(round(new_dt / phys_dt)))

        restore_dt = None
        restore_nsub = None

        if hasattr(self._env, "_control_timestep"):
            restore_dt = float(self._env._control_timestep)
            self._env._control_timestep = nsub * phys_dt

        if hasattr(self._env, "_n_sub_steps"):
            restore_nsub = int(self._env._n_sub_steps)
            self._env._n_sub_steps = nsub

        return restore_dt, restore_nsub, nsub * phys_dt  # actual dt used

    def _restore_control_dt(
        self, restore_dt: Optional[float], restore_nsub: Optional[int]
    ):
        if restore_dt is not None and hasattr(self._env, "_control_timestep"):
            self._env._control_timestep = float(restore_dt)
        if restore_nsub is not None and hasattr(self._env, "_n_sub_steps"):
            self._env._n_sub_steps = int(restore_nsub)

    # --------------------------- ContinuousEnv abstract method implementations ---------------------------

    def _reset_physics(
        self,
        *,
        seed: Optional[int],
        options: Optional[Dict[str, Any]],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # dm_control uses its own random seeding at construction; reset() ignores seed here.
        del seed, options  # unused for now
        ts = self._env.reset()
        self._elapsed_time = 0.0
        obs = _flatten_obs(ts.observation)
        self._last_obs_dmc = obs
        info: Dict[str, Any] = {}
        return obs, info

    def _step_physics(
        self,
        action: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any], float]:
        """Advance env by (approximately) `dt` seconds using dm_control's step."""
        assert dt > 0.0, "dt must be > 0 in DMCContinuousEnv._step_physics"

        restore_dt, restore_nsub, actual_dt = self._set_control_dt_for_one_step(dt)
        try:
            try:
                ts = self._env.step(np.asarray(action, dtype=np.float32))
            finally:
                self._restore_control_dt(
                    restore_dt, restore_nsub
                )  # TODO: Turn-off restore dt when not needed
        except rl.control.PhysicsError:
            obs = (
                self._last_obs_dmc
                if self._last_obs_dmc is not None
                else np.zeros(self.observation_space.shape, dtype=np.float32)
            )

            # Return 0 for actual_dt since the step was not successful
            reward = 0.0
            terminated = True
            truncated = False
            info: Dict[str, Any] = {"physics_error": True}

            return obs, reward, terminated, truncated, info, float(actual_dt)

        obs = _flatten_obs(ts.observation)
        self._last_obs_dmc = obs
        reward = float(ts.reward if ts.reward is not None else 0.0)

        is_last = bool(ts.last())
        terminated = is_last
        truncated = False  # Truncation is handled by the ContinuousEnv wrapper

        # Track elapsed time separately for diagnostics
        self._elapsed_time += float(actual_dt)

        info: Dict[str, Any] = {
            "elapsed_time": self._elapsed_time,
            "discount": ts.discount,
            "dt_requested": float(dt),
            "dt_used": float(actual_dt),
        }

        return obs, reward, terminated, truncated, info, float(actual_dt)

    # --------------------------- Convenience properties ---------------------------

    @property
    def control_dt(self) -> float:
        """Current control timestep (seconds)."""
        return self._get_control_dt()

    @property
    def physics_dt(self) -> float:
        """Physics solver timestep (seconds)."""
        return self._get_physics_dt()

    # --------------------------- Model-based dynamics (drift) ---------------------------

    def dynamics_terms(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Analytic observation-space drift ``b(obs, a) = d(obs)/dt`` from MuJoCo.

        Used by the model-based generator in CT-SAC (``dynamics_source="mujoco"``).

        Currently supports the ``cheetah`` domain, whose observation is
        ``[qpos[1:] (nq-1), qvel (nv)]`` with index-aligned planar coordinates,
        so the drift is exact without an observation Jacobian:

            d/dt obs[:nq-1] = qvel[1:nq]   (positions)
            d/dt obs[nq-1:] = qacc         (velocities; MuJoCo forward dynamics)

        The live physics state is snapshotted and restored, so this is safe to
        call during training. Computation loops over the batch on CPU.
        """
        if self.domain_name != "cheetah":
            raise NotImplementedError(
                "dynamics_terms currently supports the 'cheetah' domain only "
                "(obs = [qpos[1:], qvel]); other domains need their own obs<->state map."
            )

        if hasattr(obs, "detach"):
            obs = obs.detach().cpu().numpy()
        if hasattr(action, "detach"):
            action = action.detach().cpu().numpy()

        obs_dim = int(self.observation_space.shape[0])
        obs = np.asarray(obs, dtype=np.float64).reshape(-1, obs_dim)
        action = np.asarray(action, dtype=np.float64).reshape(obs.shape[0], -1)

        physics = self._env.physics
        data = physics.data
        nq = int(physics.model.nq)
        nv = int(physics.model.nv)
        assert obs_dim == (nq - 1) + nv, (
            f"cheetah obs_dim {obs_dim} != (nq-1)+nv = {(nq - 1) + nv}"
        )

        low = np.asarray(self.action_space.low, dtype=np.float64)
        high = np.asarray(self.action_space.high, dtype=np.float64)

        saved = (
            data.qpos.copy(),
            data.qvel.copy(),
            data.ctrl.copy(),
            float(data.time),
        )
        out = np.zeros((obs.shape[0], obs_dim), dtype=np.float32)
        try:
            for i in range(obs.shape[0]):
                qpos = np.zeros(nq, dtype=np.float64)
                qpos[1:] = obs[i, : nq - 1]
                qvel = obs[i, nq - 1 :].astype(np.float64)
                data.qpos[:] = qpos
                data.qvel[:] = qvel
                data.ctrl[:] = np.clip(action[i], low, high)
                physics.forward()
                out[i, : nq - 1] = np.asarray(data.qvel[1:nq])
                out[i, nq - 1 :] = np.asarray(data.qacc[:nv])
        finally:
            data.qpos[:] = saved[0]
            data.qvel[:] = saved[1]
            data.ctrl[:] = saved[2]
            data.time = saved[3]
            physics.forward()
        return out
