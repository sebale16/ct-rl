# ctrllib/env/dmc.py
from __future__ import annotations

import os
import threading
import weakref
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Initialize PyTorch's compiler subsystem BEFORE dm_control/MuJoCo is loaded.
# On torch 2.12.0+cu130, importing ``torch._dynamo`` *after* MuJoCo's shared
# libraries are resident segfaults (a BLAS/threading symbol conflict during
# dynamo's C-extension init). That lazy import is triggered by both
# ``torch.optim.Adam.__init__`` (via ``torch._compile``) and ``torch.func``
# (functorch), so any structured/learned CT-SAC run built after the env would
# crash at optimizer construction or the first mass-matrix Jacobian. Forcing the
# import here (env is imported before the algorithm/model everywhere) makes the
# later lazy imports no-ops, so both paths work. Harmless on torch builds
# without the conflict.
import torch as _torch  # noqa: F401
import torch._dynamo  # noqa: F401

from dm_control import suite, rl
from dm_env import specs as dm_specs

from .base import ContinuousEnv


_DRIFT_ROLLOUT_ENVS: "weakref.WeakSet[DMCContinuousEnv]" = weakref.WeakSet()


def _close_drift_rollouts_before_fork() -> None:
    """Shut native worker pools down while their threads still exist.

    A C++ ``mujoco.rollout.Rollout`` inherited across ``fork()`` cannot be used
    or destroyed safely: the child has a copy of the pool bookkeeping but none
    of its worker threads.  Closing active pools before the fork lets both the
    parent and child lazily create a process-local pool on their next drift call.
    """
    for env in list(_DRIFT_ROLLOUT_ENVS):
        env._close_drift_rollout()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=_close_drift_rollouts_before_fork)


def _default_drift_rollout_threads() -> int:
    """Choose a conservative pool size within the current CPU allocation."""
    limits = [8]
    try:
        affinity = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        affinity = None
    if affinity:
        limits.append(len(affinity))
    else:
        cpu_count = os.cpu_count()
        if cpu_count:
            limits.append(int(cpu_count))

    # Slurm installations do not universally constrain sched_getaffinity, so
    # also honor the requested CPUs when the scheduler exposes them directly.
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            limits.append(int(slurm_cpus.split("(", 1)[0]))
        except ValueError:
            pass
    return max(1, min(limits))


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
        raw_state_obs: bool = False,
        drift_backend: str = "auto",
        drift_rollout_threads: Optional[int] = None,
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

        # Validate the drift controls before constructing the comparatively
        # expensive dm_control environment. ``None`` sizes the native pool to
        # the current CPU allocation, capped at eight workers.
        drift_backend = str(drift_backend).strip().lower()
        if drift_backend not in ("auto", "rollout", "loop"):
            raise ValueError(
                f"drift_backend must be 'auto', 'rollout' or 'loop', "
                f"got {drift_backend!r}"
            )
        if drift_rollout_threads is None:
            drift_rollout_threads = _default_drift_rollout_threads()
        self.drift_backend = drift_backend
        self.drift_rollout_threads = int(drift_rollout_threads)
        if self.drift_rollout_threads < 1:
            raise ValueError(
                "drift_rollout_threads must be >= 1, got "
                f"{drift_rollout_threads!r}"
            )

        # Lazily built on first rollout-backend call:
        # (Rollout thread pool, private model copy, per-thread MjData list).
        self._drift_rollout: Optional[tuple] = None
        self._drift_rollout_pid: Optional[int] = None
        self._drift_rollout_lock = threading.RLock()
        # Register before a pool can be constructed. The pre-fork hook acquires
        # every live env's lock, so a fork cannot capture either a half-built
        # native pool or an RLock owned by a thread that will vanish in the child.
        _DRIFT_ROLLOUT_ENVS.add(self)

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

        # Raw-state observations: obs = [qpos (nq); qvel (nv)] read straight from
        # the physics, replacing the task's (often cos/sin-encoded) observation.
        # This is what the structured port-Hamiltonian model and the oracle drift
        # need: generalized coordinates whose position block satisfies
        # d(qpos)/dt = qvel exactly, which requires hinge/slide joints only
        # (nq == nv; quaternion/free joints pack qpos differently).
        self.raw_state_obs = str(raw_state_obs).strip().lower() in ("1", "true", "yes")
        if self.raw_state_obs:
            nq = int(self._env.physics.model.nq)
            nv = int(self._env.physics.model.nv)
            if nq != nv:
                raise ValueError(
                    f"raw_state_obs requires nq == nv (hinge/slide joints only); "
                    f"{domain_name} has nq={nq}, nv={nv}."
                )
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(nq + nv,), dtype=np.float32
            )

    def close(self) -> None:
        with self._drift_rollout_lock:
            try:
                self._close_drift_rollout()
                self._env.close()
                super().close()
            finally:
                _DRIFT_ROLLOUT_ENVS.discard(self)

    def __getstate__(self) -> dict:
        """Exclude process-local native rollout resources from serialization."""
        state = self.__dict__.copy()
        state["_drift_rollout"] = None
        state["_drift_rollout_pid"] = None
        state.pop("_drift_rollout_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._drift_rollout = None
        self._drift_rollout_pid = None
        self._drift_rollout_lock = threading.RLock()
        _DRIFT_ROLLOUT_ENVS.add(self)

    def _close_drift_rollout(self) -> None:
        """Close and forget this process's lazily-created rollout pool."""
        with self._drift_rollout_lock:
            rollout_state = self._drift_rollout
            if rollout_state is None:
                self._drift_rollout_pid = None
                return
            if self._drift_rollout_pid != os.getpid():
                raise RuntimeError(
                    "cannot close a mujoco rollout pool inherited from another "
                    "process; create or serialize the environment before forking"
                )
            try:
                rollout_state[0].close()
            finally:
                self._drift_rollout = None
                self._drift_rollout_pid = None

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

    def _raw_obs(self) -> np.ndarray:
        """Raw generalized state [qpos; qvel] from the live physics."""
        data = self._env.physics.data
        return np.concatenate(
            [np.asarray(data.qpos, dtype=np.float32),
             np.asarray(data.qvel, dtype=np.float32)]
        )

    def _reset_physics(
        self,
        *,
        seed: Optional[int],
        options: Optional[Dict[str, Any]],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # dm_control uses its own random seeding at construction; reset() ignores seed here.
        del seed, options  # unused for now
        with self._drift_rollout_lock:
            # Some suite tasks mutate dynamics-bearing model arrays on reset
            # (point_mass-hard randomizes actuator directions, for example).
            # A cached private model must never cross that mutation boundary.
            self._close_drift_rollout()
            ts = self._env.reset()
            self._elapsed_time = 0.0
            obs = self._raw_obs() if self.raw_state_obs else _flatten_obs(ts.observation)
            self._last_obs_dmc = obs
            info: Dict[str, Any] = {}
            return obs, info

    def _step_physics(
        self,
        action: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any], float]:
        """Advance env by (approximately) `dt` seconds using dm_control's step."""
        with self._drift_rollout_lock:
            return self._step_physics_unlocked(action, dt)

    def _step_physics_unlocked(
        self,
        action: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any], float]:
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

        obs = self._raw_obs() if self.raw_state_obs else _flatten_obs(ts.observation)
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

        Supported observation maps, both index-aligned so the drift is exact
        without an observation Jacobian:

        - ``raw_state_obs=True`` (any hinge/slide domain, nq == nv):
          obs = [qpos (nq); qvel (nv)],
          d/dt obs[:nq] = qvel, d/dt obs[nq:] = qacc.
        - the ``cheetah`` task observation, obs = [qpos[1:] (nq-1); qvel (nv)]:
          d/dt obs[:nq-1] = qvel[1:nq], d/dt obs[nq-1:] = qacc.

        Two CPU backends compute the batch (``drift_backend`` constructor
        kwarg), both deterministic and both leaving the live physics state
        unchanged:

        - ``"rollout"``: one ``mujoco.rollout`` call steps the whole batch in
          C across ``drift_rollout_threads`` worker threads, and qacc is
          recovered from the Euler velocity update ``(qvel' - qvel)/h``. The
          rollout runs on a private model copy forced to the explicit Euler
          integrator (``eulerdamp`` and unstable-state autoreset disabled); these
          settings change only the stepping rule, so the recovered qacc equals
          what ``physics.forward()`` reads under any live integrator. Live
          actuator state, applied forces, equality/mocap/user context, solver
          warm-start, and time are copied into the private rollout.
        - ``"loop"``: the historical per-sample loop, one
          ``physics.forward()`` per row with the live state snapshotted and
          restored.
        - ``"auto"`` (default): ``"rollout"`` when supported, else ``"loop"``.
        """
        with self._drift_rollout_lock:
            return self._dynamics_terms_unlocked(obs, action)

    def _dynamics_terms_unlocked(
        self, obs: np.ndarray, action: np.ndarray
    ) -> np.ndarray:
        if not self.raw_state_obs and self.domain_name != "cheetah":
            raise NotImplementedError(
                "dynamics_terms supports raw_state_obs=True (any hinge/slide "
                "domain) or the 'cheetah' task observation; other domains need "
                "their own obs<->state map."
            )

        if hasattr(obs, "detach"):
            obs = obs.detach().cpu().numpy()
        if hasattr(action, "detach"):
            action = action.detach().cpu().numpy()

        obs_dim = int(self.observation_space.shape[0])
        obs = np.asarray(obs, dtype=np.float64).reshape(-1, obs_dim)
        action = np.asarray(action, dtype=np.float64).reshape(obs.shape[0], -1)

        nq = int(self._env.physics.model.nq)
        nv = int(self._env.physics.model.nv)
        if self.raw_state_obs:
            assert obs_dim == nq + nv, f"raw obs_dim {obs_dim} != nq+nv = {nq + nv}"
            pos_width = nq  # obs = [qpos; qvel]
        else:
            assert obs_dim == (nq - 1) + nv, (
                f"cheetah obs_dim {obs_dim} != (nq-1)+nv = {(nq - 1) + nv}"
            )
            pos_width = nq - 1  # obs = [qpos[1:]; qvel], root x dropped

        low = np.asarray(self.action_space.low, dtype=np.float64)
        high = np.asarray(self.action_space.high, dtype=np.float64)
        action = np.clip(action, low, high)

        backend = self.drift_backend
        if backend == "auto":
            backend = "rollout" if self._drift_rollout_supported() else "loop"
        elif backend == "rollout" and not self._drift_rollout_supported():
            raise RuntimeError(
                "drift_backend='rollout' needs the mujoco.rollout module; "
                "use drift_backend='loop' or 'auto'."
            )
        if backend == "rollout":
            return self._dynamics_terms_rollout(obs, action, pos_width)
        return self._dynamics_terms_loop(obs, action, pos_width)

    def _dynamics_terms_loop(
        self, obs: np.ndarray, action: np.ndarray, pos_width: int
    ) -> np.ndarray:
        """Per-sample drift via ``physics.forward()`` on the live physics."""
        physics = self._env.physics
        data = physics.data
        nq = int(physics.model.nq)
        nv = int(physics.model.nv)
        obs_dim = obs.shape[1]

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
                qpos[nq - pos_width:] = obs[i, :pos_width]
                qvel = obs[i, pos_width:].astype(np.float64)
                data.qpos[:] = qpos
                data.qvel[:] = qvel
                data.ctrl[:] = action[i]
                physics.forward()
                out[i, :pos_width] = np.asarray(data.qvel[nq - pos_width:nq])
                out[i, pos_width:] = np.asarray(data.qacc[:nv])
        finally:
            data.qpos[:] = saved[0]
            data.qvel[:] = saved[1]
            data.ctrl[:] = saved[2]
            data.time = saved[3]
            physics.forward()
        return out

    def _drift_rollout_supported(self) -> bool:
        try:
            import mujoco.rollout as rollout_module
        except (ImportError, OSError):
            return False
        return hasattr(rollout_module, "Rollout")

    def _ensure_drift_rollout(self) -> tuple:
        import copy

        import mujoco
        import mujoco.rollout

        if (
            self._drift_rollout is not None
            and self._drift_rollout_pid != os.getpid()
        ):
            # Never destroy an inherited native pool here: its worker threads
            # vanished at fork and its destructor would block trying to join
            # them. The registered pre-fork hook prevents this path for Python
            # forks; the guard turns any unregistered external fork into a clear
            # error instead of a native deadlock.
            raise RuntimeError(
                "mujoco rollout pool was inherited from another process; "
                "construct or serialize the environment before forking"
            )

        if self._drift_rollout is None:
            model = copy.copy(self._env.physics.model.ptr)
            # qacc is recovered below by inverting the explicit-Euler velocity
            # update qvel' = qvel + h*qacc, so the private copy must integrate
            # with exactly that rule: force Euler (qacc from mj_forward does
            # not depend on the integrator, so RK4/implicit domains still get
            # their correct drift) and disable implicit joint damping
            # (eulerdamp), which would fold D into the update. Also disable
            # mj_step's unstable-state autoreset: mj_forward returns the large
            # acceleration to its caller, whereas autoreset would silently
            # replace qvel and make the reconstructed acceleration meaningless.
            model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
            model.opt.disableflags |= (
                mujoco.mjtDisableBit.mjDSBL_EULERDAMP
                | mujoco.mjtDisableBit.mjDSBL_AUTORESET
            )
            nthread = self.drift_rollout_threads
            rollout = mujoco.rollout.Rollout(nthread=nthread)
            try:
                datas = [mujoco.MjData(model) for _ in range(nthread)]
            except Exception:
                rollout.close()
                raise
            self._drift_rollout = (rollout, model, datas)
            self._drift_rollout_pid = os.getpid()
        rollout, model, datas = self._drift_rollout
        model.opt.timestep = float(self._env.physics.model.opt.timestep)
        return rollout, model, datas

    def _dynamics_terms_rollout(
        self, obs: np.ndarray, action: np.ndarray, pos_width: int
    ) -> np.ndarray:
        """Batched drift via a one-step ``mujoco.rollout`` on a model copy."""
        import mujoco

        rollout, model, datas = self._ensure_drift_rollout()
        nq = int(model.nq)
        nv = int(model.nv)
        batch = obs.shape[0]

        # Snapshot every integration input once, then split it into the three
        # channels accepted by mujoco.rollout. FULLPHYSICS also contains model-
        # dependent history/plugin state, so derive sizes through MuJoCo rather
        # than assuming the legacy [time;qpos;qvel;act] width.
        live_model = self._env.physics.model.ptr
        live_data = self._env.physics.data.ptr
        if (int(live_model.nq), int(live_model.nv), int(live_model.nu)) != (
            nq,
            nv,
            int(model.nu),
        ):
            raise RuntimeError("live and private MuJoCo model dimensions differ")
        integration_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        integration = np.empty(
            mujoco.mj_stateSize(live_model, integration_spec), dtype=np.float64
        )
        mujoco.mj_getState(live_model, live_data, integration, integration_spec)

        def extract_state(spec) -> np.ndarray:
            value = np.empty(mujoco.mj_stateSize(live_model, spec), dtype=np.float64)
            if hasattr(mujoco, "mj_extractState"):
                mujoco.mj_extractState(
                    live_model, integration, integration_spec, value, spec
                )
            else:
                # Compatibility with MuJoCo versions that expose Rollout but
                # predate mj_extractState.
                mujoco.mj_getState(live_model, live_data, value, spec)
            return value

        def state_offset(signature, field) -> int:
            # State fields are serialized in ascending mjtState-bit order.
            preceding = int(signature) & (int(field) - 1)
            return int(mujoco.mj_stateSize(live_model, preceding))

        full_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        base_state = extract_state(full_spec)
        if mujoco.mj_stateSize(model, full_spec) != base_state.size:
            raise RuntimeError("live and private MuJoCo model state signatures differ")
        state = np.broadcast_to(base_state, (batch, base_state.size)).copy()
        # Historical cheetah behavior fixes the omitted root translation to 0.
        # Raw observations replace every qpos entry.
        qpos_adr = state_offset(full_spec, mujoco.mjtState.mjSTATE_QPOS)
        qvel_adr = state_offset(full_spec, mujoco.mjtState.mjSTATE_QVEL)
        state[:, qpos_adr : qpos_adr + nq] = 0.0
        state[
            :, qpos_adr + (nq - pos_width) : qpos_adr + nq
        ] = obs[:, :pos_width]
        state[:, qvel_adr : qvel_adr + nv] = obs[:, pos_width:]

        # CTRL is the first component of mjSTATE_USER. All remaining components
        # retain the live qfrc/xfrc, equality, mocap, and userdata context.
        control_spec = mujoco.mjtState.mjSTATE_USER
        base_control = extract_state(control_spec)
        if mujoco.mj_stateSize(model, control_spec) != base_control.size:
            raise RuntimeError("live and private MuJoCo user-state signatures differ")
        control = np.broadcast_to(
            base_control, (batch, 1, base_control.size)
        ).copy()
        ctrl_adr = state_offset(control_spec, mujoco.mjtState.mjSTATE_CTRL)
        control[:, 0, ctrl_adr : ctrl_adr + int(model.nu)] = action

        base_warmstart = extract_state(mujoco.mjtState.mjSTATE_WARMSTART)
        if int(model.nv) != base_warmstart.size:
            raise RuntimeError("live and private MuJoCo warm-start signatures differ")
        initial_warmstart = np.broadcast_to(
            base_warmstart, (batch, base_warmstart.size)
        ).copy()

        next_state, _ = rollout.rollout(
            model,
            datas,
            state,
            control=control,
            control_spec=control_spec,
            initial_warmstart=initial_warmstart,
        )

        qvel = state[:, qvel_adr : qvel_adr + nv]
        qvel_next = next_state[:, 0, qvel_adr : qvel_adr + nv]
        qacc = (qvel_next - qvel) / float(model.opt.timestep)

        out = np.empty((batch, obs.shape[1]), dtype=np.float32)
        out[:, :pos_width] = qvel[:, nv - pos_width :]
        out[:, pos_width:] = qacc
        return out
