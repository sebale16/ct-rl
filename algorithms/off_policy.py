# algorithms/off_policy.py
from .base import BaseAlgorithm
from abc import ABC, abstractmethod
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from typing import Type

import numpy as np
import torch as th

from environment.base import ContinuousEnv
from models.base import Model
from common.buffers import ReplayBuffer
from common.schedules import Schedule
from common.utils import get_obs_shape, get_action_dim
from models.noise import ActionNoise, VectorizedActionNoise
from common.callbacks import BaseCallback


def _realized_simulated_seconds(
    t: Union[float, np.ndarray],
    next_t: Union[float, np.ndarray],
    done: Union[bool, np.ndarray],
    infos: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]],
) -> float:
    """Return the physical duration represented by one vector rollout step.

    ``VecContinuousEnv`` auto-resets completed slots, so its returned
    ``next_t`` is zero for those slots.  The actual terminal episode time is
    preserved in ``info["terminal_next_t"]`` and must be restored before
    subtracting the local episode start time.  Durations are summed across
    environment slots, just as ``num_timesteps`` counts every slot.
    """
    start = np.asarray(t, dtype=np.float64).reshape(-1)
    end = np.asarray(next_t, dtype=np.float64).reshape(-1).copy()
    dones = np.asarray(done, dtype=bool).reshape(-1)
    if start.size != end.size or start.size != dones.size:
        raise ValueError(
            "t, next_t, and done must contain one value per environment slot, "
            f"got {start.size}, {end.size}, and {dones.size}"
        )

    if isinstance(infos, (list, tuple)):
        info_items = infos
    elif isinstance(infos, dict):
        info_items = (infos,)
    else:
        info_items = ()
    for index, info in enumerate(info_items):
        if index >= end.size:
            break
        if dones[index] and isinstance(info, dict) and "terminal_next_t" in info:
            end[index] = float(info["terminal_next_t"])

    durations = end - start
    if not np.all(np.isfinite(durations)):
        raise ValueError("environment timestamps must produce finite durations")
    # Tolerate only roundoff-scale negative values. A materially negative
    # duration indicates a broken timestamp/auto-reset contract and must not
    # silently corrupt the training-time metric.
    tolerance = 1e-9
    if np.any(durations < -tolerance):
        raise ValueError(
            "environment timestamps must be non-decreasing within each episode; "
            f"got durations {durations.tolist()}"
        )
    return float(np.maximum(durations, 0.0).sum())


class OffPolicyAlgorithm(BaseAlgorithm, ABC):
    """
    Base class for off-policy CT algorithms using a ReplayBuffer.

    while num_timesteps < total_timesteps:
        1) collect one transition with current policy
        2) if enough data & at train_freq, run `train(gradient_steps, batch_size)`
    """

    def __init__(
        self,
        env: ContinuousEnv,
        model: Union[Model, str, Type[Model]],
        model_kwargs: Optional[Dict[str, Any]] = None,
        device: Union[str, th.device] = "auto",
        seed: Optional[int] = None,
        # Off-policy specific:
        gamma: float = 0.99,
        buffer_size: int = 1_000_000,
        learning_rate: Union[float, Schedule] = 3e-4,
        batch_size: int = 256,
        train_freq: int = 1,
        gradient_steps: int = 1,
        learning_starts: int = 100,
        action_noise: Optional[ActionNoise] = None,
    ) -> None:
        super().__init__(
            env=env,
            model=model,
            model_kwargs=model_kwargs,
            learning_rate=learning_rate,
            device=device,
            seed=seed,
        )

        self.gamma = float(gamma)
        self.beta = -math.log(self.gamma)
        self.buffer_size = int(buffer_size)
        self.batch_size = int(batch_size)
        self.train_freq = int(train_freq)
        self.gradient_steps = int(gradient_steps)
        self.learning_starts = int(learning_starts)
        self.action_noise = action_noise

        self.obs_shape = get_obs_shape(self.env.observation_space)
        self.action_dim = get_action_dim(self.env.action_space)

        # ReplayBuffer; allow more than 1 n_envs
        self.replay_buffer = ReplayBuffer(
            buffer_size,
            self.env.observation_space,
            self.env.action_space,
            device=self.device,
            n_envs=self.n_envs,
        )

        # Last observation from the environment
        self._last_obs: Optional[np.ndarray] = None

        # Setup vectorized action noise
        if (
            self.action_noise is not None
            and self.n_envs > 1
            and not isinstance(self.action_noise, VectorizedActionNoise)
        ):
            self.action_noise = VectorizedActionNoise(self.action_noise, self.n_envs)

    # ---------------------- Abstract methods ----------------------

    @abstractmethod
    def _policy_act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Given a single observation (np.ndarray), return action (np.ndarray).
        Implement this in concrete algorithms using the correct model:
          - CoupledVqModel.act(...)
          - ActorQCriticModel.act(...)
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, gradient_steps: int, batch_size: int) -> None:
        """
        Sample from replay buffer and update model parameters.
        Use `self.replay_buffer.sample(batch_size)` which returns ReplayBatch.
        """
        raise NotImplementedError

    # ----------------------- Rollout Logic -----------------------

    def _sample_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Exploration policy: random before learning_starts, then policy (+ noise if any).

        Returns:
        - (act_dim,) if single env
        - (n_envs, act_dim) if env is vectorized
        """
        is_vec_env = self.is_vec_env
        n_envs = self.n_envs

        obs_arr = np.asarray(obs, dtype=np.float32)
        if is_vec_env:
            # ensure obs is (n_envs, obs_dim)
            if obs_arr.ndim == 1:
                obs_arr = obs_arr[None, :]
            assert (
                obs_arr.shape[0] == n_envs
            ), f"obs first dim must be n_envs={n_envs}, got {obs_arr.shape}"
        else:
            # ensure obs is (obs_dim,)
            if obs_arr.ndim > 1:
                obs_arr = obs_arr[0]

        # Sample action
        if self.num_timesteps < self.learning_starts:
            if is_vec_env:
                action = np.stack(
                    [self.env.action_space.sample() for _ in range(n_envs)], axis=0
                ).astype(np.float32)
            else:
                action = np.asarray(self.env.action_space.sample(), dtype=np.float32)
        else:
            action = self._policy_act(obs_arr, deterministic=False)
            action = np.asarray(action, dtype=np.float32)

            # add exploration noise
            if self.action_noise is not None:
                noise = self.action_noise()
                noise = np.asarray(noise, dtype=np.float32)
                action = action + noise

                # clip
                action = np.clip(
                    action, self.env.action_space.low, self.env.action_space.high
                )

        return action

    def _store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: Union[float, np.ndarray],
        done: Union[bool, np.ndarray],
        next_obs: np.ndarray,
        t: Union[float, np.ndarray],
        next_t: Union[float, np.ndarray],
        infos: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]],
        *,
        terminated: Optional[Union[bool, np.ndarray]] = None,
        truncated: Optional[Union[bool, np.ndarray]] = None,
    ) -> None:
        """
        Wraps data to suitable arrays and calls ReplayBuffer.add().
        """
        # ensure batched
        if obs.ndim == 1:
            obs = obs[None, :]
        if next_obs.ndim == 1:
            next_obs = next_obs[None, :]
        if action.ndim == 1:
            action = action[None, :]
        reward = np.asarray(reward).reshape(-1)
        # ``done`` is the reset/episode-boundary mask supplied by the rollout
        # loop.  True task termination is the critic's bootstrap mask; an
        # ordinary time-limit truncation still bootstraps from its stashed
        # terminal observation.  Optional arguments keep direct legacy callers
        # backward compatible (their old ``done`` retains its old meaning).
        episode_end = np.asarray(done).reshape(-1)
        if terminated is None:
            objective_done = episode_end.copy()
        else:
            objective_done = np.asarray(terminated).reshape(-1)
        if truncated is not None:
            truncated_arr = np.asarray(truncated).reshape(-1)
            if truncated_arr.shape != episode_end.shape:
                raise ValueError("truncated must contain one value per env slot")
            if terminated is not None and not np.array_equal(
                episode_end.astype(bool),
                np.logical_or(
                    objective_done.astype(bool), truncated_arr.astype(bool)
                ),
            ):
                raise ValueError(
                    "done must equal terminated or truncated for every env slot"
                )
        if objective_done.shape != episode_end.shape:
            raise ValueError("terminated must contain one value per env slot")
        t = np.asarray(t).reshape(-1)
        next_t = np.asarray(next_t).reshape(-1)

        cap_failure = np.zeros_like(episode_end, dtype=np.float32)
        failure_reward_rate = np.zeros_like(episode_end, dtype=np.float32)
        failure_remaining_time = np.zeros_like(episode_end, dtype=np.float32)

        # terminal handling for reset in vec_env
        if isinstance(infos, (list, tuple)):
            for i, info in enumerate(infos):
                if (
                    episode_end[i]
                    and isinstance(info, dict)
                    and "terminal_observation" in info
                ):
                    next_obs[i] = info["terminal_observation"]
                    if "terminal_next_t" in info:
                        next_t[i] = info["terminal_next_t"]

                if not isinstance(info, dict) or not bool(
                    info.get("absorbing_failure", False)
                ):
                    continue
                cap_failure[i] = 1.0
                rate = float(info["absorbing_failure_reward_rate"])
                remaining = float(info["absorbing_failure_remaining_seconds"])
                if not np.isfinite(rate):
                    raise ValueError(
                        "absorbing failure reward rate must be finite"
                    )
                if not np.isfinite(remaining) or remaining < 0.0:
                    raise ValueError(
                        "absorbing failure remaining time must be finite and >= 0"
                    )
                if not episode_end[i] or not objective_done[i]:
                    raise ValueError(
                        "an absorbing failure must also be a true episode termination"
                    )
                failure_reward_rate[i] = rate
                failure_remaining_time[i] = remaining
        elif isinstance(infos, dict):
            if episode_end[0] and "terminal_observation" in infos:
                next_obs[0] = infos["terminal_observation"]
                if "terminal_next_t" in infos:
                    next_t[0] = infos["terminal_next_t"]
            if bool(infos.get("absorbing_failure", False)):
                cap_failure[0] = 1.0
                rate = float(infos["absorbing_failure_reward_rate"])
                remaining = float(infos["absorbing_failure_remaining_seconds"])
                if not np.isfinite(rate):
                    raise ValueError(
                        "absorbing failure reward rate must be finite"
                    )
                if not np.isfinite(remaining) or remaining < 0.0:
                    raise ValueError(
                        "absorbing failure remaining time must be finite and >= 0"
                    )
                if not episode_end[0] or not objective_done[0]:
                    raise ValueError(
                        "an absorbing failure must also be a true episode termination"
                    )
                failure_reward_rate[0] = rate
                failure_remaining_time[0] = remaining

        self.replay_buffer.add(
            obs=obs,
            next_obs=next_obs,
            action=action,
            reward=reward,
            done=objective_done,
            t=t,
            next_t=next_t,
            episode_end=episode_end,
            cap_failure=cap_failure,
            failure_reward_rate=failure_reward_rate,
            failure_remaining_time=failure_remaining_time,
        )

    # ---------------------- Learn ----------------------
    def learn(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 4,
    ) -> "OffPolicyAlgorithm":
        """
        Main off-policy loop: collect transitions and periodically train.
        """
        total_timesteps, callback = self._setup_learn(total_timesteps, callback)
        if callback:
            callback.on_training_start(locals(), globals())

        self._last_obs, _ = self.env.reset()

        while self.num_timesteps < total_timesteps:
            # Step 1: Collect one continuous-time transition
            if callback:
                callback.on_rollout_start()

            action = self._sample_action(self._last_obs)
            obs_t, t, _, reward, next_obs, next_t, terminated, truncated, infos = (
                self.env.step_dt(action)
            )
            done = np.logical_or(terminated, truncated)
            simulated_seconds = _realized_simulated_seconds(
                t=t,
                next_t=next_t,
                done=done,
                infos=infos,
            )

            self._store_transition(
                obs=obs_t,
                action=action,
                reward=reward,
                done=done,
                next_obs=next_obs,
                t=t,
                next_t=next_t,
                infos=infos,
                terminated=terminated,
                truncated=truncated,
            )
            self.num_simulated_seconds += simulated_seconds

            if self.action_noise is not None:
                if isinstance(self.action_noise, ActionNoise):
                    self.action_noise.reset()
                if isinstance(self.action_noise, VectorizedActionNoise):
                    done_indices = np.where(done)[0]
                    if len(done_indices) > 0:
                        self.action_noise.reset(indices=done_indices.tolist())

            self.num_timesteps += self.n_envs

            if callback and not callback.on_step():
                break

            # Episode handling and logging
            if isinstance(infos, (list, tuple)):
                # Vectorized env case
                dones_arr = np.asarray(done).reshape(-1).astype(bool)
                for i, d in enumerate(dones_arr):
                    if d:
                        self._update_info_buffer(infos[i])
                self._last_obs = (
                    next_obs  # Vec env already returns reset obs for done envs
                )
            else:
                done_flag = bool(done)
                if done_flag:
                    self._update_info_buffer(infos)
                    self._last_obs, _ = self.env.reset()
                else:
                    self._last_obs = next_obs

            # Update progress and learning rate
            self._update_progress_remaining()
            self._update_learning_rate()

            if callback:
                callback.on_rollout_end()

            # Step 2: Training/Optimizing model
            if (
                self.num_timesteps >= self.learning_starts
                and self.num_timesteps % self.train_freq == 0
                and self.replay_buffer.size() >= self.batch_size
            ):
                gradient_steps = self.gradient_steps
                if gradient_steps < 0:
                    # If gradient_steps < 0 then use train_freq steps as gradient steps
                    gradient_steps = self.train_freq

                self.train(gradient_steps=gradient_steps, batch_size=self.batch_size)

            # Log training statistics
            self._log_stats(log_interval)

        if callback:
            callback.on_training_end()

        return self
