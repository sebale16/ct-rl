# common/buffers.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch as th
from gymnasium import spaces

from .utils import get_device, get_obs_shape, get_action_dim


@dataclass
class ReplayBatch:
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    rewards: th.Tensor
    dones: th.Tensor
    t: th.Tensor
    next_t: th.Tensor
    dt: th.Tensor


@dataclass
class RolloutBatch:
    observations: th.Tensor
    next_observations: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    dones: th.Tensor
    episode_starts: th.Tensor
    values: th.Tensor
    log_probs: th.Tensor
    t: th.Tensor
    next_t: th.Tensor
    dt: th.Tensor


class BaseBuffer(ABC):
    """
    Base class for replay/rollout buffers.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[str, th.device] = "auto",
        n_envs: int = 1,
    ) -> None:
        super().__init__()
        self.buffer_size = int(buffer_size)
        self.observation_space = observation_space
        self.action_space = action_space

        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)

        self.device = get_device(device)
        self.n_envs = int(n_envs)

        self.pos: int = 0
        self.full: bool = False

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap axes (buffer, env) and flatten: (T, n_env, ...) -> (T*n_env, ...).
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        return self.buffer_size if self.full else self.pos

    def reset(self) -> None:
        self.pos = 0
        self.full = False

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)

    @abstractmethod
    def add(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def _get_samples(self, batch_inds: np.ndarray) -> Union[ReplayBatch, RolloutBatch]:
        raise NotImplementedError


class ReplayBuffer(BaseBuffer):
    """
    Simple off-policy replay buffer with time-awareness:
    stores (s, a, r, done, s', t, t', dt).
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[str, th.device] = "auto",
        n_envs: int = 1,
    ) -> None:
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )

        # (T, n_env, obs...)
        self.observations = np.zeros(
            (self.buffer_size, self.n_envs, *self.obs_shape),
            dtype=np.float32,
        )
        self.next_observations = np.zeros_like(self.observations)
        # (T, n_env, action_dim)
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=np.float32,
        )
        # (T, n_env)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.t = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.next_t = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dt = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_obs: np.ndarray,
        t: np.ndarray,
        next_t: np.ndarray,
    ) -> None:
        """
        Add a batch of transitions for all envs.

        All inputs are expected shape (n_envs, ...) for obs/action/etc,
        and (n_envs,) for reward/done/t/next_t.
        """
        # Ensure proper shapes for vectorized envs
        obs = np.asarray(obs, dtype=np.float32).reshape((self.n_envs, *self.obs_shape))
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(
            (self.n_envs, *self.obs_shape)
        )

        action = np.asarray(action, dtype=np.float32).reshape(
            (self.n_envs, self.action_dim)
        )
        reward = np.asarray(reward, dtype=np.float32).reshape((self.n_envs,))
        done = np.asarray(done, dtype=np.float32).reshape((self.n_envs,))
        t = np.asarray(t, dtype=np.float32).reshape((self.n_envs,))
        next_t = np.asarray(next_t, dtype=np.float32).reshape((self.n_envs,))

        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.t[self.pos] = t
        self.next_t[self.pos] = next_t
        self.dt[self.pos] = next_t - t

        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBatch:
        upper = self.buffer_size if self.full else self.pos
        assert upper > 0, "Cannot sample from an empty ReplayBuffer"
        batch_inds = np.random.randint(0, upper, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBatch:
        # Sample env index too
        env_inds = np.random.randint(0, self.n_envs, size=batch_inds.shape[0])

        obs = self.observations[batch_inds, env_inds, :]
        next_obs = self.next_observations[batch_inds, env_inds, :]
        actions = self.actions[batch_inds, env_inds, :]
        rewards = self.rewards[batch_inds, env_inds]
        dones = self.dones[batch_inds, env_inds]
        t = self.t[batch_inds, env_inds]
        next_t = self.next_t[batch_inds, env_inds]
        dt = self.dt[batch_inds, env_inds]

        # Add singleton dim for rewards/dones/time (batch, 1)
        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)
        t = t.reshape(-1, 1)
        next_t = next_t.reshape(-1, 1)
        dt = dt.reshape(-1, 1)

        return ReplayBatch(
            observations=self.to_torch(obs),
            actions=self.to_torch(actions),
            next_observations=self.to_torch(next_obs),
            rewards=self.to_torch(rewards),
            dones=self.to_torch(dones),
            t=self.to_torch(t),
            next_t=self.to_torch(next_t),
            dt=self.to_torch(dt),
        )


class RolloutBuffer(BaseBuffer):
    """
    On-policy rollout buffer with continuous-time awareness.

    Stores (s_t, a_t, r_t, done_t, s_{t+1}, dt_t) plus rollout-time V(s_t) and log π(a_t|s_t).
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[str, th.device] = "auto",
        gamma: float = 0.99,
        n_envs: int = 1,
        *,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )
        self.gamma = float(gamma)
        self.eps = float(eps)

        self.generator_ready: bool = False
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros(
            (self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32
        )
        self.next_observations = np.zeros(
            (self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32
        )
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )

        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.t = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.next_t = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dt = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.generator_ready = False
        super().reset()

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        t: np.ndarray,
        next_t: np.ndarray,
    ) -> None:
        if self.pos >= self.buffer_size:
            return

        value = value.detach()
        log_prob = log_prob.detach()

        if log_prob.ndim == 2 and log_prob.shape[1] == 1:
            log_prob = log_prob.reshape(-1)

        obs = np.asarray(obs, dtype=np.float32).reshape((self.n_envs, *self.obs_shape))
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(
            (self.n_envs, *self.obs_shape)
        )
        action = np.asarray(action, dtype=np.float32).reshape(
            (self.n_envs, self.action_dim)
        )
        reward = np.asarray(reward, dtype=np.float32).reshape((self.n_envs,))
        done = np.asarray(done, dtype=np.float32).reshape((self.n_envs,))
        episode_start = np.asarray(episode_start, dtype=np.float32).reshape(
            (self.n_envs,)
        )
        t = np.asarray(t, dtype=np.float32).reshape((self.n_envs,))
        next_t = np.asarray(next_t, dtype=np.float32).reshape((self.n_envs,))

        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.episode_starts[self.pos] = episode_start

        self.values[self.pos] = value.clone().cpu().numpy().reshape(-1)
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy().reshape(-1)

        self.t[self.pos] = t
        self.next_t[self.pos] = next_t
        self.dt[self.pos] = next_t - t

        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None):
        # assert self.full, "RolloutBuffer not full: collect data."
        current_size = self.buffer_size if self.full else self.pos
        n_samples = current_size * self.n_envs
        indices = np.random.permutation(n_samples)

        if not self.generator_ready:
            for name in [
                "observations",
                "next_observations",
                "actions",
                "rewards",
                "dones",
                "episode_starts",
                "values",
                "log_probs",
                "t",
                "next_t",
                "dt",
            ]:
                arr = getattr(self, name)
                if not self.full:
                    arr = arr[:current_size]
                setattr(self, name, self.swap_and_flatten(arr))
            self.generator_ready = True

        if batch_size is None:
            batch_size = n_samples

        start = 0
        while start < n_samples:
            batch_inds = indices[start : start + batch_size]
            yield self._get_samples(batch_inds)
            start += batch_size

    def _get_samples(self, batch_inds: np.ndarray) -> RolloutBatch:
        return RolloutBatch(
            observations=self.to_torch(self.observations[batch_inds]),
            next_observations=self.to_torch(self.next_observations[batch_inds]),
            actions=self.to_torch(self.actions[batch_inds]),
            rewards=self.to_torch(self.rewards[batch_inds]),
            dones=self.to_torch(self.dones[batch_inds]),
            episode_starts=self.to_torch(self.episode_starts[batch_inds]),
            values=self.to_torch(self.values[batch_inds]),
            log_probs=self.to_torch(self.log_probs[batch_inds]),
            t=self.to_torch(self.t[batch_inds]),
            next_t=self.to_torch(self.next_t[batch_inds]),
            dt=self.to_torch(self.dt[batch_inds]),
        )
