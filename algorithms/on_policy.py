# algorithms/on_policy.py
from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Any, Dict, Optional, Type, Union, Tuple

import numpy as np
import torch as th


from .base import BaseAlgorithm
from environment.base import ContinuousEnv
from models.base import Model
from common.utils import get_action_dim, get_obs_shape
from common.buffers import RolloutBuffer, RolloutBatch
from common.schedules import Schedule
from common.callbacks import BaseCallback, convert_callback


class OnPolicyAlgorithm(BaseAlgorithm, ABC):
    """
    Base class for on-policy CT algorithms using a RolloutBuffer.

    while num_timesteps < total_timesteps:
        1) collect n_steps rollouts with current policy into RolloutBuffer. Compute additional stats if needed
        2) run several epochs over the rollout buffer via `train()`
    """

    def __init__(
        self,
        env: ContinuousEnv,
        model: Union[Model, str, Type[Model]],
        model_kwargs: Optional[Dict[str, Any]] = None,
        device: Union[str, th.device] = "auto",
        seed: Optional[int] = None,
        # On-policy specific:
        gamma: float = 0.99,
        learning_rate: Union[float, Schedule] = 3e-4,
        batch_size: int = 64,
        n_steps: int = 2048,
        n_epochs: int = 10,
        train_actor_critic_separate_per_epoch: bool = False,
    ) -> None:
        super().__init__(
            env=env,
            model=model,
            model_kwargs=model_kwargs,
            device=device,
            seed=seed,
            learning_rate=learning_rate,
        )

        self.gamma = float(gamma)
        self.beta = -math.log(self.gamma)
        self.batch_size = int(batch_size)
        self.n_steps = int(n_steps)
        self.n_epochs = int(n_epochs)
        self.train_actor_critic_separate_per_epoch = bool(
            train_actor_critic_separate_per_epoch
        )

        self.obs_shape = get_obs_shape(self.env.observation_space)
        self.action_dim = get_action_dim(self.env.action_space)

        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.n_steps,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            device=self.device,
            gamma=self.gamma,
            n_envs=self.n_envs,
        )

    # ---------------------- Abstract methods ----------------------

    @abstractmethod
    def _policy_act_value(
        self, obs_tensor: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Given obs_tensor of shape [1, obs_dim] on device, return:
          actions:   [1, act_dim]
          log_prob:  [1]
          value:     [1]
        Concrete algorithms will implement this using:
          - CoupledVqModel (V, q, π)
          - ActorVCriticModel (V, π)
        """
        raise NotImplementedError

    @abstractmethod
    def _train_critic_batch(self, batch: RolloutBatch) -> None:
        raise NotImplementedError

    @abstractmethod
    def _train_actor_batch(self, batch: RolloutBatch) -> None:
        raise NotImplementedError

    @abstractmethod
    def _train_actor_critic_batch(self, batch: RolloutBatch) -> None:
        raise NotImplementedError

    def _train(self):
        if self.train_actor_critic_separate_per_epoch:
            for epoch in range(self.n_epochs):
                for batch in self.rollout_buffer.get(batch_size=self.batch_size):
                    self._train_critic_batch(batch)
                for batch in self.rollout_buffer.get(batch_size=self.batch_size):
                    self._train_actor_batch(batch)
        else:
            for epoch in range(self.n_epochs):
                for batch in self.rollout_buffer.get(batch_size=self.batch_size):
                    self._train_actor_critic_batch(batch)

    # -------------------- Rollout logic --------------------

    def _collect_rollouts(
        self, callback: BaseCallback, log_interval: int = 100
    ) -> bool:
        """
        Collect exactly n_steps transitions into rollout_buffer using step_dt().
        Returns True if training should continue, False otherwise.
        """
        self.rollout_buffer.reset()
        if callback:
            callback.on_rollout_start()

        obs, infos = self.env.reset()
        episode_start = np.ones((self.n_envs,), dtype=np.float32)

        for step in range(self.n_steps):
            obs_arr = np.asarray(obs, dtype=np.float32)
            if obs_arr.ndim == len(self.rollout_buffer.obs_shape):
                obs_arr = obs_arr[None, ...]  # (1, obs_dim) for single env

            obs_tensor = th.as_tensor(obs_arr, device=self.device).float()

            with th.no_grad():
                actions_th, log_prob_th, value_th = self._policy_act_value(obs_tensor)

            # Shapes are (n_envs, act_dim), (n_envs,), (n_envs,)
            actions_th = actions_th
            log_prob_th = log_prob_th.reshape(-1)
            value_th = value_th.reshape(-1)

            actions = actions_th.cpu().numpy()

            # single env expects (act_dim,); vec env expects (n_envs, act_dim)
            env_action = actions if self.is_vec_env else actions[0]

            obs_t, t, _, reward, next_obs, next_t, terminated, truncated, infos_step = (
                self.env.step_dt(env_action)
            )
            dones = np.logical_or(terminated, truncated)

            # normalize
            obs_t_arr = np.asarray(obs_t, dtype=np.float32)
            next_obs_arr = np.asarray(next_obs, dtype=np.float32)
            if obs_t_arr.ndim == len(self.rollout_buffer.obs_shape):
                obs_t_arr = obs_t_arr[None, ...]
                next_obs_arr = next_obs_arr[None, ...]
            rew_arr = np.asarray(reward, dtype=np.float32).reshape(-1)
            dones_arr = np.asarray(dones, dtype=np.float32).reshape(-1)
            t_arr = np.asarray(t, dtype=np.float32).reshape(-1)
            next_t_arr = np.asarray(next_t, dtype=np.float32).reshape(-1)

            # Terminal fix for auto-resetting vec env
            next_obs_store = next_obs_arr
            next_t_store = next_t_arr
            if isinstance(infos_step, (list, tuple)):
                next_obs_store = next_obs_arr.copy()
                next_t_store = next_t_arr.copy()
                for i, d in enumerate(dones_arr.astype(bool)):
                    if not d:
                        continue
                    info_i = infos_step[i]
                    if isinstance(info_i, dict) and "terminal_observation" in info_i:
                        next_obs_store[i] = np.asarray(
                            info_i["terminal_observation"], dtype=np.float32
                        )
                        if "terminal_next_t" in info_i:
                            next_t_store[i] = float(info_i["terminal_next_t"])

                # Update eps info buffer for envs that finished
                for i, d in enumerate(dones_arr.astype(bool)):
                    if d:
                        self._update_info_buffer(infos_step[i])
            else:
                # single env
                if bool(dones_arr[0]) and isinstance(infos_step, dict):
                    self._update_info_buffer(infos_step)

            self.rollout_buffer.add(
                obs=obs_t_arr,
                next_obs=next_obs_store,
                action=actions,
                reward=rew_arr,
                done=dones_arr,
                episode_start=episode_start,
                value=value_th,
                log_prob=log_prob_th,
                t=t_arr,
                next_t=next_t_store,
            )

            self.num_timesteps += self.n_envs
            episode_start = dones_arr.astype(np.float32)

            if callback and not callback.on_step():
                return False

            # For raw single-env (no auto-reset), we must reset manually
            if (not self.is_vec_env) and bool(dones_arr[0]):
                obs, infos = self.env.reset()
                episode_start = np.ones((self.n_envs,), dtype=np.float32)
            else:
                obs = next_obs

        if callback:
            callback.on_rollout_end()

        # Log stats
        self._log_stats(log_interval)

        return True

    # --------------------------- Learn ---------------------------

    def learn(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 1,
    ) -> OnPolicyAlgorithm:
        """
        Main on-policy loop: alternate rollouts and multiple epochs of updates.
        """
        total_timesteps, callback = self._setup_learn(total_timesteps, callback)
        if callback:
            callback.on_training_start(locals(), globals())

        while self.num_timesteps < self._total_timesteps:
            # Step 1: collect rollouts
            continue_training = self._collect_rollouts(
                callback, log_interval=log_interval
            )

            if not continue_training:
                break

            # update progress & LR (once per rollout)
            self._update_progress_remaining()
            self._update_learning_rate()

            # Step 2: run several epochs of training using the current rollout buffer
            self._train()

        if callback:
            callback.on_training_end()
        return self
