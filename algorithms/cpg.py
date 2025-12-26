# algorithms/cpg.py
from __future__ import annotations

from typing import Optional, Union, Type

import torch as th
import torch.nn.functional as F

from environment.base import ContinuousEnv
from .on_policy import OnPolicyAlgorithm
from models.actor_v_critic import ActorVCriticModel
from common.schedules import Schedule
from common.buffers import RolloutBatch


class CPG(OnPolicyAlgorithm):
    """
    Continuous-time Policy Gradient (CPG) by [Zhao et al. 2023]

    Critic semi-gradient:
      ∇_θ V(x) [ V(x') - V(x) + r(x,a) dt - α log π(a|x) dt - β V(x) dt ]

    Q estimator:
      q(x,a) = r(x,a) + (e^{-β dt} V(x') - V(x)) / dt

    Actor:
      ∇_φ ≈ (1/β) E[ ∇ log π(a|x) ( q(x,a) - α log π(a|x) - α ) ]

    [Zhao et al. 2023] Hanyang Zhao, Wenpin Tang, and David Yao. Policy optimization for continuous reinforcement learning.
        Advances in Neural Information Processing Systems, volume 36, pages 13637-13663, 2023.
    """

    def __init__(
        self,
        env: ContinuousEnv,
        model: Union[ActorVCriticModel, str, Type[ActorVCriticModel]],
        model_kwargs: Optional[dict] = None,
        device: Union[str, th.device] = "auto",
        seed: Optional[int] = None,
        gamma: float = 0.99,
        learning_rate: Union[float, Schedule] = 3e-4,
        batch_size: int = 64,
        n_steps: int = 2048,
        n_epochs: int = 10,
        # CPG-sepcific hyperparams
        alpha: float = 1.0,
        max_grad_norm: float = 0.5,
    ) -> None:
        super().__init__(
            env=env,
            model=model,
            model_kwargs=model_kwargs,
            device=device,
            seed=seed,
            gamma=gamma,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_steps=n_steps,
            n_epochs=n_epochs,
        )
        self.alpha = float(alpha)
        self.max_grad_norm = float(max_grad_norm)

        self.value_optimizer = th.optim.Adam(
            self.model.value_parameters, lr=self.lr_schedule(1.0)
        )
        self.actor_optimizer = th.optim.Adam(
            self.model.actor_parameters, lr=self.lr_schedule(1.0)
        )
        self.optimizers = [self.value_optimizer, self.actor_optimizer]

    def _policy_act_value(
        self, obs_tensor: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        actions, log_prob = self.model.act(obs_tensor, deterministic=False)
        value = self.model.value(obs_tensor).squeeze(-1)
        return actions, log_prob.view(-1), value.view(-1)

    def _train_actor_critic_batch(self, batch: RolloutBatch) -> None:
        obs = batch.observations
        next_obs = batch.next_observations
        actions = batch.actions
        rewards = batch.rewards.view(-1)
        dones = batch.dones.view(-1)
        dt = batch.dt.view(-1)
        old_logp = batch.log_probs.view(-1).detach()
        non_terminal = 1.0 - dones

        # Compute V(x)
        v = self.model.value(obs).view(-1)

        # Compute targets and q values
        with th.no_grad():
            dt_scaled = (dt * self.time_rescale).clamp(min=1e-6)
            gamma_dt = th.exp(-self.beta * dt_scaled)
            v_next_raw = self.model.value(next_obs).view(-1)
            v_next = v_next_raw * non_terminal

            # TD target regression
            td_target = (
                gamma_dt * v_next + (rewards - self.alpha * old_logp) * dt_scaled
            )

            # q(x,a) = r + (gamma * v' - v) / dt
            delta_v = (gamma_dt * v_next_raw - v.detach()) * non_terminal
            q = rewards + delta_v / dt_scaled

            # Calculate weight for actor update (policy gradient)
            weight = q - self.alpha * old_logp - self.alpha

        # Critic update, using TD error regression
        critic_loss = F.mse_loss(v, td_target)
        self.value_optimizer.zero_grad()
        critic_loss.backward()
        v_grad_norm = th.nn.utils.clip_grad_norm_(
            self.model.value_parameters, self.max_grad_norm
        )
        if th.isfinite(v_grad_norm):
            self.value_optimizer.step()
        else:
            self.value_optimizer.zero_grad()

        # Actor Update
        new_logp = self.model.log_prob(obs, actions).view(-1)
        actor_loss = -(new_logp * weight).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        a_grad_norm = th.nn.utils.clip_grad_norm_(
            self.model.actor_parameters, self.max_grad_norm
        )
        if th.isfinite(a_grad_norm):
            self.actor_optimizer.step()
        else:
            self.actor_optimizer.zero_grad()

        # Log stats
        self.logger.record("train/critic_loss", critic_loss.item())
        self.logger.record("train/actor_loss", actor_loss.item())
        self.logger.record("train/q", q.max().item())

    def _train_critic_batch(self, batch: RolloutBatch) -> None:
        obs = batch.observations
        next_obs = batch.next_observations
        rewards = batch.rewards.view(-1)
        dones = batch.dones.view(-1)
        dt = batch.dt.view(-1)
        non_terminal = 1.0 - dones

        ## Critic loss
        v = self.model.value(obs).view(-1)
        with th.no_grad():
            logp = batch.log_probs.view(-1).detach()  # Use log pi(a|s) from batch
            dt_scaled = (dt * self.time_rescale).clamp(min=1e-6)
            gamma_dt = th.exp(-self.beta * dt_scaled)
            v_next_raw = self.model.value(next_obs).view(-1)
            v_next = v_next_raw * non_terminal

            # TD target regression
            td_target = gamma_dt * v_next + (rewards - self.alpha * logp) * dt_scaled

        critic_loss = F.mse_loss(v, td_target)

        self.value_optimizer.zero_grad()
        critic_loss.backward()
        v_grad_norm = th.nn.utils.clip_grad_norm_(
            self.model.value_parameters, self.max_grad_norm
        )
        if th.isfinite(v_grad_norm):
            self.value_optimizer.step()
        else:
            self.value_optimizer.zero_grad()

        self.logger.record("train/critic_loss", critic_loss.item())

    def _train_actor_batch(self, batch: RolloutBatch) -> None:
        obs = batch.observations
        next_obs = batch.next_observations
        actions = batch.actions
        rewards = batch.rewards.view(-1)
        dones = batch.dones.view(-1)
        dt = batch.dt.view(-1)
        non_terminal = 1.0 - dones

        old_logp = batch.log_probs.view(-1).detach()
        with th.no_grad():
            v = self.model.value(obs).view(-1)
            dt_scaled = dt * self.time_rescale
            gamma_dt = th.exp(-self.beta * dt_scaled)
            v_next_raw = self.model.value(next_obs).view(-1)
            delta_v = (gamma_dt * v_next_raw - v.detach()) * non_terminal
            q = rewards + delta_v / dt_scaled

            # Calculate weight for actor update (policy gradient)
            weight = q - self.alpha * old_logp - self.alpha

        logp = self.model.log_prob(obs, actions).view(-1)
        actor_loss = -(logp * weight).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        a_grad_norm = th.nn.utils.clip_grad_norm_(
            self.model.actor_parameters, self.max_grad_norm
        )
        if th.isfinite(a_grad_norm):
            self.actor_optimizer.step()
        else:
            self.actor_optimizer.zero_grad()

        # Log stats
        self.logger.record("train/actor_loss", actor_loss.item())
        self.logger.record("train/q", q.max().item())
