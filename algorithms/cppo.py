# algorithms/cppo.py
from __future__ import annotations

import math
from typing import Optional, Union, Iterable, Type

import numpy as np
import torch as th
import torch.nn.functional as F

from environment.base import ContinuousEnv
from common.schedules import Schedule
from common.buffers import RolloutBatch
from algorithms.on_policy import OnPolicyAlgorithm
from models.actor_v_critic import ActorVCriticModel


class CPPO(OnPolicyAlgorithm):
    """
    Continuous-time Proximal Policy Optimization (CPPO) by [Zhao et al. 2023]
    Define local approximation fct: L^{π}(π') = η(π) + 1/β * E[(π'/π) (q(x, a) - α log π(a|x))]
    So steps from CPG can be repeated
    The main loop consists of:
        1. argmax_{π'} L^{π}(π') - C^k_penalty D_KL where D_KL = E_x[sqrt(D_KL(π'(.|x) || π(.|x)))]
        2. Update C^k_penalty: multiple if D_KL > (1 + eps)\delta and divide if D_KL < (1 - eps)\delta

    Steps include:
        1. CT critic semi-gradient (CPG-style):
            ∇ V(x) [ V(x') - V(x) + r dt - alpha log pi_old(a|x) dt - beta V(x) dt ]
        2. CT q estimator: q(x,a) = r_rate + (exp(-beta dt) V(x') - V(x)) / dt
        3. Define CT advantage: A_ct = q - alpha * log pi_new(a|x).
        4. Policy objective: maximize E[ ratio * A_ct ] and penalize sqrt(KL).
            where ratio = exp(log pi_new(a|x) - log pi_old(a|x))

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
        # CPPO specific hyperparameters
        alpha: float = 1.0,
        do_clip_ratio: bool = False,
        clip_range: float = 0.2,
        max_grad_norm: float = 0.5,
        # sqrt(KL) penalty control hyperparams
        target_sqrt_kl: float = 0.02,  # δ (for sqrt(KL))
        kl_update_eps: float = 0.1,  # eps in (1±eps)δ
        kl_coef_init: float = 1.0,  # C^0_penalty
        kl_coef_multiple: float = 1.5,  # KL coef multiple update
        kl_coef_update_interval: int = 10000,  # how frequent to update KL coef
        max_kl_coef: float = 10.0,
        min_kl_coef: float = 1e-8,
        kl_sqrt_eps: float = 1e-8,
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
        self.do_clip_ratio = bool(do_clip_ratio)
        self.clip_range = float(clip_range)
        self.max_grad_norm = float(max_grad_norm)

        # sqrt(KL) penalty control hyperparams
        self.target_sqrt_kl = float(target_sqrt_kl)
        self.kl_update_eps = float(kl_update_eps)
        self.kl_coef = float(kl_coef_init)
        self.kl_coef_multiple = float(kl_coef_multiple)
        self.kl_coef_update_interval = int(kl_coef_update_interval)
        self.max_kl_coef = float(max_kl_coef)
        self.min_kl_coef = float(min_kl_coef)
        self.kl_sqrt_eps = float(kl_sqrt_eps)

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

        # Compute V
        v = self.model.value(obs).view(-1)

        # Compute targets and q values
        with th.no_grad():
            dt = dt * self.time_rescale
            gamma_dt = th.exp(-self.beta * dt)
            v_next_raw = self.model.value(next_obs).view(-1)
            v_next = v_next_raw * non_terminal

            # Compute TD target: v_target
            td_target = gamma_dt * v_next + (rewards - self.alpha * old_logp) * dt

            # Compute q-fct and continuous-time advantage
            delta_v = (gamma_dt * v_next_raw - v.detach()) * non_terminal
            q = rewards + delta_v / dt
            A_ct = q - self.alpha * old_logp

        # Critic loss: use regression against TD target
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
        log_ratio = (new_logp - old_logp).clamp(-10.0, 10.0)
        ratio = th.exp(log_ratio)
        ratio_c = th.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        unclipped = ratio * A_ct
        if not self.do_clip_ratio:
            surrogate = unclipped.mean()
        else:
            clipped = ratio_c * A_ct
            surrogate = th.min(unclipped, clipped).mean()

        approx_kl = (old_logp - new_logp).mean()
        sqrt_kl = th.sqrt(th.clamp(approx_kl, min=0.0) + self.kl_sqrt_eps)
        actor_loss = -(surrogate - self.kl_coef * sqrt_kl)

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

        # If not clip ratio, then consider dynamic kl coefficient update
        if not self.do_clip_ratio and (
            self.num_timesteps % self.kl_coef_update_interval == 0
        ):
            self._update_kl_coef(sqrt_kl)

        # Log stats
        self.logger.record("train/critic_loss", float(critic_loss.item()))
        self.logger.record("train/actor_loss", float(actor_loss.item()))
        self.logger.record("train/q", q.max().item())
        self.logger.record("train/sqrt_kl", float(sqrt_kl.item()))
        self.logger.record("train/kl_coef", float(self.kl_coef))

    def _train_critic_batch(self, batch: RolloutBatch) -> None:
        obs = batch.observations
        next_obs = batch.next_observations
        rewards = batch.rewards.view(-1)
        dones = batch.dones.view(-1)
        dt = batch.dt.view(-1)
        non_terminal = 1.0 - dones

        # Old log prob
        old_logp = batch.log_probs.view(-1).detach()

        ## Critic loss
        v = self.model.value(obs).view(-1)
        with th.no_grad():
            dt_scaled = dt * self.time_rescale
            gamma_dt = th.exp(-self.beta * dt_scaled)
            v_next = self.model.value(next_obs).view(-1) * non_terminal
            td_target = (
                gamma_dt * v_next + (rewards - self.alpha * old_logp) * dt_scaled
            )

        critic_loss = F.mse_loss(v, td_target)

        # Optimize critic
        self.value_optimizer.zero_grad()
        critic_loss.backward()
        v_grad_norm = th.nn.utils.clip_grad_norm_(
            self.model.value_parameters, self.max_grad_norm
        )
        if th.isfinite(v_grad_norm):
            self.value_optimizer.step()
        else:
            self.value_optimizer.zero_grad()

        self.logger.record("train/critic_loss", float(critic_loss.item()))

    def _train_actor_batch(self, batch: RolloutBatch) -> None:
        obs = batch.observations
        next_obs = batch.next_observations
        actions = batch.actions
        rewards = batch.rewards.view(-1)
        dones = batch.dones.view(-1)
        dt = batch.dt.view(-1)
        non_terminal = 1.0 - dones

        old_logp = batch.log_probs.view(-1).detach()
        new_logp = self.model.log_prob(obs, actions).view(-1)

        with th.no_grad():
            v = self.model.value(obs).view(-1)
            dt_scaled = dt * self.time_rescale
            gamma_dt = th.exp(-self.beta * dt_scaled)
            v_next_raw = self.model.value(next_obs).view(-1)
            delta_v = (gamma_dt * v_next_raw - v.detach()) * non_terminal
            q = rewards + delta_v / dt_scaled
            A_ct = q - self.alpha * old_logp

        ## Actor loss
        log_ratio = (new_logp - old_logp).clamp(-10.0, 10.0)
        ratio = th.exp(log_ratio)
        ratio_c = th.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        unclipped = ratio * A_ct
        clipped = ratio_c * A_ct

        # Maximize E[ ratio * A_ct ] with PPO clipping
        surrogate = th.min(unclipped, clipped).mean()

        # sqrt(KL) penalty (sample-based approx): KL(π_old || π_new) ≈ E[ old_logp - new_logp ]
        approx_kl = (old_logp - new_logp).mean()
        sqrt_kl = th.sqrt(th.clamp(approx_kl, min=0.0) + self.kl_sqrt_eps)

        actor_loss = -(surrogate - self.kl_coef * sqrt_kl)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        a_grad_norm = th.nn.utils.clip_grad_norm_(
            self.model.actor_parameters, self.max_grad_norm
        )
        if th.isfinite(a_grad_norm):
            self.actor_optimizer.step()
        else:
            self.actor_optimizer.zero_grad()

        # Adaptive kl_coef update
        if self.num_timesteps % self.kl_coef_update_interval == 0:
            self._update_kl_coef(sqrt_kl)

        # Log stats
        self.logger.record("train/actor_loss", float(actor_loss.item()))
        self.logger.record("train/q", q.max().item())
        self.logger.record("train/sqrt_kl", float(sqrt_kl.item()))
        self.logger.record("train/kl_coef", float(self.kl_coef))

    def _update_kl_coef(self, sqrt_kl: th.Tensor) -> None:
        with th.no_grad():
            hi = (1.0 + self.kl_update_eps) * self.target_sqrt_kl
            lo = (1.0 - self.kl_update_eps) * self.target_sqrt_kl
            k = float(sqrt_kl.item())

            if k > hi:
                self.kl_coef *= self.kl_coef_multiple
            elif k < lo:
                self.kl_coef /= self.kl_coef_multiple

            self.kl_coef = float(
                np.clip(self.kl_coef, self.min_kl_coef, self.max_kl_coef)
            )
