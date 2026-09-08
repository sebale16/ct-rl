from typing import Callable, Union, Optional, Type
import numpy as np
import torch as th
import torch.nn.functional as F

from common.utils import get_action_dim
from environment.base import ContinuousEnv
from models.base import Model
from .off_policy import OffPolicyAlgorithm
from common.schedules import Schedule
from common.buffers import ReplayBatch
from models.noise import ActionNoise, GaussianActionNoise
from models.actor_q_critic import ActorQCriticModel


class CTTD3(OffPolicyAlgorithm):
    """
    Continuous-time Twin Delayed Deep Deterministic Policy Gradient (TD3) using our theoretical work.
    This is the continuous-time version of TD3, which improves on CT-DDPG by
    using twin critics, delayed policy updates, and target policy smoothing.
    """

    def __init__(
        self,
        env: ContinuousEnv,
        model: Union[
            ActorQCriticModel, str, Type[ActorQCriticModel]
        ] = "ActorQCriticModel",
        model_kwargs: Optional[dict] = None,
        device: Union[str, th.device] = "auto",
        seed: Optional[int] = None,
        gamma: float = 0.99,
        buffer_size: int = 1_000_000,
        learning_rate: Union[float, Schedule] = 3e-4,
        batch_size: int = 256,
        train_freq: int = 1,
        gradient_steps: int = 1,
        learning_starts: int = 100,
        action_noise: Optional[ActionNoise] = None,
        # Continuous-time TD3 specific hyperparameters
        tau: float = 0.005,  # Euler step
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,  # Used if target_action_noise is None
        target_noise_clip: float = 0.5,
        target_action_noise: Optional[ActionNoise] = None,
        demonstration_policy: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        demonstration_steps: Optional[int] = None,
        # Reward/time semantics, mirroring CTSAC.  ``target_reference_dt`` is
        # the interval T at which Q is defined, in seconds, instead of the
        # simulator's native control timestep.  ``reward_is_rate`` says the
        # environment exposes a physical reward RATE, which the CT target
        # converts to one reference interval as ``T * r`` -- without it a task
        # whose reward is a rate is credited ~1/T too heavily relative to the
        # bootstrap term (Acrobot-XK: |Q| ~ 960 against CTSAC's ~10 on the
        # identical task, because CTSAC declares reward_is_rate and this class
        # historically could not).
        target_reference_dt: Optional[float] = None,
        reward_is_rate: bool = False,
    ) -> None:
        reward_is_rate = bool(reward_is_rate)
        if reward_is_rate and target_reference_dt is None:
            raise ValueError(
                "target_reference_dt must be explicit when reward_is_rate is "
                "configured"
            )
        super().__init__(
            env=env,
            model=model,
            model_kwargs=model_kwargs,
            device=device,
            seed=seed,
            gamma=gamma,
            buffer_size=buffer_size,
            learning_rate=learning_rate,
            batch_size=batch_size,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            learning_starts=learning_starts,
            action_noise=action_noise,
        )

        if not self.model.deterministic_policy or not self.model.use_actor_target:
            raise ValueError(
                "CTTD3 requires a deterministic policy and a target actor in ActorQCriticModel."
            )

        # Reference interval T.  Overrides the env-derived dt_default so the
        # target's time rescaling and reward conversion never depend on a
        # simulator clock.  beta stays -log(gamma) per reference interval, so
        # the physical discount rate is beta / T.
        if target_reference_dt is not None:
            reference_dt = float(target_reference_dt)
            if not np.isfinite(reference_dt) or reference_dt <= 0.0:
                raise ValueError(
                    "target_reference_dt must be finite and > 0 seconds, got "
                    f"{target_reference_dt!r}"
                )
            self.dt_default = reference_dt
            self.time_rescale = 1.0 / reference_dt
        self.target_reference_dt = float(self.dt_default)
        self.reward_is_rate = reward_is_rate

        self.tau = float(tau)
        self.policy_delay = int(policy_delay)
        self.target_noise_clip = float(target_noise_clip)
        self._gradient_step_counter = 0

        if target_action_noise is None:
            # Default to Gaussian noise for target smoothing
            action_dim = get_action_dim(self.env.action_space)
            self.target_action_noise = GaussianActionNoise(
                mean=np.zeros(action_dim),
                sigma=target_policy_noise * np.ones(action_dim),
            )
        else:
            self.target_action_noise = target_action_noise

        self.actor_optimizer = th.optim.Adam(
            self.model.actor_parameters, lr=self.lr_schedule(1.0)
        )
        self.critic_optimizer = th.optim.Adam(
            self.model.critic_parameters, lr=self.lr_schedule(1.0)
        )
        self.optimizers = [self.actor_optimizer, self.critic_optimizer]

        # Demonstration warm start: for the first `demonstration_steps` calls
        # to _sample_action (default: learning_starts), _sample_action defers
        # to `demonstration_policy` instead of the base class's random-action
        # warmup, so the replay buffer starts from states the demonstrator
        # actually reaches rather than a uniform random walk's states.
        # Gradient updates are unaffected -- train() still only begins once
        # num_timesteps > learning_starts, per the base class. Mirrors
        # algorithms.ct_sac.CTSAC's demonstration_policy/demonstration_steps
        # (see there for the imitation-loss counterpart this class doesn't
        # implement).
        self.demonstration_policy = demonstration_policy
        self.demonstration_steps = (
            self.learning_starts
            if demonstration_steps is None
            else int(demonstration_steps)
        )
        if self.demonstration_steps < 0:
            raise ValueError(
                "demonstration_steps must be non-negative, got "
                f"{demonstration_steps!r}"
            )

    def _sample_action(self, obs: np.ndarray) -> np.ndarray:
        """Defer to ``demonstration_policy`` during the demonstration warm start.

        Falls through to the base class (random actions before
        ``learning_starts``, the trained policy after) once
        ``demonstration_policy`` is unset or ``num_timesteps`` reaches
        ``demonstration_steps``.
        """
        if (
            self.demonstration_policy is None
            or self.num_timesteps >= self.demonstration_steps
        ):
            return super()._sample_action(obs)

        is_vec_env = self.is_vec_env
        n_envs = self.n_envs
        obs_arr = np.asarray(obs, dtype=np.float32)
        if is_vec_env:
            if obs_arr.ndim == 1:
                obs_arr = obs_arr[None, :]
            assert (
                obs_arr.shape[0] == n_envs
            ), f"obs first dim must be n_envs={n_envs}, got {obs_arr.shape}"
            action = np.stack(
                [
                    np.asarray(
                        self.demonstration_policy(obs_arr[i]), dtype=np.float32
                    ).reshape(self.action_dim)
                    for i in range(n_envs)
                ],
                axis=0,
            )
        else:
            if obs_arr.ndim > 1:
                obs_arr = obs_arr[0]
            action = np.asarray(
                self.demonstration_policy(obs_arr), dtype=np.float32
            ).reshape(self.action_dim)
        return np.clip(action, self.env.action_space.low, self.env.action_space.high)

    def _policy_act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = th.as_tensor(obs, device=self.device).float()
        single = obs_t.ndim == 1
        if single:
            obs_t = obs_t.unsqueeze(0)  # (1, obs_dim)
        with th.no_grad():
            actions, _ = self.model.act(obs_t, deterministic=True)
        actions_np = actions.detach().cpu().numpy()

        return actions_np[0] if single else actions_np

    # ------------------------ Critic target ------------------------
    def _critic_target(
        self,
        batch: ReplayBatch,
        q_target_current: th.Tensor,
        q_target_next: th.Tensor,
    ) -> th.Tensor:
        """Build one target batch without bootstrapping through cap failures.

        Mirrors ``CTSAC._critic_target``.  Ordinary rows keep the finite-
        difference generator target; a state-cap failure takes a separate
        analytical route over the unexecuted episode remainder and is never
        passed through a learned endpoint value.  Splitting before target
        evaluation matters -- merely overwriting afterward would still let a
        non-finite learned terminal value poison the batch.

        The finite-difference branch keeps this class's rescaled-time
        convention; the absorbing branch takes seconds and rescales internally
        so that its nominal-interval test is made against ``dt_default``
        directly (see ``_absorbing_failure_target``).
        """
        dt = batch.dt * self.time_rescale
        cap_mask = batch.cap_failures.reshape(-1) > 0.5
        regular_mask = ~cap_mask
        target = th.empty_like(batch.rewards)

        if bool(th.any(cap_mask)):
            cap_dones = batch.dones[cap_mask]
            cap_ends = batch.episode_ends[cap_mask]
            if bool(th.any(cap_dones <= 0.5)) or bool(th.any(cap_ends <= 0.5)):
                raise ValueError(
                    "cap-failure replay rows must be true terminal episode ends"
                )
            target[cap_mask] = self._absorbing_failure_target(
                q_target_current[cap_mask],
                batch.rewards[cap_mask],
                batch.dt[cap_mask],
                batch.failure_reward_rates[cap_mask],
                batch.failure_remaining_times[cap_mask],
            )

        if bool(th.any(regular_mask)):
            target[regular_mask] = self._finite_difference_target(
                q_target_current[regular_mask],
                q_target_next[regular_mask],
                batch.rewards[regular_mask],
                batch.dones[regular_mask],
                dt[regular_mask],
            )

        self.logger.record(
            "train/cap_failure_fraction", cap_mask.to(th.float32).mean().item()
        )
        return target.detach()

    def _target_reward_term(self, rewards: th.Tensor) -> th.Tensor:
        """Convert configured rewards to one target-reference interval.

        ``T * r`` for a physical reward rate, ``r`` for the legacy convention
        where the environment already exposes an amount per reference
        interval.  Neither depends on the realized duration ``h``: the CT
        target credits the rate over one reference interval and lets the
        generator increment carry the duration.
        """
        return (
            rewards * self.target_reference_dt
            if self.reward_is_rate
            else rewards
        )

    def _finite_difference_target(
        self,
        q_current: th.Tensor,
        q_next: th.Tensor,
        rewards: th.Tensor,
        dones: th.Tensor,
        dt: th.Tensor,
    ) -> th.Tensor:
        """Model-free generator target over the rescaled duration ``dt``."""
        gamma_dt = th.exp(-self.beta * dt)
        fraction = (gamma_dt * q_next - q_current) / (dt + 1e-8)
        future_val = q_current + fraction
        self.logger.record("train/fraction", th.max(th.abs(fraction)).item())
        return self._target_reward_term(rewards) + (1 - dones) * future_val

    def _absorbing_failure_target(
        self,
        q_current: th.Tensor,
        rewards: th.Tensor,
        dt: th.Tensor,
        failure_reward_rates: th.Tensor,
        failure_remaining_times: th.Tensor,
    ) -> th.Tensor:
        r"""Analytical cap target over the unexecuted episode remainder.

        The counterpart of ``CTSAC._absorbing_failure_target``, written in this
        class's rescaled-time units.  For rescaled duration ``rho = h / T``,
        rescaled remaining time ``Rt = R / T``, discount exponent ``beta`` per
        reference interval and the configured reward's finite lower envelope
        ``r_F``, the frozen absorbing value is

        ``C_F = r_F (1 - exp(-beta Rt)) / beta``

        (or ``r_F Rt`` at zero discount).  It replaces the learned
        ``Q(s', a')`` endpoint.  At the nominal ``rho == 1`` interval the
        current-value anchor cancels exactly:

        ``y_F = r_F + exp(-beta) C_F``

        and off-nominal rows re-anchor through the current value exactly as an
        ordinary irregular transition does.  CT-SAC anchors on ``V(s)``; the
        deterministic-policy analogue here is the batch-action ``Q(s, a)`` this
        class already uses as its finite-difference anchor.

        ``C_F`` is expressed per reference interval, matching this class's
        per-interval reward convention -- it is CT-SAC's ``G_F`` divided by the
        reference interval ``T``, since ``beta = lambda T``.

        ``dt`` and ``failure_remaining_times`` arrive in seconds and are
        rescaled here.  The nominal test is made in seconds, as CT-SAC does:
        rescaling first would put a float multiply between the stored duration
        and the reference, and a near-miss there costs the exact cancellation
        of the anchor.
        """
        if bool(th.any(dt <= 0.0)):
            raise ValueError("dt values must be strictly positive")
        rates = failure_reward_rates
        if not bool(th.all(th.isfinite(rates))):
            raise ValueError("failure reward rates must be finite")
        if not bool(th.all(th.isfinite(failure_remaining_times))) or bool(
            th.any(failure_remaining_times < 0.0)
        ):
            raise ValueError("failure remaining times must be finite and >= 0")

        nominal = dt == dt.new_tensor(self.dt_default)
        rho = dt * self.time_rescale
        remaining = failure_remaining_times * self.time_rescale

        # C_F is linear in r_F, so the reward conversion applies to the frozen
        # remainder too; with reward_is_rate this makes C_F equal CTSAC's
        # physical G_F rather than G_F / T.
        rate_term = self._target_reward_term(rates)
        gamma_dt = th.exp(-self.beta * rho)
        if self.beta == 0.0:
            continuation = rate_term * remaining
        else:
            continuation = rate_term * (
                -th.expm1(-self.beta * remaining) / self.beta
            )

        endpoint = gamma_dt * continuation
        # rho == 1 reduces to the ordinary target; the anchor cancels exactly.
        future = th.where(
            nominal,
            endpoint,
            q_current + (endpoint - q_current) / rho,
        )
        target = self._target_reward_term(rewards) + future
        if not bool(th.all(th.isfinite(target))):
            raise ValueError("non-finite absorbing cap-failure target")
        self.logger.record(
            "train/failure_continuation_max_abs",
            continuation.abs().max().item(),
        )
        return target

    def train(self, gradient_steps: int, batch_size: int) -> None:
        for _ in range(gradient_steps):
            self._gradient_step_counter += 1
            batch: ReplayBatch = self.replay_buffer.sample(batch_size)

            obs = batch.observations
            actions = batch.actions
            next_obs = batch.next_observations

            ## Critic update
            with th.no_grad():
                # Compute Q_target(s, a) with s and a from batch
                q_target_current = self.model.target_min_q(obs, actions)

                # Generate next actions
                next_actions = self.model.act_target(next_obs)

                # Generate noise from the ActionNoise object
                noise = self.target_action_noise()
                noise = th.as_tensor(noise, dtype=th.float32, device=self.device)

                # Clip noise and add to next_actions
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                action_low = th.from_numpy(self.env.action_space.low).to(self.device)
                action_high = th.from_numpy(self.env.action_space.high).to(self.device)
                next_actions = (next_actions + noise).clamp(action_low, action_high)

                # Compute Q_target(s', a')
                q_target_next = self.model.target_min_q(next_obs, next_actions)

                # Construct Q_fast target
                q_fast_target = self._critic_target(
                    batch, q_target_current, q_target_next
                )

            # Calculate critic loss
            current_q_list = self.model.q_values(obs, actions)
            critic_loss = sum(F.mse_loss(q, q_fast_target) for q in current_q_list)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            self.logger.record("train/critic_loss", critic_loss.item())

            ## Delayed policy and target updates
            if self._gradient_step_counter % self.policy_delay == 0:
                # Actor update
                for p in self.model.critic_parameters:
                    p.requires_grad = False
                actions_pi, _ = self.model.act(obs)

                # Actor loss uses the first critic
                q_values_pi = self.model.q_values(obs, actions_pi)[0]
                actor_loss = -q_values_pi.mean()
                self.logger.record("train/actor_loss", actor_loss.item())

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for p in self.model.critic_parameters:
                    p.requires_grad = True

                # Target update
                self.model.soft_update_targets(tau=self.tau, update_actor=True)
