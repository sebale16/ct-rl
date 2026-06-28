from typing import Union, Optional, Dict, Any, Type
import numpy as np
import torch as th
import torch.nn.functional as F

from environment.base import ContinuousEnv
from models.base import Model
from .off_policy import OffPolicyAlgorithm
from common.schedules import Schedule
from common.buffers import ReplayBatch
from models.actor_q_critic import ActorQCriticModel


class CTSAC(OffPolicyAlgorithm):
    """
    Continuous-time SAC using ActorQCriticModel using our theoretical work.

    The critic target estimates the instantaneous advantage-rate
        q_V(x,a) = r + (L^a V)(x) - beta V(x)
    where (L^a V) is the controlled generator. By default this is estimated
    model-free by a finite difference over the sampled next state (Eq. 166).
    When ``use_model_based_q=True`` and a ``dynamics_model`` is supplied, the
    generator is evaluated analytically from a port-Hamiltonian drift b(x,a):
        (L^a V) = b . grad V + 1/2 Tr(sigma sigma^T Hess V)
    which removes the dependence on the sampled next state.
    """

    def __init__(
        self,
        env: ContinuousEnv,
        model: Union[
            ActorQCriticModel, str, Type[ActorQCriticModel]
        ] = "ActorQCriticModel",
        model_kwargs: Optional[Dict[str, Any]] = None,
        device: Union[str, th.device] = "auto",
        seed: Optional[int] = None,
        gamma: float = 0.99,
        buffer_size: int = 1_000_000,
        learning_rate: Union[float, Schedule] = 3e-4,
        batch_size: int = 256,
        train_freq: int = 1,
        gradient_steps: int = 1,
        learning_starts: int = 100,
        # CT-SAC specific hyperparameters
        alpha: float = 0.2,  # Entropy temperature. Auto means optimize alpha as well
        tau: float = 0.005,  # Euler step
        num_expectation_samples: int = 1,  # Num samples for expectation approximation
        target_entropy: Union[
            float, str
        ] = "auto",  # Target entropy when learning alpha
        # Model-based generator (port-Hamiltonian dynamics model)
        use_model_based_q: bool = False,
        dynamics_model: Optional[Any] = None,
        dynamics_source: str = "mujoco",
        human_input_intensity: float = 0.0,
        dynamics_lr: float = 1e-3,
        dynamics_warmup: int = 1000,
    ) -> None:
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
        )
        # Target entropy: "auto" -> -|A|, else use float value
        self.target_entropy = target_entropy
        action_dim = int(np.prod(self.env.action_space.shape))

        if self.target_entropy == "auto":
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = float(self.target_entropy)

        # Learnable log-alpha or fixed entropy coefficient alpha
        self.log_alpha: Optional[th.Tensor] = None
        self.alpha_optimizer: Optional[th.optim.Adam] = None
        self.alpha_tensor: Optional[th.Tensor] = None  # for fixed-alpha case

        if isinstance(alpha, str) and alpha.startswith("auto"):
            # Default initial value when learning alpha
            init_value = 1.0
            if "_" in alpha:
                init_value = float(alpha.split("_")[1])
                assert init_value > 0.0, "Initial alpha must be > 0"

            # log alpha is the trainable parameter
            self.log_alpha = th.log(
                th.ones(1, device=self.device) * init_value
            ).requires_grad_(True)
            self.alpha_optimizer = th.optim.Adam(
                [self.log_alpha], lr=self.lr_schedule(1.0)
            )
            self.alpha = float(init_value)
        else:
            # Fixed alpha
            alpha_float = float(alpha)
            self.alpha_tensor = th.tensor(alpha_float, device=self.device)
            self.alpha = alpha_float

        self.tau = float(tau)
        self.num_expectation_samples = int(num_expectation_samples)

        # Model-based generator configuration
        if isinstance(use_model_based_q, str):
            use_model_based_q = use_model_based_q.strip().lower() in ("1", "true", "yes")
        self.use_model_based_q = bool(use_model_based_q)
        self.dynamics_source = str(dynamics_source)
        self.human_input_intensity = float(human_input_intensity)
        self.dynamics_model = dynamics_model
        if self.dynamics_model is not None and hasattr(self.dynamics_model, "to"):
            self.dynamics_model.to(self.device)

        # Learned dynamics models (port-Hamiltonian "phast" mode) are fit online from
        # the replay buffer. Models with no trainable parameters (e.g. the MuJoCo
        # oracle) skip this and are used from the first step.
        self.dynamics_warmup = int(dynamics_warmup)
        self._dynamics_updates = 0
        self._train_dynamics = False
        self.dynamics_optimizer = None
        if self.dynamics_model is not None:
            dyn_params = [
                p for p in self.dynamics_model.parameters() if p.requires_grad
            ]
            if dyn_params:
                self._train_dynamics = True
                self.dynamics_optimizer = th.optim.Adam(
                    dyn_params, lr=float(dynamics_lr)
                )

        # The learning rate will be updated by the base algorithm class
        self.actor_optimizer = th.optim.Adam(
            self.model.actor.parameters(), lr=self.lr_schedule(1.0)
        )
        self.critic_optimizer = th.optim.Adam(
            self.model.critic_parameters, lr=self.lr_schedule(1.0)
        )
        self.optimizers = [self.actor_optimizer, self.critic_optimizer]
        if self.alpha_optimizer is not None:
            self.optimizers.append(self.alpha_optimizer)

        # For logging how many gradient updates we’ve done
        self._n_updates = 0

    def _policy_act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = th.as_tensor(obs, device=self.device).float()
        single = obs_t.ndim == 1
        if single:
            obs_t = obs_t.unsqueeze(0)  # (1, obs_dim)
        with th.no_grad():
            actions, _ = self.model.act(obs_t, deterministic=deterministic)
        actions_np = actions.detach().cpu().numpy()

        return actions_np[0] if single else actions_np

    def train(self, gradient_steps: int, batch_size: int) -> None:
        """
        Implement CT-SAC core update:

          Q_fast(s,a) = r + E_a[ Q̃_k(s,a) ]
                        + (γ E_{a'}[Q̃_k(s',a')] - E_a[Q̃_k(s,a)]) / u

        or, when ``use_model_based_q`` is set, the analytic generator target.
        The critic is trained to minimize MSE against Q_fast.
        The target networks are then updated using averaging.
        """
        for _ in range(gradient_steps):
            batch: ReplayBatch = self.replay_buffer.sample(batch_size)

            obs = batch.observations
            actions = batch.actions
            next_obs = batch.next_observations
            rewards = batch.rewards
            dones = batch.dones
            dt = batch.dt  # u in our notation

            ## Alpha (entropy temperature) update
            actions_pi, log_prob_pi = self.model.act(
                obs
            )  # Current policy on sampled states
            alpha_loss = None
            if self.alpha_optimizer is not None and self.log_alpha is not None:
                alpha_tensor = th.exp(self.log_alpha.detach())
                alpha_loss = -(
                    self.log_alpha * (log_prob_pi + self.target_entropy).detach()
                ).mean()
            else:
                # Fixed alpha
                alpha_tensor = self.alpha_tensor

            # Log alpha
            self.alpha = float(alpha_tensor.detach().item())
            self.logger.record("train/alpha", self.alpha)

            if alpha_loss is not None and self.alpha_optimizer is not None:
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.logger.record("train/alpha_loss", alpha_loss.item())

            ## Dynamics model update (learned port-Hamiltonian, fit from transitions)
            if self._train_dynamics:
                dynamics_loss = self.dynamics_model.fit_step(
                    obs, actions, next_obs, dt, self.dynamics_optimizer
                )
                self._dynamics_updates += 1
                self.logger.record("train/dynamics_loss", dynamics_loss)

            ## Critic update (target)
            # Use the model-based generator once the dynamics model is ready:
            # immediately for a non-trainable oracle, after warmup for a learned model.
            dynamics_ready = (not self._train_dynamics) or (
                self._dynamics_updates >= self.dynamics_warmup
            )
            if (
                self.use_model_based_q
                and self.dynamics_model is not None
                and dynamics_ready
            ):
                q_fast_target = self._model_based_target(
                    obs, actions, rewards, dones, alpha_tensor
                )
            else:
                q_fast_target = self._finite_difference_target(
                    obs, next_obs, rewards, dones, dt, alpha_tensor
                )

            # Calculate critic loss
            current_q_list = self.model.q_values(obs, actions)  # list of (B, 1)
            critic_loss = sum(F.mse_loss(q, q_fast_target) for q in current_q_list)

            self.logger.record("train/critic_loss", critic_loss.item())

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            ## Actor update
            for p in self.model.critic_parameters:
                p.requires_grad = False  # Freeze critic parameters for actor update

            # Re-use actions_pi, log_prob_pi from above
            q_values_pi = self.model.min_q(obs, actions_pi)
            actor_loss = (alpha_tensor * log_prob_pi - q_values_pi).mean()
            self.logger.record("train/actor_loss", actor_loss.item())

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Unfreeze critic parameters
            for p in self.model.critic_parameters:
                p.requires_grad = True

            # Target update
            self.model.soft_update_targets(tau=self.tau)

        # Track number of gradient updates
        # self._n_updates += gradient_steps
        # self.logger.record("train/n_updates", self._n_updates)

    # ------------------------ Critic targets ------------------------

    def _value_expectation(
        self,
        obs: th.Tensor,
        alpha_tensor: th.Tensor,
        deterministic: bool = False,
    ) -> th.Tensor:
        """V(x) = E_{a~pi}[ min-Q_target(x,a) - alpha log pi(a|x) ] via N reparam samples.

        Gradients w.r.t. ``obs`` flow through the (reparameterized) policy and the
        target critic, so ``autograd.grad(V.sum(), obs)`` gives the value gradient
        used by the model-based generator.
        """
        n = self.num_expectation_samples
        obs_rep = obs.repeat_interleave(n, dim=0)  # (B*N, O)
        actions, log_prob = self.model.act(obs_rep, deterministic=deterministic)
        q_min = self.model.target_min_q(obs_rep, actions)  # (B*N, 1)
        q_tilde = q_min - alpha_tensor * log_prob  # (B*N, 1)
        q_tilde = q_tilde.view(obs.shape[0], n, 1).mean(dim=1)  # (B, 1)
        return q_tilde

    def _finite_difference_target(
        self, obs, next_obs, rewards, dones, dt, alpha_tensor
    ) -> th.Tensor:
        """Model-free target (Eq. 166): generator estimated by a finite difference
        over the sampled next state.

          Q_fast = r + E[Q̃(s,a)] + (e^(-β*dt) E[Q̃(s',a')] - E[Q̃(s,a)]) / dt
        """
        with th.no_grad():
            expectation_q_tilde_next = self._value_expectation(next_obs, alpha_tensor)
            expectation_q_tilde_current = self._value_expectation(obs, alpha_tensor)

            dt = dt * self.time_rescale  # (B, 1), rescaled time
            gamma_dt = th.exp(-self.beta * dt)  # (B, 1)

            fraction = (
                gamma_dt * expectation_q_tilde_next - expectation_q_tilde_current
            ) / (dt + 1e-8)  # (B, 1) ~ (L^a V - beta V) in rescaled time
            future_val = expectation_q_tilde_current + fraction
            q_fast_target = rewards + (1 - dones) * future_val

            self.logger.record("train/fraction", th.max(th.abs(fraction)).item())
        return q_fast_target

    def _model_based_target(
        self, obs, actions, rewards, dones, alpha_tensor
    ) -> th.Tensor:
        """Model-based target: the generator is evaluated analytically from the
        port-Hamiltonian drift b(x,a), so no sampled next state is required.

          (L^a V - beta V) ~ dt_default * b . grad V - beta V   (rescaled-time
          convention matching the finite-difference target; see
          docs/port_hamiltonian_ct_sac.md, sec 2.2)

        With sigma != 0 (human input), the diffusion term
          1/2 Tr(sigma sigma^T Hess V)
        is added via Hessian-vector products.
        """
        sigma = self.dynamics_model.diffusion(obs)
        need_hessian = sigma is not None

        obs_req = obs.detach().clone().requires_grad_(True)
        V = self._value_expectation(obs_req, alpha_tensor)  # (B, 1), has graph
        (gV,) = th.autograd.grad(V.sum(), obs_req, create_graph=need_hessian)  # (B, O)

        V_det = V.detach()
        b = self.dynamics_model.drift(obs, actions)  # (B, O), physical drift (per second)
        b = th.as_tensor(b, dtype=V_det.dtype, device=V_det.device).detach()

        drift_term = (b * gV).sum(dim=-1, keepdim=True)  # b . grad V (per second)
        # Rescaled-time generator to match the finite-difference convention.
        lf = self.dt_default * drift_term - self.beta * V_det

        if need_hessian:
            sigma = sigma.to(V_det.device)
            k = sigma.shape[-1]
            hess = th.zeros_like(V_det)
            for j in range(k):
                sj = sigma[..., j]  # (B, O)
                (hvp,) = th.autograd.grad(
                    (gV * sj).sum(), obs_req, retain_graph=(j < k - 1)
                )  # (B, O) = (Hess V) sj
                hess = hess + (sj * hvp).sum(dim=-1, keepdim=True)
            lf = lf + self.dt_default * 0.5 * hess.detach()

        q_fast_target = (rewards + (1 - dones) * (V_det + lf.detach())).detach()
        self.logger.record("train/fraction", th.max(th.abs(lf)).item())
        return q_fast_target
