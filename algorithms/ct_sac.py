import os
import pathlib
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
from models.port_hamiltonian import integrate_drift


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
        dynamics_fit_horizon: int = 1,
        dynamics_integration_step: Optional[float] = None,
        generator_gate_scale: float = 0.0,
        value_warmup: int = 0,
        generator_substeps: int = 0,
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
        # Per-component trust-region blend: where |b_i * dt| <~ generator_gate_scale the
        # analytic drift is trusted; elsewhere it falls back to the realized drift
        # (x'_i - x_i)/dt. 0 (default) => pure generator (no gating).
        self.generator_gate_scale = float(generator_gate_scale)
        # Sub-step quadrature: request at least this many Euler sub-steps over the
        # nominal interval and read the value change from the V-head endpoints.
        # An exposed finer physics/integration step takes precedence.
        # 0 (default) => first-order autograd generator. >=1 => quadrature.
        self.generator_substeps = int(generator_substeps)
        # Maximum internal step used to turn the instantaneous drift into a
        # finite-duration flow. Replay keeps every irregular transition duration;
        # this only controls the numerical solver inside the fit/critic target.
        # Prefer the environment's physics resolution when it is exposed.
        if dynamics_integration_step is None:
            dynamics_integration_step = self._infer_physics_step(self.env)
        if dynamics_integration_step is not None:
            dynamics_integration_step = float(dynamics_integration_step)
            if (
                not np.isfinite(dynamics_integration_step)
                or dynamics_integration_step <= 0.0
            ):
                raise ValueError(
                    "dynamics_integration_step must be finite and > 0, got "
                    f"{dynamics_integration_step}"
                )
        self.dynamics_integration_step = dynamics_integration_step
        self.dynamics_source = str(dynamics_source)
        self.human_input_intensity = float(human_input_intensity)
        self.dynamics_model = dynamics_model
        if self.dynamics_model is not None and hasattr(self.dynamics_model, "to"):
            self.dynamics_model.to(self.device)

        # Learned dynamics models (port-Hamiltonian "phast" mode) are fit online from
        # the replay buffer. Models with no trainable parameters (e.g. the MuJoCo
        # oracle) skip this and are used from the first step.
        self.dynamics_warmup = int(dynamics_warmup)
        # Multi-transition rollout fit: with horizon H > 1 the model is fit on its
        # own flow across H replay intervals instead of a single endpoint. Each
        # irregular interval is internally integrated at _integration_max_step,
        # matching how the generator/quadrature target consumes the vector field.
        self.dynamics_fit_horizon = max(1, int(dynamics_fit_horizon))
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

        # Optional explicit scalar value head V(s): present iff the model was
        # built with one. It gives the model-based generator a clean, sample-free
        # V and grad V (no differentiation through the twin-min / stochastic
        # policy). Trained by regression to the soft value E_a[Q~] (see train()).
        self.use_value_head = bool(getattr(self.model, "has_v_head", False))
        # The head is trained from the first update, but the generator and the
        # critic targets read it only after value_warmup regression updates;
        # before that they use the sampled soft value, so the generator never
        # bootstraps on an untrained head.
        self.value_warmup = int(value_warmup)
        self._value_updates = 0
        self.value_optimizer = None
        if self.use_value_head:
            self.value_optimizer = th.optim.Adam(
                self.model.value_parameters, lr=self.lr_schedule(1.0)
            )
            self.optimizers.append(self.value_optimizer)

        # For logging how many gradient updates we’ve done
        self._n_updates = 0

    @staticmethod
    def _infer_physics_step(env) -> Optional[float]:
        """Read a physics-solver step through vector/wrapper layers when available."""
        current = env.envs[0] if hasattr(env, "envs") and env.envs else env
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            value = getattr(current, "physics_dt", None)
            if value is not None:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = None
                if value is not None and np.isfinite(value) and value > 0.0:
                    return value
            current = getattr(current, "env", None)
        return None

    def _integration_max_step(self) -> Optional[float]:
        """Internal flow resolution shared by dynamics fit and quadrature.

        ``generator_substeps`` remains a requested target resolution. If the
        environment exposes a finer physics step, both consumers use that finer
        value so the learned vector field is trained and queried consistently.
        """
        candidates = []
        if self.dynamics_integration_step is not None:
            candidates.append(float(self.dynamics_integration_step))
        if self.generator_substeps >= 1:
            candidates.append(self.dt_default / float(self.generator_substeps))
        return min(candidates) if candidates else None

    # ------------------------ persistence ------------------------

    @staticmethod
    def _dynamics_sidecar(path: Union[str, pathlib.Path]) -> str:
        """Sidecar file holding the learned dynamics model next to a checkpoint:
        ``best_model.pth`` -> ``best_model.dynamics.pth``."""
        root, ext = os.path.splitext(str(path))
        return root + ".dynamics" + (ext or ".pth")

    def save(self, path) -> None:
        """Save the actor-critic checkpoint, plus the learned dynamics model to a
        sidecar file (so the trained port-Hamiltonian can be inspected later,
        e.g. by ``evaluations/hamiltonian_recovery.py``)."""
        super().save(path)
        if self._train_dynamics and isinstance(path, (str, pathlib.Path)):
            th.save(self.dynamics_model.state_dict(), self._dynamics_sidecar(path))

    def load(self, path, strict: bool = True) -> "CTSAC":
        super().load(path, strict=strict)
        if self._train_dynamics and isinstance(path, (str, pathlib.Path)):
            sidecar = self._dynamics_sidecar(path)
            if os.path.exists(sidecar):
                self.dynamics_model.load_state_dict(
                    th.load(sidecar, map_location=self.device)
                )
        return self

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
                if self.dynamics_fit_horizon > 1:
                    # Multi-step rollout fit over a replay window: the model is
                    # rolled along its own predictions and every step regressed.
                    seq = self.replay_buffer.sample_sequences(
                        batch_size, self.dynamics_fit_horizon
                    )
                    dynamics_loss = self.dynamics_model.fit_step_rollout(
                        seq.observations, seq.actions, seq.next_observations,
                        seq.dt, seq.mask, self.dynamics_optimizer,
                        max_step=self._integration_max_step(),
                    )
                else:
                    dynamics_loss = self.dynamics_model.fit_step(
                        obs, actions, next_obs, dt, self.dynamics_optimizer,
                        max_step=self._integration_max_step(),
                    )
                self._dynamics_updates += 1
                self.logger.record("train/dynamics_loss", dynamics_loss)
                if self._integration_max_step() is not None:
                    self.logger.record(
                        "train/dynamics_integration_step",
                        self._integration_max_step(),
                    )

            ## Value head update (optional explicit scalar V(s) head)
            # Regress V_psi(s) to the soft state-value E_a[Q~(s,a)]. The label's
            # action expectation is sampled, but averaged over training into a
            # smooth V_psi; at the generator's point of use V_psi and grad V_psi
            # need no sampling and have a clean gradient.
            if self.use_value_head:
                with th.no_grad():
                    value_target = self._value_expectation(obs, alpha_tensor)
                value_pred = self.model.value(obs)
                value_loss = F.mse_loss(value_pred, value_target)
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()
                self._value_updates += 1
                self.logger.record("train/value_loss", value_loss.item())

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
                    obs, actions, next_obs, rewards, dones, dt, alpha_tensor
                )
                # A non-finite target means the model-based method has failed;
                # without this check it would silently NaN the critic, then the
                # value head and the policy, and the run would keep producing
                # 0-return evals with dead parameters. Fail loudly instead —
                # no model-free fallback, so the benchmark comparison stays a
                # pure model-based run or an explicit failure.
                if not bool(th.all(th.isfinite(q_fast_target))):
                    raise RuntimeError(
                        "Model-based critic target is non-finite (the learned "
                        "dynamics model has diverged). Terminating the run: "
                        "recovery is impossible once NaN reaches the critic, "
                        "and falling back to the model-free target would "
                        "contaminate the model-based benchmark."
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

    @property
    def _value_head_ready(self) -> bool:
        """The V-head is read by the generator/targets only after it has had
        ``value_warmup`` regression updates (insurance against bootstrapping on
        an untrained head). value_warmup=0 reads it immediately."""
        return self.use_value_head and self._value_updates >= self.value_warmup

    def _state_value(
        self, obs: th.Tensor, alpha_tensor: th.Tensor, use_target: bool = True
    ) -> th.Tensor:
        """State value V(x). With the explicit V-head this is a clean,
        sample-free read from the (lagged target) value net; otherwise it is the
        sampled soft expectation E_a[Q~] (``_value_expectation``)."""
        if self._value_head_ready:
            return (
                self.model.target_value(obs)
                if use_target
                else self.model.value(obs)
            )
        return self._value_expectation(obs, alpha_tensor)

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
            expectation_q_tilde_next = self._state_value(next_obs, alpha_tensor)
            expectation_q_tilde_current = self._state_value(obs, alpha_tensor)

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
        self, obs, actions, next_obs, rewards, dones, dt, alpha_tensor
    ) -> th.Tensor:
        """Model-based target: the generator is evaluated analytically from the
        port-Hamiltonian drift b(x,a), so no sampled next state is required.

          (L^a V - beta V) ~ dt_default * b . grad V - beta V   (rescaled-time
          convention matching the finite-difference target; see
          docs/port_hamiltonian_ct_sac.md, sec 2.2)

        When ``generator_gate_scale > 0`` the analytic drift is blended
        per-component with the realized drift ``(x' - x)/dt``: a gate
        ``g_i = exp(-|b_i * dt| / generator_gate_scale)`` trusts the analytic
        drift only where the effective step ``|b_i * dt|`` is small (the regime
        where the first-order generator is valid), and falls back to the data
        elsewhere (e.g. stiff contact coordinates). g=1 everywhere recovers the
        pure generator; g=0 everywhere is a first-order finite difference.

        With sigma != 0 (human input), the diffusion term
          1/2 Tr(sigma sigma^T Hess V)
        is added via Hessian-vector products.

        When ``generator_substeps >= 1`` the first-order autograd term is replaced
        by a sub-step quadrature: integrate the model over the nominal interval
        with the same finite-duration flow routine used by dynamics fitting and
        read the value change directly from the V-head endpoints,
        lf = (V(x_hat) - V(x)) - beta*V(x). This is autograd-free and captures
        curvature the first-order term drops. ``m=1`` is a single Euler step only
        when no finer dynamics integration step applies. See
        docs/ct_sac_substep_quadrature.md.
        """
        if self.generator_substeps >= 1:
            return self._substep_quadrature_target(
                obs, actions, rewards, dones, alpha_tensor
            )

        sigma = self.dynamics_model.diffusion(obs)
        need_hessian = sigma is not None

        obs_req = obs.detach().clone().requires_grad_(True)
        if self._value_head_ready:
            # Clean, sample-free V(x); grad flows to obs_req (not to the frozen
            # target params), giving a smooth value gradient for b . grad V.
            V = self.model.target_value(obs_req)  # (B, 1)
        else:
            V = self._value_expectation(obs_req, alpha_tensor)  # (B, 1), has graph
        (gV,) = th.autograd.grad(V.sum(), obs_req, create_graph=need_hessian)  # (B, O)

        V_det = V.detach()
        b = self.dynamics_model.drift(obs, actions)  # (B, O), per second
        b = th.as_tensor(b, dtype=V_det.dtype, device=V_det.device).detach()

        if self.generator_gate_scale > 0.0:
            dt_phys = th.as_tensor(
                dt, dtype=V_det.dtype, device=V_det.device
            ).reshape(-1, 1)  # (B, 1), seconds
            b_realized = (next_obs - obs) / (dt_phys + 1e-8)  # (B, O), per second
            b_realized = b_realized.to(V_det.dtype)
            gate = th.exp(-(b * dt_phys).abs() / self.generator_gate_scale)  # (B, O)
            b = gate * b + (1.0 - gate) * b_realized
            self.logger.record("train/gen_gate_mean", gate.mean().item())

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

    def _substep_quadrature_target(
        self, obs, actions, rewards, dones, alpha_tensor
    ) -> th.Tensor:
        """Sub-step quadrature generator target (``generator_substeps = m``).

        The drift-induced value change over the nominal interval is the integral
        of the rate along the model orbit,
          V(x') - V(x) = integral_0^dt_default (L^a V)(x(s)) ds,
        which the first-order term dt_default*(b.grad V) samples at one point. Here
        the model is integrated with explicit-Euler steps no larger than
        min(dt_default/m, dynamics_integration_step) when the latter is known. The
        V-head is read at the rolled endpoint and the value change is taken directly:
          lf = (V(x_hat) - V(x)) - beta*V(x),  target = r + V(x) + lf.
        No autograd / gradient is used; the value gradient and its curvature enter
        through the finite difference of the (clean) V-head over the predicted
        states. Larger m requests a finer integration; an even finer exposed
        physics step takes precedence. The discount is kept as the single
        -beta*V(x) lump, matching the first-order target.
        """
        with th.no_grad():
            x_hat = integrate_drift(
                self.dynamics_model.drift,
                obs.detach(),
                actions,
                self.dt_default,
                max_step=self._integration_max_step(),
            )
            V_cur = self._state_value(obs, alpha_tensor)  # (B, 1)
            V_next = self._state_value(x_hat, alpha_tensor)  # (B, 1) at rolled state
            lf = (V_next - V_cur) - self.beta * V_cur
            q_fast_target = (rewards + (1 - dones) * (V_cur + lf)).detach()
            self.logger.record("train/fraction", th.max(th.abs(lf)).item())
        return q_fast_target
