import os
import pathlib
from copy import deepcopy
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
from models.port_hamiltonian import FlowIntegrationError, integrate_drift


class ModelBasedTargetNumericalError(RuntimeError):
    """A model-based critic target failure safe for guarded fallback.

    This deliberately excludes arbitrary ``RuntimeError`` instances. Only
    explicitly detected non-finite target/flow conditions use this type, so an
    OOM or programming error cannot be mistaken for learned-model divergence.
    """


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
        dynamics_fit_horizon_warmup: int = 0,
        dynamics_duration_balance: bool = False,
        dynamics_integration_step: Optional[float] = None,
        dynamics_target_tau: float = 0.01,
        dynamics_publish_max_flow_error_ratio: Optional[float] = 2.0,
        dynamics_publish_batch_size: int = 32,
        dynamics_require_value_head: bool = True,
        generator_gate_scale: float = 0.0,
        value_warmup: int = 0,
        generator_substeps: int = 0,
        target_guard_kappa: float = 0.0,
        target_guard_cap: float = 0.0,
        dynamics_publish_interval: int = 1,
        dynamics_train_interval: int = 1,
        dynamics_rollout_interval: int = 1,
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
        # EXPLICIT guard mode for the model-based target (both default 0 = off;
        # every pre-existing mode is bit-identical with them off). kappa > 0
        # winsorizes the model-based target around the model-free finite-
        # difference anchor (per-sample outlier suppression relative to the
        # batch-consensus discrepancy); cap > 0 bounds |target| absolutely
        # (value-scale circuit breaker, ~3 x r_max/beta). Justified by the
        # corrected paired continuation (results/cartpole_fork_continuation2):
        # the learned-dynamics target causally halves recovery in model-poor
        # windows via magnitude/tail outliers while its action ordering stays
        # correct. See _guarded_model_based_target.
        self.target_guard_kappa = self._coerce_target_guard_parameter(
            "target_guard_kappa", target_guard_kappa
        )
        self.target_guard_cap = self._coerce_target_guard_parameter(
            "target_guard_cap", target_guard_cap
        )
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
        self.dynamics_target_tau = float(dynamics_target_tau)
        if not (0.0 < self.dynamics_target_tau <= 1.0):
            raise ValueError(
                "dynamics_target_tau must be in (0, 1], got "
                f"{self.dynamics_target_tau}"
            )
        if dynamics_publish_max_flow_error_ratio is not None:
            dynamics_publish_max_flow_error_ratio = float(
                dynamics_publish_max_flow_error_ratio
            )
            if (
                not np.isfinite(dynamics_publish_max_flow_error_ratio)
                or dynamics_publish_max_flow_error_ratio <= 0.0
            ):
                raise ValueError(
                    "dynamics_publish_max_flow_error_ratio must be finite and > 0 "
                    f"or None, got {dynamics_publish_max_flow_error_ratio}"
                )
        self.dynamics_publish_max_flow_error_ratio = (
            dynamics_publish_max_flow_error_ratio
        )
        self.dynamics_publish_batch_size = int(dynamics_publish_batch_size)
        if self.dynamics_publish_batch_size <= 0:
            raise ValueError(
                "dynamics_publish_batch_size must be > 0, got "
                f"{self.dynamics_publish_batch_size}"
            )
        # Full independent flow validation and target publication can be much
        # less frequent than live-model fitting. The critic continues reading
        # the last accepted frozen target between publications. A default of 1
        # preserves the historical every-update behavior.
        self.dynamics_publish_interval = int(dynamics_publish_interval)
        if self.dynamics_publish_interval <= 0:
            raise ValueError(
                "dynamics_publish_interval must be > 0, got "
                f"{self.dynamics_publish_interval}"
            )
        if isinstance(dynamics_require_value_head, str):
            dynamics_require_value_head = dynamics_require_value_head.strip().lower() in (
                "1", "true", "yes"
            )
        self.dynamics_require_value_head = bool(dynamics_require_value_head)
        # Multi-transition rollout fit: with horizon H > 1 the model is fit on its
        # own flow across H replay intervals instead of a single endpoint. Each
        # irregular interval is internally integrated at _integration_max_step,
        # matching how the generator/quadrature target consumes the vector field.
        self.dynamics_fit_horizon = max(1, int(dynamics_fit_horizon))
        # A long replay window now contains all internal physics-sized flow
        # steps. Start with single-transition fits, then enable the configured
        # outer horizon after this many dynamics updates to avoid an abrupt
        # 4 -> up-to-60-step BPTT graph on cartpole.
        self.dynamics_fit_horizon_warmup = int(dynamics_fit_horizon_warmup)
        if self.dynamics_fit_horizon_warmup < 0:
            raise ValueError("dynamics_fit_horizon_warmup must be non-negative")
        # Dynamics fitting is independent of critic optimization cadence. The
        # frozen accepted target makes skipped live-model fits safe, while the
        # replay buffer changes only slightly from one critic update to the next.
        self.dynamics_train_interval = int(dynamics_train_interval)
        if self.dynamics_train_interval <= 0:
            raise ValueError("dynamics_train_interval must be > 0")
        # After the H=1 curriculum, retain cheap local fits and inject the full
        # open-loop objective periodically instead of constructing its long BPTT
        # graph on every dynamics update. A value of 1 preserves legacy behavior.
        self.dynamics_rollout_interval = int(dynamics_rollout_interval)
        if self.dynamics_rollout_interval <= 0:
            raise ValueError("dynamics_rollout_interval must be > 0")
        if isinstance(dynamics_duration_balance, str):
            dynamics_duration_balance = dynamics_duration_balance.strip().lower() in (
                "1", "true", "yes"
            )
        # Endpoint errors grow approximately linearly in duration, so their
        # squared loss otherwise lets a sparse 30 ms tail dominate abundant
        # 2 ms data. This option retains and fully integrates every transition,
        # but normalizes its loss leverage at the internal integration scale.
        self.dynamics_duration_balance = bool(dynamics_duration_balance)
        self._dynamics_updates = 0
        self._dynamics_publications = 0
        self._last_dynamics_target_sync_update: Optional[int] = None
        self._dynamics_fit_rejections = 0
        self._dynamics_publish_rejections = 0
        self._dynamics_rollbacks = 0
        self._last_dynamics_rejection_reason: Optional[str] = None
        self._last_dynamics_mass_diagnostics: Dict[str, float] = {}
        self._train_dynamics = False
        self.dynamics_optimizer = None
        # Optional dedicated RNG for the learned-dynamics fit's replay sampling.
        # When None (default) the fit draws from the global np.random stream, as
        # before. Setting it to a numpy Generator isolates the fit's sampling so
        # it never advances the stream the critic/actor minibatches use -- this
        # keeps critic minibatches paired across target treatments that do or do
        # not fit dynamics (paired-continuation experiments).
        self._dynamics_sample_rng = None
        self.dynamics_target_model = self.dynamics_model
        if self.dynamics_model is not None:
            dyn_params = [
                p for p in self.dynamics_model.parameters() if p.requires_grad
            ]
            if dyn_params:
                self._train_dynamics = True
                self.dynamics_optimizer = th.optim.Adam(
                    dyn_params, lr=float(dynamics_lr)
                )
                # The critic never reads the live, optimizer-mutated model. It
                # reads this frozen copy, which is published only after a finite
                # post-fit flow check and then updated by EMA.
                self.dynamics_target_model = deepcopy(self.dynamics_model)
                if hasattr(self.dynamics_target_model, "to"):
                    self.dynamics_target_model.to(self.device)
                for p in self.dynamics_target_model.parameters():
                    p.requires_grad_(False)
                self.dynamics_target_model.eval()

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

        if (
            self._train_dynamics
            and self.generator_substeps >= 1
            and self.dynamics_require_value_head
            and not self.use_value_head
        ):
            raise ValueError(
                "Learned-dynamics quadrature requires an explicit V-head when "
                "dynamics_require_value_head=True; configure model_v_net_arch or "
                "set dynamics_require_value_head=False."
            )

        # For logging how many gradient updates we’ve done
        self._n_updates = 0

    @staticmethod
    def _coerce_target_guard_parameter(name: str, value: Any) -> float:
        """Parse an optional CSV guard value and reject unsafe settings."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return 0.0
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be finite and >= 0, got {value!r}"
            ) from exc
        if not np.isfinite(parsed) or parsed < 0.0:
            raise ValueError(
                f"{name} must be finite and >= 0, got {value!r}"
            )
        return parsed

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

    def _current_dynamics_fit_horizon(self) -> int:
        """Outer horizon for the next fit: curriculum plus periodic H>1 work."""
        if (
            self.dynamics_fit_horizon <= 1
            or self._dynamics_updates < self.dynamics_fit_horizon_warmup
        ):
            return 1
        fits_after_warmup = (
            self._dynamics_updates - self.dynamics_fit_horizon_warmup
        )
        if fits_after_warmup % self.dynamics_rollout_interval != 0:
            return 1
        return self.dynamics_fit_horizon

    def _dynamics_fit_due(self) -> bool:
        """Whether the current critic-gradient update also trains dynamics."""
        return self._n_updates % self.dynamics_train_interval == 0

    def _dynamics_balance_dt(self) -> Optional[float]:
        """Reference duration for flow-loss balancing, without dropping data."""
        if not self.dynamics_duration_balance:
            return None
        step = self._integration_max_step()
        if step is None:
            # No physics/integration resolution is exposed. dt_default is a
            # conservative reference and still caps only longer intervals.
            return float(self.dt_default)
        return float(step)

    def _dynamics_publication_due(self) -> bool:
        """Whether this fit update receives full independent validation."""
        return self._dynamics_updates % self.dynamics_publish_interval == 0

    def _live_dynamics_fit_failure_reason(
        self, dynamics_loss: float
    ) -> Optional[str]:
        """Cheap every-fit health check before any publication attempt.

        ``PortHamiltonianModel._fit_optimizer_step`` already verifies every
        optimized parameter after the step and exposes ``last_fit_accepted``.
        Trust that certificate instead of scanning the full state dictionary a
        second time. Generic learned models without the certificate retain the
        explicit finite-state scan.
        """
        if not np.isfinite(dynamics_loss):
            return "non-finite fit loss"
        if hasattr(self.dynamics_model, "last_fit_accepted"):
            if not bool(self.dynamics_model.last_fit_accepted):
                return "fit optimizer update was skipped"
            return None

        floating_state = [
            (name, value)
            for name, value in self.dynamics_model.state_dict().items()
            if value.is_floating_point()
        ]
        if floating_state and not bool(
            th.stack([th.isfinite(value).all() for _, value in floating_state]).all()
        ):
            bad_name = next(
                name
                for name, value in floating_state
                if not bool(th.isfinite(value).all())
            )
            return f"non-finite model state: {bad_name}"
        return None

    def _post_fit_flow_quality(
        self, obs, actions, next_obs, dt, dynamics_loss: float
    ) -> tuple[bool, float, str]:
        """Validate the live learned model before publishing it to the critic.

        The check uses the same finite-duration integrator as fitting and target
        construction.  Its scale-free error is endpoint RMSE divided by the RMS
        observed displacement, so ``1`` is the no-op predictor's error.  A
        nominal-duration roll is also required to remain finite because that is
        the exact path consumed by quadrature.
        """
        fit_failure = self._live_dynamics_fit_failure_reason(dynamics_loss)
        if fit_failure is not None:
            return False, float("inf"), fit_failure
        if (
            hasattr(self.dynamics_model, "mass_diagnostics")
            and getattr(self.dynamics_model, "mode", "structured") == "structured"
        ):
            diagnostics = self.dynamics_model.mass_diagnostics(
                obs[: min(self.dynamics_publish_batch_size, obs.shape[0])]
            )
            self._last_dynamics_mass_diagnostics = diagnostics
            for name, value in diagnostics.items():
                if not np.isfinite(value):
                    return (
                        False,
                        float("inf"),
                        f"non-finite mass diagnostic: {name}",
                    )
        try:
            with th.no_grad():
                check_n = min(self.dynamics_publish_batch_size, obs.shape[0])
                obs = obs[:check_n]
                actions = actions[:check_n]
                next_obs = next_obs[:check_n]
                dt = dt[:check_n]
                pred = integrate_drift(
                    self.dynamics_model.drift,
                    obs,
                    actions,
                    dt,
                    max_step=self._integration_max_step(),
                    check_finite=True,
                )
                nominal = integrate_drift(
                    self.dynamics_model.drift,
                    obs,
                    actions,
                    self.dt_default,
                    max_step=self._integration_max_step(),
                    check_finite=True,
                )

                target = th.as_tensor(
                    next_obs, dtype=pred.dtype, device=pred.device
                )
                start = th.as_tensor(obs, dtype=pred.dtype, device=pred.device)
                flow_rmse = th.sqrt(th.mean((pred - target) ** 2))
                displacement_rms = th.sqrt(th.mean((target - start) ** 2))
                ratio = float(
                    (flow_rmse / displacement_rms.clamp_min(1e-6)).detach()
                )
        except FlowIntegrationError as exc:
            return False, float("inf"), f"flow evaluation failed: {exc}"

        if not np.isfinite(ratio):
            return False, ratio, "non-finite flow error ratio"
        limit = self.dynamics_publish_max_flow_error_ratio
        if limit is not None and ratio > limit:
            return False, ratio, f"flow error ratio {ratio:.4g} exceeds {limit:.4g}"
        return True, ratio, "accepted"

    def _publish_dynamics_target(self) -> None:
        """Hard-publish first, then use cadence-correct EMA updates.

        If full validation occurs every ``k`` live-model updates, applying the
        configured per-update EMA coefficient once would make the target ``k``
        times slower. ``1 - (1 - tau)**k`` preserves its decay time constant.
        """
        assert self._train_dynamics and self.dynamics_target_model is not None
        if self._dynamics_publications == 0:
            tau = 1.0
        else:
            previous = self._last_dynamics_target_sync_update
            updates_since_publish = max(
                1,
                self._dynamics_updates - (
                    previous if previous is not None else self._dynamics_updates - 1
                ),
            )
            tau = 1.0 - (1.0 - self.dynamics_target_tau) ** updates_since_publish
        with th.no_grad():
            source = self.dynamics_model.state_dict()
            target = self.dynamics_target_model.state_dict()
            if source.keys() != target.keys():
                raise RuntimeError("live and target dynamics state dictionaries differ")
            for name, target_value in target.items():
                source_value = source[name].to(target_value.device)
                if target_value.is_floating_point():
                    target_value.mul_(1.0 - tau).add_(source_value, alpha=tau)
                else:
                    target_value.copy_(source_value)
        self.dynamics_target_model.eval()
        self._dynamics_publications += 1
        self._last_dynamics_target_sync_update = self._dynamics_updates

    def _rollback_live_dynamics(self) -> None:
        """Recover the learner after a non-finite fit from the last safe target."""
        self.dynamics_model.load_state_dict(self.dynamics_target_model.state_dict())
        self.dynamics_model.train()
        # Adam moments associated with the rejected trajectory can immediately
        # push the restored parameters back into the same bad region.
        self.dynamics_optimizer.state.clear()
        # The live learner now starts exactly at the accepted target. Count the
        # next EMA interval from this synchronization point, not from an older
        # publication that preceded the rejected trajectory.
        self._last_dynamics_target_sync_update = self._dynamics_updates
        self._dynamics_rollbacks += 1

    @property
    def _dynamics_ready(self) -> bool:
        if not self._train_dynamics:
            return True
        # Warmup is measured in fit updates, independently of publication
        # cadence. Even warmup=0 still requires one health-checked target.
        dynamics_ready = (
            self._dynamics_updates >= self.dynamics_warmup
            and self._dynamics_publications >= 1
        )
        value_ready = True
        if self.generator_substeps >= 1 and self.dynamics_require_value_head:
            value_ready = self._value_head_ready
        return dynamics_ready and value_ready

    @staticmethod
    def _require_finite_target_component(name: str, value: th.Tensor) -> None:
        finite = th.isfinite(value)
        if bool(th.all(finite)):
            return
        finite_values = value[finite]
        max_abs_finite = (
            float(finite_values.abs().max()) if finite_values.numel() else float("nan")
        )
        bad = int((~finite).sum())
        raise ModelBasedTargetNumericalError(
            "Model-based critic target is non-finite: "
            f"component={name}; bad={bad}/{value.numel()}, shape={tuple(value.shape)}, "
            f"max_abs_finite={max_abs_finite:.6g}."
        )

    @classmethod
    def _require_finite_target_components(
        cls, components: tuple[tuple[str, th.Tensor], ...]
    ) -> None:
        """One healthy-path synchronization, detailed diagnosis on failure."""
        all_finite = th.stack(
            [th.isfinite(value).all() for _, value in components]
        ).all()
        if bool(all_finite):
            return
        for name, value in components:
            # This second pass is taken only on failure and retains the precise
            # component/shape/count diagnostics used by benchmark failures.
            cls._require_finite_target_component(name, value)

    # ------------------------ persistence ------------------------

    @staticmethod
    def _dynamics_sidecar(path: Union[str, pathlib.Path]) -> str:
        """Sidecar file holding the learned dynamics model next to a checkpoint:
        ``best_model.pth`` -> ``best_model.dynamics.pth``."""
        root, ext = os.path.splitext(str(path))
        return root + ".dynamics" + (ext or ".pth")

    def save(self, path) -> None:
        """Save the actor-critic and the accepted dynamics used by its critic.

        The live optimizer model may be newer but rejected by the flow-health
        gate. Recovery audits and resumed targets must therefore load the frozen
        published model, not an unsafe candidate the critic never consumed.
        """
        super().save(path)
        if self._train_dynamics and isinstance(path, (str, pathlib.Path)):
            accepted = (
                self.dynamics_target_model
                if self._dynamics_publications > 0
                else self.dynamics_model
            )
            th.save(accepted.state_dict(), self._dynamics_sidecar(path))

    def load(self, path, strict: bool = True) -> "CTSAC":
        super().load(path, strict=strict)
        if self._train_dynamics and isinstance(path, (str, pathlib.Path)):
            sidecar = self._dynamics_sidecar(path)
            if os.path.exists(sidecar):
                self.dynamics_model.load_state_dict(
                    th.load(sidecar, map_location=self.device)
                )
                self.dynamics_target_model.load_state_dict(
                    self.dynamics_model.state_dict()
                )
                self.dynamics_target_model.eval()
                self._dynamics_publications = max(1, self._dynamics_publications)
                self._dynamics_updates = max(
                    self._dynamics_updates, self.dynamics_warmup
                )
                self._last_dynamics_target_sync_update = self._dynamics_updates
                if self.use_value_head:
                    self._value_updates = max(self._value_updates, self.value_warmup)
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
            if self._train_dynamics and self._dynamics_fit_due():
                fit_horizon = self._current_dynamics_fit_horizon()
                balance_dt = self._dynamics_balance_dt()
                if fit_horizon > 1:
                    # Multi-step rollout fit over a replay window: the model is
                    # rolled along its own predictions and every step regressed.
                    seq = self.replay_buffer.sample_sequences(
                        batch_size, fit_horizon, rng=self._dynamics_sample_rng
                    )
                    dynamics_loss = self.dynamics_model.fit_step_rollout(
                        seq.observations, seq.actions, seq.next_observations,
                        seq.dt, seq.mask, self.dynamics_optimizer,
                        max_step=self._integration_max_step(),
                        balance_dt=balance_dt,
                    )
                else:
                    dynamics_loss = self.dynamics_model.fit_step(
                        obs, actions, next_obs, dt, self.dynamics_optimizer,
                        max_step=self._integration_max_step(),
                        balance_dt=balance_dt,
                    )
                self._dynamics_updates += 1
                self.logger.record("train/dynamics_loss", dynamics_loss)
                if hasattr(self.dynamics_model, "last_fit_grad_norm"):
                    self.logger.record(
                        "train/dynamics_grad_norm",
                        self.dynamics_model.last_fit_grad_norm,
                    )
                self.logger.record("train/dynamics_fit_horizon", fit_horizon)
                if balance_dt is not None:
                    self.logger.record("train/dynamics_balance_dt", balance_dt)
                # Every fit receives the optimizer's cheap health certificate.
                # Full independent replay-flow validation and EMA publication
                # happen only on cadence; between them the critic keeps reading
                # the last accepted frozen target.
                fit_failure = self._live_dynamics_fit_failure_reason(
                    dynamics_loss
                )
                publish_attempted = False
                accepted = False
                flow_ratio = None
                reason = fit_failure
                if fit_failure is not None:
                    self._dynamics_fit_rejections += 1
                    self._last_dynamics_rejection_reason = fit_failure
                    self._rollback_live_dynamics()
                elif self._dynamics_publication_due():
                    publish_attempted = True
                    # Validate on a separate replay batch so publication never
                    # relies only on the data just optimized.
                    quality_batch = self.replay_buffer.sample(
                        min(batch_size, self.dynamics_publish_batch_size),
                        rng=self._dynamics_sample_rng,
                    )
                    accepted, flow_ratio, reason = self._post_fit_flow_quality(
                        quality_batch.observations,
                        quality_batch.actions,
                        quality_batch.next_observations,
                        quality_batch.dt,
                        dynamics_loss,
                    )
                    if accepted:
                        self._publish_dynamics_target()
                        self._last_dynamics_rejection_reason = None
                    else:
                        self._dynamics_publish_rejections += 1
                        self._last_dynamics_rejection_reason = reason
                        if (
                            reason.startswith("non-finite")
                            or reason.startswith("flow evaluation failed")
                            or reason.startswith("mass diagnostics failed")
                        ):
                            self._rollback_live_dynamics()
                if publish_attempted or fit_failure is not None:
                    self.logger.record(
                        "train/dynamics_publish_accepted", float(accepted)
                    )
                if flow_ratio is not None:
                    self.logger.record(
                        "train/dynamics_flow_error_ratio", flow_ratio
                    )
                self.logger.record(
                    "train/dynamics_publications", self._dynamics_publications
                )
                self.logger.record(
                    "train/dynamics_fit_rejections",
                    self._dynamics_fit_rejections,
                )
                self.logger.record(
                    "train/dynamics_publish_rejections",
                    self._dynamics_publish_rejections,
                )
                self.logger.record(
                    "train/dynamics_rollbacks", self._dynamics_rollbacks
                )
                for name, value in self._last_dynamics_mass_diagnostics.items():
                    self.logger.record(f"train/dynamics_mass_{name}", value)
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
            dynamics_ready = self._dynamics_ready
            if (
                self.use_model_based_q
                and self.dynamics_model is not None
                and dynamics_ready
            ):
                if self._target_guard_enabled:
                    q_fast_target = self._guarded_model_based_target(
                        obs, actions, next_obs, rewards, dones, dt, alpha_tensor
                    )
                else:
                    q_fast_target = self._model_based_target(
                        obs, actions, next_obs, rewards, dones, dt, alpha_tensor
                    )
                # Each model-based construction performs one aggregate finite
                # check and diagnoses the first bad component only on failure.
                # There is deliberately no model-free fallback outside the
                # explicit, separately-labeled guard mode (target_guard_*).
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

            self._n_updates += 1

        self.logger.record("train/n_updates", self._n_updates)

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
        self, obs, actions, next_obs, rewards, dones, dt, alpha_tensor,
        check: bool = True,
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
                obs, actions, rewards, dones, alpha_tensor, check=check
            )

        target_dynamics = self.dynamics_target_model
        sigma = target_dynamics.diffusion(obs)
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
        b = target_dynamics.drift(obs, actions)  # (B, O), per second
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
        if check:
            self._require_finite_target_components(
                (
                    ("V_cur", V_det),
                    ("drift", b),
                    ("value_gradient", gV),
                    ("value_increment", lf),
                    ("q_fast_target", q_fast_target),
                )
            )
        self.logger.record("train/fraction", th.max(th.abs(lf)).item())
        return q_fast_target

    def _substep_quadrature_target(
        self, obs, actions, rewards, dones, alpha_tensor, check: bool = True
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
            flow_args = (
                self.dynamics_target_model.drift,
                obs.detach(),
                actions,
                self.dt_default,
            )
            flow_kwargs = {"max_step": self._integration_max_step()}
            try:
                x_hat = integrate_drift(
                    *flow_args,
                    **flow_kwargs,
                )
            except FlowIntegrationError as exc:
                # An explicitly detected flow failure gets an eager diagnostic
                # replay; the healthy path performs no integration-time sync.
                try:
                    integrate_drift(
                        *flow_args,
                        **flow_kwargs,
                        check_finite="step",
                    )
                except FlowIntegrationError as diagnostic:
                    exc = diagnostic
                raise ModelBasedTargetNumericalError(
                    "Model-based critic target is non-finite or invalid: "
                    f"component=rolled_state; integration error: {exc}"
                ) from exc
            V_cur = self._state_value(obs, alpha_tensor)  # (B, 1)
            V_next = self._state_value(x_hat, alpha_tensor)  # (B, 1) at rolled state
            lf = (V_next - V_cur) - self.beta * V_cur
            q_fast_target = (rewards + (1 - dones) * (V_cur + lf)).detach()
            try:
                if check:
                    self._require_finite_target_components(
                        (
                            ("rolled_state", x_hat),
                            ("V_cur", V_cur),
                            ("V_next", V_next),
                            ("value_increment", lf),
                            ("q_fast_target", q_fast_target),
                        )
                    )
            except ModelBasedTargetNumericalError as component_error:
                # Recover internal-step detail only on the failed path. If the
                # flow is finite, preserve the original value-component error.
                try:
                    integrate_drift(
                        *flow_args,
                        **flow_kwargs,
                        check_finite="step",
                    )
                except FlowIntegrationError as diagnostic:
                    raise ModelBasedTargetNumericalError(
                        "Model-based critic target is non-finite or invalid: "
                        "component=rolled_state; integration error: "
                        f"{diagnostic}"
                    ) from diagnostic
                raise component_error
            self.logger.record(
                "train/model_rollout_max_abs", x_hat.abs().max().item()
            )
            self.logger.record(
                "train/model_value_next_max_abs", V_next.abs().max().item()
            )
            self.logger.record("train/fraction", th.max(th.abs(lf)).item())
        return q_fast_target

    @property
    def _target_guard_enabled(self) -> bool:
        return self.target_guard_kappa > 0.0 or self.target_guard_cap > 0.0

    def _guarded_model_based_target(
        self, obs, actions, next_obs, rewards, dones, dt, alpha_tensor
    ) -> th.Tensor:
        """Winsorized model-based target -- the EXPLICIT guard mode
        (``target_guard_kappa`` / ``target_guard_cap`` > 0; both default off).

        The corrected paired continuation (results/cartpole_fork_continuation2)
        showed the learned-dynamics target causally degrading recovery in
        model-poor windows -- return halved, variance doubled, critic loss
        ~100x -- while the within-state action ORDERING it induces stays
        correct (results/cartpole_action_grid_*). The harmful channel is
        magnitude/tail outliers inflating the critic/value scale, so the guard
        suppresses exactly that channel and passes everything else through:

          anchor     t_mf = model-free finite-difference target on the same
                            batch (realized next state; V-head read at data
                            points only, never at model predictions)
          discrepancy d   = t_model - t_mf                          (per sample)
          winsorize   t   = t_mf + med(d) + clip(d - med(d), +-kappa*MAD(d))
          cap        |t| <= target_guard_cap

        The batch median of d is kept, so the systematic higher-order
        correction the quadrature target carries over the finite difference
        survives; only per-sample outliers relative to the batch-consensus
        discrepancy are clamped. MAD is robust to the observed window
        contamination (a model-poor pocket is a minority of a uniform replay
        batch); the absolute cap backstops whole-batch corruption and the
        value-scale runaway (healthy cartpole targets are bounded by
        ~r_max/beta ~= 50; seed-1-style runaways sit orders of magnitude
        above). Non-finite model targets fall to the anchor and are counted;
        a non-finite anchor still raises -- the guard never invents a target.

        This is deliberately a separate, labeled mode: benchmark modes without
        ``target_guard_*`` keep the pure model-based-or-fail contract, and a
        guarded run is not a pure model-based result. The guard consumes no
        RNG (anchor and clamp are deterministic), so a guarded branch stays
        minibatch-paired with an unguarded one under the paired-continuation
        harness. Pairing assumes the V-head is ready (true whenever the
        quadrature target is active in the benchmark configs); an unready
        V-head would sample the soft value inside the anchor.
        """
        with th.no_grad():
            t_mf = self._finite_difference_target(
                obs, next_obs, rewards, dones, dt, alpha_tensor
            )
            self._require_finite_target_components((("guard_anchor", t_mf),))
            try:
                # The first-order generator needs autograd with respect to the
                # observation even though all guard arithmetic is no-grad.
                with th.enable_grad():
                    t_model = self._model_based_target(
                        obs, actions, next_obs, rewards, dones, dt, alpha_tensor,
                        check=False,
                    )
            except ModelBasedTargetNumericalError:
                t_model = None  # integration failure: whole batch to the anchor
            if t_model is None:
                delta = th.full_like(t_mf, float("nan"))
                finite = th.zeros_like(delta, dtype=th.bool)
                nonfinite_frac = 1.0
            else:
                if t_model.shape != t_mf.shape:
                    raise ValueError(
                        "guard model target shape must match its anchor, got "
                        f"{tuple(t_model.shape)} and {tuple(t_mf.shape)}"
                    )
                delta = t_model - t_mf
                finite = th.isfinite(delta)
                nonfinite_frac = float((~finite).float().mean())

            if bool(finite.any()):
                finite_delta = delta[finite]
                med = finite_delta.median()
                finite_centered = finite_delta - med
                mad = finite_centered.abs().median() * 1.4826
                # Exclude bad elements from both the robust statistics and the
                # arithmetic. They are restored exactly to their anchors below.
                safe_delta = th.where(finite, delta, med)
                centered = safe_delta - med
            else:
                med = delta.new_zeros(())
                mad = delta.new_zeros(())
                safe_delta = th.zeros_like(delta)
                centered = th.zeros_like(delta)

            clamp_frac = 0.0
            t = t_mf + safe_delta
            if self.target_guard_kappa > 0.0:
                # scale floor keeps the trust radius nondegenerate when the
                # model and the anchor agree to numerical precision
                floor = 1e-3 * (1.0 + t_mf.abs().median())
                radius = self.target_guard_kappa * th.clamp(mad, min=floor)
                clamped = finite & (centered.abs() > radius)
                clamp_frac = float(clamped.float().mean())
                t = t_mf + med + centered.clamp(-radius, radius)
            t = th.where(finite, t, t_mf)
            cap_frac = 0.0
            if self.target_guard_cap > 0.0:
                cap = self.target_guard_cap
                cap_frac = float((t.abs() > cap).float().mean())
                t = t.clamp(-cap, cap)

            self._require_finite_target_components((("guard_target", t),))

            self.logger.record("train/guard_delta_med", float(med))
            self.logger.record("train/guard_delta_mad", float(mad))
            self.logger.record("train/guard_clamp_frac", clamp_frac)
            self.logger.record("train/guard_cap_frac", cap_frac)
            self.logger.record("train/guard_nonfinite_frac", nonfinite_frac)
        return t.detach()
