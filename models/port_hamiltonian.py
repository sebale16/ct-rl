# models/port_hamiltonian.py
"""Port-Hamiltonian dynamics model for the model-based generator in CT-SAC.

This supplies the drift ``b(x, a)`` (the "b term" of the controlled generator,
Eq. 6 of the Hamiltonian-Flow paper) and an optional diffusion ``sigma(x)`` so
that the generator ``(L^a V) = b . grad V + 1/2 Tr(sigma sigma^T Hess V)`` can be
evaluated analytically, without sampling a successor state.

Two modes:

- ``"mujoco"``: the drift is provided by a callable (e.g.
  ``DMCContinuousEnv.dynamics_terms``), i.e. the simulator's exact
  observation-space drift. No learnable parameters. Used as the validation
  oracle (Milestone M0).

- ``"phast"``: a learned, structure-preserving drift
  ``b(x, a) = (J - R) grad H(x) + G_a a`` with ``J`` skew-symmetric, ``R`` a
  positive-semidefinite (low-rank Householder) dissipation, ``H`` a scalar energy
  network, and ``G_a`` a linear actuation (port) map. This is the UNKNOWN-regime
  PHAST model on the observation treated as a generalized phase state. The full
  state-dependent damping ``D(q)`` and the Strang integrator are deferred (see
  ``docs/port_hamiltonian_ct_sac.md``).

The diffusion is ``None`` when ``human_input_intensity == 0`` (the v1 default,
``sigma = 0``); the fluctuation-dissipation form ``sigma sigma^T = 2 T D(q)`` is a
later milestone (M2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch as th
from torch import nn
from torch.func import grad, jacfwd, vmap


class FlowIntegrationError(RuntimeError):
    """A numerical flow failure explicitly detected by ``integrate_drift``.

    The type deliberately excludes arbitrary exceptions raised by ``drift_fn``.
    Callers may recover from a detected non-finite flow without accidentally
    swallowing OOMs, tensor-shape bugs, or other programming errors.
    """


def _inverse_softplus(value: float) -> float:
    """Stable scalar inverse of softplus for positive parameter initializers."""
    x = max(float(value), 1e-8)
    return float(x + np.log(-np.expm1(-x)))


def integrate_drift(
    drift_fn: Callable[[th.Tensor, th.Tensor], th.Tensor],
    obs,
    action,
    duration,
    *,
    max_step: Optional[float] = None,
    delta_limit: Optional[th.Tensor] = None,
    clamp_final: bool = True,
    check_finite: bool | str = False,
) -> th.Tensor:
    """Integrate a batched drift over possibly different durations.

    ``duration`` is the observed/control interval for each sample.  ``max_step``
    is only the internal explicit-Euler resolution: every sample keeps its full
    duration and is advanced ``ceil(duration / max_step)`` times while its action
    is held fixed.  With ``max_step=None`` this reduces exactly to the historical
    one-Euler-step predictor.

    Active samples are gathered at each internal step rather than evaluating the
    whole batch ``max(n_steps)`` times.  This matters for the benchmark's two-tail
    schedule, where most transitions need one physics-sized step and only the
    sparse long-duration transitions need many.  ``index_copy`` keeps the roll
    differentiable, so this helper can be shared by dynamics fitting and the
    no-grad critic target. ``delta_limit`` optionally bounds the cumulative
    displacement from the interval's starting state after every internal step;
    the rollout fit uses this as its off-manifold overflow guard. Setting
    ``clamp_final=False`` guards only intermediate states and leaves the endpoint
    loss unsaturated, so a bad prediction still receives a corrective gradient.
    ``check_finite=True`` performs one healthy-path endpoint check. If the
    endpoint is non-finite, the integration is replayed once with stepwise
    checks to report the first failing internal step. Exceptions raised by
    ``drift_fn`` propagate unchanged; in particular, OOMs and programming errors
    are never relabelled as recoverable numerical flow failures. This avoids an
    accelerator synchronization after every substep while retaining actionable
    failure diagnostics. ``check_finite="step"`` requests the historical eager
    stepwise checking explicitly; ``False`` disables it.
    """
    x = obs if isinstance(obs, th.Tensor) else th.as_tensor(obs, dtype=th.float32)
    if not x.is_floating_point():
        x = x.float()
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2:
        raise ValueError(f"obs must have shape (batch, obs_dim), got {tuple(x.shape)}")

    if isinstance(check_finite, bool):
        finite_mode = "endpoint" if check_finite else "none"
    elif check_finite in ("none", "endpoint", "step"):
        finite_mode = str(check_finite)
    else:
        raise ValueError(
            "check_finite must be a bool or one of 'none', 'endpoint', 'step'"
        )

    a = th.as_tensor(action, dtype=x.dtype, device=x.device)
    if a.ndim == 1:
        a = a.unsqueeze(0)
    if a.ndim != 2 or a.shape[0] != x.shape[0]:
        raise ValueError(
            "action must have shape (batch, action_dim) with the same batch "
            f"as obs, got {tuple(a.shape)} for batch {x.shape[0]}"
        )

    dt = th.as_tensor(duration, dtype=x.dtype, device=x.device)
    if dt.numel() == 1:
        dt = dt.reshape(1, 1).expand(x.shape[0], 1)
    elif dt.numel() == x.shape[0]:
        dt = dt.reshape(x.shape[0], 1)
    else:
        raise ValueError(
            "duration must be scalar or have one value per batch item, got "
            f"shape {tuple(dt.shape)} for batch {x.shape[0]}"
        )
    if not bool(th.all(th.isfinite(dt))):
        raise ValueError("duration values must be finite")
    if bool(th.any(dt < 0.0)):
        raise ValueError("duration values must be non-negative")

    if max_step is None:
        n_steps = (dt > 0.0).to(dtype=th.long)
    else:
        max_step = float(max_step)
        if not np.isfinite(max_step) or max_step <= 0.0:
            raise ValueError(f"max_step must be finite and > 0, got {max_step}")
        # Subtract a tiny tolerance in step-count units so a float32 value such
        # as 0.010000001 does not spuriously turn five 2 ms steps into six.
        n_steps = th.ceil(dt / max_step - 1e-6).clamp_min(0).to(dtype=th.long)
        n_steps = th.where(dt > 0.0, n_steps.clamp_min(1), n_steps)

    step_size = th.where(
        n_steps > 0,
        dt / n_steps.clamp_min(1).to(dtype=dt.dtype),
        th.zeros_like(dt),
    )
    max_n = int(n_steps.max().item()) if n_steps.numel() else 0
    limit = None
    if delta_limit is not None:
        limit = th.as_tensor(delta_limit, dtype=x.dtype, device=x.device)
        if limit.numel() == 1:
            limit = limit.reshape(1, 1).expand_as(x)
        elif limit.shape == (x.shape[0], 1):
            limit = limit.expand_as(x)
        elif limit.shape != x.shape:
            raise ValueError(
                "delta_limit must be scalar, (batch, 1), or match obs shape; "
                f"got {tuple(limit.shape)} for obs {tuple(x.shape)}"
            )
        if not bool(th.all(th.isfinite(limit))) or bool(th.any(limit < 0.0)):
            raise ValueError("delta_limit values must be finite and non-negative")

    def _run(
        x_seed: th.Tensor,
        a_seed: th.Tensor,
        limit_seed: Optional[th.Tensor],
        *,
        check_steps: bool,
    ) -> th.Tensor:
        x_hat = x_seed.clone()
        x_start = x_seed.clone()
        for k in range(max_n):
            active = th.nonzero(n_steps[:, 0] > k, as_tuple=False).squeeze(-1)
            if active.numel() == 0:
                break
            x_k = x_hat.index_select(0, active)
            a_k = a_seed.index_select(0, active)
            b_k = drift_fn(x_k, a_k)
            b_k = th.as_tensor(b_k, dtype=x.dtype, device=x.device)
            if b_k.shape != x_k.shape:
                raise ValueError(
                    f"drift must return shape {tuple(x_k.shape)}, got {tuple(b_k.shape)}"
                )
            if check_steps and not bool(th.all(th.isfinite(b_k))):
                raise FlowIntegrationError(
                    "non-finite drift at internal flow step "
                    f"{k + 1}/{max_n} ({active.numel()} active samples)"
                )
            h_k = step_size.index_select(0, active)
            x_next = x_k + b_k * h_k
            if limit_seed is not None:
                x0_k = x_start.index_select(0, active)
                lim_k = limit_seed.index_select(0, active)
                bounded = x0_k + th.clamp(x_next - x0_k, -lim_k, lim_k)
                if clamp_final:
                    x_next = bounded
                else:
                    is_final = (
                        n_steps.index_select(0, active)[:, 0] == (k + 1)
                    ).unsqueeze(-1)
                    x_next = th.where(is_final, x_next, bounded)
            if check_steps and not bool(th.all(th.isfinite(x_next))):
                raise FlowIntegrationError(
                    "non-finite state at internal flow step "
                    f"{k + 1}/{max_n} ({active.numel()} active samples)"
                )
            x_hat = x_hat.index_copy(0, active, x_next)
        return x_hat

    if finite_mode == "none":
        return _run(x, a, limit, check_steps=False)
    if finite_mode == "step":
        return _run(x, a, limit, check_steps=True)

    # Production finite checking: one synchronization when the endpoint is
    # healthy. Only the exceptional path pays for a second, diagnostic rollout.
    original_error = None
    try:
        endpoint = _run(x, a, limit, check_steps=False)
    except FlowIntegrationError as exc:
        original_error = exc
        endpoint = None
    if endpoint is not None and bool(th.all(th.isfinite(endpoint))):
        return endpoint

    try:
        with th.no_grad():
            diagnostic_endpoint = _run(
                x.detach(),
                a.detach(),
                None if limit is None else limit.detach(),
                check_steps=True,
            )
            if not bool(th.all(th.isfinite(diagnostic_endpoint))):
                raise FlowIntegrationError(
                    "non-finite state at diagnostic flow endpoint "
                    f"after {max_n} internal steps"
                )
    except FlowIntegrationError as diagnostic_error:
        raise FlowIntegrationError(
            "flow integration failed; diagnostic replay found: "
            f"{diagnostic_error}"
        ) from (original_error or diagnostic_error)

    if original_error is not None:
        raise FlowIntegrationError(
            "flow integration failed, but the stepwise diagnostic replay was finite: "
            f"{original_error}"
        ) from original_error
    raise FlowIntegrationError(
        "flow endpoint is non-finite, but the stepwise diagnostic replay was finite"
    )


# --------------------------------------------------------------------------
# Analytic derivatives of the small mechanics MLPs.
#
# The structured drift needs derivatives of M(q), V(q) and the contact geometry
# with respect to position.  Taking them with ``torch.func`` (vmap + jacfwd) is
# concise but routes every operation through functorch's Python decomposition
# layer: a profile of the cheetah contact arm attributed 36% of the training
# step to four such call sites, and roughly a quarter of that was interpreter
# overhead (torch/_refs, _prims_common, inspect.Signature.bind) rather than
# arithmetic.  These helpers propagate the same derivatives through a plain
# Linear/activation stack as a handful of batched matmuls, which is both exactly
# the same function (agreement ~1e-6 relative in float32) and 5x faster on the
# mass block.  They deliberately support only the layer types the mechanics nets
# are built from, and raise on anything else rather than silently disagreeing
# with autograd.
# --------------------------------------------------------------------------


def _activation_derivative(module: nn.Module, pre: th.Tensor) -> th.Tensor:
    """Elementwise derivative of ``module`` evaluated at its pre-activation."""
    if isinstance(module, nn.SiLU):
        sig = th.sigmoid(pre)
        return sig * (1.0 + pre * (1.0 - sig))
    if isinstance(module, nn.Tanh):
        return 1.0 - th.tanh(pre).square()
    if isinstance(module, nn.ReLU):
        # Matches autograd's subgradient choice at exactly zero.
        return (pre > 0).to(pre.dtype)
    if isinstance(module, nn.ELU):
        alpha = float(module.alpha)
        return th.where(pre > 0, th.ones_like(pre), alpha * th.exp(pre))
    raise TypeError(
        "analytic mechanics derivatives do not support activation "
        f"{type(module).__name__}; add its derivative or construct the model "
        "with analytic_derivatives=False"
    )


def _mlp_trace(net: nn.Sequential, x: th.Tensor):
    """Forward pass that also returns what the JVP/VJP passes need.

    Returns ``(y, layers)`` where ``layers`` is one ``(weight, act_prime)`` pair
    per linear layer, ``act_prime`` being ``None`` for a layer with no following
    activation.  The trace is built from ordinary autograd-tracked ops, so the
    derivative passes below stay differentiable with respect to the weights --
    the dynamics fit trains through them.
    """
    # A bare Linear is a valid one-layer net (tests and recovery diagnostics
    # substitute one for the learned contact geometry).
    modules = list(net) if isinstance(net, nn.Sequential) else [net]
    layers: list[tuple[th.Tensor, Optional[th.Tensor]]] = []
    h = x
    index = 0
    while index < len(modules):
        linear = modules[index]
        if not isinstance(linear, nn.Linear):
            raise TypeError(
                "analytic mechanics derivatives expect a Linear/activation "
                f"stack, found {type(linear).__name__}"
            )
        pre = th.nn.functional.linear(h, linear.weight, linear.bias)
        index += 1
        act_prime = None
        if index < len(modules):
            activation = modules[index]
            act_prime = _activation_derivative(activation, pre)
            h = activation(pre)
            index += 1
        else:
            h = pre
        layers.append((linear.weight, act_prime))
    return h, layers


def _mlp_jvp(layers, tangent: th.Tensor) -> th.Tensor:
    """Push one input tangent forward through a traced MLP: (..., in) -> (..., out)."""
    out = tangent
    for weight, act_prime in layers:
        out = out @ weight.t()
        if act_prime is not None:
            out = out * act_prime
    return out


def _mlp_vjp(layers, cotangent: th.Tensor) -> th.Tensor:
    """Pull one output cotangent back through a traced MLP: (..., out) -> (..., in).

    Extra leading dimensions are allowed, so seeding an identity cotangent gives
    the full Jacobian in one pass (cheaper than forward mode whenever the output
    is narrower than the input, as it is for the contact geometry).
    """
    grad_out = cotangent
    for weight, act_prime in reversed(layers):
        if act_prime is not None:
            while act_prime.ndim < grad_out.ndim:
                act_prime = act_prime.unsqueeze(-2)
            grad_out = grad_out * act_prime
        grad_out = grad_out @ weight
    return grad_out


def _mlp_jacobian(layers, out_dim: int, reference: th.Tensor) -> th.Tensor:
    """Full Jacobian (batch, out_dim, in_dim) by reverse mode over the outputs."""
    eye = th.eye(out_dim, dtype=reference.dtype, device=reference.device)
    return _mlp_vjp(layers, eye.expand(reference.shape[0], out_dim, out_dim))


def _scale_diagonal(matrix: th.Tensor, scale: th.Tensor) -> th.Tensor:
    """Out-of-place ``diag(matrix) *= scale``, batched.

    The Cholesky factor's diagonal passes through softplus, so its tangent and
    its cotangent both pick up that derivative on the diagonal only.
    """
    diagonal = matrix.diagonal(dim1=-2, dim2=-1)
    return matrix - th.diag_embed(diagonal) + th.diag_embed(scale * diagonal)


@dataclass(frozen=True)
class DOFLayout:
    """Maps a flat observation to a manipulator (q, qd) phase state for the
    structured port-Hamiltonian model. Kept domain-agnostic: every DOF fact the
    model needs is a field here, so nothing about a particular robot is hardcoded.

    obs        = [ obs[pos_slice] (generalized positions q, minus any cyclic ones),
                   obs[vel_slice] (generalized velocities qd, one per config DOF) ]
    nv (= number of config DOFs) may exceed npos when cyclic coordinates are
    dropped from the observation (e.g. cheetah drops the root-x position but keeps
    its velocity). cyclic_cfg lists those config-DOF indices; M and V do not depend
    on them, so their config-gradient slot is held at zero. obs_pos_to_cfg[i] is the
    config-DOF index of the i-th observed position. act_to_cfg (if given) maps each
    actuator to the config DOF it forces; None means a dense action->force map.

    Contact-port fields (used only when the model is built with contact_force > 0):
    m_invariant_pos lists observed-position indices the mass matrix must not
    depend on (translation-invariant coordinates, e.g. the root height: rigid-body
    inertia is invariant under translating the whole mechanism vertically). They
    are excluded from the mass network's input, so dM/dq is exactly zero there and
    ground reaction cannot be absorbed into M as a contact proxy.
    contact_tangent_cfg is the config-DOF index of the horizontal translation that
    contact friction pushes against (cheetah: the cyclic root x).

    Mechanism fields are opt-in so legacy layouts retain their architecture:
    ``periodic_pos`` feeds sin/cos rather than a raw angle to the mechanical
    networks; ``potential_invariant_pos`` removes known cyclic translations from
    the learned base potential; and ``joint_limits`` adds explicit conservative
    spring storage plus localized passive damping at known rails.
    """

    obs_dim: int
    pos_slice: Tuple[int, int]
    vel_slice: Tuple[int, int]
    cyclic_cfg: Tuple[int, ...]
    obs_pos_to_cfg: Tuple[int, ...]
    act_to_cfg: Optional[Tuple[int, ...]] = None
    m_invariant_pos: Tuple[int, ...] = ()
    contact_tangent_cfg: Optional[int] = None
    # Optional structure known from the mechanism.  Defaults deliberately keep
    # the historical cheetah/raw-state architecture unchanged.
    enforce_m_invariance: bool = False
    potential_invariant_pos: Tuple[int, ...] = ()
    periodic_pos: Tuple[int, ...] = ()
    damping_log_init: float = -2.0
    base_damping_reg: float = 0.0
    # Smooth unilateral joint limits, as (observed-position index, lower,
    # upper).  The base learned potential remains independent of any coordinate
    # in ``potential_invariant_pos``; limit energy is added explicitly.
    joint_limits: Tuple[Tuple[int, float, float], ...] = ()
    joint_limit_width: float = 0.02
    joint_limit_stiffness_init: float = 1.0
    joint_limit_damping_init: float = 0.1

    @property
    def npos(self) -> int:
        return self.pos_slice[1] - self.pos_slice[0]

    @property
    def nv(self) -> int:
        return self.vel_slice[1] - self.vel_slice[0]

    def __post_init__(self) -> None:
        assert self.npos == len(self.obs_pos_to_cfg), "obs_pos_to_cfg must have npos entries"
        assert self.npos + self.nv == self.obs_dim, "pos_slice + vel_slice must cover obs_dim"
        covered = list(range(*self.pos_slice)) + list(range(*self.vel_slice))
        assert sorted(covered) == list(range(self.obs_dim)), (
            "pos_slice and vel_slice must partition [0, obs_dim) with no overlap or gaps"
        )
        assert set(self.obs_pos_to_cfg) | set(self.cyclic_cfg) == set(range(self.nv)), (
            "every config DOF must be declared either observed (obs_pos_to_cfg) or cyclic"
        )
        assert all(0 <= c < self.nv for c in self.obs_pos_to_cfg), "config index out of range"
        assert all(0 <= c < self.nv for c in self.cyclic_cfg), "cyclic index out of range"
        assert set(self.obs_pos_to_cfg).isdisjoint(self.cyclic_cfg), (
            "an observed position cannot map to a cyclic config DOF"
        )
        if self.act_to_cfg is not None:
            assert all(0 <= c < self.nv for c in self.act_to_cfg), "actuator index out of range"
        assert all(0 <= i < self.npos for i in self.m_invariant_pos), (
            "m_invariant_pos indexes observed positions, so entries must be in [0, npos)"
        )
        assert all(0 <= i < self.npos for i in self.potential_invariant_pos), (
            "potential_invariant_pos indexes observed positions"
        )
        assert all(0 <= i < self.npos for i in self.periodic_pos), (
            "periodic_pos indexes observed positions"
        )
        assert len(set(self.m_invariant_pos)) == len(self.m_invariant_pos)
        assert len(set(self.potential_invariant_pos)) == len(self.potential_invariant_pos)
        assert len(set(self.periodic_pos)) == len(self.periodic_pos)
        assert np.isfinite(self.damping_log_init)
        assert np.isfinite(self.base_damping_reg) and self.base_damping_reg >= 0.0
        assert np.isfinite(self.joint_limit_width) and self.joint_limit_width > 0.0
        assert (
            np.isfinite(self.joint_limit_stiffness_init)
            and self.joint_limit_stiffness_init > 0.0
        )
        assert (
            np.isfinite(self.joint_limit_damping_init)
            and self.joint_limit_damping_init >= 0.0
        )
        seen_limits = set()
        for pos_i, lower, upper in self.joint_limits:
            assert 0 <= pos_i < self.npos, "joint limit position index out of range"
            assert pos_i not in seen_limits, "a position may have at most one joint limit"
            assert np.isfinite(lower) and np.isfinite(upper) and lower < upper, (
                "joint limit bounds must be finite and ordered"
            )
            seen_limits.add(pos_i)
        if self.contact_tangent_cfg is not None:
            assert 0 <= self.contact_tangent_cfg < self.nv, "contact_tangent_cfg out of range"

    @classmethod
    def raw_state(cls, nv: int, act_to_cfg: Optional[Tuple[int, ...]] = None) -> "DOFLayout":
        """Layout for a raw-state observation [qpos (nv); qvel (nv)] — every
        config DOF observed as a plain coordinate (hinge/slide joints, nq == nv;
        the DMCContinuousEnv ``raw_state_obs`` option). No cyclic coordinates
        and no contact-port fields; suits smooth low-DOF validation systems
        (cartpole, acrobot, pendulum)."""
        return cls(
            obs_dim=2 * int(nv),
            pos_slice=(0, int(nv)),
            vel_slice=(int(nv), 2 * int(nv)),
            cyclic_cfg=(),
            obs_pos_to_cfg=tuple(range(int(nv))),
            act_to_cfg=act_to_cfg,
        )

    @classmethod
    def cartpole(
        cls,
        slider_limit: Tuple[float, float] = (-1.8, 1.8),
        *,
        num_poles: int = 1,
        joint_limit_width: float = 0.02,
        joint_limit_stiffness_init: float = 100.0,
        joint_limit_damping_init: float = 1.0,
        base_damping_reg: float = 1e-3,
    ) -> "DOFLayout":
        """Mechanics-aware layout for a raw-state dm_control CartPole chain.

        There is one slider plus ``num_poles`` serial hinge coordinates.  The
        state is ``[cart_x, hinge_angles..., cart_velocity, hinge_velocities...]``;
        downstream hinge angles are relative to their parent links.  Rigid-body
        inertia and gravity are invariant to cart translation, while every
        hinge coordinate is periodic.  The rail limits are represented by a
        separate smooth unilateral potential/dissipation port, and the single
        actuator applies generalized force only to the cart slider.
        """
        num_poles = int(num_poles)
        if num_poles < 1:
            raise ValueError(f"num_poles must be >= 1, got {num_poles}")
        nv = num_poles + 1
        lower, upper = (float(slider_limit[0]), float(slider_limit[1]))
        return cls(
            obs_dim=2 * nv,
            pos_slice=(0, nv),
            vel_slice=(nv, 2 * nv),
            cyclic_cfg=(),
            obs_pos_to_cfg=tuple(range(nv)),
            act_to_cfg=(0,),
            m_invariant_pos=(0,),
            enforce_m_invariance=True,
            potential_invariant_pos=(0,),
            periodic_pos=tuple(range(1, nv)),
            damping_log_init=-8.0,
            base_damping_reg=float(base_damping_reg),
            joint_limits=((0, lower, upper),),
            joint_limit_width=float(joint_limit_width),
            joint_limit_stiffness_init=float(joint_limit_stiffness_init),
            joint_limit_damping_init=float(joint_limit_damping_init),
        )

    @classmethod
    def acrobot(
        cls,
        *,
        joint_damping: float = 0.05,
        base_damping_reg: float = 1e-3,
    ) -> "DOFLayout":
        """Mechanics-aware layout for dm_control's raw-state Acrobot.

        The state is ``[shoulder, relative_elbow, shoulder_velocity,
        elbow_velocity]``.  Both hinges are periodic, the mass matrix is
        invariant to absolute shoulder rotation, gravity keeps the potential
        shoulder-dependent, and the sole motor applies torque only at the
        elbow.  ``cyclic_cfg`` remains empty because both positions are observed;
        in this layout it denotes omitted coordinates, not physical periodicity.
        """
        joint_damping = float(joint_damping)
        if not np.isfinite(joint_damping) or joint_damping <= 0.0:
            raise ValueError("joint_damping must be finite and > 0")
        return cls(
            obs_dim=4,
            pos_slice=(0, 2),
            vel_slice=(2, 4),
            cyclic_cfg=(),
            obs_pos_to_cfg=(0, 1),
            act_to_cfg=(1,),
            m_invariant_pos=(0,),
            enforce_m_invariance=True,
            potential_invariant_pos=(),
            periodic_pos=(0, 1),
            damping_log_init=_inverse_softplus(joint_damping),
            base_damping_reg=float(base_damping_reg),
        )

    @classmethod
    def cheetah(cls, obs_dim: int = 17, action_dim: int = 6) -> "DOFLayout":
        # obs = [qpos[1:] (8), qvel (9)]; root x (config DOF 0) is cyclic (dropped
        # from qpos, kept in qvel). Dense G_a(6->9) reproduces the validated prototype.
        assert obs_dim == 17 and action_dim == 6, (
            "DOFLayout.cheetah defaults assume obs_dim=17, action_dim=6; "
            "pass an explicit dof_layout for other environments."
        )
        return cls(
            obs_dim=17,
            pos_slice=(0, 8),
            vel_slice=(8, 17),
            cyclic_cfg=(0,),
            obs_pos_to_cfg=tuple(range(1, 9)),
            act_to_cfg=None,
            m_invariant_pos=(0,),   # root z: M is invariant to vertical translation
            contact_tangent_cfg=0,  # friction acts along the cyclic root x
        )


class PortHamiltonianModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        mode: str = "mujoco",
        drift_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        hidden: Sequence[int] = (256, 256),
        dissipation_rank: int = 4,
        human_input_intensity: float = 0.0,
        contact_force: int = 0,
        contact_solver: str = "constraint",
        contact_dt: float = 0.002,
        contact_iterations: int = 12,
        contact_regularization: float = 0.01,
        device: str | th.device = "cpu",
        dof_layout: Optional[DOFLayout] = None,
        structured_hidden: Sequence[int] = (128, 128),
        mass_logdet_reg: float = 0.0,
        mass_condition_reg: float = 0.0,
        mass_condition_limit: float = 1e3,
        analytic_derivatives: bool = True,
    ) -> None:
        super().__init__()
        # Position derivatives of the mechanics nets as explicit batched matmuls
        # instead of vmap(jacfwd(...)). Same function to float32 roundoff; set
        # False to fall back to the functorch reference path (the equivalence
        # tests exercise both).
        self.analytic_derivatives = bool(analytic_derivatives)
        self.mode = str(mode)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.human_input_intensity = float(human_input_intensity)
        self.contact_force = int(contact_force)
        self.contact_solver = str(contact_solver).strip().lower()
        self.contact_dt = float(contact_dt)
        self.contact_iterations = int(contact_iterations)
        self.contact_regularization = float(contact_regularization)
        self.device = th.device(device)
        self._drift_fn = drift_fn
        self.last_fit_accepted: bool = False
        self.last_fit_grad_norm: float = float("nan")
        self.mass_logdet_reg = float(mass_logdet_reg)
        self.mass_condition_reg = float(mass_condition_reg)
        self.mass_condition_limit = float(mass_condition_limit)
        if self.mass_logdet_reg < 0.0 or not np.isfinite(self.mass_logdet_reg):
            raise ValueError("mass_logdet_reg must be finite and non-negative")
        if self.mass_condition_reg < 0.0 or not np.isfinite(self.mass_condition_reg):
            raise ValueError("mass_condition_reg must be finite and non-negative")
        if self.mass_condition_limit <= 1.0 or not np.isfinite(self.mass_condition_limit):
            raise ValueError("mass_condition_limit must be finite and greater than 1")
        if self.contact_force > 0 and self.mode != "structured":
            raise ValueError(
                "contact_force > 0 (the explicit contact port) requires mode='structured'."
            )
        if self.contact_solver not in ("compliant", "constraint"):
            raise ValueError(
                "contact_solver must be either 'compliant' or 'constraint', "
                f"got {contact_solver!r}"
            )
        if not np.isfinite(self.contact_dt) or self.contact_dt <= 0.0:
            raise ValueError("contact_dt must be finite and > 0")
        if self.contact_iterations < 1:
            raise ValueError("contact_iterations must be at least 1")
        if (
            not np.isfinite(self.contact_regularization)
            or self.contact_regularization <= 0.0
        ):
            raise ValueError("contact_regularization must be finite and > 0")

        if self.mode == "mujoco":
            if drift_fn is None:
                raise ValueError(
                    "mode='mujoco' requires drift_fn (e.g. env.dynamics_terms)."
                )
        elif self.mode == "phast":
            d = self.obs_dim
            # Scalar energy H_theta(x)
            layers: list[nn.Module] = []
            last = d
            for h in hidden:
                layers += [nn.Linear(last, h), nn.SiLU()]
                last = h
            layers += [nn.Linear(last, 1)]
            self.energy = nn.Sequential(*layers)
            # Skew-symmetric J = A - A^T
            self._J_raw = nn.Parameter(0.01 * th.randn(d, d))
            # PSD dissipation R = softplus(d0) I + L L^T  (constant; PHAST Eq. 10 low-rank)
            self._d0 = nn.Parameter(th.tensor(-2.0))
            self._L = nn.Parameter(0.01 * th.randn(d, int(dissipation_rank)))
            # Agent port G_a: action -> state drift
            self.G_a = nn.Linear(self.action_dim, d, bias=False)
        elif self.mode == "structured":
            self._init_structured(dof_layout, structured_hidden)
        else:
            raise ValueError(f"Unknown mode '{self.mode}'.")

        self.to(self.device)

    # ------------------------ structured port-Hamiltonian (DeLaN core) ----------

    def _init_structured(
        self, dof_layout: Optional[DOFLayout], hidden: Sequence[int]
    ) -> None:
        """Structured model: a learned SPD mass matrix M(q) and potential V(q) give
        the Hamiltonian H = V(q) + 1/2 p^T M(q)^-1 p with the canonicalizer
        p = M(q) qd; the drift is the port-Hamiltonian flow (J - R)grad H + G_a a
        with a constant diagonal damping D on momentum (passivity dH/dt <= 0);
        contact enters through the explicit port (contact_force > 0). The
        Coriolis terms are generated from M via autodiff, not learned."""
        layout = dof_layout if dof_layout is not None else DOFLayout.cheetah(
            self.obs_dim, self.action_dim
        )
        assert layout.obs_dim == self.obs_dim, "dof_layout.obs_dim must match obs_dim"
        self.layout = layout
        self.base_damping_reg = float(layout.base_damping_reg)
        nv, npos = layout.nv, layout.npos

        def mlp(out: int, inp: int = npos) -> nn.Sequential:
            layers: list[nn.Module] = []
            last = inp
            for h in hidden:
                layers += [nn.Linear(last, h), nn.SiLU()]
                last = h
            layers += [nn.Linear(last, out)]
            return nn.Sequential(*layers)

        # Existing cheetah checkpoints excluded the translation coordinate only
        # with the explicit contact port.  ``enforce_m_invariance`` opts a layout
        # (cartpole) into the same exact invariant without changing that default.
        mass_excluded = (
            tuple(layout.m_invariant_pos)
            if layout.m_invariant_pos
            and (self.contact_force > 0 or layout.enforce_m_invariance)
            else ()
        )
        potential_excluded = tuple(layout.potential_invariant_pos)
        self._mass_excluded_pos = mass_excluded
        self._potential_excluded_pos = potential_excluded
        self._periodic_pos = tuple(layout.periodic_pos)
        # Index plans that let the analytic passes evaluate the same feature map
        # as ``_position_features`` in one gather, and push tangents/cotangents
        # through it. Non-persistent: the state_dict contract is unchanged.
        for tag, excluded in (
            ("mass", mass_excluded),
            ("potential", potential_excluded),
        ):
            source, sin_mask, cos_mask = self._feature_plan(excluded)
            self.register_buffer(f"_{tag}_feat_src", source, persistent=False)
            self.register_buffer(f"_{tag}_feat_sin", sin_mask, persistent=False)
            self.register_buffer(f"_{tag}_feat_cos", cos_mask, persistent=False)
        # Keep the legacy attribute/None-buffer contract used by recovery and
        # old checkpoints.  None buffers do not appear in a state_dict.
        if mass_excluded:
            keep = [i for i in range(npos) if i not in mass_excluded]
            self.register_buffer("_mass_in_idx", th.tensor(keep, dtype=th.long))
        else:
            self.register_buffer("_mass_in_idx", None)

        mass_in = self._position_feature_dim(mass_excluded)
        potential_in = self._position_feature_dim(potential_excluded)
        self.mass_net = mlp(nv * (nv + 1) // 2, inp=mass_in)  # Cholesky entries of M(q)
        self.potential_net = mlp(1, inp=potential_in)  # invariant base potential V(q)
        # G_a maps the action (action_dim) to a generalized force. Dense: one force
        # per config DOF (nv). Sparse: one force per (actuator -> DOF) target in
        # act_to_cfg, scattered additively onto the config axis in the drift.
        n_force = nv if layout.act_to_cfg is None else len(layout.act_to_cfg)
        self.G_a = nn.Linear(self.action_dim, n_force, bias=False)
        self._log_d = nn.Parameter(
            th.full((nv,), float(layout.damping_log_init))
        )  # base diagonal damping

        ti = th.tril_indices(nv, nv)
        self.register_buffer("_tri_flat", ti[0] * nv + ti[1])
        self.register_buffer("_eye_nv", th.eye(nv))
        self.register_buffer("_pos_to_cfg", th.tensor(layout.obs_pos_to_cfg, dtype=th.long))
        if layout.act_to_cfg is not None:
            self.register_buffer("_act_to_cfg", th.tensor(layout.act_to_cfg, dtype=th.long))

        if layout.joint_limits:
            limit_pos = [int(spec[0]) for spec in layout.joint_limits]
            limit_cfg = [int(layout.obs_pos_to_cfg[i]) for i in limit_pos]
            self.register_buffer("_limit_pos_idx", th.tensor(limit_pos, dtype=th.long))
            self.register_buffer("_limit_cfg_idx", th.tensor(limit_cfg, dtype=th.long))
            self.register_buffer(
                "_limit_lower",
                th.tensor([float(spec[1]) for spec in layout.joint_limits]),
            )
            self.register_buffer(
                "_limit_upper",
                th.tensor([float(spec[2]) for spec in layout.joint_limits]),
            )
            # Positive stiffness/damping through softplus.  The damping is
            # localized at the rail and is distinct from the near-zero base D.
            self._limit_raw = nn.Parameter(
                th.stack(
                    [
                        th.full(
                            (len(limit_pos),),
                            _inverse_softplus(layout.joint_limit_stiffness_init),
                        ),
                        th.full(
                            (len(limit_pos),),
                            _inverse_softplus(layout.joint_limit_damping_init),
                        ),
                    ]
                )
            )
        else:
            self.register_buffer("_limit_pos_idx", None)
            self.register_buffer("_limit_cfg_idx", None)
            self.register_buffer("_limit_lower", None)
            self.register_buffer("_limit_upper", None)

        if self.contact_force > 0:
            # Explicit contact-force port: K learned point contacts against the
            # ground. Each has a gap function g_i(q) and a horizontal foot offset
            # h_i(q); their gradients form the contact Jacobian. New models use
            # the coupled action-aware constraint solve below. ``compliant``
            # retains the historical Hunt--Crossley/tanh force law.
            assert layout.contact_tangent_cfg is not None, (
                "contact_force > 0 requires dof_layout.contact_tangent_cfg (the "
                "config DOF of the horizontal translation friction acts along)."
            )
            self.gap_net = mlp(self.contact_force)      # g_i(q): signed gap heights
            self.tangent_net = mlp(self.contact_force)  # h_i(q): horizontal offsets
            with th.no_grad():
                # Both formulations start quiet but reachable. The constraint
                # gate initializes just inside its compact support; the legacy
                # law uses its historical 2.5-softplus-width initialization.
                if self.contact_solver == "constraint":
                    # Start just inside the smooth candidate envelope: the
                    # contact is nearly silent but the gate derivative is live.
                    self.gap_net[-1].weight.mul_(0.01)
                    self.gap_net[-1].bias.fill_(0.98 * self._contact_gate_off)
                else:
                    self.gap_net[-1].weight.mul_(0.1)
                    self.gap_net[-1].bias.fill_(2.5 * self._contact_gap_width)
            # The persistent marker makes the force-law semantics explicit.
            # Version 0 is the historical compliant penalty law. Version 1 is
            # the action-conditioned constraint solve below. It is deliberately
            # a buffer rather than a Python-only flag so a sidecar roundtrip
            # restores the formulation that produced it.
            solver_version = 1 if self.contact_solver == "constraint" else 0
            self.register_buffer(
                "_contact_solver_version",
                th.tensor(solver_version, dtype=th.int64),
            )
            if self.contact_solver == "compliant":
                # Historical semantics and initialization, kept bit-for-bit:
                # positive stiffness k, compression damping c, and friction mu
                # through softplus; raw 0.5413 => softplus ~= 1.
                raw = th.full((3, self.contact_force), 0.5413)
            else:
                # Constraint semantics are bounded: restitution
                # e=0.5*sigmoid(raw0), stabilization beta=0.5*sigmoid(raw1),
                # and friction mu=2*sigmoid(raw2). Initialize at approximately
                # (0.05, 0.2, 0.8), respectively.
                raw = th.stack(
                    [
                        th.full((self.contact_force,), float(np.log(0.1 / 0.9))),
                        th.full((self.contact_force,), float(np.log(0.4 / 0.6))),
                        th.full((self.contact_force,), float(np.log(0.4 / 0.6))),
                    ]
                )
            self._contact_raw = nn.Parameter(raw)
            onehot = th.zeros(nv)
            onehot[layout.contact_tangent_cfg] = 1.0
            self.register_buffer("_tangent_onehot", onehot)

    # ------------------------ structure helpers (phast) ------------------------

    def _J(self) -> th.Tensor:
        return self._J_raw - self._J_raw.t()

    def _R(self) -> th.Tensor:
        d0 = th.nn.functional.softplus(self._d0)
        eye = th.eye(self.obs_dim, device=self._L.device)
        return d0 * eye + self._L @ self._L.t()

    def _grad_H(self, x: th.Tensor) -> th.Tensor:
        # enable_grad so grad H can be taken even when the caller is under
        # th.no_grad() (e.g. evaluation, or the critic's target computation).
        with th.enable_grad():
            xin = x.clone().requires_grad_(True)
            H = self.energy(xin).sum()
            (gH,) = th.autograd.grad(H, xin, create_graph=True)
        return gH

    # ---- structured helpers (per-sample; vmap'd over the batch) ----

    def _position_feature_dim(self, excluded: Tuple[int, ...]) -> int:
        """Width of the invariant/periodic feature map used by M or base V."""
        keep = [i for i in range(self.layout.npos) if i not in excluded]
        width = sum(2 if i in self._periodic_pos else 1 for i in keep)
        if width == 0:
            raise ValueError("a structured mechanics network needs at least one position feature")
        return width

    def _position_features(
        self, pos: th.Tensor, excluded: Tuple[int, ...]
    ) -> th.Tensor:
        """Map one position vector to invariant, periodic mechanics features.

        Raw generalized coordinates remain the model state, so ``qdot = qd`` is
        unchanged.  Only the learned mechanical objects see sin/cos features.
        The fast path preserves the exact historical cheetah computation.
        """
        if not excluded and not self._periodic_pos:
            return pos
        features = []
        for i in range(self.layout.npos):
            if i in excluded:
                continue
            if i in self._periodic_pos:
                features.extend((th.sin(pos[i]), th.cos(pos[i])))
            else:
                features.append(pos[i])
        return th.stack(features)

    def _feature_plan(self, excluded: Tuple[int, ...]):
        """Gather indices and sin/cos masks describing ``_position_features``.

        Column order matches ``_position_features`` exactly.  ``(None, None,
        None)`` means the map is the identity, so the analytic passes can use
        ``pos`` and its tangents untouched; a plain gather index with no masks
        means coordinates are only dropped, never transformed.
        """
        npos = self.layout.npos
        source: list[int] = []
        kinds: list[int] = []       # 0 = raw, 1 = sin, 2 = cos
        for i in range(npos):
            if i in excluded:
                continue
            if i in self._periodic_pos:
                source.extend((i, i))
                kinds.extend((1, 2))
            else:
                source.append(i)
                kinds.append(0)
        if not source:
            raise ValueError(
                "a structured mechanics network needs at least one position feature"
            )
        periodic = any(kinds)
        if not periodic and source == list(range(npos)):
            return None, None, None
        source_tensor = th.tensor(source, dtype=th.long)
        if not periodic:
            return source_tensor, None, None
        return (
            source_tensor,
            th.tensor([k == 1 for k in kinds], dtype=th.bool),
            th.tensor([k == 2 for k in kinds], dtype=th.bool),
        )

    def _batched_features(self, pos: th.Tensor, tag: str):
        """Batched ``_position_features`` plus the per-column chain-rule factor.

        Returns ``(features, source, factor)``.  ``factor[:, j] = d features[:, j]
        / d pos[:, source[j]]``; ``None`` for either means "identity", which the
        callers use to skip work on the cheetah fast path.
        """
        source = getattr(self, f"_{tag}_feat_src")
        if source is None:
            return pos, None, None
        gathered = pos.index_select(1, source)
        sin_mask = getattr(self, f"_{tag}_feat_sin")
        if sin_mask is None:
            return gathered, source, None
        cos_mask = getattr(self, f"_{tag}_feat_cos")
        sin_value, cos_value = th.sin(gathered), th.cos(gathered)
        features = th.where(sin_mask, sin_value, th.where(cos_mask, cos_value, gathered))
        factor = th.where(
            sin_mask, cos_value, th.where(cos_mask, -sin_value, th.ones_like(gathered))
        )
        return features, source, factor

    def _feature_tangent(
        self, velocity: th.Tensor, source: Optional[th.Tensor], factor: Optional[th.Tensor]
    ) -> th.Tensor:
        """Push a position-space tangent through the feature map."""
        if source is None:
            return velocity
        tangent = velocity.index_select(1, source)
        return tangent if factor is None else tangent * factor

    def _feature_cotangent(
        self,
        grad_features: th.Tensor,
        source: Optional[th.Tensor],
        factor: Optional[th.Tensor],
        like: th.Tensor,
    ) -> th.Tensor:
        """Pull a feature-space gradient back to observed-position space."""
        if source is None:
            return grad_features
        if factor is not None:
            grad_features = grad_features * factor
        zeros = like.new_zeros(like.shape[0], self.layout.npos)
        # index_add, not index_copy: a periodic coordinate contributes through
        # both its sin and its cos column.
        return zeros.index_add(1, source, grad_features)

    def _cholesky_from_entries(self, entries: th.Tensor):
        """Lower Cholesky factor with a softplus-positive diagonal, batched.

        Also returns the raw diagonal's softplus derivative, which both the
        tangent and the cotangent pass need.
        """
        factor = self._pack_lower(entries)
        raw_diag = factor.diagonal(dim1=-2, dim2=-1)
        factor = (
            factor
            - th.diag_embed(raw_diag)
            + th.diag_embed(th.nn.functional.softplus(raw_diag) + 1e-3)
        )
        return factor, th.sigmoid(raw_diag)

    def _pack_lower(self, entries: th.Tensor) -> th.Tensor:
        """Pack lower-triangular entries into a (batch, nv, nv) matrix."""
        nv = self.layout.nv
        return entries.new_zeros(entries.shape[0], nv * nv).index_copy(
            1, self._tri_flat, entries
        ).reshape(-1, nv, nv)

    def _mass_batch(self, pos: th.Tensor) -> th.Tensor:
        """Batched M(q), identical to ``vmap(self._mass)`` without functorch."""
        features, _, _ = self._batched_features(pos, "mass")
        entries = self.mass_net(features)
        factor, _ = self._cholesky_from_entries(entries)
        return th.bmm(factor, factor.transpose(1, 2)) + 1e-4 * self._eye_nv

    def _mass_terms_analytic(self, pos: th.Tensor, qd: th.Tensor):
        """``(M, Mdot @ qd, grad_pos(1/2 qd^T M qd))`` in one trace.

        Both consumers of dM/dq in the drift contract it twice with ``qd``, so
        neither needs the full (nv, nv, npos) Jacobian that ``jacfwd`` builds:
        ``Mdot`` is the directional derivative along ``qd`` (one forward tangent)
        and the Coriolis term is the gradient of a scalar (one backward pass).
        """
        nv = self.layout.nv
        features, source, factor = self._batched_features(pos, "mass")
        velocity = qd.index_select(1, self._pos_to_cfg)   # dpos/dt in observed order
        entries, layers = _mlp_trace(self.mass_net, features)
        entries_dot = _mlp_jvp(
            layers, self._feature_tangent(velocity, source, factor)
        )
        cholesky, softplus_prime = self._cholesky_from_entries(entries)
        transpose = cholesky.transpose(1, 2)
        mass = th.bmm(cholesky, transpose) + 1e-4 * self._eye_nv

        # Tangent of the factor: only the diagonal picks up the softplus factor.
        cholesky_dot = _scale_diagonal(
            self._pack_lower(entries_dot), softplus_prime
        )
        mass_dot = th.bmm(cholesky_dot, transpose) + th.bmm(
            cholesky, cholesky_dot.transpose(1, 2)
        )
        mass_dot_qd = th.bmm(mass_dot, qd.unsqueeze(-1)).squeeze(-1)

        # d/dL of s = 1/2 qd^T (L L^T) qd is qd (L^T qd)^T; the eps*I term in M
        # is constant in q and drops out.
        projected = th.bmm(transpose, qd.unsqueeze(-1)).squeeze(-1)
        grad_factor = _scale_diagonal(
            qd.unsqueeze(-1) * projected.unsqueeze(-2), softplus_prime
        )
        grad_entries = grad_factor.reshape(-1, nv * nv).index_select(1, self._tri_flat)
        coriolis = self._feature_cotangent(
            _mlp_vjp(layers, grad_entries), source, factor, pos
        )
        return mass, mass_dot_qd, coriolis

    def _potential_grad_analytic(self, pos: th.Tensor) -> th.Tensor:
        """grad_pos V(q) including the rail spring, without functorch."""
        features, source, factor = self._batched_features(pos, "potential")
        value, layers = _mlp_trace(self.potential_net, features)
        gradient = self._feature_cotangent(
            _mlp_vjp(layers, th.ones_like(value)), source, factor, pos
        )
        if self._limit_pos_idx is None:
            return gradient
        # d/dq of 1/2 k (w*softplus(pen/w))^2 on either rail.
        q = pos.index_select(1, self._limit_pos_idx)
        width = float(self.layout.joint_limit_width)
        lower_arg = (self._limit_lower - q) / width
        upper_arg = (q - self._limit_upper) / width
        lower_pen = th.nn.functional.softplus(lower_arg) * width
        upper_pen = th.nn.functional.softplus(upper_arg) * width
        stiffness = th.nn.functional.softplus(self._limit_raw[0])
        rail = stiffness * (
            upper_pen * th.sigmoid(upper_arg) - lower_pen * th.sigmoid(lower_arg)
        )
        return gradient.index_add(1, self._limit_pos_idx, rail)

    def _mechanics_terms(self, pos: th.Tensor, qd: th.Tensor):
        """``(M, dH/dq, Mdot @ qd)`` shared by the drift and the free-motion path."""
        nv, B = self.layout.nv, pos.shape[0]
        if self.analytic_derivatives:
            mass, mass_dot_qd, coriolis = self._mass_terms_analytic(pos, qd)
            grad_potential = self._potential_grad_analytic(pos)
            # Both terms live on observed positions, so one scatter carries both.
            dHdq = pos.new_zeros(B, nv).index_copy(
                1, self._pos_to_cfg, grad_potential - coriolis
            )
            return mass, dHdq, mass_dot_qd
        # functorch reference path, retained for the equivalence tests.
        mass = vmap(self._mass)(pos)
        mass_jac = vmap(jacfwd(self._mass))(pos)
        grad_potential = vmap(grad(self._potential))(pos)
        scattered = pos.new_zeros(B, nv, nv, nv).index_copy(3, self._pos_to_cfg, mass_jac)
        dHdq = pos.new_zeros(B, nv).index_copy(
            1, self._pos_to_cfg, grad_potential
        ) - 0.5 * th.einsum("nabk,na,nb->nk", scattered, qd, qd)
        mass_dot_qd = th.einsum("nabk,nk,nb->na", scattered, qd, qd)
        return mass, dHdq, mass_dot_qd

    def _mass(self, pos: th.Tensor) -> th.Tensor:
        """SPD mass matrix M(q) = L L^T + eps I from a Cholesky factor with a
        softplus-positive diagonal. Single-sample (pos: (npos,)) for vmap.
        Translation-invariant coordinates are dropped from the input when the
        contact port or the selected layout enforces that invariant."""
        nv = self.layout.nv
        l = self.mass_net(self._position_features(pos, self._mass_excluded_pos))
        L = th.zeros(nv * nv, dtype=pos.dtype, device=pos.device).scatter(
            0, self._tri_flat, l
        ).reshape(nv, nv)
        d = th.diagonal(L)
        L = L - th.diag_embed(d) + th.diag_embed(th.nn.functional.softplus(d) + 1e-3)
        return L @ L.t() + 1e-4 * self._eye_nv

    def _base_potential(self, pos: th.Tensor) -> th.Tensor:
        """Learned smooth potential, before explicit joint-limit storage."""
        return self.potential_net(
            self._position_features(pos, self._potential_excluded_pos)
        ).squeeze(-1)

    def _joint_limit_energy(self, pos: th.Tensor) -> th.Tensor:
        """Smooth unilateral spring energy for one position vector.

        ``w*softplus(penetration/w)`` is a differentiable positive-part.  Its
        squared energy yields an inward conservative force on either rail and
        exponentially vanishing force in the interior.
        """
        if self._limit_pos_idx is None:
            return pos.new_zeros(())
        q = pos.index_select(0, self._limit_pos_idx)
        w = float(self.layout.joint_limit_width)
        lower_pen = th.nn.functional.softplus((self._limit_lower - q) / w) * w
        upper_pen = th.nn.functional.softplus((q - self._limit_upper) / w) * w
        k = th.nn.functional.softplus(self._limit_raw[0])
        # A tensor-valued half keeps forward-mode Jacobians in the model dtype
        # (a Python 0.5 currently promotes the JVP tangent to float64 in PyTorch).
        return k.new_tensor(0.5) * (
            k * (lower_pen.square() + upper_pen.square())
        ).sum()

    def _potential(self, pos: th.Tensor) -> th.Tensor:
        # The explicit rail spring is part of stored mechanical energy; keeping
        # it inside V makes its force enter canonically as -grad V.
        return self._base_potential(pos) + self._joint_limit_energy(pos)

    def _damping(self, pos: th.Tensor) -> th.Tensor:
        """Constant diagonal PSD damping D = diag(softplus(log_d)) — passive joint
        damping. Contact dissipation belongs to the explicit port. (B, nv, nv)."""
        B = pos.shape[0]
        return th.diag_embed(th.nn.functional.softplus(self._log_d)).unsqueeze(0).expand(B, -1, -1)

    def _joint_limit_damping_force(
        self, pos: th.Tensor, qd: th.Tensor
    ) -> th.Tensor:
        """Localized smooth rail damping as a generalized force, shape (B,nv).

        The activation approaches one outside either rail and vanishes
        exponentially in the interior.  The force is always opposite velocity,
        so ``qd.T @ F_limit <= 0`` for every state, including during separation.
        """
        if self._limit_pos_idx is None:
            return th.zeros_like(qd)
        q = pos.index_select(1, self._limit_pos_idx)
        v = qd.index_select(1, self._limit_cfg_idx)
        w = float(self.layout.joint_limit_width)
        activation = th.sigmoid((self._limit_lower - q) / w) + th.sigmoid(
            (q - self._limit_upper) / w
        )
        c = th.nn.functional.softplus(self._limit_raw[1])
        force_limited = -c * activation * v
        return th.zeros_like(qd).index_add(1, self._limit_cfg_idx, force_limited)

    def _gaps(self, pos: th.Tensor) -> th.Tensor:
        return self.gap_net(pos)

    def _tangents(self, pos: th.Tensor) -> th.Tensor:
        return self.tangent_net(pos)

    # Gap-force smoothing width (the learned g absorbs any rescaling of it) and
    # the velocity scale of the regularized Coulomb friction.
    _contact_gap_width: float = 0.02
    _contact_stick_vel: float = 0.1
    _contact_gate_on: float = 0.0
    _contact_gate_off: float = 0.06
    _contact_max_correction_vel: float = 5.0

    def _contact_geometry(self, pos: th.Tensor, qd: th.Tensor):
        """Learned point-contact geometry shared by both force formulations."""
        B, nv, K = pos.shape[0], self.layout.nv, self.contact_force
        if self.analytic_derivatives:
            # The gap/tangent nets read raw positions, and K < npos here, so a
            # single reverse pass seeded with the identity is cheaper than
            # jacfwd's one forward tangent per coordinate.
            g, gap_layers = _mlp_trace(self.gap_net, pos)
            dg = _mlp_jacobian(gap_layers, K, pos)
            _, tangent_layers = _mlp_trace(self.tangent_net, pos)
            dh = _mlp_jacobian(tangent_layers, K, pos)
        else:
            g = self.gap_net(pos)
            dg = vmap(jacfwd(self._gaps))(pos)
            dh = vmap(jacfwd(self._tangents))(pos)
        zeros = pos.new_zeros(B, K, nv)
        J_n = zeros.index_copy(2, self._pos_to_cfg, dg)
        J_t = zeros.index_copy(2, self._pos_to_cfg, dh) + self._tangent_onehot
        gdot = th.einsum("nkv,nv->nk", J_n, qd)
        v_t = th.einsum("nkv,nv->nk", J_t, qd)
        return g, gdot, v_t, J_n, J_t

    def _contact_gate(self, g: th.Tensor) -> th.Tensor:
        """Compact, C2 contact-candidate gate (one scalar per point).

        Contacts at or below ``gate_on`` are fully enabled and contacts at or
        above ``gate_off`` are exactly disabled. The quintic smoothstep between
        them prevents a far-away point from generating friction while retaining
        a useful geometry gradient near first contact.
        """
        width = self._contact_gate_off - self._contact_gate_on
        u = ((self._contact_gate_off - g) / width).clamp(0.0, 1.0)
        # Quintic smoothstep: zero first and second derivatives at both ends.
        # At the constraint model's u=0.02 initialization this is ~7.8e-5,
        # quiet enough not to perturb the fresh model while still trainable.
        return u.pow(3) * (10.0 - 15.0 * u + 6.0 * u.square())

    @staticmethod
    def _project_contact_cones(
        impulse: th.Tensor, mu: th.Tensor
    ) -> th.Tensor:
        """Euclidean projection onto planar Coulomb cones.

        ``impulse`` is ordered ``[normal, tangent]`` per contact and ``mu`` has
        one coefficient per contact. The closed-form projection is piecewise
        differentiable and uses only PyTorch operations, so fixed-iteration
        optimization remains differentiable with respect to mechanics, geometry,
        action, and friction.
        """
        pair = impulse.reshape(*impulse.shape[:-1], -1, 2)
        normal, tangent = pair[..., 0], pair[..., 1]
        abs_tangent = tangent.abs()
        mu = mu.to(dtype=impulse.dtype, device=impulse.device)
        while mu.ndim < normal.ndim:
            mu = mu.unsqueeze(0)
        inside = (normal >= 0.0) & (abs_tangent <= mu * normal)
        polar = normal + mu * abs_tangent <= 0.0
        boundary_normal = (normal + mu * abs_tangent) / (1.0 + mu.square())
        boundary_tangent = mu * boundary_normal * tangent.sign()
        projected_normal = th.where(
            inside,
            normal,
            th.where(polar, th.zeros_like(normal), boundary_normal),
        )
        projected_tangent = th.where(
            inside,
            tangent,
            th.where(polar, th.zeros_like(tangent), boundary_tangent),
        )
        return th.stack((projected_normal, projected_tangent), dim=-1).reshape_as(
            impulse
        )

    def _constraint_contact_solve(
        self,
        pos: th.Tensor,
        qd: th.Tensor,
        M: th.Tensor,
        qdd_free: th.Tensor,
    ) -> dict[str, th.Tensor]:
        """Solve a regularized, cone-constrained contact impulse problem.

        The action has already entered ``qdd_free`` through ``G_a a``. Over the
        response horizon ``contact_dt`` this gives the free velocity. A joint
        solve through the learned Delassus matrix then chooses normal/tangential
        impulses for every point simultaneously. Static friction is represented:
        even at zero current slip, an action-induced free tangential velocity is
        cancelled when the required impulse lies inside the Coulomb cone.
        """
        B, K, nv = pos.shape[0], self.contact_force, self.layout.nv
        dt = qd.new_tensor(self.contact_dt)
        g, gdot, v_t, J_n, J_t = self._contact_geometry(pos, qd)
        gate = self._contact_gate(g)
        e = 0.5 * th.sigmoid(self._contact_raw[0])
        beta = 0.5 * th.sigmoid(self._contact_raw[1])
        mu = 2.0 * th.sigmoid(self._contact_raw[2])

        qd_free = qd + dt * qdd_free
        J_pair = th.stack((J_n, J_t), dim=2)             # (B,K,2,nv)
        J = J_pair.reshape(B, 2 * K, nv)
        contact_v_free = th.einsum("ncv,nv->nc", J, qd_free).reshape(B, K, 2)

        # The compact gate is the differentiable contact-candidate relaxation;
        # do not add positive clearance to this bias or the quiet initialization
        # would also be gradient-dead. Penetration recovery is Baumgarte-
        # stabilized and velocity-capped so a bad learned gap cannot request an
        # unbounded kick.
        penetration = (th.relu(-g) / dt).clamp_max(
            self._contact_max_correction_vel
        )
        stabilization_bias = -beta * penetration
        restitution_bias = e * th.minimum(gdot, th.zeros_like(gdot))
        normal_bias = (
            contact_v_free[..., 0]
            + stabilization_bias
            + restitution_bias
        )
        tangent_bias = contact_v_free[..., 1]
        physical_bias = th.stack((normal_bias, tangent_bias), dim=-1)

        # Let z be a latent cone impulse and p=gate*z the physical impulse. This
        # turns the smooth candidate gate into an impedance envelope without
        # changing the friction ratio. J_solver maps z directly to generalized
        # impulse, so both the quadratic kinetic term and the linear free-velocity
        # term use the same virtual-work map.
        gate_pair = gate.unsqueeze(-1).expand(-1, -1, 2).reshape(B, 2 * K)
        J_solver = J * gate_pair.unsqueeze(-1)
        M_inv_Jt_full = th.linalg.solve(M, J.transpose(1, 2))
        M_inv_Jt = M_inv_Jt_full * gate_pair.unsqueeze(1)
        W = th.bmm(J_solver, M_inv_Jt)
        W = 0.5 * (W + W.transpose(1, 2))

        # Scale regularization with the ungated Delassus matrix, retaining the
        # global mechanical gauge and keeping duplicate learned contact points
        # nonsingular. A positive diagonal also makes the QP solution unique.
        W_full = th.bmm(J, M_inv_Jt_full)
        scale = (
            W_full.diagonal(dim1=-2, dim2=-1)
            .mean(-1)
            .clamp_min(1e-6)
            .detach()
        )
        eye = th.eye(2 * K, dtype=qd.dtype, device=qd.device).unsqueeze(0)
        H = W + self.contact_regularization * scale[:, None, None] * eye
        bias = (physical_bias.reshape(B, 2 * K) * gate_pair)

        # Scaled ADMM: one batched dense factorization, then a fixed number of
        # tiny triangular solves and exact cone projections. Fixed iteration
        # count avoids data-dependent control flow/synchronization in BPTT.
        rho = H.diagonal(dim1=-2, dim2=-1).mean(-1).detach().clamp_min(1e-6)
        factor = th.linalg.cholesky(H + rho[:, None, None] * eye)
        primal = th.zeros_like(bias)
        auxiliary = th.zeros_like(bias)
        dual = th.zeros_like(bias)
        for _ in range(self.contact_iterations):
            rhs = -bias + rho[:, None] * (auxiliary - dual)
            primal = th.cholesky_solve(rhs.unsqueeze(-1), factor).squeeze(-1)
            relaxed = 1.5 * primal - 0.5 * auxiliary
            next_auxiliary = self._project_contact_cones(relaxed + dual, mu)
            dual = dual + relaxed - next_auxiliary
            auxiliary = next_auxiliary

        latent_impulse = auxiliary
        physical_impulse_flat = latent_impulse * gate_pair
        physical_impulse = physical_impulse_flat.reshape(B, K, 2)
        generalized_impulse = th.einsum(
            "ncv,nc->nv", J, physical_impulse_flat
        )
        delta_qd = th.bmm(
            M_inv_Jt_full, physical_impulse_flat.unsqueeze(-1)
        ).squeeze(-1)
        qd_post = qd_free + delta_qd
        contact_v_post = th.einsum("ncv,nv->nc", J, qd_post).reshape(B, K, 2)

        # A normalized projected-gradient residual is meaningful across the
        # learned mass gauge. Cone violation is measured on physical impulses.
        grad_qp = th.bmm(H, latent_impulse.unsqueeze(-1)).squeeze(-1) + bias
        lipschitz = H.abs().sum(-1).amax(-1).detach().clamp_min(1e-6)
        prox = self._project_contact_cones(
            latent_impulse - grad_qp / lipschitz[:, None], mu
        )
        solver_residual = (latent_impulse - prox).norm(dim=-1) / (
            1.0 + latent_impulse.norm(dim=-1)
        )
        normal_impulse = physical_impulse[..., 0]
        tangent_impulse = physical_impulse[..., 1]
        cone_violation = th.relu(tangent_impulse.abs() - mu * normal_impulse).amax(
            dim=-1
        )
        discrete_work = (
            qd_free * generalized_impulse
        ).sum(-1) + 0.5 * (generalized_impulse * delta_qd).sum(-1)
        stabilization_work = -(stabilization_bias * normal_impulse).sum(-1)

        return {
            "gap": g,
            "gap_rate": gdot,
            "tangent_velocity": v_t,
            "gate": gate,
            "J_n": J_n,
            "J_t": J_t,
            "mu": mu,
            "e": e,
            "beta": beta,
            "normal_impulse": normal_impulse,
            "tangent_impulse": tangent_impulse,
            "normal_force": normal_impulse / dt,
            "tangent_force": tangent_impulse / dt,
            "generalized_impulse": generalized_impulse,
            "generalized_force": generalized_impulse / dt,
            "contact_acceleration": delta_qd / dt,
            "free_acceleration": qdd_free,
            "v_free": qd_free,
            "v_post": qd_post,
            "contact_velocity_free": contact_v_free,
            "contact_velocity_post": contact_v_post,
            "free_contact_velocity": contact_v_free,
            "post_contact_velocity": contact_v_post,
            "cone_violation": cone_violation,
            "solver_residual": solver_residual,
            "regularization": qd.new_tensor(self.contact_regularization),
            "discrete_work": discrete_work,
            "stabilization_work": stabilization_work,
        }

    def _contact_parts(self, pos: th.Tensor, qd: th.Tensor):
        """Contact-port quantities for a batch. For each learned contact point i:

          g_i(q)      signed gap; J_n,i = dg_i/dq is the normal generalized
                      direction (a vertical point force has zero generalized
                      component along horizontal translation, so the cyclic
                      slot correctly stays zero),
          h_i(q)      horizontal foot offset relative to the root; the foot's
                      horizontal position is x_root + h_i(q), so the tangential
                      direction is J_t,i = e_tangent + dh_i/dq,
          lam_i       = phi(g_i) (k_i + c_i relu(-gdot_i)) >= 0: unilateral
                      Hunt-Crossley normal magnitude with smooth penetration
                      phi(g) = w softplus(-g/w); the spring part is a gradient
                      field (conservative) and the c-part only dissipates,
          f_t,i       = -mu_i lam_i tanh(v_t,i / v_stick): regularized Coulomb
                      friction, power f_t v_t <= 0 by construction.

        Returns (g, gdot, v_t, lam, f_t, J_n, J_t) with shapes (B,K)x5, (B,K,nv)x2."""
        g, gdot, v_t, J_n, J_t = self._contact_geometry(pos, qd)
        k, c, mu = th.nn.functional.softplus(self._contact_raw)  # each (K,)
        w = self._contact_gap_width
        phi = th.nn.functional.softplus(-g / w) * w       # smooth max(0, -g)
        lam = phi * (k + c * th.relu(-gdot))              # (B, K) >= 0
        f_t = -mu * lam * th.tanh(v_t / self._contact_stick_vel)
        return g, gdot, v_t, lam, f_t, J_n, J_t

    def _contact_force_gen(self, pos: th.Tensor, qd: th.Tensor) -> th.Tensor:
        """Generalized contact force sum_i (J_n,i lam_i + J_t,i f_t,i), (B, nv)."""
        _, _, _, lam, f_t, J_n, J_t = self._contact_parts(pos, qd)
        return th.einsum("nkv,nk->nv", J_n, lam) + th.einsum("nkv,nk->nv", J_t, f_t)

    def _structured_free_acceleration(
        self, x: th.Tensor, a: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Return ``(pos, qd, M, qdd_free)`` before ground contact.

        This mirrors the contact-free portion of :meth:`_structured_drift` and
        is intentionally public only inside the model: recovery diagnostics need
        the same action-conditioned free motion that feeds the constraint solve.
        Joint-limit forces remain part of the free mechanism; only the learned
        ground-contact port is omitted.
        """
        layout = self.layout
        nv, B = layout.nv, x.shape[0]
        pos = x[:, layout.pos_slice[0]:layout.pos_slice[1]]
        qd = x[:, layout.vel_slice[0]:layout.vel_slice[1]]
        M, dHdq, Mdot_qd = self._mechanics_terms(pos, qd)
        D_qd = th.bmm(self._damping(pos), qd.unsqueeze(-1)).squeeze(-1)
        if layout.act_to_cfg is None:
            Ga = self.G_a(a)
        else:
            Ga = th.zeros(B, nv, dtype=x.dtype, device=x.device).index_add(
                1, self._act_to_cfg, self.G_a(a)
            )
        pdot = -dHdq - D_qd + Ga + self._joint_limit_damping_force(pos, qd)
        qdd_free = th.linalg.solve(
            M, (pdot - Mdot_qd).unsqueeze(-1)
        ).squeeze(-1)
        return pos, qd, M, qdd_free

    def contact_diagnostics(self, obs, action) -> dict[str, th.Tensor]:
        """Return action-conditioned contact quantities for recovery/auditing.

        Forces are average forces over ``contact_dt`` for the constraint model;
        impulses are their time integral. Returned tensors stay on the model
        device and retain gradients unless the caller enters ``torch.no_grad``.
        """
        if self.mode != "structured" or self.contact_force <= 0:
            raise ValueError(
                "contact_diagnostics requires a structured model with contact_force > 0"
            )
        x = th.as_tensor(obs, dtype=th.float32, device=self.device)
        a = th.as_tensor(action, dtype=th.float32, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if a.ndim == 1:
            a = a.unsqueeze(0)
        if x.ndim != 2 or x.shape[1] != self.obs_dim:
            raise ValueError(
                f"obs must have shape (batch, {self.obs_dim}), got {tuple(x.shape)}"
            )
        if a.ndim != 2 or a.shape != (x.shape[0], self.action_dim):
            raise ValueError(
                "action must have shape (batch, action_dim) with the same batch as obs"
            )
        pos, qd, M, qdd_free = self._structured_free_acceleration(x, a)
        if self.contact_solver == "constraint":
            return self._constraint_contact_solve(pos, qd, M, qdd_free)

        g, gdot, v_t, lam, f_t, J_n, J_t = self._contact_parts(pos, qd)
        _, _, mu = th.nn.functional.softplus(self._contact_raw)
        dt = qd.new_tensor(self.contact_dt)
        normal_impulse = lam * dt
        tangent_impulse = f_t * dt
        generalized_force = (
            th.einsum("nkv,nk->nv", J_n, lam)
            + th.einsum("nkv,nk->nv", J_t, f_t)
        )
        generalized_impulse = generalized_force * dt
        qd_free = qd + dt * qdd_free
        delta_qd = th.linalg.solve(
            M, generalized_impulse.unsqueeze(-1)
        ).squeeze(-1)
        qd_post = qd_free + delta_qd
        J = th.stack((J_n, J_t), dim=2).reshape(
            x.shape[0], 2 * self.contact_force, self.layout.nv
        )
        contact_v_free = th.einsum("ncv,nv->nc", J, qd_free).reshape(
            x.shape[0], self.contact_force, 2
        )
        contact_v_post = th.einsum("ncv,nv->nc", J, qd_post).reshape(
            x.shape[0], self.contact_force, 2
        )
        cone_violation = th.relu(tangent_impulse.abs() - mu * normal_impulse).amax(
            dim=-1
        )
        discrete_work = (
            qd_free * generalized_impulse
        ).sum(-1) + 0.5 * (generalized_impulse * delta_qd).sum(-1)
        zeros = th.zeros_like(mu)
        return {
            "gap": g,
            "gap_rate": gdot,
            "tangent_velocity": v_t,
            "gate": th.sigmoid(-g / self._contact_gap_width),
            "J_n": J_n,
            "J_t": J_t,
            "mu": mu,
            "e": zeros,
            "beta": zeros,
            "normal_impulse": normal_impulse,
            "tangent_impulse": tangent_impulse,
            "normal_force": lam,
            "tangent_force": f_t,
            "generalized_impulse": generalized_impulse,
            "generalized_force": generalized_force,
            "contact_acceleration": delta_qd / dt,
            "free_acceleration": qdd_free,
            "v_free": qd_free,
            "v_post": qd_post,
            "contact_velocity_free": contact_v_free,
            "contact_velocity_post": contact_v_post,
            "free_contact_velocity": contact_v_free,
            "post_contact_velocity": contact_v_post,
            "cone_violation": cone_violation,
            "solver_residual": th.zeros(x.shape[0], dtype=x.dtype, device=x.device),
            "regularization": x.new_tensor(self.contact_regularization),
            "discrete_work": discrete_work,
            "stabilization_work": th.zeros_like(discrete_work),
        }

    def _structured_drift(self, x: th.Tensor, a: th.Tensor) -> th.Tensor:
        """Port-Hamiltonian drift with the canonicalizer p = M(q) qd:
          dH/dq = grad V - 1/2 qd^T (dM/dq) qd,   qd = M^-1 p (= observed velocity)
          pdot  = -dH/dq - D qd + G_a a [+ F_contact(q, qd)]
          qddot = M^-1 (pdot - Mdot qd)           (obs-space acceleration)
        The Coriolis terms come from dM/dq (autodiff), not a learned head; the
        optional contact port adds either the action-aware constraint reaction
        or the historical compliant force through learned contact geometry.
        Returns the observation-space drift [d/dt positions ; qddot]."""
        layout = self.layout
        nv = layout.nv
        B = x.shape[0]
        pos = x[:, layout.pos_slice[0]:layout.pos_slice[1]]
        qd = x[:, layout.vel_slice[0]:layout.vel_slice[1]]

        # M, dH/dq and Mdot qd come from one traced pass per mechanics net: both
        # uses of dM/dq contract it twice with qd, so a single JVP along qd and a
        # single scalar VJP replace the (nv, nv, npos) Jacobian. The cyclic config
        # DOFs are absent from _pos_to_cfg, so their gradient slots stay zero.
        M, dHdq, Mdot_qd = self._mechanics_terms(pos, qd)
        D_qd = th.bmm(self._damping(pos), qd.unsqueeze(-1)).squeeze(-1)
        if layout.act_to_cfg is None:
            Ga = self.G_a(a)                            # (B, nv)
        else:
            # additive scatter: multiple actuators on one DOF sum their forces
            Ga = th.zeros(B, nv, dtype=x.dtype, device=x.device).index_add(
                1, self._act_to_cfg, self.G_a(a)
            )
        pdot = -dHdq - D_qd + Ga
        # The conservative rail spring is already in gV through _potential;
        # this separate port contributes only non-positive damping power.
        pdot = pdot + self._joint_limit_damping_force(pos, qd)
        free_rhs = pdot - Mdot_qd
        if self.contact_force > 0 and self.contact_solver == "compliant":
            # Historical state-only Hunt--Crossley/Coulomb port.
            free_rhs = free_rhs + self._contact_force_gen(pos, qd)
            qddot = th.linalg.solve(M, free_rhs.unsqueeze(-1)).squeeze(-1)
        else:
            qdd_free = th.linalg.solve(M, free_rhs.unsqueeze(-1)).squeeze(-1)
            if self.contact_force > 0:
                # The constraint port reads qdd_free, hence candidate action,
                # before solving the coupled stance reaction.
                contact = self._constraint_contact_solve(pos, qd, M, qdd_free)
                qddot = qdd_free + contact["contact_acceleration"]
            else:
                qddot = qdd_free
        b_pos = qd[:, self._pos_to_cfg]                 # d/dt of each observed position
        return th.cat([b_pos, qddot], dim=-1)           # (B, obs_dim)

    # ------------------------ public API ------------------------

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load dynamics weights while preserving contact-law compatibility.

        Contact sidecars written before the constraint solver have no formulation
        marker. Such a sidecar is unambiguously the historical compliant model:
        inject a version-zero marker so strict loading still succeeds, and switch
        this instance to that law before it is used. New sidecars carry the marker
        and therefore restore their own formulation even when the constructor's
        default changes.
        """
        incoming = state_dict
        if self.contact_force > 0 and hasattr(self, "_contact_solver_version"):
            marker_name = "_contact_solver_version"
            if marker_name not in state_dict:
                incoming = state_dict.copy()
                if hasattr(state_dict, "_metadata"):
                    incoming._metadata = state_dict._metadata
                incoming[marker_name] = self._contact_solver_version.new_zeros(())
                version = 0
            else:
                marker = th.as_tensor(state_dict[marker_name])
                if marker.numel() != 1:
                    raise RuntimeError(
                        "contact solver version marker must contain one scalar"
                    )
                version = int(marker.detach().cpu().item())
            if version not in (0, 1):
                raise RuntimeError(f"unsupported contact solver version {version}")
            self.contact_solver = "constraint" if version == 1 else "compliant"
        try:
            result = super().load_state_dict(incoming, strict=strict, assign=assign)
        except TypeError:  # PyTorch releases before the ``assign`` keyword.
            result = super().load_state_dict(incoming, strict=strict)
        if self.contact_force > 0 and hasattr(self, "_contact_solver_version"):
            self._contact_solver_version.fill_(
                1 if self.contact_solver == "constraint" else 0
            )
        return result

    def to(self, *args, **kwargs):
        """Move module state and keep the explicit dynamics device in sync.

        ``drift`` accepts NumPy inputs and therefore cannot infer a device from
        them.  Keeping this attribute synchronized is also required when the
        shared integrator gathers active sub-batches after CT-SAC moves a model
        from its construction device.
        """
        module = super().to(*args, **kwargs)
        state_tensor = next(module.parameters(), None)
        if state_tensor is None:
            state_tensor = next(module.buffers(), None)
        if state_tensor is not None:
            self.device = state_tensor.device
        else:
            requested = kwargs.get("device")
            if requested is None and args:
                candidate = args[0]
                if isinstance(candidate, (str, th.device)):
                    requested = candidate
                elif isinstance(candidate, th.Tensor):
                    requested = candidate.device
            if requested is not None:
                self.device = th.device(requested)
        return module

    def drift(self, obs, action) -> th.Tensor:
        """Return the drift b(obs, action) of shape (B, obs_dim) on ``self.device``."""
        if self.mode == "mujoco":
            b = self._drift_fn(obs, action)  # numpy (B, d)
            return th.as_tensor(np.asarray(b), dtype=th.float32, device=self.device)

        x = th.as_tensor(obs, dtype=th.float32, device=self.device)
        a = th.as_tensor(action, dtype=th.float32, device=self.device)

        if self.mode == "structured":
            return self._structured_drift(x, a)

        gH = self._grad_H(x)  # (B, d)
        JR = self._J() - self._R()  # (d, d)
        return gH @ JR.t() + self.G_a(a)

    def diffusion(self, obs) -> Optional[th.Tensor]:
        """Return sigma(obs) of shape (B, obs_dim, k), or None if sigma == 0."""
        if self.human_input_intensity <= 0.0:
            return None
        batch = obs.shape[0] if hasattr(obs, "shape") else len(obs)
        scale = float(self.human_input_intensity) ** 0.5
        eye = th.eye(self.obs_dim, device=self.device) * scale
        return eye.unsqueeze(0).expand(batch, -1, -1)

    # Fit gradients are clipped to this global norm. The one-step fit rarely
    # needs it, but the rollout fit backpropagates through a chained Euler roll
    # whose forward pass can transiently explode (the Coriolis term is
    # quadratic in velocity), and one unclipped step is enough to NaN the
    # parameters and, through the critic target, the whole agent.
    fit_max_grad_norm: float = 10.0

    @staticmethod
    def _duration_balance_weights(
        dt: th.Tensor, balance_dt: Optional[float]
    ) -> th.Tensor:
        """Endpoint weights that remove the quadratic duration leverage.

        A flow endpoint error is approximately ``dt * drift_error``, so its
        squared loss otherwise gives a 30 ms transition 225 times the leverage
        of a 2 ms transition.  ``balance_dt`` retains and fully integrates every
        interval but caps that leverage with
        ``w=(balance_dt/max(dt,balance_dt))^2``.  The caller normalizes by the
        sum of weights.
        """
        if balance_dt is None:
            return th.ones_like(dt)
        ref = float(balance_dt)
        if not np.isfinite(ref) or ref <= 0.0:
            raise ValueError("balance_dt must be finite and > 0")
        ref_t = dt.new_tensor(ref)
        return (ref_t / th.maximum(dt, ref_t)).square()

    def _mass_regularization(self, obs: th.Tensor) -> th.Tensor:
        """Optional numerical gauge and condition penalties for structured M.

        The mean log-determinant anchor chooses one representative of the
        otherwise free global mechanical scale.  The condition penalty is zero
        below ``mass_condition_limit``.  Both default to zero, preserving all
        existing fits and checkpoints.
        """
        if self.mode != "structured" or (
            self.mass_logdet_reg == 0.0 and self.mass_condition_reg == 0.0
        ):
            return obs.new_zeros(())
        pos = obs[:, self.layout.pos_slice[0]:self.layout.pos_slice[1]]
        M = self._mass_batch(pos) if self.analytic_derivatives else vmap(self._mass)(pos)
        penalty = obs.new_zeros(())
        if self.mass_logdet_reg > 0.0:
            logdet = th.linalg.slogdet(M).logabsdet
            penalty = penalty + self.mass_logdet_reg * logdet.mean().square()
        if self.mass_condition_reg > 0.0:
            eig = th.linalg.eigvalsh(M)
            cond = eig[:, -1] / eig[:, 0].clamp_min(1e-8)
            log_limit = cond.new_tensor(np.log(self.mass_condition_limit))
            excess = th.relu(th.log(cond) - log_limit)
            penalty = penalty + self.mass_condition_reg * excess.square().mean()
        return penalty

    def _base_damping_regularization(self) -> th.Tensor:
        """Optional relative-scale prior that keeps near-zero base damping small."""
        if self.mode != "structured" or self.base_damping_reg == 0.0:
            # Every learned structured/phast mode has at least one parameter;
            # use it only to construct a correctly placed scalar zero.
            return next(self.parameters()).new_zeros(())
        target = self._log_d.new_tensor(float(self.layout.damping_log_init))
        return self.base_damping_reg * (self._log_d - target).square().mean()

    def mass_diagnostics(self, obs) -> dict[str, float]:
        """Condition/gauge diagnostics on a batch of structured-model states."""
        if self.mode != "structured":
            raise ValueError("mass_diagnostics requires mode='structured'")
        x = th.as_tensor(obs, dtype=th.float32, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        pos = x[:, self.layout.pos_slice[0]:self.layout.pos_slice[1]]
        with th.no_grad():
            M = self._mass_batch(pos) if self.analytic_derivatives else vmap(self._mass)(pos)
            eig = th.linalg.eigvalsh(M)
            cond = eig[:, -1] / eig[:, 0].clamp_min(1e-12)
            logdet = th.linalg.slogdet(M).logabsdet
        return {
            "min_eig": float(eig[:, 0].min()),
            "max_eig": float(eig[:, -1].max()),
            "condition_mean": float(cond.mean()),
            "condition_max": float(cond.max()),
            "logdet_mean": float(logdet.mean()),
        }

    def _fit_optimizer_step(self, loss: th.Tensor, optimizer) -> float:
        """backward + clipped step, skipping the update (not the report) when
        the loss or the gradient norm is non-finite, so a pathological batch
        cannot write NaN into the parameters."""
        self.last_fit_accepted = False
        self.last_fit_grad_norm = float("nan")
        optimizer.zero_grad()
        if not th.isfinite(loss):
            return float(loss.detach())
        loss.backward()
        total_norm = th.nn.utils.clip_grad_norm_(
            [p for g in optimizer.param_groups for p in g["params"]],
            self.fit_max_grad_norm,
        )
        self.last_fit_grad_norm = float(total_norm.detach())
        if th.isfinite(total_norm):
            optimizer.step()
            params = [
                p for group in optimizer.param_groups for p in group["params"]
            ]
            self.last_fit_accepted = bool(
                th.stack([th.isfinite(p).all() for p in params]).all()
            )
        optimizer.zero_grad(set_to_none=True)
        return float(loss.detach())

    def fit_step(
        self, obs, action, next_obs, dt, optimizer, *,
        max_step: Optional[float] = None,
        balance_dt: Optional[float] = None,
    ) -> float:
        """One supervised flow-matching step.

        The learned drift is integrated across each transition's complete observed
        duration, using physics-sized internal steps when ``max_step`` is supplied,
        before its endpoint is compared with ``x'``.  This keeps irregular replay
        intervals while avoiding the old coarse-secant target ``x + b(x,a)*dt``.
        ``balance_dt`` optionally weights endpoint losses by
        ``(balance_dt/max(dt,balance_dt))**2`` without shortening any interval.
        """
        assert self.mode in ("phast", "structured"), (
            "fit_step only applies to learned modes ('phast', 'structured')."
        )
        x = th.as_tensor(obs, dtype=th.float32, device=self.device)
        a = th.as_tensor(action, dtype=th.float32, device=self.device)
        xp = th.as_tensor(next_obs, dtype=th.float32, device=self.device)
        dt_t = th.as_tensor(dt, dtype=th.float32, device=self.device).reshape(-1, 1)
        if dt_t.shape[0] == 1 and x.shape[0] != 1:
            dt_t = dt_t.expand(x.shape[0], 1)
        elif dt_t.shape[0] != x.shape[0]:
            raise ValueError("dt must be scalar or have one value per sample")
        # Healthy predictions stay well inside this generous data-scaled bound;
        # a runaway internal roll saturates before a later drift call sees inf.
        delta_limit = 5.0 * (xp - x).abs() + 1e-3 if max_step is not None else None
        pred = integrate_drift(
            self.drift,
            x,
            a,
            dt_t,
            max_step=max_step,
            delta_limit=delta_limit,
            clamp_final=False,
        )
        weight = self._duration_balance_weights(dt_t, balance_dt)
        loss = (weight * (pred - xp).square()).sum() / (
            weight.sum() * self.obs_dim + 1e-8
        )
        loss = (
            loss
            + self._mass_regularization(x)
            + self._base_damping_regularization()
        )
        return self._fit_optimizer_step(loss, optimizer)

    def fit_step_rollout(
        self, obs, actions, next_obs, dt, mask, optimizer, *,
        max_step: Optional[float] = None,
        balance_dt: Optional[float] = None,
    ) -> float:
        """Multi-transition flow-matching step.

        Roll the model across ``H`` replay transitions and regress every endpoint
        onto the observed sequence.  Each outer transition keeps its recorded
        duration and action; internally, :func:`integrate_drift` resolves that
        interval with steps no larger than ``max_step``:

          x_hat_0 = x_t,
          x_hat_{k+1} = Phi_hat(dt_{t+k}, x_hat_k, a_{t+k}),
          loss = sum_k m_k ||x_hat_{k+1} - x_{t+k+1}||^2 / (obs_dim * sum_k m_k).

        Gradients flow through the whole roll (backprop through time), so the
        model is trained where the generator/quadrature targets consume it: on
        its own predictions, not only on buffer states. ``mask`` (1 valid,
        0 invalid, cumulative) zeroes the steps of a window that crosses an
        episode end or the ring seam; ``horizon=1`` with a full mask is exactly
        ``fit_step`` when both use the same ``max_step`` and ``balance_dt``.

        The compounding steps evaluate the drift at model-predicted states, where
        the velocity-quadratic Coriolis term can overflow. Internal states and
        off-manifold outer endpoints are therefore bounded by a generous multiple
        of the window's observed increments. The first real-state endpoint remains
        unclamped in the loss (so it keeps a corrective gradient and horizon=1 is
        exactly ``fit_step``), although a bounded copy seeds a longer rollout.
        The optimizer step is also gradient-clipped and skipped on non-finite loss.

        Shapes: obs (B, O); actions (B, H, A); next_obs (B, H, O);
        dt, mask (B, H, 1)."""
        assert self.mode in ("phast", "structured"), (
            "fit_step_rollout only applies to learned modes ('phast', 'structured')."
        )
        x_hat = th.as_tensor(obs, dtype=th.float32, device=self.device)
        a = th.as_tensor(actions, dtype=th.float32, device=self.device)
        xp = th.as_tensor(next_obs, dtype=th.float32, device=self.device)
        batch = x_hat.shape[0]
        dt_t = th.as_tensor(dt, dtype=th.float32, device=self.device).reshape(
            batch, -1, 1
        )
        m = th.as_tensor(mask, dtype=th.float32, device=self.device).reshape(
            batch, -1, 1
        )
        horizon = a.shape[1]
        duration_weight = self._duration_balance_weights(dt_t, balance_dt)
        endpoint_weight = m * duration_weight

        # Per-sample, per-dimension bound on a compounding Euler increment:
        # 5x the largest observed one-step change in this window.
        obs_steps = xp - th.cat([x_hat.unsqueeze(1), xp[:, :-1]], dim=1)  # (B, H, O)
        step_limit = 5.0 * (obs_steps.abs() * m).amax(dim=1) + 1e-3      # (B, O)

        loss = x_hat.new_zeros(())
        for k in range(horizon):
            # Zero duration keeps masked tails fixed and, importantly, prevents
            # their drift from being evaluated at all inside integrate_drift.
            x_next = integrate_drift(
                self.drift,
                x_hat,
                a[:, k],
                dt_t[:, k] * m[:, k],
                max_step=max_step,
                delta_limit=step_limit if max_step is not None else None,
                # The first real-state endpoint is the same unsaturated label as
                # fit_step. Later off-manifold endpoints retain the legacy clamp.
                clamp_final=(k > 0),
            )
            step = x_next - x_hat
            if k == 0:
                # Regress the true, unclamped endpoint, but keep an explosive
                # prediction from becoming the starting state of the next outer
                # transition. Healthy predictions never touch this guard.
                x_loss = x_next
                if max_step is not None:
                    step = th.clamp(step, -step_limit, step_limit)
                x_hat = x_hat + step
            else:
                step = th.clamp(step, -step_limit, step_limit)
                x_hat = x_hat + step
                x_loss = x_hat
            loss = loss + (
                endpoint_weight[:, k] * (x_loss - xp[:, k]).square()
            ).sum()
        loss = loss / (endpoint_weight.sum() * self.obs_dim + 1e-8)
        loss = (
            loss
            + self._mass_regularization(
                th.as_tensor(obs, dtype=th.float32, device=self.device)
            )
            + self._base_damping_regularization()
        )
        return self._fit_optimizer_step(loss, optimizer)
