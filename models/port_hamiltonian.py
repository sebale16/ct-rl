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
from torch.func import jacfwd, vmap


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
    """

    obs_dim: int
    pos_slice: Tuple[int, int]
    vel_slice: Tuple[int, int]
    cyclic_cfg: Tuple[int, ...]
    obs_pos_to_cfg: Tuple[int, ...]
    act_to_cfg: Optional[Tuple[int, ...]] = None

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
        contact_aware: bool = False,
        device: str | th.device = "cpu",
        dof_layout: Optional[DOFLayout] = None,
        structured_hidden: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        self.mode = str(mode)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.human_input_intensity = float(human_input_intensity)
        self.contact_aware = bool(contact_aware)
        self.device = th.device(device)
        self._drift_fn = drift_fn

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
            # Contact-aware damping: the dissipation weights beta_i become a
            # function of the state and the incoming velocity jump dx = x - x_prev
            # (a contact shows up as a large dx). R = softplus(d0) I + L diag(beta) L^T
            # stays PSD (beta >= 0); beta == const recovers the constant-R form.
            if self.contact_aware:
                self.beta_net = nn.Sequential(
                    nn.Linear(2 * d, 64), nn.SiLU(), nn.Linear(64, int(dissipation_rank))
                )
        elif self.mode == "structured":
            self._init_structured(dof_layout, structured_hidden, int(dissipation_rank))
        else:
            raise ValueError(f"Unknown mode '{self.mode}'.")

        self.to(self.device)

    # ------------------------ structured port-Hamiltonian (DeLaN core) ----------

    def _init_structured(
        self, dof_layout: Optional[DOFLayout], hidden: Sequence[int], rank: int
    ) -> None:
        """Structured model: a learned SPD mass matrix M(q) and potential V(q) give
        the Hamiltonian H = V(q) + 1/2 p^T M(q)^-1 p with the canonicalizer
        p = M(q) qd; the drift is the port-Hamiltonian flow (J - R)grad H + G_a a
        with R = diag(D(q,dv)) acting on momentum (passivity dH/dt <= 0). The
        Coriolis terms are generated from M via autodiff, not learned."""
        layout = dof_layout if dof_layout is not None else DOFLayout.cheetah(
            self.obs_dim, self.action_dim
        )
        assert layout.obs_dim == self.obs_dim, "dof_layout.obs_dim must match obs_dim"
        self.layout = layout
        nv, npos = layout.nv, layout.npos

        def mlp(out: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            last = npos
            for h in hidden:
                layers += [nn.Linear(last, h), nn.SiLU()]
                last = h
            layers += [nn.Linear(last, out)]
            return nn.Sequential(*layers)

        self.mass_net = mlp(nv * (nv + 1) // 2)  # lower-triangular Cholesky entries of M(q)
        self.potential_net = mlp(1)              # scalar potential V(q)
        # G_a maps the action (action_dim) to a generalized force. Dense: one force
        # per config DOF (nv). Sparse: one force per (actuator -> DOF) target in
        # act_to_cfg, scattered additively onto the config axis in the drift.
        n_force = nv if layout.act_to_cfg is None else len(layout.act_to_cfg)
        self.G_a = nn.Linear(self.action_dim, n_force, bias=False)
        self._log_d = nn.Parameter(th.full((nv,), -2.0))  # base diagonal damping

        ti = th.tril_indices(nv, nv)
        self.register_buffer("_tri_flat", ti[0] * nv + ti[1])
        self.register_buffer("_eye_nv", th.eye(nv))
        self.register_buffer("_pos_to_cfg", th.tensor(layout.obs_pos_to_cfg, dtype=th.long))
        if layout.act_to_cfg is not None:
            self.register_buffer("_act_to_cfg", th.tensor(layout.act_to_cfg, dtype=th.long))

        if self.contact_aware:
            # Contact-gated dissipation directions (Householder-style, PSD) driven by
            # the velocity jump dv: D(q,dv) = diag(softplus(log_d)) + K diag(beta) K^T,
            # beta = softplus(beta_net([q, dv])) >= 0. Off => diagonal base only.
            self._damp_dirs = nn.Parameter(0.01 * th.randn(nv, rank))
            self.beta_net = nn.Sequential(
                nn.Linear(npos + nv, 64), nn.SiLU(), nn.Linear(64, rank)
            )

    # ------------------------ structure helpers (phast) ------------------------

    def _J(self) -> th.Tensor:
        return self._J_raw - self._J_raw.t()

    def _R(self) -> th.Tensor:
        d0 = th.nn.functional.softplus(self._d0)
        eye = th.eye(self.obs_dim, device=self._L.device)
        return d0 * eye + self._L @ self._L.t()

    def _R_batch(self, x: th.Tensor, dx: th.Tensor) -> th.Tensor:
        """Per-sample PSD dissipation R = softplus(d0) I + L diag(beta) L^T with
        beta = softplus(beta_net([x, dx])) >= 0 gated by the incoming velocity
        jump dx (large dx -> contact -> more damping). Shape (B, d, d)."""
        beta = th.nn.functional.softplus(self.beta_net(th.cat([x, dx], dim=-1)))  # (B, r)
        d0 = th.nn.functional.softplus(self._d0)
        eye = th.eye(self.obs_dim, device=x.device)
        Lb = self._L.unsqueeze(0) * beta.unsqueeze(1)  # (B, d, r): columns scaled by beta
        R = d0 * eye + th.bmm(Lb, self._L.t().unsqueeze(0).expand(x.shape[0], -1, -1))
        return R  # (B, d, d)

    def _grad_H(self, x: th.Tensor) -> th.Tensor:
        # enable_grad so grad H can be taken even when the caller is under
        # th.no_grad() (e.g. evaluation, or the critic's target computation).
        with th.enable_grad():
            xin = x.clone().requires_grad_(True)
            H = self.energy(xin).sum()
            (gH,) = th.autograd.grad(H, xin, create_graph=True)
        return gH

    # ---- structured helpers (per-sample; vmap'd over the batch) ----

    def _mass(self, pos: th.Tensor) -> th.Tensor:
        """SPD mass matrix M(q) = L L^T + eps I from a Cholesky factor with a
        softplus-positive diagonal. Single-sample (pos: (npos,)) for vmap."""
        nv = self.layout.nv
        l = self.mass_net(pos)
        L = th.zeros(nv * nv, dtype=pos.dtype, device=pos.device).scatter(
            0, self._tri_flat, l
        ).reshape(nv, nv)
        d = th.diagonal(L)
        L = L - th.diag_embed(d) + th.diag_embed(th.nn.functional.softplus(d) + 1e-3)
        return L @ L.t() + 1e-4 * self._eye_nv

    def _potential(self, pos: th.Tensor) -> th.Tensor:
        return self.potential_net(pos).squeeze(-1)

    def _damping(self, pos: th.Tensor, dv: th.Tensor) -> th.Tensor:
        """Symmetric PSD damping D(q, dv) = diag(softplus(log_d)) [+ contact term].
        Contact term K diag(softplus(beta([q,dv]))) K^T is added only when
        contact_aware; it stays PSD and vanishes as beta -> 0. Shape (B, nv, nv)."""
        B = pos.shape[0]
        base = th.diag_embed(th.nn.functional.softplus(self._log_d)).unsqueeze(0).expand(B, -1, -1)
        if not self.contact_aware:
            return base
        beta = th.nn.functional.softplus(self.beta_net(th.cat([pos, dv], dim=-1)))  # (B, r)
        Kb = self._damp_dirs.unsqueeze(0) * beta.unsqueeze(1)  # (B, nv, r)
        return base + th.bmm(Kb, self._damp_dirs.t().unsqueeze(0).expand(B, -1, -1))

    def _structured_drift(self, x: th.Tensor, a: th.Tensor, prev_obs) -> th.Tensor:
        """Port-Hamiltonian drift with the canonicalizer p = M(q) qd:
          dH/dq = grad V - 1/2 qd^T (dM/dq) qd,   qd = M^-1 p (= observed velocity)
          pdot  = -dH/dq - D(q,dv) qd + G_a a     (R on momentum -> dH/dt <= 0)
          qddot = M^-1 (pdot - Mdot qd)           (obs-space acceleration)
        The Coriolis terms come from dM/dq (autodiff), not a learned head. Returns
        the observation-space drift [d/dt positions ; qddot]."""
        layout = self.layout
        nv = layout.nv
        B = x.shape[0]
        pos = x[:, layout.pos_slice[0]:layout.pos_slice[1]]
        qd = x[:, layout.vel_slice[0]:layout.vel_slice[1]]

        M = vmap(self._mass)(pos)                       # (B, nv, nv), SPD
        dM_pos = vmap(jacfwd(self._mass))(pos)          # (B, nv, nv, npos) = dM/dq_pos
        gV_pos = vmap(jacfwd(self._potential))(pos)     # (B, npos)
        # scatter the position-config gradients into the nv config axis; cyclic
        # config DOFs are absent from _pos_to_cfg, so their slot stays zero.
        dM = th.zeros(B, nv, nv, nv, dtype=x.dtype, device=x.device).index_copy(
            3, self._pos_to_cfg, dM_pos
        )
        gV = th.zeros(B, nv, dtype=x.dtype, device=x.device).index_copy(1, self._pos_to_cfg, gV_pos)

        if prev_obs is None:
            dv = th.zeros_like(qd)
        else:
            prev = th.as_tensor(prev_obs, dtype=x.dtype, device=x.device)
            dv = qd - prev[:, layout.vel_slice[0]:layout.vel_slice[1]]

        dHdq = gV - 0.5 * th.einsum("nabk,na,nb->nk", dM, qd, qd)
        D_qd = th.bmm(self._damping(pos, dv), qd.unsqueeze(-1)).squeeze(-1)
        if layout.act_to_cfg is None:
            Ga = self.G_a(a)                            # (B, nv)
        else:
            # additive scatter: multiple actuators on one DOF sum their forces
            Ga = th.zeros(B, nv, dtype=x.dtype, device=x.device).index_add(
                1, self._act_to_cfg, self.G_a(a)
            )
        pdot = -dHdq - D_qd + Ga
        Mdot_qd = th.einsum("nabk,nk,nb->na", dM, qd, qd)
        qddot = th.linalg.solve(M, (pdot - Mdot_qd).unsqueeze(-1)).squeeze(-1)  # (B, nv)
        b_pos = qd[:, self._pos_to_cfg]                 # d/dt of each observed position
        return th.cat([b_pos, qddot], dim=-1)           # (B, obs_dim)

    # ------------------------ public API ------------------------

    def drift(self, obs, action, prev_obs=None) -> th.Tensor:
        """Return the drift b(obs, action) of shape (B, obs_dim) on ``self.device``.

        ``prev_obs`` (the previous observation) supplies the incoming velocity
        jump for the contact-aware damping; it is ignored by the mujoco oracle
        and by the constant-R phast model. When it is None the jump is taken as
        zero (graceful fallback to a state-only R(x))."""
        if self.mode == "mujoco":
            b = self._drift_fn(obs, action)  # numpy (B, d)
            return th.as_tensor(np.asarray(b), dtype=th.float32, device=self.device)

        x = th.as_tensor(obs, dtype=th.float32, device=self.device)
        a = th.as_tensor(action, dtype=th.float32, device=self.device)

        if self.mode == "structured":
            return self._structured_drift(x, a, prev_obs)

        gH = self._grad_H(x)  # (B, d)
        if self.contact_aware:
            if prev_obs is None:
                dx = th.zeros_like(x)
            else:
                dx = x - th.as_tensor(prev_obs, dtype=th.float32, device=self.device)
            JR = self._J().unsqueeze(0) - self._R_batch(x, dx)  # (B, d, d)
            b = th.bmm(JR, gH.unsqueeze(-1)).squeeze(-1)  # (B, d)
            return b + self.G_a(a)
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

    def _fit_optimizer_step(self, loss: th.Tensor, optimizer) -> float:
        """backward + clipped step, skipping the update (not the report) when
        the loss or the gradient norm is non-finite, so a pathological batch
        cannot write NaN into the parameters."""
        optimizer.zero_grad()
        if not th.isfinite(loss):
            return float(loss.detach())
        loss.backward()
        total_norm = th.nn.utils.clip_grad_norm_(
            [p for g in optimizer.param_groups for p in g["params"]],
            self.fit_max_grad_norm,
        )
        if th.isfinite(total_norm):
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return float(loss.detach())

    def fit_step(self, obs, action, next_obs, dt, optimizer, prev_obs=None) -> float:
        """One supervised PHAST data step: minimize ||(x + b*dt) - x'||^2 (mode='phast').

        ``prev_obs`` feeds the contact-aware damping (ignored by the constant-R model)."""
        assert self.mode in ("phast", "structured"), (
            "fit_step only applies to learned modes ('phast', 'structured')."
        )
        x = th.as_tensor(obs, dtype=th.float32, device=self.device)
        a = th.as_tensor(action, dtype=th.float32, device=self.device)
        xp = th.as_tensor(next_obs, dtype=th.float32, device=self.device)
        dt_t = th.as_tensor(dt, dtype=th.float32, device=self.device).reshape(-1, 1)
        pred = x + self.drift(x, a, prev_obs=prev_obs) * dt_t
        loss = ((pred - xp) ** 2).mean()
        return self._fit_optimizer_step(loss, optimizer)

    def fit_step_rollout(
        self, obs, actions, next_obs, dt, mask, optimizer, prev_obs=None
    ) -> float:
        """Multi-step (rollout) data step: roll the model H explicit Euler steps
        from x_t along the recorded actions and regress every rolled state onto
        the observed sequence,

          x_hat_0 = x_t,   x_hat_{k+1} = x_hat_k + b(x_hat_k, a_{t+k}; x_hat_{k-1}) dt_{t+k},
          loss = sum_k m_k ||x_hat_{k+1} - x_{t+k+1}||^2 / (obs_dim * sum_k m_k).

        Gradients flow through the whole roll (backprop through time), so the
        model is trained where the generator/quadrature targets consume it: on
        its own predictions, not only on buffer states. The velocity jump for
        the contact gate uses the real predecessor at k=0 and the rolled states
        after, the convention of the sub-step quadrature roll. ``mask`` (1 valid,
        0 invalid, cumulative) zeroes the steps of a window that crosses an
        episode end or the ring seam; ``horizon=1`` with a full mask is exactly
        ``fit_step``.

        The compounding steps (k >= 1) evaluate the drift at the model's own
        predictions, where nothing bounds it — the Coriolis term is quadratic
        in velocity, so one bad prediction can blow up the rest of the roll,
        overflow the loss, and NaN the parameters through BPTT. Each such
        increment is therefore clamped elementwise to a generous multiple of
        the window's own observed increments; healthy rolls never touch the
        bound, and a run-away roll saturates instead of compounding (saturated
        entries also stop back-propagating). The first step is taken from a
        real buffer state and is left unclamped, so horizon=1 remains exactly
        ``fit_step``. The optimizer step itself is clipped and skipped on a
        non-finite loss (``_fit_optimizer_step``).

        Shapes: obs (B, O); actions (B, H, A); next_obs (B, H, O);
        dt, mask (B, H, 1); prev_obs (B, O) or None."""
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

        # Per-sample, per-dimension bound on a compounding Euler increment:
        # 5x the largest observed one-step change in this window.
        obs_steps = xp - th.cat([x_hat.unsqueeze(1), xp[:, :-1]], dim=1)  # (B, H, O)
        step_limit = 5.0 * obs_steps.abs().amax(dim=1) + 1e-3            # (B, O)

        prev = prev_obs
        loss = x_hat.new_zeros(())
        for k in range(horizon):
            b = self.drift(x_hat, a[:, k], prev_obs=prev)
            prev = x_hat
            # The update is gated by the mask so an invalid tail cannot run the
            # state off to non-finite values (0 * inf would poison the masked loss).
            step = b * (dt_t[:, k] * m[:, k])
            if k > 0:
                step = th.clamp(step, -step_limit, step_limit)
            x_hat = x_hat + step
            loss = loss + (m[:, k] * (x_hat - xp[:, k]) ** 2).sum()
        loss = loss / (m.sum() * self.obs_dim + 1e-8)
        return self._fit_optimizer_step(loss, optimizer)
