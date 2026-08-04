#!/usr/bin/env python
"""Reusable probe for the contact solver's compliance, independent of the fit.

Every number this module produces isolates the *contact* code: MuJoCo's own mass
matrix (``mj_fullM``) and its own contact-free generalized acceleration are
substituted for the model's learned mechanics, so a freshly-constructed
(random-weight) ``PortHamiltonianModel`` still gives physically meaningful
forces. What remains under test is exactly the contact geometry plus
``_constraint_contact_solve``.

The entry points all take the model as an argument, so a fixed model can be
handed to the identical code later and the before/after numbers are comparable
by construction:

    make_probe_model(...)                 build a kinematic/constraint model
    MuJoCoCheetahMechanics()              M(q) and qdd_free from the simulator
    probe_qpos(model, mech, gap)          a held state at a chosen minimum gap
    static_force(model, qpos, ...)        normal force at a held state
    qp_matrices(model, ...)               W_full, S, R, Rtilde, H, cond(H)
    drop_test(model, ...)                 rest gap / force / settle diagnostics
    mujoco_reference_rest(mech, ...)      what the simulator itself does
    reg_sweep(...) / constitutive_sweep(...)

``qp_matrices`` recovers the solver's own ``H`` by intercepting the Cholesky
call rather than re-deriving it, so it keeps reporting the true regularizer even
after ``R`` is changed. ``R`` is then read off as ``H - sym(S W_full S)`` and the
physical compliance as ``Rtilde = S^-1 R S^-1``.

Also provides two standalone algebra checks that do not need MuJoCo at all:

    verify_cone_invariance()      P_Cone(S x) == S P_Cone(x) for uniform S
    verify_change_of_variables()  the latent minimizer y and the physical
                                  minimizer Lambda satisfy Lambda = S y
"""
from __future__ import annotations

import hashlib
import os
from typing import Callable, Iterable, Optional, Sequence

os.environ.setdefault("MUJOCO_GL", "disable")

import numpy as np
import torch as th

from models.port_hamiltonian import PortHamiltonianModel, DOFLayout

# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def md5(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.md5(handle.read()).hexdigest()


DEPENDENCIES = ("models/port_hamiltonian.py", "evaluations/contact_compliance_probe.py")


def dependency_md5(root: str) -> dict[str, str]:
    """md5 of every source this module's numbers depend on."""
    return {name: md5(os.path.join(root, name)) for name in DEPENDENCIES}


# ---------------------------------------------------------------------------
# mechanics substitution
# ---------------------------------------------------------------------------


class MuJoCoCheetahMechanics:
    """MuJoCo's mass matrix and contact-free acceleration for the cheetah.

    ``mass_and_free`` deliberately ignores ``data.qfrc_constraint``: the free
    acceleration is what the body would do with no contact at all, which is the
    input the model's contact solver expects.
    """

    def __init__(self, domain: str = "cheetah", task: str = "run"):
        import mujoco  # noqa: PLC0415  (kept local so the algebra checks import clean)
        from dm_control import suite  # noqa: PLC0415

        self._mujoco = mujoco
        self._env = suite.load(domain, task)
        self.physics = self._env.physics
        self.mj_model = self.physics.model
        self.mj_data = self.physics.data
        self.nq = int(self.mj_model.nq)
        self.nv = int(self.mj_model.nv)
        self.total_mass = float(np.sum(self.mj_model.body_mass))
        self.gravity = float(-self.mj_model.opt.gravity[2])
        self.weight_N = self.total_mass * self.gravity

    def mass_and_free(self, qpos, qvel, ctrl=None):
        self.mj_data.qpos[:] = qpos
        self.mj_data.qvel[:] = qvel
        self.mj_data.ctrl[:] = 0.0 if ctrl is None else ctrl
        self.physics.forward()
        M = np.zeros((self.nv, self.nv))
        self._mujoco.mj_fullM(self.mj_model.ptr, M, self.mj_data.qM)
        force = (
            self.mj_data.qfrc_actuator[: self.nv]
            + self.mj_data.qfrc_passive[: self.nv]
            - self.mj_data.qfrc_bias[: self.nv]
        )
        return M, np.linalg.solve(M, force)

    def step_reference(self, qpos0, qvel0=None, steps: int = 3000):
        """Let MuJoCo itself run from the same initial state."""
        self.mj_data.qpos[:] = qpos0
        self.mj_data.qvel[:] = 0.0 if qvel0 is None else qvel0
        self.mj_data.ctrl[:] = 0.0
        self._mujoco.mj_forward(self.mj_model.ptr, self.mj_data.ptr)
        for _ in range(steps):
            self._mujoco.mj_step(self.mj_model.ptr, self.mj_data.ptr)
        return np.array(self.mj_data.qpos), np.array(self.mj_data.qvel)


# ---------------------------------------------------------------------------
# model construction and state plumbing
# ---------------------------------------------------------------------------


def make_probe_model(
    seed: int = 0,
    contact_regularization: float = 0.01,
    contact_gate_off: Optional[float] = None,
    contact_force: int = 6,
    contact_geometry: str = "kinematic",
    contact_solver: str = "constraint",
    contact_dt: float = 0.002,
    contact_iterations: int = 12,
    structured_hidden: Sequence[int] = (128, 128),
    double: bool = True,
    **kwargs,
) -> PortHamiltonianModel:
    """A fresh cheetah model. Weights are random; only contact code is probed."""
    th.manual_seed(seed)
    model = PortHamiltonianModel(
        obs_dim=17,
        action_dim=6,
        mode="structured",
        structured_hidden=tuple(structured_hidden),
        contact_force=contact_force,
        contact_solver=contact_solver,
        contact_geometry=contact_geometry,
        contact_gate_off=contact_gate_off,
        contact_dt=contact_dt,
        contact_iterations=contact_iterations,
        contact_regularization=contact_regularization,
        dof_layout=DOFLayout.cheetah(),
        **kwargs,
    )
    model.eval()
    return model.double() if double else model


def split_state(model: PortHamiltonianModel, qpos, qvel):
    """MuJoCo (qpos, qvel) -> the (pos, qd) tensors the contact code consumes."""
    lo, hi = model.layout.pos_slice
    npos = hi - lo
    dtype = next(model.parameters()).dtype
    pos = th.as_tensor(np.asarray(qpos)[1 : 1 + npos], dtype=dtype)[None]
    qd = th.as_tensor(np.asarray(qvel)[: model.layout.nv], dtype=dtype)[None]
    return pos, qd


def contact_solve(model, mech, qpos, qvel=None, ctrl=None):
    """Run the model's own contact solver on MuJoCo mechanics at one state."""
    qpos = np.asarray(qpos, dtype=float)
    qvel = np.zeros(mech.nv) if qvel is None else np.asarray(qvel, dtype=float)
    M, qdd_free = mech.mass_and_free(qpos, qvel, ctrl)
    pos, qd = split_state(model, qpos, qvel)
    dtype = pos.dtype
    Mt = th.as_tensor(M, dtype=dtype)[None]
    ft = th.as_tensor(qdd_free, dtype=dtype)[None]
    with th.no_grad():
        out = model._constraint_contact_solve(pos, qd, Mt, ft)
    return out, M, qdd_free, pos, qd


def model_gaps(model, qpos, qvel=None, nv: Optional[int] = None) -> np.ndarray:
    nv = model.layout.nv if nv is None else nv
    pos, qd = split_state(model, qpos, np.zeros(nv) if qvel is None else qvel)
    with th.no_grad():
        g = model._contact_geometry(pos, qd)[0]
    return g[0].numpy()


def probe_qpos(model, mech, gap: float, base_qpos=None) -> np.ndarray:
    """A zero-velocity state whose *minimum* contact gap equals ``gap``.

    All contact gaps are affine in the root height with unit slope, so a single
    shift places the lowest point exactly at the requested clearance.
    """
    qpos = np.zeros(mech.nq) if base_qpos is None else np.array(base_qpos, dtype=float)
    qpos[1] = 0.0
    current = float(model_gaps(model, qpos, nv=mech.nv).min())
    qpos[1] = gap - current
    return qpos


# ---------------------------------------------------------------------------
# QP introspection
# ---------------------------------------------------------------------------


def qp_matrices(model, mech, qpos, qvel=None, ctrl=None) -> dict:
    """The solver's own H, plus the physical-coordinate decomposition.

    ``H`` is captured from the argument the solver hands to ``cholesky_ex``
    (``H + rho I``), so this stays correct when the definition of ``R`` changes.
    ``rho`` is rebuilt here exactly as the solver builds it, from the gated
    Delassus diagonal plus the conditioning floor, so the recovery does not
    depend on ``R`` either. ``W_full = J M^-1 J^T`` and the gate come from the
    same geometry call the solver makes, hence

        R      = H - sym(S W_full S)
        Rtilde = S^-1 R S^-1        (the physical compliance)
    """
    captured: list[th.Tensor] = []
    real_cholesky = th.linalg.cholesky_ex

    def spy(A, *args, **kw):
        captured.append(A.detach().clone())
        return real_cholesky(A, *args, **kw)

    th.linalg.cholesky_ex = spy
    try:
        out, M, qdd_free, pos, qd = contact_solve(model, mech, qpos, qvel, ctrl)
    finally:
        th.linalg.cholesky_ex = real_cholesky
    if not captured:
        raise RuntimeError("solver did not factor a matrix; cannot recover H")
    A = captured[-1]
    eye = th.eye(A.shape[-1], dtype=A.dtype)[None]

    with th.no_grad():
        g, gdot, v_t, J_n, J_t = model._contact_geometry(pos, qd)
        gate = model._contact_gate(g)
        B, K = gate.shape
        J = th.stack((J_n, J_t), dim=2).reshape(B, 2 * K, -1)
        Mt = th.as_tensor(M, dtype=pos.dtype)[None]
        M_inv_Jt = th.linalg.solve(Mt, J.transpose(1, 2))
        W_full = th.bmm(J, M_inv_Jt)
        scale = W_full.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-6)
        S = gate.unsqueeze(-1).expand(-1, -1, 2).reshape(B, 2 * K)
        W = th.bmm(J * S.unsqueeze(-1), M_inv_Jt * S.unsqueeze(1))
        W = 0.5 * (W + W.transpose(1, 2))
        # The solver's own step size: the mean of the gated Delassus diagonal
        # plus the conditioning floor, which is independent of R by
        # construction. Reproduced branch for branch (the two expressions agree
        # elementwise but not bit for bit, because a mean over a strided
        # diagonal view does not round like a mean over a contiguous vector).
        reg_f = float(model.contact_regularization)
        if getattr(model, "_contact_compliance_raw", None) is None:
            legacy = W + reg_f * scale[:, None, None] * eye
            rho = legacy.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-6)
        else:
            rho = (
                (W.diagonal(dim1=-2, dim2=-1) + reg_f * scale[:, None])
                .mean(-1)
                .clamp_min(1e-6)
            )
        H = A - rho[:, None, None] * eye
        R = H - W
        inv_S = 1.0 / S.clamp_min(1e-300)
        Rtilde = R * inv_S.unsqueeze(-1) * inv_S.unsqueeze(1)

    return {
        "H": H,
        "H_plus_rho": A,
        "rho": rho,
        "W": W,
        "W_full": W_full,
        "R": R,
        "Rtilde": Rtilde,
        "gate": gate,
        "gate_pair": S,
        "scale": scale,
        "cond_H": float(np.linalg.cond(H[0].numpy())),
        "cond_H_plus_rho": float(np.linalg.cond(A[0].numpy())),
        "cond_W_full": float(np.linalg.cond(W_full[0].numpy())),
        "R_diag": R.diagonal(dim1=-2, dim2=-1)[0].numpy(),
        "Rtilde_diag": Rtilde.diagonal(dim1=-2, dim2=-1)[0].numpy(),
        "solver_out": out,
    }


# ---------------------------------------------------------------------------
# held-state force
# ---------------------------------------------------------------------------


def _scalar_or_list(tensor):
    flat = tensor.detach().reshape(-1)
    return float(flat[0]) if flat.numel() == 1 else [float(v) for v in flat]


def _summarize(out, mech=None) -> dict:
    normal = out["normal_force"][0].numpy()
    tangent = out["tangent_force"][0].numpy()
    row = {
        "gap_m": [float(v) for v in out["gap"][0].numpy()],
        "min_gap_m": float(out["gap"][0].min()),
        "gate": [float(v) for v in out["gate"][0].numpy()],
        "normal_force_N": [float(v) for v in normal],
        "total_normal_force_N": float(normal.sum()),
        "total_tangent_force_N": float(tangent.sum()),
        "solver_residual": float(out["solver_residual"][0]),
        "cone_violation": float(out["cone_violation"][0]),
        "mu": _scalar_or_list(out["mu"]),
        "e": _scalar_or_list(out["e"]),
        "beta": _scalar_or_list(out["beta"]),
        "regularization": float(out["regularization"]),
    }
    if mech is not None:
        row["total_normal_force_over_weight"] = row["total_normal_force_N"] / mech.weight_N
    return row


def static_force(model, mech, qpos, qvel=None, ctrl=None, with_cond: bool = False) -> dict:
    """Normal force the solver produces at a held state."""
    out, _, _, _, _ = contact_solve(model, mech, qpos, qvel, ctrl)
    row = _summarize(out, mech)
    if with_cond:
        qp = qp_matrices(model, mech, qpos, qvel, ctrl)
        row["cond_H"] = qp["cond_H"]
        row["cond_H_plus_rho"] = qp["cond_H_plus_rho"]
        row["cond_W_full"] = qp["cond_W_full"]
        row["scale"] = float(qp["scale"][0])
        row["R_diag"] = [float(v) for v in qp["R_diag"]]
        row["Rtilde_diag"] = [float(v) for v in qp["Rtilde_diag"]]
    return row


def rigid_reference(
    model, mech, qpos, qvel=None, ctrl=None, gate_threshold: float = 1e-12,
    reg: float = 1e-12, iterations: int = 4000,
) -> dict:
    """The reg -> 0 limit: a hard (non-compliant) solve on the same active set.

    Solves the physical-coordinate cone QP ``min 0.5 L' W_full L + b' L`` over
    the contacts the gate currently admits, with only a numerically negligible
    regularizer. This is the force a rigid contact would carry at this state and
    is the yardstick the compliant answers are compared against.
    """
    qpos = np.asarray(qpos, dtype=float)
    qvel = np.zeros(mech.nv) if qvel is None else np.asarray(qvel, dtype=float)
    M, qdd_free = mech.mass_and_free(qpos, qvel, ctrl)
    pos, qd = split_state(model, qpos, qvel)
    dtype = pos.dtype
    dt = float(model.contact_dt)
    with th.no_grad():
        g, gdot, v_t, J_n, J_t = model._contact_geometry(pos, qd)
        gate = model._contact_gate(g)
        B, K = gate.shape
        J = th.stack((J_n, J_t), dim=2).reshape(B, 2 * K, -1)
        Mt = th.as_tensor(M, dtype=dtype)[None]
        W_full = th.bmm(J, th.linalg.solve(Mt, J.transpose(1, 2)))
        scale = W_full.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-6)

        e = 0.5 * th.sigmoid(model._contact_raw[0])
        beta = 0.5 * th.sigmoid(model._contact_raw[1])
        mu = 2.0 * th.sigmoid(model._contact_raw[2])
        qd_free = qd + dt * th.as_tensor(qdd_free, dtype=dtype)[None]
        v_free = th.einsum("ncv,nv->nc", J, qd_free).reshape(B, K, 2)
        penetration = (th.relu(-g) / dt).clamp_max(model._contact_max_correction_vel)
        b = th.stack(
            (
                v_free[..., 0] - beta * penetration
                + e * th.minimum(gdot, th.zeros_like(gdot)),
                v_free[..., 1],
            ),
            dim=-1,
        ).reshape(B, 2 * K)

        active = (gate > gate_threshold)[0]
        idx = th.nonzero(active.repeat_interleave(2)).squeeze(-1)
        if idx.numel() == 0:
            return {"active_contacts": 0, "total_normal_force_N": 0.0,
                    "normal_force_N": [0.0] * K, "residual": 0.0}
        Wa = W_full[:, idx][:, :, idx]
        ba = b[:, idx]
        mu_a = mu.expand(K)[active] if mu.numel() == K else mu.reshape(-1).expand(K)[active]
        Ha = Wa + reg * scale[:, None, None] * th.eye(idx.numel(), dtype=dtype)[None]
        lam = solve_cone_qp(Ha, ba, mu_a, iterations)
        resid = float(cone_qp_residual(lam, Ha, ba, mu_a)[0])
        full = th.zeros(1, 2 * K, dtype=dtype)
        full[:, idx] = lam
        normal = (full.reshape(1, K, 2)[..., 0] / dt)[0].numpy()
    return {
        "active_contacts": int(active.sum()),
        "gate": [float(v) for v in gate[0].numpy()],
        "normal_force_N": [float(v) for v in normal],
        "total_normal_force_N": float(normal.sum()),
        "residual": resid,
        "reg": reg,
    }


# ---------------------------------------------------------------------------
# drop test
# ---------------------------------------------------------------------------


def drop_test(
    model,
    mech,
    steps: int = 3000,
    dt: Optional[float] = None,
    z0: float = 0.6,
    settle_tol: float = 1e-3,
    ctrl=None,
) -> dict:
    """Drop the cheetah and report where the model's contact lets it rest.

    Semi-implicit Euler at the solver's own ``contact_dt``; the mass matrix and
    free acceleration come from MuJoCo at every step, so only the contact
    response is the model's.
    """
    dt = float(model.contact_dt) if dt is None else float(dt)
    qpos = np.zeros(mech.nq)
    qpos[1] = z0
    qvel = np.zeros(mech.nv)
    speed_trace = np.empty(steps)
    out = None
    diverged = False
    for i in range(steps):
        out, _, qdd_free, _, _ = contact_solve(model, mech, qpos, qvel, ctrl)
        acc = qdd_free + out["contact_acceleration"][0].numpy()
        qvel = qvel + dt * acc
        qpos[: mech.nv] = qpos[: mech.nv] + dt * qvel
        speed_trace[i] = np.abs(qvel).max()
        if not np.all(np.isfinite(qpos)):
            diverged = True
            speed_trace = speed_trace[: i + 1]
            break

    settled = np.nonzero(speed_trace >= settle_tol)[0]
    steps_to_settle = int(settled[-1] + 1) if settled.size else 0
    if steps_to_settle >= len(speed_trace):
        steps_to_settle = -1  # never settled

    # Re-solve at the final state so the reported gap/force belong to the same
    # configuration as the reported rest qpos, not to the step before it.
    if not diverged:
        out, _, _, _, _ = contact_solve(model, mech, qpos, qvel, ctrl)
    row = _summarize(out, mech)
    row.update(
        {
            "diverged": bool(diverged),
            "steps": int(len(speed_trace)),
            "dt": dt,
            "z0": z0,
            "contact_gate_off": float(model._contact_gate_off),
            "contact_regularization": float(model.contact_regularization),
            "rest_root_z_m": float(qpos[1]),
            "rest_qpos": [float(v) for v in qpos],
            "rest_max_abs_qvel": float(np.abs(qvel).max()),
            "settle_tol": settle_tol,
            "steps_to_settle": steps_to_settle,
            "rest_gap_m": row.pop("gap_m"),
            "rest_min_gap_m": row.pop("min_gap_m"),
            "rest_total_normal_force_N": row.pop("total_normal_force_N"),
        }
    )
    row["rest_total_normal_force_over_weight"] = (
        row["rest_total_normal_force_N"] / mech.weight_N
    )
    row.pop("total_normal_force_over_weight", None)
    return row


def mujoco_reference_rest(model, mech, z0: float = 0.6, steps: int = 3000) -> dict:
    """Where MuJoCo itself rests, read through the model's own gap function."""
    qpos0 = np.zeros(mech.nq)
    qpos0[1] = z0
    qpos, qvel = mech.step_reference(qpos0, steps=steps)
    gaps = model_gaps(model, qpos, nv=mech.nv)
    return {
        "steps": steps,
        "z0": z0,
        "rest_root_z_m": float(qpos[1]),
        "rest_qpos": [float(v) for v in qpos],
        "rest_max_abs_qvel": float(np.abs(qvel).max()),
        "model_gap_at_mujoco_rest_m": [float(v) for v in gaps],
        "model_min_gap_at_mujoco_rest_m": float(gaps.min()),
    }


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------


def reg_sweep(
    model_factory: Callable[..., PortHamiltonianModel],
    mech,
    regs: Iterable[float],
    gap: float,
    gate_off: Optional[float] = None,
    **factory_kwargs,
) -> list[dict]:
    """Normal force at a fixed probe gap as ``contact_regularization`` varies."""
    rows = []
    for reg in regs:
        model = model_factory(
            contact_regularization=float(reg), contact_gate_off=gate_off, **factory_kwargs
        )
        qpos = probe_qpos(model, mech, gap)
        row = static_force(model, mech, qpos, with_cond=True)
        row["contact_regularization"] = float(reg)
        row["contact_gate_off"] = float(model._contact_gate_off)
        row["probe_gap_m"] = float(gap)
        rows.append(row)
    return rows


# physical <-> raw parameterization used by _constraint_contact_solve
_RAW_SLOT = {"e": (0, 0.5), "beta": (1, 0.5), "mu": (2, 2.0)}


def set_constitutive(model, name: str, value: float) -> None:
    """Set a learned constitutive parameter by its *physical* value."""
    if name in _RAW_SLOT:
        slot, span = _RAW_SLOT[name]
        frac = float(value) / span
        if not 0.0 < frac < 1.0:
            raise ValueError(f"{name}={value} outside (0, {span})")
        with th.no_grad():
            model._contact_raw[slot] = float(np.log(frac / (1.0 - frac)))
        return
    # Forward-compatible: a later phase adds a learned compliance parameter.
    for attr in (f"_contact_{name}", f"_contact_{name}_raw", name):
        if hasattr(model, attr):
            tensor = getattr(model, attr)
            with th.no_grad():
                tensor.fill_(float(value))
            return
    raise AttributeError(f"model has no constitutive parameter {name!r}")


def constitutive_sweep(
    model_factory: Callable[..., PortHamiltonianModel],
    mech,
    name: str,
    values: Iterable[float],
    gap: float,
    gate_off: Optional[float] = None,
    contact_regularization: float = 0.01,
    **factory_kwargs,
) -> list[dict]:
    """Normal force at a fixed probe gap as one constitutive parameter varies."""
    rows = []
    for value in values:
        model = model_factory(
            contact_regularization=float(contact_regularization),
            contact_gate_off=gate_off,
            **factory_kwargs,
        )
        set_constitutive(model, name, value)
        qpos = probe_qpos(model, mech, gap)
        row = static_force(model, mech, qpos, with_cond=True)
        row["parameter"] = name
        row["value"] = float(value)
        row["contact_regularization"] = float(contact_regularization)
        row["contact_gate_off"] = float(model._contact_gate_off)
        row["probe_gap_m"] = float(gap)
        rows.append(row)
    return rows


def compliance_identity(model, mech, qpos, qvel=None, ctrl=None,
                        gate_threshold: float = 1e-6) -> dict:
    """Measure the contact compliance the solver actually implements.

    Two things are checked against the code as it runs, not against a rederived
    formula:

      1. ``Rtilde = S^-1 (H - sym(S W_full S)) S^-1`` is the physical-coordinate
         regularizer. It is compared to the closed form implied by the current
         ``R``; at baseline that form is ``reg * scale / gate^2``.
      2. At an interior optimum, ``v+ - v* = -Rtilde Lambda`` exactly, i.e. the
         solver leaves a velocity-level constraint violation proportional to the
         impulse. That residual IS the compliance.

    Only contacts the gate admits are used; a gated-off contact has S = 0, so
    Rtilde is infinite and Lambda is zero and the identity is vacuous.
    """
    qp = qp_matrices(model, mech, qpos, qvel, ctrl)
    out = qp["solver_out"]
    dtype = qp["H"].dtype
    dt = float(model.contact_dt)
    gate = qp["gate"]
    S = qp["gate_pair"]
    B, K = gate.shape
    scale = qp["scale"]
    reg = float(model.contact_regularization)

    baseline_form = (reg * scale[:, None] / S.clamp_min(1e-300).square())[0].numpy()
    measured = np.asarray(qp["Rtilde_diag"], dtype=float)

    active = (gate > gate_threshold)[0]
    idx = th.nonzero(active.repeat_interleave(2)).squeeze(-1)

    # rebuild the solver's own bias vector so v* is available
    qpos = np.asarray(qpos, dtype=float)
    qvel = np.zeros(mech.nv) if qvel is None else np.asarray(qvel, dtype=float)
    M, qdd_free = mech.mass_and_free(qpos, qvel, ctrl)
    pos, qd = split_state(model, qpos, qvel)
    with th.no_grad():
        g, gdot, v_t, J_n, J_t = model._contact_geometry(pos, qd)
        J = th.stack((J_n, J_t), dim=2).reshape(B, 2 * K, -1)
        e = 0.5 * th.sigmoid(model._contact_raw[0])
        beta = 0.5 * th.sigmoid(model._contact_raw[1])
        qd_free = qd + dt * th.as_tensor(qdd_free, dtype=dtype)[None]
        v_free = th.einsum("ncv,nv->nc", J, qd_free).reshape(B, K, 2)
        penetration = (th.relu(-g) / dt).clamp_max(model._contact_max_correction_vel)
        b = th.stack(
            (
                v_free[..., 0] - beta * penetration
                + e * th.minimum(gdot, th.zeros_like(gdot)),
                v_free[..., 1],
            ),
            dim=-1,
        ).reshape(B, 2 * K)
        v_free_flat = v_free.reshape(B, 2 * K)
        v_target = v_free_flat - b
        v_post = out["contact_velocity_post"].reshape(B, 2 * K)
        lam = th.stack((out["normal_impulse"], out["tangent_impulse"]),
                       dim=-1).reshape(B, 2 * K)
        gap_error = v_post - v_target
        predicted = -th.bmm(qp["Rtilde"][:, idx][:, :, idx],
                            lam[:, idx].unsqueeze(-1)).squeeze(-1) if idx.numel() \
            else th.zeros(B, 0, dtype=dtype)
        actual = gap_error[:, idx]
    denom = float(actual.abs().max()) if idx.numel() else 0.0
    err = float((actual - predicted).abs().max()) if idx.numel() else 0.0
    return {
        "active_contacts": int(active.sum()),
        "gate": [float(v) for v in gate[0].numpy()],
        "scale": float(scale[0]),
        "contact_regularization": reg,
        "Rtilde_diag_measured": [float(v) for v in measured],
        "Rtilde_diag_reg_over_gate_squared": [float(v) for v in baseline_form],
        "Rtilde_matches_reg_over_gate_squared": bool(
            np.all(np.isfinite(baseline_form[np.asarray(active.numpy()).repeat(2)]))
            and np.allclose(measured[np.asarray(active.numpy()).repeat(2)],
                            baseline_form[np.asarray(active.numpy()).repeat(2)],
                            rtol=1e-9, atol=0.0)
        ),
        "impedance_Z_active": [float(1.0 / v) for v in
                               measured[np.asarray(active.numpy()).repeat(2)]],
        "velocity_violation_measured": [float(v) for v in actual[0].numpy()],
        "velocity_violation_predicted": [float(v) for v in predicted[0].numpy()],
        "velocity_identity_max_abs_error": err,
        "velocity_identity_max_rel_error": err / denom if denom > 0 else 0.0,
        "velocity_identity_holds": bool(denom > 0 and err < 1e-6 * denom),
        "solver_residual": float(out["solver_residual"][0]),
    }


def gate_sweep(
    model_factory,
    mech,
    gate_offs,
    gap: float,
    contact_regularization: float = 0.01,
    **factory_kwargs,
) -> list[dict]:
    """Force vs the gate at a fixed geometry: the gate's only channel is Rtilde.

    Widening the band raises the gate at a fixed gap. At a large regularizer that
    changes the force a lot (the gate IS the stiffness); as reg -> 0 the same
    sweep flattens onto the rigid answer (the gate cancels).
    """
    rows = []
    for off in gate_offs:
        model = model_factory(contact_regularization=float(contact_regularization),
                              contact_gate_off=float(off), **factory_kwargs)
        qpos = probe_qpos(model, mech, gap)
        row = static_force(model, mech, qpos, with_cond=True)
        row["contact_gate_off"] = float(off)
        row["contact_regularization"] = float(contact_regularization)
        row["probe_gap_m"] = float(gap)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# standalone algebra checks (no MuJoCo)
# ---------------------------------------------------------------------------


def solve_cone_qp(H, c, mu, iterations: int = 40000, over_relax: float = 1.5):
    """min 0.5 y'Hy + c'y over the product of planar Coulomb cones.

    Same scaled-ADMM shape the model uses, run far past its 12 iterations so the
    minimizer is exact to solver tolerance.
    """
    B, n = c.shape
    eye = th.eye(n, dtype=H.dtype)[None]
    rho = H.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12)
    factor, info = th.linalg.cholesky_ex(H + rho[:, None, None] * eye)
    if bool(info.any()):
        raise RuntimeError("H + rho I not positive definite")
    primal = th.zeros_like(c)
    auxiliary = th.zeros_like(c)
    dual = th.zeros_like(c)
    for _ in range(iterations):
        rhs = -c + rho[:, None] * (auxiliary - dual)
        primal = th.cholesky_solve(rhs.unsqueeze(-1), factor).squeeze(-1)
        relaxed = over_relax * primal - (over_relax - 1.0) * auxiliary
        nxt = PortHamiltonianModel._project_contact_cones(relaxed + dual, mu)
        dual = dual + relaxed - nxt
        auxiliary = nxt
    return auxiliary


def cone_qp_residual(y, H, c, mu):
    """Normalized projected-gradient (fixed-point) optimality residual."""
    grad = th.bmm(H, y.unsqueeze(-1)).squeeze(-1) + c
    lip = H.abs().sum(-1).amax(-1).clamp_min(1e-12)
    prox = PortHamiltonianModel._project_contact_cones(y - grad / lip[:, None], mu)
    return ((y - prox).norm(dim=-1) / (1.0 + y.norm(dim=-1)))


def verify_cone_invariance(trials: int = 2000, K: int = 6, seed: int = 0) -> dict:
    """P_Cone(S x) == S P_Cone(x) for S uniform and positive within each block."""
    gen = th.Generator().manual_seed(seed)
    x = th.randn(trials, 2 * K, generator=gen, dtype=th.float64) * 10.0
    # deliberately include the interior, the polar cone, and the boundary region
    x[: trials // 3, 0::2] = x[: trials // 3, 0::2].abs() * 1e-3
    gate = th.rand(trials, K, generator=gen, dtype=th.float64).clamp_min(1e-6)
    S = gate.unsqueeze(-1).expand(-1, -1, 2).reshape(trials, 2 * K)
    mu = th.rand(K, generator=gen, dtype=th.float64) * 2.0
    proj = PortHamiltonianModel._project_contact_cones
    left = proj(S * x, mu)
    right = S * proj(x, mu)
    err = (left - right).abs()
    denom = right.abs().amax().clamp_min(1e-12)
    return {
        "trials": trials,
        "K": K,
        "max_abs_error": float(err.max()),
        "max_rel_error": float(err.max() / denom),
        "passed": bool(err.max() < 1e-12 * float(denom) + 1e-14),
    }


def verify_change_of_variables(
    trials: int = 24,
    K: int = 4,
    seed: int = 0,
    reg: float = 1e-2,
    iterations: int = 40000,
    tol: float = 1e-8,
) -> dict:
    """Latent minimizer y and physical minimizer Lambda satisfy Lambda = S y.

    Random PSD ``W_full``, random gates in (0, 1], random bias, random mu. The
    bias is scaled over several decades so that across the batch some contacts
    saturate the friction cone, some are pushed to ``lambda_n = 0``, and some
    stay interior.
    """
    gen = th.Generator().manual_seed(seed)
    n = 2 * K
    rows = []
    worst = {"max_rel_error": 0.0}
    for t in range(trials):
        A = th.randn(1, n, n, generator=gen, dtype=th.float64)
        W_full = th.bmm(A, A.transpose(1, 2)) / n + 1e-3 * th.eye(n, dtype=th.float64)[None]
        gate = th.rand(1, K, generator=gen, dtype=th.float64).clamp_min(1e-3)
        S = gate.unsqueeze(-1).expand(-1, -1, 2).reshape(1, n)
        mu = th.rand(K, generator=gen, dtype=th.float64).clamp_min(0.05) * 2.0
        b = th.randn(1, n, generator=gen, dtype=th.float64)
        # decade-spaced magnitudes and a strongly tangential branch push
        # different trials into different active sets
        b = b * (10.0 ** (t % 5 - 2))
        if t % 3 == 1:
            b[:, 1::2] *= 25.0            # heavy tangential drive -> cone boundary
        if t % 3 == 2:
            b[:, 0::2] = b[:, 0::2].abs()  # separating normals -> lambda_n = 0

        scale = W_full.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-6)
        eye = th.eye(n, dtype=th.float64)[None]

        # latent problem exactly as the code builds it: W = sym(S W_full S)
        W = th.bmm(S.unsqueeze(-1) * W_full, th.diag_embed(S))
        W = 0.5 * (W + W.transpose(1, 2))
        H_latent = W + reg * scale[:, None, None] * eye
        c_latent = S * b
        y = solve_cone_qp(H_latent, c_latent, mu, iterations)

        # physical problem: Rtilde = S^-1 R S^-1
        inv_S = 1.0 / S
        Rtilde = th.diag_embed(reg * scale[:, None] * inv_S * inv_S)
        H_phys = W_full + Rtilde
        lam = solve_cone_qp(H_phys, b, mu, iterations)

        pred = S * y
        err = (lam - pred).abs().max()
        denom = lam.abs().max().clamp_min(1e-12)
        rel = float(err / denom)
        # active-set bookkeeping so we can prove the cone was engaged
        normal = lam.reshape(1, K, 2)[..., 0]
        tangent = lam.reshape(1, K, 2)[..., 1]
        slack = (mu * normal - tangent.abs()).abs()
        active_bound = int(((normal > 1e-9) & (slack < 1e-7 * normal.clamp_min(1e-9))).sum())
        at_zero = int((normal.abs() <= 1e-9 * denom).sum())
        row = {
            "trial": t,
            "max_rel_error": rel,
            "max_abs_error": float(err),
            "latent_residual": float(cone_qp_residual(y, H_latent, c_latent, mu)[0]),
            "physical_residual": float(cone_qp_residual(lam, H_phys, b, mu)[0]),
            "contacts_on_cone_boundary": active_bound,
            "contacts_at_zero_normal": at_zero,
            "min_gate": float(gate.min()),
        }
        rows.append(row)
        if rel > worst["max_rel_error"]:
            worst = row
    boundary = sum(r["contacts_on_cone_boundary"] for r in rows)
    zeros = sum(r["contacts_at_zero_normal"] for r in rows)
    return {
        "trials": trials,
        "K": K,
        "reg": reg,
        "admm_iterations": iterations,
        "tol": tol,
        "worst_max_rel_error": worst["max_rel_error"],
        "worst_trial": worst,
        "max_latent_residual": max(r["latent_residual"] for r in rows),
        "max_physical_residual": max(r["physical_residual"] for r in rows),
        "total_contacts_on_cone_boundary": boundary,
        "total_contacts_at_zero_normal": zeros,
        "cone_was_active": bool(boundary > 0 and zeros > 0),
        "passed": bool(worst["max_rel_error"] < tol and boundary > 0 and zeros > 0),
        "rows": rows,
    }
