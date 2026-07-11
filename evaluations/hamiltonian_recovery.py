# evaluations/hamiltonian_recovery.py
"""Audit a trained structured port-Hamiltonian model against the simulator
(cheetah, or any raw-state hinge/slide domain via ``--raw_state_obs``).

The audit separates three kinds of accuracy, and a model should not be called
accurate on the strength of one alone:

  * predictive accuracy   — does the model predict accelerations and open-loop
                            rollouts (``predictive_report``)?
  * physical recovery     — does it recover the true mass, potential, Coriolis,
                            damping, actuator, and contact terms
                            (``recovery_report``)?
  * control-relevant accuracy — does model error corrupt the generator and
                            quadrature targets CT-SAC consumes
                            (``generator_report``, ``quadrature_report``)?

Physical recovery fixes two gauge freedoms first:

  * global scale: M -> cM, V -> cV, D -> cD, G_a -> cG_a leaves the flow
    invariant (the Coriolis force scales with M automatically). One scalar c*
    is fit on the mass matrices and applied to every learned term — four
    objects share a single gauge parameter, which is itself part of the test.
    No per-term scales are fit: that would hide inconsistent attribution.
  * potential offset: V and V + const generate the same flow.

After c* is fixed, only the potential (and total energy) retain an offset
freedom. Affine-fit R^2 values are therefore reported as *shape* diagnostics
(``*_shape_R2``); the strict recovery numbers are the scale-locked errors
(``*_locked_nrmse``, ``damping_rel_err``), and every force term carries a
magnitude-sensitive NRMSE beside its correlation:

    NRMSE(f) = sqrt(sum_i |f_hat_i - f_i|^2) / (sqrt(sum_i |f_i|^2) + eps).

The mass dM/dz probe is an architectural/leakage check, not a recovery
metric: once root height is excluded from the mass input (contact port
active), a zero derivative is guaranteed by construction
(``mass_dMdz_structural``) and says nothing about the rest of M.

Ground truth comes from the same physics the oracle drift uses: mj_fullM for
M(q); the MuJoCo energy flag for potential/kinetic energy; qfrc_bias /
qfrc_passive at zero velocity for the gravity+spring torque; qfrc_bias(q,v) -
qfrc_bias(q,0) for the Coriolis force; dof_damping for the true damping
diagonal; unit-control qfrc_actuator columns for G_a; and qfrc_constraint for
the generalized contact force (caveat: joint-limit forces are included too).

Evaluation distributions: the audit is not limited to each checkpoint's own
visited states. ``main`` evaluates on a fixed broad-exploration reference set
(OU noise, fixed seed) and, when checkpoints are given, on the current
policy's distribution and a frozen best-policy distribution; the generator and
quadrature errors are additionally measured under policy-sampled candidate
actions crossed with the visited states.

Usage (offline fit, OU exploration data):
    python -m evaluations.hamiltonian_recovery --fit_steps 8000 --fit_horizon 4

Usage (model trained by an RL run, saved by the CTSAC checkpoint sidecar):
    python -m evaluations.hamiltonian_recovery \
        --dynamics_path saved_models/.../best_model.dynamics.pth --contact_force 4 \
        [--checkpoint saved_models/.../best_model.pth --mode mbq_structured_quad_cforce_roll] \
        [--best_checkpoint saved_models/.../peak_model.pth]

``--checkpoint`` rolls the saved policy to collect the evaluation states (the
on-policy distribution) and supplies the V-head for the control-relevant
metrics; ``--best_checkpoint`` adds a frozen best-policy distribution.
"""

from __future__ import annotations

import argparse
import json

import mujoco
import numpy as np
import torch as th
from torch.func import jacfwd, vmap

from common.buffers import ReplayBuffer
from environment.dmc import DMCContinuousEnv
from models.port_hamiltonian import DOFLayout, PortHamiltonianModel, integrate_drift
from models.noise import OrnsteinUhlenbeckActionNoise


# --------------------------- data collection ---------------------------


def collect(env, n, policy=None, ou_sigma=0.4, seed=0):
    """Roll the env for n steps (saved policy if given, else OU exploration).
    Returns float32 arrays: obs, actions, next_obs, dt, dones."""
    env.action_space.seed(seed)
    ad = int(np.prod(env.action_space.shape))
    ou = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(ad), sigma=ou_sigma * np.ones(ad), theta=0.15, dt=0.01
    )
    O, A, NO, DT, DN = [], [], [], [], []
    obs, _ = env.reset()
    for _ in range(n):
        if policy is not None:
            with th.no_grad():
                a_t, _ = policy.act(th.as_tensor(obs, dtype=th.float32).unsqueeze(0))
            a = a_t.squeeze(0).numpy()
        else:
            a = np.clip(ou(), env.action_space.low, env.action_space.high)
        o, t, _, r, no, nt, term, trunc, _ = env.step_dt(a)
        O.append(o); A.append(a); NO.append(no); DT.append(nt - t)
        DN.append(1.0 if (term or trunc) else 0.0)
        if term or trunc:
            obs, _ = env.reset(); ou.reset()
        else:
            obs = no
    to = lambda x: np.asarray(x, dtype=np.float32)
    return to(O), to(A), to(NO), to(DT), to(DN)


# --------------------------- ground truth (MuJoCo) ---------------------------


def _set_state(data, nq, pos_width, obs_row):
    qpos = np.zeros(nq)
    qpos[nq - pos_width:] = obs_row[:pos_width]
    data.qpos[:] = qpos
    data.qvel[:] = obs_row[pos_width:]


def ground_truth(env: DMCContinuousEnv, obs: np.ndarray):
    """Extract the true mechanical terms at each observed state. Two observation
    maps are supported: raw state (obs = [qpos (nq); qvel (nv)], the env's
    ``raw_state_obs`` option) and the cheetah task observation
    (obs = [qpos[1:] (nq-1); qvel (nv)], root x set to 0 — every extracted
    quantity is translation-invariant). The live physics state is
    snapshotted/restored.

    Returns a dict of numpy arrays:
      M (B,nv,nv), e_pot (B,), e_kin (B,), g_pot (B,nv) gravity+spring torque,
      coriolis (B,nv) = C(q,v)v, dof_damping (nv,), G (nv,nu),
      qfrc_contact (B,nv) generalized constraint force (contacts + joint
      limits), contact_flag (B,) any active contact, contact_geom_act (B,Gc)
      per-geom contact activity over the geoms seen in contact,
      contact_geom_ids (Gc,).
    """
    raw = bool(getattr(env, "raw_state_obs", False))
    assert raw or env.domain_name == "cheetah", (
        "ground_truth supports raw_state_obs envs or the cheetah domain"
    )
    physics = env._env.physics
    model, data = physics.model, physics.data
    nq, nv, nu = int(model.nq), int(model.nv), int(model.nu)
    pos_width = nq if raw else nq - 1
    obs = np.asarray(obs, dtype=np.float64).reshape(-1, pos_width + nv)
    B = obs.shape[0]

    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY

    saved = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), float(data.time))
    M = np.zeros((B, nv, nv))
    e_pot = np.zeros(B); e_kin = np.zeros(B)
    g_pot = np.zeros((B, nv)); coriolis = np.zeros((B, nv))
    qfrc_contact = np.zeros((B, nv))
    contact_flag = np.zeros(B, dtype=bool)
    geom_hits: list[dict] = []
    G = np.zeros((nv, nu))
    try:
        # Actuator port: qfrc_actuator response to unit control, one column per
        # actuator (exact for the cheetah's linear torque motors).
        data.qpos[:] = saved[0]; data.qvel[:] = 0.0
        for j in range(nu):
            data.ctrl[:] = 0.0; data.ctrl[j] = 1.0
            physics.forward()
            G[:, j] = data.qfrc_actuator[:nv]
        data.ctrl[:] = 0.0

        for i in range(B):
            # with velocity: energy, mass matrix, full bias force, constraints
            _set_state(data, nq, pos_width, obs[i])
            physics.forward()
            mujoco.mj_fullM(model.ptr, M[i], data.qM)
            e_pot[i], e_kin[i] = data.energy[0], data.energy[1]
            bias_v = data.qfrc_bias[:nv].copy()
            qfrc_contact[i] = data.qfrc_constraint[:nv]
            ncon = int(data.ncon)
            contact_flag[i] = ncon > 0
            hits: dict = {}
            for j in range(ncon):
                con = data.contact[j]
                for gid in (int(con.geom1), int(con.geom2)):
                    hits[gid] = True
            geom_hits.append(hits)
            # at zero velocity: qfrc_bias = gravity, qfrc_passive = spring
            data.qvel[:] = 0.0
            physics.forward()
            g_pot[i] = data.qfrc_bias[:nv] - data.qfrc_passive[:nv]
            coriolis[i] = bias_v - data.qfrc_bias[:nv]
    finally:
        data.qpos[:] = saved[0]; data.qvel[:] = saved[1]
        data.ctrl[:] = saved[2]; data.time = saved[3]
        physics.forward()

    # per-geom activity matrix over the geoms ever seen in contact, most-active
    # first, skipping the most common geom (the shared floor/ground)
    counts: dict = {}
    for hits in geom_hits:
        for gid in hits:
            counts[gid] = counts.get(gid, 0) + 1
    geom_ids = sorted(counts, key=lambda g: -counts[g])
    if len(geom_ids) > 1:
        geom_ids = geom_ids[1:]  # drop the floor (touches in every contact)
    geom_act = np.zeros((B, len(geom_ids)))
    for i, hits in enumerate(geom_hits):
        for k, gid in enumerate(geom_ids):
            geom_act[i, k] = 1.0 if gid in hits else 0.0

    return dict(
        M=M, e_pot=e_pot, e_kin=e_kin, g_pot=g_pot, coriolis=coriolis,
        dof_damping=np.asarray(model.dof_damping[:nv]).copy(), G=G,
        qfrc_contact=qfrc_contact, contact_flag=contact_flag,
        contact_geom_act=geom_act,
        contact_geom_ids=np.asarray(geom_ids, dtype=np.int64),
    )


def mujoco_transition(env: DMCContinuousEnv, obs: np.ndarray, actions: np.ndarray,
                      dt: float) -> np.ndarray:
    """Integrate the true physics from each observed state under a constant
    (clipped) action for ``dt`` seconds; returns the next observations. The
    live physics state is snapshotted/restored."""
    raw = bool(getattr(env, "raw_state_obs", False))
    assert raw or env.domain_name == "cheetah", (
        "mujoco_transition supports raw_state_obs envs or the cheetah domain"
    )
    physics = env._env.physics
    model, data = physics.model, physics.data
    nq, nv = int(model.nq), int(model.nv)
    pos_width = nq if raw else nq - 1
    obs = np.asarray(obs, dtype=np.float64).reshape(-1, pos_width + nv)
    actions = np.asarray(actions, dtype=np.float64).reshape(obs.shape[0], -1)
    low = np.asarray(env.action_space.low, dtype=np.float64)
    high = np.asarray(env.action_space.high, dtype=np.float64)
    phys_dt = float(model.opt.timestep)
    nstep = max(1, int(round(dt / phys_dt)))

    saved = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), float(data.time))
    out = np.zeros_like(obs, dtype=np.float64)
    try:
        for i in range(obs.shape[0]):
            _set_state(data, nq, pos_width, obs[i])
            data.ctrl[:] = np.clip(actions[i], low, high)
            # deterministic, state-consistent constraint-solver warm start:
            # without it each sample inherits the previous sample's warmstart
            # and the integration becomes order-dependent
            physics.forward()
            data.qacc_warmstart[:] = data.qacc
            for _ in range(nstep):
                physics.step()
            out[i, :pos_width] = data.qpos[nq - pos_width:]
            out[i, pos_width:] = data.qvel[:nv]
    finally:
        data.qpos[:] = saved[0]; data.qvel[:] = saved[1]
        data.ctrl[:] = saved[2]; data.time = saved[3]
        physics.forward()
    return out.astype(np.float32)


# --------------------------- learned terms ---------------------------


def learned_terms(m: PortHamiltonianModel, obs: np.ndarray):
    """Evaluate the structured model's M, V, kinetic/total energy, potential
    gradient g_pot = +grad V (matching ``ground_truth``'s convention), Coriolis
    force C(q,qd)qd (left-hand-side convention, matching qfrc_bias), base
    damping diagonal and damping force, and G_a, all scattered onto the nv-wide
    config axis. With the contact port: the combined conservative gradient,
    the generalized contact force and its power, and per-contact diagnostics."""
    assert m.mode == "structured", "hamiltonian recovery applies to mode='structured'"
    lay = m.layout
    nv = lay.nv
    x = th.as_tensor(obs, dtype=th.float32, device=m.device)
    B = x.shape[0]
    pos = x[:, lay.pos_slice[0]:lay.pos_slice[1]]
    qd = x[:, lay.vel_slice[0]:lay.vel_slice[1]]

    with th.no_grad():
        M = vmap(m._mass)(pos)                                   # (B, nv, nv)
        V = vmap(m._potential)(pos)                              # (B,)
        dM_pos = vmap(jacfwd(m._mass))(pos)                      # (B, nv, nv, npos)
        gV_pos = vmap(jacfwd(m._potential))(pos)                 # (B, npos)
        dM = th.zeros(B, nv, nv, nv).index_copy(3, m._pos_to_cfg, dM_pos)
        gV = th.zeros(B, nv).index_copy(1, m._pos_to_cfg, gV_pos)
        # Coriolis force in the momentum balance M qdd + c(q,qd) = -grad V + ...
        c_hat = (
            th.einsum("nabk,nk,nb->na", dM, qd, qd)
            - 0.5 * th.einsum("nabk,na,nb->nk", dM, qd, qd)
        )
        e_kin = 0.5 * th.einsum("na,nab,nb->n", qd, M, qd)
        if lay.act_to_cfg is None:
            G = m.G_a.weight.detach().clone()                     # (nv, na)
        else:
            G = th.zeros(nv, m.action_dim).index_add(
                0, m._act_to_cfg, m.G_a.weight.detach()
            )
        d_base = th.nn.functional.softplus(m._log_d).detach()
        # per-position-coordinate mean |dM/dq| (translation-invariance probe)
        dM_mag = dM_pos.abs().mean(dim=(0, 1, 2))                 # (npos,)

        out = dict(
            M=M.numpy(), V=V.numpy(), e_kin=e_kin.numpy(), g_pot=gV.numpy(),
            coriolis=c_hat.numpy(), d_base=d_base.numpy(), G=G.numpy(),
            dM_mag=dM_mag.numpy(), qd=qd.numpy(),
            f_damp=(d_base.unsqueeze(0) * qd).numpy(),
            # architectural flag: dM/dq = 0 on m_invariant_pos coordinates is
            # guaranteed by construction when the mass input excludes them
            m_input_excluded=bool(getattr(m, "_mass_in_idx", None) is not None),
        )

        # Contact port: the gap springs k_i phi(g_i) grad g_i are themselves a
        # conservative field, and on always-in-contact data they are nearly
        # degenerate with grad V (gravity migrates into the port). Report the
        # combined conservative gradient grad V - sum_i k_i phi(g_i) J_n,i (a
        # force is minus a gradient) so the potential comparison sees the whole
        # field, plus the full generalized contact force, its power, and
        # per-contact diagnostics of the split.
        if getattr(m, "contact_force", 0) > 0:
            g, gdot, v_t, lam, f_t, J_n, J_t = m._contact_parts(pos, qd)
            k_i, c_i, mu_i = th.nn.functional.softplus(m._contact_raw)  # each (K,)
            w = m._contact_gap_width
            phi = th.nn.functional.softplus(-g / w) * w
            F_spring = th.einsum("nkv,nk->nv", J_n, phi * k_i)    # (B, nv)
            F_n = th.einsum("nkv,nk->nv", J_n, lam)
            F_t = th.einsum("nkv,nk->nv", J_t, f_t)
            out["g_pot_combined"] = (gV - F_spring).numpy()
            out["contact_F"] = (F_n + F_t).numpy()                # (B, nv)
            out["contact_F_n"] = F_n.numpy()
            out["contact_F_t"] = F_t.numpy()
            out["contact_power"] = (lam * gdot + f_t * v_t).sum(1).numpy()
            out["contact_lam"] = lam.numpy()                      # (B, K)
            out["contact_gap"] = g.numpy()
            out["contact_in_frac"] = float((g < 0).float().mean())
            out["contact_in_frac_per"] = (g < 0).float().mean(0).numpy()
            out["contact_spring_ratio"] = float(
                F_spring.norm(dim=1).mean() / (gV.norm(dim=1).mean() + 1e-12)
            )
            out["contact_kcm"] = th.stack([k_i, c_i, mu_i]).numpy()  # (3, K)

    return out


# --------------------------- metric helpers ---------------------------


def pearson(a, b):
    a, b = np.ravel(a), np.ravel(b)
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).mean() / (a.std() * b.std() + 1e-12))


def affine_fit(x, y):
    """Least-squares y ~ a*x + b; returns (a, b, R^2)."""
    x, y = np.ravel(x), np.ravel(y)
    A = np.stack([x, np.ones_like(x)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1.0 - ((a * x + b - y) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)
    return float(a), float(b), float(r2)


def nrmse(f_hat, f_true):
    """sqrt(sum |f_hat - f|^2) / (sqrt(sum |f|^2) + eps): magnitude-sensitive,
    complements scale-free correlations."""
    f_hat, f_true = np.asarray(f_hat, np.float64), np.asarray(f_true, np.float64)
    return float(np.linalg.norm(f_hat - f_true) / (np.linalg.norm(f_true) + 1e-12))


def _per_sample_err(f_hat, f_true):
    d = np.asarray(f_hat, np.float64) - np.asarray(f_true, np.float64)
    return np.abs(d) if d.ndim == 1 else np.linalg.norm(d, axis=-1)


def _tail_p95(f_hat, f_true):
    """p95 of per-sample error norms, normalized by the truth's RMS norm."""
    err = _per_sample_err(f_hat, f_true)
    t = np.asarray(f_true, np.float64)
    rms = np.sqrt((np.abs(t) ** 2 if t.ndim == 1 else (t ** 2).sum(-1)).mean())
    return float(np.percentile(err, 95) / (rms + 1e-12))


def _strata(speed, contact_flag):
    """Speed-tercile and contact-phase masks for stratified reporting."""
    lo, hi = np.quantile(speed, [1 / 3, 2 / 3])
    strata = {
        "speed_low": speed <= lo, "speed_mid": (speed > lo) & (speed <= hi),
        "speed_high": speed > hi,
    }
    if contact_flag is not None:
        strata["flight"] = ~contact_flag
        strata["contact"] = contact_flag
    return strata


def _stratified_nrmse(f_hat, f_true, strata):
    out = {}
    for name, mask in strata.items():
        out[name] = nrmse(f_hat[mask], f_true[mask]) if mask.any() else float("nan")
    return out


def force_metrics(rep, name, f_hat, f_true, strata=None):
    """Correlation + NRMSE + tail error (and optional strata) for one force."""
    rep[f"{name}_corr"] = pearson(f_hat, f_true)
    rep[f"{name}_nrmse"] = nrmse(f_hat, f_true)
    rep[f"{name}_err_p95"] = _tail_p95(f_hat, f_true)
    if strata is not None:
        rep[f"{name}_nrmse_strata"] = _stratified_nrmse(f_hat, f_true, strata)


def gauge_scale(M_hat, M_true):
    """c* = argmin_c ||c*M_hat - M_true||_F^2 over the eval states."""
    num = float((M_hat * M_true).sum())
    den = float((M_hat * M_hat).sum())
    return num / (den + 1e-12)


def _match_activity(pred_act, true_act):
    """Greedy permutation-invariant matching of learned contact activity
    columns (B,K) to true per-geom activity columns (B,Gc) by Pearson
    correlation; returns the matched correlations, best first."""
    K, Gc = pred_act.shape[1], true_act.shape[1]
    if K == 0 or Gc == 0:
        return []
    corr = np.full((K, Gc), -np.inf)
    for i in range(K):
        for j in range(Gc):
            if pred_act[:, i].std() > 0 and true_act[:, j].std() > 0:
                corr[i, j] = pearson(pred_act[:, i], true_act[:, j])
    matched = []
    used_i, used_j = set(), set()
    for _ in range(min(K, Gc)):
        best = None
        for i in range(K):
            for j in range(Gc):
                if i in used_i or j in used_j or not np.isfinite(corr[i, j]):
                    continue
                if best is None or corr[i, j] > corr[best]:
                    best = (i, j)
        if best is None:
            break
        matched.append(round(float(corr[best]), 3))
        used_i.add(best[0]); used_j.add(best[1])
    return matched


def _edge_timing(pred, true, dones, max_lag=10):
    """Mean |offset| (in steps) between true and predicted contact edges of the
    same type, matched to the nearest within ``max_lag``; edges that cross an
    episode boundary are skipped. Returns (mean_offset, n_true_edges,
    matched_frac)."""
    pred = np.asarray(pred, bool); true = np.asarray(true, bool)
    valid = np.ones(len(pred) - 1, bool)
    if dones is not None:
        valid &= np.asarray(dones[:-1], np.float64) == 0.0
    offs = []
    n_edges = 0
    for kind in (True, False):  # touchdown (rise), liftoff (fall)
        t_edges = [i for i in range(len(true) - 1)
                   if valid[i] and true[i + 1] != true[i] and true[i + 1] == kind]
        p_edges = [i for i in range(len(pred) - 1)
                   if valid[i] and pred[i + 1] != pred[i] and pred[i + 1] == kind]
        n_edges += len(t_edges)
        for t in t_edges:
            if p_edges:
                d = min(abs(t - p) for p in p_edges)
                if d <= max_lag:
                    offs.append(d)
    mean_off = float(np.mean(offs)) if offs else float("nan")
    matched = float(len(offs) / n_edges) if n_edges else float("nan")
    return mean_off, n_edges, matched


# --------------------------- physical recovery ---------------------------


def recovery_report(truth: dict, learned: dict, actions=None, dt=None,
                    dones=None) -> dict:
    """Gauge-fix the learned terms against the truth and compute the physical
    recovery metrics. ``actions``/``dt``/``dones`` (the transitions behind the
    eval states) enable the actuator-force, inverse-mass-response, impulse, and
    contact-timing metrics. Returns a JSON-serializable dict.

    Shape vs strict recovery: ``*_shape_R2`` metrics grant an affine refit and
    diagnose shape only; the strict numbers are the scale-locked errors
    (``*_locked_nrmse``, ``damping_rel_err``, the force NRMSEs)."""
    c = gauge_scale(learned["M"], truth["M"])
    Mh, Mt = c * learned["M"], truth["M"]

    rep: dict = {"gauge_scale_c": c}

    # ---- mass matrix ----
    rel = np.linalg.norm(Mh - Mt, axis=(1, 2)) / (
        np.linalg.norm(Mt, axis=(1, 2)) + 1e-12
    )
    rep["mass_rel_frob_err"] = float(rel.mean())
    rep["mass_entry_corr"] = pearson(Mh, Mt)
    nv = Mt.shape[-1]
    di = np.arange(nv)
    rep["mass_diag_rel_err"] = float(
        np.linalg.norm(Mh[:, di, di] - Mt[:, di, di])
        / (np.linalg.norm(Mt[:, di, di]) + 1e-12)
    )
    iu = np.triu_indices(nv, k=1)
    rep["mass_offdiag_rel_err"] = float(
        np.linalg.norm(Mh[:, iu[0], iu[1]] - Mt[:, iu[0], iu[1]])
        / (np.linalg.norm(Mt[:, iu[0], iu[1]]) + 1e-12)
    )
    iu0 = np.triu_indices(nv, k=0)  # unique upper-triangular entries incl diag
    rep["mass_uppertri_corr"] = pearson(Mh[:, iu0[0], iu0[1]], Mt[:, iu0[0], iu0[1]])
    eh = np.linalg.eigvalsh(Mh); et = np.linalg.eigvalsh(Mt)
    rep["mass_eig_rel_err"] = float(np.abs(eh - et).sum() / (np.abs(et).sum() + 1e-12))
    rep["mass_cond_ratio"] = float(
        ((eh[:, -1] / np.maximum(eh[:, 0], 1e-12))
         / (et[:, -1] / np.maximum(et[:, 0], 1e-12))).mean()
    )
    # inverse-mass response to generalized forces representative of the data:
    # (c M_hat)^-1 tau vs M^-1 tau with tau = G_true a — directly connected to
    # acceleration accuracy.
    if actions is not None:
        tau = np.asarray(actions, np.float64) @ truth["G"].T           # (B, nv)
        resp_h = np.linalg.solve(Mh.astype(np.float64), tau[..., None])[..., 0]
        resp_t = np.linalg.solve(Mt.astype(np.float64), tau[..., None])[..., 0]
        rep["mass_inverse_response_nrmse"] = nrmse(resp_h, resp_t)

    # ---- architectural / leakage checks (not recovery metrics) ----
    # translation-invariance probe: the true M does not depend on the first
    # observed position (cheetah: root height). When the mass input excludes
    # it (contact port active) a zero is guaranteed by construction and is no
    # longer evidence about the remaining mass matrix.
    dM_mag = learned["dM_mag"]
    rep["mass_dMdz_ratio"] = float(dM_mag[0] / (dM_mag[1:].mean() + 1e-12))
    rep["mass_dMdz_structural"] = bool(learned.get("m_input_excluded", False))

    # ---- energies ----
    # potential: offset gauge is physical; the affine slope is a shape
    # diagnostic and its ratio to c* measures scale attribution.
    a, b, r2 = affine_fit(learned["V"], truth["e_pot"])
    rep["potential_shape_R2"] = r2
    rep["potential_slope_a"] = a
    rep["potential_slope_ratio"] = float(a / (c + 1e-12))
    res = c * learned["V"] - truth["e_pot"]
    res = res - res.mean()  # offset gauge only; scale locked to c*
    rep["potential_locked_nrmse"] = float(
        np.linalg.norm(res)
        / (np.linalg.norm(truth["e_pot"] - truth["e_pot"].mean()) + 1e-12)
    )
    ek_h = c * learned["e_kin"]
    rep["kinetic_R2"] = 1.0 - float(
        ((ek_h - truth["e_kin"]) ** 2).sum()
        / (((truth["e_kin"] - truth["e_kin"].mean()) ** 2).sum() + 1e-12)
    )
    rep["kinetic_nrmse"] = nrmse(ek_h, truth["e_kin"])  # no offset: T(qd=0)=0
    Hh = c * (learned["V"] + learned["e_kin"])
    Ht = truth["e_pot"] + truth["e_kin"]
    _, _, r2h = affine_fit(Hh, Ht)
    rep["total_H_shape_R2"] = r2h
    resH = Hh - Ht; resH = resH - resH.mean()
    rep["total_H_locked_nrmse"] = float(
        np.linalg.norm(resH) / (np.linalg.norm(Ht - Ht.mean()) + 1e-12)
    )

    # ---- forces: correlation + magnitude (NRMSE) + tail, stratified ----
    qd = learned.get("qd")
    strata = None
    if qd is not None:
        speed = np.linalg.norm(qd, axis=1)
        strata = _strata(speed, truth.get("contact_flag"))
    force_metrics(rep, "gradV_force", c * learned["g_pot"], truth["g_pot"])
    force_metrics(rep, "coriolis_force", c * learned["coriolis"],
                  truth["coriolis"], strata=strata)
    if qd is not None:
        # damping force d ∘ qd (coefficient-weighted velocity, both conventions)
        force_metrics(rep, "damping_force", c * learned["f_damp"],
                      truth["dof_damping"][None, :] * qd)
    if actions is not None:
        # actuator force over the dataset's actions
        act = np.asarray(actions, np.float64)
        force_metrics(rep, "actuator_force", act @ (c * learned["G"]).T,
                      act @ truth["G"].T)

    # ---- contact port ----
    # with the contact port, V alone is not the whole learned conservative
    # field: compare the combined gradient too, and record how much of the
    # field lives in the port springs (the gravity-migration diagnostic).
    # V and the gap-spring potentials are only identified in sum, so they are
    # evaluated jointly through gradV_combined.
    if "g_pot_combined" in learned:
        force_metrics(rep, "gradV_combined", c * learned["g_pot_combined"],
                      truth["g_pot"], strata=strata)
        rep["contact_in_frac"] = learned["contact_in_frac"]
        rep["contact_spring_ratio"] = learned["contact_spring_ratio"]
        # Port parameters. Contact forces share the global gauge, so k and the
        # compression damping are reported at scale c*; mu is a force ratio and
        # gauge-free. k also trades against the learned gap scale, so absolute
        # values are soft — the meaningful read is the trend across checkpoints
        # of one run (stiffness creep) and the per-contact activity split.
        k_i, c_i, mu_i = learned["contact_kcm"]
        rep["contact_k"] = [round(float(v) * c, 4) for v in k_i]
        rep["contact_c"] = [round(float(v) * c, 4) for v in c_i]
        rep["contact_mu"] = [round(float(v), 4) for v in mu_i]
        rep["contact_in_frac_per"] = [
            round(float(v), 3) for v in learned["contact_in_frac_per"]
        ]
        g = learned["contact_gap"]
        rep["contact_gap_mean"] = float(g.mean())
        rep["contact_gap_min"] = float(g.min())

        # contact force recovery against the generalized constraint force
        # (includes joint-limit forces — a caveat, not a bias, on cheetah gaits)
        if "qfrc_contact" in truth and np.linalg.norm(truth["qfrc_contact"]) > 0:
            force_metrics(rep, "contact_force", c * learned["contact_F"],
                          truth["qfrc_contact"], strata=strata)
            if qd is not None:
                p_true = (truth["qfrc_contact"] * qd).sum(-1)
                force_metrics(rep, "contact_power", c * learned["contact_power"],
                              p_true)
            if dt is not None:
                w = np.asarray(dt, np.float64)[:, None]
                imp_h = (c * learned["contact_F"] * w).sum(0)
                imp_t = (truth["qfrc_contact"] * w).sum(0)
                rep["contact_impulse_rel_err"] = nrmse(imp_h, imp_t)
        # contact-state detection and timing against the true contact flag
        if "contact_flag" in truth:
            pred = (learned["contact_gap"] < 0).any(axis=1)
            true = np.asarray(truth["contact_flag"], bool)
            tp = float((pred & true).sum())
            rep["contact_precision"] = tp / (float(pred.sum()) + 1e-12)
            rep["contact_recall"] = tp / (float(true.sum()) + 1e-12)
            off, n_edges, matched = _edge_timing(pred, true, dones)
            rep["contact_edge_offset_steps"] = off
            rep["contact_edge_count"] = n_edges
            rep["contact_edge_matched_frac"] = matched
        if "contact_geom_act" in truth and truth["contact_geom_act"].shape[1] > 0:
            rep["contact_match_corr"] = _match_activity(
                (learned["contact_gap"] < 0).astype(np.float64),
                truth["contact_geom_act"],
            )

    # ---- damping (PHAST identifiability axis) ----
    # evaluated directly at scale c* — no affine refit: after the shared gauge,
    # the damping diagonal retains no slope or offset freedom.
    dh, dt_ = c * learned["d_base"], truth["dof_damping"]
    rep["damping_learned"] = [round(float(v), 4) for v in dh]
    rep["damping_true"] = [round(float(v), 4) for v in dt_]
    rep["damping_rel_err"] = nrmse(dh, dt_)
    rep["damping_abs_err_per_dof"] = [round(float(v), 4) for v in np.abs(dh - dt_)]
    rep["damping_rel_err_per_dof"] = [
        round(float(np.abs(h - t) / (abs(t) + 1e-12)), 4) for h, t in zip(dh, dt_)
    ]
    rep["damping_locked_R2"] = 1.0 - float(
        ((dh - dt_) ** 2).sum() / (((dt_ - dt_.mean()) ** 2).sum() + 1e-12)
    )

    # ---- actuator port ----
    Gh, Gt = c * learned["G"], truth["G"]
    rep["G_rel_frob_err"] = float(np.linalg.norm(Gh - Gt) / (np.linalg.norm(Gt) + 1e-12))
    cos = [
        float((Gh[:, j] @ Gt[:, j]) /
              (np.linalg.norm(Gh[:, j]) * np.linalg.norm(Gt[:, j]) + 1e-12))
        for j in range(Gt.shape[1])
    ]
    rep["G_actuator_cosine"] = [round(v, 3) for v in cos]
    return rep


# --------------------------- predictive accuracy ---------------------------


def predictive_report(m: PortHamiltonianModel, obs, actions, next_obs, dt,
                      dones, horizons=(4, 8), max_step=None) -> dict:
    """One-step and open-loop prediction accuracy. The recorded next states
    ARE the true (MuJoCo) rollout under the recorded actions, so the open-loop
    comparison is a paired learned-vs-simulator rollout with identical actions.
    ``max_step`` is the internal Euler resolution (as in training)."""
    lay = m.layout
    vel0 = lay.vel_slice[0]
    obs = np.asarray(obs, np.float32); next_obs = np.asarray(next_obs, np.float32)
    dt = np.asarray(dt, np.float32); dones = np.asarray(dones, np.float32)
    actions = np.asarray(actions, np.float32)
    with th.no_grad():
        b = m.drift(obs, actions).cpu().numpy()
    realized = (next_obs - obs) / (dt[:, None] + 1e-12)
    rep = {
        "accel_corr": pearson(b[:, vel0:], realized[:, vel0:]),
        "accel_nrmse": nrmse(b[:, vel0:], realized[:, vel0:]),
        "drift_nrmse": nrmse(b, realized),
    }
    # open-loop: roll the model with the recorded actions/durations over
    # windows with no episode boundary; displacement-normalized error.
    for H in horizons:
        errs = []
        t = 0
        while t + H <= len(obs):
            if dones[t:t + H].any():
                t += 1
                continue
            x = obs[t][None].copy()
            disp = 0.0
            with th.no_grad():
                for k in range(H):
                    x = integrate_drift(m.drift, x, actions[t + k][None],
                                        float(dt[t + k]),
                                        max_step=max_step).cpu().numpy()
                    disp += float(np.linalg.norm(next_obs[t + k] - obs[t + k]))
            errs.append(float(np.linalg.norm(x[0] - next_obs[t + H - 1]) / (disp + 1e-12)))
            t += H
        rep[f"rollout_rel_err_H{H}"] = float(np.mean(errs)) if errs else float("nan")
    return rep


# --------------------------- control-relevant accuracy ---------------------------


def _value_and_grad(policy, obs):
    """V_psi(x) and grad V_psi(x) from the checkpoint's value head."""
    x = th.as_tensor(np.asarray(obs), dtype=th.float32).requires_grad_(True)
    V = policy.value(x)
    (gV,) = th.autograd.grad(V.sum(), x)
    return V.detach().numpy().ravel(), gV.numpy()


def _err_stats(e):
    e = np.asarray(e, np.float64)
    return {
        "rmse": float(np.sqrt((e ** 2).mean())),
        "bias": float(e.mean()),
        "p95": float(np.percentile(np.abs(e), 95)),
        "p99": float(np.percentile(np.abs(e), 99)),
    }


def generator_report(m: PortHamiltonianModel, env, policy, obs, actions,
                     contact_flag=None, actions_pi=None) -> dict:
    """Error of the generator projection CT-SAC consumes:
        e_gen(x,a) = (b_hat(x,a) - b_MJ(x,a)) . grad V_psi(x).
    The projection weighs drift components by the value gradient, so this is
    the direct measure of how learned-dynamics error enters the critic target.
    Stratified by speed and contact phase; ``actions_pi`` (policy-sampled
    candidate actions at the same states) adds the action-novelty read."""
    obs = np.asarray(obs, np.float32)
    _, gV = _value_and_grad(policy, obs)
    lay = m.layout
    qd = obs[:, lay.vel_slice[0]:lay.vel_slice[1]]
    speed = np.linalg.norm(qd, axis=1)
    strata = _strata(speed, np.asarray(contact_flag, bool)
                     if contact_flag is not None else None)

    def _one(a):
        with th.no_grad():
            b_hat = m.drift(obs, a).cpu().numpy()
        b_mj = np.asarray(env.dynamics_terms(obs, a), np.float64)
        proj_hat = (b_hat * gV).sum(-1)
        proj_mj = (b_mj * gV).sum(-1)
        e = proj_hat - proj_mj
        out = _err_stats(e)
        out["corr"] = pearson(proj_hat, proj_mj)
        out["nrmse"] = nrmse(proj_hat, proj_mj)
        out["rmse_strata"] = {
            k: (float(np.sqrt((e[msk] ** 2).mean())) if msk.any() else float("nan"))
            for k, msk in strata.items()
        }
        return out, e

    rep = {}
    rep["data_actions"], e_data = _one(np.asarray(actions, np.float32))
    if actions_pi is not None:
        actions_pi = np.asarray(actions_pi, np.float32)
        rep["policy_actions"], e_pi = _one(actions_pi)
        novelty = np.linalg.norm(actions_pi - np.asarray(actions, np.float32), axis=1)
        rep["err_vs_action_novelty_corr"] = pearson(np.abs(e_pi), novelty)
    return rep


def quadrature_report(m: PortHamiltonianModel, env, policy, obs, actions,
                      dt_default, max_step=None, contact_flag=None) -> dict:
    """Error of the sub-step quadrature label: from the same state and constant
    action, integrate the learned model over dt_default (same internal Euler
    resolution as training) and MuJoCo over the same interval, and compare the
    value increments the target reads:
        dV_model = V_psi(x_hat) - V_psi(x0)  vs  dV_true = V_psi(x_MJ) - V_psi(x0).
    This is the closest available test of the learned model's contribution to
    the actual CT-SAC label. Also reports the endpoint state error (predictive
    axis of the same paired rollout)."""
    obs = np.asarray(obs, np.float32)
    actions = np.asarray(actions, np.float32)
    with th.no_grad():
        x_hat = integrate_drift(m.drift, obs, actions, float(dt_default),
                                max_step=max_step).cpu().numpy()
    x_mj = mujoco_transition(env, obs, actions, float(dt_default))
    V0, _ = _value_and_grad(policy, obs)
    Vh, _ = _value_and_grad(policy, x_hat)
    Vt, _ = _value_and_grad(policy, x_mj)
    dV_model, dV_true = Vh - V0, Vt - V0
    e = dV_model - dV_true
    rep = _err_stats(e)
    rep["corr"] = pearson(dV_model, dV_true)
    rep["nrmse"] = nrmse(dV_model, dV_true)
    rep["sign_disagree_frac"] = float((np.sign(dV_model) != np.sign(dV_true)).mean())
    rep["endpoint_state_nrmse"] = nrmse(x_hat - obs, x_mj - obs)
    lay = m.layout
    qd = obs[:, lay.vel_slice[0]:lay.vel_slice[1]]
    strata = _strata(np.linalg.norm(qd, axis=1),
                     np.asarray(contact_flag, bool) if contact_flag is not None else None)
    rep["rmse_strata"] = {
        k: (float(np.sqrt((e[msk] ** 2).mean())) if msk.any() else float("nan"))
        for k, msk in strata.items()
    }
    return rep


# --------------------------- energy balance ---------------------------


def energy_balance_report(m: PortHamiltonianModel, obs, actions) -> dict:
    """Forced energy balance along the model's own drift. A bare "energy must
    decrease" test is invalid under control — actuator work can raise the
    mechanical energy — so the residual accounts for every port:

        dE/dt = qd.G_a a  -  qd.D qd  +  sum_i (lam_i gdot_i + f_t,i v_t,i),

    with E = V + T the mechanical energy; the contact sum contains the
    conservative spring exchange (storage, sign-indefinite) plus the
    compression and friction dissipation (<= 0). ``residual_nrmse`` checks the
    identity itself; ``passivity_violation_frac`` counts states whose energy
    rise is NOT explained by actuator work plus spring release."""
    assert m.mode == "structured"
    lay = m.layout
    x = th.as_tensor(np.asarray(obs), dtype=th.float32).requires_grad_(True)
    a = th.as_tensor(np.asarray(actions), dtype=th.float32)
    pos = x[:, lay.pos_slice[0]:lay.pos_slice[1]]
    qd = x[:, lay.vel_slice[0]:lay.vel_slice[1]]
    M = vmap(m._mass)(pos)
    E = vmap(m._potential)(pos) + 0.5 * th.einsum("na,nab,nb->n", qd, M, qd)
    (gE,) = th.autograd.grad(E.sum(), x)
    with th.no_grad():
        b = m.drift(x.detach(), a)
        Edot = (gE * b).sum(-1)
        pos_d, qd_d = pos.detach(), qd.detach()
        if lay.act_to_cfg is None:
            Ga = m.G_a(a)
        else:
            Ga = th.zeros(x.shape[0], lay.nv).index_add(1, m._act_to_cfg, m.G_a(a))
        p_act = (qd_d * Ga).sum(-1)
        d = th.nn.functional.softplus(m._log_d)
        p_damp = -(qd_d * d * qd_d).sum(-1)
        p_contact = th.zeros_like(p_act)
        p_spring = th.zeros_like(p_act)
        if getattr(m, "contact_force", 0) > 0:
            g, gdot, v_t, lam, f_t, _, _ = m._contact_parts(pos_d, qd_d)
            k_i, _, _ = th.nn.functional.softplus(m._contact_raw)
            w = m._contact_gap_width
            phi = th.nn.functional.softplus(-g / w) * w
            p_contact = (lam * gdot + f_t * v_t).sum(-1)
            p_spring = (k_i * phi * gdot).sum(-1)
        rhs = p_act + p_damp + p_contact
        resid = (Edot - rhs).numpy()
        scale = float(np.abs(Edot.numpy()).mean() + np.abs(rhs.numpy()).mean()) + 1e-12
        # unexplained energy rise: dE/dt beyond actuator input + spring release
        unexplained = (Edot - p_act - p_spring).numpy()
    return {
        "residual_nrmse": float(np.linalg.norm(resid) / (np.linalg.norm(rhs.numpy()) + 1e-12)),
        "passivity_violation_frac": float((unexplained > 1e-3 * scale).mean()),
        "power_actuator_mean": float(p_act.mean()),
        "power_damping_mean": float(p_damp.mean()),
        "power_contact_mean": float(p_contact.mean()),
        "power_spring_mean": float(p_spring.mean()),
        "dE_dt_mean": float(Edot.mean()),
    }


# --------------------------- sanity checks ---------------------------


def sanity_check_truth(truth: dict, obs: np.ndarray, pos_width: int,
                       check_root_invariance: bool = True):
    """Internal consistency of the extraction: M SPD, MuJoCo's kinetic energy
    equals 1/2 v^T M v, and (cheetah) the gravity+spring torque has no root-x
    component. ``pos_width`` is the observation's position-block width."""
    eig = np.linalg.eigvalsh(truth["M"])
    assert eig.min() > 0, "true mass matrix not SPD"
    v = obs[:, pos_width:]
    ek = 0.5 * np.einsum("na,nab,nb->n", v, truth["M"], v)
    err = np.abs(ek - truth["e_kin"]).max() / (np.abs(truth["e_kin"]).max() + 1e-12)
    assert err < 1e-6, f"kinetic energy mismatch ({err:.2e}): extraction inconsistent"
    if check_root_invariance:
        assert np.abs(truth["g_pot"][:, 0]).max() < 1e-9, "gravity torque has root-x part"


# --------------------------- offline fit ---------------------------


def fit_model(env, O, A, NO, DT, DN, steps, horizon, batch=128, lr=1e-3,
              contact_force=0, hidden=(128, 128), seed=1, log_every=1000,
              dof_layout=None, integration_step=None):
    th.manual_seed(seed)
    od = int(env.observation_space.shape[0]); ad = int(env.action_space.shape[0])
    m = PortHamiltonianModel(od, ad, mode="structured",
                             structured_hidden=hidden,
                             contact_force=contact_force,
                             dof_layout=dof_layout)
    buf = ReplayBuffer(len(O), env.observation_space, env.action_space,
                       device="cpu", n_envs=1)
    for i in range(len(O)):
        buf.add(O[i:i+1], A[i:i+1], np.zeros(1, np.float32), DN[i:i+1],
                NO[i:i+1], np.zeros(1, np.float32), DT[i:i+1])
    opt = th.optim.Adam(m.parameters(), lr=lr)
    if integration_step is None:
        integration_step = getattr(env, "physics_dt", None)
    for s in range(steps):
        if horizon > 1:
            seq = buf.sample_sequences(batch, horizon)
            loss = m.fit_step_rollout(seq.observations, seq.actions,
                                      seq.next_observations, seq.dt, seq.mask,
                                      opt, max_step=integration_step)
        else:
            bt = buf.sample(batch)
            loss = m.fit_step(bt.observations, bt.actions, bt.next_observations,
                              bt.dt, opt, max_step=integration_step)
        if log_every and (s + 1) % log_every == 0:
            print(f"[fit] step {s+1}/{steps}: loss {loss:.5f}", flush=True)
    return m


# --------------------------- report assembly ---------------------------


def _jsonable(v):
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _headline(axes: dict) -> dict:
    """The cross-distribution summary row: one number per axis."""
    phys, pred = axes.get("physical_recovery", {}), axes.get("predictive", {})
    gen, quad = axes.get("generator", {}), axes.get("quadrature", {})
    row = {
        "accel_corr": pred.get("accel_corr"),
        "accel_nrmse": pred.get("accel_nrmse"),
        "mass_rel_frob_err": phys.get("mass_rel_frob_err"),
        "coriolis_corr": phys.get("coriolis_force_corr"),
        "coriolis_nrmse": phys.get("coriolis_force_nrmse"),
        "gradV_combined_corr": phys.get("gradV_combined_corr",
                                        phys.get("gradV_force_corr")),
    }
    if gen:
        row["gen_rmse"] = gen.get("data_actions", {}).get("rmse")
    if quad:
        row["quad_corr"] = quad.get("corr")
        row["quad_sign_disagree"] = quad.get("sign_disagree_frac")
    return row


def evaluate_dataset(m, env, data, n_eval, policy=None, max_step=None,
                     with_energy=False) -> dict:
    """All three accuracy axes on one dataset's held-out tail."""
    O, A, NO, DT, DN = data
    obs, act = O[-n_eval:], A[-n_eval:]
    nxt, dt, dn = NO[-n_eval:], DT[-n_eval:], DN[-n_eval:]

    truth = ground_truth(env, obs)
    learned = learned_terms(m, obs)
    axes = {
        "physical_recovery": recovery_report(truth, learned, actions=act,
                                             dt=dt, dones=dn),
        "predictive": predictive_report(m, obs, act, nxt, dt, dn,
                                        max_step=max_step),
    }
    if policy is not None and getattr(policy, "has_v_head", False):
        with th.no_grad():
            a_t, _ = policy.act(th.as_tensor(obs, dtype=th.float32))
            a_pi = a_t.numpy()
        axes["generator"] = generator_report(
            m, env, policy, obs, act,
            contact_flag=truth["contact_flag"], actions_pi=a_pi)
        axes["quadrature"] = quadrature_report(
            m, env, policy, obs, a_pi,
            dt_default=float(env.dt_default), max_step=max_step,
            contact_flag=truth["contact_flag"])
    if with_energy:
        axes["energy_balance"] = energy_balance_report(m, obs, act)
    axes["headline"] = _headline(axes)
    return axes


# --------------------------- main ---------------------------


def _print_primary(axes: dict):
    rep = axes["physical_recovery"]
    pred = axes["predictive"]
    print("\n== Predictive accuracy ==")
    print(f"accel corr / nrmse            : {pred['accel_corr']:.3f} / {pred['accel_nrmse']:.3f}")
    for k, v in pred.items():
        if k.startswith("rollout_rel_err"):
            print(f"{k:<30s}: {v:.3f}")

    print("\n== Physical recovery (gauge-fixed) ==")
    print(f"gauge scale c*                : {rep['gauge_scale_c']:.4f}")
    print(f"mass  rel Frobenius err       : {rep['mass_rel_frob_err']:.3f}"
          f"  (diag {rep['mass_diag_rel_err']:.3f},"
          f" offdiag {rep['mass_offdiag_rel_err']:.3f})")
    print(f"mass  uppertri corr           : {rep['mass_uppertri_corr']:.3f}"
          f"  (eig err {rep['mass_eig_rel_err']:.3f},"
          f" cond ratio {rep['mass_cond_ratio']:.2f})")
    if "mass_inverse_response_nrmse" in rep:
        print(f"mass  inverse-response nrmse  : {rep['mass_inverse_response_nrmse']:.3f}")
    tag = " [structural zero]" if rep["mass_dMdz_structural"] else ""
    print(f"mass  |dM/dz|/|dM/dq| (arch.) : {rep['mass_dMdz_ratio']:.3f}  (true: 0){tag}")
    print(f"potential shape R^2           : {rep['potential_shape_R2']:.3f}"
          f"  (slope ratio {rep['potential_slope_ratio']:.2f},"
          f" locked nrmse {rep['potential_locked_nrmse']:.3f})")
    print(f"kinetic R^2 / nrmse (scale c*): {rep['kinetic_R2']:.3f} / {rep['kinetic_nrmse']:.3f}")
    print(f"total H shape R^2 / locked    : {rep['total_H_shape_R2']:.3f}"
          f" / {rep['total_H_locked_nrmse']:.3f}")
    for name in ("gradV_force", "gradV_combined", "coriolis_force",
                 "damping_force", "actuator_force", "contact_force",
                 "contact_power"):
        if f"{name}_corr" in rep:
            print(f"{name:<22s} corr/nrmse : {rep[f'{name}_corr']:.3f}"
                  f" / {rep[f'{name}_nrmse']:.3f}  (p95 {rep[f'{name}_err_p95']:.3f})")
    if "contact_in_frac" in rep:
        print(f"  in-contact frac / spring    : {rep['contact_in_frac']:.2f}"
              f" / {rep['contact_spring_ratio']:.2f}")
        print(f"  port c*.k                   : {rep['contact_k']}")
        print(f"  port c*.c / mu              : {rep['contact_c']} / {rep['contact_mu']}")
        if "contact_precision" in rep:
            print(f"  contact precision / recall  : {rep['contact_precision']:.2f}"
                  f" / {rep['contact_recall']:.2f}"
                  f"  (edge offset {rep['contact_edge_offset_steps']}"
                  f" steps over {rep['contact_edge_count']} edges)")
        if "contact_match_corr" in rep:
            print(f"  matched contact activity    : {rep['contact_match_corr']}")
        if "contact_impulse_rel_err" in rep:
            print(f"  impulse rel err             : {rep['contact_impulse_rel_err']:.3f}")
    print(f"damping rel err / locked R^2  : {rep['damping_rel_err']:.3f}"
          f" / {rep['damping_locked_R2']:.3f}")
    print(f"  learned c*.softplus(log_d)  : {rep['damping_learned']}")
    print(f"  true dof_damping            : {rep['damping_true']}")
    print(f"G_a rel Frobenius err         : {rep['G_rel_frob_err']:.3f}")
    print(f"G_a per-actuator cosine       : {rep['G_actuator_cosine']}")

    if "generator" in axes:
        gen = axes["generator"]
        print("\n== Control-relevant accuracy ==")
        for which in ("data_actions", "policy_actions"):
            if which in gen:
                g = gen[which]
                print(f"generator e_gen ({which:<14s}): rmse {g['rmse']:.4f}"
                      f"  bias {g['bias']:+.4f}  corr {g['corr']:.3f}"
                      f"  p95 {g['p95']:.4f}  p99 {g['p99']:.4f}")
                print(f"  rmse by stratum             : "
                      + "  ".join(f"{k} {v:.4f}" for k, v in g["rmse_strata"].items()))
        if "err_vs_action_novelty_corr" in gen:
            print(f"  |e_gen| vs action novelty   : {gen['err_vs_action_novelty_corr']:.3f}")
    if "quadrature" in axes:
        q = axes["quadrature"]
        print(f"quadrature dV: rmse {q['rmse']:.4f}  bias {q['bias']:+.4f}"
              f"  corr {q['corr']:.3f}  sign-disagree {q['sign_disagree_frac']:.3f}"
              f"  p95 {q['p95']:.4f}")
        print(f"  endpoint state nrmse        : {q['endpoint_state_nrmse']:.3f}")
    if "energy_balance" in axes:
        e = axes["energy_balance"]
        print(f"\nenergy balance: residual nrmse {e['residual_nrmse']:.4f},"
              f" unexplained-rise frac {e['passivity_violation_frac']:.3f}")
        print(f"  mean powers  act {e['power_actuator_mean']:+.3f}"
              f"  damp {e['power_damping_mean']:+.3f}"
              f"  contact {e['power_contact_mean']:+.3f}"
              f"  spring {e['power_spring_mean']:+.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--env_id", default="cheetah-run")
    p.add_argument("--n", type=int, default=4000, help="transitions to collect")
    p.add_argument("--n_eval", type=int, default=800,
                   help="held-out tail states for the comparison")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ou_sigma", type=float, default=0.4)
    p.add_argument("--dynamics_path", default=None,
                   help="state_dict of a trained structured model "
                        "(the CTSAC checkpoint sidecar *.dynamics.pth); "
                        "otherwise a model is fit offline on the collected data")
    p.add_argument("--fit_steps", type=int, default=8000)
    p.add_argument("--fit_horizon", type=int, default=4)
    p.add_argument("--contact_force", type=int, default=0,
                   help="number of learned contact points for the explicit "
                        "contact-force port (must match --dynamics_path if given)")
    p.add_argument("--checkpoint", default=None,
                   help="RL checkpoint (*.pth): its policy collects the "
                        "on-policy data and its V-head enables the "
                        "control-relevant metrics")
    p.add_argument("--best_checkpoint", default=None,
                   help="frozen best-policy checkpoint: adds a best-policy "
                        "evaluation distribution (forgetting probe)")
    p.add_argument("--mode", default="mbq_structured_quad_contact_roll",
                   help="CSV mode row used to size the policy nets for --checkpoint")
    p.add_argument("--reference_seed", type=int, default=123,
                   help="fixed seed of the broad-exploration reference set")
    p.add_argument("--no_reference", action="store_true",
                   help="skip the fixed OU reference distribution")
    p.add_argument("--raw_state_obs", action="store_true",
                   help="raw-state observations [qpos; qvel] (hinge/slide "
                        "domains, e.g. cartpole/acrobot; must match the run)")
    p.add_argument("--out", default=None, help="write the report as JSON here")
    args = p.parse_args()

    th.manual_seed(args.seed); np.random.seed(args.seed)
    domain, task = args.env_id.split("-", 1)
    env = DMCContinuousEnv(domain, task, time_sampling="uniform", dt=0.01,
                           physics_dt=0.002, episode_duration=20.0, seed=args.seed,
                           raw_state_obs=args.raw_state_obs)
    layout = (DOFLayout.raw_state(nv=int(env.observation_space.shape[0]) // 2)
              if args.raw_state_obs else None)
    max_step = getattr(env, "physics_dt", None)

    def _load_policy(path):
        from common.utils import load_ct_hyperparams_from_table
        from models.actor_q_critic import ActorQCriticModel
        _, _, model_kwargs, _, _ = load_ct_hyperparams_from_table(
            "ct_sac", args.env_id, args.mode)
        pol = ActorQCriticModel(
            observation_space=env.observation_space,
            action_space=env.action_space, device="cpu", **model_kwargs)
        pol.load_state(path)
        return pol

    policy = _load_policy(args.checkpoint) if args.checkpoint else None
    best_policy = _load_policy(args.best_checkpoint) if args.best_checkpoint else None
    if policy is not None:
        print(f"collecting with the policy from {args.checkpoint}")

    # evaluation distributions: fixed broad-exploration reference, current
    # policy, frozen best policy (the audit is not limited to the visited set)
    datasets: dict = {}
    if not args.no_reference or policy is None:
        datasets["reference"] = collect(env, args.n, policy=None,
                                        ou_sigma=args.ou_sigma,
                                        seed=args.reference_seed)
    if policy is not None:
        datasets["policy"] = collect(env, args.n, policy=policy, seed=args.seed)
    if best_policy is not None:
        datasets["best_policy"] = collect(env, args.n, policy=best_policy,
                                          seed=args.seed + 1)
    primary = "policy" if "policy" in datasets else "reference"

    O, A, NO, DT, DN = datasets[primary]
    if args.dynamics_path:
        od = int(env.observation_space.shape[0])
        ad = int(env.action_space.shape[0])
        m = PortHamiltonianModel(od, ad, mode="structured",
                                 structured_hidden=(128, 128),
                                 contact_force=args.contact_force,
                                 dof_layout=layout)
        m.load_state_dict(th.load(args.dynamics_path, map_location="cpu"))
        print(f"loaded dynamics model from {args.dynamics_path}")
    else:
        m = fit_model(env, O, A, NO, DT, DN, args.fit_steps, args.fit_horizon,
                      contact_force=args.contact_force, seed=args.seed + 1,
                      dof_layout=layout)

    # sanity of the truth extraction on the primary tail
    eval_obs = O[-args.n_eval:]
    truth = ground_truth(env, eval_obs)
    nq = int(env._env.physics.model.nq)
    sanity_check_truth(truth, eval_obs.astype(np.float64),
                       pos_width=nq if args.raw_state_obs else nq - 1,
                       check_root_invariance=not args.raw_state_obs)

    value_policy = policy if (policy is not None and
                              getattr(policy, "has_v_head", False)) else None
    if policy is not None and value_policy is None:
        print("checkpoint has no V-head: control-relevant metrics skipped")

    report = {"primary": primary, "datasets": {}}
    for name, data in datasets.items():
        report["datasets"][name] = evaluate_dataset(
            m, env, data, args.n_eval, policy=value_policy,
            max_step=max_step, with_energy=(name == primary))

    _print_primary(report["datasets"][primary])

    if len(report["datasets"]) > 1:
        print("\n== Across evaluation distributions ==")
        keys = list(_headline(report["datasets"][primary]).keys())
        print(f"{'dataset':<14s}" + "".join(f"{k:>22s}" for k in keys))
        for name, axes in report["datasets"].items():
            row = axes["headline"]
            cells = "".join(
                f"{row.get(k):>22.3f}" if isinstance(row.get(k), float)
                else f"{'—':>22s}" for k in keys
            )
            print(f"{name:<14s}" + cells)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(_jsonable(report), f, indent=2)
        print(f"\nreport written to {args.out}")


if __name__ == "__main__":
    main()
