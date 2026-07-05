# evaluations/hamiltonian_recovery.py
"""Compare a trained structured port-Hamiltonian model against the real cheetah
Hamiltonian.

The structured model (models/port_hamiltonian.py, mode="structured") learns the
physical objects themselves — mass matrix M(q), potential V(q), damping D, and
actuator port G_a — in the same generalized coordinates MuJoCo integrates, so a
term-by-term comparison against the simulator is well-posed. Two gauge freedoms
must be fixed first:

  * global scale: M -> cM, V -> cV, D -> cD, G_a -> cG_a leaves the flow
    invariant (the Coriolis force scales with M automatically). One scalar c*
    is fit on the mass matrices and applied to every learned term — four
    objects share a single gauge parameter, which is itself part of the test.
  * potential offset: V and V + const generate the same flow, so V is compared
    through an affine fit.

Everything is evaluated on visited states only (the model is unconstrained off
the data manifold).

Ground truth comes from the same physics the oracle drift uses:
mj_fullM for M(q); the MuJoCo energy flag for potential/kinetic energy;
qfrc_bias/qfrc_passive at zero velocity for the gravity+spring torque;
qfrc_bias(q,v) - qfrc_bias(q,0) for the Coriolis force; dof_damping for the
true damping diagonal; and unit-control qfrc_actuator columns for G_a.

Usage (offline fit, OU exploration data):
    python -m evaluations.hamiltonian_recovery --fit_steps 8000 --fit_horizon 4

Usage (model trained by an RL run, saved by the CTSAC checkpoint sidecar):
    python -m evaluations.hamiltonian_recovery \
        --dynamics_path saved_models/.../best_model.dynamics.pth --contact_aware \
        [--checkpoint saved_models/.../best_model.pth --mode mbq_structured_quad_contact_roll]

``--checkpoint`` rolls the saved policy to collect the evaluation states (the
on-policy distribution); without it, OU exploration is used.
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
from models.port_hamiltonian import PortHamiltonianModel
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


def ground_truth(env: DMCContinuousEnv, obs: np.ndarray):
    """Extract the true mechanical terms at each observed state. Cheetah obs is
    [qpos[1:] (nq-1); qvel (nv)]; root x is set to 0 (every extracted quantity
    is translation-invariant). The live physics state is snapshotted/restored.

    Returns a dict of numpy arrays:
      M (B,nv,nv), e_pot (B,), e_kin (B,), g_pot (B,nv) gravity+spring torque,
      coriolis (B,nv) = C(q,v)v, dof_damping (nv,), G (nv,nu).
    """
    assert env.domain_name == "cheetah", "ground_truth supports the cheetah domain"
    physics = env._env.physics
    model, data = physics.model, physics.data
    nq, nv, nu = int(model.nq), int(model.nv), int(model.nu)
    obs = np.asarray(obs, dtype=np.float64).reshape(-1, (nq - 1) + nv)
    B = obs.shape[0]

    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY

    saved = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), float(data.time))
    M = np.zeros((B, nv, nv))
    e_pot = np.zeros(B); e_kin = np.zeros(B)
    g_pot = np.zeros((B, nv)); coriolis = np.zeros((B, nv))
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
            qpos = np.zeros(nq); qpos[1:] = obs[i, : nq - 1]
            qvel = obs[i, nq - 1:]
            # with velocity: energy, mass matrix, full bias force
            data.qpos[:] = qpos; data.qvel[:] = qvel
            physics.forward()
            mujoco.mj_fullM(model.ptr, M[i], data.qM)
            e_pot[i], e_kin[i] = data.energy[0], data.energy[1]
            bias_v = data.qfrc_bias[:nv].copy()
            # at zero velocity: qfrc_bias = gravity, qfrc_passive = spring
            data.qvel[:] = 0.0
            physics.forward()
            g_pot[i] = data.qfrc_bias[:nv] - data.qfrc_passive[:nv]
            coriolis[i] = bias_v - data.qfrc_bias[:nv]
    finally:
        data.qpos[:] = saved[0]; data.qvel[:] = saved[1]
        data.ctrl[:] = saved[2]; data.time = saved[3]
        physics.forward()

    return dict(
        M=M, e_pot=e_pot, e_kin=e_kin, g_pot=g_pot, coriolis=coriolis,
        dof_damping=np.asarray(model.dof_damping[:nv]).copy(), G=G,
    )


# --------------------------- learned terms ---------------------------


def learned_terms(m: PortHamiltonianModel, obs: np.ndarray):
    """Evaluate the structured model's M, V, kinetic/total energy, potential
    gradient g_pot = +grad V (matching ``ground_truth``'s convention), Coriolis
    force C(q,qd)qd (left-hand-side convention, matching qfrc_bias), base
    damping diagonal, and G_a, all scattered onto the nv-wide config axis."""
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
        # per-position-coordinate mean |dM/dq| (z-independence probe)
        dM_mag = dM_pos.abs().mean(dim=(0, 1, 2))                 # (npos,)

        out = dict(
            M=M.numpy(), V=V.numpy(), e_kin=e_kin.numpy(), g_pot=gV.numpy(),
            coriolis=c_hat.numpy(), d_base=d_base.numpy(), G=G.numpy(),
            dM_mag=dM_mag.numpy(),
        )

        # Contact port: the gap springs k_i phi(g_i) grad g_i are themselves a
        # conservative field, and on always-in-contact data they are nearly
        # degenerate with grad V (gravity migrates into the port). Report the
        # combined conservative gradient grad V - sum_i k_i phi(g_i) J_n,i (a
        # force is minus a gradient) so the potential comparison sees the whole
        # field, plus per-contact diagnostics of the split.
        if getattr(m, "contact_force", 0) > 0:
            g = m.gap_net(pos)                                    # (B, K)
            dg = vmap(jacfwd(m._gaps))(pos)                       # (B, K, npos)
            J_n = th.zeros(B, m.contact_force, nv).index_copy(
                2, m._pos_to_cfg, dg
            )
            k_i, _, _ = th.nn.functional.softplus(m._contact_raw)  # (K,)
            w = m._contact_gap_width
            phi = th.nn.functional.softplus(-g / w) * w
            F_spring = th.einsum("nkv,nk->nv", J_n, phi * k_i)    # (B, nv)
            out["g_pot_combined"] = (gV - F_spring).numpy()
            out["contact_gap"] = g.numpy()
            out["contact_in_frac"] = float((g < 0).float().mean())
            out["contact_spring_ratio"] = float(
                F_spring.norm(dim=1).mean() / (gV.norm(dim=1).mean() + 1e-12)
            )

    return out


# --------------------------- metrics ---------------------------


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


def gauge_scale(M_hat, M_true):
    """c* = argmin_c ||c*M_hat - M_true||_F^2 over the eval states."""
    num = float((M_hat * M_true).sum())
    den = float((M_hat * M_hat).sum())
    return num / (den + 1e-12)


def recovery_report(truth: dict, learned: dict) -> dict:
    """Gauge-fix the learned terms against the truth and compute the comparison
    metrics. Returns a flat dict (JSON-serializable)."""
    c = gauge_scale(learned["M"], truth["M"])
    Mh = c * learned["M"]

    rep: dict = {"gauge_scale_c": c}

    # mass matrix
    rel = np.linalg.norm(Mh - truth["M"], axis=(1, 2)) / (
        np.linalg.norm(truth["M"], axis=(1, 2)) + 1e-12
    )
    rep["mass_rel_frob_err"] = float(rel.mean())
    rep["mass_entry_corr"] = pearson(Mh, truth["M"])
    # z-independence probe: the true M does not depend on root height (obs
    # position coordinate 0); a faithful M(q) should have |dM/dz| << |dM/dq_j|.
    dM_mag = learned["dM_mag"]
    rep["mass_dMdz_ratio"] = float(dM_mag[0] / (dM_mag[1:].mean() + 1e-12))

    # potential (affine gauge) and energies
    a, b, r2 = affine_fit(learned["V"], truth["e_pot"])
    rep["potential_affine_R2"] = r2
    rep["potential_slope_a"] = a
    ek_h = c * learned["e_kin"]
    rep["kinetic_R2"] = 1.0 - float(
        ((ek_h - truth["e_kin"]) ** 2).sum()
        / (((truth["e_kin"] - truth["e_kin"].mean()) ** 2).sum() + 1e-12)
    )
    _, _, r2h = affine_fit(c * learned["V"] + c * learned["e_kin"],
                           truth["e_pot"] + truth["e_kin"])
    rep["total_H_affine_R2"] = r2h

    # forces
    rep["gradV_force_corr"] = pearson(c * learned["g_pot"], truth["g_pot"])
    rep["coriolis_force_corr"] = pearson(c * learned["coriolis"], truth["coriolis"])
    # with the contact port, V alone is not the whole learned conservative
    # field: compare the combined gradient too, and record how much of the
    # field lives in the port springs (the gravity-migration diagnostic).
    if "g_pot_combined" in learned:
        rep["gradV_combined_corr"] = pearson(
            c * learned["g_pot_combined"], truth["g_pot"]
        )
        rep["contact_in_frac"] = learned["contact_in_frac"]
        rep["contact_spring_ratio"] = learned["contact_spring_ratio"]

    # damping (PHAST identifiability axis): base diagonal vs dof_damping
    dh, dt_ = c * learned["d_base"], truth["dof_damping"]
    rep["damping_learned"] = [round(float(v), 4) for v in dh]
    rep["damping_true"] = [round(float(v), 4) for v in dt_]
    _, _, r2d = affine_fit(dh, dt_)
    rep["damping_affine_R2"] = r2d

    # actuator port
    Gh, Gt = c * learned["G"], truth["G"]
    rep["G_rel_frob_err"] = float(np.linalg.norm(Gh - Gt) / (np.linalg.norm(Gt) + 1e-12))
    cos = [
        float((Gh[:, j] @ Gt[:, j]) /
              (np.linalg.norm(Gh[:, j]) * np.linalg.norm(Gt[:, j]) + 1e-12))
        for j in range(Gt.shape[1])
    ]
    rep["G_actuator_cosine"] = [round(v, 3) for v in cos]
    return rep


def sanity_check_truth(truth: dict, obs: np.ndarray, nq: int):
    """Internal consistency of the extraction: M SPD, MuJoCo's kinetic energy
    equals 1/2 v^T M v, and the gravity+spring torque has no root-x component."""
    eig = np.linalg.eigvalsh(truth["M"])
    assert eig.min() > 0, "true mass matrix not SPD"
    v = obs[:, nq - 1:]
    ek = 0.5 * np.einsum("na,nab,nb->n", v, truth["M"], v)
    err = np.abs(ek - truth["e_kin"]).max() / (np.abs(truth["e_kin"]).max() + 1e-12)
    assert err < 1e-6, f"kinetic energy mismatch ({err:.2e}): extraction inconsistent"
    assert np.abs(truth["g_pot"][:, 0]).max() < 1e-9, "gravity torque has root-x part"


# --------------------------- offline fit ---------------------------


def fit_model(env, O, A, NO, DT, DN, steps, horizon, batch=128, lr=1e-3,
              contact_aware=True, contact_force=0, hidden=(128, 128), seed=1,
              log_every=1000):
    th.manual_seed(seed)
    od = int(env.observation_space.shape[0]); ad = int(env.action_space.shape[0])
    m = PortHamiltonianModel(od, ad, mode="structured",
                             structured_hidden=hidden, contact_aware=contact_aware,
                             contact_force=contact_force)
    buf = ReplayBuffer(len(O), env.observation_space, env.action_space,
                       device="cpu", n_envs=1)
    for i in range(len(O)):
        buf.add(O[i:i+1], A[i:i+1], np.zeros(1, np.float32), DN[i:i+1],
                NO[i:i+1], np.zeros(1, np.float32), DT[i:i+1])
    opt = th.optim.Adam(m.parameters(), lr=lr)
    for s in range(steps):
        if horizon > 1:
            seq = buf.sample_sequences(batch, horizon)
            loss = m.fit_step_rollout(seq.observations, seq.actions,
                                      seq.next_observations, seq.dt, seq.mask,
                                      opt, prev_obs=seq.prev_observations)
        else:
            bt = buf.sample(batch)
            loss = m.fit_step(bt.observations, bt.actions, bt.next_observations,
                              bt.dt, opt, prev_obs=bt.prev_observations)
        if log_every and (s + 1) % log_every == 0:
            print(f"[fit] step {s+1}/{steps}: loss {loss:.5f}", flush=True)
    return m


# --------------------------- main ---------------------------


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
    p.add_argument("--contact_aware", action="store_true",
                   help="build the model with the contact-gated damping "
                        "(must match --dynamics_path if given)")
    p.add_argument("--contact_force", type=int, default=0,
                   help="number of learned contact points for the explicit "
                        "contact-force port (must match --dynamics_path if given)")
    p.add_argument("--checkpoint", default=None,
                   help="RL checkpoint (*.pth) whose policy collects the data")
    p.add_argument("--mode", default="mbq_structured_quad_contact_roll",
                   help="CSV mode row used to size the policy nets for --checkpoint")
    p.add_argument("--out", default=None, help="write the report as JSON here")
    args = p.parse_args()

    th.manual_seed(args.seed); np.random.seed(args.seed)
    domain, task = args.env_id.split("-", 1)
    env = DMCContinuousEnv(domain, task, time_sampling="uniform", dt=0.01,
                           physics_dt=0.002, episode_duration=20.0, seed=args.seed)

    policy = None
    if args.checkpoint:
        from common.utils import load_ct_hyperparams_from_table
        from models.actor_q_critic import ActorQCriticModel
        _, _, model_kwargs, _, _ = load_ct_hyperparams_from_table(
            "ct_sac", args.env_id, args.mode)
        policy = ActorQCriticModel(
            observation_space=env.observation_space,
            action_space=env.action_space, device="cpu", **model_kwargs)
        policy.load_state(args.checkpoint)
        print(f"collecting with the policy from {args.checkpoint}")

    O, A, NO, DT, DN = collect(env, args.n, policy=policy,
                               ou_sigma=args.ou_sigma, seed=args.seed)

    if args.dynamics_path:
        od = int(env.observation_space.shape[0])
        ad = int(env.action_space.shape[0])
        m = PortHamiltonianModel(od, ad, mode="structured",
                                 structured_hidden=(128, 128),
                                 contact_aware=args.contact_aware,
                                 contact_force=args.contact_force)
        m.load_state_dict(th.load(args.dynamics_path, map_location="cpu"))
        print(f"loaded dynamics model from {args.dynamics_path}")
    else:
        m = fit_model(env, O, A, NO, DT, DN, args.fit_steps, args.fit_horizon,
                      contact_aware=args.contact_aware,
                      contact_force=args.contact_force, seed=args.seed + 1)

    eval_obs = O[-args.n_eval:]
    truth = ground_truth(env, eval_obs)
    sanity_check_truth(truth, eval_obs.astype(np.float64),
                       nq=int(env._env.physics.model.nq))
    learned = learned_terms(m, eval_obs)
    rep = recovery_report(truth, learned)

    print("\n== Hamiltonian recovery (gauge-fixed) ==")
    print(f"gauge scale c*                : {rep['gauge_scale_c']:.4f}")
    print(f"mass  rel Frobenius err       : {rep['mass_rel_frob_err']:.3f}")
    print(f"mass  per-entry corr          : {rep['mass_entry_corr']:.3f}")
    print(f"mass  |dM/dz| / |dM/dq| ratio : {rep['mass_dMdz_ratio']:.3f}  (true: 0)")
    print(f"potential affine R^2          : {rep['potential_affine_R2']:.3f}"
          f"  (slope {rep['potential_slope_a']:.3f} vs c* {rep['gauge_scale_c']:.3f})")
    print(f"kinetic R^2 (scale c*)        : {rep['kinetic_R2']:.3f}")
    print(f"total H affine R^2            : {rep['total_H_affine_R2']:.3f}")
    print(f"grad-V force corr             : {rep['gradV_force_corr']:.3f}")
    if "gradV_combined_corr" in rep:
        print(f"grad-(V+port springs) corr    : {rep['gradV_combined_corr']:.3f}"
              f"  (in-contact frac {rep['contact_in_frac']:.2f},"
              f" spring/gradV ratio {rep['contact_spring_ratio']:.2f})")
    print(f"Coriolis force corr           : {rep['coriolis_force_corr']:.3f}")
    print(f"damping affine R^2 (per DOF)  : {rep['damping_affine_R2']:.3f}")
    print(f"  learned c*.softplus(log_d)  : {rep['damping_learned']}")
    print(f"  true dof_damping            : {rep['damping_true']}")
    print(f"G_a rel Frobenius err         : {rep['G_rel_frob_err']:.3f}")
    print(f"G_a per-actuator cosine       : {rep['G_actuator_cosine']}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2)
        print(f"\nreport written to {args.out}")


if __name__ == "__main__":
    main()
