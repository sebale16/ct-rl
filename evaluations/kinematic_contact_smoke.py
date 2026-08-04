"""Short A/B fit: learned vs kinematic contact geometry, identical data/seed.

Deliverable A of the kinematic-contact payoff measurement. This is deliberately
small enough to run on a login node under ``timeout 900``; the conclusive run is
the 100k-step sbatch sweep (benchmarks/kinematic_contact_upper_bound.slurm).

Two things are measured that the standard recovery report cannot give:

  1. the in-contact fraction AS A FUNCTION OF FIT STEP. The learned port is
     known to eject itself within ~50 optimizer steps and never return, so the
     end-of-fit number alone cannot distinguish "never engaged" from
     "engaged then collapsed".
  2. the same accuracy axes as evaluations.hamiltonian_recovery, computed by
     calling that module's own evaluate_dataset, so the numbers are directly
     comparable to results/dynamics_upper_bound_100k*/.

Usage:
    python -m evaluations.kinematic_contact_smoke collect --data D.npz --n 3000
    python -m evaluations.kinematic_contact_smoke fit --data D.npz \
        --arm learned --contact_geometry learned --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch as th

from common.buffers import ReplayBuffer
from environment.dmc import DMCContinuousEnv
from models.port_hamiltonian import PortHamiltonianModel
from evaluations.hamiltonian_recovery import (
    _jsonable,
    _parse_contact_compliance,
    _select_recovery_dof_layout,
    collect,
    evaluate_dataset,
)


def _make_env(seed: int) -> DMCContinuousEnv:
    return DMCContinuousEnv("cheetah", "run", time_sampling="uniform", dt=0.01,
                            physics_dt=0.002, episode_duration=20.0, seed=seed,
                            raw_state_obs=False)


def _compliance_row(m, obs, act, d) -> dict:
    """Constitutive state of the gate-shaped compliance on a fixed batch.

    c0 is a LEARNED softness knob, so the fit can make the contact vanish
    without ever touching the gate band. The numbers that make that visible:

      c0            softplus(raw) + floor, per contact point (the model's own
                    ``compliance_c0``);
      Rtilde_nn     the PHYSICAL compliance the solver actually applies,
                    scale [c0 (1/s^2 - 1) + reg/s^2] with s = gate. This is
                    recomputed here from the model's own M and J rather than
                    read off the solver, because ``scale`` (the detached mean of
                    diag J M^-1 J^T) is not returned;
      stiffness     beta / (Rtilde_nn dt^2), in N/m IN THE LEARNED MASS GAUGE.
                    At static rest v+_n = 0 and gdot = 0, so stationarity of the
                    physical QP gives beta (-g)/dt = Rtilde_nn Lambda_n, i.e.
                    a sustained penetration -g = Rtilde_nn F dt^2 / beta;
      rest_gap      that penetration evaluated at the normal force the solver
                    actually returned on this batch, i.e. how far off the floor
                    the fitted contact law would hold this load.

    ``scale`` and the mass gauge are recorded for EVERY arm, compliance or not,
    because R is scale-multiplied by construction (deliberately, so that R is
    covariant under a rescaling of the learned mass matrix). That makes the mass
    head a second, unlabelled softness knob: it can inflate Rtilde without
    touching any contact parameter. An arm with the compliance disabled has the
    same channel open through R = reg * scale * I, so the control needs the same
    column or the comparison cannot tell the two channels apart.
    """
    with th.no_grad():
        raw = getattr(m, "_contact_compliance_raw", None)
        c0 = (
            d["compliance_c0"].double() if raw is not None
            else th.zeros(m.contact_force, dtype=th.float64)
        )
        gate = d["gate"].double()                              # (B,K)
        _, _, M, _ = m._structured_free_acceleration(
            th.as_tensor(obs, dtype=th.float32),
            th.as_tensor(act, dtype=th.float32))
        J = th.stack((d["J_n"], d["J_t"]), dim=2).reshape(
            gate.shape[0], 2 * gate.shape[1], -1).double()
        M = M.double()
        W_full = th.bmm(J, th.linalg.solve(M, J.transpose(1, 2)))
        scale = W_full.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-6)
        reg = float(m.contact_regularization)
        beta = float(0.5 * th.sigmoid(m._contact_raw[1].double()).mean())
        dt = float(m.contact_dt)
        s2 = gate.square().clamp_min(1e-30)
        rtilde = scale[:, None] * (c0[None, :] * (1.0 / s2 - 1.0) + reg / s2)
        rtilde = th.where(gate > 0, rtilde, th.full_like(rtilde, float("inf")))
        stiff = beta / (rtilde * dt * dt)
        fn = (d["normal_force"].double()).clamp_min(0.0)
        rest_gap = rtilde * fn * dt * dt / max(beta, 1e-12)
        engaged = gate > 1e-6
        def _m(t, mask):
            v = t[mask]
            return float(v.mean()) if v.numel() else float("nan")
        row = {
            "scale_mean": float(scale.mean()),
            "mass_diag_mean": float(M.diagonal(dim1=-2, dim2=-1).mean()),
            "mass_logdet_mean": float(th.linalg.slogdet(M)[1].mean()),
            "beta": beta,
            "Rtilde_nn_engaged_mean": _m(rtilde, engaged),
            "stiffness_N_per_m_engaged_mean": _m(stiff, engaged),
            "stiffness_N_per_m_at_gate1": float(
                beta / (scale.mean() * reg * dt * dt)),
            "implied_rest_gap_engaged_mean_m": _m(rest_gap, engaged),
            "implied_rest_gap_max_m": float(rest_gap[engaged].max())
            if bool(engaged.any()) else float("nan"),
        }
        if raw is not None:
            row.update({
                "c0": [float(v) for v in c0],
                "c0_mean": float(c0.mean()),
                "c0_min": float(c0.min()),
                "c0_max": float(c0.max()),
                "c0_ratio_to_init": (
                    float(c0.mean()) / float(m.contact_compliance)),
            })
        return row


def _trace(m, obs, act) -> dict:
    """One row of the in-contact-fraction-vs-step trace."""
    with th.no_grad():
        d = m.contact_diagnostics(obs, act)
        gap = d["gap"].double()
        gate = d["gate"].double()
        ni = d["normal_impulse"].double()
        scale = float(np.percentile(np.abs(ni.numpy()), 95))
        thr = max(1e-8, 1e-4 * scale)
        return {
            **_compliance_row(m, obs, act, d),
            "in_contact_frac": float((ni > thr).double().mean()),
            "in_contact_frac_any": float((ni > thr).any(dim=1).double().mean()),
            "active_threshold": thr,
            "gate_mean": float(gate.mean()),
            "gate_nonzero_frac": float((gate > 1e-6).double().mean()),
            "gap_mean": float(gap.mean()),
            "gap_median": float(gap.median()),
            "gap_min": float(gap.min()),
            "normal_impulse_mean": float(ni.abs().mean()),
            "normal_impulse_max": float(ni.abs().max()),
            "contact_accel_rms": float(
                d["contact_acceleration"].double().pow(2).mean().sqrt()),
            "free_accel_rms": float(
                d["free_acceleration"].double().pow(2).mean().sqrt()),
        }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("collect", "fit"))
    p.add_argument("--data", required=True)
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--n_eval", type=int, default=500)
    p.add_argument("--reference_seed", type=int, default=123)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arm", default="arm")
    p.add_argument("--contact_geometry", choices=("learned", "kinematic"),
                   default="learned")
    p.add_argument("--contact_solver", choices=("compliant", "constraint"),
                   default="constraint")
    p.add_argument("--contact_force", type=int, default=6)
    p.add_argument("--contact_gate_off", type=float, default=None)
    p.add_argument("--contact_iterations", type=int, default=12)
    p.add_argument("--contact_regularization", type=float, default=0.01)
    p.add_argument(
        "--contact_compliance", default=None,
        help="initial c0 for the gate-shaped compliance ('true' = model "
             "default, a number = that c0, omitted = the historical fixed "
             "Delassus regularizer)")
    p.add_argument(
        "--freeze_c0", action="store_true",
        help="hold c0 at its calibrated init: _contact_compliance_raw is "
             "detached from autograd AND withheld from the optimizer, so this "
             "separates 'the compliance model is better' from 'the extra "
             "parameter is being exploited'")
    p.add_argument("--fit_steps", type=int, default=2000)
    p.add_argument("--fit_horizon", type=int, default=1)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--trace_every", type=int, default=10)
    p.add_argument("--trace_batch", type=int, default=256)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.cmd == "collect":
        env = _make_env(args.reference_seed)
        O, A, NO, DT, DN = collect(env, args.n, policy=None, ou_sigma=0.4,
                                   seed=args.reference_seed)
        np.savez(args.data, O=O, A=A, NO=NO, DT=DT, DN=DN)
        print(f"wrote {args.n} transitions to {args.data}")
        return

    d = np.load(args.data)
    O, A, NO, DT, DN = d["O"], d["A"], d["NO"], d["DT"], d["DN"]
    env = _make_env(args.reference_seed)
    layout = _select_recovery_dof_layout(
        "cheetah", False, int(env.observation_space.shape[0]), None)
    max_step = getattr(env, "physics_dt", None)

    th.manual_seed(args.seed + 1)
    od = int(env.observation_space.shape[0])
    ad = int(env.action_space.shape[0])
    m = PortHamiltonianModel(
        od, ad, mode="structured", structured_hidden=(128, 128),
        contact_force=args.contact_force,
        contact_solver=args.contact_solver,
        contact_geometry=args.contact_geometry,
        contact_gate_off=args.contact_gate_off,
        contact_dt=max_step or 0.002,
        contact_iterations=args.contact_iterations,
        contact_regularization=args.contact_regularization,
        contact_compliance=_parse_contact_compliance(args.contact_compliance),
        dof_layout=layout)

    frozen_c0 = None
    if args.freeze_c0:
        if m._contact_compliance_raw is None:
            raise SystemExit("--freeze_c0 requires --contact_compliance")
        m._contact_compliance_raw.requires_grad_(False)
        frozen_c0 = [
            float(v) for v in (
                m._contact_compliance_floor
                + th.nn.functional.softplus(m._contact_compliance_raw)
            ).detach()
        ]

    buf = ReplayBuffer(len(O), env.observation_space, env.action_space,
                       device="cpu", n_envs=1)
    for i in range(len(O)):
        buf.add(O[i:i+1], A[i:i+1], np.zeros(1, np.float32), DN[i:i+1],
                NO[i:i+1], np.zeros(1, np.float32), DT[i:i+1])
    # requires_grad=False alone would still let Adam touch a parameter if a
    # stale .grad were present, so the frozen tensor is withheld outright.
    opt = th.optim.Adam([q for q in m.parameters() if q.requires_grad], lr=1e-3)

    # A FIXED probe batch, so the trace measures the model changing and not the
    # states changing. Held-out tail, i.e. the same rows the report scores.
    probe_obs = O[-args.trace_batch:]
    probe_act = A[-args.trace_batch:]

    trace = [dict(step=0, loss=None, **_trace(m, probe_obs, probe_act))]
    t0 = time.time()
    losses = []
    for s in range(args.fit_steps):
        if args.fit_horizon > 1:
            seq = buf.sample_sequences(args.batch, args.fit_horizon)
            loss = m.fit_step_rollout(seq.observations, seq.actions,
                                      seq.next_observations, seq.dt, seq.mask,
                                      opt, max_step=max_step)
        else:
            bt = buf.sample(args.batch)
            loss = m.fit_step(bt.observations, bt.actions, bt.next_observations,
                              bt.dt, opt, max_step=max_step)
        losses.append(float(loss))
        step = s + 1
        if args.trace_every and (step % args.trace_every == 0
                                 or step in (1, 2, 5)):
            trace.append(dict(step=step, loss=float(loss),
                              **_trace(m, probe_obs, probe_act)))
        if step % 200 == 0:
            print(f"[fit] step {step}/{args.fit_steps}: loss {loss:.5f} "
                  f"in_contact {trace[-1]['in_contact_frac']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    fit_seconds = time.time() - t0

    axes = evaluate_dataset(m, env, (O, A, NO, DT, DN), args.n_eval,
                            policy=None, max_step=max_step, with_energy=False)
    report = {
        "arm": args.arm,
        "config": {
            "contact_geometry": args.contact_geometry,
            "contact_solver": args.contact_solver,
            "contact_force": args.contact_force,
            "contact_gate_off_resolved": float(m._contact_gate_off),
            "contact_iterations": args.contact_iterations,
            "contact_regularization": args.contact_regularization,
            "contact_compliance": m.contact_compliance,
            "contact_compliance_floor": (
                m._contact_compliance_floor
                if m.contact_compliance is not None else None),
            "freeze_c0": bool(args.freeze_c0),
            "frozen_c0": frozen_c0,
            "fit_steps": args.fit_steps,
            "fit_horizon": args.fit_horizon,
            "batch": args.batch,
            "seed": args.seed,
            "n": int(len(O)),
            "n_eval": args.n_eval,
            "reference_seed": args.reference_seed,
        },
        "fit_seconds": fit_seconds,
        "loss_first10_mean": float(np.mean(losses[:10])),
        "loss_last100_mean": float(np.mean(losses[-100:])),
        "contact_trace": trace,
        "axes": axes,
        "headline": axes["headline"],
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(_jsonable(report), f, indent=2)
        print(f"report written to {args.out}")
    h = axes["headline"]
    print(f"[{args.arm}] accel_nrmse {h['accel_nrmse']:.4f} "
          f"mass_rel_frob {h['mass_rel_frob_err']:.4f} "
          f"final in_contact {trace[-1]['in_contact_frac']:.4f}")


if __name__ == "__main__":
    main()
