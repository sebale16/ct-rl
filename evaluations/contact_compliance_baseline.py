#!/usr/bin/env python
"""Drive evaluations.contact_compliance_probe over the whole baseline battery.

Kept as a script, not a notebook, so the pre-fix and post-fix numbers are
produced by byte-identical code:

    python -m evaluations.contact_compliance_baseline \
        --out results/contact_compliance/baseline.json

Later phases point ``--out`` somewhere else and change nothing else. Every
source the numbers depend on is md5'd before and after the whole battery and the
run aborts if anything moved underneath it.
"""
import argparse
import functools
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch as th

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("MUJOCO_GL", "disable")

from models.port_hamiltonian import PortHamiltonianModel  # noqa: E402
from evaluations.contact_compliance_probe import (  # noqa: E402
    DEPENDENCIES, MuJoCoCheetahMechanics, compliance_identity, constitutive_sweep,
    dependency_md5, drop_test, gate_sweep, make_probe_model, model_gaps,
    mujoco_reference_rest, probe_qpos, qp_matrices, reg_sweep, rigid_reference,
    static_force, verify_change_of_variables, verify_cone_invariance,
)

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="results/contact_compliance/baseline.json")
parser.add_argument("--label", default="baseline")
parser.add_argument("--drop-steps", type=int, default=3000)
parser.add_argument("--algebra-cache", default=None)
parser.add_argument(
    "--compliance", type=float, default=None, metavar="C0",
    help="enable the gate-shaped compliance with this initial c0; omitted (the "
         "default) reproduces the pre-fix battery exactly",
)
args = parser.parse_args()

# The ONLY difference between the baseline and the post-fix run: which model the
# whole battery below is built from. Every grid, protocol and probe state is the
# same code either way.
if args.compliance is None:
    probe_model = make_probe_model
else:
    probe_model = functools.partial(
        make_probe_model, contact_compliance=float(args.compliance)
    )

SELF = os.path.abspath(__file__)
t0 = time.time()
md5_before = dependency_md5(ROOT)
md5_before["evaluations/contact_compliance_baseline.py"] = hashlib.md5(
    open(SELF, "rb").read()).hexdigest()
print("md5 before:", json.dumps(md5_before, indent=1), flush=True)

out = {
    "description": (
        "Baseline pathology of the cone-constrained contact solver before the "
        "gate-shaped compliance fix. MuJoCo's mj_fullM mass matrix and its "
        "contact-free generalized acceleration are substituted for the learned "
        "mechanics, so only the contact geometry and _constraint_contact_solve "
        "are under test. contact_geometry='kinematic', contact_solver="
        "'constraint', K=6, float64, contact_dt=0.002, contact_iterations=12."
    ),
    "harness": "evaluations/contact_compliance_probe.py",
    "md5_before": md5_before,
}

# ---------------------------------------------------------------- algebra ----
print("== algebra ==", flush=True)
CACHE = args.algebra_cache
if CACHE and os.path.exists(CACHE):
    cached = json.load(open(CACHE))
    out["cone_invariance"], cov = cached["cone_invariance"], cached["change_of_variables"]
    print("  (reusing cached algebra checks)", flush=True)
else:
    out["cone_invariance"] = verify_cone_invariance(trials=2000, K=6, seed=0)
    cov = verify_change_of_variables(trials=24, K=4, seed=0, reg=1e-2, iterations=20000)
    if CACHE:
        json.dump({"cone_invariance": out["cone_invariance"],
                   "change_of_variables": cov}, open(CACHE, "w"))
print("  cone invariance:", out["cone_invariance"]["passed"],
      out["cone_invariance"]["max_abs_error"], flush=True)
out["change_of_variables"] = cov
print("  Lambda = S y:", cov["passed"], "worst rel", cov["worst_max_rel_error"],
      "boundary", cov["total_contacts_on_cone_boundary"],
      "zero-normal", cov["total_contacts_at_zero_normal"], flush=True)

mech = MuJoCoCheetahMechanics()
out["mujoco"] = {"total_mass_kg": mech.total_mass, "gravity": mech.gravity,
                 "body_weight_N": mech.weight_N, "nq": mech.nq, "nv": mech.nv}

# ------------------------------------------------- H-recovery cross-check ----
# qp_matrices reads H off the solver's Cholesky call. R must come back as the
# closed form of whichever law is active -- reg * scale * I when the compliance is
# disabled, [c0 (1 - s^2) + reg] * scale per coordinate when it is enabled. If it
# does not, every conditioning number below is suspect.
mref = probe_model(contact_regularization=1e-3, contact_gate_off=0.06)
qref = probe_qpos(mref, mech, 0.050)
qp = qp_matrices(mref, mech, qref)
scale_ref = float(qp["scale"][0])
gate_pair = qp["gate_pair"][0].numpy()
if mref._contact_compliance_raw is None:
    expect = np.full(gate_pair.shape, 1e-3 * scale_ref)
    law = "reg * scale * I"
else:
    c0 = (
        mref._contact_compliance_floor
        + th.nn.functional.softplus(mref._contact_compliance_raw.detach())
    ).numpy().repeat(2)
    expect = (c0 * (1.0 - gate_pair ** 2) + 1e-3) * scale_ref
    law = "[c0 (1 - s^2) + reg] * scale per coordinate"
Rd = np.asarray(qp["R_diag"], dtype=float)
off = float(np.abs(qp["R"][0].numpy() - np.diag(Rd)).max())
dev = float(np.abs(Rd - expect).max())
tol = 1e-10 * float(np.abs(expect).max())
out["H_recovery_crosscheck"] = {
    "contact_regularization": 1e-3,
    "law": law,
    "scale": scale_ref,
    "expected_R_diag": [float(v) for v in expect],
    "recovered_R_diag_max_abs_dev": dev,
    "recovered_R_max_offdiag": off,
    "recovered_R_diag": [float(v) for v in Rd],
    "recovered_Rtilde_diag": [float(v) for v in qp["Rtilde_diag"]],
    "passed": bool(dev < tol and off < tol),
}
print("  H recovery:", out["H_recovery_crosscheck"]["passed"],
      "R dev", out["H_recovery_crosscheck"]["recovered_R_diag_max_abs_dev"], flush=True)

# ------------------------------------------------------------- reg sweep ----
print("== reg sweep, gap +0.050 m, gate_off 0.06 ==", flush=True)
REGS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
rows = reg_sweep(probe_model, mech, REGS, gap=0.050, gate_off=0.06)
for r in rows:
    print("  reg=%-8g F=%8.4f N  cond(H)=%12.5g  cond(H+rhoI)=%10.4g  gate=%.6f  resid=%.2e"
          % (r["contact_regularization"], r["total_normal_force_N"], r["cond_H"],
             r["cond_H_plus_rho"], r["gate"][0], r["solver_residual"]), flush=True)
mrig = probe_model(contact_regularization=1e-2, contact_gate_off=0.06)
qrig = probe_qpos(mrig, mech, 0.050)
rigid = rigid_reference(mrig, mech, qrig)
out["reg_sweep_gap_0.050_gate_off_0.06"] = {
    "probe_gap_m": 0.050, "contact_gate_off": 0.06,
    "gate_at_probe": rows[0]["gate"][0],
    "rigid_reference": rigid,
    "rows": rows,
}
print("  rigid reference (reg->0, hard solve on the gated active set): %.4f N"
      % rigid["total_normal_force_N"], flush=True)

# same sweep at the shipped kinematic default band, for completeness
rows005 = reg_sweep(probe_model, mech, REGS, gap=0.050, gate_off=0.005)
out["reg_sweep_gap_0.050_gate_off_0.005"] = {
    "probe_gap_m": 0.050, "contact_gate_off": 0.005,
    "note": "gap 0.050 is outside the 0.005 band, so every gate is exactly 0",
    "rows": rows005,
}
print("  (gate_off 0.005 at the same gap: F =",
      [round(r["total_normal_force_N"], 6) for r in rows005], ")", flush=True)

# --------------------------------------------------- constitutive sweeps ----
print("== constitutive sweeps, gap +0.030 m, gate_off 0.06, reg 1e-2 ==", flush=True)
mbase = probe_model(contact_regularization=1e-2, contact_gate_off=0.06)
qbase = probe_qpos(mbase, mech, 0.030)
base = static_force(mbase, mech, qbase, with_cond=True)
print("  untrained baseline: F=%.4f N  e=%s beta=%s mu=%s gate=%s"
      % (base["total_normal_force_N"], np.round(base["e"], 4),
         np.round(base["beta"], 4), np.round(base["mu"], 4),
         np.round(base["gate"], 6)), flush=True)
GRIDS = {
    "e": (0.001, 0.01, 0.05, 0.1, 0.25, 0.4, 0.499),
    "beta": (0.001, 0.01, 0.05, 0.1, 0.25, 0.4, 0.499),
    "mu": (0.005, 0.05, 0.2, 0.5, 1.0, 1.5, 1.995),
}
const = {}
for name, grid in GRIDS.items():
    srows = constitutive_sweep(probe_model, mech, name, grid, gap=0.030,
                               gate_off=0.06, contact_regularization=1e-2)
    const[name] = srows
    forces = [r["total_normal_force_N"] for r in srows]
    print("  %-5s %s" % (name, "  ".join("%.4g:%.4fN" % (v, f)
                                         for v, f in zip(grid, forces))), flush=True)
    print("        spread = %.4g N over the whole range"
          % (max(forces) - min(forces)), flush=True)
out["constitutive_sweep_gap_0.030_gate_off_0.06"] = {
    "probe_gap_m": 0.030, "contact_gate_off": 0.06, "contact_regularization": 1e-2,
    "untrained_baseline": base,
    "rigid_reference": rigid_reference(mbase, mech, qbase),
    "rigid_reference_gate_1e-4": rigid_reference(mbase, mech, qbase, gate_threshold=1e-4),
    "sweeps": const,
    "spread_N": {k: float(max(r["total_normal_force_N"] for r in v)
                          - min(r["total_normal_force_N"] for r in v))
                 for k, v in const.items()},
}

# a reg sweep at the SAME state as the constitutive sweeps, so the comparison
# "numerical knob vs physical knobs" is at one state and not two
regrows030 = reg_sweep(probe_model, mech, REGS, gap=0.030, gate_off=0.06)
out["constitutive_sweep_gap_0.030_gate_off_0.06"]["reg_sweep_same_state"] = regrows030
print("  reg at the SAME state: %s" %
      "  ".join("%g:%.4fN" % (r["contact_regularization"], r["total_normal_force_N"])
                for r in regrows030), flush=True)

# -------------------------------------------- compliance identity + gate ----
# Rtilde is the compliance: check the closed form and the velocity-level
# constraint violation it predicts, measured against the running solver.
print("== compliance identity, gap +0.050 m, gate_off 0.06 ==", flush=True)
ident = {}
for reg in (1e-2, 1e-4, 1e-6):
    m = probe_model(contact_regularization=reg, contact_gate_off=0.06)
    q = probe_qpos(m, mech, 0.050)
    ci = compliance_identity(m, mech, q)
    ident["reg_%g" % reg] = ci
    print("  reg=%-8g Rtilde_n=%.6g (reg*scale/gate^2=%.6g, match=%s)  "
          "v+-v* identity: rel_err=%.2e holds=%s  Z=%.6g"
          % (reg, ci["Rtilde_diag_measured"][0],
             ci["Rtilde_diag_reg_over_gate_squared"][0],
             ci["Rtilde_matches_reg_over_gate_squared"],
             ci["velocity_identity_max_rel_error"], ci["velocity_identity_holds"],
             ci["impedance_Z_active"][0]), flush=True)
out["compliance_identity_gap_0.050"] = {
    "probe_gap_m": 0.050, "contact_gate_off": 0.06, "by_regularization": ident,
}

# The identity misses at small reg with the shipped 12 ADMM iterations. Show
# that this is solver truncation in the reported velocity field and not an
# algebra failure: the force is already converged at 12 iterations, the
# post-contact velocity is not.
iters_rows = []
for reg in (1e-2, 1e-4, 1e-6):
    for iters in (12, 200, 4000):
        m = probe_model(contact_regularization=reg, contact_gate_off=0.06,
                             contact_iterations=iters)
        q = probe_qpos(m, mech, 0.050)
        ci = compliance_identity(m, mech, q)
        sf = static_force(m, mech, q)
        iters_rows.append({
            "contact_regularization": reg, "contact_iterations": iters,
            "total_normal_force_N": sf["total_normal_force_N"],
            "solver_residual": ci["solver_residual"],
            "velocity_identity_max_rel_error": ci["velocity_identity_max_rel_error"],
            "velocity_identity_holds": ci["velocity_identity_holds"],
        })
        print("  reg=%-8g iters=%-5d F=%9.5f N  resid=%.2e  v+-v* rel_err=%.3e %s"
              % (reg, iters, sf["total_normal_force_N"], ci["solver_residual"],
                 ci["velocity_identity_max_rel_error"],
                 ci["velocity_identity_holds"]), flush=True)
out["compliance_identity_gap_0.050"]["iteration_crosscheck"] = {
    "note": ("with contact_iterations >= 200 the identity v+ - v* = -Rtilde Lambda "
             "holds to ~1e-13 at every regularizer; the misses at the shipped 12 "
             "iterations are ADMM truncation in the reported post-contact "
             "velocity, and the normal force is already converged at 12"),
    "rows": iters_rows,
}

print("== gate sweep at gap +0.050 m (the gate's only channel is Rtilde) ==",
      flush=True)
BANDS = (0.055, 0.06, 0.08, 0.12, 0.2, 0.5)
gsweep = {}
for reg in (1e-2, 1e-6):
    rows_g = gate_sweep(probe_model, mech, BANDS, gap=0.050,
                        contact_regularization=reg)
    gsweep["reg_%g" % reg] = rows_g
    print("  reg=%-8g %s" % (reg, "  ".join(
        "band%.3g(gate%.4f):%.3fN" % (r["contact_gate_off"], r["gate"][0],
                                      r["total_normal_force_N"]) for r in rows_g)),
        flush=True)
out["gate_sweep_gap_0.050"] = {
    "probe_gap_m": 0.050,
    "note": ("at reg=1e-2 the gate sets the force (it IS the stiffness); as "
             "reg -> 0 the same sweep collapses onto the rigid answer, i.e. the "
             "gate has cancelled out of W_full and b and survives only in Rtilde"),
    "by_regularization": gsweep,
}

# ------------------------------------------------------------- drop test ----
print("== drop test ==", flush=True)
drops = {}
for off in (0.005, 0.06):
    m = probe_model(contact_regularization=1e-2, contact_gate_off=off)
    d = drop_test(m, mech, steps=args.drop_steps, z0=0.6)
    drops["gate_off_%g" % off] = d
    print("  gate_off=%-6g rest_root_z=%.5f  min_gap=%+.6f m  F=%8.3f N (%.3f x weight)"
          "  |qvel|=%.2e  settle=%d  resid=%.2e  cone=%.2e"
          % (off, d["rest_root_z_m"], d["rest_min_gap_m"],
             d["rest_total_normal_force_N"], d["rest_total_normal_force_over_weight"],
             d["rest_max_abs_qvel"], d["steps_to_settle"], d["solver_residual"],
             d["cone_violation"]), flush=True)
mref2 = probe_model(contact_gate_off=0.005)
ref = mujoco_reference_rest(mref2, mech, z0=0.6, steps=args.drop_steps)
print("  MUJOCO reference: rest_root_z=%.5f  model gap there min=%+.3e m"
      % (ref["rest_root_z_m"], ref["model_min_gap_at_mujoco_rest_m"]), flush=True)
out["drop_test"] = {
    "protocol": ("cheetah released from root z = 0.6 m, zero action, semi-implicit "
                 "Euler at dt = contact_dt = 0.002 s for 3000 steps; MuJoCo M(q) "
                 "and contact-free qdd recomputed every step"),
    "body_weight_N": mech.weight_N,
    "contact_regularization": 1e-2,
    "arms": drops,
    "mujoco_reference": ref,
}

# ------------------------------------- reg sweep near full engagement ------
# The gate-shaped law is deliberately identical to the old one at s = 1, where
# R = reg * scale * I exactly. Sweeping reg at a nearly-closed gap therefore
# still moves the force, in both arms; recorded so that limit is not mistaken
# for a failure of the regularizer-insensitivity test above.
print("== reg sweep, gap +0.001 m, gate_off 0.06 (near full engagement) ==",
      flush=True)
rows001 = reg_sweep(probe_model, mech, REGS, gap=0.001, gate_off=0.06)
for r in rows001:
    print("  reg=%-8g F=%8.4f N  cond(H)=%12.5g  gate=%.6f"
          % (r["contact_regularization"], r["total_normal_force_N"], r["cond_H"],
             r["gate"][0]), flush=True)
out["reg_sweep_gap_0.001_gate_off_0.06"] = {
    "probe_gap_m": 0.001, "contact_gate_off": 0.06,
    "note": ("R = reg * scale * I at s = 1 by construction, so this is the "
             "regime the fix deliberately leaves alone"),
    "rows": rows001,
}

# --------------------------------------------------- learned compliance ----
out["contact_compliance_c0_init"] = args.compliance
if args.compliance is not None:
    # AC2, in two halves, because c0 = floor + softplus(raw) with
    # floor = fraction * c0_init:
    #   (a) the RAW parameter at a fixed initialization -- what a fit can reach
    #       without changing the configuration, i.e. [floor, inf);
    #   (b) the PHYSICAL c0 set through the constructor, which moves the floor
    #       with it -- the full range of the constitutive parameter.
    print("== compliance sweep (raw parameter, c0_init=%g), gap +0.030 m, "
          "gate_off 0.06, reg 1e-2 ==" % args.compliance, flush=True)
    RAW_GRID = (-10.0, -6.0, -3.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0, 40.0)
    crows = constitutive_sweep(probe_model, mech, "compliance", RAW_GRID,
                               gap=0.030, gate_off=0.06,
                               contact_regularization=1e-2)
    floor = (
        PortHamiltonianModel._CONTACT_COMPLIANCE_FLOOR_FRACTION
        * float(args.compliance)
    )
    for row, raw in zip(crows, RAW_GRID):
        row["c0_raw"] = float(raw)
        row["compliance_floor"] = float(floor)
        # the value the solver uses, floor included
        row["c0"] = floor + float(th.nn.functional.softplus(th.tensor(raw)))
        print("  c0=%-10.5g (raw %6.1f) F=%9.4f N  Rtilde_n=%.5g  cond(H)=%.5g"
              % (row["c0"], raw, row["total_normal_force_N"],
                 row["Rtilde_diag"][0], row["cond_H"]), flush=True)
    forces = [r["total_normal_force_N"] for r in crows]
    out["compliance_sweep_gap_0.030_gate_off_0.06"] = {
        "probe_gap_m": 0.030, "contact_gate_off": 0.06,
        "contact_regularization": 1e-2,
        "note": ("the sweep grid is the RAW parameter that set_constitutive "
                 "writes; the effective c0 = floor + softplus(raw) with "
                 "floor = %g is reported per row, so this sweep is bounded "
                 "below by the floor -- which is the point of the floor"
                 % floor),
        "compliance_floor": float(floor),
        "rows": crows,
        "force_ratio_max_over_min": float(max(forces) / max(min(forces), 1e-300)),
    }
    print("  force ratio max/min = %.4g (raw sweep, floored at c0=%g)"
          % (out["compliance_sweep_gap_0.030_gate_off_0.06"]["force_ratio_max_over_min"],
             floor),
          flush=True)

    print("== compliance sweep (physical c0 through the constructor), "
          "gap +0.030 m, gate_off 0.06, reg 1e-2 ==", flush=True)
    C0_GRID = (1e-4, 1e-3, 1e-2, 0.1, 1.0, 4.0, 10.0, 40.0, 400.0)
    prows = []
    for c0 in C0_GRID:
        m_c0 = make_probe_model(contact_regularization=1e-2,
                                contact_gate_off=0.06, contact_compliance=c0)
        row = static_force(m_c0, mech, probe_qpos(m_c0, mech, 0.030),
                           with_cond=True)
        row["c0"] = float(c0)
        row["compliance_floor"] = float(
            PortHamiltonianModel._CONTACT_COMPLIANCE_FLOOR_FRACTION * c0
        )
        prows.append(row)
        print("  c0=%-10.5g F=%9.4f N  Rtilde_n=%.5g  cond(H)=%.5g"
              % (c0, row["total_normal_force_N"], row["Rtilde_diag"][0],
                 row["cond_H"]), flush=True)
    pforces = [r["total_normal_force_N"] for r in prows]
    out["compliance_sweep_physical_c0_gap_0.030_gate_off_0.06"] = {
        "probe_gap_m": 0.030, "contact_gate_off": 0.06,
        "contact_regularization": 1e-2,
        "note": ("c0 requested through the constructor, so the floor moves with "
                 "it and the effective c0 is exactly the requested value"),
        "rows": prows,
        "force_ratio_max_over_min": float(
            max(pforces) / max(min(pforces), 1e-300)
        ),
        "monotone_decreasing": bool(
            all(b < a for a, b in zip(pforces, pforces[1:]))
        ),
    }
    print("  force ratio max/min = %.4g"
          % out["compliance_sweep_physical_c0_gap_0.030_gate_off_0.06"][
              "force_ratio_max_over_min"], flush=True)

    # AC5. d(loss)/d(c0) on a batch that actually touches the floor, through the
    # full drift (cholesky_ex + cholesky_solve + cone projections included).
    print("== gradient reaches c0 ==", flush=True)
    gm = probe_model(contact_regularization=1e-2, contact_gate_off=0.06)
    qpos_touch = probe_qpos(gm, mech, -0.002)
    lo, hi = gm.layout.pos_slice
    npos = hi - lo
    th.manual_seed(0)
    xb = th.zeros(8, gm.obs_dim, dtype=th.float64)
    xb[:, :npos] = th.as_tensor(qpos_touch[1:1 + npos], dtype=th.float64)
    xb[:, :npos] += 0.01 * th.randn(8, npos, dtype=th.float64)
    xb[:, npos:] = 0.05 * th.randn(8, gm.layout.nv, dtype=th.float64)
    ab = 0.1 * th.randn(8, gm.action_dim, dtype=th.float64)
    gm.zero_grad(set_to_none=True)
    # _structured_drift, not drift: drift() hard-casts its inputs to float32 and
    # the probe model is float64. Same graph, same contact solve.
    loss = gm._structured_drift(xb, ab).square().mean()
    loss.backward()
    grad = gm._contact_compliance_raw.grad
    out["gradient_to_c0"] = {
        "loss": float(loss),
        "grad_per_contact": [float(v) for v in grad],
        "grad_norm": float(grad.norm()),
        "grad_max_abs": float(grad.abs().max()),
        "all_finite": bool(th.isfinite(grad).all()),
        "nonzero_entries": int((grad != 0).sum()),
        "reference_grad_norm_contact_raw": float(gm._contact_raw.grad.norm()),
        "reference_grad_norm_mass": float(
            max(float(p.grad.norm()) for n, p in gm.named_parameters()
                if p.grad is not None and "mass" in n)
        ) if any("mass" in n for n, _ in gm.named_parameters()) else None,
    }
    print("  loss=%.6g  |dL/dc0_raw|=%.6g  max|.|=%.6g  nonzero=%d/%d  finite=%s"
          % (loss, out["gradient_to_c0"]["grad_norm"],
             out["gradient_to_c0"]["grad_max_abs"],
             out["gradient_to_c0"]["nonzero_entries"], grad.numel(),
             out["gradient_to_c0"]["all_finite"]), flush=True)
    print("  (for scale: |dL/d_contact_raw| = %.6g)"
          % out["gradient_to_c0"]["reference_grad_norm_contact_raw"], flush=True)

md5_after = dependency_md5(ROOT)
md5_after["evaluations/contact_compliance_baseline.py"] = hashlib.md5(
    open(SELF, "rb").read()).hexdigest()
out["label"] = args.label
out["md5_after"] = md5_after
out["md5_stable"] = bool(md5_before == md5_after)
out["wall_seconds"] = time.time() - t0
assert md5_before == md5_after, "source changed mid-measurement: %s vs %s" % (
    md5_before, md5_after)

path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
with open(path, "w") as handle:
    json.dump(out, handle, indent=1, default=float)
print("wrote", path, "in %.1f s" % out["wall_seconds"], flush=True)
