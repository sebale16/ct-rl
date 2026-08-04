#!/usr/bin/env python
"""Regularizer insensitivity where the force is a real body weight.

The fixed-state reg sweep in ``contact_compliance_baseline`` is taken at a
+0.050 m clearance. With the gate-shaped compliance the force there is ~1e-3 N by
design (5 cm above the floor is not a contact), so "insensitive to reg" is a weak
statement at that state. This script measures the same question at the state that
matters: the settled drop-test rest pose, which carries exactly body weight.

Also records what the fix costs the ADMM solve, by re-running the drop at several
iteration counts. This is where the ``rho = mean(diag H)`` regression showed up:
the gated-off contacts contribute c0 * scale to that mean, so the step size was
two orders of magnitude too small and 12 iterations rested 100x too deep. The
step size is now taken from the gated Delassus diagonal plus the conditioning
floor, which is c0-independent, and these rows are the check that the rest pose
no longer moves with the iteration count.

    python -m evaluations.contact_compliance_rest_probe \
        --out results/contact_compliance/rest_reg_insensitivity.json
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

from evaluations.contact_compliance_probe import (  # noqa: E402
    MuJoCoCheetahMechanics, dependency_md5, drop_test, make_probe_model,
    mujoco_reference_rest, qp_matrices, static_force,
)

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="results/contact_compliance/rest_reg_insensitivity.json")
parser.add_argument("--drop-steps", type=int, default=3000)
parser.add_argument("--c0", type=float, default=40.0)
args = parser.parse_args()

SELF = os.path.abspath(__file__)
t0 = time.time()
md5_before = dependency_md5(ROOT)
md5_before["evaluations/contact_compliance_rest_probe.py"] = hashlib.md5(
    open(SELF, "rb").read()).hexdigest()
print("md5 before:", json.dumps(md5_before, indent=1), flush=True)

mech = MuJoCoCheetahMechanics()
REGS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
ARMS = {
    "baseline": make_probe_model,
    "compliance_c0_%g" % args.c0: functools.partial(
        make_probe_model, contact_compliance=float(args.c0)),
}
out = {
    "description": (
        "Regularizer sweep at the settled rest pose (force == body weight), for "
        "the pre-fix regularizer and for the gate-shaped compliance. Each row "
        "re-runs the whole 3000-step drop, so the rest pose is re-found under "
        "that regularizer rather than held fixed from another one."
    ),
    "md5_before": md5_before,
    "body_weight_N": mech.weight_N,
    "c0": args.c0,
}

ref = mujoco_reference_rest(make_probe_model(contact_gate_off=0.005), mech,
                            z0=0.6, steps=args.drop_steps)
out["mujoco_reference"] = ref
print("MuJoCo rest: root_z=%.6f, model gap there min=%+.4e m"
      % (ref["rest_root_z_m"], ref["model_min_gap_at_mujoco_rest_m"]), flush=True)

drops = {}
for arm, factory in ARMS.items():
    for band in (0.005, 0.06):
        rows = []
        for reg in REGS:
            m = factory(contact_regularization=float(reg), contact_gate_off=band)
            d = drop_test(m, mech, steps=args.drop_steps, z0=0.6)
            # the same rest pose, re-solved as a held static state
            held = static_force(m, mech, d["rest_qpos"], with_cond=True)
            qp = qp_matrices(m, mech, d["rest_qpos"])
            rows.append({
                "contact_regularization": float(reg),
                "rest_min_gap_m": d["rest_min_gap_m"],
                "rest_gap_m": d["rest_gap_m"],
                "rest_total_normal_force_N": d["rest_total_normal_force_N"],
                "rest_total_normal_force_over_weight":
                    d["rest_total_normal_force_over_weight"],
                "rest_max_abs_qvel": d["rest_max_abs_qvel"],
                "steps_to_settle": d["steps_to_settle"],
                "solver_residual": d["solver_residual"],
                "cone_violation": d["cone_violation"],
                "held_total_normal_force_N": held["total_normal_force_N"],
                "held_gate": held["gate"],
                "cond_H": qp["cond_H"],
                "cond_H_plus_rho": qp["cond_H_plus_rho"],
                "Rtilde_diag": [float(v) for v in qp["Rtilde_diag"]],
                "R_diag": [float(v) for v in qp["R_diag"]],
                "scale": float(qp["scale"][0]),
            })
            print("  %-22s band=%-6g reg=%-8g rest_gap=%+.6e F=%9.4f N (%.4f w)  "
                  "|qvel|=%.2e settle=%-5d resid=%.2e cond(H)=%.4g"
                  % (arm, band, reg, d["rest_min_gap_m"],
                     d["rest_total_normal_force_N"],
                     d["rest_total_normal_force_over_weight"],
                     d["rest_max_abs_qvel"], d["steps_to_settle"],
                     d["solver_residual"], qp["cond_H"]), flush=True)
        gaps = [r["rest_min_gap_m"] for r in rows]
        forces = [r["rest_total_normal_force_N"] for r in rows]
        drops["%s_band_%g" % (arm, band)] = {
            "rows": rows,
            "rest_gap_spread_m": float(max(gaps) - min(gaps)),
            "force_ratio_max_over_min": float(max(forces) / max(min(forces), 1e-300)),
            "gap_error_vs_mujoco_m": [
                float(g - ref["model_min_gap_at_mujoco_rest_m"]) for g in gaps],
        }
        print("    -> rest gap spread over 5 decades of reg = %.3e m; force ratio %.5f"
              % (drops["%s_band_%g" % (arm, band)]["rest_gap_spread_m"],
                 drops["%s_band_%g" % (arm, band)]["force_ratio_max_over_min"]),
              flush=True)
out["drop_reg_sweep"] = drops

# ------------------------------------------- ADMM iteration sensitivity ----
print("== ADMM iteration sensitivity at rest ==", flush=True)
iters_rows = []
for arm, factory in ARMS.items():
    for iters in (12, 25, 50, 200):
        m = factory(contact_regularization=1e-2, contact_gate_off=0.06,
                    contact_iterations=iters)
        d = drop_test(m, mech, steps=args.drop_steps, z0=0.6)
        qp = qp_matrices(m, mech, d["rest_qpos"])
        row = {
            "arm": arm, "contact_iterations": iters,
            "rest_min_gap_m": d["rest_min_gap_m"],
            "rest_total_normal_force_N": d["rest_total_normal_force_N"],
            "rest_max_abs_qvel": d["rest_max_abs_qvel"],
            "steps_to_settle": d["steps_to_settle"],
            "solver_residual": d["solver_residual"],
            "cone_violation": d["cone_violation"],
            "rho_over_active_R": float(
                qp["rho"][0] / max(float(np.min(qp["R_diag"])), 1e-300)),
        }
        iters_rows.append(row)
        print("  %-22s iters=%-5d rest_gap=%+.6e F=%9.4f N |qvel|=%.2e settle=%-5d "
              "resid=%.2e" % (arm, iters, d["rest_min_gap_m"],
                              d["rest_total_normal_force_N"], d["rest_max_abs_qvel"],
                              d["steps_to_settle"], d["solver_residual"]), flush=True)
out["admm_iterations_at_rest"] = {
    "contact_regularization": 1e-2, "contact_gate_off": 0.06, "rows": iters_rows,
}

md5_after = dependency_md5(ROOT)
md5_after["evaluations/contact_compliance_rest_probe.py"] = hashlib.md5(
    open(SELF, "rb").read()).hexdigest()
out["md5_after"] = md5_after
out["md5_stable"] = bool(md5_before == md5_after)
out["wall_seconds"] = time.time() - t0
assert md5_before == md5_after, "source changed mid-measurement"

path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
json.dump(out, open(path, "w"), indent=1, default=float)
print("wrote", path, "in %.1f s" % out["wall_seconds"], flush=True)
