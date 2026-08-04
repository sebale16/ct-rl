#!/usr/bin/env python
"""AC1 over eight decades of ``contact_regularization``, at BOTH kinds of state.

The headline claim of the gate-shaped compliance is "sweeping the regularizer no
longer moves the contact force". That is true where the gate is tapering, and it
is NOT true at full engagement -- by construction: R = reg * scale * I exactly at
s = 1, which is the property that keeps the fully engaged limit identical to the
pre-fix solver. Reporting only the tapering states would overstate the fix, and
reporting only the engaged ones would hide it, so this measures both, over a
deliberately wider reg range than the battery, with enough ADMM iterations that
truncation cannot be mistaken for physics.

    python -m evaluations.contact_compliance_reg_range \
        --out results/contact_compliance/ac1_reg_range.json
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("MUJOCO_GL", "disable")

from evaluations.contact_compliance_probe import (  # noqa: E402
    MuJoCoCheetahMechanics, dependency_md5, make_probe_model, probe_qpos,
    static_force,
)

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="results/contact_compliance/ac1_reg_range.json")
parser.add_argument("--iterations", type=int, default=2000,
                    help="ADMM iterations, high enough that truncation is not "
                         "what is being measured")
parser.add_argument("--c0", type=float, default=40.0)
args = parser.parse_args()

SELF = os.path.abspath(__file__)
t0 = time.time()
md5_before = dependency_md5(ROOT)
md5_before["evaluations/contact_compliance_reg_range.py"] = hashlib.md5(
    open(SELF, "rb").read()).hexdigest()
print("md5 before:", json.dumps(md5_before, indent=1), flush=True)

mech = MuJoCoCheetahMechanics()
REGS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
# (label, gate band, minimum contact gap). The first three are load-bearing
# states at s = 1; the last three are in the taper.
STATES = (
    ("engaged_band_0.005_gap_-0.002", 0.005, -0.002),
    ("engaged_band_0.005_gap_-0.0006", 0.005, -6.0e-4),
    ("engaged_band_0.06_gap_-0.002", 0.06, -0.002),
    ("taper_band_0.005_gap_+0.001", 0.005, 0.001),
    ("taper_band_0.06_gap_+0.030", 0.06, 0.030),
    ("taper_band_0.06_gap_+0.050", 0.06, 0.050),
)
ARMS = {"baseline": None, "compliance_c0_%g" % args.c0: float(args.c0)}

out = {
    "description": __doc__,
    "md5_before": md5_before,
    "contact_iterations": args.iterations,
    "regs": list(REGS),
    "body_weight_N": mech.weight_N,
    "c0": args.c0,
    "states": {},
}

for label, band, gap in STATES:
    entry = {"contact_gate_off": band, "probe_gap_m": gap, "arms": {}}
    for arm, c0 in ARMS.items():
        kwargs = {} if c0 is None else {"contact_compliance": c0}
        rows = []
        for reg in REGS:
            model = make_probe_model(contact_regularization=float(reg),
                                     contact_gate_off=band,
                                     contact_iterations=args.iterations,
                                     **kwargs)
            row = static_force(model, mech, probe_qpos(model, mech, gap))
            row["contact_regularization"] = float(reg)
            rows.append(row)
        forces = [r["total_normal_force_N"] for r in rows]
        gates = rows[0]["gate"]
        entry["arms"][arm] = {
            "rows": rows,
            "max_gate": float(max(gates)),
            "force_N_by_reg": {"%g" % reg: f for reg, f in zip(REGS, forces)},
            "force_ratio_max_over_min": float(
                max(forces) / max(min(forces), 1e-300)
            ),
        }
        print("  %-30s %-20s max_gate=%.6f ratio=%.4f  F=[%.6g .. %.6g]"
              % (label, arm, max(gates),
                 entry["arms"][arm]["force_ratio_max_over_min"],
                 forces[0], forces[-1]), flush=True)
    entry["ratio_improvement"] = float(
        (entry["arms"]["baseline"]["force_ratio_max_over_min"] - 1.0)
        / max(entry["arms"]["compliance_c0_%g" % args.c0]
              ["force_ratio_max_over_min"] - 1.0, 1e-300)
    )
    out["states"][label] = entry

md5_after = dependency_md5(ROOT)
md5_after["evaluations/contact_compliance_reg_range.py"] = hashlib.md5(
    open(SELF, "rb").read()).hexdigest()
out["md5_after"] = md5_after
out["md5_stable"] = bool(md5_before == md5_after)
out["wall_seconds"] = time.time() - t0
assert md5_before == md5_after, "source changed mid-measurement"

path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
json.dump(out, open(path, "w"), indent=1, default=float)
print("wrote", path, "in %.1f s" % out["wall_seconds"], flush=True)
