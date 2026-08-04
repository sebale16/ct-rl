#!/usr/bin/env python
"""Default-disabled must be EXACTLY the pre-fix tree, not merely close to it.

Loads a reference copy of ``models/port_hamiltonian.py`` from before the
gate-shaped-compliance edit as a second module and compares ``drift`` bit for
bit against the working tree, for fresh models across five seeds and five
contact configurations, and for every sidecar in
``results/contact_sidecar_corpus.json``.

    python -m evaluations.contact_compliance_bit_exactness \
        --reference /path/to/pre-fix/port_hamiltonian.py \
        --expect-reference-md5 a5310e33eb06cf8b5cd79a182b3f7f8a

The reference file is read only. The permanent in-tree guard against this
regression is ``tests/test_contact_gate_shaped_compliance.py``; this script is
the one-off before/after measurement, which needs a copy of "before".
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import torch as th

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "disable")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--reference", required=True,
                    help="a copy of models/port_hamiltonian.py from before the edit")
parser.add_argument("--expect-reference-md5", default=None)
parser.add_argument("--out",
                    default="results/contact_compliance/ac4_bit_exactness.json")
args = parser.parse_args()
REF = os.path.abspath(args.reference)


def md5(path):
    with open(path, "rb") as h:
        return hashlib.md5(h.read()).hexdigest()


ref_md5 = md5(REF)
if args.expect_reference_md5:
    assert ref_md5 == args.expect_reference_md5, (ref_md5, args.expect_reference_md5)
new_md5 = md5(os.path.join(ROOT, "models/port_hamiltonian.py"))

spec = importlib.util.spec_from_file_location("ph_ref", REF)
ph_ref = importlib.util.module_from_spec(spec)
sys.modules["ph_ref"] = ph_ref
spec.loader.exec_module(ph_ref)

import models.port_hamiltonian as ph_new  # noqa: E402

out = {
    "reference_module": REF,
    "reference_md5": ref_md5,
    "reference_source": "/work2/10976/sebale16/frontera/ct-rl-kin/models/port_hamiltonian.py",
    "edited_md5_before": new_md5,
}

CFGS = [
    ("kinematic_K6", dict(contact_force=6, contact_geometry="kinematic",
                          contact_solver="constraint")),
    ("kinematic_K6_band_0.06", dict(contact_force=6, contact_geometry="kinematic",
                                    contact_solver="constraint",
                                    contact_gate_off=0.06)),
    ("learned_K4_constraint", dict(contact_force=4, contact_geometry="learned",
                                   contact_solver="constraint")),
    ("learned_K4_compliant", dict(contact_force=4, contact_geometry="learned",
                                  contact_solver="compliant")),
    ("no_contact", dict(contact_force=0)),
]


def build(mod, seed, cfg, hidden=(128, 128)):
    th.manual_seed(seed)
    np.random.seed(seed)
    return mod.PortHamiltonianModel(
        obs_dim=17, action_dim=6, mode="structured", structured_hidden=hidden,
        dof_layout=mod.DOFLayout.cheetah(), **cfg,
    ).eval()


def batch(model, seed, touch=True):
    """A batch that puts several contact points through and around the floor."""
    lo, hi = model.layout.pos_slice
    npos = hi - lo
    th.manual_seed(1000 + seed)
    x = th.zeros(12, model.obs_dim, dtype=th.float64)
    x[:, :npos] = 0.15 * th.randn(12, npos, dtype=th.float64)
    if touch and model.contact_force > 0 and model.contact_geometry == "kinematic":
        # sweep the root height across the floor so gates cover [0, 1]
        x[:, 0] = th.linspace(-0.05, 0.10, 12, dtype=th.float64)
    x[:, npos:] = 0.3 * th.randn(12, model.layout.nv, dtype=th.float64)
    a = 0.3 * th.randn(12, model.action_dim, dtype=th.float64)
    return x, a


fresh = []
for name, cfg in CFGS:
    for seed in (0, 1, 2, 3, 4):
        a = build(ph_ref, seed, cfg)
        b = build(ph_new, seed, cfg)
        ka, kb = list(a.state_dict().keys()), list(b.state_dict().keys())
        sd_a, sd_b = a.state_dict(), b.state_dict()
        params_equal = all(th.equal(sd_a[k], sd_b[k]) for k in ka) and ka == kb
        x, u = batch(a, seed)
        with th.no_grad():
            # drift() hard-casts its inputs to float32, so this is the real
            # production path; _structured_drift is compared in float64 too.
            da, db = a.drift(x, u), b.drift(x, u)
            a64, b64 = build(ph_ref, seed, cfg).double(), build(ph_new, seed, cfg).double()
            d64a = a64._structured_drift(x, u)
            d64b = b64._structured_drift(x, u)
        gates = None
        if a.contact_force > 0:
            with th.no_grad():
                lo, hi = a.layout.pos_slice
                pos = x[:, : hi - lo]
                gates = a64._contact_gate(a64._contact_geometry(pos, x[:, hi - lo:])[0])
        row = {
            "config": name, "seed": seed,
            "state_dict_keys_equal": bool(ka == kb),
            "n_keys": len(ka),
            "state_dict_values_bit_equal": bool(params_equal),
            "drift_bit_equal": bool(th.equal(da, db)),
            "drift_max_abs_diff": float((da - db).abs().max()),
            "drift_finite": bool(th.isfinite(da).all()),
            "drift_float64_bit_equal": bool(th.equal(d64a, d64b)),
            "drift_float64_max_abs_diff": float((d64a - d64b).abs().max()),
            "gate_min": None if gates is None else float(gates.min()),
            "gate_max": None if gates is None else float(gates.max()),
            "n_gates_partial": None if gates is None else int(
                ((gates > 1e-9) & (gates < 1 - 1e-9)).sum()),
        }
        fresh.append(row)
        print("  %-24s seed %d keys=%d equal=%s drift_bit_equal=%s gate[%s,%s] partial=%s"
              % (name, seed, row["n_keys"], row["state_dict_keys_equal"],
                 row["drift_bit_equal"], row["gate_min"], row["gate_max"],
                 row["n_gates_partial"]), flush=True)
out["fresh_models"] = fresh

# ------------------------------------------------------------ corpus ----
corpus = json.load(open(os.path.join(ROOT, "results/contact_sidecar_corpus.json")))
rows = []
for entry in corpus["checkpoints"]:
    path = entry["path"]
    if not os.path.exists(path):
        rows.append({"path": path, "status": "missing"})
        print("  MISSING", path, flush=True)
        continue
    assert md5(path) == entry["md5"], path
    sd = th.load(path, map_location="cpu", weights_only=True)
    K = sd["_contact_raw"].shape[1]
    solver = "constraint" if int(sd.get("_contact_solver_version", 0)) == 1 else "compliant"
    geom = "kinematic" if int(sd.get("_contact_geometry_version", 0)) == 1 else "learned"
    hidden = (
        sd["mass_net.0.weight"].shape[0],
        sd["mass_net.2.weight"].shape[0],
    ) if "mass_net.0.weight" in sd else (128, 128)
    cfg = dict(contact_force=K, contact_geometry=geom, contact_solver=solver)
    a = build(ph_ref, 0, cfg, hidden)
    b = build(ph_new, 0, cfg, hidden)
    a.load_state_dict(sd)
    b.load_state_dict(sd)
    x, u = batch(a, 0)
    with th.no_grad():
        da, db = a.drift(x, u), b.drift(x, u)
        a64 = build(ph_ref, 0, cfg, hidden); a64.load_state_dict(sd); a64 = a64.double()
        b64 = build(ph_new, 0, cfg, hidden); b64.load_state_dict(sd); b64 = b64.double()
        d64a, d64b = a64._structured_drift(x, u), b64._structured_drift(x, u)
    row = {
        "path": path, "md5": entry["md5"], "K": int(K), "solver": solver,
        "geometry": geom, "n_keys_ref": len(a.state_dict()),
        "n_keys_new": len(b.state_dict()),
        "keys_equal": bool(list(a.state_dict()) == list(b.state_dict())),
        "drift_bit_equal": bool(th.equal(da, db)),
        "drift_max_abs_diff": float((da - db).abs().max()),
        "drift_finite": bool(th.isfinite(da).all()),
        "drift_float64_bit_equal": bool(th.equal(d64a, d64b)),
        "drift_float64_max_abs_diff": float((d64a - d64b).abs().max()),
        "status": "ok",
    }
    rows.append(row)
    print("  %-70s K=%d %-10s %-9s keys=%d/%d bit_equal=%s"
          % (os.path.basename(path), K, solver, geom, row["n_keys_ref"],
             row["n_keys_new"], row["drift_bit_equal"]), flush=True)
out["corpus"] = rows
out["corpus_count"] = len(rows)
out["all_fresh_bit_equal"] = bool(
    all(r["drift_bit_equal"] and r["drift_float64_bit_equal"] for r in fresh))
out["all_fresh_keys_equal"] = bool(all(r["state_dict_keys_equal"] for r in fresh))
out["all_corpus_bit_equal"] = bool(
    all(r.get("drift_bit_equal") and r.get("drift_float64_bit_equal")
        for r in rows if r["status"] == "ok"))
out["all_corpus_loaded"] = bool(all(r["status"] == "ok" for r in rows))
out["passed"] = bool(out["all_fresh_bit_equal"] and out["all_fresh_keys_equal"]
                     and out["all_corpus_bit_equal"] and out["all_corpus_loaded"])
out["edited_md5_after"] = md5(os.path.join(ROOT, "models/port_hamiltonian.py"))
assert out["edited_md5_after"] == out["edited_md5_before"], "source moved mid-run"

dest = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
os.makedirs(os.path.dirname(dest), exist_ok=True)
json.dump(out, open(dest, "w"), indent=1, default=float)
print("PASSED =", out["passed"], "-> wrote", dest, flush=True)
