#!/usr/bin/env python
"""Collate the compliance-envelope A/B smoke into JSON/CSV.

Reads the per-arm reports written by ``evaluations.kinematic_contact_smoke`` and
emits two things:

  summary.csv / summary.json   one row per arm: the accuracy axes plus the
                               contact-engagement endpoints, in the same column
                               order as results/kinematic_contact_smoke/ so the
                               two batches can be diffed directly.
  runaway.json                 the c0 trajectory on the FIXED probe batch. c0 is
                               a learnable softness knob and the documented
                               failure mode of this model is that the fit ejects
                               the contact port so the mass head can explain the
                               ground reaction, so the question "did c0 run away
                               to make contact vanish?" gets its own file:
                               c0 vs step, the taper it implies (Rtilde, the
                               equivalent stiffness and rest gap), and the
                               engagement/impulse series it would have to move.

Usage:
    python -m evaluations.contact_compliance_smoke_summary \
        --dir results/contact_compliance_smoke
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os

ARM_ORDER = (
    "learned_constraint",
    "kinematic_constraint",
    "kinematic_compliance",
    "kinematic_compliance_frozen",
)

CSV_COLS = (
    "arm", "contact_geometry", "contact_solver", "gate_off",
    "contact_compliance_c0_init", "freeze_c0",
    "accel_nrmse", "accel_corr", "mass_rel_frob_err",
    "contact_force_corr", "contact_force_nrmse", "contact_active_frac",
    "contact_solver_residual_mean",
    "loss_first10_mean", "loss_last100_mean",
    "in_contact_step0", "in_contact_step500", "in_contact_step3000",
    "normal_impulse_mean_step0", "normal_impulse_mean_step3000",
    "impulse_decay_ratio_0_to_end",
    "c0_mean_step0", "c0_mean_step_end", "c0_ratio_to_init_end",
    "fit_seconds",
)


def _at(trace, step):
    """The trace row at ``step``, or the nearest one at or below it."""
    best = None
    for row in trace:
        if row["step"] <= step and (best is None or row["step"] > best["step"]):
            best = row
    return best or trace[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results/contact_compliance_smoke")
    args = p.parse_args()

    reports = {}
    for path in sorted(glob.glob(os.path.join(args.dir, "*.json"))):
        name = os.path.basename(path)[:-5]
        if name in ("summary", "runaway", "aggregate"):
            continue
        with open(path) as f:
            reports[name] = json.load(f)
        reports[name]["_md5"] = hashlib.md5(
            open(path, "rb").read()).hexdigest()

    order = [a for a in ARM_ORDER if a in reports] + [
        a for a in sorted(reports) if a not in ARM_ORDER]

    rows = []
    runaway = {}
    for arm in order:
        r = reports[arm]
        cfg, trace, h = r["config"], r["contact_trace"], r["headline"]
        # contact_force_corr is a physical-recovery axis, not a headline one.
        phys = r["axes"]["physical_recovery"]
        end = trace[-1]
        first, mid = trace[0], _at(trace, 500)
        ni0, ni_end = first["normal_impulse_mean"], end["normal_impulse_mean"]
        rows.append({
            "arm": arm,
            "contact_geometry": cfg["contact_geometry"],
            "contact_solver": cfg["contact_solver"],
            "gate_off": cfg["contact_gate_off_resolved"],
            "contact_compliance_c0_init": cfg.get("contact_compliance"),
            "freeze_c0": cfg.get("freeze_c0"),
            "accel_nrmse": h.get("accel_nrmse"),
            "accel_corr": h.get("accel_corr"),
            "mass_rel_frob_err": h.get("mass_rel_frob_err"),
            "contact_force_corr": phys.get("contact_force_corr"),
            "contact_force_nrmse": phys.get("contact_force_nrmse"),
            "contact_active_frac": phys.get("contact_active_frac"),
            "contact_solver_residual_mean": phys.get(
                "contact_solver_residual_mean"),
            "loss_first10_mean": r["loss_first10_mean"],
            "loss_last100_mean": r["loss_last100_mean"],
            "in_contact_step0": first["in_contact_frac"],
            "in_contact_step500": mid["in_contact_frac"],
            "in_contact_step3000": end["in_contact_frac"],
            "normal_impulse_mean_step0": ni0,
            "normal_impulse_mean_step3000": ni_end,
            "impulse_decay_ratio_0_to_end": (ni0 / ni_end) if ni_end else None,
            "c0_mean_step0": first.get("c0_mean"),
            "c0_mean_step_end": end.get("c0_mean"),
            "c0_ratio_to_init_end": end.get("c0_ratio_to_init"),
            "fit_seconds": r["fit_seconds"],
        })

        keys = ("step", "c0_mean", "c0_min", "c0_max", "c0_ratio_to_init",
                "in_contact_frac", "in_contact_frac_any",
                "normal_impulse_mean", "gate_mean", "gate_nonzero_frac",
                "scale_mean", "mass_diag_mean", "mass_logdet_mean", "beta",
                "free_accel_rms", "contact_accel_rms",
                "Rtilde_nn_engaged_mean", "stiffness_N_per_m_engaged_mean",
                "stiffness_N_per_m_at_gate1",
                "implied_rest_gap_engaged_mean_m", "implied_rest_gap_max_m")
        rt0, rte = first.get("Rtilde_nn_engaged_mean"), end.get(
            "Rtilde_nn_engaged_mean")
        sc0, sce = first.get("scale_mean"), end.get("scale_mean")
        runaway[arm] = {
            "c0_init": cfg.get("contact_compliance"),
            "c0_floor": cfg.get("contact_compliance_floor"),
            "freeze_c0": cfg.get("freeze_c0"),
            "c0_per_contact_step0": first.get("c0"),
            "c0_per_contact_end": end.get("c0"),
            "c0_ratio_to_init_end": end.get("c0_ratio_to_init"),
            "c0_monotone_increasing": all(
                b.get("c0_mean", 0) >= a.get("c0_mean", 0) - 1e-12
                for a, b in zip(trace, trace[1:])),
            # Which channel actually softened the contact. Rtilde = scale
            # [c0 (1/s^2 - 1) + reg/s^2]: on exact kinematic geometry the gate s
            # is a function of position only, so on a FIXED probe batch it cannot
            # move at all, and the only two factors left are c0 and scale (the
            # learned mass gauge). Comparing their fold-changes attributes the
            # softening to one or the other with no further modelling.
            "Rtilde_fold_change": (rte / rt0) if rt0 else None,
            "scale_fold_change": (sce / sc0) if sc0 else None,
            "c0_fold_change": (
                end["c0_mean"] / first["c0_mean"]
                if first.get("c0_mean") else None),
            "gate_mean_fold_change": (
                end["gate_mean"] / first["gate_mean"]
                if first.get("gate_mean") else None),
            "normal_impulse_fold_change": (ni_end / ni0) if ni0 else None,
            "series": {k: [row.get(k) for row in trace] for k in keys},
        }

    # Per-arm-family aggregate. The 100k sweep this smoke previews measured a
    # seed spread of 0.507..0.695 accel_nrmse for ONE fixed config, so a single
    # seed cannot resolve a 0.02 difference between arms; the spread is reported
    # next to the means so the comparison is read with the right error bar.
    families = {}
    for row in rows:
        fam = row["arm"].rsplit("_s", 1)[0] if "_s" in row["arm"] else row["arm"]
        families.setdefault(fam, []).append(row)
    agg = {}
    metrics = ("accel_nrmse", "accel_corr", "mass_rel_frob_err",
               "contact_force_corr", "contact_force_nrmse",
               "contact_active_frac", "loss_last100_mean",
               "in_contact_step3000", "normal_impulse_mean_step3000",
               "impulse_decay_ratio_0_to_end", "c0_ratio_to_init_end")
    for fam, group in sorted(families.items()):
        entry = {"n_seeds": len(group),
                 "seeds": [g["arm"] for g in group]}
        for k in metrics:
            vals = [g[k] for g in group
                    if isinstance(g.get(k), (int, float))
                    and not isinstance(g.get(k), bool)]
            if not vals:
                entry[k] = None
                continue
            mean = sum(vals) / len(vals)
            var = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
                   if len(vals) > 1 else 0.0)
            entry[k] = {"mean": mean, "sd": var ** 0.5,
                        "min": min(vals), "max": max(vals), "n": len(vals)}
        agg[fam] = entry

    os.makedirs(args.dir, exist_ok=True)
    with open(os.path.join(args.dir, "aggregate.json"), "w") as f:
        json.dump(agg, f, indent=2)
    with open(os.path.join(args.dir, "summary.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_COLS))
        w.writeheader()
        for row in rows:
            w.writerow(row)
    with open(os.path.join(args.dir, "summary.json"), "w") as f:
        json.dump({
            "rows": rows,
            "aggregate": agg,
            "report_md5": {a: reports[a]["_md5"] for a in order},
            "prior_batch_for_comparison": "results/kinematic_contact_smoke/",
        }, f, indent=2)
    if runaway:
        # Which channel softened the contact, as a number rather than a claim.
        # Rtilde = scale [c0 (1/s^2 - 1) + reg/s^2] has exactly three factors, so
        # log(Rtilde fold) splits into log(scale fold) + log(c0 term fold) +
        # log(gate term fold). Reporting the residual makes the split checkable:
        # if scale explains it, the residual is ~0.
        for arm, d in runaway.items():
            rt, sc = d.get("Rtilde_fold_change"), d.get("scale_fold_change")
            d["softening_attribution"] = {
                "Rtilde_fold": rt,
                "scale_fold": sc,
                "c0_fold": d.get("c0_fold_change"),
                "gate_mean_fold": d.get("gate_mean_fold_change"),
                "Rtilde_over_scale": (rt / sc) if (rt and sc) else None,
                "note_units": "gate is a function of position only under exact "
                              "kinematic geometry, so on a FIXED probe batch it "
                              "cannot move; scale and c0 are the only free "
                              "factors in Rtilde.",
            }
        with open(os.path.join(args.dir, "runaway.json"), "w") as f:
            json.dump(runaway, f, indent=2)

    hdr = ("arm", "nrmse", "mass", "ccorr", "loss100", "in_c_end",
           "imp_end", "c0_end")
    print("%-30s %8s %8s %8s %10s %9s %11s %10s" % hdr)
    for row in rows:
        def _f(v, w=8, p=4):
            return ("%*.*f" % (w, p, v)) if isinstance(v, float) else "%*s" % (w, "-")
        print("%-30s %s %s %s %s %s %s %s" % (
            row["arm"], _f(row["accel_nrmse"]), _f(row["mass_rel_frob_err"]),
            _f(row["contact_force_corr"]), _f(row["loss_last100_mean"], 10, 5),
            _f(row["in_contact_step3000"], 9), _f(row["normal_impulse_mean_step3000"], 11, 7),
            _f(row["c0_mean_step_end"], 10, 3)))
    if any(e["n_seeds"] > 1 for e in agg.values()):
        print("\n== per-arm mean +- sd over seeds ==")
        print("%-32s %18s %18s %18s" % (
            "arm", "accel_nrmse", "mass_rel_frob", "contact_force_corr"))
        for fam, e in agg.items():
            def _c(k):
                v = e.get(k)
                return ("%9.4f +-%6.4f" % (v["mean"], v["sd"])) if v else "%17s" % "-"
            print("%-32s %18s %18s %18s" % (
                fam, _c("accel_nrmse"), _c("mass_rel_frob_err"),
                _c("contact_force_corr")))


if __name__ == "__main__":
    main()
