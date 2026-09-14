#!/usr/bin/env python
"""Consolidate the advantage-parameterization audits into claim-by-claim verdicts.

Reads whatever ``evaluations.action_error_target_audit`` and
``evaluations.one_step_advantage_vs_dt`` wrote into ``results/`` and prints:

  1. per-arm acrobot-XK ledger at each run's own control interval, aggregated
     over seeds (mean +- sd across seeds, so seed spread is visible);
  2. the dt sweep with one fixed critic, plus the log-log slope of the action
     signal against the control interval -- the parameterization claim predicts
     slope 1 for ``range_y`` and slope 0 for the rate ``range_y / T``;
  3. the same slopes from the network-free rollout ground truth;
  4. the cross-environment ledger at each environment's single dt.

    python -m evaluations.summarize_advantage_validation
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np

RESULTS = "results"


def read(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if v in ("", "None"):
                r[k] = None
                continue
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows


def agg(rows, key):
    vals = np.array([r[key] for r in rows if isinstance(r.get(key), float)])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(vals.mean()), float(vals.std(ddof=1) if vals.size > 1 else 0.0)


def loglog_slope(x, y):
    """Slope of log|y| against log x -- the exponent of the power law in dt."""
    x = np.asarray(x, dtype=float)
    y = np.abs(np.asarray(y, dtype=float))
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])


def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def per_arm_table(rows, dists):
    hdr = (
        f"{'arm':24s} {'dist':9s} {'n':>3s} {'T':>7s} {'range_y':>16s} "
        f"{'rate=range_y/T':>15s} {'range_Q':>9s} {'shape err':>10s} {'snr':>7s} "
        f"{'rho':>6s} {'V_range':>8s}"
    )
    print(hdr)
    arms = []
    for r in rows:
        if r["arm"] not in arms:
            arms.append(r["arm"])
    for arm in arms:
        for dist in dists:
            sel = [r for r in rows if r["arm"] == arm and r["dist"] == dist]
            if not sel:
                continue
            ry, ry_sd = agg(sel, "range_y")
            rq, _ = agg(sel, "range_qv")
            rQ, _ = agg(sel, "range_q")
            er, _ = agg(sel, "rms_shape_err")
            sn, sn_sd = agg(sel, "snr")
            rho, _ = agg(sel, "spearman")
            vr, _ = agg(sel, "V_range")
            T = sel[0]["T"]
            print(
                f"{arm:24s} {dist:9s} {len(sel):3d} {T:7.4f} "
                f"{ry:9.5f}+-{ry_sd:<5.4f} {rq:15.2f} {rQ:9.5f} {er:10.5f} "
                f"{sn:7.2f} {rho:+6.2f} {vr:8.3f}"
            )


def delta_table(rows, dists):
    print(
        f"\n{'arm':24s} {'dist':9s} {'delta_y':>12s} {'|delta_y|':>11s} "
        f"{'delta/T':>10s} {'|t|':>6s} {'neg frac':>9s}"
    )
    arms = []
    for r in rows:
        if r["arm"] not in arms:
            arms.append(r["arm"])
    for arm in arms:
        for dist in dists:
            sel = [r for r in rows if r["arm"] == arm and r["dist"] == dist]
            if not sel:
                continue
            d, d_sd = agg(sel, "delta_y")
            ad, _ = agg(sel, "abs_delta_y")
            dr, _ = agg(sel, "delta_y_rate")
            t, _ = agg(sel, "delta_y_t")
            nf, _ = agg(sel, "delta_y_negative_frac")
            print(
                f"{arm:24s} {dist:9s} {d:+7.5f}+-{d_sd:<4.4f} {ad:11.5f} "
                f"{dr:+10.3f} {abs(t):6.2f} {nf:9.2f}"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RESULTS)
    args = ap.parse_args()
    R = args.results

    acro = read(f"{R}/action_error_target_audit_acrobot-swingup-xk.csv")
    if acro:
        dists = []
        for r in acro:
            if r["dist"] not in dists:
                dists.append(r["dist"])
        section("1. acrobot-XK -- each arm at its OWN control interval")
        per_arm_table(acro, dists)
        print(
            "\n   range_y  = action-range of the critic TARGET (the signal the "
            "critic must store)\n"
            "   rate     = the same signal as an advantage rate; the claim is "
            "that this is dt-invariant\n"
            "   shape err= RMS error of the trained critic in representing the "
            "action dependence\n"
            "   snr      = range_y / shape err; below 1 means the signal is "
            "under the critic's own noise"
        )
        section("1b. cost of a 5.8 N.m action error, through the trained value")
        delta_table(acro, dists)

    sweep = read(f"{R}/action_error_target_audit_acrobot-swingup-xk_dtsweep.csv")
    if sweep:
        section("2. dt sweep -- ONE trained critic, target re-formed at each interval")
        arms = []
        for r in sweep:
            if r["arm"] not in arms:
                arms.append(r["arm"])
        Ts = sorted({r["T"] for r in sweep})
        print(f"{'arm':24s} " + "".join(f"{T*1000:>11.0f}ms" for T in Ts) + "   slope")
        for label, key in (("range_y", "range_y"), ("rate range_y/T", "range_qv")):
            print(f"-- {label}")
            for arm in arms:
                vals = []
                for T in Ts:
                    sel = [r for r in sweep if r["arm"] == arm and r["T"] == T]
                    vals.append(agg(sel, key)[0] if sel else float("nan"))
                slope = loglog_slope(Ts, vals)
                print(
                    f"{arm:24s} "
                    + "".join(f"{v:13.4f}" for v in vals)
                    + f"   {slope:+.2f}"
                )
        print(
            "\n   Predicted by the advantage-parameterization claim: slope +1.00 "
            "for range_y,\n   slope 0.00 for the rate.  A flat rate is the "
            "claim that the continuous-time\n   object is well conditioned and "
            "only its storage inside Q is not."
        )

    truth = read(f"{R}/one_step_advantage_vs_dt.csv")
    grid = read(f"{R}/one_step_advantage_vs_dt_grid.csv")
    if truth:
        section("3. ground truth by exact rollout -- no networks")
        print(
            f"{'dt (ms)':>8s} {'n':>4s} {'mean A':>11s} {'sd':>9s} {'|t|':>6s} "
            f"{'95% CI':>24s} {'A/dt':>10s} {'pos':>9s}"
        )
        for r in truth:
            print(
                f"{r['dt']*1000:8.1f} {int(r['n']):4d} {r['mean']:+11.5f} "
                f"{r['sd']:9.5f} {abs(r['t']):6.2f} "
                f"[{r['ci_lo']:+.5f},{r['ci_hi']:+.5f}] {r['rate']:10.3f} "
                f"{int(r['positive']):4d}/{int(r['n'])}"
            )
        print(
            f"\n   log-log slope of |mean A| vs dt: "
            f"{loglog_slope([r['dt'] for r in truth], [r['mean'] for r in truth]):+.2f}"
            f"   (of the rate A/dt: "
            f"{loglog_slope([r['dt'] for r in truth], [r['rate'] for r in truth]):+.2f})"
        )
    if grid and "range_q_true" in grid[0]:
        print(
            f"\n   TRUE action-range of Q^pi over the full torque range "
            f"(n={int(grid[0]['n'])} states):"
        )
        print(
            f"{'dt (ms)':>8s} {'range Q^pi':>12s} {'sd':>9s} {'rate=range/dt':>14s}"
        )
        for r in grid:
            print(
                f"{r['dt']*1000:8.1f} {r['range_q_true']:12.5f} "
                f"{r['range_q_true_sd']:9.5f} {r['range_qv_true']:14.3f}"
            )
        print(
            f"   log-log slope of range Q^pi vs dt: "
            f"{loglog_slope([r['dt'] for r in grid], [r['range_q_true'] for r in grid]):+.2f}"
            f"   (of the rate: "
            f"{loglog_slope([r['dt'] for r in grid], [r['range_qv_true'] for r in grid]):+.2f})"
        )

    section("4. other environments, at their single control interval")
    printed = False
    for path in sorted(glob.glob(f"{R}/action_error_target_audit_*.csv")):
        if "acrobot" in path or "dtsweep" in path:
            continue
        rows = read(path)
        if not rows:
            continue
        printed = True
        print(f"\n-- {rows[0]['env_id']}")
        dists = []
        for r in rows:
            if r["dist"] not in dists:
                dists.append(r["dist"])
        per_arm_table(rows, dists)
        delta_table(rows, dists)
    if not printed:
        print("(none found)")


if __name__ == "__main__":
    main()
