#!/usr/bin/env python
"""Aggregate evaluations.eval_reward_rate JSONs into a per-arm comparison.

``rate_executed`` averages reward rate over the episode as actually run;
``rate_horizon`` charges a cap-terminated episode for the 20 s horizon it
discarded by freezing the remainder at the reward's lower envelope.  The two
diverge only for arms that actually cap, so their gap is a direct read on how
much a family's score depends on episodes ending early.

    python -m evaluations.summarize_reward_rate
"""
from __future__ import annotations

import argparse, csv, glob, json, os, statistics as st

ENV_ID = "acrobot-swingup-xk"
CAP_CUR = "eval/strict_capture_success_rate"
CAP_BEST = "eval/best_strict_capture_success_rate"


def capture_stats(rec, log_root="logs"):
    """Capture for one rate-eval record, joined from its own training log.

    ``best`` reads the run's stored running max rather than recomputing it, so
    it stays correct for the SB3 arms whose progress.csv retains only the tail
    after a logger reconfiguration on resume.  ``mean_draw`` averages the
    per-evaluation draws that survive in the file, which for those same arms is
    a partial view -- hence the reported draw count.
    """
    pat = os.path.join(log_root, rec["algo"], ENV_ID, rec["mode"],
                       f"seed_{rec['seed']}", f"*{rec['run_id']}*", "progress.csv")
    hits = sorted(glob.glob(pat))
    if not hits:
        return None
    best, draws = None, []
    with open(hits[0], newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get(CAP_BEST) not in ("", None):
                try: best = float(row[CAP_BEST])
                except ValueError: pass
            if row.get(CAP_CUR) not in ("", None):
                try: draws.append(float(row[CAP_CUR]))
                except ValueError: pass
    return {"best": best, "draws": draws}

FAMILY_LABEL = {
    "ms10": "ct_sac 10ms uniform", "irr_old": "ct_sac irr tau1p25e2",
    "irr_tau": "ct_sac irr tau5e3", "min2ms": "ct_sac irr min2ms(5x)",
    "irr_old/ct_td3": "ct_td3 irr tau1p25e2", "irr_tau/ct_td3": "ct_td3 irr tau5e3",
    "min2ms/ct_td3": "ct_td3 irr min2ms(5x)",
    "irr_old/sac": "sac (discrete) irr", "irr_old/td3": "td3 (discrete) irr",
}
STEM = "xk_r3_eta0p23_ctrl10ms_h2s_temp0p01_xkdot_q2dot4pi_logrecip_"


def arm_label(mode):
    s = mode.replace(STEM, "").replace("_irregular1m", "").replace("_min2ms", "")
    for a, b in (("xkklrev_xkdemo20k_tau1p25e2_anneal0span60k", "+demo+KL->0"),
                 ("xkklrev_xkdemo20k_tau1p25e2_anneal0p5span60k", "+demo+KL->0.5"),
                 ("xkklrev_xkdemo20k_tau5e3_anneal0span60k", "+demo+KL->0"),
                 ("xkklrev_xkdemo20k_tau5e3_anneal0p5span60k", "+demo+KL->0.5"),
                 ("xkklrev_xkdemo50k_tau5e3_anneal0span150k", "+demo+KL->0"),
                 ("xkklrev_xkdemo50k_tau5e3_anneal0p5span150k", "+demo+KL->0.5"),
                 ("xkdemo20k_tau1p25e2", "+demo"), ("xkdemo20k_tau5e3", "+demo"),
                 ("xkdemo50k_tau5e3", "+demo"),
                 ("xkklrev_xkdemo40k_tau6e3_anneal0span120k", "+demo+KL->0"),
                 ("xkklrev_xkdemo40k_tau6e3_anneal0p5span120k", "+demo+KL->0.5"),
                 ("xkdemo40k_tau6e3", "+demo"),
                 ("tau6e3_ls20k", "baseline"),
                 ("tau5e3_ls25k", "baseline"), ("tau1p25e2", "baseline"), ("tau5e3", "baseline")):
        if s == a:
            return b
    return s


def fmt(v):
    return f"{st.fmean(v):+.4f}±{st.stdev(v):.4f}" if len(v) > 1 else (f"{v[0]:+.4f}" if v else "--")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results/reward_rate")
    args = ap.parse_args(argv)
    groups = {}
    for p in sorted(glob.glob(os.path.join(args.results_dir, "*.json"))):
        parts = os.path.basename(p).split("__")
        fam = parts[0]
        d = json.load(open(p))
        algo = d.get("algo", "ct_sac")
        key = fam if algo == "ct_sac" else f"{fam}/{algo}"
        groups.setdefault((key, arm_label(d["mode"])), []).append(d)
    if not groups:
        print("no results yet in", args.results_dir); return 1
    order = ["ms10", "irr_old", "irr_tau", "min2ms",
             "irr_old/ct_td3", "irr_tau/ct_td3", "min2ms/ct_td3",
             "irr_old/sac", "irr_old/td3"]
    arms = ["baseline", "+demo", "+demo+KL->0", "+demo+KL->0.5"]
    print(f"{'family':22s} {'arm':14s} {'n':>2s} {'rate_executed':>18s} "
          f"{'rate_horizon':>18s} {'cap_frac':>9s} {'ep_len':>7s} "
          f"{'capture':>14s} {'mean_draw':>14s} {'drw':>4s}")
    print("-" * 128)
    for fam in order:
        for arm in arms:
            g = groups.get((fam, arm))
            if not g:
                continue
            caps = [capture_stats(d) for d in g]
            best = [c["best"] for c in caps if c and c["best"] is not None]
            mdraw = [st.fmean(c["draws"]) for c in caps if c and c["draws"]]
            ndraw = [len(c["draws"]) for c in caps if c and c["draws"]]
            def cfmt(v):
                if not v: return "     --"
                return f"{st.fmean(v):.3f}" + (f"±{st.stdev(v):.3f}" if len(v) > 1 else "")
            print(f"{FAMILY_LABEL.get(fam, fam):22s} {arm:14s} {len(g):2d} "
                  f"{fmt([d['rate_executed_mean'] for d in g]):>18s} "
                  f"{fmt([d['rate_horizon_mean'] for d in g]):>18s} "
                  f"{st.fmean([d['cap_fraction'] for d in g]):9.3f} "
                  f"{st.fmean([d['mean_duration_s'] for d in g]):7.2f} "
                  f"{cfmt(best):>14s} {cfmt(mdraw):>14s} "
                  f"{(round(st.fmean(ndraw)) if ndraw else 0):4d}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
