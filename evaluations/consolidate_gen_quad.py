#!/usr/bin/env python
"""Consolidate the final-aligned control-relevant recovery reports.

For each learned-cartpole seed's report (produced by hamiltonian_recovery on the
FINAL checkpoint, on-policy distribution), pull the generator_report and
quadrature_report metrics -- for BOTH the on-policy data actions and the
policy-sampled candidate actions -- and join the seed's FINAL evaluation return.

    python -m evaluations.consolidate_gen_quad \
        out/recovery/final_aligned/cartpole_seed*_final.json \
        --out results/cartpole_final_aligned_gen_quad

Writes <out>.csv and <out>.json and prints a summary + return-vs-metric
correlations.
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np

MODE = ("cartpole-swingup", "mbq_structured_quad_roll")


def final_return(seed):
    vals = []
    for f in glob.glob(
        f"logs/ct_sac/{MODE[0]}/{MODE[1]}/seed_{seed}/*cforce_grid_chain*/eval/evaluations.npz"
    ):
        try:
            d = np.load(f, allow_pickle=True)
            vals.append((int(d["timesteps"][-1]), float(np.mean(d["results"][-1]))))
        except Exception:
            pass
    if not vals:
        return None
    vals.sort()
    return vals[-1][1]


def seed_of(path):
    m = re.search(r"seed(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


# (output column, path within datasets["policy"])
COLS = [
    ("gen_data_rmse", ("generator", "data_actions", "rmse")),
    ("gen_data_corr", ("generator", "data_actions", "corr")),
    ("gen_data_nrmse", ("generator", "data_actions", "nrmse")),
    ("gen_data_bias", ("generator", "data_actions", "bias")),
    ("gen_data_p95", ("generator", "data_actions", "p95")),
    ("gen_pi_rmse", ("generator", "policy_actions", "rmse")),
    ("gen_pi_corr", ("generator", "policy_actions", "corr")),
    ("gen_pi_nrmse", ("generator", "policy_actions", "nrmse")),
    ("gen_pi_bias", ("generator", "policy_actions", "bias")),
    ("gen_pi_p95", ("generator", "policy_actions", "p95")),
    ("gen_err_vs_action_novelty_corr", ("generator", "err_vs_action_novelty_corr")),
    ("quad_rmse", ("quadrature", "rmse")),
    ("quad_corr", ("quadrature", "corr")),
    ("quad_nrmse", ("quadrature", "nrmse")),
    ("quad_bias", ("quadrature", "bias")),
    ("quad_sign_disagree_frac", ("quadrature", "sign_disagree_frac")),
    ("quad_p95", ("quadrature", "p95")),
    ("quad_endpoint_state_nrmse", ("quadrature", "endpoint_state_nrmse")),
]


def dig(d, path):
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--out", required=True, help="output path prefix (.csv/.json)")
    args = ap.parse_args()

    rows = []
    for path in sorted(args.reports, key=lambda p: (seed_of(p) if seed_of(p) is not None else 999)):
        try:
            rep = json.load(open(path))
        except Exception as e:
            print(f"skip {path}: {e}")
            continue
        s = seed_of(path)
        pol = rep.get("datasets", {}).get("policy", {})
        if not pol:
            print(f"skip {path}: no 'policy' dataset (checkpoint had no V-head?)")
            continue
        row = {"seed": s, "final_return": final_return(s)}
        for name, p in COLS:
            row[name] = dig(pol, p)
        rows.append(row)

    if not rows:
        print("no valid reports")
        return

    cols = ["seed", "final_return"] + [c for c, _ in COLS]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})
    json.dump(rows, open(args.out + ".json", "w"), indent=2)

    print(f"\nwrote {args.out}.csv / .json  ({len(rows)} seeds)")
    R = np.array([r["final_return"] for r in rows], float)
    print(f"\nfinal return: median={np.median(R):.1f}  "
          f"IQR={np.percentile(R,75)-np.percentile(R,25):.1f}  "
          f"<500={int((R<500).sum())}  <1000={int((R<1000).sum())}")

    # print the two-distribution generator comparison + quadrature, per seed
    print("\nseed  return | gen_data(corr/rmse)  gen_pi(corr/rmse)  novelty | "
          "quad(corr/nrmse/sign-dis)  quad_endpt_nrmse")
    for r in rows:
        g = lambda k: (f"{r[k]:.3f}" if isinstance(r.get(k), (int, float)) else "  - ")
        print(f"{r['seed']:>4d}  {r['final_return']:6.0f} | "
              f"{g('gen_data_corr')}/{g('gen_data_rmse')}   "
              f"{g('gen_pi_corr')}/{g('gen_pi_rmse')}   {g('gen_err_vs_action_novelty_corr')} | "
              f"{g('quad_corr')}/{g('quad_nrmse')}/{g('quad_sign_disagree_frac')}   "
              f"{g('quad_endpoint_state_nrmse')}")

    # return-vs-metric correlations (Pearson), where the metric is finite for all
    print("\nreturn vs metric (Pearson r over seeds):")
    for name, _ in COLS:
        x = np.array([r.get(name) for r in rows], float)
        msk = np.isfinite(x) & np.isfinite(R)
        if msk.sum() >= 3 and np.std(x[msk]) > 0:
            r_ = np.corrcoef(x[msk], R[msk])[0, 1]
            print(f"  {name:32s} r={r_:+.3f}  (n={int(msk.sum())})")


if __name__ == "__main__":
    main()
