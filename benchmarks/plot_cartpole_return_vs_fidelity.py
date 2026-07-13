#!/usr/bin/env python
"""Learned-dynamics cartpole: control return vs model-fidelity metrics.

Joins each of the 12 learned-cartpole seeds (mode ``mbq_structured_quad_roll``)
final evaluation return to its row in the Hamiltonian-recovery audit
(``out/recovery/cartpole_recovery_consolidated.csv``) and scatters return against
four fidelity metrics: acceleration NRMSE, rollout error at H4 and H8, and the
full drift NRMSE. Highlights seed1 (final return ~88, the single control
failure) as a diamond.

Finding: return is *positively* correlated with one-step/rollout error
(r ~ +0.70) -- higher-return policies visit a richer state distribution that is
harder to fit, so model fidelity does not predict control quality here. The
degenerate residual (force) NRMSE (~2e-5 for every seed) is deliberately
replaced by drift NRMSE, which carries the same signal as acceleration NRMSE.

Outputs (tracked, under results/):
  results/cartpole_learned_return_vs_fidelity.png
  results/cartpole_return_vs_fidelity.csv
"""
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

MODE = ("cartpole-swingup", "mbq_structured_quad_roll")
RECOVERY_CSV = "out/recovery/cartpole_recovery_consolidated.csv"
BLUE, VERM, GRAY = "#0072B2", "#D55E00", "#999999"   # Okabe-Ito, CVD-safe
INK, MUT = "#222222", "#666666"

# Return on y; each panel is a fidelity metric on x. Panel 4 is drift NRMSE
# (the residual/force NRMSE is ~2e-5 for all seeds and carries no signal).
PANELS = [
    ("accel_nrmse", "Acceleration NRMSE"),
    ("rollout_rel_err_H4", "Rollout error, H4"),
    ("rollout_rel_err_H8", "Rollout error, H8"),
    ("drift_nrmse", "Drift NRMSE"),
]


def per_seed_final_return():
    ret = {}
    for f in glob.glob(
        f"logs/ct_sac/{MODE[0]}/{MODE[1]}/seed_*/*cforce_grid_chain*/eval/evaluations.npz"
    ):
        s = int(f.split("/seed_")[1].split("/")[0])
        d = np.load(f, allow_pickle=True)
        ret[s] = float(np.mean(d["results"][-1]))
    return ret


def main():
    ret = per_seed_final_return()
    rows = {r["label"]: r for r in csv.DictReader(open(RECOVERY_CSV))}
    seeds = sorted(ret)
    R = np.array([ret[s] for s in seeds])
    col = lambda k: np.array([float(rows[f"cartpole_seed{s}"][k]) for s in seeds])

    # --- data dump (tracked) ---
    os.makedirs("results", exist_ok=True)
    with open("results/cartpole_return_vs_fidelity.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "final_return"] + [k for k, _ in PANELS])
        for i, s in enumerate(seeds):
            w.writerow([s, f"{R[i]:.2f}"] + [f"{col(k)[i]:.6g}" for k, _ in PANELS])

    # --- summary stats ---
    q1, med, q3 = np.percentile(R, [25, 50, 75])
    print(f"n={len(R)}  median={med:.1f}  IQR={q3 - q1:.1f} (Q1={q1:.1f}, Q3={q3:.1f})")
    print(f"count<500={int((R < 500).sum())}  count<1000={int((R < 1000).sum())}")
    seed88 = seeds[int(np.argmin(np.abs(R - 88)))]
    print(f"return~88 -> seed{seed88} ({R[seeds.index(seed88)]:.1f})")

    # --- figure ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), dpi=150)
    for ax, (k, lbl) in zip(axes.ravel(), PANELS):
        x = col(k)
        r = pearsonr(x, R)[0]
        b, a = np.polyfit(x, R, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, a + b * xs, color=GRAY, lw=1.5, ls="--", alpha=0.8, zorder=1)
        is88 = np.isclose(R, R[seeds.index(seed88)], atol=2)
        ax.scatter(x[~is88], R[~is88], s=80, color=BLUE, edgecolor="white", lw=1.0, zorder=3)
        ax.scatter(x[is88], R[is88], s=140, color=VERM, edgecolor="white", lw=1.2,
                   zorder=4, marker="D")
        for xi, ri, s in zip(x, R, seeds):
            ax.annotate(f"s{s}", (xi, ri), textcoords="offset points", xytext=(6, 3),
                        fontsize=7.5, color=MUT, zorder=5)
        ax.set_xlabel(lbl, fontsize=10, color=INK)
        ax.set_ylabel("Final return (10-ep mean)", fontsize=10, color=INK)
        ax.set_title(f"{lbl}   (Pearson r = {r:+.2f})", fontsize=10.5, color=INK)
        ax.grid(axis="both", alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axhline(500, color=GRAY, lw=0.7, alpha=0.5)
        ax.axhline(1000, color=GRAY, lw=0.7, alpha=0.5)

    fig.suptitle(
        "Learned-dynamics cartpole: control return vs model-fidelity metrics\n"
        f"12 seeds, final return joined to recovery-audit row; ◆ = seed{seed88} "
        "(return 88)",
        fontsize=12, y=0.99)
    fig.text(0.5, 0.005,
             "Positive slope: seeds with WORSE one-step/rollout fit reach HIGHER return "
             "(richer state visitation → harder audit). Gray lines: return=500 / 1000.",
             ha="center", fontsize=8.5, color=MUT)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    out = "results/cartpole_learned_return_vs_fidelity.png"
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
