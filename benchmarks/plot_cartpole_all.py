#!/usr/bin/env python
"""Cartpole-swingup learning curves across the three dynamics modes, mean +/- std
over 12 seeds. Colour = family: model-free (blue), oracle-in-loop (vermillion),
learned port-Hamiltonian in-loop (green). Reads each cell's evaluations.npz.

Outputs (tracked, under results/):
  results/cartpole_modes_500k.png
"""
import glob
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAP = 500_000
BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"   # Okabe-Ito, CVD-safe
# mode -> (label, colour)
MODES = {
    "top":                      ("model-free",              BLUE),
    "mbq_vhead_quad":           ("oracle (MuJoCo in-loop)", VERM),
    "mbq_structured_quad_roll": ("learned port-Hamiltonian", GREEN),
}


def curve(mode):
    by_step = defaultdict(list)
    for f in glob.glob(
        f"logs/ct_sac/cartpole-swingup/{mode}/seed_*/*cforce_grid_chain*/eval/evaluations.npz"
    ):
        try:
            d = np.load(f, allow_pickle=True)
            for t, res in zip(d["timesteps"], d["results"]):
                if int(t) <= CAP:
                    by_step[int(t)].append(float(np.mean(res)))
        except Exception:
            pass
    steps = sorted(by_step)
    nmax = max((len(by_step[s]) for s in steps), default=0)
    steps = [s for s in steps if len(by_step[s]) == nmax]   # only fully-populated steps
    m = np.array([np.mean(by_step[s]) for s in steps])
    sd = np.array([np.std(by_step[s]) for s in steps])
    return np.array(steps), m, sd, nmax


fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
end_labels = []
for mode, (label, color) in MODES.items():
    x, m, sd, n = curve(mode)
    if len(x) == 0:
        continue
    ax.fill_between(x / 1e3, m - sd, m + sd, color=color, alpha=0.12, linewidth=0)
    ax.plot(x / 1e3, m, color=color, lw=2, label=f"{label}  (n={n})")
    end_labels.append([x[-1] / 1e3, float(m[-1]), color, f"{m[-1]:.0f}"])
    print(f"{mode:26s} n={n} final@{int(x[-1])}={m[-1]:.1f}±{sd[-1]:.1f} peak={m.max():.1f}")

# spread near-equal end labels vertically
MIN_GAP = 90.0
end_labels.sort(key=lambda e: e[1])
placed = []
for e in end_labels:
    y = e[1]
    if placed and y - placed[-1] < MIN_GAP:
        y = placed[-1] + MIN_GAP
    placed.append(y)
    e.append(y)
for x_end, y_true, color, text, y_lab in end_labels:
    if abs(y_lab - y_true) >= 1:
        ax.plot([x_end, x_end + 8], [y_true, y_lab], color=color, lw=0.6, alpha=0.7,
                clip_on=False)
    ax.text(x_end + 11, y_lab, text, color=color, fontsize=9, fontweight="bold",
            va="center", ha="left", clip_on=False)

ax.set_xlabel("Environment steps (thousands)")
ax.set_ylabel("Evaluation return (10-episode mean)")
ax.set_title("cartpole-swingup: model-free vs oracle vs learned dynamics\n"
             "mean ± std over 12 seeds, to 500k steps")
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(0, CAP / 1e3)
ax.set_ylim(bottom=0)
ax.legend(loc="lower right", frameon=False, title="colour = family")
fig.tight_layout()
os.makedirs("results", exist_ok=True)
out = "results/cartpole_modes_500k.png"
fig.savefig(out, bbox_inches="tight")
print("saved", out)
