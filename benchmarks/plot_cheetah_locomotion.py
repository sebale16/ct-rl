#!/usr/bin/env python
"""Cheetah locomotion metrics (energy efficiency + gait consistency) for the four
video-aligned policies. Small multiples: one bar panel per metric (scales differ,
so never a shared/dual axis). Colour = family (model-free blue, oracle vermillion);
hatch = 300k buffer (solid = 1M). Reads results/cheetah_locomotion/*.agg.json.

Output (tracked): results/cheetah_locomotion_metrics.png
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, VERM = "#0072B2", "#D55E00"     # Okabe-Ito, CVD-safe
INK, MUT = "#222222", "#666666"

# (mode key, short label, colour, is_300k_buffer)
MODES = [
    ("top", "mf 300k", BLUE, True),
    ("top_buf1m", "mf 1M", BLUE, False),
    ("mbq_vhead_quad", "orc 300k", VERM, True),
    ("mbq_vhead_quad_buf1m", "orc 1M", VERM, False),
]
# (metric key, title, value fmt, lower_is_better)
PANELS = [
    ("return", "Return", "{:.0f}", False),
    ("mean_speed_mps", "Speed [m/s]", "{:.2f}", False),
    ("cost_of_transport", "Cost of transport", "{:.2f}", True),
    ("return_per_joule", "Return / joule", "{:.3f}", False),
    ("autocorr_peak", "Periodicity (autocorr)", "{:.2f}", False),
    ("stride_cv", "Stride-period CV", "{:.3f}", True),
    ("limb_plv", "Front/back phase-lock", "{:.2f}", False),
    ("peak_power_frac_mean", "Fundamental power frac", "{:.2f}", False),
]

agg = {}
for m, *_ in MODES:
    p = f"results/cheetah_locomotion/cheetah_{m}_locomotion.agg.json"
    agg[m] = json.load(open(p))

labels = [lbl for _, lbl, _, _ in MODES]
colors = [c for _, _, c, _ in MODES]
hatches = ["//" if is300 else "" for _, _, _, is300 in MODES]
x = np.arange(len(MODES))

fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), dpi=150)
for ax, (key, title, fmt, lower) in zip(axes.ravel(), PANELS):
    means = np.array([agg[m].get(f"{key}_mean", np.nan) for m, *_ in MODES])
    stds = np.array([agg[m].get(f"{key}_std", 0.0) for m, *_ in MODES])
    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="white",
                  linewidth=1.5, error_kw=dict(ecolor=MUT, lw=1))
    for b, h in zip(bars, hatches):
        if h:
            b.set_hatch(h)
    for xi, mv in zip(x, means):
        ax.text(xi, mv + (stds.max() * 0.15 if stds.max() else abs(mv) * 0.02),
                fmt.format(mv), ha="center", va="bottom", fontsize=8.5,
                fontweight="bold", color=INK)
    arrow = " ↓" if lower else " ↑"
    ax.set_title(title + arrow, fontsize=10.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, color=INK)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(y=0.18)

# legend: colour = family, hatch = 300k buffer
from matplotlib.patches import Patch
handles = [
    Patch(facecolor=BLUE, edgecolor="white", label="model-free"),
    Patch(facecolor=VERM, edgecolor="white", label="oracle (MuJoCo in-loop)"),
    Patch(facecolor="#cccccc", edgecolor="white", hatch="//", label="300k buffer (solid = 1M)"),
]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
           fontsize=10, bbox_to_anchor=(0.5, 0.99))
fig.suptitle("cheetah-run locomotion: energy efficiency & gait consistency  "
             "(video-aligned best policies, 8 episodes)\n"
             "↑ higher is better · ↓ lower is better", fontsize=12.5, y=1.06)
fig.tight_layout(rect=[0, 0, 1, 0.95])
os.makedirs("results", exist_ok=True)
out = "results/cheetah_locomotion_metrics.png"
fig.savefig(out, bbox_inches="tight")
print("saved", out)
