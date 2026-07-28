"""Render the planar Coulomb cone diagram for docs/contact_impulse_qp.md."""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
BLUE = "#2a78d6"

MU = 0.5
XMAX = 3.8

fig, ax = plt.subplots(figsize=(7.2, 4.1), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.set_aspect("equal")

# feasible wedge |Lambda_t| <= mu * Lambda_n, Lambda_n >= 0
ax.fill([0, XMAX, XMAX], [0, MU * XMAX, -MU * XMAX],
        color=BLUE, alpha=0.14, lw=0, zorder=1)

# saturation boundaries
ax.plot([0, XMAX], [0, MU * XMAX], color=BLUE, lw=2, zorder=3,
        solid_capstyle="round")
ax.plot([0, XMAX], [0, -MU * XMAX], color=BLUE, lw=2, zorder=3,
        solid_capstyle="round")

# axes as arrows through the origin
arrow = dict(arrowstyle="-|>", color=INK2, lw=1.2, shrinkA=0, shrinkB=0,
             mutation_scale=14)
ax.annotate("", xy=(4.25, 0), xytext=(-0.55, 0), arrowprops=arrow, zorder=2)
ax.annotate("", xy=(0, 2.25), xytext=(0, -2.25), arrowprops=arrow, zorder=2)
ax.text(4.25, -0.16, r"$\Lambda_n$", color=INK2, fontsize=13,
        ha="right", va="top")
ax.text(-0.12, 2.22, r"$\Lambda_t$", color=INK2, fontsize=13,
        ha="right", va="top")

# apex marker + label
ax.plot([0], [0], "o", ms=7, color=BLUE, zorder=4)
ax.annotate("apex: zero impulse",
            xy=(-0.04, 0.07), xytext=(-1.4, 0.95),
            color=INK, fontsize=11, ha="left",
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9,
                            shrinkA=2, shrinkB=4,
                            connectionstyle="arc3,rad=-0.18"))

# boundary labels, rotated along the rays (aspect equal -> true slope)
ang = np.degrees(np.arctan(MU))
ax.text(2.1, MU * 2.1 + 0.14, r"$\Lambda_t=+\mu\,\Lambda_n$  (friction saturated)",
        color=INK, fontsize=11, rotation=ang,
        rotation_mode="anchor", ha="center", va="bottom")
ax.text(2.1, -MU * 2.1 - 0.14, r"$\Lambda_t=-\mu\,\Lambda_n$  (friction saturated)",
        color=INK, fontsize=11, rotation=-ang,
        rotation_mode="anchor", ha="center", va="top")

# region labels
ax.text(2.8, 0.62, "feasible impulses\n" + r"$\mathcal{C}_\mu$",
        color=INK, fontsize=12, ha="center", va="center")
ax.text(0.8, 1.85, "infeasible:\n" + r"$|\Lambda_t|>\mu\,\Lambda_n$",
        color=INK2, fontsize=10, ha="center", va="center")
ax.text(-0.75, -1.25, "infeasible:\n" + r"$\Lambda_n<0$",
        color=INK2, fontsize=10, ha="center", va="center")

ax.set_xlim(-1.5, 4.35)
ax.set_ylim(-2.35, 2.45)
ax.axis("off")
fig.tight_layout(pad=0.3)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coulomb_cone.svg")
fig.savefig(out, facecolor=SURFACE)
print("wrote", out)
