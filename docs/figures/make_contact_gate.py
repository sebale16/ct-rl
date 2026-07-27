"""Render the quintic contact-gate diagram for docs/contact_impulse_qp.md."""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8983"
BLUE = "#2a78d6"

G_ON, G_OFF = 0.0, 0.06
WIDTH = G_OFF - G_ON


def u_of(g):
    return np.clip((G_OFF - g) / WIDTH, 0.0, 1.0)


def quintic(u):
    return u ** 3 * (10.0 - 15.0 * u + 6.0 * u ** 2)


def cubic(u):
    return u ** 2 * (3.0 - 2.0 * u)


# second derivatives in u; du/dg = -1/WIDTH so d2/dg2 = s''(u)/WIDTH**2
def quintic_dd(u):
    return 60.0 * u * (2.0 * u - 1.0) * (u - 1.0)


def cubic_dd(u):
    return 6.0 - 12.0 * u


fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(10.4, 3.9), facecolor=SURFACE,
    gridspec_kw={"width_ratios": [1.32, 1.0]})

g = np.linspace(-0.022, 0.082, 1400)
u = u_of(g)
band = (g > G_ON) & (g < G_OFF)

for ax in (axA, axB):
    ax.set_facecolor(SURFACE)
    ax.axvspan(G_ON, G_OFF, color=BLUE, alpha=0.07, lw=0, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9, length=3, width=0.8)
    ax.set_xlim(-0.022, 0.082)
    ax.set_xticks([G_ON, G_OFF])
    ax.set_xticklabels([r"$g_{\mathrm{on}}$", r"$g_{\mathrm{off}}$"], fontsize=11)
    ax.set_xlabel(r"learned gap $g_i(q)$", color=INK2, fontsize=10.5)

# ---- panel A: the gate itself -------------------------------------------
axA.plot(g, u, color=MUTED, lw=1.3, ls=":", zorder=2)
axA.plot(g, cubic(u), color=MUTED, lw=1.3, ls="--", zorder=2)
axA.plot(g, quintic(u), color=BLUE, lw=2.2, zorder=4, solid_capstyle="round")

axA.set_ylim(-0.13, 1.2)
axA.set_yticks([0, 0.5, 1])
axA.set_ylabel(r"gate $s_i$", color=INK2, fontsize=10.5)
axA.set_title("the gate over the transition band",
              color=INK, fontsize=11.5, pad=10, loc="left")

axA.text(-0.019, 1.06, "fully enabled\n" + r"$s_i=1$", color=INK2,
         fontsize=9.5, ha="left", va="center")
axA.text(0.079, 0.14, "exactly disabled\n" + r"$s_i=0$", color=INK2,
         fontsize=9.5, ha="right", va="center")
axA.annotate("quintic  " + r"$6u^5-15u^4+10u^3$",
             xy=(0.030, 0.5), xytext=(0.036, 0.86),
             color=BLUE, fontsize=10.5, ha="left",
             arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.9,
                             shrinkA=2, shrinkB=3,
                             connectionstyle="arc3,rad=-0.25"))
axA.annotate("cubic", xy=(0.0175, cubic(u_of(0.0175))), xytext=(0.0035, 0.30),
             color=INK2, fontsize=9.5, ha="left",
             arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                             shrinkA=2, shrinkB=3))
axA.annotate("linear clip", xy=(0.0425, u_of(0.0425)), xytext=(0.047, 0.52),
             color=INK2, fontsize=9.5, ha="left",
             arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                             shrinkA=2, shrinkB=3))

# ---- panel B: second derivative ------------------------------------------
qdd = np.where(band, quintic_dd(u) / WIDTH ** 2, 0.0)
cdd = np.where(band, cubic_dd(u) / WIDTH ** 2, 0.0)

axB.axhline(0, color=MUTED, lw=0.8, zorder=1)
axB.plot(g, cdd, color=MUTED, lw=1.3, ls="--", zorder=2)
axB.plot(g, qdd, color=BLUE, lw=2.2, zorder=4, solid_capstyle="round")

# the cubic's jumps at the two joins
for gj, val in ((G_ON, cubic_dd(1.0) / WIDTH ** 2), (G_OFF, cubic_dd(0.0) / WIDTH ** 2)):
    axB.plot([gj, gj], [0, val], color=MUTED, lw=1.0, ls=(0, (1, 2)), zorder=3)
    axB.plot([gj], [val], "o", ms=5, mfc=SURFACE, mec=MUTED, mew=1.2, zorder=5)
axB.plot([G_ON, G_OFF], [0, 0], "o", ms=5.5, color=BLUE, zorder=6)

axB.set_ylim(-2450, 2450)
axB.set_yticks([0])
axB.set_ylabel(r"$d^2s_i/dg^2$", color=INK2, fontsize=10.5)
axB.set_title("curvature at the joins", color=INK, fontsize=11.5, pad=10, loc="left")

axB.annotate("quintic returns to zero\nat both ends: " + r"$C^2$",
             xy=(G_OFF, 0), xytext=(0.064, -1500),
             color=BLUE, fontsize=9.5, ha="center", va="top",
             arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.9,
                             shrinkA=3, shrinkB=4,
                             connectionstyle="arc3,rad=0.25"))
axB.annotate("cubic jumps: only " + r"$C^1$",
             xy=(G_ON, cubic_dd(1.0) / WIDTH ** 2), xytext=(0.004, -2050),
             color=INK2, fontsize=9.5, ha="left", va="center",
             arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                             shrinkA=3, shrinkB=4,
                             connectionstyle="arc3,rad=-0.3"))

fig.tight_layout(pad=0.6, w_pad=2.4)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "contact_gate.svg")
fig.savefig(out, facecolor=SURFACE)
fig.savefig("/tmp/claude-1000/-home-seb-Documents-bajaj-code-ct-rl/756a26f6-226b-4f4e-9fda-9444271b3f63/scratchpad/contact_gate.png",
            dpi=160, facecolor=SURFACE)
print("wrote", out)
