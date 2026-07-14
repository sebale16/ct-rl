import glob, csv, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SUB = "logs/ct_sac/cartpole-swingup/mbq_structured_quad_roll"
CRITIC, VNEXT, ROLLB = "train/critic_loss", "train/model_value_next_max_abs", "train/dynamics_rollbacks"


def tb_series(seed, tag):
    xs, ys = [], []
    for f in glob.glob(f"{SUB}/seed_{seed}/*/events.out.tfevents.*"):
        try:
            ea = EventAccumulator(f, size_guidance={'scalars': 0}); ea.Reload()
            if tag in ea.Tags().get('scalars', []):
                for e in ea.Scalars(tag):
                    xs.append(e.step); ys.append(e.value)
        except Exception:
            pass
    if not xs:
        return np.array([]), np.array([])
    o = np.argsort(xs)
    return np.array(xs)[o], np.array(ys)[o]


rows = list(csv.DictReader(open("results/cartpole_target_audit_trajectory.csv")))


def audit(seed, dist="onpolicy"):
    sub = sorted([r for r in rows if int(r["seed"]) == seed and r["distribution"] == dist],
                 key=lambda r: int(r["step"]))
    return (np.array([int(r["step"]) for r in sub]),
            np.array([float(r["eps_T_rms"]) for r in sub]))


SEEDS = [0, 1, 6, 11]
# cache TB series
cache = {}
def cs(s, tag):
    cache.setdefault((s, tag), tb_series(s, tag))
    return cache[(s, tag)]

print("=== spike-alignment: audit eps_T>1 vs nearest train critic_loss / value_next ===")
for s in range(12):
    ax, ay = audit(s)
    cl_x, cl_y = cs(s, CRITIC); vn_x, vn_y = cs(s, VNEXT)
    for x, y in zip(ax, ay):
        if y > 1.0:
            cl = dcl = vn = float('nan')
            if len(cl_x):
                ci = int(np.argmin(np.abs(cl_x - x))); cl = cl_y[ci]; dcl = abs(cl_x[ci] - x)
            if len(vn_x):
                vi = int(np.argmin(np.abs(vn_x - x))); vn = vn_y[vi]
            print(f"  seed{s:2d} @{x:>7d}: audit eps_T={y:8.2f} | train critic_loss={cl:11.3f} "
                  f"(dstep {dcl:.0f}) value_next_max={vn:8.2f}")

C = {"audit eps_T": "#D55E00", "critic_loss": "#0072B2", "value_next_max": "#009E73"}
fig, axes = plt.subplots(1, len(SEEDS), figsize=(5 * len(SEEDS), 4.5), dpi=150, sharey=True)
for ax, s in zip(axes, SEEDS):
    series = {"audit eps_T": audit(s), "critic_loss": cs(s, CRITIC), "value_next_max": cs(s, VNEXT)}
    for lab, (xs, ys) in series.items():
        if len(xs) == 0:
            continue
        pos = ys[ys > 0]; med = np.median(pos) if pos.size else 1.0
        ax.plot(xs / 1e3, np.maximum(ys, 1e-9) / max(med, 1e-9),
                color=C[lab], lw=1.0, alpha=0.85, label=lab)
    rx, ry = cs(s, ROLLB)
    if len(rx) > 1:
        for xr in rx[1:][np.diff(ry) > 0]:
            ax.axvline(xr / 1e3, color="#999999", lw=0.4, alpha=0.4)
    ax.set_yscale("log"); ax.set_title(f"seed {s}", fontsize=11)
    ax.set_xlabel("training step (k)"); ax.grid(alpha=0.2, which="both")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
axes[0].set_ylabel("fold over own median (log)")
axes[0].legend(loc="upper right", fontsize=8, frameon=False)
fig.suptitle("cartpole learned dynamics: audit eps_T vs training critic_loss / rolled-value magnitude\n"
             "fold-over-median, log; grey = dynamics rollbacks. aligned spikes = real transient, audit-only = save artifact",
             fontsize=11, y=1.05)
fig.tight_layout(rect=[0, 0, 1, 0.93])
os.makedirs("results", exist_ok=True)
fig.savefig("results/cartpole_epsT_vs_training.png", bbox_inches="tight")
print("\nsaved results/cartpole_epsT_vs_training.png")
