#!/usr/bin/env python
"""Poincaré phase portraits of the cheetah gait for one or more saved policies.

For each policy this rolls out one episode and plots the reference leg joint's
angle vs. its velocity (θ, θ̇) over the steady-cruise window. A tight closed band
is a clean limit-cycle gait; a diffuse cloud is irregular. Overlaid are the
Poincaré section (θ = mean, crossed going up) and, in the bottom row, the return
map θ̇ₙ₊₁ vs θ̇ₙ at successive crossings — a tight cluster on the diagonal means a
stable cycle. Colour = family (model-free blue, oracle vermillion); the annotated
numbers are the same autocorr / stride-CV / Poincaré-dispersion the battery reports.

Needs the checkpoints (not in the repo — trained policies live elsewhere). Example
for the oracle pair::

    python -m benchmarks.plot_cheetah_phase_portrait \
        --checkpoint /path/orc_300k.pt --label "orc 300k" \
        --checkpoint /path/orc_1M.pt   --label "orc 1M" \
        --out results/cheetah_phase_portraits.png

All checkpoints in one call must share the observation layout (pass
``--no-raw-state-obs`` for model-free task-obs policies).
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluations.locomotion_metrics import rollout_cheetah, phase_portrait_data
from evaluations.run_locomotion_metrics import build_env, load_policy, _parse_arch

BLUE, VERM = "#0072B2", "#D55E00"     # Okabe-Ito, CVD-safe
INK, MUT = "#222222", "#666666"


def _color_for(label: str) -> str:
    lo = label.lower()
    if "random" in lo:
        return MUT
    if lo.startswith("mf") or "model-free" in lo or "free" in lo:
        return BLUE
    return VERM


def build_portrait_from_checkpoint(
    checkpoint: Optional[str], env, *, pi_arch, activation, deterministic_policy,
    warmup_s: float, seed_reset: int = 0, ref_joint: Optional[object] = None,
) -> Optional[Dict[str, Any]]:
    if checkpoint is None:
        policy = lambda _o: env.action_space.sample()  # noqa: E731
    else:
        policy = load_policy(
            checkpoint, env, pi_arch=pi_arch, activation=activation,
            deterministic_policy=deterministic_policy,
        )
    # Roll a few episodes; keep the one with the most detected strides (clearest cycle).
    best = None
    for _ in range(3):
        roll = rollout_cheetah(policy, env)
        pd = phase_portrait_data(roll, warmup_s=warmup_s, ref_joint=ref_joint)
        if pd is None:
            continue
        if best is None or pd["metrics"].get("n_strides", 0) > best["metrics"].get("n_strides", 0):
            best = pd
    return best


def plot_phase_portraits(portraits: List[Dict[str, Any]], labels: List[str],
                         out: str, colors: Optional[List[str]] = None) -> None:
    n = len(portraits)
    colors = colors or [_color_for(l) for l in labels]
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8), dpi=150, squeeze=False)

    for j, (pd, label, col) in enumerate(zip(portraits, labels, colors)):
        top, bot = axes[0][j], axes[1][j]
        if pd is None:
            for ax in (top, bot):
                ax.text(0.5, 0.5, "no steady gait", ha="center", va="center",
                        transform=ax.transAxes, color=MUT)
            top.set_title(label, color=INK)
            continue

        theta, td = pd["theta"], pd["theta_dot"]
        m = pd["metrics"]

        # --- phase portrait: (θ, θ̇) limit cycle + Poincaré section ---
        top.plot(theta, td, color=col, lw=0.7, alpha=0.55)
        top.axvline(pd["section_theta"], color=MUT, lw=1.0, ls="--", alpha=0.8)
        top.scatter(np.full_like(pd["cross_theta_dot"], pd["section_theta"]),
                    pd["cross_theta_dot"], s=26, color=col, edgecolor="white",
                    linewidth=0.6, zorder=3)
        top.set_title(label, color=INK, fontsize=12)
        top.set_xlabel(f"{pd['joint']} angle  " + r"$\theta$ [rad]", fontsize=9)
        top.set_ylabel(r"$\dot\theta$ [rad/s]", fontsize=9)
        note = (f"autocorr {m['autocorr_peak']:.2f}\nstride-CV {m['stride_cv']:.3f}\n"
                f"Poincaré disp {m['poincare_dispersion']:.2f}")
        top.text(0.03, 0.97, note, transform=top.transAxes, va="top", ha="left",
                 fontsize=8.5, color=INK,
                 bbox=dict(boxstyle="round", fc="white", ec=MUT, alpha=0.8))

        # --- Poincaré return map: θ̇ₙ₊₁ vs θ̇ₙ ---
        c = pd["cross_theta_dot"]
        if len(c) >= 2:
            bot.scatter(c[:-1], c[1:], s=30, color=col, edgecolor="white", linewidth=0.6)
            lo, hi = float(np.min(c)), float(np.max(c))
            pad = 0.05 * (hi - lo + 1e-9)
            bot.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUT, lw=0.8, ls=":")
            bot.set_aspect("equal", adjustable="box")
        bot.set_xlabel(r"$\dot\theta_n$ at crossing [rad/s]", fontsize=9)
        bot.set_ylabel(r"$\dot\theta_{n+1}$ [rad/s]", fontsize=9)

        for ax in (top, bot):
            ax.grid(alpha=0.25)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.suptitle("cheetah-run gait phase portraits & Poincaré return maps\n"
                 "tight closed band / clustered return map = clean limit-cycle gait",
                 fontsize=13, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", action="append", default=[],
                   help="Checkpoint path (repeat for several policies).")
    p.add_argument("--label", action="append", default=[],
                   help="Label per checkpoint (repeat, same order).")
    p.add_argument("--random", action="store_true",
                   help="Add a random-policy column (no checkpoint).")
    p.add_argument("--env-id", type=str, default="cheetah-run")
    p.add_argument("--raw-state-obs", dest="raw_state_obs", action="store_true", default=True)
    p.add_argument("--no-raw-state-obs", dest="raw_state_obs", action="store_false")
    p.add_argument("--pi-arch", type=str, default="400,300")
    p.add_argument("--activation", type=str, default="relu")
    p.add_argument("--deterministic-policy", action="store_true")
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--episode-seconds", type=float, default=10.0)
    p.add_argument("--warmup-s", type=float, default=2.0)
    p.add_argument("--joint", type=str, default=None,
                   help="pin the reference joint across all policies "
                        "(name: bthigh/bshin/bfoot/fthigh/fshin/ffoot, or index); "
                        "default auto-selects per episode")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="results/cheetah_phase_portraits.png")
    args = p.parse_args(argv)

    checkpoints = list(args.checkpoint)
    labels = list(args.label) or [f"policy {i}" for i in range(len(checkpoints))]
    if args.random:
        checkpoints.append(None)
        labels.append("random")
    if not checkpoints:
        p.error("provide at least one --checkpoint (with --label) or --random")

    env = build_env(args.env_id, raw_state_obs=args.raw_state_obs, dt=args.dt,
                    episode_seconds=args.episode_seconds, seed=args.seed)
    pi_arch = _parse_arch(args.pi_arch)
    portraits = [
        build_portrait_from_checkpoint(
            ckpt, env, pi_arch=pi_arch, activation=args.activation,
            deterministic_policy=args.deterministic_policy, warmup_s=args.warmup_s,
            ref_joint=args.joint,
        )
        for ckpt in checkpoints
    ]
    plot_phase_portraits(portraits, labels, args.out)


if __name__ == "__main__":
    main()
