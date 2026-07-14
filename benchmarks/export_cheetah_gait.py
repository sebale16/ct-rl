#!/usr/bin/env python
"""Roll out the four video-aligned cheetah policies on BOTH time bases
(regular/uniform dt=0.01 and irregular/training-matched), extract the
per-leg-joint phase-portrait data for every leg joint, and:

  * write the two phase-portrait figures (regular + irregular; pinned bthigh)
  * export all the underlying data as one JSON (+ a flat metrics CSV) for an
    interactive visualizer that selects (time-interval type, joint).

The heavy work (policy rollouts) is done once per (time base, mode); every
joint's (theta, theta-dot, Poincare section) is derived from the same rollout,
so switching joints in the visualizer needs no re-rollout.

    python -m benchmarks.export_cheetah_gait
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import csv
import json

import numpy as np

from evaluations.locomotion_metrics import (
    rollout_cheetah, phase_portrait_data, _JOINT_NAMES,
)
from evaluations.run_locomotion_metrics import build_env, load_policy, _parse_arch
from benchmarks.plot_cheetah_phase_portrait import plot_phase_portraits, _color_for

# (mode key, best seed, short label)
MODES = [
    ("top", 6, "mf 300k"),
    ("top_buf1m", 0, "mf 1M"),
    ("mbq_vhead_quad", 7, "orc 300k"),
    ("mbq_vhead_quad_buf1m", 8, "orc 1M"),
]
# time base -> build_env kwargs + the agg dir holding the 8-episode metrics
TIME_BASES = {
    "regular": dict(
        env=dict(time_sampling="uniform", dt=0.01, episode_seconds=10.0),
        agg_dir="results/cheetah_locomotion_regular",
    ),
    "irregular": dict(
        env=dict(time_sampling="irregular", dt=0.01, episode_seconds=10.0,
                 physics_dt=0.002, min_dt=0.002, max_dt=0.03, max_steps=2000,
                 time_sampling_kwargs="tail_p=0.99;tail_split=0.9"),
        agg_dir="results/cheetah_locomotion",
    ),
}
PI_ARCH = "400,300"
ACTIVATION = "relu"
DETERMINISTIC = False         # policies were trained with a stochastic actor
                              # (matches the phase-portrait figure invocation)
WARMUP_S = 2.0
PIN_JOINT = "bthigh"          # reference joint pinned in the phase-portrait figures
N_ROLL = 3                    # keep the clearest (most strides) of N episodes
OUT_DIR = "results/cheetah_gait_data"
# aggregate metrics exported per (time base, mode) for the visualizer's summary panel
AGG_KEYS = [
    "return", "mean_speed_mps", "cost_of_transport", "return_per_joule",
    "autocorr_peak", "stride_cv", "poincare_dispersion", "limb_plv",
    "peak_power_frac_mean", "stride_freq_hz",
]


def _rnd(a, p=5):
    return [round(float(x), p) for x in np.asarray(a).ravel()]


def best_rollout(policy, env):
    """Roll N episodes; keep the one with the most detected strides on the
    pinned reference joint (clearest limit cycle), matching the portrait."""
    best, best_n = None, -1
    for _ in range(N_ROLL):
        roll = rollout_cheetah(policy, env)
        pd = phase_portrait_data(roll, warmup_s=WARMUP_S, ref_joint=PIN_JOINT)
        n = pd["metrics"].get("n_strides", 0) if pd else -1
        if n > best_n:
            best, best_n = roll, n
    return best


def joint_block(roll):
    """Per-joint phase-portrait data for every leg joint from one rollout."""
    out = {}
    for j in _JOINT_NAMES:
        pd = phase_portrait_data(roll, warmup_s=WARMUP_S, ref_joint=j)
        if pd is None:
            out[j] = None
            continue
        out[j] = {
            "theta": _rnd(pd["theta"]),
            "theta_dot": _rnd(pd["theta_dot"]),
            "time": _rnd(pd["time"], 4),
            "section_theta": round(float(pd["section_theta"]), 5),
            "cross_theta_dot": _rnd(pd["cross_theta_dot"]),
            "metrics": {k: (round(float(v), 5) if np.isfinite(v) else None)
                        for k, v in pd["metrics"].items()},
        }
    # auto-selected reference (most periodic) joint, for reference
    auto = phase_portrait_data(roll, warmup_s=WARMUP_S, ref_joint=None)
    return out, (auto["joint"] if auto else None)


def load_agg(agg_dir, mode):
    p = f"{agg_dir}/cheetah_{mode}_locomotion.agg.json"
    if not os.path.exists(p):
        return {}
    a = json.load(open(p))
    return {k: a.get(f"{k}_mean") for k in AGG_KEYS if f"{k}_mean" in a}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pi_arch = _parse_arch(PI_ARCH)
    export = {
        "meta": {
            "joints": list(_JOINT_NAMES),
            "time_bases": list(TIME_BASES),
            "pinned_joint_in_figures": PIN_JOINT,
            "warmup_s": WARMUP_S,
            "modes": [{"key": m, "seed": s, "label": lbl} for m, s, lbl in MODES],
            "agg_keys": AGG_KEYS,
            "configs": {tb: cfg["env"] for tb, cfg in TIME_BASES.items()},
        },
        "data": {},
    }
    flat_rows = []
    for tb, cfg in TIME_BASES.items():
        print(f"==== time base: {tb} ====", flush=True)
        env = build_env("cheetah-run", raw_state_obs=False, seed=0, **cfg["env"])
        export["data"][tb] = {}
        pin_portraits, labels, colors = [], [], []
        for mode, seed, lbl in MODES:
            bm = _glob_best(mode, seed)
            policy = load_policy(bm, env, pi_arch=pi_arch, activation=ACTIVATION,
                                 deterministic_policy=DETERMINISTIC)
            roll = best_rollout(policy, env)
            joints, auto_ref = joint_block(roll)
            agg = load_agg(cfg["agg_dir"], mode)
            export["data"][tb][mode] = {
                "label": lbl, "seed": seed, "auto_ref_joint": auto_ref,
                "aggregate": agg, "joints": joints,
            }
            # phase-portrait figure uses the pinned joint
            pj = joints.get(PIN_JOINT)
            pin_portraits.append(
                None if pj is None else {
                    "joint": PIN_JOINT,
                    "theta": np.asarray(pj["theta"]),
                    "theta_dot": np.asarray(pj["theta_dot"]),
                    "time": np.asarray(pj["time"]),
                    "section_theta": pj["section_theta"],
                    "cross_theta_dot": np.asarray(pj["cross_theta_dot"]),
                    "metrics": {k: (v if v is not None else float("nan"))
                                for k, v in pj["metrics"].items()},
                })
            labels.append(lbl); colors.append(_color_for(lbl))
            for j, jb in joints.items():
                if jb is None:
                    continue
                m = jb["metrics"]
                flat_rows.append({
                    "time_base": tb, "mode": mode, "label": lbl, "joint": j,
                    "is_auto_ref": int(j == auto_ref),
                    "n_points": len(jb["theta"]), "n_crossings": len(jb["cross_theta_dot"]),
                    "section_theta": jb["section_theta"],
                    "autocorr_peak": m.get("autocorr_peak"),
                    "stride_cv": m.get("stride_cv"),
                    "poincare_dispersion": m.get("poincare_dispersion"),
                    "limb_plv": m.get("limb_plv"),
                    "n_strides": m.get("n_strides"),
                })
            print(f"  {mode:22s} auto_ref={auto_ref} "
                  f"bthigh_strides={pj['metrics'].get('n_strides') if pj else 'NA'}",
                  flush=True)
        out = f"results/cheetah_phase_portraits_{tb}.png"
        plot_phase_portraits(pin_portraits, labels, out, colors=colors)

    json_path = f"{OUT_DIR}/gait_data.json"
    with open(json_path, "w") as f:
        json.dump(export, f)
    csv_path = f"{OUT_DIR}/gait_metrics_long.csv"
    cols = list(flat_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(flat_rows)
    sz = os.path.getsize(json_path) / 1e6
    print(f"\nwrote {json_path} ({sz:.1f} MB), {csv_path} ({len(flat_rows)} rows)")


def _glob_best(mode, seed):
    import glob
    hits = glob.glob(
        f"saved_models/ct_sac/cheetah-run/{mode}/seed_{seed}/*cforce_grid_chain*/best_model/best_model.pth"
    )
    if not hits:
        raise FileNotFoundError(f"no best_model for {mode} seed {seed}")
    return hits[0]


if __name__ == "__main__":
    main()
