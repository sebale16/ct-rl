"""Stratified and multi-step scoring of the fitted cheetah dynamics models.

Two questions this answers, on one common held-out dataset that none of the
models was fitted on:

1. Does the arm ranking survive a contact-weighted metric?  ``accel_nrmse``
   averages over every sample, and most samples are flight, so a model that
   predicts flight well and stance badly can outrank one that does the reverse.
   Scores are therefore reported separately on stance samples (MuJoCo reports at
   least one active contact) and flight samples.

2. Does the arm ranking survive a rollout?  The fits minimize one-step error at
   ``fit_horizon 1``.  Open-loop rollouts over increasing horizons are where a
   systematically wrong decomposition should begin to cost prediction accuracy.

Every model is scored on identical states, actions and step durations, so the
comparison is paired.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch as th

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.dmc import DMCContinuousEnv  # noqa: E402
from evaluations.hamiltonian_recovery import collect, ground_truth  # noqa: E402
from models.port_hamiltonian import (  # noqa: E402
    DOFLayout,
    PortHamiltonianModel,
    integrate_drift,
)

# A saved sidecar records how it was built, so nothing here needs a table of
# arm names. ``_contact_geometry_version`` selects the geometry and
# ``_contact_solver_version`` the contact law, using the same vocabulary the
# model's own loader reports:
#
#   geometry 0 learned          1 kinematic
#   solver   0 compliant        1 fixed-regularizer constraint
#            2 gate-shaped compliance
#            3 predicted-crossing physical stiffness
#
# Inferring the configuration rather than declaring it means a directory of
# mixed arms, including ones added later, scores without edits.
GEOMETRY = {0: "learned", 1: "kinematic"}
SOLVER = {0: "compliant", 1: "constraint", 2: "constraint", 3: "constraint"}
SOLVER_NAME = {
    0: "compliant",
    1: "constraint",
    2: "constraint+gate_compliance",
    3: "constraint+predicted_crossing",
}


def config_from_sidecar(state: dict) -> dict:
    """The constructor keywords a sidecar was written with."""
    solver_version = int(state["_contact_solver_version"])
    geometry_version = int(state.get("_contact_geometry_version", 0))
    if solver_version not in SOLVER or geometry_version not in GEOMETRY:
        raise ValueError(
            f"unsupported markers: solver {solver_version}, "
            f"geometry {geometry_version}"
        )
    cfg = {
        "contact_geometry": GEOMETRY[geometry_version],
        "contact_solver": SOLVER[solver_version],
        "contact_force": int(state["_contact_raw"].shape[1]),
        "structured_hidden": _hidden_from(state),
    }
    if solver_version == 2:
        cfg["contact_compliance"] = True
    elif solver_version == 3:
        cfg["contact_stiffness"] = True
    return cfg, SOLVER_NAME[solver_version]


def _hidden_from(state: dict) -> tuple:
    """Hidden widths read off the mass head's own weight shapes."""
    widths = []
    index = 0
    while f"mass_net.{index}.weight" in state:
        widths.append(int(state[f"mass_net.{index}.weight"].shape[0]))
        index += 2
    return tuple(widths[:-1]) if len(widths) > 1 else tuple(widths)


def nrmse(pred, true):
    pred, true = np.asarray(pred, np.float64), np.asarray(true, np.float64)
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12))


def build(cfg, obs_dim, act_dim, contact_dt):
    return PortHamiltonianModel(
        obs_dim,
        act_dim,
        mode="structured",
        contact_dt=contact_dt,
        contact_iterations=12,
        contact_regularization=0.01,
        dof_layout=DOFLayout.cheetah(),
        **cfg,
    ).eval()


def stratified_accel(model, obs, act, nxt, dt, stance, load):
    """Acceleration nrmse overall and stratified by how much ground reaction the
    sample actually carries.

    Under exploration the cheetah is in contact almost always, so a
    contact/no-contact split is close to degenerate.  What separates the samples
    is the *size* of the true generalized contact force, so the strata are
    quantiles of that magnitude: the top decile is where contact modelling has
    to be right, and the bottom decile is nearly ballistic motion.
    """
    vel0 = model.layout.vel_slice[0]
    with th.no_grad():
        drift = model.drift(obs, act).cpu().numpy()
    realized = (nxt - obs) / (dt[:, None] + 1e-12)
    a_hat, a_true = drift[:, vel0:], realized[:, vel0:]
    out = {
        "accel_nrmse_all": nrmse(a_hat, a_true),
        "frac_stance": float(stance.mean()),
    }
    if stance.any() and not stance.all():
        out["accel_nrmse_stance"] = nrmse(a_hat[stance], a_true[stance])
        out["accel_nrmse_flight"] = nrmse(a_hat[~stance], a_true[~stance])
    # contact-load strata
    for name, lo, hi in (("q0_25", 0.0, 0.25), ("q25_75", 0.25, 0.75),
                         ("q75_90", 0.75, 0.90), ("top10", 0.90, 1.0)):
        a, b = np.quantile(load, lo), np.quantile(load, hi)
        sel = (load >= a) & (load <= b) if hi == 1.0 else (load >= a) & (load < b)
        if sel.any():
            out[f"accel_nrmse_{name}"] = nrmse(a_hat[sel], a_true[sel])
    # load-weighted: each sample's squared error weighted by its ground reaction
    w = load / (load.mean() + 1e-12)
    num = float(np.sum(w * np.sum((a_hat - a_true) ** 2, axis=1)))
    den = float(np.sum(w * np.sum(a_true ** 2, axis=1)))
    out["accel_nrmse_load_weighted"] = float(np.sqrt(num / (den + 1e-12)))
    return out


def rollout_errors(model, obs, act, nxt, dt, dones, horizons, max_step, stance,
                   n_windows):
    """Open-loop relative error at each horizon.

    The recorded next states are MuJoCo's own trajectory under the same actions,
    so this is a paired learned-vs-simulator rollout.  Error is normalized by the
    distance the true trajectory actually travelled over the window, so a model
    that simply stands still does not score well.  Windows are additionally split
    by whether the true trajectory touches the ground anywhere inside them.
    """
    out = {}
    H_max = max(horizons)
    starts = [
        t for t in range(0, len(obs) - H_max)
        if not dones[t:t + H_max].any()
    ]
    if len(starts) > n_windows:
        idx = np.linspace(0, len(starts) - 1, n_windows).astype(int)
        starts = [starts[i] for i in idx]
    starts = np.asarray(starts, dtype=int)
    if starts.size == 0:
        return {f"rollout_H{H}": float("nan") for H in horizons}

    x = obs[starts].copy()
    disp = np.zeros(len(starts))
    touched = np.zeros(len(starts), dtype=bool)
    # A sample is retired once its state leaves any plausible range: cheetah
    # observations are order 10, so 1e4 is far outside anything physical and
    # keeps a blown-up trajectory from reaching the contact solve, where an
    # infinite mass matrix would abort the whole batch.  Divergence is then a
    # reported outcome rather than a crash.
    BOUND = 1e4
    alive = np.ones(len(starts), dtype=bool)
    diverged_at = np.full(len(starts), np.inf)
    per_h = {}
    with th.no_grad():
        for k in range(H_max):
            step = starts + k
            if alive.any():
                try:
                    nxt_x = integrate_drift(
                        model.drift, x[alive], act[step[alive]],
                        float(np.mean(dt[step])), max_step=max_step,
                    ).cpu().numpy()
                except Exception:
                    nxt_x = np.full((int(alive.sum()), x.shape[1]), np.nan)
                x[alive] = nxt_x
            bad = alive & (~np.isfinite(x).all(axis=1)
                           | (np.abs(x).max(axis=1) > BOUND))
            diverged_at[bad] = k + 1
            alive &= ~bad
            disp += np.linalg.norm(nxt[step] - obs[step], axis=1)
            touched |= stance[step]
            H = k + 1
            if H in horizons:
                err = np.linalg.norm(x - nxt[step], axis=1) / (disp + 1e-12)
                per_h[H] = (err, touched.copy(), alive.copy())

    for H, (err, touch, ok) in per_h.items():
        ok = ok & np.isfinite(err)
        out[f"rollout_H{H}"] = float(np.mean(err[ok])) if ok.any() else float("nan")
        out[f"rollout_H{H}_diverged"] = float(1.0 - ok.mean())
        sel = ok & touch
        out[f"rollout_H{H}_stance"] = float(np.mean(err[sel])) if sel.any() else float("nan")
        sel = ok & ~touch
        out[f"rollout_H{H}_flight"] = float(np.mean(err[sel])) if sel.any() else float("nan")
    finite_div = diverged_at[np.isfinite(diverged_at)]
    out["diverged_frac"] = float(np.isfinite(diverged_at).mean())
    out["diverged_median_step"] = (float(np.median(finite_div))
                                   if finite_div.size else float("nan"))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_eval", type=int, default=3000)
    p.add_argument("--n_windows", type=int, default=400)
    p.add_argument("--eval_seed", type=int, default=9137)
    p.add_argument("--horizons", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16, 32, 64])
    args = p.parse_args()

    th.manual_seed(args.eval_seed)
    np.random.seed(args.eval_seed)
    env = DMCContinuousEnv("cheetah", "run", time_sampling="uniform", dt=0.01,
                           physics_dt=0.002, episode_duration=20.0,
                           seed=args.eval_seed)
    max_step = getattr(env, "physics_dt", None)

    print(f"collecting {args.n_eval} fresh evaluation transitions "
          f"(seed {args.eval_seed}) ...", flush=True)
    O, A, NO, DT, DN = collect(env, args.n_eval, seed=args.eval_seed)
    truth = ground_truth(env, O, A)
    stance = np.asarray(truth["contact_flag"], dtype=bool)
    # magnitude of the true generalized ground reaction at each sample; this is
    # what actually varies across an exploration dataset in which the cheetah is
    # almost never airborne.
    load = np.linalg.norm(np.asarray(truth["qfrc_contact"], np.float64), axis=1)
    qs = np.quantile(load, [0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    print(f"stance fraction {stance.mean():.4f}", flush=True)
    print("contact-load quantiles (0/25/50/75/90/100): "
          + " ".join(f"{v:.3f}" for v in qs), flush=True)

    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])

    rows = []
    for name in sorted(os.listdir(args.models_dir)):
        if not name.endswith(".dynamics.pth"):
            continue
        tag = name[: -len(".dynamics.pth")]
        state = th.load(os.path.join(args.models_dir, name), map_location="cpu")
        try:
            cfg, arm = config_from_sidecar(state)
        except (KeyError, ValueError) as exc:
            print(f"  skip {tag}: {exc}", flush=True)
            continue
        try:
            model = build(cfg, obs_dim, act_dim, max_step)
            model.load_state_dict(state)
        except (ValueError, RuntimeError) as exc:
            # A sidecar written by a configuration this code no longer supports
            # is reported rather than silently dropped.
            print(f"  skip {tag}: {type(exc).__name__}: {str(exc)[:130]}",
                  flush=True)
            continue
        model.eval()
        arm = f"{cfg['contact_geometry']}_{arm}"

        row = {"tag": tag, "arm": arm}
        row.update(stratified_accel(model, O, A, NO, DT, stance, load))
        row.update(rollout_errors(model, O, A, NO, DT, DN, set(args.horizons),
                                  max_step, stance, args.n_windows))
        rows.append(row)
        print(f"  {tag}: all {row['accel_nrmse_all']:.4f}  "
              f"top10 {row.get('accel_nrmse_top10', float('nan')):.4f}  "
              f"q0_25 {row.get('accel_nrmse_q0_25', float('nan')):.4f}  "
              f"wtd {row['accel_nrmse_load_weighted']:.4f}  "
              f"H8 {row.get('rollout_H8', float('nan')):.4f}  "
              f"H64 {row.get('rollout_H64', float('nan')):.4f}", flush=True)

    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    summary = {}
    for arm, rs in by_arm.items():
        keys = [k for k in rs[0] if k not in ("tag", "arm")]
        summary[arm] = {
            k: {
                "mean": float(np.nanmean([r[k] for r in rs])),
                "sd": float(np.nanstd([r[k] for r in rs], ddof=1))
                if len(rs) > 1 else 0.0,
                "seeds": [r[k] for r in rs],
            }
            for k in keys
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({
            "eval_seed": args.eval_seed,
            "n_eval": args.n_eval,
            "n_windows": args.n_windows,
            "stance_fraction": float(stance.mean()),
            "horizons": args.horizons,
            "per_model": rows,
            "by_arm": summary,
        }, fh, indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
