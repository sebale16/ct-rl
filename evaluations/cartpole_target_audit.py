#!/usr/bin/env python
"""Target-decomposition audit for the learned-dynamics cartpole.

At a loaded checkpoint, with the TARGET V-head, integrate each (x,a) over
dt_default under the learned published dynamics (x_hat) and true MuJoCo physics
(x_mj), and decompose the value increment the critic target reads:

    dV_learned = V_target(x_hat) - V_target(x_cur)
    dV_oracle  = V_target(x_mj)  - V_target(x_cur)
    eps_T      = V_target(x_hat) - V_target(x_mj)  = dV_learned - dV_oracle

Distributions (the state x action sets that matter for learning):
  * replay_data -- checkpoint's buffer.npz samples with their logged actions
                   (the critic-target distribution)
  * replay_pi   -- replay STATES with fresh policy-sampled actions
                   (the actor-update distribution)
  * onpolicy    -- fresh states+actions from the loaded policy
  * ou          -- shared broad OU-exploration set

Metrics per (checkpoint, distribution): for eps_T, the L-O target gap, and each
Q critic vs each target -- RMS, BIAS, and p95/p99 tails (not RMS alone); plus
eps_T RELATIVE to the oracle value-increment scale (RMS eps_T / RMS dV_oracle);
the dV SIGN-disagreement fraction; the dV and target RANK (Spearman) agreement;
and, for policy-action sets, the ACTION GAP (||a - a_ref||) and its correlation
with |eps_T|. Per seed: actual checkpoint step and direct_return (the only
checkpoint-valid return; logged eval returns are off-lineage, MAE ~221/RMSE ~313).

--trajectory additionally runs the eps_T decomposition at every periodic
checkpoint (on onpolicy + ou; replay buffers are not saved intermediate), so a
clean final model cannot hide harmful transient errors earlier in training.

    python -m evaluations.cartpole_target_audit --out results/cartpole_target_audit_by_distribution
    python -m evaluations.cartpole_target_audit --trajectory --out results/cartpole_target_audit_trajectory
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re

import numpy as np
import torch as th
from scipy.stats import pearsonr, spearmanr

from models.port_hamiltonian import integrate_drift
from evaluations.hamiltonian_recovery import mujoco_transition
from evaluations.cartpole_critic_audit import (
    build_algorithm, collect_fixed_dataset, final_return, rms, CHAIN, SEEDS,
)

SAVED = "saved_models/ct_sac/cartpole-swingup/mbq_structured_quad_roll"


def periodic_checkpoints(seed):
    out = []
    for f in glob.glob(f"{SAVED}/seed_{seed}/*/ct_sac_*_*_steps.pth"):
        if f.endswith(".dynamics.pth"):
            continue
        m = re.search(r"_(\d+)_steps\.pth$", f)
        if m:
            out.append((int(m.group(1)), f))
    return sorted(out)


def load_buffer_dataset(ck, n, seed):
    d = np.load(f"{ck}/buffer.npz", allow_pickle=False)
    pos, full = int(d["pos"]), bool(d["full"])
    valid = d["observations"].shape[0] if full else pos
    rng = np.random.default_rng(seed)
    idx = rng.choice(valid, size=min(n, valid), replace=False)
    return (d["observations"][idx, 0, :].astype(np.float32),
            d["actions"][idx, 0, :].astype(np.float32),
            d["rewards"][idx].astype(np.float32).reshape(-1, 1),
            d["dones"][idx].astype(np.float32).reshape(-1, 1))


def collect_policy_dataset(algo, env, n, seed):
    O, A, R, DN = [], [], [], []
    obs, _ = env.reset(seed=seed)
    for _ in range(n):
        with th.no_grad():
            a_t, _ = algo.model.act(
                th.as_tensor(obs, dtype=th.float32, device=algo.device).unsqueeze(0),
                deterministic=False)
        a = a_t.cpu().numpy()[0]
        o, t, _, r, no, nt, term, trunc, _ = env.step_dt(a)
        O.append(o); A.append(a); R.append([r]); DN.append([1.0 if (term or trunc) else 0.0])
        obs = env.reset()[0] if (term or trunc) else no
    f = lambda x: np.asarray(x, dtype=np.float32)
    return f(O), f(A), f(R), f(DN)


def policy_actions(algo, O):
    with th.no_grad():
        a, _ = algo.model.act(th.as_tensor(O, dtype=th.float32, device=algo.device),
                              deterministic=False)
    return a.cpu().numpy().astype(np.float32)


def _corr(fn, a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return round(float(fn(a, b)[0]), 4)


def _estats(x, prefix, d):
    ax = np.abs(x)
    d[f"{prefix}_rms"] = round(rms(x), 4)
    d[f"{prefix}_bias"] = round(float(np.mean(x)), 4)
    d[f"{prefix}_p95"] = round(float(np.percentile(ax, 95)), 4)
    d[f"{prefix}_p99"] = round(float(np.percentile(ax, 99)), 4)


def decompose(algo, env, O, A, R, DN, beta, dt_default, x_mj=None, A_ref=None,
              full=True):
    """Rich target decomposition on one state x action set at the loaded checkpoint."""
    dev = algo.device
    O_t = th.as_tensor(O, dtype=th.float32, device=dev)
    A_t = th.as_tensor(A, dtype=th.float32, device=dev)
    R_t = th.as_tensor(R, dtype=th.float32, device=dev).reshape(-1, 1)
    DN_t = th.as_tensor(DN, dtype=th.float32, device=dev).reshape(-1, 1)
    keep = 1.0 - DN_t
    if x_mj is None:
        x_mj = th.as_tensor(mujoco_transition(env, O, A, dt_default),
                            dtype=th.float32, device=dev)
    with th.no_grad():
        x_hat = integrate_drift(algo.dynamics_target_model.drift, O_t, A_t,
                                dt_default, max_step=algo._integration_max_step())
        V_cur = algo.model.target_value(O_t)
        V_hat = algo.model.target_value(x_hat)
        V_mj = algo.model.target_value(x_mj)
        dV_l = (V_hat - V_cur).cpu().numpy().ravel()
        dV_o = (V_mj - V_cur).cpu().numpy().ravel()
        eps = (V_hat - V_mj).cpu().numpy().ravel()
        tl = (R_t + keep * (V_hat - beta * V_cur)).cpu().numpy().ravel()
        to = (R_t + keep * (V_mj - beta * V_cur)).cpu().numpy().ravel()
        qs = [q.cpu().numpy().ravel() for q in algo.model.q_values(O_t, A_t)]

    d = {"n_states": len(O)}
    _estats(eps, "eps_T", d)
    d["rel_eps_T_over_dVoracle"] = round(rms(eps) / (rms(dV_o) + 1e-9), 4)
    d["dV_sign_err_frac"] = round(float(np.mean(np.sign(dV_l) != np.sign(dV_o))), 4)
    d["dV_spearman"] = _corr(spearmanr, dV_l, dV_o)
    d["target_spearman"] = _corr(spearmanr, tl, to)
    d["rms_target_learned"] = round(rms(tl), 4)
    d["rms_target_oracle"] = round(rms(to), 4)
    if full:
        _estats(tl - to, "L_minus_O_tgt", d)
        for i, q in enumerate(qs):
            _estats(q - tl, f"Q{i}_vs_learned", d)
        for i, q in enumerate(qs):
            _estats(q - to, f"Q{i}_vs_oracle", d)
    if A_ref is not None:
        gap = np.linalg.norm(np.asarray(A, np.float64) - np.asarray(A_ref, np.float64), axis=1)
        d["action_gap_rms"] = round(rms(gap), 4)
        d["action_gap_p95"] = round(float(np.percentile(gap, 95)), 4)
        d["abs_eps_vs_gap_corr"] = _corr(pearsonr, np.abs(eps), gap)
    return d


def direct_return(algo, env, n_episodes=5, seed0=9000):
    rets = []
    for k in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + k)
        R = 0.0
        for _ in range(6000):
            with th.no_grad():
                a, _ = algo.model.act(
                    th.as_tensor(obs, dtype=th.float32, device=algo.device).unsqueeze(0),
                    deterministic=True)
            o, t, _, r, no, nt, term, trunc, _ = env.step_dt(a.cpu().numpy()[0])
            R += float(r); obs = no
            if term or trunc:
                break
        rets.append(R)
    return round(float(np.mean(rets)), 2), round(float(np.std(rets)), 2)


def _set_alpha(algo, ts):
    if ts is not None and algo.log_alpha is not None and "log_alpha" in ts:
        algo.log_alpha.data.copy_(th.as_tensor(ts["log_alpha"]).to(algo.device))


def run_final(algo, env, beta, dt_default, n, dataset_seed, eval_episodes, out):
    OU_O, OU_A, _, OU_R, OU_DN = collect_fixed_dataset(env, n, dataset_seed)
    OU_xmj = th.as_tensor(mujoco_transition(env, OU_O, OU_A, dt_default),
                          dtype=th.float32, device=algo.device)
    rows = []
    for s in SEEDS:
        ck = f"{CHAIN}/seed_{s}/checkpoint"
        algo.load(f"{ck}/model.pth")
        ts = th.load(f"{ck}/train_state.pt", map_location="cpu", weights_only=False)
        step = int(ts.get("counters", {}).get("num_timesteps", -1))
        _set_alpha(algo, ts)
        dret_m, dret_s = direct_return(algo, env, eval_episodes)

        rO, rA, rR, rDN = load_buffer_dataset(ck, n, seed=s)
        oO, oA, oR, oDN = collect_policy_dataset(algo, env, n, seed=1000 + s)
        rpiA = policy_actions(algo, rO)                       # policy actions at replay states
        dists = [
            ("replay_data", rO, rA, rR, rDN, None, None),
            ("replay_pi", rO, rpiA, rR, rDN, None, rA),       # actor-update dist; gap vs logged action
            ("onpolicy", oO, oA, oR, oDN, None, None),
            ("ou", OU_O, OU_A, OU_R, OU_DN, OU_xmj, None),
        ]
        for name, O, A, R, DN, xmj, Aref in dists:
            m = decompose(algo, env, O, A, R, DN, beta, dt_default, x_mj=xmj, A_ref=Aref)
            rows.append({"seed": s, "distribution": name, "checkpoint_step": step,
                         "direct_return": dret_m, "direct_return_std": dret_s,
                         "logged_final_return": round(final_return(s), 2), **m})
            print(f"seed {s:2d} @{step} [{name:11s}]: eps_T rms={m['eps_T_rms']:.3f} "
                  f"bias={m['eps_T_bias']:+.3f} p99={m['eps_T_p99']:.3f} "
                  f"rel={m['rel_eps_T_over_dVoracle']:.3f} sign={m['dV_sign_err_frac']:.3f} "
                  f"tgt_rho={m['target_spearman']}" +
                  (f" gap={m.get('action_gap_rms')}" if Aref is not None else ""), flush=True)
    _write(rows, out)
    _final_summary(rows)


def run_trajectory(algo, env, beta, dt_default, n, dataset_seed, out):
    OU_O, OU_A, _, OU_R, OU_DN = collect_fixed_dataset(env, n, dataset_seed)
    OU_xmj = th.as_tensor(mujoco_transition(env, OU_O, OU_A, dt_default),
                          dtype=th.float32, device=algo.device)
    rows = []
    for s in SEEDS:
        for step, path in periodic_checkpoints(s):
            algo.load(path)                                   # model + dynamics sidecar; no train_state
            oO, oA, oR, oDN = collect_policy_dataset(algo, env, n, seed=7000 + s)
            for name, O, A, R, DN, xmj in [
                ("onpolicy", oO, oA, oR, oDN, None),
                ("ou", OU_O, OU_A, OU_R, OU_DN, OU_xmj),
            ]:
                m = decompose(algo, env, O, A, R, DN, beta, dt_default,
                              x_mj=xmj, full=False)
                rows.append({"seed": s, "step": step, "distribution": name, **m})
            print(f"seed {s:2d} @{step:>7d}: onp eps_T rms="
                  f"{rows[-2]['eps_T_rms']:.3f}/p99={rows[-2]['eps_T_p99']:.3f} "
                  f"sign={rows[-2]['dV_sign_err_frac']:.3f} | ou eps_T rms="
                  f"{rows[-1]['eps_T_rms']:.3f}/p99={rows[-1]['eps_T_p99']:.3f}", flush=True)
    _write(rows, out)


def _write(rows, out):
    cols = list(dict.fromkeys(k for r in rows for k in r))   # union, order-preserving
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(out + ".json", "w"), indent=2)
    print(f"\nwrote {out}.csv / .json ({len(rows)} rows)")


def _final_summary(rows):
    print("\n=== median over seeds, by distribution ===")
    hdr = ["eps_rms", "eps_bias", "eps_p99", "rel_dVo", "sign", "tgt_rho", "Q0|L_rms"]
    print(f"{'dist':12s}" + "".join(f"{h:>10s}" for h in hdr))
    for name in ("replay_data", "replay_pi", "onpolicy", "ou"):
        sub = [r for r in rows if r["distribution"] == name]
        if not sub:
            continue
        med = lambda k: float(np.nanmedian([r.get(k, np.nan) for r in sub]))
        print(f"{name:12s}{med('eps_T_rms'):10.3f}{med('eps_T_bias'):10.3f}"
              f"{med('eps_T_p99'):10.3f}{med('rel_eps_T_over_dVoracle'):10.3f}"
              f"{med('dV_sign_err_frac'):10.3f}{med('target_spearman'):10.3f}"
              f"{med('Q0_vs_learned_rms'):10.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="states per distribution")
    ap.add_argument("--dataset-seed", type=int, default=20260713)
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--trajectory", action="store_true",
                    help="sweep intermediate periodic checkpoints (onpolicy+ou) instead")
    ap.add_argument("--out", default="results/cartpole_target_audit_by_distribution")
    args = ap.parse_args()

    algo, env = build_algorithm()
    beta, dt_default = float(algo.beta), float(algo.dt_default)
    print(f"beta={beta:.4f} dt_default={dt_default:.4f}", flush=True)
    if args.trajectory:
        run_trajectory(algo, env, beta, dt_default, args.n, args.dataset_seed, args.out)
    else:
        run_final(algo, env, beta, dt_default, args.n, args.dataset_seed,
                  args.eval_episodes, args.out)


if __name__ == "__main__":
    main()
