#!/usr/bin/env python
"""Target-decomposition audit for the learned-dynamics cartpole, across the three
state distributions that matter for learning.

For each learned-cartpole seed, at its actual loaded checkpoint, and on each of:
  * replay   -- the checkpoint's own buffer.npz replay samples (states that
                actually drove the critic updates)
  * onpolicy -- fresh states rolled from that exact loaded policy
  * ou       -- the shared fixed OU-exploration set (broad, seed-independent)

integrate each (x,a) over dt_default under the learned published dynamics
(x_hat) and true MuJoCo physics (x_mj), and, with the TARGET V-head, measure the
value-endpoint discrepancy the critic target inherits from model error:

    eps_T = V_target(x_hat) - V_target(x_mj)

Per (seed, distribution): RMS eps_T; the learned vs oracle substep-quadrature
targets  target = r + (1-done)*(V_target(x_next) - beta*V_cur)  and RMS of their
difference (= RMS (1-done)*eps_T); and each Q critic vs the learned target and vs
the oracle target. Per seed also: the actual checkpoint step, and direct_return
from rolling out the loaded checkpoint (the ONLY return valid for checkpoint-level
conclusions -- logged eval returns are off-lineage, MAE ~221 / RMSE ~313).

Comparing across distributions reveals whether model-target error is localized to
the replay / on-policy regions that drive learning, or only shows up on the broad
OU set.

    python -m evaluations.cartpole_target_audit --out results/cartpole_target_audit
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch as th

from models.port_hamiltonian import integrate_drift
from evaluations.hamiltonian_recovery import mujoco_transition
from evaluations.cartpole_critic_audit import (
    build_algorithm, collect_fixed_dataset, final_return, rms, CHAIN, SEEDS,
)


def load_buffer_dataset(ck, n, seed):
    """Sample n replay transitions from a checkpoint's buffer.npz (valid region)."""
    d = np.load(f"{ck}/buffer.npz", allow_pickle=False)
    pos, full = int(d["pos"]), bool(d["full"])
    valid = d["observations"].shape[0] if full else pos
    rng = np.random.default_rng(seed)
    idx = rng.choice(valid, size=min(n, valid), replace=False)
    O = d["observations"][idx, 0, :].astype(np.float32)
    A = d["actions"][idx, 0, :].astype(np.float32)
    R = d["rewards"][idx].astype(np.float32).reshape(-1, 1)
    DN = d["dones"][idx].astype(np.float32).reshape(-1, 1)
    return O, A, R, DN


def collect_policy_dataset(algo, env, n, seed):
    """Fresh on-policy transitions from the loaded (stochastic) policy."""
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


def decompose(algo, env, O, A, R, DN, beta, dt_default, x_mj=None):
    """Target-decomposition metrics on one state set at the loaded checkpoint."""
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
        eps = (V_hat - V_mj).cpu().numpy()
        tl = (R_t + keep * (V_hat - beta * V_cur)).cpu().numpy().ravel()
        to = (R_t + keep * (V_mj - beta * V_cur)).cpu().numpy().ravel()
        qs = [q.cpu().numpy().ravel() for q in algo.model.q_values(O_t, A_t)]
    d = {
        "n_states": len(O),
        "rms_eps_T": round(rms(eps), 4),
        "rms_target_learned": round(rms(tl), 4),
        "rms_target_oracle": round(rms(to), 4),
        "rms_learned_minus_oracle_target": round(rms(tl - to), 4),
    }
    for i, q in enumerate(qs):
        d[f"rms_Q{i}_vs_learned"] = round(rms(q - tl), 4)
    for i, q in enumerate(qs):
        d[f"rms_Q{i}_vs_oracle"] = round(rms(q - to), 4)
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
    return float(np.mean(rets)), float(np.std(rets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="states per distribution")
    ap.add_argument("--dataset-seed", type=int, default=20260713,
                    help="shared-OU seed (matches cartpole_critic_audit)")
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--out", default="results/cartpole_target_audit")
    args = ap.parse_args()

    algo, env = build_algorithm()
    dev = algo.device
    beta = float(algo.beta)
    dt_default = float(algo.dt_default)

    # shared OU set (seed-independent) + its true-physics endpoints, computed once
    OU_O, OU_A, _, OU_R, OU_DN = collect_fixed_dataset(env, args.n, args.dataset_seed)
    OU_xmj = th.as_tensor(mujoco_transition(env, OU_O, OU_A, dt_default),
                          dtype=th.float32, device=dev)
    print(f"shared OU set: {len(OU_O)}; beta={beta:.4f} dt_default={dt_default:.4f}", flush=True)

    rows = []
    for s in SEEDS:
        ck = f"{CHAIN}/seed_{s}/checkpoint"
        algo.load(f"{ck}/model.pth")
        ts = th.load(f"{ck}/train_state.pt", map_location="cpu", weights_only=False)
        step = int(ts.get("counters", {}).get("num_timesteps", -1))
        if algo.log_alpha is not None and "log_alpha" in ts:
            algo.log_alpha.data.copy_(th.as_tensor(ts["log_alpha"]).to(dev))
        dret_m, dret_s = direct_return(algo, env, args.eval_episodes)

        dists = {
            "replay": (load_buffer_dataset(ck, args.n, seed=s), None),
            "onpolicy": (collect_policy_dataset(algo, env, args.n, seed=1000 + s), None),
            "ou": ((OU_O, OU_A, OU_R, OU_DN), OU_xmj),
        }
        for name, (data, xmj) in dists.items():
            O, A, R, DN = data
            m = decompose(algo, env, O, A, R, DN, beta, dt_default, x_mj=xmj)
            row = {
                "seed": s, "distribution": name, "checkpoint_step": step,
                "direct_return": round(dret_m, 2), "direct_return_std": round(dret_s, 2),
                "logged_final_return": round(final_return(s), 2),
                **m,
            }
            rows.append(row)
            print(f"seed {s:2d} @{step} [{name:8s}]: eps_T={m['rms_eps_T']:.3f} "
                  f"tgtL={m['rms_target_learned']:.3f} L-O={m['rms_learned_minus_oracle_target']:.3f} "
                  f"Q0|L={m['rms_Q0_vs_learned']:.3f} Q1|L={m['rms_Q1_vs_learned']:.3f} "
                  f"(dret {dret_m:.0f})", flush=True)

    cols = list(rows[0].keys())
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(args.out + ".json", "w"), indent=2)

    # per-distribution medians of the headline errors
    print("\n=== median over seeds, by distribution ===")
    print(f"{'dist':9s}{'eps_T':>9s}{'L-O tgt':>9s}{'Q0|L':>8s}{'Q1|L':>8s}{'tgtL':>8s}")
    for name in ("replay", "onpolicy", "ou"):
        sub = [r for r in rows if r["distribution"] == name]
        med = lambda k: float(np.median([r[k] for r in sub]))
        print(f"{name:9s}{med('rms_eps_T'):9.3f}{med('rms_learned_minus_oracle_target'):9.3f}"
              f"{med('rms_Q0_vs_learned'):8.3f}{med('rms_Q1_vs_learned'):8.3f}{med('rms_target_learned'):8.3f}")
    print(f"\nwrote {args.out}.csv / .json ({len(rows)} rows)")


if __name__ == "__main__":
    main()
