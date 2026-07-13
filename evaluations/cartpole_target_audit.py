#!/usr/bin/env python
"""Decisive target-decomposition audit for the learned-dynamics cartpole.

On the SAME fixed states as cartpole_critic_audit (same dataset-seed / n), with
the TARGET V-head, integrate each (x,a) forward over dt_default under (i) the
learned published dynamics -> x_hat_learned, and (ii) the true MuJoCo physics ->
x_mujoco, and measure the value-endpoint discrepancy the critic target inherits
from model error:

    eps_T = V_target(x_hat_learned) - V_target(x_mujoco)

Per seed (all at the loaded checkpoint), report:
  * checkpoint step (num_timesteps actually in the checkpoint)
  * RMS eps_T
  * learned target vs oracle target: the two substep-quadrature targets
        target = r + (1-done)*(V_target(x_next) - beta*V_cur)
    built with x_hat_learned vs x_mujoco, and RMS of their difference
    (= RMS of (1-done)*eps_T)
  * each Q critic vs the learned target:  RMS(Q_i - target_learned)
  * each Q critic vs the oracle  target:  RMS(Q_i - target_oracle)
  * direct return from rolling out the loaded checkpoint policy (deterministic)
  * the logged final eval return (for cross-check)

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


def direct_return(algo, env, n_episodes=5, seed0=9000):
    """Roll the loaded deterministic policy on the (training-config) env."""
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
            R += float(r)
            obs = no
            if term or trunc:
                break
        rets.append(R)
    return float(np.mean(rets)), float(np.std(rets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="fixed-dataset transitions")
    ap.add_argument("--dataset-seed", type=int, default=20260713,
                    help="MUST match cartpole_critic_audit for the same fixed states")
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--out", default="results/cartpole_target_audit")
    args = ap.parse_args()

    algo, env = build_algorithm()
    dev = algo.device
    beta = float(algo.beta)
    dt_default = float(algo.dt_default)

    # the SAME fixed states as the critic audit
    O, A, NO, R, DN = collect_fixed_dataset(env, args.n, args.dataset_seed)
    print(f"fixed dataset: {len(O)} transitions (seed {args.dataset_seed}); "
          f"beta={beta:.4f} dt_default={dt_default:.4f}", flush=True)
    O_t = th.as_tensor(O, dtype=th.float32, device=dev)
    A_t = th.as_tensor(A, dtype=th.float32, device=dev)
    R_t = th.as_tensor(R, dtype=th.float32, device=dev)          # (B,1)
    DN_t = th.as_tensor(DN, dtype=th.float32, device=dev)        # (B,1)
    keep = (1.0 - DN_t)
    # true next state under real physics over dt_default (seed-independent)
    x_mj = th.as_tensor(mujoco_transition(env, O, A, dt_default),
                        dtype=th.float32, device=dev)

    rows = []
    for s in SEEDS:
        ck = f"{CHAIN}/seed_{s}/checkpoint"
        algo.load(f"{ck}/model.pth")
        ts = th.load(f"{ck}/train_state.pt", map_location="cpu", weights_only=False)
        step = int(ts.get("counters", {}).get("num_timesteps", -1))
        if algo.log_alpha is not None and "log_alpha" in ts:
            algo.log_alpha.data.copy_(th.as_tensor(ts["log_alpha"]).to(dev))

        with th.no_grad():
            max_step = algo._integration_max_step()
            x_hat = integrate_drift(algo.dynamics_target_model.drift, O_t, A_t,
                                    dt_default, max_step=max_step)         # learned endpoint
            V_cur = algo.model.target_value(O_t)                           # (B,1)
            V_hat = algo.model.target_value(x_hat)                         # learned
            V_mj = algo.model.target_value(x_mj)                           # oracle
            eps_T = (V_hat - V_mj)                                         # (B,1)

            tgt_learned = R_t + keep * (V_hat - beta * V_cur)              # (B,1)
            tgt_oracle = R_t + keep * (V_mj - beta * V_cur)

            qs = algo.model.q_values(O_t, A_t)                            # list of (B,1)
            q_np = [q.cpu().numpy().ravel() for q in qs]

        tl = tgt_learned.cpu().numpy().ravel()
        to = tgt_oracle.cpu().numpy().ravel()
        dret_m, dret_s = direct_return(algo, env, args.eval_episodes)

        row = {
            "seed": s,
            "checkpoint_step": step,
            "logged_final_return": round(final_return(s), 2),
            "direct_return": round(dret_m, 2),
            "direct_return_std": round(dret_s, 2),
            "rms_eps_T": round(rms(eps_T.cpu().numpy()), 4),
            "rms_target_learned": round(rms(tl), 4),
            "rms_target_oracle": round(rms(to), 4),
            "rms_learned_minus_oracle_target": round(rms(tl - to), 4),
        }
        for i, q in enumerate(q_np):
            row[f"rms_Q{i}_vs_learned"] = round(rms(q - tl), 4)
        for i, q in enumerate(q_np):
            row[f"rms_Q{i}_vs_oracle"] = round(rms(q - to), 4)
        rows.append(row)
        qcols_l = " ".join(f"Q{i}|L={row[f'rms_Q{i}_vs_learned']:.3f}" for i in range(len(q_np)))
        qcols_o = " ".join(f"Q{i}|O={row[f'rms_Q{i}_vs_oracle']:.3f}" for i in range(len(q_np)))
        print(f"seed {s:2d} @{step}: ret(direct)={dret_m:7.1f} (logged {row['logged_final_return']:7.1f}) "
              f"eps_T={row['rms_eps_T']:.3f} tgtL={row['rms_target_learned']:.3f} "
              f"tgtO={row['rms_target_oracle']:.3f} L-O={row['rms_learned_minus_oracle_target']:.3f} | "
              f"{qcols_l} {qcols_o}", flush=True)

    cols = list(rows[0].keys())
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(args.out + ".json", "w"), indent=2)
    print(f"\nwrote {args.out}.csv / .json ({len(rows)} seeds)")


if __name__ == "__main__":
    main()
