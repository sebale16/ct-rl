#!/usr/bin/env python
"""Per-seed critic/value audit for the learned-dynamics cartpole, aligned.

Every seed's audit and its return are read at EXACTLY the same checkpoint -- the
end-of-training chain checkpoint (num_timesteps=500000), whose model.pth holds
the aligned policy/critic/V-head and whose model.dynamics.pth holds the published
dynamics the critic target actually consumed (CTSAC.load restores both the live
and the target dynamics model + marks the V-head ready). All seeds are evaluated
on ONE FIXED dataset of states (fixed-seed OU exploration on cartpole), so
per-seed differences reflect the value/critic/model, not the state distribution.

Per seed, on the fixed dataset, we report the RMS of:
  * V            -- target V-head value V(s)
  * grad V       -- ||d V(s)/d s||         (the value gradient)
  * oracle b.gradV -- (b_MJ(s,a) . grad V)  with the ORACLE (MuJoCo) drift b_MJ,
                     i.e. the true generator projection the critic target approximates
  * target       -- the model-based critic target q_fast_target the critic regresses to
                     (substep-quadrature generator target, learned dynamics)
  * Bellman err  -- min_i Q_i(s,a) - target   (critic TD residual)

and joins the seed's FINAL evaluation return.

    python -m evaluations.cartpole_critic_audit --out results/cartpole_critic_audit
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np
import torch as th

from common.utils import load_ct_hyperparams_from_table
from benchmarks.run_ct_rl import (
    make_ct_env, _select_structured_dof_layout, _pop_structured_model_kwargs,
)
from models import ActorQCriticModel
from models.port_hamiltonian import DOFLayout, PortHamiltonianModel
from models.noise import OrnsteinUhlenbeckActionNoise
from algorithms.ct_sac import CTSAC

ENV_ID = "cartpole-swingup"
MODE = "mbq_structured_quad_roll"
CHAIN = ".chain_mbq_cforce_grid/cartpole-swingup__mbq_structured_quad_roll"
SEEDS = list(range(12))


def final_return(seed):
    vals = []
    for f in glob.glob(
        f"logs/ct_sac/{ENV_ID}/{MODE}/seed_{seed}/*cforce_grid_chain*/eval/evaluations.npz"
    ):
        d = np.load(f, allow_pickle=True)
        vals.append((int(d["timesteps"][-1]), float(np.mean(d["results"][-1]))))
    vals.sort()
    return vals[-1][1] if vals else float("nan")


def collect_fixed_dataset(env, n, seed, ou_sigma=0.4):
    """One fixed OU-exploration dataset (obs, act, next_obs, reward, done)."""
    env.action_space.seed(seed)
    ad = int(np.prod(env.action_space.shape))
    ou = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(ad), sigma=ou_sigma * np.ones(ad), theta=0.15, dt=0.01
    )
    O, A, NO, R, DN = [], [], [], [], []
    obs, _ = env.reset(seed=seed)
    for _ in range(n):
        a = np.clip(ou(), env.action_space.low, env.action_space.high).astype(np.float32)
        o, t, _, r, no, nt, term, trunc, _ = env.step_dt(a)
        O.append(o); A.append(a); NO.append(no); R.append(r)
        DN.append(1.0 if (term or trunc) else 0.0)
        if term or trunc:
            obs, _ = env.reset(); ou.reset()
        else:
            obs = no
    f = lambda x: np.asarray(x, dtype=np.float32)
    return f(O), f(A), f(NO), f(R).reshape(-1, 1), f(DN).reshape(-1, 1)


def rms(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean(x ** 2)))


def build_algorithm():
    tt, env_kwargs, model_kwargs, algo_kwargs, log_kwargs = load_ct_hyperparams_from_table(
        "ct_sac", ENV_ID, MODE, hyperparams_dir="benchmarks/hyperparams")
    env_kwargs.pop("n_envs", None)
    env_kwargs.pop("eval_n_envs", None)
    env = make_ct_env(env_id=ENV_ID, seed=0, env_kwargs=dict(env_kwargs))
    # replicate run_ct_rl's structured-dynamics construction
    contact_force = int(str(algo_kwargs.pop("dynamics_contact_force", "") or "").strip() or 0)
    structured_model_kwargs = _pop_structured_model_kwargs(algo_kwargs)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    dof_layout = _select_structured_dof_layout(env, obs_dim, DOFLayout)
    algo_kwargs["dynamics_model"] = PortHamiltonianModel(
        obs_dim, act_dim, mode="structured",
        human_input_intensity=float(algo_kwargs.get("human_input_intensity", 0.0) or 0.0),
        contact_force=contact_force, dof_layout=dof_layout, **structured_model_kwargs)
    algo = CTSAC(env=env, model=ActorQCriticModel, model_kwargs=model_kwargs,
                 seed=0, **algo_kwargs)
    return algo, env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="fixed-dataset transitions")
    ap.add_argument("--dataset-seed", type=int, default=20260713)
    ap.add_argument("--out", default="results/cartpole_critic_audit")
    args = ap.parse_args()

    algo, env = build_algorithm()

    # ONE fixed dataset, shared by every seed
    O, A, NO, R, DN = collect_fixed_dataset(env, args.n, args.dataset_seed)
    print(f"fixed dataset: {len(O)} transitions (seed {args.dataset_seed})", flush=True)
    dev = algo.device
    O_t = th.as_tensor(O, dtype=th.float32, device=dev)
    A_t = th.as_tensor(A, dtype=th.float32, device=dev)
    NO_t = th.as_tensor(NO, dtype=th.float32, device=dev)
    R_t = th.as_tensor(R, dtype=th.float32, device=dev)
    DN_t = th.as_tensor(DN, dtype=th.float32, device=dev)
    b_oracle = np.asarray(env.dynamics_terms(O, A), dtype=np.float64)   # ORACLE drift, seed-independent

    rows = []
    for s in SEEDS:
        ck = f"{CHAIN}/seed_{s}/checkpoint"
        algo.load(f"{ck}/model.pth")                       # restores policy/critic/V-head + published dynamics
        ts = th.load(f"{ck}/train_state.pt", map_location="cpu", weights_only=False)
        la = ts.get("counters", {}).get("alpha")
        if algo.log_alpha is not None and "log_alpha" in ts:
            algo.log_alpha.data.copy_(th.as_tensor(ts["log_alpha"]).to(dev))
        alpha = (th.exp(algo.log_alpha.detach()) if algo.log_alpha is not None
                 else algo.alpha_tensor)

        # V and grad V from the TARGET V-head (the value the generator target reads)
        obs_req = O_t.clone().requires_grad_(True)
        V = algo.model.target_value(obs_req)               # (B,1)
        (gV,) = th.autograd.grad(V.sum(), obs_req)         # (B,O)
        V_np, gV_np = V.detach().cpu().numpy(), gV.detach().cpu().numpy()

        b_proj = (b_oracle * gV_np).sum(-1)                # oracle b . grad V  (per second)

        target = algo._model_based_target(O_t, A_t, NO_t, R_t, DN_t,
                                          th.ones(len(O), 1, device=dev) * algo.dt_default,
                                          alpha).detach().cpu().numpy()   # (B,1)
        with th.no_grad():
            qs = algo.model.q_values(O_t, A_t)             # list of (B,1)
            q_min = th.stack(qs, 0).min(0).values.cpu().numpy()
        bellman = q_min.ravel() - target.ravel()

        row = {
            "seed": s,
            "final_return": round(final_return(s), 2),
            "rms_V": round(rms(V_np), 4),
            "rms_gradV": round(rms(np.linalg.norm(gV_np, axis=1)), 4),
            "rms_oracle_b_gradV": round(rms(b_proj), 4),
            "rms_target": round(rms(target), 4),
            "rms_bellman_err": round(rms(bellman), 4),
        }
        rows.append(row)
        print(f"seed {s:2d}: ret={row['final_return']:7.1f}  RMS V={row['rms_V']:.3f}  "
              f"gradV={row['rms_gradV']:.3f}  b.gradV={row['rms_oracle_b_gradV']:.3f}  "
              f"target={row['rms_target']:.3f}  bellman={row['rms_bellman_err']:.4f}", flush=True)

    cols = ["seed", "final_return", "rms_V", "rms_gradV", "rms_oracle_b_gradV",
            "rms_target", "rms_bellman_err"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(args.out + ".json", "w"), indent=2)
    print(f"\nwrote {args.out}.csv / .json ({len(rows)} seeds)")


if __name__ == "__main__":
    main()
