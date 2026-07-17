#!/usr/bin/env python
"""Offline validation of the re-anchored quadrature target on saved checkpoints.

For real transition tuples (x, a, r, x', d, dt) drawn from a checkpoint's
replay buffer / fresh policy rollouts / an OU exploration walk, build the
critic target four ways through the SAME target V-head and score each against
the oracle target (MuJoCo endpoint at dt_default):

  quad     x_hat = Phi_hat^a_{T}(x)          -- current pure model roll
  re       x_re  = reanchored_endpoint(...)  -- data-anchored transport of x'
                                                across the duration mismatch
  gate     lambda-blend of re with fd by the per-sample innovation rate
                                                (exactly the trainer's formula)
  fd       model-free finite difference over the sampled x'

The decisive comparison is eps_* = t_* - t_oracle on the seed-5/11 window
checkpoints: the re-anchor is validated if it collapses the eps tails
(p95/p99) where the pure roll is poor while matching it where the model is
healthy. All constructions share the logged reward and done flag, so eps
isolates the endpoint substitution. Per-probe RNG is isolated via SeedSequence
keys and checkpoint hashes are recorded (cartpole_action_grid_audit
conventions).

    python -m evaluations.cartpole_reanchor_audit --out results/cartpole_reanchor_audit
    python -m evaluations.cartpole_reanchor_audit --trajectory \
        --seeds 1,3,5,11 --min-step 100000 --max-step 400000 \
        --out results/cartpole_reanchor_audit_trajectory
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch as th

from algorithms.ct_sac import reanchored_endpoint
from models.port_hamiltonian import integrate_drift
from models.noise import OrnsteinUhlenbeckActionNoise
from evaluations.hamiltonian_recovery import mujoco_transition
from evaluations.cartpole_critic_audit import build_algorithm, rms, CHAIN, SEEDS
from evaluations.cartpole_target_audit import periodic_checkpoints
from evaluations.cartpole_action_grid_audit import _checkpoint_hashes

_PROBE = {"replay": 11, "onpolicy_env": 12, "onpolicy_torch": 13, "ou": 14}


def _rng(dataset_seed, probe, seed=0, step=0):
    return np.random.default_rng(
        np.random.SeedSequence([dataset_seed, _PROBE[probe], seed, step]))


def _seed_int(dataset_seed, probe, seed=0, step=0):
    ss = np.random.SeedSequence([dataset_seed, _PROBE[probe], seed, step])
    return int(ss.generate_state(1)[0])


def load_buffer_transitions(ck, n, rng):
    d = np.load(f"{ck}/buffer.npz", allow_pickle=False)
    pos, full = int(d["pos"]), bool(d["full"])
    valid = d["observations"].shape[0] if full else pos
    idx = rng.choice(valid, size=min(n, valid), replace=False)
    f = lambda k: d[k][idx, 0].astype(np.float32)
    return (f("observations"), f("actions"), f("rewards").reshape(-1, 1),
            f("next_observations"), f("dones").reshape(-1, 1),
            f("dt").reshape(-1, 1))


def collect_transitions(env, action_fn, n, env_seed):
    """Walk the env recording full (x, a, r, x', done, dt) tuples."""
    O, A, R, NO, DN, DT = [], [], [], [], [], []
    obs, _ = env.reset(seed=env_seed)
    for _ in range(n):
        a = action_fn(obs)
        o, t, _, r, no, nt, term, trunc, _ = env.step_dt(a)
        O.append(o); A.append(a); R.append([r]); NO.append(no)
        DN.append([1.0 if (term or trunc) else 0.0]); DT.append([nt - t])
        obs = env.reset()[0] if (term or trunc) else no
    f = lambda x: np.asarray(x, dtype=np.float32)
    return f(O), f(A), f(R), f(NO), f(DN), f(DT)


def policy_action_fn(algo):
    def fn(obs):
        with th.no_grad():
            a, _ = algo.model.act(
                th.as_tensor(obs, dtype=th.float32,
                             device=algo.device).unsqueeze(0),
                deterministic=False)
        return a.cpu().numpy()[0]
    return fn


def ou_action_fn(env, seed, sigma=0.4):
    env.action_space.seed(seed)
    ad = int(np.prod(env.action_space.shape))
    ou = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(ad), sigma=sigma * np.ones(ad), theta=0.15, dt=0.01)

    def fn(obs):
        return np.clip(ou(), env.action_space.low,
                       env.action_space.high).astype(np.float32)
    return fn


def build_targets(algo, env, O, A, R, NO, DN, DT, gate_rho):
    """All four target constructions + the oracle, through the target V-head."""
    dev = algo.device
    beta, T = float(algo.beta), float(algo.dt_default)
    max_step = algo._integration_max_step()
    O_t = th.as_tensor(O, dtype=th.float32, device=dev)
    A_t = th.as_tensor(A, dtype=th.float32, device=dev)
    NO_t = th.as_tensor(NO, dtype=th.float32, device=dev)
    R_t = th.as_tensor(R, dtype=th.float32, device=dev)
    DT_t = th.as_tensor(DT, dtype=th.float32, device=dev)
    keep = 1.0 - th.as_tensor(DN, dtype=th.float32, device=dev)
    x_mj = th.as_tensor(mujoco_transition(env, O, A, T),
                        dtype=th.float32, device=dev)
    with th.no_grad():
        V = lambda x: algo.model.target_value(x)
        v_cur = V(O_t)
        tgt = lambda v_end: (R_t + keep * (v_end - beta * v_cur))

        t_or = tgt(V(x_mj))
        dv_or = V(x_mj) - v_cur

        x_quad = integrate_drift(algo.dynamics_target_model.drift, O_t, A_t,
                                 T, max_step=max_step)
        t_quad = tgt(V(x_quad))

        x_re, innov, model_seconds = reanchored_endpoint(
            algo.dynamics_target_model.drift, O_t, A_t, NO_t, DT_t, T,
            max_step=max_step)
        t_re = tgt(V(x_re))

        # model-free finite difference (rescaled-time convention)
        u = (DT_t / T).clamp_min(1e-8)
        frac = (th.exp(-beta * u) * V(NO_t) - v_cur) / u
        t_fd = R_t + keep * (v_cur + frac)

        # the trainer's innovation gate, verbatim
        dt_col = DT_t.clamp_min(1e-8)
        rate_scale = ((NO_t - O_t).norm(dim=-1, keepdim=True) / dt_col
                      ).median() + 1e-8
        rho = (innov.norm(dim=-1, keepdim=True) / dt_col) / rate_scale
        lam = th.exp(-(rho / gate_rho).square()) if gate_rho > 0 else None
        t_gate = (lam * t_re + (1 - lam) * t_fd) if lam is not None else None

    out = {"or": t_or, "quad": t_quad, "re": t_re, "fd": t_fd}
    if t_gate is not None:
        out["gate"] = t_gate
    aux = {"dv_or": dv_or, "rho": rho, "model_seconds": model_seconds,
           "lam": lam}
    return out, aux


def _estats(eps, dv_or, prefix, d):
    e = eps.cpu().numpy().ravel().astype(np.float64)
    e = e[np.isfinite(e)]
    if not len(e):
        for k in ("rms", "bias", "p95", "p99", "rel"):
            d[f"{prefix}_{k}"] = float("nan")
        return
    d[f"{prefix}_rms"] = round(rms(e), 4)
    d[f"{prefix}_bias"] = round(float(np.mean(e)), 4)
    d[f"{prefix}_p95"] = round(float(np.percentile(np.abs(e), 95)), 4)
    d[f"{prefix}_p99"] = round(float(np.percentile(np.abs(e), 99)), 4)
    d[f"{prefix}_rel"] = round(rms(e) / (rms(dv_or.cpu().numpy()) + 1e-9), 4)


def audit_transitions(algo, env, tup, gate_rho):
    targets, aux = build_targets(algo, env, *tup, gate_rho=gate_rho)
    d = {"n": len(tup[0])}
    for name in ("quad", "re", "gate", "fd"):
        if name in targets:
            _estats(targets[name] - targets["or"], aux["dv_or"], f"eps_{name}", d)
    rho = aux["rho"].cpu().numpy().ravel()
    rho = rho[np.isfinite(rho)]
    d["rho_med"] = round(float(np.median(rho)), 4) if len(rho) else float("nan")
    d["rho_p95"] = round(float(np.percentile(rho, 95)), 4) if len(rho) else float("nan")
    if aux["lam"] is not None:
        d["lambda_mean"] = round(float(aux["lam"].mean()), 4)
    d["model_seconds_mean"] = round(float(aux["model_seconds"].mean()), 6)
    return d


def _print_row(tag, m):
    print(f"{tag}: rel quad={m['eps_quad_rel']:.3f} re={m['eps_re_rel']:.3f} "
          f"gate={m.get('eps_gate_rel', float('nan')):.3f} "
          f"fd={m['eps_fd_rel']:.3f} | p99 quad={m['eps_quad_p99']:.3f} "
          f"re={m['eps_re_p99']:.3f} | rho_med={m['rho_med']:.3f}", flush=True)


def run_final(algo, env, args, seeds):
    rows, meta = [], {}
    for s in seeds:
        ck = f"{CHAIN}/seed_{s}/checkpoint"
        algo.load(f"{ck}/model.pth")
        ts = th.load(f"{ck}/train_state.pt", map_location="cpu",
                     weights_only=False)
        step = int(ts.get("counters", {}).get("num_timesteps", -1))
        meta[f"seed_{s}"] = {**_checkpoint_hashes(f"{ck}/model.pth",
                                                  f"{ck}/buffer.npz"),
                             "step": step}
        rep = load_buffer_transitions(ck, args.n,
                                      _rng(args.dataset_seed, "replay", seed=s))
        th.manual_seed(_seed_int(args.dataset_seed, "onpolicy_torch", seed=s))
        onp = collect_transitions(
            env, policy_action_fn(algo), args.n,
            _seed_int(args.dataset_seed, "onpolicy_env", seed=s) % (2 ** 31))
        for name, tup in (("replay", rep), ("onpolicy", onp)):
            m = audit_transitions(algo, env, tup, args.gate_rho)
            rows.append({"seed": s, "distribution": name,
                         "checkpoint_step": step, **m})
            _print_row(f"seed {s:2d} @{step} [{name:9s}]", m)
    return rows, meta


def run_trajectory(algo, env, args, seeds):
    ou_tup = None
    rows, meta = [], {}
    for s in seeds:
        for step, path in periodic_checkpoints(s):
            if not (args.min_step <= step <= args.max_step):
                continue
            algo.load(path)
            meta[f"seed_{s}_step_{step}"] = {**_checkpoint_hashes(path),
                                             "step": step}
            if ou_tup is None:  # shared fixed exploration set
                ou_tup = collect_transitions(
                    env, ou_action_fn(env, args.dataset_seed), args.n,
                    _seed_int(args.dataset_seed, "ou") % (2 ** 31))
            th.manual_seed(_seed_int(args.dataset_seed, "onpolicy_torch",
                                     seed=s, step=step))
            onp = collect_transitions(
                env, policy_action_fn(algo), args.n,
                _seed_int(args.dataset_seed, "onpolicy_env",
                          seed=s, step=step) % (2 ** 31))
            for name, tup in (("onpolicy", onp), ("ou", ou_tup)):
                m = audit_transitions(algo, env, tup, args.gate_rho)
                rows.append({"seed": s, "step": step, "distribution": name, **m})
                _print_row(f"seed {s:2d} @{step:>7d} [{name:8s}]", m)
    return rows, meta


def _write(rows, meta, out):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(out + ".json", "w"), indent=2)
    json.dump(meta, open(out + "_meta.json", "w"), indent=2)
    print(f"\nwrote {out}.csv / .json / _meta.json ({len(rows)} rows)")


def _summary(rows):
    print("\n=== median over rows, by distribution ===")
    keys = ["eps_quad_rel", "eps_re_rel", "eps_gate_rel", "eps_fd_rel",
            "eps_quad_p99", "eps_re_p99", "rho_med"]
    hdr = ["quad_rel", "re_rel", "gate_rel", "fd_rel", "quad_p99", "re_p99",
           "rho_med"]
    dists = list(dict.fromkeys(r["distribution"] for r in rows))
    print(f"{'dist':10s}" + "".join(f"{h:>10s}" for h in hdr))
    for name in dists:
        sub = [r for r in rows if r["distribution"] == name]
        med = lambda k: float(np.nanmedian([r.get(k, np.nan) for r in sub]))
        print(f"{name:10s}" + "".join(f"{med(k):10.3f}" for k in keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="transitions per distribution")
    ap.add_argument("--gate-rho", type=float, default=1.0)
    ap.add_argument("--dataset-seed", type=int, default=20260713)
    ap.add_argument("--seeds", default="all")
    ap.add_argument("--trajectory", action="store_true")
    ap.add_argument("--min-step", type=int, default=0)
    ap.add_argument("--max-step", type=int, default=10 ** 12)
    ap.add_argument("--out", default="results/cartpole_reanchor_audit")
    args = ap.parse_args()

    seeds = SEEDS if args.seeds == "all" else [int(x) for x in args.seeds.split(",")]
    algo, env = build_algorithm()
    print(f"beta={float(algo.beta):.4f} T={float(algo.dt_default):.4f} "
          f"gate_rho={args.gate_rho} n={args.n}", flush=True)
    runner = run_trajectory if args.trajectory else run_final
    rows, meta_ckpts = runner(algo, env, args, seeds)
    meta = {"mode": "trajectory" if args.trajectory else "final",
            "chain": CHAIN, "seeds": seeds, "n": args.n,
            "gate_rho": args.gate_rho, "dataset_seed": args.dataset_seed,
            "beta": float(algo.beta), "dt_default": float(algo.dt_default),
            "checkpoints": meta_ckpts}
    _write(rows, meta, args.out)
    _summary(rows)


if __name__ == "__main__":
    main()
