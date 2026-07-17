#!/usr/bin/env python
"""Offline validation of the re-anchored quadrature target on saved checkpoints.

For real transition tuples (x, a, r, x', d, dt) drawn from a checkpoint's
replay buffer / fresh policy rollouts / an OU exploration walk, build the
critic target four ways through the SAME target V-head and score each against
the oracle target (MuJoCo endpoint at dt_default):

  quad     x_hat = Phi_hat^a_{T}(x)          -- current pure model roll
  re       x_re  = reanchored_endpoint(...)  -- data-anchored transport of x'
                                                across the duration mismatch
  gate     lambda-blend of re with fd by the transport-aware innovation rate,
           using the trainer's minibatch size and finite fallback semantics
  fd       model-free finite difference over the sampled x'

The decisive comparison is eps_* = t_* - t_oracle on the seed-5/11 window
checkpoints: the re-anchor is validated if it collapses the eps tails
(p95/p99) where the pure roll is poor while matching it where the model is
healthy. Endpoint errors, their non-finite fractions, and their normalization
exclude terminal rows: all constructions collapse to the logged reward there,
so including them would only dilute the endpoint comparison. Per-probe RNG is
isolated via SeedSequence keys and checkpoint hashes are recorded
(cartpole_action_grid_audit conventions).

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

from algorithms.ct_sac import reanchor_gate_statistics, reanchored_endpoint
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


def _chunked_gate(
    algo, O, NO, innovation, DT, T, gate_rho, t_re, t_fd,
):
    """Apply the trainer's re-anchor gate in training-sized minibatches.

    The gate scale is a batch median, so evaluating an entire audit dataset as
    one batch does not reproduce training.  Preserve input order and use the
    configured training batch size, including a final short chunk when needed.
    Non-finite re-anchored rows receive exactly the trainer's finite-difference
    fallback; a non-finite anchor remains visible in the returned target and is
    reported by the audit instead of being silently discarded.
    """
    n = int(O.shape[0])
    batch_size = max(1, int(getattr(algo, "batch_size", n or 1)))
    rho_chunks = []
    innovation_rho_chunks = []
    mismatch_chunks = []
    lambda_chunks = []
    gate_chunks = []
    fallback_chunks = []

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        sl = slice(start, stop)
        rho, innovation_rho, mismatch = reanchor_gate_statistics(
            O[sl], NO[sl], innovation[sl], DT[sl], T,
        )
        rho_chunks.append(rho)
        innovation_rho_chunks.append(innovation_rho)
        mismatch_chunks.append(mismatch)

        if gate_rho > 0:
            finite = th.isfinite(t_re[sl]) & th.isfinite(rho)
            lam = th.where(
                finite,
                th.exp(-(rho / gate_rho).square()),
                th.zeros_like(rho),
            )
            gated = th.where(
                finite,
                lam * t_re[sl] + (1.0 - lam) * t_fd[sl],
                t_fd[sl],
            )
            lambda_chunks.append(lam)
            gate_chunks.append(gated)
            fallback_chunks.append(~finite)

    empty = DT.new_empty((0, 1))
    aux = {
        "rho": th.cat(rho_chunks) if rho_chunks else empty,
        "innovation_rho": (
            th.cat(innovation_rho_chunks) if innovation_rho_chunks else empty
        ),
        "mismatch_fraction": (
            th.cat(mismatch_chunks) if mismatch_chunks else empty
        ),
        "gate_batch_size": batch_size,
        "gate_chunks": (n + batch_size - 1) // batch_size,
    }
    if gate_rho > 0:
        aux.update({
            "lam": th.cat(lambda_chunks) if lambda_chunks else empty,
            "gate_fallback": (
                th.cat(fallback_chunks)
                if fallback_chunks
                else empty.to(dtype=th.bool)
            ),
        })
        target = th.cat(gate_chunks) if gate_chunks else empty
    else:
        aux.update({"lam": None, "gate_fallback": None})
        target = None
    return target, aux


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
    DN_t = th.as_tensor(DN, dtype=th.float32, device=dev)
    keep = 1.0 - DN_t
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
        # Match CTSAC._finite_difference_target exactly, including the additive
        # denominator epsilon rather than clamping the rescaled duration.
        u = DT_t * float(algo.time_rescale)
        frac = (th.exp(-beta * u) * V(NO_t) - v_cur) / (u + 1e-8)
        t_fd = R_t + keep * (v_cur + frac)

        t_gate, gate_aux = _chunked_gate(
            algo, O_t, NO_t, innov, DT_t, T, gate_rho, t_re, t_fd,
        )

    out = {"or": t_or, "quad": t_quad, "re": t_re, "fd": t_fd}
    if t_gate is not None:
        out["gate"] = t_gate
    aux = {
        "dv_or": dv_or,
        "model_seconds": model_seconds,
        "dt": DT_t,
        "nonterminal": DN_t < 0.5,
        "fd_anchor_finite": th.isfinite(t_fd),
        **gate_aux,
    }
    return out, aux


def _estats(eps, dv_or, nonterminal, prefix, d):
    active = nonterminal.cpu().numpy().ravel().astype(bool)
    e_all = eps.cpu().numpy().ravel().astype(np.float64)[active]
    finite = np.isfinite(e_all)
    d[f"{prefix}_nonfinite_frac"] = (
        round(float(np.mean(~finite)), 6) if len(e_all) else float("nan")
    )
    d[f"{prefix}_finite_n"] = int(finite.sum())
    e = e_all[finite]
    if not len(e):
        for k in ("rms", "bias", "p95", "p99", "rel"):
            d[f"{prefix}_{k}"] = float("nan")
        return
    d[f"{prefix}_rms"] = round(rms(e), 4)
    d[f"{prefix}_bias"] = round(float(np.mean(e)), 4)
    d[f"{prefix}_p95"] = round(float(np.percentile(np.abs(e), 95)), 4)
    d[f"{prefix}_p99"] = round(float(np.percentile(np.abs(e), 99)), 4)
    dv_all = dv_or.cpu().numpy().ravel().astype(np.float64)[active]
    paired = finite & np.isfinite(dv_all)
    e_rel = e_all[paired]
    dv = dv_all[paired]
    d[f"{prefix}_rel"] = (
        round(rms(e_rel) / (rms(dv) + 1e-9), 4)
        if len(dv)
        else float("nan")
    )


def _tensor_fraction(mask, active=None):
    values = mask.reshape(-1)
    if active is not None:
        values = values[active.reshape(-1)]
    return float(values.float().mean()) if values.numel() else float("nan")


def audit_transitions(algo, env, tup, gate_rho):
    targets, aux = build_targets(algo, env, *tup, gate_rho=gate_rho)
    nonterminal = aux["nonterminal"]
    d = {
        "n": len(tup[0]),
        "n_nonterminal": int(nonterminal.sum().item()),
        "terminal_frac": _tensor_fraction(~nonterminal),
        "gate_batch_size": aux["gate_batch_size"],
        "gate_chunks": aux["gate_chunks"],
        "fd_anchor_nonfinite_frac": _tensor_fraction(
            ~aux["fd_anchor_finite"]
        ),
        "fd_anchor_nonfinite_nonterminal_frac": _tensor_fraction(
            ~aux["fd_anchor_finite"], nonterminal,
        ),
    }
    finite_dv = th.isfinite(aux["dv_or"])
    d["oracle_increment_nonfinite_frac"] = _tensor_fraction(
        ~finite_dv, nonterminal,
    )
    for name in ("quad", "re", "gate", "fd"):
        if name in targets:
            _estats(
                targets[name] - targets["or"], aux["dv_or"], nonterminal,
                f"eps_{name}", d,
            )
    rho_all = aux["rho"].cpu().numpy().ravel()
    rho_finite = rho_all[np.isfinite(rho_all)]
    d["rho_nonfinite_frac"] = (
        round(float(np.mean(~np.isfinite(rho_all))), 6)
        if len(rho_all)
        else float("nan")
    )
    d["rho_med"] = (
        round(float(np.median(rho_finite)), 4)
        if len(rho_finite)
        else float("nan")
    )
    d["rho_p95"] = (
        round(float(np.percentile(rho_finite, 95)), 4)
        if len(rho_finite)
        else float("nan")
    )
    mismatch = aux["mismatch_fraction"]
    innovation_rho = aux["innovation_rho"].cpu().numpy().ravel()
    innovation_valid = mismatch.cpu().numpy().ravel() > 0.0
    innovation_rho = innovation_rho[
        innovation_valid & np.isfinite(innovation_rho)
    ]
    d["innovation_rho_med"] = (
        round(float(np.median(innovation_rho)), 4)
        if len(innovation_rho)
        else float("nan")
    )
    d["mismatch_fraction_mean"] = (
        round(float(mismatch.mean()), 6)
        if mismatch.numel()
        else float("nan")
    )
    d["innovation_valid_frac"] = _tensor_fraction(mismatch > 0.0)
    if aux["lam"] is not None:
        d["lambda_mean"] = round(float(aux["lam"].mean()), 4)
        d["gate_fallback_frac"] = _tensor_fraction(aux["gate_fallback"])
        d["gate_fallback_nonterminal_frac"] = _tensor_fraction(
            aux["gate_fallback"], nonterminal,
        )
        # Trainer-compatible name plus the more descriptive audit name above.
        d["reanchor_nonfinite_frac"] = d["gate_fallback_frac"]
        d["reanchor_nonfinite_nonterminal_frac"] = (
            d["gate_fallback_nonterminal_frac"]
        )
    d["transport_seconds_mean"] = round(
        float(aux["model_seconds"].mean()), 6
    )
    innovation_seconds = th.where(
        mismatch > 0.0, aux["dt"], th.zeros_like(aux["dt"])
    )
    d["model_seconds_mean"] = round(
        float((innovation_seconds + aux["model_seconds"]).mean()), 6
    )
    d["long_frac"] = round(
        float((aux["dt"] > float(algo.dt_default)).float().mean()), 6
    )
    return d


def _print_row(tag, m):
    print(f"{tag}: rel quad={m['eps_quad_rel']:.3f} re={m['eps_re_rel']:.3f} "
          f"gate={m.get('eps_gate_rel', float('nan')):.3f} "
          f"fd={m['eps_fd_rel']:.3f} | p99 quad={m['eps_quad_p99']:.3f} "
          f"re={m['eps_re_p99']:.3f} | rho_med={m['rho_med']:.3f} "
          f"fallback={m.get('gate_fallback_frac', float('nan')):.3f} "
          f"nf(re/gate)={m['eps_re_nonfinite_frac']:.3f}/"
          f"{m.get('eps_gate_nonfinite_frac', float('nan')):.3f}", flush=True)


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

    print("\n=== median audit health fractions, by distribution ===")
    health_keys = [
        "gate_fallback_frac", "eps_quad_nonfinite_frac",
        "eps_re_nonfinite_frac", "eps_gate_nonfinite_frac",
        "eps_fd_nonfinite_frac", "fd_anchor_nonfinite_frac",
    ]
    health_hdr = ["fallback", "nf_quad", "nf_re", "nf_gate", "nf_fd", "nf_anchor"]
    print(f"{'dist':10s}" + "".join(f"{h:>10s}" for h in health_hdr))
    for name in dists:
        sub = [r for r in rows if r["distribution"] == name]
        med = lambda k: float(np.nanmedian([r.get(k, np.nan) for r in sub]))
        print(f"{name:10s}" + "".join(f"{med(k):10.3f}" for k in health_keys))


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
            "gate_batch_size": int(algo.batch_size),
            "gate_chunking": "contiguous_input_order",
            "beta": float(algo.beta), "dt_default": float(algo.dt_default),
            "checkpoints": meta_ckpts}
    _write(rows, meta, args.out)
    _summary(rows)


if __name__ == "__main__":
    main()
