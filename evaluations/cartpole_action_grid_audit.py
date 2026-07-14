#!/usr/bin/env python
"""Within-state ACTION-ordering audit for the learned-dynamics cartpole.

The target audit (evaluations/cartpole_target_audit.py) samples one action per
state, so its Spearman metric ranks different STATES. The actor instead needs
the correct ordering of alternative ACTIONS competing at the same state. Here,
at representative states, a dense 1-D action grid is evaluated under both the
learned published dynamics (x_hat) and true MuJoCo physics (x_mj), the
action-conditioned reward is recomputed for every action (the earlier
replay_pi probe reused the logged action's reward and was therefore
inconsistent), and the two critic targets are formed exactly as the
substep-quadrature trainer target does (keep=1: swingup never terminates):

    t_L(a) = r(x,a) + V_tgt(x_hat(x,a)) - beta * V_tgt(x)
    t_O(a) = r(x,a) + V_tgt(x_mj (x,a)) - beta * V_tgt(x)

Alongside the targets, the grid is scored by the CURRENT live critic
Q(a) = min_i Q_i(x,a) (the actor's objective) and, when the entropy
temperature is known, by the pointwise SAC actor score
S(a) = Q(a) - alpha * log pi(a|x).

Per state the audit separates three failure locations:

  1. rank(t_L) vs rank(t_O) disagree (spearman_tl_to, pairwise_agree,
     regret_lgreedy): the dynamics-mediated target itself mis-orders actions
     -- dynamics error is directly misleading action learning.
  2. targets agree but Q mis-orders (spearman_q_to, regret_qgreedy,
     qslope_sign_agree): critic extrapolation/optimization is the bottleneck.
  3. Q orders correctly but the policy action is poor (regret_pi,
     regret_pi_sampled): actor optimization / entropy / exploration is the
     bottleneck.

All regrets are ORACLE regrets, t_O(best) - t_O(chosen), reported raw (target
units) and normalized by the oracle target range over the grid so states with
exploded value scales (seed 1) remain comparable.

Measurement hygiene (requested before the causal continuation experiment):
every probe derives its RNG from an isolated SeedSequence keyed by
(dataset-seed, probe, seed, step, distribution), and the artifacts record the
checkpoint paths and SHA-256 hashes actually read. Recomputed rewards are
validated against logged replay rewards at the LOGGED per-row dt (cartpole is
constraint-free, so the replay reward must reproduce to float32 precision).

    python -m evaluations.cartpole_action_grid_audit \
        --out results/cartpole_action_grid_audit
    python -m evaluations.cartpole_action_grid_audit --trajectory \
        --seeds 1,3,5,11 --min-step 100000 --max-step 400000 \
        --out results/cartpole_action_grid_trajectory
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os

import numpy as np
import torch as th
from dm_control import rl
from scipy.stats import spearmanr

from models.port_hamiltonian import integrate_drift
from evaluations.hamiltonian_recovery import _set_state
from evaluations.cartpole_critic_audit import (
    build_algorithm, collect_fixed_dataset, final_return, rms, CHAIN, SEEDS,
)
from evaluations.cartpole_target_audit import (
    collect_policy_dataset, periodic_checkpoints, _set_alpha, SAVED,
)

# Probe ids for RNG isolation (SeedSequence keys; never reuse across probes).
_PROBE = {"ou_states": 1, "replay_states": 2, "onpolicy_states": 3,
          "policy_samples": 4, "reward_validation": 5, "onpolicy_env": 6,
          "onpolicy_torch": 7}
_DIST_ID = {"replay": 1, "onpolicy": 2, "ou": 3}


def _rng(dataset_seed, probe, seed=0, step=0, dist=0):
    return np.random.default_rng(
        np.random.SeedSequence([dataset_seed, _PROBE[probe], seed, step, dist]))


def _torch_seed(dataset_seed, probe, seed=0, step=0, dist=0):
    ss = np.random.SeedSequence([dataset_seed, _PROBE[probe], seed, step, dist])
    return int(ss.generate_state(1)[0])


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------ physics: transition + true reward ------------------------


def action_grid(env, n):
    """Dense 1-D action grid over the Box, endpoints included."""
    low = np.asarray(env.action_space.low, dtype=np.float64).ravel()
    high = np.asarray(env.action_space.high, dtype=np.float64).ravel()
    if low.size != 1:
        raise ValueError("dense-grid action ordering audit is 1-D (cartpole) only")
    return np.linspace(low[0], high[0], n, dtype=np.float64).reshape(-1, 1)


def mujoco_transition_reward(env, obs, actions, dt):
    """Like hamiltonian_recovery.mujoco_transition, but also returns the true
    action-conditioned reward: ctrl is held at the (clipped) action for
    round(dt/phys_dt) physics substeps and the task reward is read at the
    endpoint -- exactly DMCContinuousEnv._step_physics's reward path
    (return_reward_increment off). ``dt`` may be a scalar or a per-row array.
    Requires raw_state_obs and a task whose reward has no per-step
    accumulation state (true for cartpole-swingup). Rows whose integration
    raises a PhysicsError are returned as NaN."""
    assert bool(getattr(env, "raw_state_obs", False)), (
        "mujoco_transition_reward requires raw_state_obs envs"
    )
    physics = env._env.physics
    task = getattr(env._env, "task", None) or env._env._task
    model, data = physics.model, physics.data
    nq, nv = int(model.nq), int(model.nv)
    obs = np.asarray(obs, dtype=np.float64).reshape(-1, nq + nv)
    actions = np.asarray(actions, dtype=np.float64).reshape(obs.shape[0], -1)
    dt = np.broadcast_to(np.asarray(dt, dtype=np.float64).ravel(), (obs.shape[0],))
    low = np.asarray(env.action_space.low, dtype=np.float64)
    high = np.asarray(env.action_space.high, dtype=np.float64)
    phys_dt = float(model.opt.timestep)

    saved = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), float(data.time))
    out = np.full_like(obs, np.nan, dtype=np.float64)
    rew = np.full(obs.shape[0], np.nan, dtype=np.float64)
    try:
        for i in range(obs.shape[0]):
            _set_state(data, nq, nq, obs[i])
            data.ctrl[:] = np.clip(actions[i], low, high)
            # deterministic, state-consistent constraint-solver warm start
            # (same discipline as mujoco_transition)
            physics.forward()
            data.qacc_warmstart[:] = data.qacc
            try:
                for _ in range(max(1, int(round(dt[i] / phys_dt)))):
                    physics.step()
            except rl.control.PhysicsError:
                continue
            out[i, :nq] = data.qpos[:nq]
            out[i, nq:] = data.qvel[:nv]
            rew[i] = float(task.get_reward(physics))
    finally:
        data.qpos[:] = saved[0]; data.qvel[:] = saved[1]
        data.ctrl[:] = saved[2]; data.time = saved[3]
        physics.forward()
    return out.astype(np.float32), rew.astype(np.float32)


# ------------------------ grid tables ------------------------


def evaluate_tables(algo, env, S, grid, extras, beta, dt_default, alpha,
                    grid_mj=None):
    """Score every (state, action) pair of grid + per-state extra actions.

    S (n_s, D); grid (n_g, 1) shared across states; extras (n_s, n_e, 1).
    Returns (tables, grid_mj): tables of (n_s, n_g + n_e) float64 arrays
    [t_l, t_o, q, s_score(None w/o alpha), r, eps, a] plus v_cur (n_s,);
    grid_mj = (x_mj, r) for the grid block, reusable when S is shared.
    """
    dev = algo.device
    n_s, n_g, n_e = len(S), len(grid), extras.shape[1]
    S_g = np.repeat(S, n_g, axis=0)
    A_g = np.tile(grid, (n_s, 1))
    S_e = np.repeat(S, n_e, axis=0)
    A_e = extras.reshape(n_s * n_e, -1)

    if grid_mj is None:
        grid_mj = mujoco_transition_reward(env, S_g, A_g, dt_default)
    xg_mj, rg = grid_mj
    xe_mj, re = mujoco_transition_reward(env, S_e, A_e, dt_default)

    S_flat = np.concatenate([S_g, S_e], axis=0)
    A_flat = np.concatenate([A_g, A_e], axis=0).astype(np.float32)
    x_mj = np.concatenate([xg_mj, xe_mj], axis=0)

    with th.no_grad():
        S_t = th.as_tensor(S, dtype=th.float32, device=dev)
        Sf_t = th.as_tensor(S_flat, dtype=th.float32, device=dev)
        Af_t = th.as_tensor(A_flat, dtype=th.float32, device=dev)
        x_hat = integrate_drift(algo.dynamics_target_model.drift, Sf_t, Af_t,
                                dt_default, max_step=algo._integration_max_step())
        v_cur = algo.model.target_value(S_t).cpu().numpy().astype(np.float64).ravel()
        v_hat = algo.model.target_value(x_hat).cpu().numpy().astype(np.float64).ravel()
        v_mj = algo.model.target_value(
            th.as_tensor(x_mj, dtype=th.float32, device=dev)
        ).cpu().numpy().astype(np.float64).ravel()
        q = algo.model.min_q(Sf_t, Af_t).cpu().numpy().astype(np.float64).ravel()
        log_pi = None
        if hasattr(algo.model.actor, "log_prob"):
            log_pi = algo.model.actor.log_prob(Sf_t, Af_t)
            log_pi = log_pi.cpu().numpy().astype(np.float64).ravel()

    def stitch(flat):
        flat = np.asarray(flat, dtype=np.float64).ravel()
        return np.hstack([flat[: n_s * n_g].reshape(n_s, n_g),
                          flat[n_s * n_g:].reshape(n_s, n_e)])

    r = stitch(np.concatenate([rg, re]))
    v_hat, v_mj, q = stitch(v_hat), stitch(v_mj), stitch(q)
    vc = v_cur.reshape(n_s, 1)
    tables = {
        "t_l": r + v_hat - beta * vc,
        "t_o": r + v_mj - beta * vc,
        "q": q,
        "s_score": (q - alpha * stitch(log_pi))
        if (alpha is not None and log_pi is not None) else None,
        "r": r,
        "eps": v_hat - v_mj,
        "a": stitch(np.concatenate([A_g.ravel(), A_e.ravel()])),
        "v_cur": v_cur,
    }
    return tables, grid_mj


def q_action_grad(algo, S, a_pi):
    """d min-Q / da at the deterministic policy action (the actor's local
    learning signal)."""
    dev = algo.device
    S_t = th.as_tensor(S, dtype=th.float32, device=dev)
    a_t = th.as_tensor(a_pi, dtype=th.float32, device=dev).requires_grad_(True)
    qv = algo.model.min_q(S_t, a_t)
    (g,) = th.autograd.grad(qv.sum(), a_t)
    return g.detach().cpu().numpy().astype(np.float64)


def policy_extras(algo, env, S, n_samples, torch_seed, grad_h):
    """Per-state extra actions: [a_pi, a_pi-h, a_pi+h (clipped into bounds),
    n_samples policy draws]. Returns (extras (n_s, 3+n_samples, 1), a_pi)."""
    dev = algo.device
    low = float(env.action_space.low.ravel()[0])
    high = float(env.action_space.high.ravel()[0])
    S_t = th.as_tensor(S, dtype=th.float32, device=dev)
    with th.no_grad():
        a_pi, _ = algo.model.act(S_t, deterministic=True)
        th.manual_seed(torch_seed)  # isolated draw for the sampled-policy regret
        S_rep = S_t.repeat_interleave(n_samples, dim=0)
        a_smp, _ = algo.model.act(S_rep, deterministic=False)
    a_pi = a_pi.cpu().numpy().astype(np.float64).reshape(len(S), 1)
    a_smp = a_smp.cpu().numpy().astype(np.float64).reshape(len(S), n_samples, 1)
    a_lo = np.clip(a_pi - grad_h, low, high)
    a_hi = np.clip(a_pi + grad_h, low, high)
    extras = np.concatenate(
        [a_pi[:, None, :], a_lo[:, None, :], a_hi[:, None, :], a_smp], axis=1)
    return extras, a_pi


# ------------------------ per-state metrics ------------------------


def _spearman(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b)[0])


def _pairwise_agreement(tl, to):
    """Fraction of action pairs ordered consistently, excluding pairs the
    oracle leaves (numerically) tied."""
    dl = tl[:, None] - tl[None, :]
    do = to[:, None] - to[None, :]
    iu = np.triu_indices(len(tl), k=1)
    dl, do = dl[iu], do[iu]
    keep = np.abs(do) > 1e-9 * max(float(np.ptp(to)), 1e-300)
    if keep.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(dl[keep]) == np.sign(do[keep])))


def _slope(t_row, a_row, i_lo, i_hi):
    da = a_row[i_hi] - a_row[i_lo]
    if not np.isfinite(da) or abs(da) < 1e-9:
        return float("nan")
    return float((t_row[i_hi] - t_row[i_lo]) / da)


def _sign_agree(x, y):
    if not (np.isfinite(x) and np.isfinite(y)):
        return float("nan")
    return float(np.sign(x) == np.sign(y))


def state_row(i, tab, n_g, n_samples, dq_da, k_top):
    """All within-state ordering metrics for state i. Grid indices [0, n_g);
    extras: n_g -> a_pi, n_g+1 -> a_pi-h, n_g+2 -> a_pi+h, then samples."""
    t_l, t_o, q = tab["t_l"][i], tab["t_o"][i], tab["q"][i]
    a, r, eps = tab["a"][i], tab["r"][i], tab["eps"][i]
    s_sc = tab["s_score"][i] if tab["s_score"] is not None else None
    g = slice(0, n_g)
    i_pi, i_lo, i_hi = n_g, n_g + 1, n_g + 2
    i_smp = slice(n_g + 3, n_g + 3 + n_samples)

    fin = np.isfinite(t_l[g]) & np.isfinite(t_o[g]) & np.isfinite(q[g])
    row = {"finite_frac": round(float(np.mean(fin)), 4)}
    if fin.sum() < 5:
        return row
    tl_g, to_g, q_g, a_g = t_l[g][fin], t_o[g][fin], q[g][fin], a[g][fin]

    # oracle-best reference over every finite evaluated action (grid + extras)
    fin_all = np.isfinite(t_o)
    to_best = float(np.max(t_o[fin_all]))
    to_rng = float(np.ptp(to_g))
    nrm = to_rng if to_rng > 1e-9 else float("nan")

    # 1) learned-target vs oracle-target ordering (dynamics-mediated)
    i_l, i_o = int(np.argmax(tl_g)), int(np.argmax(to_g))
    top_l = np.argsort(-tl_g)[:k_top]
    top_o = np.argsort(-to_g)[:k_top]
    reg_l = to_best - float(to_g[i_l])
    row.update({
        "spearman_tl_to": round(_spearman(tl_g, to_g), 4),
        "pairwise_agree": round(_pairwise_agreement(tl_g, to_g), 4),
        "topk_overlap": round(len(set(top_l) & set(top_o)) / k_top, 4),
        "best_a_oracle": round(float(a_g[i_o]), 4),
        "best_a_learned": round(float(a_g[i_l]), 4),
        "argmax_disagree": int(i_l != i_o),
        "argmax_dist": round(float(abs(a_g[i_l] - a_g[i_o])), 4),
        "regret_lgreedy": round(reg_l, 6),
        "regret_lgreedy_norm": round(reg_l / nrm, 4),
    })

    # 2) critic ordering (live min-Q, the actor's objective)
    reg_q = to_best - float(to_g[int(np.argmax(q_g))])
    row.update({
        "spearman_q_to": round(_spearman(q_g, to_g), 4),
        "spearman_q_tl": round(_spearman(q_g, tl_g), 4),
        "regret_qgreedy": round(reg_q, 6),
        "regret_qgreedy_norm": round(reg_q / nrm, 4),
    })

    # local slopes at the policy mean (finite differences over the +-h probes;
    # dq_da is the exact autograd critic slope the actor ascends)
    sl_l = _slope(t_l, a, i_lo, i_hi)
    sl_o = _slope(t_o, a, i_lo, i_hi)
    row.update({
        "a_pi": round(float(a[i_pi]), 4),
        "slope_l_pi": round(sl_l, 6),
        "slope_o_pi": round(sl_o, 6),
        "slope_q_pi": round(float(dq_da[i]), 6),
        "tslope_sign_agree": _sign_agree(sl_l, sl_o),
        "qslope_sign_agree": _sign_agree(float(dq_da[i]), sl_o),
    })

    # 3) actor: pointwise SAC score and the policy's realized choices
    if s_sc is not None:
        s_g = s_sc[g][fin]
        reg_s = to_best - float(to_g[int(np.argmax(s_g))])
        row.update({
            "spearman_s_to": round(_spearman(s_g, to_g), 4),
            "regret_sgreedy_norm": round(reg_s / nrm, 4),
        })
    reg_pi = to_best - float(t_o[i_pi])
    smp = t_o[i_smp]
    smp = smp[np.isfinite(smp)]
    reg_smp = float(np.mean(to_best - smp)) if len(smp) else float("nan")
    row.update({
        "regret_pi": round(reg_pi, 6),
        "regret_pi_norm": round(reg_pi / nrm, 4),
        "regret_pi_sampled_norm": round(reg_smp / nrm, 4),
    })

    # context: how much ordering signal exists, and where it comes from
    row.update({
        "v_cur": round(float(tab["v_cur"][i]), 4),
        "to_range": round(to_rng, 6),
        "tl_range": round(float(np.ptp(tl_g)), 6),
        "r_range": round(float(np.ptp(r[g][fin])), 6),
        "r_to_range_ratio": round(float(np.ptp(r[g][fin])) / nrm, 4),
        "eps_rms_grid": round(rms(eps[g][fin]), 6),
    })
    return row


# ------------------------ aggregation / io ------------------------

_MED_KEYS = (
    "spearman_tl_to", "pairwise_agree", "topk_overlap", "argmax_dist",
    "regret_lgreedy_norm", "spearman_q_to", "spearman_q_tl",
    "regret_qgreedy_norm", "spearman_s_to", "regret_sgreedy_norm",
    "regret_pi_norm", "regret_pi_sampled_norm", "r_to_range_ratio",
)
_P90_KEYS = ("regret_lgreedy_norm", "regret_qgreedy_norm", "regret_pi_norm",
             "regret_pi_sampled_norm")
_P10_KEYS = ("spearman_tl_to", "spearman_q_to")
_FRAC_KEYS = ("argmax_disagree", "tslope_sign_agree", "qslope_sign_agree")


def summarize_states(rows):
    """Nan-aware aggregate of per-state rows -> one summary dict."""
    col = lambda k: np.asarray([r.get(k, np.nan) for r in rows], dtype=np.float64)
    out = {"n_states": len(rows),
           "mean_finite_frac": round(float(np.nanmean(col("finite_frac"))), 4)}
    with np.errstate(all="ignore"):
        for k in _MED_KEYS:
            v = col(k)
            out[f"med_{k}"] = (round(float(np.nanmedian(v)), 4)
                               if np.isfinite(v).any() else float("nan"))
        for k in _P90_KEYS:
            v = col(k)
            out[f"p90_{k}"] = (round(float(np.nanpercentile(v[np.isfinite(v)], 90)), 4)
                               if np.isfinite(v).any() else float("nan"))
        for k in _P10_KEYS:
            v = col(k)
            out[f"p10_{k}"] = (round(float(np.nanpercentile(v[np.isfinite(v)], 10)), 4)
                               if np.isfinite(v).any() else float("nan"))
        for k in _FRAC_KEYS:
            v = col(k)
            out[f"frac_{k}"] = (round(float(np.nanmean(v)), 4)
                                if np.isfinite(v).any() else float("nan"))
    return out


def _write_csv(rows, path):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _write_outputs(state_rows, summary_rows, meta, out):
    _write_csv(state_rows, out + "_states.csv")
    _write_csv(summary_rows, out + ".csv")
    json.dump(summary_rows, open(out + ".json", "w"), indent=2)
    json.dump(meta, open(out + "_meta.json", "w"), indent=2)
    print(f"\nwrote {out}.csv/.json ({len(summary_rows)} summaries), "
          f"{out}_states.csv ({len(state_rows)} states), {out}_meta.json")


def _checkpoint_hashes(model_path, buffer_path=None):
    d = {"model_path": model_path, "model_sha256": sha256_file(model_path)}
    sidecar = os.path.splitext(model_path)[0] + ".dynamics.pth"
    if os.path.exists(sidecar):
        d["dynamics_path"] = sidecar
        d["dynamics_sha256"] = sha256_file(sidecar)
    if buffer_path and os.path.exists(buffer_path):
        d["buffer_path"] = buffer_path
        d["buffer_sha256"] = sha256_file(buffer_path)
    return d


def _current_alpha(algo):
    if algo.log_alpha is not None:
        return float(th.exp(algo.log_alpha.detach()).cpu())
    at = getattr(algo, "alpha_tensor", None)
    return float(at) if at is not None else None


def _load_buffer_states(ck, n, rng):
    """Replay states + logged (action, reward, dt) rows via an isolated rng."""
    d = np.load(f"{ck}/buffer.npz", allow_pickle=False)
    pos, full = int(d["pos"]), bool(d["full"])
    valid = d["observations"].shape[0] if full else pos
    idx = rng.choice(valid, size=min(n, valid), replace=False)
    return (d["observations"][idx, 0, :].astype(np.float32),
            d["actions"][idx, 0, :].astype(np.float32),
            d["rewards"][idx, 0].astype(np.float32),
            d["dt"][idx, 0].astype(np.float32))


def validate_replay_rewards(env, O, A, R_logged, DT):
    """Recompute the logged replay rewards at the LOGGED per-row dt. Cartpole
    is constraint-free, so up to the float32 storage of the buffer state the
    recomputation must reproduce the logged reward; a large gap means the
    audit env no longer matches the env that produced the checkpoint."""
    _, r_rec = mujoco_transition_reward(env, O, A, DT)
    diff = np.abs(r_rec.astype(np.float64) - R_logged.astype(np.float64))
    diff = diff[np.isfinite(diff)]
    return {"n": int(len(diff)), "mae": round(float(np.mean(diff)), 6),
            "p99_abs": round(float(np.percentile(diff, 99)), 6),
            "max_abs": round(float(np.max(diff)), 6)}


# ------------------------ audit drivers ------------------------


def audit_state_set(algo, env, S, beta, dt_default, alpha, grid, n_samples,
                    torch_seed, grad_h, k_top, grid_mj=None):
    """Run the full within-state audit on one state set; returns
    (per-state rows, summary, grid_mj cache)."""
    extras, a_pi = policy_extras(algo, env, S, n_samples, torch_seed, grad_h)
    tab, grid_mj = evaluate_tables(algo, env, S, grid, extras, beta,
                                   dt_default, alpha, grid_mj=grid_mj)
    dq_da = q_action_grad(algo, S, a_pi).ravel()
    rows = []
    for i in range(len(S)):
        row = {"state_idx": i}
        row.update({f"obs{j}": round(float(S[i, j]), 5) for j in range(S.shape[1])})
        row.update(state_row(i, tab, len(grid), n_samples, dq_da, k_top))
        rows.append(row)
    return rows, summarize_states(rows), grid_mj


def _print_summary_line(tag, m):
    def f(k):
        v = m.get(k)
        return "  nan" if v is None or not np.isfinite(v) else f"{v:5.2f}"
    print(f"{tag}: rho_T={f('med_spearman_tl_to')} agree={f('med_pairwise_agree')} "
          f"topk={f('med_topk_overlap')} dis={f('frac_argmax_disagree')} "
          f"regL={f('med_regret_lgreedy_norm')}/{f('p90_regret_lgreedy_norm')} "
          f"rho_Q={f('med_spearman_q_to')} regQ={f('med_regret_qgreedy_norm')} "
          f"regPi={f('med_regret_pi_norm')} qslope={f('frac_qslope_sign_agree')}",
          flush=True)


def _subsample(S, n, rng):
    if len(S) <= n:
        return S
    return S[rng.choice(len(S), size=n, replace=False)]


def run_final(algo, env, args, grid, seeds):
    beta, dt_default = float(algo.beta), float(algo.dt_default)
    grad_h = args.grad_h or float(grid[1, 0] - grid[0, 0])
    ds = args.dataset_seed

    OU_O, _, _, _, _ = collect_fixed_dataset(env, args.n_collect, ds)
    S_ou = _subsample(OU_O, args.n_states, _rng(ds, "ou_states"))
    ou_grid_mj = None  # oracle grid transitions at shared states: seed-independent

    state_rows, summary_rows, meta_ckpts = [], [], {}
    for s in seeds:
        ck = f"{CHAIN}/seed_{s}/checkpoint"
        algo.load(f"{ck}/model.pth")
        ts = th.load(f"{ck}/train_state.pt", map_location="cpu", weights_only=False)
        step = int(ts.get("counters", {}).get("num_timesteps", -1))
        _set_alpha(algo, ts)
        alpha = _current_alpha(algo) if "log_alpha" in ts else None
        meta = _checkpoint_hashes(f"{ck}/model.pth", f"{ck}/buffer.npz")
        meta.update({"step": step, "alpha": alpha})

        rO, rA, rR, rDT = _load_buffer_states(
            ck, max(args.n_states, args.validate_replay_rewards),
            _rng(ds, "replay_states", seed=s))
        if args.validate_replay_rewards > 0:
            m = min(args.validate_replay_rewards, len(rO))
            meta["reward_validation"] = validate_replay_rewards(
                env, rO[:m], rA[:m], rR[:m], rDT[:m])
        meta_ckpts[f"seed_{s}"] = meta

        th.manual_seed(_torch_seed(ds, "onpolicy_torch", seed=s))
        oO, _, _, _ = collect_policy_dataset(
            algo, env, args.n_collect,
            seed=_torch_seed(ds, "onpolicy_env", seed=s) % (2 ** 31))
        dists = [
            ("replay", rO[: args.n_states], None),
            ("onpolicy", _subsample(oO, args.n_states,
                                    _rng(ds, "onpolicy_states", seed=s)), None),
            ("ou", S_ou, ou_grid_mj),
        ]
        dists = [d for d in dists if d[0] in args.dists]
        for name, S, gmj in dists:
            rows, summ, gmj = audit_state_set(
                algo, env, np.asarray(S, np.float32), beta, dt_default, alpha,
                grid, args.n_policy_samples,
                _torch_seed(ds, "policy_samples", seed=s, dist=_DIST_ID[name]),
                grad_h, args.k_top, grid_mj=gmj)
            if name == "ou":
                ou_grid_mj = gmj
            head = {"seed": s, "distribution": name, "checkpoint_step": step}
            state_rows += [{**head, **r} for r in rows]
            summary_rows.append({**head, "alpha": alpha,
                                 "logged_final_return": round(final_return(s), 2),
                                 **summ})
            _print_summary_line(f"seed {s:2d} @{step} [{name:9s}]", summ)
    return state_rows, summary_rows, meta_ckpts


def run_trajectory(algo, env, args, grid, seeds):
    beta, dt_default = float(algo.beta), float(algo.dt_default)
    grad_h = args.grad_h or float(grid[1, 0] - grid[0, 0])
    ds = args.dataset_seed

    OU_O, _, _, _, _ = collect_fixed_dataset(env, args.n_collect, ds)
    S_ou = _subsample(OU_O, args.n_states, _rng(ds, "ou_states"))
    ou_grid_mj = None

    state_rows, summary_rows, meta_ckpts = [], [], {}
    for s in seeds:
        for step, path in periodic_checkpoints(s):
            if not (args.min_step <= step <= args.max_step):
                continue
            algo.load(path)  # model + dynamics sidecar; no train_state -> no alpha
            meta_ckpts[f"seed_{s}_step_{step}"] = {
                **_checkpoint_hashes(path), "step": step}
            th.manual_seed(_torch_seed(ds, "onpolicy_torch", seed=s, step=step))
            oO, _, _, _ = collect_policy_dataset(
                algo, env, args.n_collect,
                seed=_torch_seed(ds, "onpolicy_env", seed=s, step=step) % (2 ** 31))
            dists = [
                ("onpolicy", _subsample(oO, args.n_states,
                                        _rng(ds, "onpolicy_states", seed=s,
                                             step=step)), None),
                ("ou", S_ou, ou_grid_mj),
            ]
            dists = [d for d in dists if d[0] in args.dists]
            for name, S, gmj in dists:
                rows, summ, gmj = audit_state_set(
                    algo, env, np.asarray(S, np.float32), beta, dt_default,
                    None, grid, args.n_policy_samples,
                    _torch_seed(ds, "policy_samples", seed=s, step=step,
                                dist=_DIST_ID[name]),
                    grad_h, args.k_top, grid_mj=gmj)
                if name == "ou":
                    ou_grid_mj = gmj
                head = {"seed": s, "step": step, "distribution": name}
                state_rows += [{**head, **r} for r in rows]
                summary_rows.append({**head, **summ})
                _print_summary_line(f"seed {s:2d} @{step:>7d} [{name:8s}]", summ)
    return state_rows, summary_rows, meta_ckpts


def _final_table(summary_rows, dists):
    print("\n=== median over seeds, by distribution ===")
    keys = ["med_spearman_tl_to", "med_pairwise_agree", "med_topk_overlap",
            "frac_argmax_disagree", "med_regret_lgreedy_norm",
            "med_spearman_q_to", "med_regret_qgreedy_norm",
            "med_regret_pi_norm", "frac_qslope_sign_agree"]
    hdr = ["rho_T", "agree", "topk", "dis", "regL", "rho_Q", "regQ", "regPi", "qslope"]
    print(f"{'dist':10s}" + "".join(f"{h:>8s}" for h in hdr))
    for name in dists:
        sub = [r for r in summary_rows if r["distribution"] == name]
        if not sub:
            continue
        med = lambda k: float(np.nanmedian([r.get(k, np.nan) for r in sub]))
        print(f"{name:10s}" + "".join(f"{med(k):8.3f}" for k in keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-states", type=int, default=256,
                    help="representative states per distribution")
    ap.add_argument("--n-actions", type=int, default=101, help="dense grid size")
    ap.add_argument("--k-top", type=int, default=10)
    ap.add_argument("--n-policy-samples", type=int, default=16,
                    help="policy draws per state for the sampled regret")
    ap.add_argument("--n-collect", type=int, default=2000,
                    help="env steps walked before subsampling states")
    ap.add_argument("--grad-h", type=float, default=None,
                    help="finite-difference half-width at a_pi (default: grid spacing)")
    ap.add_argument("--dataset-seed", type=int, default=20260713)
    ap.add_argument("--seeds", default="all", help="comma list or 'all'")
    ap.add_argument("--dists", default=None,
                    help="comma subset of replay,onpolicy,ou")
    ap.add_argument("--validate-replay-rewards", type=int, default=256,
                    help="replay rows for the logged-dt reward check (0 = off)")
    ap.add_argument("--trajectory", action="store_true",
                    help="sweep periodic checkpoints (onpolicy+ou) instead of final")
    ap.add_argument("--min-step", type=int, default=0)
    ap.add_argument("--max-step", type=int, default=10 ** 12)
    ap.add_argument("--out", default="results/cartpole_action_grid_audit")
    args = ap.parse_args()

    seeds = SEEDS if args.seeds == "all" else [int(x) for x in args.seeds.split(",")]
    default_dists = ("onpolicy", "ou") if args.trajectory else ("replay", "onpolicy", "ou")
    args.dists = tuple(args.dists.split(",")) if args.dists else default_dists

    algo, env = build_algorithm()
    grid = action_grid(env, args.n_actions)
    print(f"beta={float(algo.beta):.4f} dt_default={float(algo.dt_default):.4f} "
          f"grid={len(grid)} pts in [{grid[0, 0]:.2f}, {grid[-1, 0]:.2f}] "
          f"n_states={args.n_states} dists={','.join(args.dists)}", flush=True)

    runner = run_trajectory if args.trajectory else run_final
    state_rows, summary_rows, meta_ckpts = runner(algo, env, args, grid, seeds)

    meta = {
        "mode": "trajectory" if args.trajectory else "final",
        "chain": CHAIN, "saved": SAVED, "seeds": seeds,
        "beta": float(algo.beta), "dt_default": float(algo.dt_default),
        "physics_dt": float(env.physics_dt),
        "grid": {"n": args.n_actions, "low": float(grid[0, 0]),
                 "high": float(grid[-1, 0])},
        "n_states": args.n_states, "k_top": args.k_top,
        "n_policy_samples": args.n_policy_samples, "n_collect": args.n_collect,
        "grad_h": args.grad_h or float(grid[1, 0] - grid[0, 0]),
        "dataset_seed": args.dataset_seed, "dists": list(args.dists),
        "checkpoints": meta_ckpts,
    }
    _write_outputs(state_rows, summary_rows, meta, args.out)
    _final_table(summary_rows, args.dists)


if __name__ == "__main__":
    main()
