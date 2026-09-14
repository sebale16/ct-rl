#!/usr/bin/env python
"""True one-step advantage of a torque error, as a function of control interval.

No networks.  This is the ground-truth companion to
``evaluations.action_error_target_audit``: it measures, by exact rollout under
the analytical Xin-Kaneda controller, the quantity CT-SAC's ``Q(s,a)`` is
defined to price --

    A(s) = G(deviate for ONE control interval, then follow the law)
         - G(follow the law throughout)

with ``G`` the beta-discounted return over a fixed physical horizon.  Both
branches are deterministic, so each ``A(s)`` is exact and the only statistics
are over the state distribution (the protocol of
``.claude_scratch/advantage_n185.py``, and the source of the numbers quoted in
``docs/ct_sac_advantage_parameterization.md`` sec. 9).

Sweeping the control interval is the point: the horizon, the state set, the
torque error and the physics substep are all held fixed, so the only thing that
changes is how long the deviated action is applied before the law resumes.  The
advantage-parameterization claim predicts ``A`` proportional to the interval and
``A / T`` -- the advantage RATE -- flat.

    python -m evaluations.one_step_advantage_vs_dt \
        --intervals 0.001,0.002,0.005,0.01,0.02,0.05 --n 185 \
        --out results/one_step_advantage_vs_dt
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
from scipy import stats

os.environ.setdefault("MUJOCO_GL", "egl")

from common.utils import load_ct_hyperparams_from_table
from controllers.xin_kaneda import AcrobotParams, Gains, XinKanedaController
from environment.acrobot_xk import (
    DEFAULT_LYAPUNOV_K_D,
    DEFAULT_LYAPUNOV_K_P,
    DEFAULT_LYAPUNOV_K_V,
    DEFAULT_TORQUE_LIMIT,
)
from environment.dmc import DMCContinuousEnv

ENV_ID = "acrobot-swingup-xk"
MODE = (
    "xk_r3_eta0p23_fixed1ms_h2s_temp0p01_xkdot_q2dot4pi_logrecip_xkklrev"
    "_xkdemo200k_tau1p25e3_anneal0span600k"
)
DELTA = 0.29  # 5.8 N.m against the 20 N.m torque limit
LAMBDA = 0.5  # discount rate, s^-1
HORIZON_SECONDS = 2.0
PHYSICS_DT = 0.001


def task_config():
    _, env_kwargs, _, _, _ = load_ct_hyperparams_from_table(
        "ct_sac", ENV_ID, MODE, hyperparams_dir="benchmarks/hyperparams"
    )
    task_kwargs = dict(env_kwargs.get("task_kwargs", {}) or {})
    task_kwargs.update(uniform_start=False, paper_start=False, release_start=True)
    gains = Gains(
        k_v=float(task_kwargs.get("k_v", DEFAULT_LYAPUNOV_K_V)),
        k_d=float(task_kwargs.get("k_d", DEFAULT_LYAPUNOV_K_D)),
        k_p=float(task_kwargs.get("k_p", DEFAULT_LYAPUNOV_K_P)),
    )
    return task_kwargs, gains, float(task_kwargs.get("torque_limit",
                                                     DEFAULT_TORQUE_LIMIT))


def make_env(task_kwargs, dt):
    return DMCContinuousEnv(
        domain_name="acrobot",
        task_name="swingup-xk",
        seed=20000,
        raw_state_obs=True,
        time_sampling="uniform",
        dt=dt,
        physics_dt=min(PHYSICS_DT, dt),
        max_steps=int(round(20.0 / dt)),
        episode_duration=20.0,
        return_reward_increment=False,
        task_kwargs=task_kwargs,
    )


def collect_tube_states(task_kwargs, gains, torque_limit, n, seeds):
    """In-tube states, gathered once at the finest interval and shared by all."""
    env = make_env(task_kwargs, PHYSICS_DT)
    ctrl = XinKanedaController(
        AcrobotParams.from_physics(env._env.physics), gains, torque_limit=torque_limit
    )
    pool = []
    for episode_seed in seeds:
        obs, _ = env.reset(seed=episode_seed)
        obs = np.asarray(obs, np.float32)
        ctrl.reset()
        for _ in range(int(round(20.0 / PHYSICS_DT))):
            act = np.asarray(ctrl.actions(obs[None, :]), np.float32).reshape(-1)
            _, _, _, _, next_obs, _, term, trunc, info = env.step_dt(act)
            if float(info.get("acrobot_xk_homoclinic_capture", 0.0)) > 0.5:
                p = env._env.physics
                pool.append((p.data.qpos.copy(), p.data.qvel.copy()))
            if term or trunc:
                break
            obs = np.asarray(next_obs, np.float32)
    if not pool:
        raise RuntimeError("no in-tube states collected")
    idx = np.linspace(0, len(pool) - 1, min(n, len(pool))).astype(int)
    return [pool[i] for i in idx]


def rollout(env, ctrl, qpos, qvel, dt, steps, deviate, override=None):
    """Discounted return from a physical state.

    ``deviate`` adds the fixed torque error to the first control step only;
    ``override`` instead REPLACES the first action with a given normalized
    torque, which is how the action grid is swept.  Every later step follows the
    analytical law in both cases, so the return differs only through the state
    the first interval leaves behind -- exactly the comparison ``Q(s,a)`` makes.
    """
    env.reset(seed=20000)
    p = env._env.physics
    with p.reset_context():
        p.data.qpos[:] = qpos
        p.data.qvel[:] = qvel
    env._step_index = 0
    env.cur_t = 0.0
    obs = np.concatenate([p.data.qpos.copy(), p.data.qvel.copy()]).astype(np.float32)
    env._last_obs = obs
    env._last_obs_dmc = obs
    total = 0.0
    for k in range(steps):
        act = np.asarray(ctrl.actions(obs[None, :]), np.float32).reshape(-1)
        if k == 0:
            if override is not None:
                act = np.array([override], dtype=np.float32)
            elif deviate:
                act = np.clip(act + DELTA, -1.0, 1.0)
        _, _, _, reward, next_obs, _, term, trunc, _ = env.step_dt(act)
        total += float(reward) * dt * np.exp(-LAMBDA * k * dt)
        if term or trunc:
            break
        obs = np.asarray(next_obs, np.float32)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intervals", default="0.001,0.002,0.005,0.01,0.02,0.05")
    ap.add_argument("--n", type=int, default=185)
    ap.add_argument(
        "--action-grid",
        type=int,
        default=0,
        help="if > 1, also roll out this many evenly spaced actions per state "
        "to measure the TRUE action-range of Q^pi -- the quantity sec. 3 of "
        "the advantage-parameterization note tabulates",
    )
    ap.add_argument("--horizon", type=float, default=HORIZON_SECONDS)
    ap.add_argument("--episode-seeds", default="20000,20001,20002,20003")
    ap.add_argument("--out", default="results/one_step_advantage_vs_dt")
    args = ap.parse_args()

    task_kwargs, gains, torque_limit = task_config()
    seeds = [int(s) for s in args.episode_seeds.split(",")]
    states = collect_tube_states(task_kwargs, gains, torque_limit, args.n, seeds)
    print(f"in-tube states: {len(states)}  horizon {args.horizon}s  delta {DELTA}",
          flush=True)

    rows = []
    for dt in [float(x) for x in args.intervals.split(",") if x.strip()]:
        env = make_env(task_kwargs, dt)
        ctrl = XinKanedaController(
            AcrobotParams.from_physics(env._env.physics), gains,
            torque_limit=torque_limit,
        )
        steps = int(round(args.horizon / dt))
        A = np.array(
            [
                rollout(env, ctrl, qp, qv, dt, steps, True)
                - rollout(env, ctrl, qp, qv, dt, steps, False)
                for qp, qv in states
            ]
        )
        n = len(A)
        sem = A.std(ddof=1) / np.sqrt(n)
        tcrit = stats.t.ppf(0.975, n - 1)
        row = dict(
            dt=dt,
            n=n,
            mean=float(A.mean()),
            sd=float(A.std(ddof=1)),
            sem=float(sem),
            t=float(A.mean() / sem) if sem else float("nan"),
            ci_lo=float(A.mean() - tcrit * sem),
            ci_hi=float(A.mean() + tcrit * sem),
            rate=float(A.mean() / dt),
            positive=int((A > 0).sum()),
            cohen_d=float(A.mean() / A.std(ddof=1)),
        )
        if args.action_grid > 1:
            g = np.linspace(-1.0, 1.0, args.action_grid)
            spans = []
            for qp, qv in states:
                q = np.array(
                    [
                        rollout(env, ctrl, qp, qv, dt, steps, False, override=float(a))
                        for a in g
                    ]
                )
                spans.append(q.max() - q.min())
            spans = np.asarray(spans)
            row["range_q_true"] = float(spans.mean())
            row["range_q_true_sd"] = float(spans.std(ddof=1))
            row["range_qv_true"] = float(spans.mean() / dt)

        rows.append(row)
        print(
            f"  dt={dt*1000:6.1f}ms  mean={row['mean']:+.6f}  sd={row['sd']:.5f}  "
            f"|t|={abs(row['t']):5.2f}  95%CI=[{row['ci_lo']:+.5f},{row['ci_hi']:+.5f}]"
            f"  rate=mean/dt={row['rate']:+.4f}  pos={row['positive']}/{n}"
            + (
                f"  range_Q_true={row['range_q_true']:.5f}"
                f"  range_qv_true={row['range_qv_true']:.3f}"
                if "range_q_true" in row
                else ""
            ),
            flush=True,
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(args.out + ".json", "w"), indent=2)

    ref = rows[0]
    print("\nscaling relative to the finest interval:")
    for r in rows:
        print(
            f"  dt={r['dt']*1000:6.1f}ms  x{r['dt']/ref['dt']:6.1f} in dt  "
            f"x{r['mean']/ref['mean']:6.2f} in mean advantage  "
            f"rate ratio {r['rate']/ref['rate']:5.2f}"
        )
    print(f"\nwrote {args.out}.csv / .json")


if __name__ == "__main__":
    main()
