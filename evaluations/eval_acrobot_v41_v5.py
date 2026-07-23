#!/usr/bin/env python
"""Unified acrobot policy eval for the v4.1 / v5 batch (CT-SAC + SB3).

Discovers the run checkpoints under saved_models, loads each policy (CT
ActorQCriticModel from .pth, or SB3 SAC/PPO from .zip), rolls N_EVAL held-out
deterministic episodes on the cell's own training timing, and records per
checkpoint the mean return, max tip height, height occupancy (dt-weighted time
with tip_z>3), and hold occupancy (dt-weighted info['acrobot_hold']).

Each checkpoint is evaluated under both start distributions, side by side:
``uniform`` (the training reset: uniform random joint angles) and ``hanging``
(the canonical swing-up-from-down task).  The ``start`` column distinguishes
them, so the true from-hanging capability is reported alongside the
capture-from-anywhere number the uniform training reset produces.

No rendering -> MUJOCO_GL=disable. Writes results/acrobot_v41_v5_eval.csv.
"""
import os
os.environ.setdefault("MUJOCO_GL", "disable")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import glob
import csv
import numpy as np
import torch as th

from benchmarks.run_ct_rl import make_ct_env
from common.utils import (
    load_ct_hyperparams_from_table,
    load_sb3_hyperparams_from_table,
)
from models import ActorQCriticModel
from stable_baselines3 import SAC, PPO

N_EVAL = int(os.environ.get("N_EVAL", "20"))
SEED0 = 20000
OUT = os.environ.get("ACRO_EVAL_OUT", "results/acrobot_v41_v5_eval.csv")

# Start distributions to evaluate, run alongside each other per checkpoint.
# "uniform" is the training reset; "hanging" is the canonical swing-up task.
_START_UNIFORM = {"uniform": True, "hanging": False}
STARTS = [
    (label, _START_UNIFORM[label])
    for label in os.environ.get("ACRO_EVAL_STARTS", "uniform,hanging").split(",")
    if label.strip()
]


def env_kwargs_for(framework, algo, env_id, mode):
    if framework == "ct":
        _, ek, mk, _, _ = load_ct_hyperparams_from_table(
            algo="ct_sac", env_id=env_id, mode=mode, hyperparams_dir="benchmarks/hyperparams"
        )
    else:
        _, ek, _, _, _ = load_sb3_hyperparams_from_table(
            algo=algo, env_id=env_id, mode=mode, hyperparams_dir="benchmarks/hyperparams"
        )
        mk = None
    ek = dict(ek)
    for k in ("n_envs", "eval_n_envs", "id"):
        ek.pop(k, None)
    return ek, mk


def load_policy(framework, algo, path, env, mk):
    if framework == "ct":
        m = ActorQCriticModel(
            observation_space=env.observation_space,
            action_space=env.action_space, **mk,
        )
        m.load_state(path)
        return ("ct", m)
    cls = SAC if algo == "sac" else PPO
    return ("sb3", cls.load(path, device="cpu"))


def act(pol, obs):
    kind, m = pol
    if kind == "ct":
        ot = th.as_tensor(obs, dtype=th.float32).unsqueeze(0)
        with th.no_grad():
            a, _ = m.act(ot, deterministic=True)
        return a.detach().cpu().numpy()[0]
    a, _ = m.predict(obs, deterministic=True)
    return a


def rollout(env, pol, seed):
    obs, _ = env.reset(seed=seed)
    obs = np.asarray(obs, dtype=np.float32)
    ret, maxtip, occ_h, occ_hold, T, done = 0.0, -1e9, 0.0, 0.0, 0.0, False
    while not done:
        a = act(pol, obs)
        _, t, _, r, nobs, nt, term, trunc, info = env.step_dt(a)
        dt = float(nt) - float(t)
        T += dt
        tip = float(info.get("acrobot_tip_height", -1e9))
        maxtip = max(maxtip, tip)
        occ_h += dt * (1.0 if tip > 3.0 else 0.0)
        occ_hold += dt * float(info.get("acrobot_hold", 0.0))
        ret += float(r)
        obs = np.asarray(nobs, dtype=np.float32)
        done = bool(term or trunc)
    return ret, maxtip, occ_h / max(T, 1e-9), occ_hold / max(T, 1e-9)


def discover():
    specs = []  # dict(framework, algo, env_id, mode, seed, kind, path)
    # CT-SAC runs: (env_id, mode, run_tag)
    ct = [
        ("acrobot-swingup-v4.1", "fork_v41", "acrov41_fork_v1"),
        ("acrobot-swingup-v4.1", "final_mf", "acrov41_mf_v1"),
        ("acrobot-swingup-v5", "mf_hz_g0999", "acrov5_rs_v1"),
        ("acrobot-swingup-v5", "mf_hz_g09995", "acrov5_rs_v1"),
    ]
    for env_id, mode, tag in ct:
        for d in sorted(glob.glob(f"saved_models/ct_sac/{env_id}/{mode}/seed_*/*{tag}")):
            seed = int(d.split("/seed_")[1].split("/")[0])
            for kind, p in [("final", f"{d}/final_model.pth"),
                            ("best", f"{d}/best_model/best_model.pth")]:
                if os.path.isfile(p):
                    specs.append(dict(framework="ct", algo="ct_sac", env_id=env_id,
                                      mode=mode, seed=seed, kind=kind, path=p))
    # SB3 runs: both v4.1 (any dir) and v5 (desc 'rs' only, to skip stale run)
    for algo in ("sac", "ppo"):
        for env_id, dirglob in [("acrobot-swingup-v4.1", "*"),
                                ("acrobot-swingup-v5", "*rs*")]:
            for d in sorted(glob.glob(
                    f"saved_models/{algo}/{env_id}/final_mf/seed_*/{dirglob}")):
                if not os.path.isdir(d):
                    continue
                seed = int(d.split("/seed_")[1].split("/")[0])
                for kind, p in [("final", f"{d}/final_model.zip"),
                                ("best", f"{d}/best_model/best_model.zip")]:
                    if os.path.isfile(p):
                        specs.append(dict(framework="sb3", algo=algo, env_id=env_id,
                                          mode="final_mf", seed=seed, kind=kind, path=p))
    return specs


def _with_start(env_kwargs, uniform_start):
    """Copy env_kwargs with the acrobot start distribution overridden."""
    ek = dict(env_kwargs)
    task_kwargs = dict(ek.get("task_kwargs", {}))
    task_kwargs["uniform_start"] = uniform_start
    ek["task_kwargs"] = task_kwargs
    return ek


def main():
    specs = discover()
    print(f"discovered {len(specs)} checkpoints x {len(STARTS)} starts "
          f"({[s for s, _ in STARTS]})", flush=True)
    rows = []
    total = len(specs) * len(STARTS)
    n = 0
    for s in specs:
        ek, mk = env_kwargs_for(s["framework"], s["algo"], s["env_id"], s["mode"])
        for start_label, uniform_start in STARTS:
            ek_start = _with_start(ek, uniform_start)
            rets, tips, hocc, holdocc = [], [], [], []
            for j in range(N_EVAL):
                env = make_ct_env(
                    env_id=s["env_id"], seed=SEED0 + j, env_kwargs=ek_start
                )
                pol = load_policy(s["framework"], s["algo"], s["path"], env, mk)
                r, mt, ho, hd = rollout(env, pol, SEED0 + j)
                env.close()
                rets.append(r); tips.append(mt); hocc.append(ho); holdocc.append(hd)
            row = dict(
                framework=s["framework"], algo=s["algo"], env_id=s["env_id"],
                mode=s["mode"], seed=s["seed"], ckpt=s["kind"],
                start=start_label, n_eval=N_EVAL,
                mean_return=round(float(np.mean(rets)), 2),
                max_tip_height=round(float(np.max(tips)), 3),
                mean_height_occ=round(float(np.mean(hocc)), 4),
                mean_hold_occ=round(float(np.mean(holdocc)), 4),
                frac_tip_gt3=round(float(np.mean([t > 3.0 for t in tips])), 3),
            )
            rows.append(row)
            n += 1
            print(f"[{n}/{total}] {s['algo']}/{s['env_id'].split('-')[-1]}/"
                  f"{s['mode']}/s{s['seed']}/{s['kind']}/{start_label}: "
                  f"ret={row['mean_return']} tip={row['max_tip_height']} "
                  f"hocc={row['mean_height_occ']} hold={row['mean_hold_occ']}",
                  flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
