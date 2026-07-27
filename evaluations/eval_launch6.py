#!/usr/bin/env python
"""Unified held-out eval for the launch6_v1 batch (5 arms x 2 envs).

Covers CT-SAC and CT-TD3 (ActorQCriticModel, .pth), and discrete PPO/SAC
(SB3, .zip), on acrobot-swingup-v4.2 and cartpole-two_poles-curriculum. Each
final/best checkpoint is rolled N_EVAL held-out deterministic episodes under two
start distributions -- ``uniform`` (uniform_start=True) and ``hanging``
(uniform_start=False, the stock near-hanging swing-up) -- with the reset
curriculum forced OFF so the start is fixed.

Universal metrics per checkpoint x start: mean return, mean episode sim-seconds,
mean dt-weighted reward-rate (return/T, a length-normalised quality density).
For acrobot-swingup-v4.2 additionally: max tip height, height occupancy
(tip_z>3), hold occupancy (info['acrobot_hold']), frac tip>3, and the strict
sustained-capture success rate + mean max residence (distance<0.2, speed<0.2,
>=1 physical second) applied uniformly to every algorithm on that env.
Cartpole rows leave the acrobot-only columns as nan.

``LAUNCH_ENVS``, ``LAUNCH_CT_ARMS`` and ``LAUNCH_SB3_ARMS`` override the env and
arm lists ("algo:mode" pairs, comma separated) so a later batch reuses this
harness unchanged; unset, they reproduce launch6_v1.  On a task without a
``hold`` reward term (acrobot-swingup-v6) the hold-occupancy column reads 0 and
the height and strict-capture columns carry the comparison.

Sharding: set SHARD="i/N" to process only spec indices with index % N == i and
write to ``${ACRO_EVAL_OUT%.csv}_shard{i}.csv`` (merge externally). No sharding
=> single CSV at ACRO_EVAL_OUT.  MUJOCO_GL=disable (no rendering).
"""
import os
os.environ.setdefault("MUJOCO_GL", "disable")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import glob
import csv
import numpy as np
import torch as th

from benchmarks.run_ct_rl import make_ct_env
from common.utils import (
    load_ct_hyperparams_from_table,
    load_sb3_hyperparams_from_table,
)
from evaluations.sustained_capture import (
    SustainedCaptureSpec,
    SustainedCaptureTracker,
    capture_selection_rank,
)
from environment.tip_curriculum import PERFORMANCE_CURRICULUM_ENV_IDS
from models import ActorQCriticModel
from stable_baselines3 import SAC, PPO

N_EVAL = int(os.environ.get("N_EVAL", "20"))
SEED0 = 20000
OUT = os.environ.get("ACRO_EVAL_OUT", "results/launch6_eval.csv")
TAG = os.environ.get("LAUNCH_TAG", "launch6_v1")

def _arms(var, default):
    """Parse an "algo:mode,algo:mode" override, else keep the launch6 default."""
    raw = os.environ.get(var, "").strip()
    if not raw:
        return default
    return [tuple(p.split(":", 1)) for p in raw.split(",") if p.strip()]


# Overridable so a later batch (e.g. the acrobot-swingup-v6 quadratic-cost pair)
# reuses this harness without editing it; the defaults reproduce launch6_v1.
ENVS = [
    e for e in os.environ.get(
        "LAUNCH_ENVS", "acrobot-swingup-v4.2,cartpole-two_poles-curriculum"
    ).split(",") if e.strip()
]
CT_ARMS = _arms(
    "LAUNCH_CT_ARMS",
    [("ct_sac", "final_mf"), ("ct_sac", "final_oracle_rollout"),
     ("ct_td3", "final_mf")],
)
SB3_ARMS = _arms("LAUNCH_SB3_ARMS", [("ppo", "final_mf"), ("sac", "final_mf")])


def _eval_starts(env_id):
    """Per-env eval start distributions as task_kwargs overrides.

    Mastery-gated tip tasks (v4.3/v6.1/cartpole-two_poles-v2) take only a
    ``curriculum`` flag; ``curriculum=False`` is the fixed hanging-at-rest task
    used for checkpoint selection, and there is no ``uniform_start`` for them.
    Older uniform-start / fraction-curriculum tasks keep both starts.
    """
    if env_id in PERFORMANCE_CURRICULUM_ENV_IDS:
        # ``curriculum=True`` pins the reset ladder at level 0 -- the near-upright
        # braking start ("balance from near the top", the first mastery stage) --
        # since nothing advances the stage during eval. Gated so the committed
        # default (fixed hanging start for checkpoint selection) is unchanged.
        if os.environ.get("EVAL_NEAR_TOP", "") == "1":
            return [("near_top", {"curriculum": True})]
        return [("hanging", {"curriculum": False})]
    return [("uniform", {"uniform_start": True, "curriculum": False}),
            ("hanging", {"uniform_start": False, "curriculum": False})]


def env_kwargs_for(framework, algo, env_id, mode):
    if framework == "ct":
        _, ek, mk, _, _ = load_ct_hyperparams_from_table(
            algo=algo, env_id=env_id, mode=mode, hyperparams_dir="benchmarks/hyperparams"
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


def _with_start(env_kwargs, overrides):
    """Copy env_kwargs applying a start's task_kwargs overrides."""
    ek = dict(env_kwargs)
    tk = dict(ek.get("task_kwargs", {}))
    tk.update(overrides)
    ek["task_kwargs"] = tk
    return ek


def rollout(env, pol, seed, capture_spec):
    obs, reset_info = env.reset(seed=seed)
    obs = np.asarray(obs, dtype=np.float32)
    ret, rw_dt, T, steps, done = 0.0, 0.0, 0.0, 0, False
    maxtip, occ_h, occ_hold = -1e9, 0.0, 0.0
    tracker = (
        SustainedCaptureTracker(1, capture_spec, [reset_info])
        if capture_spec is not None else None
    )
    cap = None
    while not done:
        a = act(pol, obs)
        _, t, _, r, nobs, nt, term, trunc, info = env.step_dt(a)
        done = bool(term or trunc)
        if tracker is not None:
            cap = tracker.update_slot(0, info, done=done)
        dt = float(nt) - float(t)
        T += dt; steps += 1; ret += float(r); rw_dt += float(r) * dt
        tip = float(info.get("acrobot_tip_height", -1e9))
        maxtip = max(maxtip, tip)
        occ_h += dt * (1.0 if tip > 3.0 else 0.0)
        occ_hold += dt * float(info.get("acrobot_hold", 0.0))
        obs = np.asarray(nobs, dtype=np.float32)
    return dict(ret=ret, rw_dt=rw_dt, T=T, steps=steps, maxtip=maxtip,
                height_occ=occ_h / max(T, 1e-9),
                hold_occ=occ_hold / max(T, 1e-9), cap=cap)


def discover():
    specs = []
    for algo, mode in CT_ARMS:
        for env_id in ENVS:
            for d in sorted(glob.glob(
                    f"saved_models/{algo}/{env_id}/{mode}/seed_*/*{TAG}")):
                seed = int(d.split("/seed_")[1].split("/")[0])
                for kind, p in [("final", f"{d}/final_model.pth"),
                                ("best", f"{d}/best_model/best_model.pth")]:
                    if os.path.isfile(p):
                        specs.append(dict(framework="ct", algo=algo, env_id=env_id,
                                          mode=mode, seed=seed, kind=kind, path=p))
    for algo, mode in SB3_ARMS:
        for env_id in ENVS:
            for d in sorted(glob.glob(
                    f"saved_models/{algo}/{env_id}/{mode}/seed_*/*{TAG}*")):
                if not os.path.isdir(d):
                    continue
                seed = int(d.split("/seed_")[1].split("/")[0])
                for kind, p in [("final", f"{d}/final_model.zip"),
                                ("best", f"{d}/best_model/best_model.zip")]:
                    if os.path.isfile(p):
                        specs.append(dict(framework="sb3", algo=algo, env_id=env_id,
                                          mode=mode, seed=seed, kind=kind, path=p))
    return specs


def main():
    specs = discover()
    shard = os.environ.get("SHARD", "")
    out = OUT
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        specs = [s for k, s in enumerate(specs) if k % n == i]
        out = f"{OUT[:-4] if OUT.endswith('.csv') else OUT}_shard{i}.csv"
    print(f"[{shard or 'all'}] {len(specs)} specs", flush=True)
    rows = []
    for s in specs:
        ek, mk = env_kwargs_for(s["framework"], s["algo"], s["env_id"], s["mode"])
        is_acro = s["env_id"].startswith("acrobot")
        capture_spec = SustainedCaptureSpec() if is_acro else None
        for start_label, overrides in _eval_starts(s["env_id"]):
            ek_s = _with_start(ek, overrides)
            rets, Ts, rates, tavg, rps, tips, hocc, holdocc = [], [], [], [], [], [], [], []
            csucc, cdur = [], []
            for j in range(N_EVAL):
                env = make_ct_env(env_id=s["env_id"], seed=SEED0 + j, env_kwargs=ek_s)
                try:
                    pol = load_policy(s["framework"], s["algo"], s["path"], env, mk)
                    m = rollout(env, pol, SEED0 + j, capture_spec)
                finally:
                    env.close()
                rets.append(m["ret"]); Ts.append(m["T"])
                rates.append(m["ret"] / max(m["T"], 1e-9))
                # dt-weighted time-average reward (fair, timing-invariant) + per-step
                tavg.append(m["rw_dt"] / max(m["T"], 1e-9))
                rps.append(m["ret"] / max(m["steps"], 1))
                tips.append(m["maxtip"]); hocc.append(m["height_occ"])
                holdocc.append(m["hold_occ"])
                if capture_spec is not None and m["cap"] is not None:
                    csucc.append(m["cap"].success)
                    cdur.append(m["cap"].max_duration_seconds)
            if capture_spec is not None and csucc:
                cap_rate, cap_meandur = capture_selection_rank(csucc, cdur)
            else:
                cap_rate = cap_meandur = float("nan")
            row = dict(
                framework=s["framework"], algo=s["algo"], env_id=s["env_id"],
                mode=s["mode"], seed=s["seed"], ckpt=s["kind"], start=start_label,
                n_eval=N_EVAL,
                mean_return=round(float(np.mean(rets)), 2),
                mean_T=round(float(np.mean(Ts)), 2),
                mean_timeavg_reward=round(float(np.mean(tavg)), 4),
                mean_reward_per_step=round(float(np.mean(rps)), 4),
                mean_reward_rate=round(float(np.mean(rates)), 4),
                max_tip_height=round(float(np.max(tips)), 3) if is_acro else "",
                mean_height_occ=round(float(np.mean(hocc)), 4) if is_acro else "",
                mean_hold_occ=round(float(np.mean(holdocc)), 4) if is_acro else "",
                frac_tip_gt3=round(float(np.mean([t > 3.0 for t in tips])), 3) if is_acro else "",
                strict_capture_success_rate=round(float(cap_rate), 6) if is_acro else "",
                strict_capture_mean_max_duration=round(float(cap_meandur), 6) if is_acro else "",
            )
            rows.append(row)
            print(f"  {s['algo']}/{s['env_id'].split('-')[0] if is_acro else 'cartpole'}"
                  f"/s{s['seed']}/{s['kind']}/{start_label}: ret={row['mean_return']} "
                  f"timeavg_r={row['mean_timeavg_reward']} strict={row['strict_capture_success_rate']}",
                  flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
