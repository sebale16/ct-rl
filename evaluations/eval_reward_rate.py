#!/usr/bin/env python
"""Termination-aware reward RATE over the fixed Acrobot-XK eval protocol.

``compare_r0_r1_returns`` sums per-step rewards, which is not comparable
across arms that cap at different times: an episode killed by a state cap at
2 s accumulates a smaller sum simply for being short, and the 18 s it never
executed are not represented at all.  A rate fixes both.

For each of the 32 fixed evaluation episodes (seeds 20000-20031) this records

    executed  = sum_i r_i * h_i           over the steps actually taken
    t_end     = sum_i h_i                 realized episode duration
    R         = info["absorbing_failure_remaining_seconds"]   (0 if no cap)

and reports two rates:

    rate_executed = executed / t_end
        the average reward rate while the episode was alive.  Flatters a
        policy that caps early out of a good state.

    rate_horizon  = (executed + r_F * R) / (t_end + R)
        the average rate over the FULL nominal 20 s horizon, with the
        unexecuted remainder frozen at the reward's own lower envelope r_F --
        the same quantity CT-SAC's absorbing target integrates as G_F, here
        undiscounted.  This is the termination-aware number: a cap is charged
        for the horizon it threw away, so capped and full episodes are
        commensurable.

``r_F`` and ``R`` come from the env's own cap block
(``DMCContinuousEnv._step_physics_unlocked``), so the envelope matches the one
training scored against.  Note the terminal step's reward is ALREADY r_F, and
``R`` covers only the unexecuted remainder, so there is no double count.

The eval env is built exactly as ``run_ct_rl`` builds it -- the fixed xk_eval
row with the training arm's dt/physics_dt/max_steps/episode_duration and cap
limits carried across (``_align_acrobot_xk_eval_termination_limits``) -- so
these numbers sit on the same protocol as the reported capture rates.  This
differs from ``compare_r0_r1_returns``, which evaluates at the xk_eval row's
own dt=1 ms regardless of the arm's control rate.

    python -m evaluations.eval_reward_rate --algo ct_sac --mode <mode> \
        --seed 0 --run_id irregular1m_v1 --output results/rate/arm_seed0.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch as th

from pathlib import Path as _Path

from stable_baselines3 import SAC, TD3

from benchmarks.run_ct_rl import _align_acrobot_xk_eval_termination_limits
from benchmarks.run_discrete_rl import make_env as make_sb3_env
from common.demonstration import ACROBOT_XK_ENV_ID, ACROBOT_XK_EVAL_SEEDS
from common.utils import (
    build_save_path,
    load_ct_hyperparams_from_table,
    load_sb3_hyperparams_from_table,
)
from data.trading.config import EVAL_NPZ
from environment.dmc import DMCContinuousEnv
from evaluations.evaluate_swingup_final import _load_model

CT_ALGOS = ("ct_sac", "ct_td3")
SB3_ALGOS = ("sac", "td3")


#: task keys that define WHICH reward is paid out.  ``--train-reward`` carries
#: these from the training row so the rate is measured in the reward the arm
#: was actually optimising, instead of xk_eval's fixed r0.
TRAIN_REWARD_KEYS = ("reward_kind", "eta", "reward_transform", "reward_base",
                     "lyapunov_rate_source", "k_v", "k_d", "k_p", "discount_rate")


def _aligned_meta(algo: str, mode: str, hyperparams_dir: str,
                  train_reward: bool = False):
    """xk_eval's env block aligned to the arm's own control rate and caps.

    With ``train_reward`` the reward-defining task keys are carried across too,
    so the episode is scored by the arm's own training reward on the fixed
    evaluation start distribution.  The cap envelope r_F follows automatically:
    the env derives it from whichever reward_kind is configured."""
    loader = (load_ct_hyperparams_from_table if algo in CT_ALGOS
              else load_sb3_hyperparams_from_table)
    _, train_env, model_kwargs, _, _ = loader(
        algo, ACROBOT_XK_ENV_ID, mode, hyperparams_dir=hyperparams_dir
    )
    _, eval_env, _, _, _ = loader(
        algo, ACROBOT_XK_ENV_ID, "xk_eval", hyperparams_dir=hyperparams_dir
    )
    aligned = _align_acrobot_xk_eval_termination_limits(
        ACROBOT_XK_ENV_ID, train_env, eval_env
    )
    if train_reward:
        task = dict(aligned.get("task_kwargs") or {})
        train_task = dict(train_env.get("task_kwargs") or {})
        for key in TRAIN_REWARD_KEYS:
            if key in train_task:
                task[key] = train_task[key]
        aligned = dict(aligned)
        aligned["task_kwargs"] = task
    return aligned, train_env, model_kwargs


def build_sb3_eval_env(algo: str, mode: str, seed: int, hyperparams_dir: str,
                       train_reward: bool = False):
    """Same aligned protocol, built through run_discrete_rl's SB3 env factory."""
    aligned, train_env, _ = _aligned_meta(algo, mode, hyperparams_dir, train_reward)
    env = make_sb3_env(
        env_id=ACROBOT_XK_ENV_ID,
        monitor_root=_Path("results") / "_rate_monitor",
        seed=seed + 1000,
        env_meta=aligned,
        dataset_path=EVAL_NPZ,
    )
    return env, train_env


def rollout_sb3(model, env, episode_seeds):
    """SB3 goes through the gym 5-tuple, so h comes from info['dt_used']."""
    episodes = []
    for ep_seed in episode_seeds:
        obs, _ = env.reset(seed=int(ep_seed))
        executed = t_end = remaining = rate_F = 0.0
        capped = False
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            h = float(info.get("dt_used", 0.0))
            executed += float(reward) * h
            t_end += h
            if float(info.get("absorbing_failure", 0.0)) > 0.5:
                capped = True
                rate_F = float(info["absorbing_failure_reward_rate"])
                remaining = float(info["absorbing_failure_remaining_seconds"])
            if bool(terminated) or bool(truncated):
                break
        horizon = t_end + remaining
        episodes.append({
            "seed": int(ep_seed), "executed_integral": executed, "t_end": t_end,
            "capped": capped, "failure_rate": rate_F, "remaining": remaining,
            "rate_executed": executed / t_end if t_end > 0 else float("nan"),
            "rate_horizon": (executed + rate_F * remaining) / horizon if horizon > 0 else float("nan"),
        })
    return episodes


def build_eval_env(algo: str, mode: str, hyperparams_dir: str,
                   train_reward: bool = False):
    """The arm's reported eval env: xk_eval row, aligned to its control rate."""
    aligned, train_env, model_kwargs = _aligned_meta(
        algo, mode, hyperparams_dir, train_reward
    )
    env = DMCContinuousEnv(
        domain_name="acrobot",
        task_name="swingup-xk",
        seed=ACROBOT_XK_EVAL_SEEDS[0],
        raw_state_obs=True,
        time_sampling=aligned.get("time_sampling", "uniform"),
        dt=aligned["dt"],
        physics_dt=aligned.get("physics_dt", aligned["dt"]),
        max_steps=aligned["max_steps"],
        episode_duration=aligned.get("episode_duration"),
        task_kwargs=aligned.get("task_kwargs") or {},
    )
    return env, train_env, model_kwargs


def rollout(model, env, episode_seeds):
    """Per-episode executed reward-time integral, duration and cap remainder."""
    episodes = []
    for ep_seed in episode_seeds:
        obs, _ = env.reset(seed=int(ep_seed))
        executed = 0.0
        t_end = 0.0
        remaining = 0.0
        rate_F = 0.0
        capped = False
        while True:
            obs_t = th.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
            with th.no_grad():
                action_t, _ = model.act(obs_t, deterministic=True)
            action = action_t.squeeze(0).cpu().numpy()
            _, t, _, reward, next_obs, next_t, terminated, truncated, info = env.step_dt(action)
            h = float(next_t) - float(t)
            executed += float(reward) * h
            t_end += h
            obs = next_obs
            if float(info.get("absorbing_failure", 0.0)) > 0.5:
                capped = True
                rate_F = float(info["absorbing_failure_reward_rate"])
                remaining = float(info["absorbing_failure_remaining_seconds"])
            if bool(terminated) or bool(truncated):
                break
        horizon = t_end + remaining
        episodes.append({
            "seed": int(ep_seed),
            "executed_integral": executed,
            "t_end": t_end,
            "capped": capped,
            "failure_rate": rate_F,
            "remaining": remaining,
            "rate_executed": executed / t_end if t_end > 0 else float("nan"),
            "rate_horizon": (executed + rate_F * remaining) / horizon if horizon > 0 else float("nan"),
        })
    return episodes


def summarize(episodes):
    ex = np.array([e["rate_executed"] for e in episodes], dtype=float)
    hz = np.array([e["rate_horizon"] for e in episodes], dtype=float)
    dur = np.array([e["t_end"] for e in episodes], dtype=float)
    cap = np.array([e["capped"] for e in episodes], dtype=bool)
    return {
        "n_episodes": len(episodes),
        "rate_executed_mean": float(ex.mean()), "rate_executed_std": float(ex.std()),
        "rate_horizon_mean": float(hz.mean()), "rate_horizon_std": float(hz.std()),
        "mean_duration_s": float(dur.mean()),
        "cap_fraction": float(cap.mean()),
        "mean_remaining_s": float(np.mean([e["remaining"] for e in episodes])),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", default="ct_sac", choices=CT_ALGOS + SB3_ALGOS)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--run_id", default="irregular1m_v1")
    ap.add_argument("--checkpoint", default="best_model",
                    choices=["best_model", "final_model"])
    ap.add_argument("--hyperparams_dir", default="benchmarks/hyperparams")
    ap.add_argument("--n-episodes", type=int, default=len(ACROBOT_XK_EVAL_SEEDS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int,
                    help="evaluate the numbered checkpoint at this step count "
                         "instead of best/final")
    ap.add_argument("--train-reward", action="store_true",
                    help="score with the arm's own training reward rather than "
                         "xk_eval's fixed r0")
    ap.add_argument("--output")
    args = ap.parse_args(argv)

    if args.algo in SB3_ALGOS:
        env, train_env = build_sb3_eval_env(args.algo, args.mode, args.seed,
                                            args.hyperparams_dir, args.train_reward)
        save_dir = build_save_path("saved_models", args.algo, ACROBOT_XK_ENV_ID,
                                   args.mode, args.seed, train_env, "", run_id=args.run_id)
        ckpt = _pick_ckpt(save_dir, args, "zip")
        if not ckpt.is_file():
            env.close()
            raise FileNotFoundError(f"No checkpoint at {ckpt}")
        model = {"sac": SAC, "td3": TD3}[args.algo].load(str(ckpt), env=None)
        try:
            eps = rollout_sb3(model, env, ACROBOT_XK_EVAL_SEEDS[: args.n_episodes])
        finally:
            env.close()
    else:
        env, train_env, model_kwargs = build_eval_env(
            args.algo, args.mode, args.hyperparams_dir, args.train_reward)
        save_dir = build_save_path("saved_models", args.algo, ACROBOT_XK_ENV_ID,
                                   args.mode, args.seed, train_env, "", run_id=args.run_id)
        ckpt = _pick_ckpt(save_dir, args, "pth")
        if not ckpt.is_file():
            env.close()
            raise FileNotFoundError(f"No checkpoint at {ckpt}")
        model = _load_model(env, model_kwargs, ckpt, args.device)
        try:
            eps = rollout(model, env, ACROBOT_XK_EVAL_SEEDS[: args.n_episodes])
        finally:
            env.close()

    out = {"algo": args.algo, "mode": args.mode, "seed": args.seed,
           "run_id": args.run_id, "checkpoint": str(ckpt),
           "eval_dt": aligned_dt(train_env), "steps": args.steps,
           "train_reward": bool(args.train_reward),
           **summarize(eps), "episodes": eps}
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "episodes"}, indent=2))
    return 0


def _pick_ckpt(save_dir, args, ext):
    """best/final, or the numbered checkpoint nearest --steps."""
    if args.steps is None:
        return (save_dir / "best_model" / f"best_model.{ext}"
                if args.checkpoint == "best_model"
                else save_dir / f"final_model.{ext}")
    hits = sorted(save_dir.glob(f"*_{args.steps}_steps.{ext}"))
    if not hits:
        raise FileNotFoundError(f"No {args.steps}-step checkpoint in {save_dir}")
    return hits[0]


def aligned_dt(train_env):
    return float(train_env.get("dt", float("nan")))


if __name__ == "__main__":
    raise SystemExit(main())
