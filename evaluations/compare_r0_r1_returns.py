"""Compare final-checkpoint episode returns under the r0 and r1 reward
functions across the irregular1m sweep's 10 arms.

environment.acrobot_xk.BalanceXK.get_reward always computes r0, r1, r2, r3
(raw and transformed) simultaneously via xk_reward_terms -- only the
configured reward_kind picks which one is actually paid out as `reward`;
the rest ride along in `info` every step (see
environment.dmc.DMCContinuousEnv._acrobot_reward_info). So one rollout
under the standard xk_eval env (uniform time, dt=0.001, reward_kind=r0,
DEFAULT reward_base="lyapunov"/k_v/k_d/k_p -- i.e. one fixed r0/r1
definition applied identically to every arm, not each arm's own training
reward_base) is enough to score both: sum info["acrobot_xk_raw_r0"] and
info["acrobot_xk_raw_r1"] per episode instead of (only) the env's own `reward` output.

Runs the same fixed 32-episode protocol (seeds 20000-20031) as
common.sb3_callbacks.evaluate_sb3_policy_at_fixed_seeds /
evaluations.evaluation_helpers.evaluate_policy_per_episode's
exact_episode_seeds branch, so results here are on the same footing as
those. One-off analysis script (not part of the training/eval pipeline).

Usage:
    python -m evaluations.compare_r0_r1_returns --algo ct_sac \\
        --mode <training-mode> --seed 0 --run_id irregular1m_v1 \\
        --output results/irregular1m_r0_r1/ct_sac_baseline_seed0.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import SAC, TD3

from benchmarks.run_ct_rl import ACROBOT_XK_TERMINATION_TASK_KEYS
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


def _xk_eval_task_kwargs(algo: str, mode: str, hyperparams_dir: str) -> dict:
    """xk_eval's own reward_kind=r0 task, with the training arm's own
    termination limits carried over (matching evaluate_sb3_checkpoint /
    run_ct_rl.py's eval_mode resolution) -- reward_base/k_v/k_d/k_p are
    left at xk_eval's defaults so r1 means the same thing for every arm."""
    loader = load_ct_hyperparams_from_table if algo in CT_ALGOS else load_sb3_hyperparams_from_table
    _, eval_meta, *_ = loader(algo, ACROBOT_XK_ENV_ID, "xk_eval", hyperparams_dir=hyperparams_dir)
    _, train_meta, *_ = loader(algo, ACROBOT_XK_ENV_ID, mode, hyperparams_dir=hyperparams_dir)
    task = dict(eval_meta.get("task_kwargs") or {})
    train_task = dict(train_meta.get("task_kwargs") or {})
    for key in ACROBOT_XK_TERMINATION_TASK_KEYS:
        if key in train_task:
            task[key] = train_task[key]
    return task, eval_meta, train_meta


def _run_ct(algo: str, mode: str, seed: int, run_id: str, hyperparams_dir: str) -> dict:
    task, eval_meta, train_meta = _xk_eval_task_kwargs(algo, mode, hyperparams_dir)
    _, _, model_kwargs, _, _ = load_ct_hyperparams_from_table(
        algo, ACROBOT_XK_ENV_ID, mode, hyperparams_dir=hyperparams_dir
    )
    env = DMCContinuousEnv(
        domain_name="acrobot",
        task_name="swingup-xk",
        seed=ACROBOT_XK_EVAL_SEEDS[0],
        raw_state_obs=True,
        time_sampling=eval_meta.get("time_sampling", "uniform"),
        dt=eval_meta["dt"],
        physics_dt=eval_meta.get("physics_dt", eval_meta["dt"]),
        max_steps=eval_meta["max_steps"],
        episode_duration=eval_meta.get("episode_duration"),
        task_kwargs=task,
    )
    save_dir = build_save_path(
        "saved_models", algo, ACROBOT_XK_ENV_ID, mode, seed, train_meta, "", run_id=run_id
    )
    checkpoint = save_dir / "final_model.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No checkpoint at {checkpoint}")
    try:
        model = _load_model(env, model_kwargs, checkpoint, "cpu")
        return _rollout_ct(model, env, ACROBOT_XK_EVAL_SEEDS), str(checkpoint)
    finally:
        env.close()


def _rollout_ct(model, env, episode_seeds) -> dict:
    r0_returns, r1_returns, native_returns = [], [], []
    for seed in episode_seeds:
        obs, _ = env.reset(seed=seed)
        r0_sum = r1_sum = native_sum = 0.0
        while True:
            obs_t = th.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
            with th.no_grad():
                action_t, _ = model.act(obs_t, deterministic=True)
            action = action_t.squeeze(0).cpu().numpy()
            obs, _, _, reward, next_obs, _, terminated, truncated, info = env.step_dt(action)
            r0_sum += float(info["acrobot_xk_raw_r0"])
            r1_sum += float(info["acrobot_xk_raw_r1"])
            native_sum += float(reward)
            obs = next_obs
            if bool(terminated) or bool(truncated):
                break
        r0_returns.append(r0_sum)
        r1_returns.append(r1_sum)
        native_returns.append(native_sum)
    return {
        "r0_mean_return": float(np.mean(r0_returns)),
        "r0_std_return": float(np.std(r0_returns)),
        "r1_mean_return": float(np.mean(r1_returns)),
        "r1_std_return": float(np.std(r1_returns)),
        "native_mean_return": float(np.mean(native_returns)),
        "n_episodes": len(episode_seeds),
    }


def _run_sb3(algo: str, mode: str, seed: int, run_id: str, hyperparams_dir: str) -> dict:
    task, eval_meta, train_meta = _xk_eval_task_kwargs(algo, mode, hyperparams_dir)
    env_meta = dict(eval_meta)
    env_meta["task_kwargs"] = task

    save_dir = build_save_path(
        "saved_models", algo, ACROBOT_XK_ENV_ID, mode, seed, train_meta, "", run_id=run_id
    )
    checkpoint = save_dir / "final_model.zip"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No checkpoint at {checkpoint}")

    AlgoClass = {"sac": SAC, "td3": TD3}[algo]
    model = AlgoClass.load(str(checkpoint), env=None)
    env = make_sb3_env(
        env_id=ACROBOT_XK_ENV_ID,
        monitor_root=Path("results") / "_r0_r1_monitor",
        seed=seed + 1000,
        env_meta=env_meta,
        dataset_path=EVAL_NPZ,
    )
    try:
        return _rollout_sb3(model, env, ACROBOT_XK_EVAL_SEEDS), str(checkpoint)
    finally:
        env.close()


def _rollout_sb3(model, env, episode_seeds) -> dict:
    r0_returns, r1_returns, native_returns = [], [], []
    for seed in episode_seeds:
        obs, _ = env.reset(seed=seed)
        r0_sum = r1_sum = native_sum = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            r0_sum += float(info["acrobot_xk_raw_r0"])
            r1_sum += float(info["acrobot_xk_raw_r1"])
            native_sum += float(reward)
            if terminated or truncated:
                break
        r0_returns.append(r0_sum)
        r1_returns.append(r1_sum)
        native_returns.append(native_sum)
    return {
        "r0_mean_return": float(np.mean(r0_returns)),
        "r0_std_return": float(np.std(r0_returns)),
        "r1_mean_return": float(np.mean(r1_returns)),
        "r1_std_return": float(np.std(r1_returns)),
        "native_mean_return": float(np.mean(native_returns)),
        "n_episodes": len(episode_seeds),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, choices=CT_ALGOS + SB3_ALGOS)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--hyperparams_dir", default="benchmarks/hyperparams")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    runner = _run_ct if args.algo in CT_ALGOS else _run_sb3
    metrics, checkpoint = runner(
        args.algo, args.mode, args.seed, args.run_id, args.hyperparams_dir
    )
    summary = {
        "algo": args.algo,
        "mode": args.mode,
        "seed": args.seed,
        "checkpoint": checkpoint,
        **metrics,
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
