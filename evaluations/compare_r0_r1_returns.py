"""Compare final-checkpoint episode returns under the r0 and r1 reward
functions across the irregular1m sweep's 10 arms.

environment.acrobot_xk.BalanceXK.get_reward always computes r0, r1, r2, r3
(raw and transformed) simultaneously via xk_reward_terms -- only the
configured reward_kind picks which one is actually paid out as `reward`;
the rest ride along in `info` every step (see
environment.dmc.DMCContinuousEnv._acrobot_reward_info). That live diagnostic
value is NOT what training/scoring actually pays out on a cap-terminated
episode, though: DMCContinuousEnv._step_physics_unlocked's
cap_terminal_penalty block substitutes the terminal step's `reward` with
the configured reward_kind's own failure-rate lower bound, and that
substitution only applies to whichever reward_kind the env is actually
built with. So this runs the standard xk_eval env (uniform time, dt=0.001,
DEFAULT reward_base="lyapunov"/k_v/k_d/k_p -- one fixed r0/r1 definition
applied identically to every arm, not each arm's own training reward_base)
TWICE per checkpoint, once with reward_kind='r0' and once with 'r1': the
env's own returned `reward`, summed, is "r0/r1 with the termination reward"
(*_return_mean below); info's raw_r0/raw_r1 diagnostic, summed, is the same
metric without that substitution (*_notermination_mean) -- the number the
first version of this script reported.

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


def _xk_eval_task_kwargs(
    algo: str, mode: str, hyperparams_dir: str, reward_kind: str
) -> tuple[dict, dict, dict]:
    """xk_eval's own task, with the training arm's own termination limits
    carried over (matching evaluate_sb3_checkpoint / run_ct_rl.py's
    eval_mode resolution) and reward_kind swapped to the requested one --
    reward_base/k_v/k_d/k_p stay at xk_eval's defaults so r0/r1 mean the
    same thing for every arm.

    ``reward_kind`` isn't just which diagnostic to read after the fact: the
    env only applies its cap-termination reward substitution (see
    environment.dmc.DMCContinuousEnv._step_physics_unlocked's
    cap_terminal_penalty block) to whichever reward_kind is actually
    configured. A rollout with reward_kind='r0' gives info['acrobot_xk_r1']
    every step, but that value is never substituted on a cap failure the
    way the r0 the env actually pays out is -- getting r1 "with the
    termination reward" requires a separate rollout with reward_kind='r1'
    configured, not just reading r1 out of the r0 rollout's diagnostics.
    """
    loader = load_ct_hyperparams_from_table if algo in CT_ALGOS else load_sb3_hyperparams_from_table
    _, eval_meta, *_ = loader(algo, ACROBOT_XK_ENV_ID, "xk_eval", hyperparams_dir=hyperparams_dir)
    _, train_meta, *_ = loader(algo, ACROBOT_XK_ENV_ID, mode, hyperparams_dir=hyperparams_dir)
    task = dict(eval_meta.get("task_kwargs") or {})
    train_task = dict(train_meta.get("task_kwargs") or {})
    for key in ACROBOT_XK_TERMINATION_TASK_KEYS:
        if key in train_task:
            task[key] = train_task[key]
    task["reward_kind"] = reward_kind
    return task, eval_meta, train_meta


def _make_ct_env(algo: str, mode: str, hyperparams_dir: str, reward_kind: str):
    task, eval_meta, train_meta = _xk_eval_task_kwargs(
        algo, mode, hyperparams_dir, reward_kind
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
    return env, train_meta


def _run_ct(algo: str, mode: str, seed: int, run_id: str, hyperparams_dir: str) -> dict:
    _, _, model_kwargs, _, _ = load_ct_hyperparams_from_table(
        algo, ACROBOT_XK_ENV_ID, mode, hyperparams_dir=hyperparams_dir
    )
    # Build once against the r0 env just to resolve save_dir/checkpoint and
    # load the model; both reward_kind passes reuse this same model.
    probe_env, train_meta = _make_ct_env(algo, mode, hyperparams_dir, "r0")
    save_dir = build_save_path(
        "saved_models", algo, ACROBOT_XK_ENV_ID, mode, seed, train_meta, "", run_id=run_id
    )
    checkpoint = save_dir / "final_model.pth"
    if not checkpoint.is_file():
        probe_env.close()
        raise FileNotFoundError(f"No checkpoint at {checkpoint}")
    model = _load_model(probe_env, model_kwargs, checkpoint, "cpu")

    metrics = {}
    for reward_kind, env in (("r0", probe_env), ("r1", None)):
        if env is None:
            env, _ = _make_ct_env(algo, mode, hyperparams_dir, reward_kind)
        try:
            metrics[reward_kind] = _rollout_ct(model, env, ACROBOT_XK_EVAL_SEEDS)
        finally:
            env.close()
    return _combine(metrics), str(checkpoint)


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


def _make_sb3_env(algo: str, mode: str, seed: int, hyperparams_dir: str, reward_kind: str):
    task, eval_meta, train_meta = _xk_eval_task_kwargs(
        algo, mode, hyperparams_dir, reward_kind
    )
    env_meta = dict(eval_meta)
    env_meta["task_kwargs"] = task
    env = make_sb3_env(
        env_id=ACROBOT_XK_ENV_ID,
        monitor_root=Path("results") / "_r0_r1_monitor",
        seed=seed + 1000,
        env_meta=env_meta,
        dataset_path=EVAL_NPZ,
    )
    return env, train_meta


def _run_sb3(algo: str, mode: str, seed: int, run_id: str, hyperparams_dir: str) -> dict:
    # Just needs train_meta for the checkpoint path; the env itself is
    # rebuilt fresh per reward_kind pass below.
    probe_env, train_meta = _make_sb3_env(algo, mode, seed, hyperparams_dir, "r0")
    probe_env.close()

    save_dir = build_save_path(
        "saved_models", algo, ACROBOT_XK_ENV_ID, mode, seed, train_meta, "", run_id=run_id
    )
    checkpoint = save_dir / "final_model.zip"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No checkpoint at {checkpoint}")

    AlgoClass = {"sac": SAC, "td3": TD3}[algo]
    model = AlgoClass.load(str(checkpoint), env=None)

    metrics = {}
    for reward_kind in ("r0", "r1"):
        env, _ = _make_sb3_env(algo, mode, seed, hyperparams_dir, reward_kind)
        try:
            metrics[reward_kind] = _rollout_sb3(model, env, ACROBOT_XK_EVAL_SEEDS)
        finally:
            env.close()
    return _combine(metrics), str(checkpoint)


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


def _combine(metrics: dict) -> dict:
    """Merge the reward_kind='r0' and reward_kind='r1' rollout passes.

    ``*_notermination`` is the live per-step diagnostic sum (info's
    raw_r0/raw_r1, computed every step regardless of which reward_kind is
    configured, but never substituted on a cap-termination). ``*_return``
    is the env's own native reward sum *for that pass's configured
    reward_kind* -- the one number that actually reflects the
    cap-termination substitution (see _xk_eval_task_kwargs), i.e. "r0/r1
    with the reward termination applied."
    """
    r0, r1 = metrics["r0"], metrics["r1"]
    assert r0["n_episodes"] == r1["n_episodes"]
    return {
        "r0_notermination_mean": r0["r0_mean_return"],
        "r0_notermination_std": r0["r0_std_return"],
        "r0_return_mean": r0["native_mean_return"],
        "r1_notermination_mean": r1["r1_mean_return"],
        "r1_notermination_std": r1["r1_std_return"],
        "r1_return_mean": r1["native_mean_return"],
        "n_episodes": r0["n_episodes"],
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
