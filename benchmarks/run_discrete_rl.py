# benchmarks/run_discrete_benchmarks.py

from __future__ import annotations

from datetime import datetime
from functools import partial
import argparse
from pathlib import Path
import os
from itertools import count


import gymnasium as gym
from environment.dmc import DMCContinuousEnv
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
    LogEveryNTimesteps,
)

try:
    from sb3_contrib import TRPO
except ImportError:
    TRPO = None

from common.utils import build_save_path, load_sb3_hyperparams_from_table

_MONITOR_COUNTER = count()


def make_env(env_id, monitor_root, seed, env_meta=None):
    """Build a single environment instance from DM Control suite."""
    domain_name, task_name = env_id.split("-", 1)

    # Map env_* columns (already parsed into env_meta) to DMCContinuousEnv kwargs
    dmc_kwargs = {}
    for k in [
        "time_sampling",  # "uniform" / "irregular"
        "dt",
        "physics_dt",
        "min_dt",
        "max_dt",
        "max_steps",  # step limit
        "episode_duration",  # T
        "time_sampling_kwargs",
    ]:
        if k in env_meta and env_meta[k] is not None:
            dmc_kwargs[k] = env_meta[k]
    if "dt" in dmc_kwargs and "physics_dt" not in dmc_kwargs:
        dmc_kwargs["physics_dt"] = dmc_kwargs["dt"]

    env = DMCContinuousEnv(
        domain_name=domain_name,
        task_name=task_name,
        seed=seed,
        **dmc_kwargs,
    )

    # SB3 Monitor wrapper (for ep_reward_mean, ep_len_mean in TensorBoard)
    monitor_idx = next(_MONITOR_COUNTER)
    monitor_path = monitor_root / f"monitor_{os.getpid()}_{monitor_idx}.csv"
    env = Monitor(env, str(monitor_path))
    return env


def run_sb3_benchmark(
    algo: str,
    env_id: str,
    mode: str,
    eval_mode: str | None,
    seed: int,
    hyperparams_dir: str,
    log_root_dir: str,
    save_root_dir: str,
    total_timesteps_override: int | None,
    desc: str,
    increment_modeling: bool,
    n_eval_episodes: int = 10,
):
    """
    Runs a single Stable-Baselines3 benchmark experiment.
    """
    print(
        f"\n{'='*50}\nRunning SB3 {algo.upper()} on {env_id} (mode: {mode}, eval_mode: {eval_mode or mode}, seed: {seed})\n{'='*50}"
    )

    total_timesteps, env_meta, policy_kwargs, algo_kwargs, log_kwargs = (
        load_sb3_hyperparams_from_table(
            algo=algo, env_id=env_id, mode=mode, hyperparams_dir=hyperparams_dir
        )
    )

    if eval_mode:
        _, eval_env_meta, _, _, _ = load_sb3_hyperparams_from_table(
            algo=algo, env_id=env_id, mode=eval_mode, hyperparams_dir=hyperparams_dir
        )
    else:
        # If eval_mode is not specified, use the same env settings as training
        eval_env_meta = env_meta.copy()

    if total_timesteps_override is not None:
        total_timesteps = total_timesteps_override

    # If increment modeling is enabled, adjust env
    if increment_modeling:
        env_meta["return_reward_increment"] = True

    # Build logs and saved_models save paths
    log_dir = build_save_path(log_root_dir, algo, env_id, mode, seed, env_meta, desc)
    save_dir = build_save_path(save_root_dir, algo, env_id, mode, seed, env_meta, desc)

    # Create train environment through make_vec_env helper
    n_envs = int(env_meta.get("n_envs", 4))
    train_env = make_vec_env(
        make_env,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=dict(
            env_id=env_id, monitor_root=log_dir / "train", seed=seed, env_meta=env_meta
        ),
    )

    # Create evaluation environment
    eval_n_envs = int(eval_env_meta.get("n_envs", 1))
    eval_env = make_vec_env(
        make_env,
        n_envs=eval_n_envs,
        seed=seed + 1000,
        env_kwargs=dict(
            env_id=env_id,
            monitor_root=log_dir / "eval",
            seed=seed + 1000,
            env_meta=eval_env_meta,
        ),
    )

    # If increment modeling, adjust algo parameters gamma
    dt = env_meta.get("dt")
    if dt is not None and "gamma" in algo_kwargs:
        original_gamma = algo_kwargs["gamma"]
        dt_default = train_env.get_attr("dt_default", indices=0)[0]
        algo_kwargs["gamma"] = original_gamma ** (dt / dt_default)

    # Setup algorithms
    if algo == "sac":
        DefaultAlgo = partial(SAC, policy_kwargs=policy_kwargs)
    elif algo == "ppo":
        DefaultAlgo = partial(PPO, policy_kwargs=policy_kwargs)
    elif algo == "td3":
        DefaultAlgo = partial(TD3, policy_kwargs=policy_kwargs)
    elif algo == "trpo":
        if TRPO is None:
            raise ImportError(
                "TRPO selected but sb3_contrib is not installed. "
                "Install with `pip install sb3-contrib`."
            )
        DefaultAlgo = partial(TRPO, policy_kwargs=policy_kwargs)
    else:
        raise ValueError(f"Unsupported algo '{algo}'")

    # Currently use logger instead of setting params: tensorboard_log=str(log_dir), verbose=0
    logger = configure(str(log_dir), ["tensorboard", "csv", "json"])
    model = DefaultAlgo(
        "MlpPolicy",
        train_env,
        seed=seed,
        **algo_kwargs,
    )
    model.set_logger(logger)

    # Setup callbacks
    save_freq = log_kwargs.get("save_freq", 100000)
    eval_freq = log_kwargs.get("eval_freq", 10000)
    step_log_interval = log_kwargs.get("interval", 10000)

    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=n_eval_episodes,
        best_model_save_path=str(save_dir / "best_model"),
        log_path=str(log_dir / "eval"),
        eval_freq=max(eval_freq // n_envs, 1),
        deterministic=True,
        render=False,
    )

    # Checkpoint freq is based on calls to step(), so we need to adjust for n_envs
    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq // n_envs, 1),
        save_path=str(save_dir),
        name_prefix=f"{algo}_{env_id}",
    )

    # Timestep-based logging callback
    log_callback = LogEveryNTimesteps(n_steps=max(step_log_interval // n_envs, 1))
    # progress_bar_callback = ProgressBarCallback()

    # Combine callbacks
    callback = CallbackList([checkpoint_callback, eval_callback, log_callback])

    # Train
    print(
        f"\n[SB3 {algo.upper()}] env={env_id} mode={mode}\n"
        f"total_timesteps={total_timesteps}\n\n"
        f"increment_modeling={increment_modeling}"
    )
    if increment_modeling:
        print(f"  new_gamma={algo_kwargs.get('gamma')}\n")
    print(f"\npolicy_kwargs={policy_kwargs}\n")
    print(f"env_meta={env_meta}\n")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        tb_log_name=f"{algo}_{env_id}",
        log_interval=10**9,  # Disable SB3's episode-based logging
    )
    model.save(str(save_dir / "final_model"))
    print("Training finished.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algos",
        type=str,
        default="sac",
        help="Comma-separated list of SB3 algorithms to run (e.g., 'sac,ppo').",
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default="cheetah-run",
        help="DMC env as 'domain-task', e.g. 'cheetah-run', 'walker-walk'.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="default",
        help="Mode key matching the CSV 'mode' column.",
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        default=None,
        help="Evaluation mode key for EvalCallback. Defaults to --mode if not set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--hyperparams_dir",
        type=str,
        default="benchmarks/hyperparams",
        help="Directory containing per-algo hyperparam CSVs.",
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default="logs",
        help="Log root relative to project dir.",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="saved_models",
        help="Model save root relative to project dir.",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=None,
        help="Override total timesteps (otherwise use table).",
    )
    parser.add_argument(
        "--desc",
        type=str,
        default="",
        help="Optional description to append to run directory name.",
    )
    parser.add_argument(
        "--increment_modeling",
        action="store_true",
        help="If set, use reward increment modeling with adjusted gamma.",
    )
    parser.add_argument(
        "--n_eval_episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate during EvalCallback.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    algos_to_run = [algo.strip() for algo in args.algos.split(",") if algo.strip()]

    for algo_name in algos_to_run:
        try:
            run_sb3_benchmark(
                algo=algo_name.lower(),
                env_id=args.env_id,
                mode=args.mode,
                eval_mode=args.eval_mode,
                seed=args.seed,
                hyperparams_dir=args.hyperparams_dir,
                log_root_dir=args.log_root,
                save_root_dir=args.save_root,
                total_timesteps_override=args.total_timesteps,
                desc=args.desc,
                increment_modeling=args.increment_modeling,
                n_eval_episodes=args.n_eval_episodes,
            )
        except (FileNotFoundError, KeyError) as e:
            print(f"\nCould not run {algo_name} due to a configuration error: {e}\n")


if __name__ == "__main__":
    main()
