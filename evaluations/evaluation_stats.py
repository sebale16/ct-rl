# evaluations/evaluation_stats.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Union, Type

import numpy as np

from evaluations.evaluation_helpers import (
    evaluate_policy_per_step,
    evaluate_sb3_policy_per_step,
)
from stable_baselines3.common.base_class import BaseAlgorithm
from environment.base import ContinuousEnv
from models.base import Model


def benchmark_progression(
    model: Model,
    model_dir: str,
    env: ContinuousEnv,
    n_eval_episodes: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates a model's performance across multiple checkpoints and returns numerical stats.

    :param model: An instance of the model (e.g., ActorQCriticModel) to load states into.
    :param model_dir: Directory containing model files (e.g., '..._N_steps.pth' or 'best_model.pth').
    :param env: The environment to evaluate on.
    :param n_eval_episodes: Number of episodes to run for each checkpoint.
    :return: A dictionary mapping step string or model name to a dict of stats {'mean_reward': ..., 'std_reward': ...}.
    """
    path = Path(model_dir)
    files = list(path.glob("*.pth"))

    # Sort files: numerically if they are checkpoints, alphabetically otherwise
    if any("_steps" in f.name for f in files):
        files = sorted(
            [f for f in files if "_steps" in f.name],
            key=lambda f: int(re.search(r"_(\d+)_steps\.pth$", str(f)).group(1)),
        )
    else:
        files = sorted(files)

    if not files:
        print(f"No .pth files found in {model_dir}")
        return {}

    results = {}
    for file_path in files:
        print(f"Benchmarking model file: {file_path.name}")
        model.load_state(str(file_path))

        evaluate_results = evaluate_policy_per_step(
            model,
            env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
        )
        rewards = evaluate_results["episode_returns"]

        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))

        step_match = re.search(r"_(\d+)_steps\.pth$", str(file_path))
        if step_match:
            key = step_match.group(1)
        else:
            key = file_path.stem
        results[key] = {"mean_reward": mean_reward, "std_reward": std_reward}
        print(
            f"  - Step/Model '{key}': Mean Reward = {mean_reward:.2f} +/- {std_reward:.2f}"
        )

    return results


def benchmark_sb3_progression(
    sb3_algo_class: Type[BaseAlgorithm],
    model_dir: str,
    env: ContinuousEnv,
    n_eval_episodes: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates a Stable-Baselines3 model's performance across multiple checkpoints.

    :param sb3_algo_class: The SB3 algorithm class (e.g., `stable_baselines3.SAC`).
    :param model_dir: Directory containing SB3 model files (e.g., '..._N_steps.zip').
    :param env: The environment to evaluate on.
    :param n_eval_episodes: Number of episodes to run for each checkpoint.
    :return: A dictionary mapping step string or model name to a dict of stats {'mean_reward': ..., 'std_reward': ...}.
    """
    path = Path(model_dir)
    files = list(path.glob("*.zip"))

    if any("_steps" in f.name for f in files):
        files = sorted(
            [f for f in files if "_steps" in f.name],
            key=lambda f: int(re.search(r"_(\d+)_steps\.zip$", str(f)).group(1)),
        )
    else:
        files = sorted(files)

    if not files:
        print(f"No SB3 .zip files found in {model_dir}")
        return {}

    results = {}
    for file_path in files:
        print(f"Benchmarking SB3 model file: {file_path.name}")
        sb3_model = sb3_algo_class.load(str(file_path), env=env)

        evaluate_results = evaluate_sb3_policy_per_step(
            sb3_model,
            env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
        )
        rewards = evaluate_results["episode_returns"]
        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))

        step_match = re.search(r"_(\d+)_steps\.zip$", str(file_path))
        if step_match:
            key = step_match.group(1)
        else:
            key = file_path.stem
        results[key] = {"mean_reward": mean_reward, "std_reward": std_reward}
        print(
            f"  - Step/Model '{key}': Mean Reward = {mean_reward:.2f} +/- {std_reward:.2f}"
        )

    return results


def benchmark_comparison(
    models_to_compare: Dict[Union[Model, Type[BaseAlgorithm]], str],
    env: ContinuousEnv,
    n_eval_episodes: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Runs a numerical benchmark comparing multiple trained model type.

    :param models_to_compare: A dictionary mapping a model object to its checkpoint path.
    :param env: The environment to evaluate on.
    :param n_eval_episodes: Number of episodes to run for each model.
    :return: A dictionary mapping model name to its performance stats.
    """
    results = {}
    for model_obj, model_path in models_to_compare.items():
        model_name = Path(model_path).stem
        print(f"Benchmarking model: {model_name}")

        if isinstance(model_obj, Model):
            # Custom model instance
            model_obj.load_state(model_path)
            rewards, _, _ = evaluate_policy_per_step(
                model_obj,
                env,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False,
            )
        elif isinstance(model_obj, type) and issubclass(model_obj, BaseAlgorithm):
            # SB3 Algorithm Class
            sb3_model = model_obj.load(model_path, env=env)
            rewards, _, _ = evaluate_sb3_policy_per_step(
                sb3_model,
                env,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False,
            )
        else:
            raise TypeError(f"Unsupported model type in dictionary: {type(model_obj)}")

        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))
        results[model_name] = {"mean_reward": mean_reward, "std_reward": std_reward}
        print(f"  - {model_name}: Mean Reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    return results
