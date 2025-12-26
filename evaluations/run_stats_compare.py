# experiments/run_stats_compare.py
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

from stable_baselines3.common.base_class import BaseAlgorithm
from models.base import Model
from evaluations.evaluation_stats import (
    benchmark_progression,
    benchmark_sb3_progression,
)
from evaluations.evaluation_helpers import (
    create_evaluation_env_and_model,
    ALGO_CLASS_MAP,
)


def run_stats_comparison(config: Dict[str, Any]):
    """
    Main function to run numerical comparisons between different models.
    The models to compare are defined in the config dictionary using strings.
    """
    # Setup
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    env_id: str = config["env_id"]
    mode: str = config["mode"]
    seed: int = config["seed"]
    n_eval_episodes: int = config["n_eval_episodes"]
    models_config: Dict[str, Dict[str, str]] = config["models_to_compare"]

    # Create a single environment instance to be shared for evaluation
    env, _ = create_evaluation_env_and_model(
        env_id,
        model_class=None,  # No model needed yet
        seed=seed,
        algo="ct_sac",
        mode=mode,
    )

    print("--- Running Numerical Benchmarks ---")
    all_results_dfs: List[pd.DataFrame] = []

    for model_title, model_conf in models_config.items():
        model_dir = model_conf["dir"]
        algo = model_conf["algo"]
        print(f"\n--- Benchmarking '{model_title}' ({algo}) ---")

        if algo not in ALGO_CLASS_MAP:
            raise ValueError(
                f"Unknown algorithm '{algo}' in config. Please add it to ALGO_CLASS_MAP."
            )

        model_or_algo_class = ALGO_CLASS_MAP[algo]
        results_dict = {}

        if issubclass(model_or_algo_class, Model):
            # For custom algorithms, create a model instance to load state into
            _, model_instance = create_evaluation_env_and_model(
                env_id,
                model_class=model_or_algo_class,
                seed=seed,
                algo=algo,
                mode=mode,
            )
            results_dict = benchmark_progression(
                model_instance, model_dir, env, n_eval_episodes
            )
        elif issubclass(model_or_algo_class, BaseAlgorithm):
            # For SB3 algorithms, we use the algorithm class directly
            results_dict = benchmark_sb3_progression(
                model_or_algo_class, model_dir, env, n_eval_episodes
            )

        if results_dict:
            df = pd.DataFrame.from_dict(results_dict, orient="index")
            df[model_title] = df.apply(
                lambda row: f"{row['mean_reward']:.2f} ± {row['std_reward']:.2f}",
                axis=1,
            )
            all_results_dfs.append(df[[model_title]])

    # Combine table
    if not all_results_dfs:
        print("No results were generated. Exiting.")
        return

    comparison_df = pd.concat(all_results_dfs, axis=1)
    comparison_df = comparison_df.sort_index(
        key=lambda x: pd.to_numeric(x, errors="coerce").fillna(float("inf"))
    )
    comparison_df.index.name = "Step / Model"

    print("\n--- Benchmark Comparison Table ---")
    print(comparison_df.to_string())
    print("-" * 50)

    # Save to CSV
    csv_path = output_dir / "benchmark_comparison.csv"
    comparison_df.to_csv(csv_path)
    print(f"Benchmark table saved to {csv_path}")


if __name__ == "__main__":
    env_id = "quadruped-run"
    mode = "irregular_dt"
    prefix_ct = "saved_models/ct_sac/" + env_id + "/"
    prefix_discrete = "saved_models/discrete_benchmarks/sac/" + env_id + "/"
    best_model = True
    suffix = "/best_model" if best_model else ""
    config = {
        "models_to_compare": {
            "CT-SAC": {
                "dir": prefix_ct
                + "irregular_pdt_0_005_dt_0_02_max_steps_2000_irregular_dt_hard_2025-12-20_18-01-20"
                + suffix,
                "algo": "ct_sac",
            },
            "SB3-SAC": {
                "dir": prefix_discrete
                + "irregular_pdt_0_005_dt_0_02_max_steps_2000_irregular_dt_hard_2025-12-20_18-01-54"
                + suffix,
                "algo": "sac",
            },
        },
        "env_id": env_id,
        "mode": mode,
        "seed": 0,
        "output_dir": "out/initial_compare/" + env_id + "/" + mode,
        "n_eval_episodes": 20,
    }
    run_stats_comparison(config)
