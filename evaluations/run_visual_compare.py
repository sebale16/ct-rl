# experiments/run_visual_compare.py
from pathlib import Path
from typing import Dict, Any

from stable_baselines3.common.base_class import BaseAlgorithm

from models.base import Model
from evaluations.evaluation_helpers import (
    create_evaluation_env_and_model,
    ALGO_CLASS_MAP,
)
from evaluations.evaluation_visualize import create_comparison_video


def run_visual_compare(config: Dict[str, Any]):
    """
    Main function to run visual comparisons between different models.
    The models to compare are defined in the config dictionary using strings.
    """
    # Setup
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    env_id: str = config["env_id"]
    mode: str = config["mode"]
    seed: int = config["seed"]
    render_interval: int = config["render_interval"]
    fps: int = config["fps"]
    models_config: Dict[str, Dict[str, str]] = config["models_to_compare"]

    # Create a single environment instance to be shared for evaluation
    env, _ = create_evaluation_env_and_model(
        env_id,
        model_class=None,  # No model needed yet
        seed=seed,
        algo="ct_sac",
        mode=mode,
    )

    # Prepare the models for comparison based on the string configuration
    models_to_compare = {}
    for model_title, model_conf in models_config.items():
        model_dir = model_conf["dir"]
        algo = model_conf["algo"]

        if algo not in ALGO_CLASS_MAP:
            raise ValueError(
                f"Unknown algorithm '{algo}' in config. Please add it to ALGO_CLASS_MAP."
            )

        model_or_algo_class = ALGO_CLASS_MAP[algo]

        if issubclass(model_or_algo_class, Model):
            # For custom algorithms, create a model instance to load state into
            _, model_instance = create_evaluation_env_and_model(
                env_id,
                model_class=model_or_algo_class,
                seed=seed,
                algo=algo,
                mode=mode,
            )
            models_to_compare[model_instance] = (model_dir, model_title)
        elif issubclass(model_or_algo_class, BaseAlgorithm):
            # For SB3 algorithms, we use the algorithm class directly
            models_to_compare[model_or_algo_class] = (model_dir, model_title)

    output_path = output_dir / "visual_comparison.mp4"
    create_comparison_video(
        models_to_compare,
        env,
        output_path,
        render_interval,
        fps=fps,
    )
    print("-" * 50)


#################################### MAIN RUN ####################################

if __name__ == "__main__":
    env_id = "humanoid-walk"
    mode = "irregular_dt"
    prefix_ct = "saved_models/ct_sac/" + env_id + "/"
    prefix_discrete = "saved_models/discrete_benchmarks/sac/" + env_id + "/"
    best_model = True
    suffix = "/best_model" if best_model else ""
    config = {
        "models_to_compare": {
            "CT-SAC": {
                "dir": prefix_ct
                + "irregular_pdt_0_005_dt_0_025_max_steps_1000_irregular_dt_2025-12-20_20-07-31"
                + suffix,
                "algo": "ct_sac",
            },
            "SB3-SAC": {
                "dir": prefix_discrete
                + "irregular_pdt_0_005_dt_0_025_max_steps_1000_irregular_dt_2025-12-20_20-07-50"
                + suffix,
                "algo": "sac",
            },
        },
        "env_id": env_id,
        "mode": mode,
        "seed": 42,
        "output_dir": "out/initial_compare/" + env_id + "/" + mode,
        "render_interval": 1,
        "fps": 30,
    }

    run_visual_compare(config)
