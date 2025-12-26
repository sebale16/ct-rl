# evaluations/run_visual_evaluation.py
from pathlib import Path
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np

from stable_baselines3.common.base_class import BaseAlgorithm
from models.base import Model
from environment.dmc import DMCContinuousEnv
from evaluations.evaluation_helpers import (
    create_evaluation_env_and_model,
    evaluate_policy_per_step,
    evaluate_sb3_policy_per_step,
    ALGO_CLASS_MAP,
)
from common.utils import (
    load_ct_hyperparams_from_table,
    load_sb3_hyperparams_from_table,
)


def generate_single_model_video(config):
    """
    Generates a video of a single trajectory for a trained agent.
    Also plots the rewards from a final evaluation run.
    """
    # Setup
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = config.get("checkpoint_dir")
    env_id: str = config["env_id"]
    mode: str = config["mode"]
    algo: str = config["algo"]
    seed: int = config["seed"]
    title: str = config["title"]
    render_interval: int = config["render_interval"]

    if algo not in ALGO_CLASS_MAP:
        raise ValueError(
            f"Unknown algorithm '{algo}'. Please add it to ALGO_CLASS_MAP."
        )

    model_or_algo_class = ALGO_CLASS_MAP[algo]

    # --- Create Environment ---
    # Load hyperparameters to get environment settings
    env_kwargs = {}
    if issubclass(model_or_algo_class, Model):
        _, env_kwargs, _, _, _ = load_ct_hyperparams_from_table(
            algo=algo, env_id=env_id, mode=mode
        )
    elif issubclass(model_or_algo_class, BaseAlgorithm):
        _, env_kwargs, _, _, _ = load_sb3_hyperparams_from_table(
            algo=algo, env_id=env_id, mode=mode
        )
        env_kwargs.pop("id")
        print(env_kwargs)

    # Instantiate the environment directly
    env_kwargs.pop("n_envs")
    domain_name, task_name = env_id.split("-", 1)
    env = DMCContinuousEnv(
        domain_name=domain_name,
        task_name=task_name,
        seed=seed,
        **env_kwargs,
    )

    # --- Load final model ---
    print(f"--- Loading final model for {title} ---")
    final_model = None
    if checkpoint_dir is not None:
        if issubclass(model_or_algo_class, Model):
            # Custom model
            _, final_model = create_evaluation_env_and_model(
                env_id,
                model_class=model_or_algo_class,
                seed=seed,
                algo=algo,
                mode=mode,
                env_kwargs=env_kwargs,
            )
            # Assumes the checkpoint_dir points to the final model file or a dir with best_model
            model_path = Path(checkpoint_dir)
            if model_path.is_dir():
                model_path = model_path / "best_model.pth"
            if model_path.exists():
                final_model.load_state(model_path)
                print(f"Loaded custom model from {model_path}")
            else:
                print(
                    f"Warning: Model file not found at {model_path}, using un-trained model."
                )
        elif issubclass(model_or_algo_class, BaseAlgorithm):
            # SB3 model
            model_path = Path(checkpoint_dir)
            if model_path.is_dir():
                model_path = model_path / "best_model.zip"
            best_model_path = model_path
            if best_model_path.exists():
                final_model = model_or_algo_class.load(best_model_path, env=env)
                print(f"Loaded SB3 model from {best_model_path}")
            else:
                print(
                    f"Warning: SB3 model file not found at {best_model_path}, skipping evaluation."
                )
    else:
        print("No checkpoint_dir provided. Using random policy (model=None).")

    # --- Run Evaluation to get Frames and Metrics ---
    if final_model is not None or checkpoint_dir is None:
        print("\n--- Running Final Evaluation for Video and Plot ---")
        if final_model is None or isinstance(final_model, Model):
            eval_results = evaluate_policy_per_step(
                final_model,
                env,
                n_eval_episodes=1,
                render=True,
                render_interval=render_interval,
            )
        else:  # SB3 model
            eval_results = evaluate_sb3_policy_per_step(
                final_model,
                env,
                n_eval_episodes=1,
                render=True,
                render_interval=render_interval,
            )

        if eval_results["episode_timestamps"]:
            timestamps = eval_results["episode_timestamps"][0]
            step_rewards = eval_results["episode_step_rewards"][0]

            # --- Save Video ---
            frames = eval_results["episode_frames"][0]
            if frames:
                video_path = output_dir / f"{title}_trajectory.mp4"
                print(f"Saving video to {video_path}...")
                imageio.mimsave(str(video_path), frames, fps=config.get("fps", 30))
                print("Done.")
            else:
                print("No frames were generated during evaluation.")

            # --- Plotting single trajectory evaluation ---
            plt.figure(figsize=(12, 7))
            plt.plot(timestamps, step_rewards, marker="o", linestyle="-", markersize=4)
            plt.title(f"Single Trajectory Reward for {title} on {env_id}")
            plt.xlabel("Time (s)")
            plt.ylabel("Cumulative Reward")
            plt.grid(True)
            plot_path = output_dir / f"{title}_trajectory_reward.png"
            plt.savefig(plot_path)
            print(f"Trajectory plot saved to {plot_path}")
            print("\nPlotting results. Close the plot window to exit.")
            plt.show()
        else:
            print("Could not generate trajectory plot: No evaluation data.")


if __name__ == "__main__":
    env_id = "quadruped-run"
    prefix_dir = "saved_models/ct_sac/" + env_id + "/"
    dir_name = None  # prefix_dir + "uniform_pdt_0_0025_dt_0_005_max_steps_5000_small_dt_2025-12-19_14-46-53/best_model"
    # dir_name = "saved_models/discrete_benchmarks/sac/cheetah-run/irregular_pdt_0_002_dt_0_01_max_steps_1000_irregular_dt_hard_2025-12-17_20-04-17/best_model"
    algo = "ct_sac"
    title = "CT-SAC-Walker"  # "SB3-SAC-Cheetah"  #
    config = {
        "checkpoint_dir": None,
        "env_id": env_id,
        "algo": algo,
        "mode": "normal",
        "title": title,
        "seed": 42,
        "output_dir": "out/finegrained_visualization/" + env_id,
        "render_interval": 5,
        "fps": 30,
    }
    generate_single_model_video(config)
