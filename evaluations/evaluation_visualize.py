# evaluations/evaluation_visualize.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Type

import numpy as np
import imageio
import cv2

from stable_baselines3.common.base_class import BaseAlgorithm
from environment.base import ContinuousEnv
from models.base import Model
from .evaluation_helpers import (
    evaluate_policy_per_step,
    evaluate_sb3_policy_per_step,
)


def _generate_episode_frames(
    model_or_path: Union[Model, str],
    env: ContinuousEnv,
    is_sb3_model: bool = False,
    sb3_algo_class: Optional[Type[BaseAlgorithm]] = None,
    render_interval: int = 10,
) -> List[np.ndarray]:
    """
    Internal helper to load a model state and generate frames for one episode.
    This function can handle both custom models and Stable-Baselines3 models.

    :param model_or_path: Either a path to a model file (.pth or .zip) or an instantiated custom Model.
    :param env: The environment to evaluate on.
    :param is_sb3_model: Flag to indicate if the model is a Stable-Baselines3 model.
    :param sb3_algo_class: The SB3 algorithm class (e.g., SAC) to use if loading a .zip file.
    :return: A list of image frames for the episode.
    """
    frames_list = []
    if is_sb3_model:
        if not isinstance(model_or_path, str) or not model_or_path.endswith(".zip"):
            raise ValueError(
                "For SB3 models, `model_or_path` must be a path to a .zip file."
            )
        if sb3_algo_class is None:
            raise ValueError("`sb3_algo_class` must be provided for SB3 models.")

        model_path = model_or_path
        print(f"  - Generating frames for SB3 model: {Path(model_path).name}")
        sb3_model = sb3_algo_class.load(model_path, env=env)

        # Use the dedicated SB3 evaluation function to get frames
        evaluate_results_dict = evaluate_sb3_policy_per_step(
            sb3_model,
            env,
            n_eval_episodes=1,
            deterministic=True,
            render=True,
            render_interval=render_interval,
        )
        frames_list = evaluate_results_dict.get("episode_frames", [])

    elif isinstance(model_or_path, Model):
        # Continuous-time model evaluation
        print(f"  - Generating frames for custom model.")
        evaluate_results_dict = evaluate_policy_per_step(
            model_or_path,
            env,
            n_eval_episodes=1,
            deterministic=True,
            render=True,
            render_interval=render_interval,
        )
        frames_list = evaluate_results_dict.get("episode_frames", [])

    elif isinstance(model_or_path, str) and model_or_path.endswith(".pth"):
        model_path = model_or_path
        raise ValueError(
            f"For .pth files ('{model_path}'), please pass an instantiated model object, not a path string."
        )

    if frames_list and frames_list[0]:
        return frames_list[0]
    return []


def generate_progression_frames(
    model: Model,
    model_dir: str,
    env: ContinuousEnv,
    title: str,
    render_interval: int = 1,
) -> List[np.ndarray]:
    """
    Generates a sequence of frames showing the learning progression of a custom agent.

    :param model: An instance of the model (e.g., ActorQCriticModel) to load states into.
    :param model_dir: Directory containing model files (e.g., '..._10000_steps.pth' or 'best_model.pth').
    :param env: The environment to evaluate on.
    :return: A list of all frames for the progression video.
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
        return []

    all_frames = []
    for file_path in files:
        # For continuous-time models, load state into the provided model instance
        print(f"Processing model file: {file_path.name}")
        model.load_state(str(file_path))
        frames = _generate_episode_frames(
            model, env, is_sb3_model=False, render_interval=render_interval
        )

        if frames:
            # Add a title card before each segment
            step_match = re.search(r"_(\d+)_steps\.pth$", str(file_path))
            if step_match:
                step_count = step_match.group(1)
                card_title = f"{title}: {step_count} steps"
            else:
                card_title = f"{title}: {file_path.stem}"
            title_card = np.zeros_like(frames[0])
            cv2.putText(
                title_card,
                card_title,
                (50, title_card.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            # Add the title card for a short duration
            all_frames.extend([title_card] * 60)  # 2s at 30fps
            all_frames.extend(frames)

    return all_frames


def generate_sb3_progression_frames(
    sb3_algo_class: Type[BaseAlgorithm],
    model_dir: str,
    env: ContinuousEnv,
    title: str,
    render_interval: int = 1,
) -> List[np.ndarray]:
    """
    Generates a sequence of frames showing the learning progression of an SB3 agent.

    :param sb3_algo_class: The SB3 algorithm class (e.g., `stable_baselines3.SAC`).
    :param model_dir: Directory containing SB3 model files (e.g., '..._N_steps.zip').
    :param env: The environment to evaluate on.
    :return: A list of all frames for the progression video.
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
        return []

    all_frames = []
    for file_path in files:
        print(f"Processing SB3 model file: {file_path.name}")
        frames = _generate_episode_frames(
            str(file_path),
            env,
            is_sb3_model=True,
            sb3_algo_class=sb3_algo_class,
            render_interval=render_interval,
        )

        if frames:
            # Add a title card before each segment
            step_match = re.search(r"_(\d+)_steps\.zip$", str(file_path))
            if step_match:
                step_count = step_match.group(1)
                card_title = f"{title}: {step_count} steps"
            else:
                card_title = f"{title}: {file_path.stem}"
            title_card = np.zeros_like(frames[0])
            cv2.putText(
                title_card,
                card_title,
                (50, title_card.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            # Add the title card for a short duration
            all_frames.extend([title_card] * 60)  # 2s at 30fps
            all_frames.extend(frames)

    return all_frames


def create_comparison_video(
    models_to_compare: Dict[Union[Model, Type[BaseAlgorithm]], Tuple[str, str]],
    env: ContinuousEnv,
    output_path: str,
    render_interval: int = 1,
    fps: int = 30,
) -> None:
    """
    Creates a side-by-side comparison video of multiple trained agents.

    :param models_to_compare: A dictionary mapping a model object/class to its model directory and title.
    :param env: The environment to evaluate on.
    :param output_path: Path to save the final comparison video.
    :param fps: Frames per second for the output video.
    """

    # Generate frames for each model
    all_model_frames = []
    for model_obj, (model_dir, model_title) in models_to_compare.items():
        if isinstance(model_obj, Model):
            frames = generate_progression_frames(
                model_obj,
                model_dir,
                env,
                title=model_title,
                render_interval=render_interval,
            )
        elif isinstance(model_obj, type) and issubclass(model_obj, BaseAlgorithm):
            frames = generate_sb3_progression_frames(
                model_obj,
                model_dir,
                env,
                title=model_title,
                render_interval=render_interval,
            )
        else:
            raise TypeError(f"Unsupported model type in dictionary: {type(model_obj)}")
        all_model_frames.append(frames)

    # Align and combine frames
    max_len = (
        max(len(frames) for frames in all_model_frames if frames)
        if any(all_model_frames)
        else 0
    )
    if max_len == 0:
        print("No frames were generated for any model. Cannot create video.")
        return

    for i, frames in enumerate(all_model_frames):
        # Pad shorter episodes with their last frame
        if not frames:
            h, w, c = all_model_frames[0][0].shape
            all_model_frames[i] = [np.zeros((h, w, c), dtype=np.uint8)] * max_len
        else:
            last_frame = frames[-1]
            padding = [last_frame] * (max_len - len(frames))
            all_model_frames[i].extend(padding)

    # Create the side-by-side video
    combined_video_frames = []
    for i in range(max_len):
        frames_to_combine = [model_frames[i] for model_frames in all_model_frames]
        combined_frame = np.hstack(frames_to_combine)
        combined_video_frames.append(combined_frame)

    if combined_video_frames:
        print(f"Saving comparison video to {output_path}")
        imageio.mimsave(output_path, combined_video_frames, fps=fps)
