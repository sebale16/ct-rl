#!/usr/bin/env python
"""Render one deterministic episode of the final checkpoint for each of the
10 irregular1m arms (release-from-rest, seed 20000, the standard xk_eval
release-from-rest condition -- same task/time-regime used throughout the
irregular1m comparison).

Headless (EGL). One-off analysis script (not part of the training/eval
pipeline), mirroring benchmarks/render_acrobot_videos.py's structure but
generalized to both CT (ct_sac/ct_td3) and SB3 (sac/td3) checkpoints via
evaluations.evaluation_helpers.evaluate_policy_per_step /
evaluate_sb3_policy_per_step, which already handle both model families
against the same raw ContinuousEnv.

Episode length is capped (env_max_steps override) purely to keep clips a
reasonable size for embedding in an artifact -- this is for visualization,
not scoring, so it isn't run through the fixed 32-seed protocol.

Usage:
    MUJOCO_GL=egl python -m benchmarks.render_irregular1m_arm_videos \\
        --out-dir videos/irregular1m
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio
import numpy as np
from stable_baselines3 import SAC, TD3

from benchmarks.run_ct_rl import ACROBOT_XK_TERMINATION_TASK_KEYS, make_ct_env
from common.demonstration import ACROBOT_XK_ENV_ID
from common.utils import (
    build_save_path,
    load_ct_hyperparams_from_table,
    load_sb3_hyperparams_from_table,
)
from evaluations.evaluate_swingup_final import _load_model
from evaluations.evaluation_helpers import (
    evaluate_policy_per_step,
    evaluate_sb3_policy_per_step,
)

STEM = "xk_r3_eta0p23_ctrl10ms_h2s_temp0p01_xkdot_q2dot4pi_logrecip"
BASELINE_MODE = f"{STEM}_tau1p25e2_irregular1m"
DEMO_MODE = f"{STEM}_xkdemo20k_tau1p25e2_irregular1m"
ANNEAL0_MODE = f"{STEM}_xkklrev_xkdemo20k_tau1p25e2_anneal0span60k_irregular1m"
ANNEAL05_MODE = f"{STEM}_xkklrev_xkdemo20k_tau1p25e2_anneal0p5span60k_irregular1m"

ARMS = [
    ("ct_sac", BASELINE_MODE, "ct_sac_baseline", "ct_sac baseline"),
    ("ct_sac", DEMO_MODE, "ct_sac_demo", "ct_sac +demo"),
    ("ct_sac", ANNEAL0_MODE, "ct_sac_anneal0", "ct_sac +demo+KL→0"),
    ("ct_sac", ANNEAL05_MODE, "ct_sac_anneal05", "ct_sac +demo+KL→0.5"),
    ("ct_td3", BASELINE_MODE, "ct_td3_baseline", "ct_td3 baseline"),
    ("ct_td3", DEMO_MODE, "ct_td3_demo", "ct_td3 +demo"),
    ("sac", BASELINE_MODE, "sac_baseline", "sac baseline"),
    ("sac", DEMO_MODE, "sac_demo", "sac +demo"),
    ("td3", BASELINE_MODE, "td3_baseline", "td3 baseline"),
    ("td3", DEMO_MODE, "td3_demo", "td3 +demo"),
]

CT_ALGOS = ("ct_sac", "ct_td3")
SEED = 20000
RUN_ID = "irregular1m_v1"
HP = "benchmarks/hyperparams"
RENDER_MAX_STEPS = 6000  # cap episode length for clip size, not scoring
RENDER_INTERVAL = 15
WIDTH, HEIGHT = 320, 240
FPS = 24


def _eval_env_kwargs(algo: str, mode: str) -> tuple[dict, dict]:
    loader = load_ct_hyperparams_from_table if algo in CT_ALGOS else load_sb3_hyperparams_from_table
    _, eval_meta, *_ = loader(algo, ACROBOT_XK_ENV_ID, "xk_eval", hyperparams_dir=HP)
    _, train_meta, *_ = loader(algo, ACROBOT_XK_ENV_ID, mode, hyperparams_dir=HP)
    task = dict(eval_meta.get("task_kwargs") or {})
    train_task = dict(train_meta.get("task_kwargs") or {})
    for key in ACROBOT_XK_TERMINATION_TASK_KEYS:
        if key in train_task:
            task[key] = train_task[key]
    env_kwargs = dict(eval_meta)
    env_kwargs["task_kwargs"] = task
    env_kwargs["max_steps"] = RENDER_MAX_STEPS
    env_kwargs.pop("n_envs", None)
    return env_kwargs, train_meta


def render_arm(algo: str, mode: str, label: str, title: str, out_dir: str) -> str | None:
    env_kwargs, train_meta = _eval_env_kwargs(algo, mode)
    env = make_ct_env(env_id=ACROBOT_XK_ENV_ID, seed=SEED, env_kwargs=env_kwargs)

    save_dir = build_save_path(
        "saved_models", algo, ACROBOT_XK_ENV_ID, mode, 0, train_meta, "", run_id=RUN_ID
    )
    try:
        if algo in CT_ALGOS:
            checkpoint = save_dir / "final_model.pth"
            if not checkpoint.is_file():
                print(f"{label}: NO CHECKPOINT at {checkpoint}")
                return None
            _, _, model_kwargs, _, _ = load_ct_hyperparams_from_table(
                algo, ACROBOT_XK_ENV_ID, mode, hyperparams_dir=HP
            )
            model = _load_model(env, model_kwargs, checkpoint, "cpu")
            out = evaluate_policy_per_step(
                model, env, n_eval_episodes=1, deterministic=True,
                render=True, render_interval=RENDER_INTERVAL,
            )
        else:
            checkpoint = save_dir / "final_model.zip"
            if not checkpoint.is_file():
                print(f"{label}: NO CHECKPOINT at {checkpoint}")
                return None
            AlgoClass = {"sac": SAC, "td3": TD3}[algo]
            sb3_model = AlgoClass.load(str(checkpoint), env=None)
            out = evaluate_sb3_policy_per_step(
                sb3_model, env, n_eval_episodes=1, deterministic=True,
                render=True, render_interval=RENDER_INTERVAL,
            )
    finally:
        env.close()

    frames = out.get("episode_frames") or []
    if not frames or not frames[0]:
        print(f"{label}: NO FRAMES")
        return None
    frames = [np.asarray(f, dtype=np.uint8) for f in frames[0]]
    ep_len = out["episode_lengths"][0]
    ep_return = out["episode_returns"][0]
    path = os.path.join(out_dir, f"{label}.mp4")
    imageio.mimsave(path, frames, fps=FPS, macro_block_size=None)
    print(
        f"{label} ({title}): {len(frames)} frames, ep_len={ep_len}, "
        f"return={ep_return:.1f} -> {path}",
        flush=True,
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="videos/irregular1m")
    parser.add_argument("--only", default=None, help="Comma-separated labels to render")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    only = set(args.only.split(",")) if args.only else None
    for algo, mode, label, title in ARMS:
        if only is not None and label not in only:
            continue
        render_arm(algo, mode, label, title, args.out_dir)


if __name__ == "__main__":
    main()
