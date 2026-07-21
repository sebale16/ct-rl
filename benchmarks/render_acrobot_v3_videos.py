#!/usr/bin/env python
"""Render one full deterministic episode of the v3-pilot best_model policy for
model-free and oracle on acrobot-swingup-v3 -> per-mode MP4 + a 1x2 grid.
Headless EGL (GPU node). Full 10s episode; grid uses every frame (same
speed/length as the per-mode clips).
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import gc, glob
import numpy as np
import imageio
import cv2

from benchmarks.run_ct_rl import make_ct_env
from common.utils import load_ct_hyperparams_from_table
from models import ActorQCriticModel
from evaluations.evaluation_visualize import generate_progression_frames

ENV_ID = "acrobot-swingup-v3"
RUN_TAG = "acrov3_pilot_v1"
SEED = 0
EP_DURATION = 10.0
FPS = int(os.environ.get("ACRO_VIDEO_FPS", "25"))
OUT = "out/media/videos_acrobot_v3"
os.makedirs(OUT, exist_ok=True)

MODES = [("final_mf", "model-free"), ("final_oracle_rollout", "oracle")]
grid_seq = {}

for mode, title in MODES:
    tt, ek, mk, ak, lk = load_ct_hyperparams_from_table(
        algo="ct_sac", env_id=ENV_ID, mode=mode, hyperparams_dir="benchmarks/hyperparams"
    )
    ek.pop("n_envs", None)
    ek["episode_duration"] = EP_DURATION
    env = make_ct_env(env_id=ENV_ID, seed=SEED + 1000, env_kwargs=ek)
    model = ActorQCriticModel(observation_space=env.observation_space,
                              action_space=env.action_space, **mk)
    cand = glob.glob(f"saved_models/ct_sac/{ENV_ID}/{mode}/seed_{SEED}/*{RUN_TAG}*/best_model")
    if not cand:
        print(f"{mode}: NO best_model dir", flush=True); continue
    frames, _ = generate_progression_frames(model, cand[0], env, title=title)
    if not frames:
        print(f"{mode}: NO FRAMES", flush=True); continue
    frames = [np.asarray(f, dtype=np.uint8) for f in frames]
    imageio.mimsave(f"{OUT}/acrobot_v3_{mode}.mp4", frames, fps=FPS)
    print(f"{mode}: {len(frames)} frames -> {OUT}/acrobot_v3_{mode}.mp4", flush=True)
    grid_seq[mode] = ([cv2.resize(f, (0, 0), fx=0.5, fy=0.5) for f in frames], title)
    del frames; gc.collect()

# ---- 1x2 comparison grid (same frame count/speed as per-mode clips) ----
order = [m for m, _ in MODES if m in grid_seq]
if len(order) == 2:
    h, w = grid_seq[order[0]][0][0].shape[:2]
    n = max(len(grid_seq[m][0]) for m in order)
    def labelled(fr, txt):
        fr = fr.copy()
        cv2.putText(fr, txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return fr
    grid = []
    for i in range(n):
        cells = []
        for m in order:
            seq, t = grid_seq[m]
            f = seq[min(i, len(seq) - 1)]
            if f.shape[:2] != (h, w): f = cv2.resize(f, (w, h))
            cells.append(labelled(f, t))
        grid.append(np.hstack(cells))
    imageio.mimsave(f"{OUT}/acrobot_v3_grid.mp4", grid, fps=FPS)
    print(f"grid: {len(grid)} frames -> {OUT}/acrobot_v3_grid.mp4", flush=True)
