#!/usr/bin/env python
"""Render one deterministic episode of the latest (best_model) policy for each
of the seven acrobot-swingup-v2 final modes -> an MP4 per mode + a 2x4 grid.
Headless EGL (CPU software render; no GPU needed). Login-node-frugal: renders
one mode at a time, frees frames between modes, and caps the episode short.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import gc
import glob
import numpy as np
import imageio
import cv2

from benchmarks.run_ct_rl import make_ct_env
from common.utils import load_ct_hyperparams_from_table
from models import ActorQCriticModel
from evaluations.evaluation_visualize import generate_progression_frames

ENV_ID = "acrobot-swingup-v2"
SEED = 0
EP_DURATION = 10.0         # full episode (matches training ~10s / ~1973 steps)
FPS = int(os.environ.get("ACRO_VIDEO_FPS", "25"))   # ~4x slow-mo
OUT = "out/media/videos_acrobot_v2"
os.makedirs(OUT, exist_ok=True)

MODES = [
    ("final_mf", "model-free"),
    ("final_mf_vhead", "mf+vhead"),
    ("final_oracle_rollout", "oracle"),
    ("final_structured", "learned"),
    ("final_guarded", "guarded"),
    ("final_reanchor", "reanchor"),
    ("final_guarded_reanchor", "guard+reanchor"),
]

grid_lo = {}   # mode -> downsampled frame list (for the comparison grid)
# GRID_STRIDE=1 keeps every frame so the grid plays at the SAME speed/length as
# the per-mode videos; GRID_SCALE only shrinks cell resolution (not the speed).
GRID_STRIDE, GRID_SCALE = 1, 0.5

for mode, title in MODES:
    tt, ek, mk, ak, lk = load_ct_hyperparams_from_table(
        algo="ct_sac", env_id=ENV_ID, mode=mode, hyperparams_dir="benchmarks/hyperparams"
    )
    ek.pop("n_envs", None)
    ek["episode_duration"] = EP_DURATION
    env = make_ct_env(env_id=ENV_ID, seed=SEED + 1000, env_kwargs=ek)
    model = ActorQCriticModel(
        observation_space=env.observation_space,
        action_space=env.action_space,
        **mk,
    )
    cand = glob.glob(
        f"saved_models/ct_sac/{ENV_ID}/{mode}/seed_{SEED}/*swingup_final_v1*/best_model"
    )
    if not cand:
        print(f"{mode}: NO best_model dir", flush=True)
        continue
    frames, _ = generate_progression_frames(model, cand[0], env, title=title)
    if not frames:
        print(f"{mode}: NO FRAMES", flush=True)
        continue
    frames = [np.asarray(f, dtype=np.uint8) for f in frames]
    imageio.mimsave(f"{OUT}/acrobot_v2_{mode}.mp4", frames, fps=FPS)
    print(f"{mode}: {len(frames)} frames -> {OUT}/acrobot_v2_{mode}.mp4", flush=True)
    # keep only a small downsampled copy for the grid, then free the rest
    lo = [cv2.resize(frames[i], (0, 0), fx=GRID_SCALE, fy=GRID_SCALE)
          for i in range(0, len(frames), GRID_STRIDE)]
    grid_lo[mode] = (lo, title)
    del frames
    gc.collect()

# ---- 2x4 comparison grid (7 modes + 1 blank), from downsampled frames ----
order = [m for m, _ in MODES if m in grid_lo]
if order:
    h, w = grid_lo[order[0]][0][0].shape[:2]
    n = max(len(grid_lo[m][0]) for m in order)

    def labelled(fr, txt):
        fr = fr.copy()
        cv2.putText(fr, txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return fr

    blank = np.zeros((h, w, 3), dtype=np.uint8)
    grid = []
    for i in range(n):
        cells = []
        for m in order:
            seq, t = grid_lo[m]
            f = seq[min(i, len(seq) - 1)]
            if f.shape[:2] != (h, w):
                f = cv2.resize(f, (w, h))
            cells.append(labelled(f, t))
        while len(cells) < 8:
            cells.append(blank)
        row0 = np.hstack(cells[0:4])
        row1 = np.hstack(cells[4:8])
        grid.append(np.vstack([row0, row1]))
    imageio.mimsave(f"{OUT}/acrobot_v2_all_modes_grid.mp4", grid, fps=FPS)
    print(f"grid: {len(grid)} frames -> {OUT}/acrobot_v2_all_modes_grid.mp4", flush=True)
