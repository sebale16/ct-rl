#!/usr/bin/env python
"""Reconstruct resumable fork checkpoints for the paired continuation experiment.

For each fork point (seed, step K) build a full checkpoint that the grid can
--resume, assembled from artifacts that DO exist:
  * model.pth / model.dynamics.pth  <- the periodic weights+dynamics save nearest K
  * buffer.npz                      <- the FINAL buffer's prefix [0:K] (insertion
                                       order => the replay contents at step K)
  * train_state.pt                  <- fresh optimizer (empty), counters with
                                       num_timesteps=K, log_alpha read from the TB
                                       log at K, and ONE SHARED rng state so the
                                       three target-branches start bit-identically.

The reconstructed start is NOT bit-identical to the original run at K (original
optimizer moments / RNG were never saved), but the three branches ARE identical
to each other from this common start, differing only in the critic-target
treatment -- which is what the causal A/B needs.

    python -m benchmarks.build_fork_checkpoints
"""
import glob
import os
import re
import shutil

import numpy as np
import torch as th

LEARNED = "mbq_structured_quad_roll"
SAVED = f"saved_models/ct_sac/cartpole-swingup/{LEARNED}"
CHAIN = f".chain_mbq_cforce_grid/cartpole-swingup__{LEARNED}"
FORK_ROOT = ".chain_fork"
BRANCHES = ["base", "oracle", "mf"]          # -> fork_base / fork_oracle / fork_mf
N_CONT = 120_000                             # continuation length beyond K
RNG_SEED = 12345

# fork points: (name, seed, target step)
FORKS = [
    ("seed1_at100k", 1, 100_000),
    ("seed5_at124k", 5, 124_000),
    ("seed11_at356k", 11, 356_000),
    ("ctrl_seed6_at239k", 6, 240_000),
]


def nearest_periodic(seed, target):
    cands = []
    for f in glob.glob(f"{SAVED}/seed_{seed}/*/ct_sac_*_*_steps.pth"):
        if f.endswith(".dynamics.pth"):
            continue
        m = re.search(r"_(\d+)_steps\.pth$", f)
        if m:
            cands.append((int(m.group(1)), f))
    return min(cands, key=lambda kf: abs(kf[0] - target))


def alpha_at(seed, step):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    xs, ys = [], []
    for f in glob.glob(f"logs/ct_sac/cartpole-swingup/{LEARNED}/seed_{seed}/*/events.out.tfevents.*"):
        try:
            ea = EventAccumulator(f, size_guidance={'scalars': 0}); ea.Reload()
            if "train/alpha" in ea.Tags().get('scalars', []):
                for e in ea.Scalars("train/alpha"):
                    xs.append(e.step); ys.append(e.value)
        except Exception:
            pass
    if not xs:
        return 0.1
    xs, ys = np.array(xs), np.array(ys)
    return float(ys[np.argmin(np.abs(xs - step))])


def shared_rng_state():
    import random
    random.seed(RNG_SEED); np.random.seed(RNG_SEED); th.manual_seed(RNG_SEED)
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": th.get_rng_state(), "torch_cuda": None}


def build_one(name, seed, K, rng):
    k_actual, mpth = nearest_periodic(seed, K)
    dpth = mpth.replace(".pth", ".dynamics.pth")
    # final buffer, kept at full buffer_size with only [0:K] valid (pos=K,
    # full=False); [K:] zeroed so no future transitions leak. checkpoint.load
    # assigns into the full-size ring arrays, so the shape must match buffer_size.
    bnz = np.load(f"{CHAIN}/seed_{seed}/checkpoint/buffer.npz", allow_pickle=False)
    pos = min(int(k_actual), int(bnz["pos"]))
    log_alpha = th.tensor(float(np.log(max(alpha_at(seed, k_actual), 1e-6))))
    state = {"optimizers": {}, "counters": {"num_timesteps": int(k_actual)},
             "log_alpha": log_alpha, "rng": rng, "extra": {}}
    for br in BRANCHES:
        sub = f"{name}__{br}"
        ck = f"{FORK_ROOT}/{sub}/seed_{seed}/checkpoint"
        os.makedirs(ck, exist_ok=True)
        shutil.copy(mpth, f"{ck}/model.pth")
        shutil.copy(dpth, f"{ck}/model.dynamics.pth")
        arrs = {}
        for kk in ("observations", "next_observations", "actions", "rewards",
                   "dones", "t", "next_t", "dt"):
            a = bnz[kk].copy()          # full buffer_size shape
            a[pos:] = 0                 # invalidate everything after the fork step
            arrs[kk] = a
        np.savez_compressed(f"{ck}/buffer.npz", pos=np.int64(pos),
                            full=np.bool_(False), **arrs)
        th.save(state, f"{ck}/train_state.pt")
    return k_actual, pos


def main():
    rng = shared_rng_state()
    print(f"fork root={FORK_ROOT}  N_cont={N_CONT}  rng_seed={RNG_SEED}")
    lines = []
    for name, seed, K in FORKS:
        k_actual, pos = build_one(name, seed, K, rng)
        total = int(k_actual) + N_CONT
        for br in BRANCHES:
            lines.append(f"fork_{br}  {seed}  {total}  {name}__{br}")
        print(f"{name}: seed{seed} K~{k_actual} buffer_prefix={pos} -> total={total}")
    print("\n# CELLS for the fork slurm (mode seed total subdir):")
    print("\n".join("    \"" + ln + "\"" for ln in lines))


if __name__ == "__main__":
    main()
