#!/usr/bin/env python
"""Reconstruct resumable fork checkpoints for the CORRECTED paired continuation.

This supersedes build_fork_checkpoints.py (point 2). The point-2 forks started
the three target branches from a common reconstructed state but let their RNG
streams desync during the continuation (the learned branch draws extra replay
minibatches to fit dynamics, so from step 2 on its critic minibatches differ
from oracle/mf). The corrected experiment fixes that in the trainer (a dedicated
dynamics-fit RNG + a one-time paired global reseed via
--continuation_rng_seed), and here we additionally:

  * persist ``_n_updates`` and ``_value_updates`` in the reconstructed
    train_state so every branch -- including model-free, whose CTSAC.load does
    NOT bump value-head readiness -- starts with an IDENTICAL, value-head-ready
    counter state. (If mf's V-head were not ready its target would sample the
    soft value and consume torch RNG the model-based branches do not, breaking
    the pairing.)
  * add a ``sham`` branch: a second copy of the learned baseline. Run under the
    same paired RNG it must reproduce ``base`` (an A/A null that measures the
    residual float-nondeterminism floor the target contrasts are judged against).

Counters (train_freq=1, gradient_steps=1, learning_starts=10000):
  _n_updates = _value_updates = K - learning_starts  (>> value_warmup=5000).
``_dynamics_updates`` is intentionally left to CTSAC.load, which sets it
identically for the two dynamics-fitting branches (base, sham).

    python -m benchmarks.build_fork_checkpoints2
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
FORK_ROOT = ".chain_fork2"
BRANCHES = ["base", "oracle", "mf", "sham"]   # -> fork_base/fork_oracle/fork_mf/fork_sham
N_CONT = 120_000                              # continuation length beyond K
LEARNING_STARTS = 10_000
RNG_SEED = 12345                              # placeholder; overridden at run time

# fork points: (name, seed, target step)  -- catastrophic / partial / healthy
FORKS = [
    ("seed1_at100k", 1, 100_000),
    ("seed5_at124k", 5, 124_000),
    ("seed11_at356k", 11, 356_000),
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
    bnz = np.load(f"{CHAIN}/seed_{seed}/checkpoint/buffer.npz", allow_pickle=False)
    pos = min(int(k_actual), int(bnz["pos"]))
    updates = max(0, int(k_actual) - LEARNING_STARTS)
    log_alpha = th.tensor(float(np.log(max(alpha_at(seed, k_actual), 1e-6))))
    # counters identical across all four branches -> identical, V-head-ready
    # start. (_dynamics_updates left to CTSAC.load, identical for base/sham.)
    state = {"optimizers": {},
             "counters": {"num_timesteps": int(k_actual),
                          "_n_updates": updates,
                          "_value_updates": updates},
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
            a = bnz[kk].copy()
            a[pos:] = 0
            arrs[kk] = a
        np.savez_compressed(f"{ck}/buffer.npz", pos=np.int64(pos),
                            full=np.bool_(False), **arrs)
        th.save(state, f"{ck}/train_state.pt")
    return k_actual, pos, updates


def main():
    rng = shared_rng_state()
    print(f"fork root={FORK_ROOT}  N_cont={N_CONT}  branches={BRANCHES}")
    lines = []
    for name, seed, K in FORKS:
        k_actual, pos, updates = build_one(name, seed, K, rng)
        total = int(k_actual) + N_CONT
        for br in BRANCHES:
            lines.append(f"fork_{br}  {seed}  {total}  {name}__{br}")
        print(f"{name}: seed{seed} K~{k_actual} prefix={pos} updates={updates} "
              f"-> total={total}")
    print("\n# base cells (mode seed total subdir); the slurm crosses these with 3 RNGs:")
    print("\n".join("    \"" + ln + "\"" for ln in lines))


if __name__ == "__main__":
    main()
