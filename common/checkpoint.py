# common/checkpoint.py
"""
Full training-state checkpointing for off-policy continuous-time algorithms
(CT-SAC / CT-TD3 / CT-DDPG).

The built-in ``algorithm.save`` only persists the actor-critic weights (plus a
learned-dynamics sidecar). That is enough to *evaluate* a trained model but not
to *resume* training: the replay buffer, optimizer moments, entropy temperature,
global timestep counter and RNG state are all lost. On a queue with a hard wall
time (e.g. LS6 ``development`` at 2 h) a >2 h run must therefore checkpoint the
*entire* trainer state near the wall and pick up exactly where it left off in
the next job of a resubmission chain.

A checkpoint is a directory with three parts:

  <ckpt>/model.pth        # actor-critic (+ .dynamics.pth sidecar) via algorithm.save
  <ckpt>/buffer.npz       # replay buffer arrays + ring pointers (pos, full)
  <ckpt>/train_state.pt   # optimizers, log_alpha, counters, RNG, + caller `extra`

Writes are atomic: the checkpoint is built in a sibling temp directory and swapped
into place with ``os.replace`` so an interrupted write never corrupts the last
good checkpoint.
"""

from __future__ import annotations

import os
import shutil
import random
import pathlib
from typing import Any, Dict, Optional

import numpy as np
import torch as th


# Optimizer / scalar attributes that may or may not exist on a given algorithm
# instance (e.g. ``alpha_optimizer`` only exists for auto-entropy, ``value_optimizer``
# only when the model carries a V-head). Each is saved iff present and non-None.
_OPTIMIZER_ATTRS = (
    "actor_optimizer",
    "critic_optimizer",
    "alpha_optimizer",
    "value_optimizer",
    "dynamics_optimizer",
)

_COUNTER_ATTRS = (
    "num_timesteps",
    "_n_updates",
    "_value_updates",
    "_dynamics_updates",
    "alpha",
)

_MODEL_NAME = "model.pth"
_BUFFER_NAME = "buffer.npz"
_STATE_NAME = "train_state.pt"


def _is_complete(ckpt_dir: str) -> bool:
    return all(
        os.path.exists(os.path.join(ckpt_dir, n))
        for n in (_MODEL_NAME, _BUFFER_NAME, _STATE_NAME)
    )


def save_checkpoint(
    algorithm: Any,
    ckpt_dir: str,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Atomically write the full trainer state of ``algorithm`` to ``ckpt_dir``.

    ``extra`` is any JSON/pickle-able dict the caller wants preserved across the
    resume (used here to carry the EvalCallback's best-reward and eval history so
    the eval curve stays whole and ``best_model.pth`` is not clobbered by a worse
    early eval in the next chunk).
    """
    extra = extra or {}
    ckpt_dir = str(ckpt_dir)
    parent = os.path.dirname(os.path.abspath(ckpt_dir)) or "."
    os.makedirs(parent, exist_ok=True)
    tmp_dir = ckpt_dir + ".tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    # 1) actor-critic (+ dynamics sidecar) via the algorithm's own save().
    algorithm.save(os.path.join(tmp_dir, _MODEL_NAME))

    # 2) replay buffer arrays + ring pointers.
    buf = algorithm.replay_buffer
    np.savez(
        os.path.join(tmp_dir, _BUFFER_NAME),
        observations=buf.observations,
        next_observations=buf.next_observations,
        actions=buf.actions,
        rewards=buf.rewards,
        dones=buf.dones,
        t=buf.t,
        next_t=buf.next_t,
        dt=buf.dt,
        pos=np.int64(buf.pos),
        full=np.bool_(buf.full),
    )

    # 3) optimizers, scalars, entropy temperature, RNG, and caller extra.
    state: Dict[str, Any] = {"optimizers": {}, "counters": {}}
    for name in _OPTIMIZER_ATTRS:
        opt = getattr(algorithm, name, None)
        if opt is not None:
            state["optimizers"][name] = opt.state_dict()
    for name in _COUNTER_ATTRS:
        if hasattr(algorithm, name):
            state["counters"][name] = getattr(algorithm, name)

    log_alpha = getattr(algorithm, "log_alpha", None)
    if log_alpha is not None:
        state["log_alpha"] = log_alpha.detach().cpu()

    state["rng"] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": th.get_rng_state(),
        "torch_cuda": (
            th.cuda.get_rng_state_all() if th.cuda.is_available() else None
        ),
    }
    dynamics_rng = getattr(algorithm, "_dynamics_sample_rng", None)
    if isinstance(dynamics_rng, np.random.Generator):
        state["dynamics_sample_rng_state"] = dynamics_rng.bit_generator.state
    state["extra"] = extra
    th.save(state, os.path.join(tmp_dir, _STATE_NAME))

    # Atomic swap: keep the previous good checkpoint until the new one is whole.
    old_dir = ckpt_dir + ".old"
    if os.path.exists(ckpt_dir):
        if os.path.exists(old_dir):
            shutil.rmtree(old_dir)
        os.replace(ckpt_dir, old_dir)
    os.replace(tmp_dir, ckpt_dir)
    if os.path.exists(old_dir):
        shutil.rmtree(old_dir)
    return ckpt_dir


def load_checkpoint(
    algorithm: Any,
    ckpt_dir: str,
    strict: bool = True,
) -> Dict[str, Any]:
    """Restore the full trainer state saved by :func:`save_checkpoint` into
    ``algorithm`` in place and return the caller's ``extra`` dict.

    Sets ``algorithm._resumed_from_checkpoint = True`` so ``_setup_learn`` keeps
    the restored ``num_timesteps`` instead of zeroing it.
    """
    ckpt_dir = str(ckpt_dir)
    if not _is_complete(ckpt_dir):
        raise FileNotFoundError(f"Incomplete or missing checkpoint at {ckpt_dir}")

    # 1) actor-critic weights (+ dynamics sidecar).
    algorithm.load(os.path.join(ckpt_dir, _MODEL_NAME), strict=strict)

    # 2) replay buffer.
    buf = algorithm.replay_buffer
    with np.load(os.path.join(ckpt_dir, _BUFFER_NAME), allow_pickle=False) as data:
        for name in (
            "observations",
            "next_observations",
            "actions",
            "rewards",
            "dones",
            "t",
            "next_t",
            "dt",
        ):
            arr = data[name]
            getattr(buf, name)[...] = arr
        buf.pos = int(data["pos"])
        buf.full = bool(data["full"])

    # 3) optimizers, scalars, entropy temperature, RNG.
    state = th.load(
        os.path.join(ckpt_dir, _STATE_NAME),
        map_location=algorithm.device,
        weights_only=False,
    )

    if "log_alpha" in state and getattr(algorithm, "log_alpha", None) is not None:
        with th.no_grad():
            algorithm.log_alpha.data.copy_(state["log_alpha"].to(algorithm.device))

    for name, opt_state in state.get("optimizers", {}).items():
        opt = getattr(algorithm, name, None)
        if opt is not None:
            opt.load_state_dict(opt_state)

    for name, value in state.get("counters", {}).items():
        setattr(algorithm, name, value)

    rng = state.get("rng", {})
    if "python" in rng:
        random.setstate(rng["python"])
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    if "torch" in rng:
        # RNG state must live on CPU as a ByteTensor.
        th.set_rng_state(rng["torch"].cpu() if hasattr(rng["torch"], "cpu") else rng["torch"])
    if rng.get("torch_cuda") is not None and th.cuda.is_available():
        try:
            th.cuda.set_rng_state_all(rng["torch_cuda"])
        except Exception:
            pass
    if "dynamics_sample_rng_state" in state and hasattr(
        algorithm, "_dynamics_sample_rng"
    ):
        dynamics_rng = np.random.default_rng()
        dynamics_rng.bit_generator.state = state["dynamics_sample_rng_state"]
        algorithm._dynamics_sample_rng = dynamics_rng

    algorithm._resumed_from_checkpoint = True
    return state.get("extra", {})
