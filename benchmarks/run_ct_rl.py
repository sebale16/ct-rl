# experiments/run_ct_rl.py

from __future__ import annotations

import os
import random
from datetime import datetime
from functools import partial
import argparse
from pathlib import Path

import numpy as np
import torch as th

from environment.dmc import DMCContinuousEnv
from environment.monitor import Monitor
from environment.vec_env import VecContinuousEnv
from environment.trading_env import TradingContinuousEnv
from environment.base import ContinuousEnv
from algorithms.ct_sac import CTSAC
from algorithms.ct_td3 import CTTD3
from algorithms.ct_ddpg import CTDDPG
from algorithms.q_learning import qLearning
from algorithms.cpg import CPG
from algorithms.cppo import CPPO
from models import ActorQCriticModel, CoupledVqModel, ActorVCriticModel
from models.noise import (
    ActionNoise,
    GaussianActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    WallClockCheckpointCallback,
)
from common.checkpoint import load_checkpoint

from common.logger import configure
from common.utils import (
    load_ct_hyperparams_from_table,
    build_save_path,
    normalize_eval_range,
    get_eval_episode_count,
    set_seed,
)
from data.trading.config import TRAIN_NPZ, EVAL_NPZ, GROUPS


def _create_action_noise_from_hyperparams(
    env: DMCContinuousEnv, algo_kwargs: dict
) -> ActionNoise | None:
    noise_type = algo_kwargs.pop("noise_type", None)
    noise_params_str = algo_kwargs.pop("noise_params", None)
    if not noise_type:
        return None

    params = {}
    if noise_params_str:
        for part in noise_params_str.split(";"):
            key, val = part.strip().split("=")
            params[key] = float(val)

    action_dim = int(np.prod(env.action_space.shape))
    mean = params.get("mean", 0.0) * np.ones(action_dim)

    if noise_type == "gaussian" or noise_type == "normal":
        return GaussianActionNoise(
            mean=mean, sigma=params.get("sigma", 0.1) * np.ones(action_dim)
        )
    elif noise_type == "ornstein":
        dt = params.get("dt", env.dt)
        return OrnsteinUhlenbeckActionNoise(
            mean=mean,
            sigma=params.get("sigma", 0.1) * np.ones(action_dim),
            theta=params.get("theta", 0.15),
            dt=dt,
        )
    raise ValueError(f"Unknown noise type: {noise_type}")


def make_ct_env(
    env_id: str,
    seed: int,
    env_kwargs: dict,
    log_dir: Path | str | None = None,
    npz_path: str | None = None,
) -> ContinuousEnv:
    """
    Build a single continuous-time environment instance.
    """
    if env_id.startswith("trading"):
        env = TradingContinuousEnv(
            npz_path=npz_path,
            seed=seed,
            **env_kwargs,
        )
    else:
        if "-" not in env_id:
            raise ValueError("env-id must be 'domain-task', e.g. 'cheetah-run'.")
        domain_name, task_name = env_id.split("-", 1)
        env = DMCContinuousEnv(
            domain_name=domain_name,
            task_name=task_name,
            seed=seed,
            **env_kwargs,
        )

    # Continuous-time Monitor wrapper
    if log_dir:
        env = Monitor(env)
    return env


def _select_structured_dof_layout(env, obs_dim: int, layout_cls):
    """Choose a mechanics-aware layout through vector/wrapper layers.

    Raw cartpole has information that the generic ``[q; qdot]`` layout cannot
    infer from tensor shapes: its cart translation is invariant and its single
    actuator drives only the slider. Other raw hinge/slide domains keep the
    generic fallback until they have an equally explicit layout.

    ``layout_cls`` is injected to keep the optional model-based import local and
    make this selection rule independently testable.
    """
    current = env.envs[0] if hasattr(env, "envs") and env.envs else env
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "raw_state_obs"):
            break
        current = getattr(current, "env", None)

    if current is None or not getattr(current, "raw_state_obs", False):
        return None
    if getattr(current, "domain_name", None) == "cartpole":
        return layout_cls.cartpole()
    if getattr(current, "domain_name", None) == "acrobot":
        return layout_cls.acrobot()
    return layout_cls.raw_state(nv=int(obs_dim) // 2)


def _pop_structured_model_kwargs(algo_kwargs: dict) -> dict:
    """Move dynamics-model regularizers out of the CTSAC kwargs namespace."""
    return {
        "mass_logdet_reg": float(
            str(algo_kwargs.pop("dynamics_mass_logdet_reg", "") or "0").strip()
        ),
        "mass_condition_reg": float(
            str(algo_kwargs.pop("dynamics_mass_condition_reg", "") or "0").strip()
        ),
        "mass_condition_limit": float(
            str(
                algo_kwargs.pop("dynamics_mass_condition_limit", "") or "1000"
            ).strip()
        ),
    }


def run_algorithm(
    algo: str,
    env_id: str,
    mode: str,
    eval_mode: str | None,
    seed: int,
    hyperparams_dir: str,
    log_root_dir: str,
    save_root_dir: str,
    total_timesteps_override: int | None,
    desc: str,
    n_eval_episodes: int = 10,
    eval_range: str | None = None,
    resume: bool = False,
    max_seconds: float | None = None,
    checkpoint_dir: str | None = None,
    run_id: str | None = None,
    continuation_rng_seed: int | None = None,
    init_weights: str | None = None,
    best_model_gate: str | None = None,
    eval_hanging: bool = False,
) -> bool:
    """
    Runs a single RL algorithm experiment.

    Returns True if training reached ``total_timesteps`` (finished), False if it
    paused for a wall-time/signal checkpoint and should be resumed. When
    ``max_seconds`` is set, a resumable checkpoint (model + replay buffer +
    optimizers + counters + RNG) is written as the job approaches that budget and
    the loop exits cleanly; passing ``resume=True`` on a later run reloads the
    latest checkpoint and continues from the exact timestep.
    """
    # Root every fresh-run object -- including structured dynamics constructed
    # before the algorithm -- in the requested seed.  BaseAlgorithm separately
    # restarts the policy/runtime stream after model construction.
    set_seed(int(seed))

    print(
        f"\n{'='*50}\nRunning: {algo} on {env_id} (mode: {mode}, eval_mode: {eval_mode or mode}, seed: {seed})\n{'='*50}"
    )

    (
        total_timesteps,
        env_kwargs,
        model_kwargs,
        algo_kwargs,
        log_kwargs,
    ) = load_ct_hyperparams_from_table(
        algo=algo,
        env_id=env_id,
        mode=mode,
        hyperparams_dir=hyperparams_dir,
    )

    if eval_mode:
        _, eval_env_kwargs, _, _, _ = load_ct_hyperparams_from_table(
            algo=algo,
            env_id=env_id,
            mode=eval_mode,
            hyperparams_dir=hyperparams_dir,
        )
    else:
        eval_env_kwargs = env_kwargs.copy()

    if env_id.startswith("trading") and eval_range is not None:
        eval_env_kwargs = eval_env_kwargs.copy()
        eval_range = normalize_eval_range(eval_range)
        eval_env_kwargs["eval_range"] = eval_range
        eval_env_kwargs["eval_cycle_tickers"] = True

        # Approximate number of episodes (2-weeks trading periods)
        n_time_windows = get_eval_episode_count(eval_range)
        n_ticker_cycles = max(len(v) for v in GROUPS.values())
        n_eval_episodes = n_time_windows * n_ticker_cycles

    if total_timesteps_override is not None:
        total_timesteps = total_timesteps_override

    # Build logs and saved_models save paths
    log_dir = build_save_path(
        log_root_dir, algo, env_id, mode, seed, env_kwargs, desc, run_id=run_id
    )
    save_dir = build_save_path(
        save_root_dir, algo, env_id, mode, seed, env_kwargs, desc, run_id=run_id
    )

    # Resumable-checkpoint location and whether a complete one exists to resume.
    from common.checkpoint import _is_complete

    ckpt_dir = checkpoint_dir or str(save_dir / "checkpoint")
    resume_active = bool(resume) and _is_complete(ckpt_dir)

    # When resuming, append to the existing logs so the learning curve stays
    # continuous across the resubmission chain instead of being truncated.
    configure(
        folder=str(log_dir),
        output_formats=["csv", "json", "tensorboard", "log"],
        append=resume_active,
    )

    # Create (vectorized) train environments
    n_envs = int(env_kwargs.pop("n_envs", 1))

    make_train_env_fn = partial(
        make_ct_env,
        env_id=env_id,
        env_kwargs=env_kwargs,
        log_dir=log_dir / "train",
        npz_path=TRAIN_NPZ,
    )
    train_env = (
        VecContinuousEnv(
            [lambda i=i: make_train_env_fn(seed=seed + i) for i in range(n_envs)]
        )
        if n_envs > 1
        else make_train_env_fn(seed=seed)
    )

    # Create (vectorized) evaluation environment
    eval_n_envs = int(eval_env_kwargs.pop("n_envs", 1))
    make_eval_env_fn = partial(
        make_ct_env,
        env_id=env_id,
        env_kwargs=eval_env_kwargs,
        npz_path=EVAL_NPZ,
    )
    eval_env = (
        VecContinuousEnv(
            [
                lambda i=i: make_eval_env_fn(seed=seed + 1000 + i)
                for i in range(eval_n_envs)
            ]
        )
        if eval_n_envs > 1
        else make_eval_env_fn(seed=seed + 1000)
    )

    # Create algorithm
    algo_map = {
        "ct_sac": CTSAC,
        "ct_td3": CTTD3,
        "ct_ddpg": CTDDPG,
        "q_learning": qLearning,
        "cpg": CPG,
        "cppo": CPPO,
    }
    if algo not in algo_map:
        raise ValueError(f"Unknown algorithm: {algo}")

    AlgoClass = algo_map[algo]
    if AlgoClass is None:
        print(f"Algorithm '{algo}' is not implemented. Skipping.")
        return

    # Handle model selection and action noise
    if algo == "q_learning":
        model_class = CoupledVqModel
    elif algo == "cppo" or algo == "cpg":
        #  CPG and CPPO is an on-policy algorithm
        model_class = ActorVCriticModel
    else:
        # CT-SAC, CT-TD3, and CT-DDPG
        model_class = ActorQCriticModel
        # For DDPG/TD3, action_noise is for exploration
        if algo in ["ct_ddpg", "ct_td3"]:
            algo_kwargs["action_noise"] = _create_action_noise_from_hyperparams(
                train_env, algo_kwargs
            )

    # Optional: model-based generator (port-Hamiltonian dynamics model) for CT-SAC.
    if algo == "ct_sac" and str(
        algo_kwargs.get("use_model_based_q", "")
    ).strip().lower() in ("1", "true", "yes"):
        from models.port_hamiltonian import DOFLayout, PortHamiltonianModel

        source = str(algo_kwargs.get("dynamics_source", "mujoco"))
        intensity = float(algo_kwargs.get("human_input_intensity", 0.0) or 0.0)
        contact_force = int(
            str(algo_kwargs.pop("dynamics_contact_force", "") or "").strip() or 0
        )
        # Structured-model-only regularizers live in the benchmark's algo_*
        # namespace for configuration, but are consumed by the dynamics model
        # constructor rather than CTSAC itself.
        structured_model_kwargs = _pop_structured_model_kwargs(algo_kwargs)
        obs_dim = int(np.prod(train_env.observation_space.shape))
        act_dim = int(np.prod(train_env.action_space.shape))
        # Raw cartpole gets its known invariances and sparse actuation. Other
        # raw hinge/slide domains keep the generic layout; non-raw structured
        # models retain their existing domain default.
        dof_layout = _select_structured_dof_layout(
            train_env, obs_dim, DOFLayout
        )
        if source == "mujoco":
            base_env = train_env
            while not hasattr(base_env, "dynamics_terms") and hasattr(base_env, "env"):
                base_env = base_env.env
            if not hasattr(base_env, "dynamics_terms"):
                raise ValueError(
                    "dynamics_source='mujoco' requires a single DMC env exposing "
                    "dynamics_terms()."
                )
            algo_kwargs["dynamics_model"] = PortHamiltonianModel(
                obs_dim,
                act_dim,
                mode="mujoco",
                drift_fn=base_env.dynamics_terms,
                human_input_intensity=intensity,
            )
        elif source == "phast":
            # Learned port-Hamiltonian; CT-SAC fits it online from the replay buffer
            # (warmup, then it takes over from the finite-difference target).
            algo_kwargs["dynamics_model"] = PortHamiltonianModel(
                obs_dim,
                act_dim,
                mode="phast",
                human_input_intensity=intensity,
            )
        elif source == "structured":
            # Structured port-Hamiltonian (DeLaN core): learned SPD mass M(q) and
            # potential V(q) generate the Coriolis terms; canonicalizer p = M(q)qd;
            # constant diagonal damping on momentum; optional explicit contact
            # port (dynamics_contact_force = number of learned contact points,
            # which also makes M translation-invariant). Raw cartpole uses its
            # mechanics-aware layout; other raw-state envs use the generic
            # layout, and the non-raw default remains cheetah's.
            algo_kwargs["dynamics_model"] = PortHamiltonianModel(
                obs_dim,
                act_dim,
                mode="structured",
                human_input_intensity=intensity,
                contact_force=contact_force,
                dof_layout=dof_layout,
                **structured_model_kwargs,
            )
        else:
            raise ValueError(f"Unknown dynamics_source '{source}'.")

    # model_kwargs from CSV: q_net_arch, pi_net_arch, n_critics, activation_fn, ...
    # algo_kwargs from CSV: learning_rate, buffer_size, batch_size, gamma, tau, ...
    algorithm = AlgoClass(
        env=train_env,
        model=model_class,
        model_kwargs=model_kwargs,
        seed=seed,
        **algo_kwargs,
    )
    # Learned-dynamics replay sampling has its own stream.  Without this,
    # structured fitting advances global NumPy state and changes later critic
    # minibatches relative to MF/oracle arms for reasons unrelated to data.
    if hasattr(algorithm, "_dynamics_sample_rng"):
        algorithm._dynamics_sample_rng = np.random.default_rng(int(seed) + 999983)

    # Warm start (fork): graft a previously trained policy/critic onto a fresh
    # trainer.  ``load`` restores only the model weights (actor + Q-critics +
    # their targets); alpha, optimizers, and the replay buffer stay fresh from
    # this run's hyperparameters, so a fork can continue the loaded policy under
    # a different alpha/lr/tau regime.  Mutually exclusive with --resume.
    if init_weights:
        if resume_active:
            raise ValueError("--init_weights cannot be combined with --resume")
        algorithm.load(init_weights, strict=True)
        print(f"[fork] warm-started weights from {init_weights}", flush=True)

    # Setup callbacks
    save_freq = log_kwargs.get("save_freq", 100000)
    eval_freq = log_kwargs.get("eval_freq", 10000)
    log_interval = log_kwargs.get("interval", 1000)

    # Optional best-model gate: "occupancy_key:min_occupancy:min_reward"
    # (e.g. "acrobot_hold:0.05:400") -> best_model only updates on evals whose
    # dt-weighted mean info[key] and mean reward both clear the floors.
    gate_key, gate_occ, gate_rew = None, 0.0, float("-inf")
    if best_model_gate:
        parts = best_model_gate.split(":")
        if len(parts) != 3:
            raise ValueError(
                "--best_model_gate must be 'key:min_occupancy:min_reward'"
            )
        gate_key = parts[0].strip()
        gate_occ = float(parts[1])
        gate_rew = float(parts[2])
        print(
            f"[gate] best_model gated on {gate_key}>={gate_occ} and "
            f"eval_reward>={gate_rew}",
            flush=True,
        )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        reset_seed=seed + 1000,
        best_model_save_path=str(save_dir / "best_model"),
        log_path=str(log_dir / "eval"),
        verbose=1,
        gate_occupancy_key=gate_key,
        gate_min_occupancy=gate_occ,
        gate_min_reward=gate_rew,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(save_dir),
        name_prefix=f"{algo}_{env_id}_{mode}",
        verbose=1,
    )

    # Resume: reload the full trainer state (buffer, optimizers, counters, RNG)
    # and restore the EvalCallback's cumulative history so the eval curve stays
    # whole and best_model.pth is not clobbered by a worse early eval.
    if resume_active:
        extra = load_checkpoint(algorithm, ckpt_dir)
        eval_state = extra.get("eval", {})
        if eval_state:
            eval_callback.best_mean_reward = eval_state.get(
                "best_mean_reward", eval_callback.best_mean_reward
            )
            eval_callback.last_mean_reward = eval_state.get(
                "last_mean_reward", eval_callback.last_mean_reward
            )
            eval_callback._last_eval_timesteps = eval_state.get(
                "last_eval_timesteps", 0
            )
            eval_callback.evaluations_timesteps = eval_state.get(
                "evaluations_timesteps", []
            )
            eval_callback.evaluations_results = eval_state.get(
                "evaluations_results", []
            )
            eval_callback.evaluations_lengths = eval_state.get(
                "evaluations_lengths", []
            )
        print(
            f"[resume] loaded checkpoint from {ckpt_dir}: "
            f"num_timesteps={algorithm.num_timesteps}, "
            f"buffer_size={algorithm.replay_buffer.size()}, "
            f"eval_best={eval_callback.best_mean_reward:.3f}",
            flush=True,
        )

    # Paired-continuation reseeding (optional). When --continuation_rng_seed is
    # set on a resumed run, re-seed the global python/numpy/torch RNGs to a value
    # that is IDENTICAL across target treatments -- so their critic/actor
    # minibatches and exploration noise are paired -- but differs across
    # replicate seeds, and give the learned-dynamics fit its OWN isolated
    # sampling stream so a treatment that fits dynamics never advances the shared
    # minibatch stream. The global reseed is applied ONCE (first chunk); a
    # sibling marker records that so later chunks continue the stream instead of
    # restarting it, while the dynamics-fit stream is re-isolated every chunk.
    if resume_active and continuation_rng_seed is not None:
        algorithm._dynamics_sample_rng = np.random.default_rng(
            int(continuation_rng_seed) + 999983
        )
        marker = str(ckpt_dir).rstrip("/") + ".continuation_seeded"
        if not os.path.exists(marker):
            random.seed(int(continuation_rng_seed))
            np.random.seed(int(continuation_rng_seed))
            th.manual_seed(int(continuation_rng_seed))
            with open(marker, "w") as f:
                f.write(f"continuation_rng_seed={int(continuation_rng_seed)}\n")
            print(
                f"[continuation] one-time global reseed to {continuation_rng_seed}; "
                f"dynamics-fit sampling isolated on a dedicated stream.",
                flush=True,
            )
        else:
            print(
                "[continuation] resume chunk: kept checkpoint RNG; "
                "dynamics-fit sampling re-isolated on its dedicated stream.",
                flush=True,
            )

    callbacks = [checkpoint_callback, eval_callback]

    # Optional second eval track from the canonical hanging start, run
    # alongside the (uniform-start) primary eval.  For acrobot v4.1/v5 the
    # training resets are uniform random, so the primary eval and its
    # best_model measure capture-from-anywhere; this hanging eval reports the
    # true swing-up-from-down task and saves its own gated best_model_hanging/,
    # without disturbing the primary best_model.
    if eval_hanging:
        hanging_eval_env_kwargs = dict(eval_env_kwargs)
        hanging_task_kwargs = dict(hanging_eval_env_kwargs.get("task_kwargs", {}))
        hanging_task_kwargs["uniform_start"] = False
        hanging_eval_env_kwargs["task_kwargs"] = hanging_task_kwargs
        make_hanging_env_fn = partial(
            make_ct_env,
            env_id=env_id,
            env_kwargs=hanging_eval_env_kwargs,
            npz_path=EVAL_NPZ,
        )
        hanging_eval_env = (
            VecContinuousEnv(
                [
                    lambda i=i: make_hanging_env_fn(seed=seed + 2000 + i)
                    for i in range(eval_n_envs)
                ]
            )
            if eval_n_envs > 1
            else make_hanging_env_fn(seed=seed + 2000)
        )
        hanging_eval_callback = EvalCallback(
            eval_env=hanging_eval_env,
            eval_freq=max(eval_freq // n_envs, 1),
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            reset_seed=seed + 2000,
            best_model_save_path=str(save_dir / "best_model_hanging"),
            log_path=str(log_dir / "eval_hanging"),
            verbose=1,
            gate_occupancy_key=gate_key,
            gate_min_occupancy=gate_occ,
            gate_min_reward=gate_rew,
            log_prefix="eval_hanging",
        )
        callbacks.append(hanging_eval_callback)

    # Wall-clock checkpoint: near the time budget, write a resumable checkpoint
    # and stop cleanly so the resubmission chain can continue.
    wall_cb = None
    if max_seconds is not None and max_seconds > 0:
        def _collect_extra():
            return {
                "eval": {
                    "best_mean_reward": float(eval_callback.best_mean_reward),
                    "last_mean_reward": float(eval_callback.last_mean_reward),
                    "last_eval_timesteps": int(eval_callback._last_eval_timesteps),
                    "evaluations_timesteps": eval_callback.evaluations_timesteps,
                    "evaluations_results": eval_callback.evaluations_results,
                    "evaluations_lengths": eval_callback.evaluations_lengths,
                }
            }

        wall_cb = WallClockCheckpointCallback(
            ckpt_dir=ckpt_dir,
            max_seconds=max_seconds,
            extra_state_fn=_collect_extra,
            verbose=1,
        )
        callbacks.append(wall_cb)

    callback = CallbackList(callbacks)

    # Training
    env_kwargs["n_envs"] = n_envs  # Put back n_envs only for printing
    env_kwargs["eval_n_envs"] = eval_n_envs
    print(
        f"\n[{algo.upper()}] env={env_id} mode={mode}\n"
        f"total_timesteps={total_timesteps}; n_eval_episodes={n_eval_episodes}\n\n"
        f"env_kwargs={env_kwargs}\n\n"
        f"eval_env_kwargs={eval_env_kwargs}\n\n"
        f"model_kwargs={model_kwargs}\n\n"
        f"algo_kwargs={algo_kwargs}\n\n"
        f"log_kwargs={log_kwargs}\n\n"
    )
    algorithm.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=log_interval,
    )

    # Distinguish "reached total_timesteps" from "paused for a wall-time
    # checkpoint". The runner writes a marker file the batch script reads to
    # decide whether to resubmit the next chunk of the chain.
    paused = bool(wall_cb is not None and wall_cb.stopped)
    finished = (algorithm.num_timesteps >= total_timesteps) and not paused

    if max_seconds is not None:
        os.makedirs(ckpt_dir, exist_ok=True)
        marker = os.path.join(ckpt_dir, "STATUS")
        with open(marker, "w") as f:
            f.write(
                ("DONE" if finished else "INCOMPLETE")
                + f" num_timesteps={algorithm.num_timesteps}"
                + f" total_timesteps={total_timesteps}\n"
            )

    if finished:
        final_model_path = save_dir / "final_model.pth"
        algorithm.save(final_model_path)
        print(
            f"Training finished: reached {algorithm.num_timesteps}/{total_timesteps} "
            f"steps; saved exact endpoint to {final_model_path}.",
            flush=True,
        )
    else:
        print(
            f"Training paused at {algorithm.num_timesteps}/{total_timesteps} steps "
            f"(will resume from checkpoint).",
            flush=True,
        )
    return finished


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algos",
        type=str,
        default="ct_sac",
        help="Comma-separated list of algorithms to run (e.g., 'ct_sac, cppo'). "
        "Choices: ct_sac, ct_td3, ct_ddpg, q_learning, cpg, cppo.",
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default="cheetah-run",
        help="env e.g. 'cheetah-run', 'walker-run' or 'trading'.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="top",
        help="Mode key matching the CSV row",
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        default=None,
        help="Evaluation mode key for EvalCallback. Defaults to --mode if not set.",
    )
    parser.add_argument(
        "--eval_hanging",
        action="store_true",
        help="Add a second eval track from the canonical hanging start "
        "(saves best_model_hanging/, logs eval_hanging/*) alongside the "
        "uniform-start primary eval. For acrobot v4.1/v5.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--hyperparams_dir",
        type=str,
        default="benchmarks/hyperparams",
        help="Directory containing per-algo hyperparam CSVs.",
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default="logs",
        help="Log root relative to project dir.",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="saved_models",
        help="Model save root relative to project dir.",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=None,
        help="Override total_timesteps from CSV.",
    )
    parser.add_argument(
        "--desc",
        type=str,
        default="",
        help="Optional description to append to run directory name.",
    )
    parser.add_argument(
        "--n_eval_episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate during EvalCallback.",
    )
    parser.add_argument(
        "--eval_range",
        type=str,
        default="Q3_2025",
        help="Evaluation quarters for the trading environment",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint under the checkpoint dir if present.",
    )
    parser.add_argument(
        "--max_seconds",
        type=float,
        default=None,
        help="Wall-clock budget (seconds). When set, a resumable checkpoint is "
        "written as this budget is approached and training exits cleanly so a "
        "resubmission chain can continue. Omit for a normal single run.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Override the checkpoint directory (default: <save_dir>/checkpoint).",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Fixed run identifier used in the run directory name instead of a "
        "wall-clock timestamp. Give all chunks of a resubmission chain the same "
        "run_id so they share one log/save/checkpoint directory.",
    )
    parser.add_argument(
        "--init_weights",
        type=str,
        default=None,
        help="Warm-start (fork): path to a saved model (e.g. best_model.pth) "
        "whose actor/critic weights are loaded into a fresh trainer before "
        "learning. alpha/optimizers/replay buffer stay fresh, so the loaded "
        "policy continues under this run's hyperparameters. Not for --resume.",
    )
    parser.add_argument(
        "--best_model_gate",
        type=str,
        default=None,
        help="Gate best_model saving on 'key:min_occupancy:min_reward' "
        "(e.g. 'acrobot_hold:0.05:400'): best_model only updates on evals whose "
        "dt-weighted mean info[key] and mean reward both clear the floors.",
    )
    parser.add_argument(
        "--continuation_rng_seed",
        type=int,
        default=None,
        help="Paired-continuation replicate seed. On a resumed run, re-seeds the "
        "global RNGs identically across target treatments (paired minibatches / "
        "exploration) and isolates the learned-dynamics fit's sampling. Applied "
        "once per chain via a sibling '.continuation_seeded' marker.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    algos_to_run = [algo.strip() for algo in args.algos.split(",") if algo.strip()]
    env_id = args.env_id
    if env_id.startswith("trading"):
        eval_range = args.eval_range
    else:
        eval_range = None

    # Exit code convention (used by the resubmission chain): 0 = all runs
    # finished, 42 = at least one run paused for a wall-time checkpoint and
    # should be resumed.
    all_finished = True
    for algo in algos_to_run:
        try:
            finished = run_algorithm(
                algo=algo,
                env_id=env_id,
                mode=args.mode,
                eval_mode=args.eval_mode,
                seed=args.seed,
                hyperparams_dir=args.hyperparams_dir,
                log_root_dir=args.log_root,
                save_root_dir=args.save_root,
                total_timesteps_override=args.total_timesteps,
                desc=args.desc,
                n_eval_episodes=args.n_eval_episodes,
                eval_range=eval_range,
                resume=args.resume,
                max_seconds=args.max_seconds,
                checkpoint_dir=args.checkpoint_dir,
                run_id=args.run_id,
                continuation_rng_seed=args.continuation_rng_seed,
                init_weights=args.init_weights,
                best_model_gate=args.best_model_gate,
                eval_hanging=args.eval_hanging,
            )
            all_finished = all_finished and bool(finished)
        except (FileNotFoundError, KeyError) as e:
            print(f"\nCould not run {algo} due to a configuration error: {e}\n")

    if args.max_seconds is not None and not all_finished:
        import sys

        sys.exit(42)


if __name__ == "__main__":
    main()
