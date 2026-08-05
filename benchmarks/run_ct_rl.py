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
    CurriculumFractionCallback,
    EvalCallback,
    MasteryCurriculumCallback,
    WallClockCheckpointCallback,
)
from common.checkpoint import load_checkpoint
from evaluations.sustained_capture import (
    curriculum_mastery_capture_spec_for,
    strict_capture_spec_for,
)

from common.logger import configure
from common.utils import (
    load_ct_hyperparams_from_table,
    build_save_path,
    normalize_eval_range,
    get_eval_episode_count,
    set_seed,
)
from data.trading.config import TRAIN_NPZ, EVAL_NPZ, GROUPS
from environment.tip_curriculum import (
    FRACTION_CURRICULUM_ENV_IDS,
    PERFORMANCE_CURRICULUM_ENV_IDS,
)


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
        env = Monitor(env, info_keywords=rollout_info_keys(env_id))
    return env


# Per-step diagnostics logged from the behavior policy's own rollouts, by task.
# The acrobot v6 set factors its velocity cost into the two things it confounds:
# ``energy_norm`` is the swing-up energy budget currently held, which pumping
# raises, and ``coordination_loss`` is where that energy sits between the
# cheapest and dearest directions of the mass metric at the current pose.  A
# stalled run is diagnosable from the pair — no energy going in is a collapsed
# policy, energy going in at coordination_loss near 1 is flailing.
ROLLOUT_INFO_KEYS = {
    "acrobot-swingup-v6": (
        "acrobot_energy_norm",
        "acrobot_kinetic_norm",
        "acrobot_velocity_cost",
        "acrobot_velocity_cost_per_joule",
        "acrobot_coordination_loss",
    ),
}
ROLLOUT_INFO_KEYS["acrobot-swingup-v6-uniform"] = ROLLOUT_INFO_KEYS[
    "acrobot-swingup-v6"
]
ROLLOUT_INFO_KEYS["acrobot-swingup-v6.1"] = ROLLOUT_INFO_KEYS[
    "acrobot-swingup-v6"
]


def rollout_info_keys(env_id: str) -> tuple[str, ...]:
    """Per-step info scalars the Monitor should log for this task."""
    return ROLLOUT_INFO_KEYS.get(env_id, ())


# Fraction of the training budget over which the reset curriculum widens from
# its near-upright band to the full circle; the remaining budget trains on the
# full (uniform) start distribution with the capture value already in place.
CURRICULUM_SPAN_FRAC = 0.5


def _iter_dmc_envs(env):
    """Yield the ``DMCContinuousEnv`` instances inside vector/monitor wrappers."""
    subs = getattr(env, "envs", None)
    if subs:
        for sub in subs:
            yield from _iter_dmc_envs(sub)
        return
    seen = set()
    cur = env
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, DMCContinuousEnv):
            yield cur
            return
        cur = getattr(cur, "env", None)


def _select_structured_dof_layout(env, obs_dim: int, layout_cls):
    """Choose a mechanics-aware layout through vector/wrapper layers.

    Raw CartPole chains and Acrobot have mechanism information that the generic
    ``[q; qdot]`` layout cannot infer from tensor shapes: exact periodicity,
    mass invariances, joint limits, and sparse actuator placement. Other raw
    hinge/slide domains keep the generic fallback until they have an equally
    explicit layout.

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
        num_poles = int(obs_dim) // 2 - 1
        if num_poles == 1:
            # Keep the no-argument call compatible with layout implementations
            # and tests written before serial CartPole chains were supported.
            return layout_cls.cartpole()
        return layout_cls.cartpole(num_poles=num_poles)
    if getattr(current, "domain_name", None) == "acrobot":
        return layout_cls.acrobot()
    return layout_cls.raw_state(nv=int(obs_dim) // 2)


def _pop_structured_model_kwargs(
    algo_kwargs: dict, *, contact_dt_default: float | None = None
) -> dict:
    """Move structured-model controls out of the CTSAC kwargs namespace.

    Blank CSV cells are omitted by ``load_ct_hyperparams_from_table``.  When a
    contact solver is selected without an explicit step, use the environment's
    physics step so the differentiable solve runs at the same resolution as
    the simulator rather than at the irregular control duration.
    """
    model_kwargs = {
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

    contact_solver = str(
        algo_kwargs.pop("dynamics_contact_solver", "") or ""
    ).strip()
    contact_geometry = str(
        algo_kwargs.pop("dynamics_contact_geometry", "") or ""
    ).strip()
    if contact_geometry:
        model_kwargs["contact_geometry"] = contact_geometry
    contact_gate_off = algo_kwargs.pop("dynamics_contact_gate_off", None)
    contact_dt = algo_kwargs.pop("dynamics_contact_dt", None)
    contact_iterations = algo_kwargs.pop("dynamics_contact_iterations", None)
    contact_regularization = algo_kwargs.pop(
        "dynamics_contact_regularization", None
    )
    if contact_gate_off is not None and str(contact_gate_off).strip():
        model_kwargs["contact_gate_off"] = float(contact_gate_off)

    def _optional_contact_parameter(name: str):
        value = str(algo_kwargs.pop(name, "") or "").strip()
        if not value:
            return None, False
        lowered = value.lower()
        if lowered in ("true", "yes"):
            return True, True
        if lowered in ("false", "no", "none"):
            return None, True
        return float(value), True

    # Legacy gate-shaped c0 and the predicted-crossing physical k laws are
    # separate, mutually exclusive experiments. A numeric stiffness is N/m;
    # 'true' selects the model's 1e5 N/m default. A numeric stiffness ratio
    # selects the version-4 depth curve; tangent compliance is its independent
    # positive velocity-level QP softness in 1/kg.
    contact_compliance, has_compliance = _optional_contact_parameter(
        "dynamics_contact_compliance"
    )
    contact_stiffness, has_stiffness = _optional_contact_parameter(
        "dynamics_contact_stiffness"
    )
    contact_attenuation = str(
        algo_kwargs.pop("dynamics_contact_attenuation", "") or ""
    ).strip()
    contact_stiffness_ratio_value = algo_kwargs.pop(
        "dynamics_contact_stiffness_ratio", ""
    )
    contact_stiffness_ratio = (
        ""
        if contact_stiffness_ratio_value is None
        else str(contact_stiffness_ratio_value).strip()
    )
    contact_tangent_compliance_value = algo_kwargs.pop(
        "dynamics_contact_tangent_compliance", ""
    )
    contact_tangent_compliance = (
        ""
        if contact_tangent_compliance_value is None
        else str(contact_tangent_compliance_value).strip()
    )
    if has_compliance:
        model_kwargs["contact_compliance"] = contact_compliance
    if has_stiffness:
        model_kwargs["contact_stiffness"] = contact_stiffness
    if contact_attenuation:
        model_kwargs["contact_attenuation"] = float(contact_attenuation)
    if contact_stiffness_ratio:
        if contact_stiffness_ratio.lower() in (
            "true", "yes", "false", "no", "none"
        ):
            raise ValueError("dynamics_contact_stiffness_ratio must be numeric")
        model_kwargs["contact_stiffness_ratio"] = float(contact_stiffness_ratio)
    if contact_tangent_compliance:
        if contact_tangent_compliance.lower() in (
            "true", "yes", "false", "no", "none"
        ):
            raise ValueError("dynamics_contact_tangent_compliance must be numeric")
        model_kwargs["contact_tangent_compliance"] = float(
            contact_tangent_compliance
        )

    if contact_solver:
        model_kwargs["contact_solver"] = contact_solver
        if contact_dt is None:
            contact_dt = contact_dt_default
    if contact_dt is not None:
        model_kwargs["contact_dt"] = float(contact_dt)
    if contact_iterations is not None:
        model_kwargs["contact_iterations"] = int(contact_iterations)
    if contact_regularization is not None:
        model_kwargs["contact_regularization"] = float(contact_regularization)

    return model_kwargs


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
    curriculum_steps: int | None = None,
    curriculum_success_threshold: float = 0.8,
    curriculum_consecutive_evals: int = 1,
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

    # Keep the historical fraction curricula and the new mastery curricula
    # separate. Both use a fixed final-task distribution for checkpoint
    # selection; mastery tasks additionally get a changing-stage probe env.
    is_fraction_curriculum = env_id in FRACTION_CURRICULUM_ENV_IDS
    is_performance_curriculum = env_id in PERFORMANCE_CURRICULUM_ENV_IDS
    is_curriculum_env = is_fraction_curriculum or is_performance_curriculum
    curriculum_probe_env_kwargs = None
    if is_curriculum_env:
        if is_performance_curriculum:
            curriculum_probe_env_kwargs = dict(eval_env_kwargs)
            probe_task_kwargs = dict(
                curriculum_probe_env_kwargs.get("task_kwargs", {})
            )
            probe_task_kwargs["curriculum"] = True
            curriculum_probe_env_kwargs["task_kwargs"] = probe_task_kwargs
        _eval_task_kwargs = dict(eval_env_kwargs.get("task_kwargs", {}))
        _eval_task_kwargs["curriculum"] = False
        eval_env_kwargs = dict(eval_env_kwargs)
        eval_env_kwargs["task_kwargs"] = _eval_task_kwargs
    if is_fraction_curriculum:
        curr_total = (
            int(curriculum_steps)
            if curriculum_steps is not None
            else int(CURRICULUM_SPAN_FRAC * total_timesteps)
        )

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

    curriculum_probe_env = None
    if curriculum_probe_env_kwargs is not None:
        probe_kwargs = dict(curriculum_probe_env_kwargs)
        probe_n_envs = int(probe_kwargs.pop("n_envs", eval_n_envs))
        make_probe_env_fn = partial(
            make_ct_env,
            env_id=env_id,
            env_kwargs=probe_kwargs,
            npz_path=EVAL_NPZ,
        )
        curriculum_probe_env = (
            VecContinuousEnv(
                [
                    lambda i=i: make_probe_env_fn(seed=seed + 3000 + i)
                    for i in range(probe_n_envs)
                ]
            )
            if probe_n_envs > 1
            else make_probe_env_fn(seed=seed + 3000)
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
        # Structured-model-only regularizers and contact-solver controls live
        # in the benchmark's algo_* namespace, but are consumed by the
        # dynamics-model constructor rather than CTSAC itself.
        structured_model_kwargs = _pop_structured_model_kwargs(
            algo_kwargs,
            contact_dt_default=env_kwargs.get("physics_dt"),
        )
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
            # constant diagonal damping on momentum; optional learned or exact
            # contact geometry (dynamics_contact_force = number of points), with
            # a historical penalty law, the smooth-gate constraint law, or the
            # predicted-crossing physical-stiffness constraint prototypes. Raw
            # cartpole uses its mechanics-aware layout; other raw-state envs
            # use the generic layout, and the non-raw default remains cheetah's.
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
    capture_spec = strict_capture_spec_for(algorithm=algo, env_id=env_id)
    curriculum_capture_spec = (
        curriculum_mastery_capture_spec_for(
            algorithm=algo,
            env_id=env_id,
        )
        if is_performance_curriculum
        else None
    )
    if capture_spec is not None:
        print(
            "[selection] best_model uses strict capture: distance<0.2, "
            "speed<0.2, sustained for >=1 physical second",
            flush=True,
        )

    # Optional best-model gate: "occupancy_key:min_occupancy:min_reward"
    # (e.g. "acrobot_hold:0.05:400") -> best_model only updates on evals whose
    # dt-weighted mean info[key] and mean reward both clear the floors.
    gate_key, gate_occ, gate_rew = None, 0.0, float("-inf")
    if best_model_gate and capture_spec is not None:
        print(
            "[selection] strict sustained capture supersedes "
            "--best_model_gate for this task",
            flush=True,
        )
    elif best_model_gate:
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
        capture_spec=capture_spec,
    )

    mastery_curriculum_callback = None
    curriculum_probe_callback = None
    if curriculum_probe_env is not None:
        if curriculum_capture_spec is None:
            raise ValueError(
                f"{env_id} requires a sustained-capture spec for curriculum "
                "mastery"
            )
        train_curriculum_envs = list(_iter_dmc_envs(train_env))
        probe_curriculum_envs = list(_iter_dmc_envs(curriculum_probe_env))
        all_curriculum_envs = train_curriculum_envs + probe_curriculum_envs
        stage_counts = {
            env.num_curriculum_stages for env in all_curriculum_envs
        }
        if None in stage_counts or len(stage_counts) != 1:
            raise RuntimeError(
                "performance curriculum environments must expose one shared "
                f"stage count, got {stage_counts}"
            )
        num_curriculum_stages = int(next(iter(stage_counts)))

        def _set_mastery_curriculum_stage(
            stage: int, _envs=all_curriculum_envs
        ) -> None:
            for curriculum_env in _envs:
                curriculum_env.set_curriculum_stage(stage)

        def _get_mastery_curriculum_metrics(
            _env=train_curriculum_envs[0],
        ) -> dict[str, float]:
            return _env.curriculum_log_metrics()

        mastery_curriculum_callback = MasteryCurriculumCallback(
            set_stage=_set_mastery_curriculum_stage,
            num_stages=num_curriculum_stages,
            success_threshold=curriculum_success_threshold,
            consecutive_evals=curriculum_consecutive_evals,
            get_curriculum_metrics=_get_mastery_curriculum_metrics,
            verbose=1,
        )
        curriculum_probe_callback = EvalCallback(
            eval_env=curriculum_probe_env,
            eval_freq=max(eval_freq // n_envs, 1),
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            reset_seed=seed + 3000,
            log_path=str(log_dir / "curriculum_probe"),
            best_model_save_path=None,
            verbose=1,
            callback_after_eval=mastery_curriculum_callback,
            capture_spec=curriculum_capture_spec,
            log_prefix="curriculum",
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(save_dir),
        name_prefix=f"{algo}_{env_id}_{mode}",
        verbose=1,
    )

    def _eval_callback_state(callback: EvalCallback) -> dict:
        return {
            "best_mean_reward": float(callback.best_mean_reward),
            "last_mean_reward": float(callback.last_mean_reward),
            "best_capture_success_rate": float(
                callback.best_capture_success_rate
            ),
            "best_capture_duration": float(callback.best_capture_duration),
            "last_capture_success_rate": callback.last_capture_success_rate,
            "last_capture_duration": callback.last_capture_duration,
            "last_eval_timesteps": int(callback._last_eval_timesteps),
            "evaluations_timesteps": callback.evaluations_timesteps,
            "evaluations_results": callback.evaluations_results,
            "evaluations_lengths": callback.evaluations_lengths,
            "evaluations_capture_timesteps": (
                callback.evaluations_capture_timesteps
            ),
            "evaluations_capture_successes": (
                callback.evaluations_capture_successes
            ),
            "evaluations_capture_durations": (
                callback.evaluations_capture_durations
            ),
        }

    def _restore_eval_callback_state(
        callback: EvalCallback, state: dict
    ) -> None:
        if not state:
            return
        callback.best_mean_reward = state.get(
            "best_mean_reward", callback.best_mean_reward
        )
        callback.last_mean_reward = state.get(
            "last_mean_reward", callback.last_mean_reward
        )
        callback.best_capture_success_rate = state.get(
            "best_capture_success_rate",
            callback.best_capture_success_rate,
        )
        callback.best_capture_duration = state.get(
            "best_capture_duration", callback.best_capture_duration
        )
        callback.last_capture_success_rate = state.get(
            "last_capture_success_rate", callback.last_capture_success_rate
        )
        callback.last_capture_duration = state.get(
            "last_capture_duration", callback.last_capture_duration
        )
        callback._last_eval_timesteps = state.get("last_eval_timesteps", 0)
        callback.evaluations_timesteps = state.get(
            "evaluations_timesteps", []
        )
        callback.evaluations_results = state.get("evaluations_results", [])
        callback.evaluations_lengths = state.get("evaluations_lengths", [])
        callback.evaluations_capture_timesteps = state.get(
            "evaluations_capture_timesteps", []
        )
        callback.evaluations_capture_successes = state.get(
            "evaluations_capture_successes", []
        )
        callback.evaluations_capture_durations = state.get(
            "evaluations_capture_durations", []
        )

    # Resume: reload the full trainer state (buffer, optimizers, counters, RNG)
    # and restore the EvalCallback's cumulative history so the eval curve stays
    # whole and best_model.pth is not clobbered by a worse early eval.
    resume_extra = {}
    if resume_active:
        resume_extra = load_checkpoint(algorithm, ckpt_dir)
        _restore_eval_callback_state(
            eval_callback, resume_extra.get("eval", {})
        )
        if curriculum_probe_callback is not None:
            _restore_eval_callback_state(
                curriculum_probe_callback,
                resume_extra.get("curriculum_probe_eval", {}),
            )
        if (
            mastery_curriculum_callback is not None
            and "mastery_curriculum" in resume_extra
        ):
            mastery_curriculum_callback.load_state_dict(
                resume_extra["mastery_curriculum"]
            )
        if capture_spec is None:
            best_label = f"eval_best_reward={eval_callback.best_mean_reward:.3f}"
        else:
            best_label = (
                "eval_best_capture="
                f"{eval_callback.best_capture_success_rate:.3f}, "
                "duration="
                f"{eval_callback.best_capture_duration:.3f}s"
            )
        print(
            f"[resume] loaded checkpoint from {ckpt_dir}: "
            f"num_timesteps={algorithm.num_timesteps}, "
            f"buffer_size={algorithm.replay_buffer.size()}, {best_label}",
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
    if curriculum_probe_callback is not None:
        callbacks.append(curriculum_probe_callback)

    # Drive the reset curriculum from global training progress.  The setter
    # fans the fraction out to every training env; eval envs were built with
    # the curriculum off and are left untouched.
    if is_fraction_curriculum:
        curriculum_envs = list(_iter_dmc_envs(train_env))

        def _set_curriculum_fraction(frac, _envs=curriculum_envs):
            for e in _envs:
                e.set_curriculum_fraction(frac)

        def _get_fraction_curriculum_metrics(
            _env=curriculum_envs[0],
        ) -> dict[str, float]:
            return _env.curriculum_log_metrics()

        # Run before any evaluation callback that may flush the logger at the
        # same timestep, so every emitted row describes the current fraction.
        callbacks.insert(
            0,
            CurriculumFractionCallback(
                set_fraction=_set_curriculum_fraction,
                total_steps=curr_total,
                get_curriculum_metrics=_get_fraction_curriculum_metrics,
                verbose=1,
            ),
        )
        print(
            f"[curriculum] reset band widens to full over {curr_total} steps "
            f"({len(curriculum_envs)} train env(s)); eval starts fixed.",
            flush=True,
        )
    elif is_performance_curriculum:
        assert mastery_curriculum_callback is not None
        assert curriculum_capture_spec is not None
        terminal_requirement = (
            " through episode end"
            if curriculum_capture_spec.require_terminal_hold
            else ""
        )
        print(
            "[curriculum] tip-height ladder advances only after "
            f"{n_eval_episodes} deterministic probe episode(s) reach "
            f"{curriculum_success_threshold:.0%} stabilization held for "
            f">={curriculum_capture_spec.duration_seconds:g} physical "
            f"seconds{terminal_requirement} "
            f"for {curriculum_consecutive_evals} consecutive evaluation(s); "
            "checkpoint selection stays on the fixed hanging task.",
            flush=True,
        )

    # Optional second eval track from the canonical hanging start, run
    # alongside the (uniform-start) primary eval.  For acrobot v4.1/v5 the
    # training resets are uniform random, so the primary eval and its
    # best_model measure capture-from-anywhere; this hanging eval reports the
    # true swing-up-from-down task and saves its own best_model_hanging/,
    # without disturbing the primary best_model.
    hanging_eval_callback = None
    hanging_eval_env = None
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
            capture_spec=capture_spec,
            log_prefix="eval_hanging",
        )
        _restore_eval_callback_state(
            hanging_eval_callback,
            resume_extra.get("eval_hanging", {}),
        )
        callbacks.append(hanging_eval_callback)

    # Wall-clock checkpoint: near the time budget, write a resumable checkpoint
    # and stop cleanly so the resubmission chain can continue.
    wall_cb = None
    if max_seconds is not None and max_seconds > 0:
        def _collect_extra():
            state = {"eval": _eval_callback_state(eval_callback)}
            if curriculum_probe_callback is not None:
                state["curriculum_probe_eval"] = _eval_callback_state(
                    curriculum_probe_callback
                )
            if mastery_curriculum_callback is not None:
                state["mastery_curriculum"] = (
                    mastery_curriculum_callback.state_dict()
                )
            if hanging_eval_callback is not None:
                state["eval_hanging"] = _eval_callback_state(
                    hanging_eval_callback
                )
            elif "eval_hanging" in resume_extra:
                # Preserve an existing optional track even if this chunk was
                # launched without --eval_hanging.
                state["eval_hanging"] = resume_extra["eval_hanging"]
            return state

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
    try:
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
                "Training finished: reached "
                f"{algorithm.num_timesteps}/{total_timesteps} steps; "
                f"saved exact endpoint to {final_model_path}.",
                flush=True,
            )
        else:
            print(
                "Training paused at "
                f"{algorithm.num_timesteps}/{total_timesteps} steps "
                "(will resume from checkpoint).",
                flush=True,
            )
        return finished
    finally:
        # VecContinuousEnv has no aggregate close method. Release every native
        # DMC environment explicitly, including the extra changing-stage probe,
        # so multi-algorithm invocations do not accumulate MuJoCo resources.
        closed: set[int] = set()
        for root_env in (
            train_env,
            eval_env,
            curriculum_probe_env,
            hanging_eval_env,
        ):
            if root_env is None:
                continue
            for dmc_env in _iter_dmc_envs(root_env):
                if id(dmc_env) not in closed:
                    dmc_env.close()
                    closed.add(id(dmc_env))


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
        "--curriculum_steps",
        type=int,
        default=None,
        help="Steps over which the acrobot v4.2 reset band widens to the full "
        "circle. Default: half the training budget. Ignored for other tasks.",
    )
    parser.add_argument(
        "--curriculum_success_threshold",
        type=float,
        default=0.8,
        help="Deterministic sustained-stabilization success rate required to "
        "advance a performance-gated tip-height curriculum.",
    )
    parser.add_argument(
        "--curriculum_consecutive_evals",
        type=int,
        default=1,
        help="Consecutive mastery probe evaluations that must clear the "
        "success threshold before lowering the curriculum tip height.",
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
                curriculum_steps=args.curriculum_steps,
                curriculum_success_threshold=(
                    args.curriculum_success_threshold
                ),
                curriculum_consecutive_evals=(
                    args.curriculum_consecutive_evals
                ),
            )
            all_finished = all_finished and bool(finished)
        except (FileNotFoundError, KeyError) as e:
            print(f"\nCould not run {algo} due to a configuration error: {e}\n")

    if args.max_seconds is not None and not all_finished:
        import sys

        sys.exit(42)


if __name__ == "__main__":
    main()
