# evaluations/evaluation_helpers.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import os
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import SAC, TD3, PPO

try:
    from sb3_contrib import TRPO
except ImportError:
    TRPO = None

from environment.base import ContinuousEnv
from environment.dmc import DMCContinuousEnv
from environment.vec_env import VecContinuousEnv
from environment.trading_env import TradingContinuousEnv
from data.trading.config import EVAL_NPZ
from models.base import Model
from models.actor_q_critic import ActorQCriticModel
from models.actor_v_critic import ActorVCriticModel
from models.coupled_vq import CoupledVqModel
from evaluations.sustained_capture import (
    CaptureEpisodeResult,
    SustainedCaptureSpec,
    SustainedCaptureTracker,
)
from common.utils import (
    load_ct_hyperparams_from_table,
    get_device,
    normalize_eval_range,
)


@dataclass
class EpisodeEvaluationResults:
    """Per-episode outputs, including optional physical-time diagnostics."""

    returns: List[float]
    lengths: List[int]
    occupancies: Optional[List[float]] = None
    capture_successes: Optional[List[bool]] = None
    capture_durations: Optional[List[float]] = None


# Global map from algorithm string to the corresponding model or algorithm class
ALGO_CLASS_MAP = {
    "ct_sac": ActorQCriticModel,
    "ct_td3": ActorQCriticModel,
    "ct_ddpg": ActorQCriticModel,
    "q_learning": CoupledVqModel,
    "cpg": ActorVCriticModel,
    "cppo": ActorVCriticModel,
    "coupled_sarsa": CoupledVqModel,
    "sac": SAC,
    "td3": TD3,
    "ppo": PPO,
    "trpo": TRPO,
}


def _is_vec_env(env: Any) -> bool:
    return hasattr(env, "num_envs") and isinstance(getattr(env, "num_envs"), int)


# Episodic evaluation on either single env or vectorized(multiple)-env
def evaluate_policy_per_episode(
    model: Model,
    env: Union[ContinuousEnv, VecContinuousEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    reset_seed: Optional[int] = None,
    occupancy_key: Optional[str] = None,
    capture_spec: Optional[SustainedCaptureSpec] = None,
    return_metrics: bool = False,
):
    """
    Episodic eval for continuous-time model.
    The implementation is aware of whether the env is a Monitor wrapper:
      - if info has "episode", use info["episode"]["r"] and ["l"]
      - otherwise use accumulated sums

    Returns:
      episode_returns: List[float] length n_eval_episodes
      episode_lengths: List[int]   length n_eval_episodes

    If ``occupancy_key`` is given, a per-step scalar is read from
    ``info[occupancy_key]`` each step and its dt-weighted mean over each episode
    is returned as a third list ``episode_occupancies`` (used e.g. for the
    acrobot hold-occupancy best-model gate).

    If ``capture_spec`` is given, strict-capture residence is accumulated in
    physical time between consecutive qualifying endpoints. With
    ``return_metrics=True`` all outputs are returned in
    :class:`EpisodeEvaluationResults`; the legacy tuple return is preserved by
    default.
    """
    is_vec_env = _is_vec_env(env)
    n_envs = int(getattr(env, "num_envs", 1)) if is_vec_env else 1

    # A fixed initial reset makes every callback invocation evaluate the same
    # episode/reset and irregular-time streams.  Subsequent episode resets
    # advance those freshly rooted local streams deterministically.
    obs, reset_infos = env.reset(seed=reset_seed)
    if not is_vec_env:
        obs = np.asarray(obs, dtype=np.float32)
        reset_info_list = [reset_infos]
    else:
        reset_info_list = (
            list(reset_infos)
            if isinstance(reset_infos, (list, tuple))
            else [reset_infos] * n_envs
        )

    running_returns = np.zeros((n_envs,), dtype=np.float64)
    running_lengths = np.zeros((n_envs,), dtype=np.int64)
    running_occ_w = np.zeros((n_envs,), dtype=np.float64)  # dt-weighted occupancy sum
    running_dt = np.zeros((n_envs,), dtype=np.float64)

    episode_returns: List[float] = []
    episode_lengths: List[int] = []
    episode_occupancies: List[float] = []
    episode_capture_successes: List[bool] = []
    episode_capture_durations: List[float] = []
    capture_tracker = (
        SustainedCaptureTracker(n_envs, capture_spec, reset_info_list)
        if capture_spec is not None
        else None
    )

    while len(episode_returns) < n_eval_episodes:
        obs_batch = obs if is_vec_env else obs[None, ...]
        obs_tensor = th.as_tensor(obs_batch, dtype=th.float32, device=model.device)

        with th.no_grad():
            act_tensor, _ = model.act(obs_tensor, deterministic=deterministic)
        actions = act_tensor.detach().cpu().numpy()
        action_for_env = actions if is_vec_env else actions[0]

        obs_t, t, _, reward, next_obs, next_t, terminated, truncated, infos = (
            env.step_dt(action_for_env)
        )
        done = np.logical_or(terminated, truncated)

        # normalize to arrays
        rew_arr = (
            np.asarray(reward, dtype=np.float64).reshape(-1)
            if is_vec_env
            else np.asarray([reward], dtype=np.float64)
        )
        done_arr = (
            np.asarray(done, dtype=bool).reshape(-1)
            if is_vec_env
            else np.asarray([bool(done)], dtype=bool)
        )

        running_returns += rew_arr
        running_lengths += 1

        info_list = (
            list(infos)
            if is_vec_env and isinstance(infos, (list, tuple))
            else [infos] * n_envs
        )

        if occupancy_key is not None:
            dt_arr = np.asarray(
                [
                    float(info_list[i].get("dt_used", np.nan))
                    for i in range(n_envs)
                ],
                dtype=np.float64,
            )
            missing_dt = ~np.isfinite(dt_arr) | (dt_arr <= 0.0)
            if np.any(missing_dt):
                t_arr = np.asarray(t, dtype=np.float64).reshape(-1)
                next_t_arr = np.asarray(next_t, dtype=np.float64).reshape(-1)
                if t_arr.size == 1:
                    t_arr = np.full(n_envs, float(t_arr[0]))
                if next_t_arr.size == 1:
                    next_t_arr = np.full(n_envs, float(next_t_arr[0]))
                fallback = next_t_arr - t_arr
                for i in np.where(missing_dt)[0]:
                    if done_arr[i] and "terminal_next_t" in info_list[i]:
                        fallback[i] = (
                            float(info_list[i]["terminal_next_t"]) - t_arr[i]
                        )
                dt_arr[missing_dt] = fallback[missing_dt]
            if not np.all(np.isfinite(dt_arr)) or np.any(dt_arr <= 0.0):
                raise ValueError(
                    f"evaluation step durations must be finite and > 0, got {dt_arr}"
                )
            occ_arr = np.asarray(
                [
                    float(info_list[i].get(occupancy_key, 0.0))
                    for i in range(n_envs)
                ],
                dtype=np.float64,
            )
            running_occ_w += occ_arr * dt_arr
            running_dt += dt_arr

        capture_results: List[Optional[CaptureEpisodeResult]] = [None] * n_envs
        if capture_tracker is not None:
            for i in range(n_envs):
                reset_info = (
                    info_list[i].get("reset_info")
                    if bool(done_arr[i]) and is_vec_env
                    else None
                )
                capture_results[i] = capture_tracker.update_slot(
                    i,
                    info_list[i],
                    done=bool(done_arr[i]),
                    reset_info=reset_info,
                )

        if is_vec_env:
            # Record stats at the end of episode of a subset of envs inside the vec_env
            done_indices = np.where(done_arr)[0]
            for i in done_indices:
                info_i = info_list[i]
                if isinstance(info_i, dict) and "episode" in info_i:
                    ep_r = float(info_i["episode"]["r"])
                    ep_l = int(info_i["episode"]["l"])
                else:
                    ep_r = float(running_returns[i])
                    ep_l = int(running_lengths[i])

                episode_returns.append(ep_r)
                episode_lengths.append(ep_l)
                if occupancy_key is not None:
                    episode_occupancies.append(
                        float(running_occ_w[i] / running_dt[i])
                        if running_dt[i] > 0 else 0.0
                    )
                if capture_tracker is not None:
                    capture_result = capture_results[i]
                    if capture_result is None:
                        raise RuntimeError("missing terminal capture result")
                    episode_capture_successes.append(capture_result.success)
                    episode_capture_durations.append(
                        capture_result.max_duration_seconds
                    )

                running_returns[i] = 0.0
                running_lengths[i] = 0
                running_occ_w[i] = 0.0
                running_dt[i] = 0.0

                if len(episode_returns) >= n_eval_episodes:
                    break

            # VecContinuousEnv already auto-resets done envs internally;
            # next_obs is already the reset obs for those env slots.
            obs = next_obs

        else:
            if bool(done_arr[0]):
                # Record stats at the end of episode
                info = infos
                if isinstance(info, dict) and "episode" in info:
                    ep_r = float(info["episode"]["r"])
                    ep_l = int(info["episode"]["l"])
                else:
                    ep_r = float(running_returns[0])
                    ep_l = int(running_lengths[0])

                episode_returns.append(ep_r)
                episode_lengths.append(ep_l)
                if occupancy_key is not None:
                    episode_occupancies.append(
                        float(running_occ_w[0] / running_dt[0])
                        if running_dt[0] > 0 else 0.0
                    )
                if capture_tracker is not None:
                    capture_result = capture_results[0]
                    if capture_result is None:
                        raise RuntimeError("missing terminal capture result")
                    episode_capture_successes.append(capture_result.success)
                    episode_capture_durations.append(
                        capture_result.max_duration_seconds
                    )

                running_returns[0] = 0.0
                running_lengths[0] = 0
                running_occ_w[0] = 0.0
                running_dt[0] = 0.0

                # single env must reset explicitly
                obs, reset_info = env.reset()
                obs = np.asarray(obs, dtype=np.float32)
                if capture_tracker is not None:
                    capture_tracker.reset_slot(0, reset_info)
            else:
                obs = np.asarray(next_obs, dtype=np.float32)

    metrics = EpisodeEvaluationResults(
        returns=episode_returns,
        lengths=episode_lengths,
        occupancies=(episode_occupancies if occupancy_key is not None else None),
        capture_successes=(
            episode_capture_successes if capture_spec is not None else None
        ),
        capture_durations=(
            episode_capture_durations if capture_spec is not None else None
        ),
    )
    if return_metrics:
        return metrics
    if occupancy_key is not None and capture_spec is not None:
        return (
            episode_returns,
            episode_lengths,
            episode_occupancies,
            episode_capture_successes,
            episode_capture_durations,
        )
    if capture_spec is not None:
        return (
            episode_returns,
            episode_lengths,
            episode_capture_successes,
            episode_capture_durations,
        )
    if occupancy_key is not None:
        return episode_returns, episode_lengths, episode_occupancies
    return episode_returns, episode_lengths


# Step-level evaluation
def evaluate_policy_per_step(
    model: Optional[Model],
    env: ContinuousEnv,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    render_interval: int = 10,
    probe_fn: Optional[Callable[[ContinuousEnv], Any]] = None,
) -> Dict[str, Any]:
    """
    Step-level eval for continuous-time model on a Single ContinuousEnv.

    `probe_fn`, if given, is called after each step with the env and its
    return value is recorded in `episode_step_probes` (parallel to rewards).

    Returns a dict with the usual fields plus, if probe_fn is provided,
    `episode_step_probes: List[List[Any]]`.
    """
    episode_step_rewards: List[List[float]] = []
    episode_timestamps: List[List[float]] = []
    episode_lengths: List[int] = []
    episode_returns: List[float] = []
    episode_frames: Optional[List[List[np.ndarray]]] = [] if render else None
    episode_step_probes: Optional[List[List[Any]]] = (
        [] if probe_fn is not None else None
    )

    device = model.device if model is not None else "auto"
    device = get_device(device)
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        obs = np.asarray(obs, dtype=np.float32)

        done = False
        step_rewards: List[float] = []
        ts: List[float] = []
        frames: List[np.ndarray] = []
        probes: List[Any] = []
        step_idx = 0

        while not done:
            if model is None:
                action = env.action_space.sample()
            else:
                obs_tensor = th.as_tensor(
                    obs, dtype=th.float32, device=device
                ).unsqueeze(0)
                with th.no_grad():
                    act_tensor, _ = model.act(obs_tensor, deterministic=deterministic)
                    action = act_tensor.detach().cpu().numpy()[0]

            obs_t, t, _, reward, next_obs, next_t, terminated, truncated, info = (
                env.step_dt(action)
            )
            done = bool(terminated or truncated)

            step_rewards.append(float(reward))
            ts.append(float(t))
            obs = np.asarray(next_obs, dtype=np.float32)

            if probe_fn is not None:
                probes.append(probe_fn(env))

            if render and (step_idx % render_interval == 0):
                frame = env.render(mode="rgb_array")
                if frame is not None:
                    frames.append(frame)

            step_idx += 1

        episode_step_rewards.append(step_rewards)
        episode_timestamps.append(ts)
        episode_lengths.append(len(step_rewards))
        episode_returns.append(float(np.sum(step_rewards)))
        if render and episode_frames is not None:
            episode_frames.append(frames)
        if episode_step_probes is not None:
            episode_step_probes.append(probes)

    out: Dict[str, Any] = {
        "episode_step_rewards": episode_step_rewards,
        "episode_timestamps": episode_timestamps,
        "episode_lengths": episode_lengths,
        "episode_returns": episode_returns,
        "episode_frames": episode_frames,
    }
    if episode_step_probes is not None:
        out["episode_step_probes"] = episode_step_probes
    return out


def evaluate_sb3_policy_per_step(
    sb3_model: Any,
    env: ContinuousEnv,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    render_interval: int = 10,
    probe_fn: Optional[Callable[[ContinuousEnv], Any]] = None,
) -> Dict[str, Any]:
    """
    Step-level eval for SB3 model, for visualization/benchmarking.

    See `evaluate_policy_per_step` for the `probe_fn` semantics.
    """
    episode_step_rewards: List[List[float]] = []
    episode_timestamps: List[List[float]] = []
    episode_lengths: List[int] = []
    episode_returns: List[float] = []
    episode_frames: Optional[List[List[np.ndarray]]] = [] if render else None
    episode_step_probes: Optional[List[List[Any]]] = (
        [] if probe_fn is not None else None
    )

    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False

        step_rewards: List[float] = []
        ts: List[float] = []
        frames: List[np.ndarray] = []
        probes: List[Any] = []
        step_idx = 0

        while not done:
            action, _ = sb3_model.predict(obs, deterministic=deterministic)
            obs_t, t, _, reward, next_obs, next_t, terminated, truncated, info = (
                env.step_dt(action)
            )
            done = bool(terminated or truncated)

            step_rewards.append(float(reward))
            ts.append(float(t))
            obs = next_obs

            if probe_fn is not None:
                probes.append(probe_fn(env))

            if render and (step_idx % render_interval == 0):
                frame = env.render(mode="rgb_array")
                if frame is not None:
                    frames.append(frame)

            step_idx += 1

        episode_step_rewards.append(step_rewards)
        episode_timestamps.append(ts)
        episode_lengths.append(len(step_rewards))
        episode_returns.append(float(np.sum(step_rewards)))
        if render and episode_frames is not None:
            episode_frames.append(frames)
        if episode_step_probes is not None:
            episode_step_probes.append(probes)

    out: Dict[str, Any] = {
        "episode_step_rewards": episode_step_rewards,
        "episode_timestamps": episode_timestamps,
        "episode_lengths": episode_lengths,
        "episode_returns": episode_returns,
        "episode_frames": episode_frames,
    }
    if episode_step_probes is not None:
        out["episode_step_probes"] = episode_step_probes
    return out


def _force_dmc_regular_time(env) -> None:
    """
    For DMC envs, force "regular" mode to match dm_control's default control timestep.
    This overrides any irregular/small-time settings that may be present in env_kwargs.
    """
    try:
        if hasattr(env, "time_sampling"):
            env.time_sampling = "uniform"

        # DMCContinuousEnv defines dt_default = env.control_timestep (dm_control)
        if hasattr(env, "dt_default"):
            env.dt = float(env.dt_default)

        # Clear irregular bounds if present
        if hasattr(env, "min_dt"):
            env.min_dt = None
        if hasattr(env, "max_dt"):
            env.max_dt = None
        if hasattr(env, "time_sampling_kwargs"):
            env.time_sampling_kwargs = {}

    except Exception:
        # best effort only; don't crash evaluation
        pass


def create_evaluation_env_and_model(
    env_id: str,
    model_class: Optional[Type[Model]],
    seed: int,
    algo: str,
    mode: Optional[str] = None,
    hyperparams_dir: Optional[str] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    quarters: Optional[List[str]] = None,
) -> Tuple[ContinuousEnv, Optional[Model]]:
    # Load/get hyperparams
    if mode:
        if hyperparams_dir is None:
            hyperparams_dir = "benchmarks/hyperparams"
        try:
            _, loaded_env_kwargs, loaded_model_kwargs, _, _ = (
                load_ct_hyperparams_from_table(
                    algo=algo,
                    env_id=env_id,
                    mode=mode,
                    hyperparams_dir=hyperparams_dir,
                )
            )
            env_kwargs = (
                loaded_env_kwargs
                if env_kwargs is None
                else {**loaded_env_kwargs, **env_kwargs}
            )
            model_kwargs = (
                loaded_model_kwargs
                if model_kwargs is None
                else {**loaded_model_kwargs, **model_kwargs}
            )
        except Exception as e:
            print(
                f"[WARN] Failed to load hyperparams for (algo={algo}, env={env_id}, mode={mode}): {e}"
            )
            print("[WARN] Falling back to caller-provided evaluation defaults.")
            env_kwargs = dict(env_kwargs or {})
            model_kwargs = dict(model_kwargs or {})
    else:
        print(f"[WARN] Unknown mode for (algo={algo}, env={env_id})")
        return None, None

    # Create the env
    if "n_envs" in env_kwargs:
        env_kwargs.pop("n_envs")  # Don't support visualize vectorized env yet.

    if env_id.startswith("trading") and quarters is not None:
        env_kwargs["eval_range"] = normalize_eval_range(quarters)
        env_kwargs["eval_cycle_tickers"] = True

    if env_id.startswith("trading"):
        env = TradingContinuousEnv(
            npz_path=EVAL_NPZ,
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

    # DMC regular mode: dm_control default dt
    if env_id != "trading" and mode in {"regular", "normal"}:
        _force_dmc_regular_time(env)

    # Create the model.
    model_instance = None
    if model_class is not None:
        model_instance = model_class(
            observation_space=env.observation_space,
            action_space=env.action_space,
            **model_kwargs,
        )
    return env, model_instance


def get_latest_run_dir(base_dir: Union[str, Path]) -> str:
    path = Path(base_dir)
    if not path.exists():
        return str(base_dir)
    subdirs = [d for d in path.iterdir() if d.is_dir()]
    if not subdirs:
        return str(base_dir)

    latest_subdir = max(subdirs, key=lambda p: p.stat().st_mtime)
    return str(latest_subdir)
