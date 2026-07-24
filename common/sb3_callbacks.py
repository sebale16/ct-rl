"""Stable-Baselines3 callbacks shared with the continuous-time benchmarks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import sync_envs_normalization

from evaluations.sustained_capture import (
    CaptureEpisodeResult,
    SustainedCaptureSpec,
    SustainedCaptureTracker,
    capture_selection_rank,
)


@dataclass(frozen=True)
class SB3CaptureEvaluation:
    """Per-episode reward and strict-capture results from one evaluation."""

    rewards: list[float]
    lengths: list[int]
    capture_successes: list[bool]
    capture_durations: list[float]


def _capture_reset_infos(
    env: Any, spec: SustainedCaptureSpec
) -> list[Mapping[str, Any]]:
    """Find Gymnasium reset infos through optional VecEnv wrappers."""

    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        infos = getattr(current, "reset_infos", None)
        if (
            isinstance(infos, (list, tuple))
            and len(infos) == int(env.num_envs)
            and all(
                isinstance(info, Mapping) and spec.info_key in info
                for info in infos
            )
        ):
            return list(infos)
        current = getattr(current, "venv", None)
    raise KeyError(
        f"strict capture evaluation requires reset info[{spec.info_key!r}] "
        "for every vector environment slot"
    )


def evaluate_sb3_policy_with_capture(
    model: Any,
    env: Any,
    *,
    n_eval_episodes: int,
    deterministic: bool,
    render: bool,
    capture_spec: SustainedCaptureSpec,
) -> SB3CaptureEvaluation:
    """Evaluate an SB3 policy and measure conservative physical-time capture.

    As in the CT evaluator, an interval counts only when both of its observed
    endpoints satisfy the strict predicate. Vector slots receive a fixed
    episode quota, matching SB3's bias-free evaluation allocation.
    """

    n_envs = int(env.num_envs)
    if n_envs <= 0:
        raise ValueError("evaluation environment must have at least one slot")
    if int(n_eval_episodes) <= 0:
        raise ValueError("n_eval_episodes must be positive")

    observations = env.reset()
    tracker = SustainedCaptureTracker(
        n_envs, capture_spec, _capture_reset_infos(env, capture_spec)
    )

    episode_counts = np.zeros(n_envs, dtype=np.int64)
    episode_targets = np.asarray(
        [(int(n_eval_episodes) + i) // n_envs for i in range(n_envs)],
        dtype=np.int64,
    )
    running_rewards = np.zeros(n_envs, dtype=np.float64)
    running_lengths = np.zeros(n_envs, dtype=np.int64)
    episode_starts = np.ones(n_envs, dtype=bool)
    states: Optional[Any] = None

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    capture_successes: list[bool] = []
    capture_durations: list[float] = []

    while np.any(episode_counts < episode_targets):
        actions, states = model.predict(
            observations,
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        observations, rewards, dones, infos = env.step(actions)
        rewards = np.asarray(rewards, dtype=np.float64)
        dones = np.asarray(dones, dtype=bool)
        episode_starts = dones

        for i in range(n_envs):
            active = episode_counts[i] < episode_targets[i]
            if active:
                running_rewards[i] += rewards[i]
                running_lengths[i] += 1

            reset_info = None
            if dones[i]:
                reset_info = _capture_reset_infos(env, capture_spec)[i]
            capture_result = tracker.update_slot(
                i,
                infos[i],
                done=bool(dones[i]),
                reset_info=reset_info,
            )

            if not active or not dones[i]:
                continue
            if capture_result is None:
                raise RuntimeError("missing terminal strict-capture result")

            monitor_episode = infos[i].get("episode")
            if monitor_episode is None:
                episode_rewards.append(float(running_rewards[i]))
                episode_lengths.append(int(running_lengths[i]))
            else:
                episode_rewards.append(float(monitor_episode["r"]))
                episode_lengths.append(int(monitor_episode["l"]))
            capture_successes.append(capture_result.success)
            capture_durations.append(capture_result.max_duration_seconds)
            episode_counts[i] += 1
            running_rewards[i] = 0.0
            running_lengths[i] = 0

        if render:
            env.render()

    if len(episode_rewards) != int(n_eval_episodes):
        raise RuntimeError(
            f"expected {n_eval_episodes} episodes, got {len(episode_rewards)}"
        )
    return SB3CaptureEvaluation(
        rewards=episode_rewards,
        lengths=episode_lengths,
        capture_successes=capture_successes,
        capture_durations=capture_durations,
    )


class SustainedCaptureEvalCallback(EvalCallback):
    """SB3 evaluation callback whose best checkpoint uses strict capture."""

    def __init__(
        self,
        eval_env: Any,
        *,
        capture_spec: SustainedCaptureSpec,
        reset_seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(eval_env, **kwargs)
        self.capture_spec = capture_spec
        self.reset_seed = None if reset_seed is None else int(reset_seed)
        self.best_capture_success_rate = -np.inf
        self.best_capture_duration = -np.inf
        self.evaluations_capture_successes: list[list[bool]] = []
        self.evaluations_capture_durations: list[list[float]] = []

    def _on_step(self) -> bool:
        continue_training = True
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return continue_training

        if self.model.get_vec_normalize_env() is not None:
            try:
                sync_envs_normalization(self.training_env, self.eval_env)
            except AttributeError as exc:
                raise AssertionError(
                    "training and evaluation environments must use matching "
                    "VecNormalize wrappers"
                ) from exc

        # VecEnv.seed() applies to its next reset. Reapplying it here makes
        # every candidate checkpoint face the same reset and time streams.
        if self.reset_seed is not None:
            self.eval_env.seed(self.reset_seed)

        results = evaluate_sb3_policy_with_capture(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
            render=self.render,
            capture_spec=self.capture_spec,
        )
        rewards = np.asarray(results.rewards, dtype=np.float64)
        lengths = np.asarray(results.lengths, dtype=np.int64)
        capture_rate, mean_capture_duration = capture_selection_rank(
            results.capture_successes, results.capture_durations
        )

        self.evaluations_timesteps.append(self.num_timesteps)
        self.evaluations_results.append(results.rewards)
        self.evaluations_length.append(results.lengths)
        self.evaluations_capture_successes.append(results.capture_successes)
        self.evaluations_capture_durations.append(results.capture_durations)
        if self.log_path is not None:
            np.savez(
                self.log_path,
                timesteps=self.evaluations_timesteps,
                results=self.evaluations_results,
                ep_lengths=self.evaluations_length,
                capture_successes=self.evaluations_capture_successes,
                capture_durations=self.evaluations_capture_durations,
            )

        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))
        mean_length = float(np.mean(lengths))
        std_length = float(np.std(lengths))
        self.last_mean_reward = mean_reward

        if self.verbose >= 1:
            print(
                f"Eval num_timesteps={self.num_timesteps}, "
                f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
            )
            print(
                f"Episode length: {mean_length:.2f} +/- {std_length:.2f}; "
                f"strict capture={capture_rate:.3f}, "
                f"mean max duration={mean_capture_duration:.3f}s"
            )

        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/mean_ep_length", mean_length)
        self.logger.record("eval/strict_capture_success_rate", capture_rate)
        self.logger.record(
            "eval/strict_capture_mean_max_duration", mean_capture_duration
        )
        self.logger.record(
            "time/total_timesteps",
            self.num_timesteps,
            exclude="tensorboard",
        )

        rank = (capture_rate, mean_capture_duration)
        best_rank = (
            self.best_capture_success_rate,
            self.best_capture_duration,
        )
        if rank > best_rank:
            # Keep this legacy attribute tied to the selected checkpoint, as
            # the CT callback does; raw reward remains logged every eval.
            self.best_mean_reward = mean_reward
            self.best_capture_success_rate = capture_rate
            self.best_capture_duration = mean_capture_duration
            if self.best_model_save_path is not None:
                if self.verbose >= 1:
                    print(
                        "New best strict capture score; saving model to "
                        f"{self.best_model_save_path}"
                    )
                self.model.save(
                    os.path.join(self.best_model_save_path, "best_model")
                )
            if self.callback_on_new_best is not None:
                continue_training = self.callback_on_new_best.on_step()

        self.logger.record(
            "eval/best_strict_capture_success_rate",
            self.best_capture_success_rate,
        )
        self.logger.record(
            "eval/best_strict_capture_mean_max_duration",
            self.best_capture_duration,
        )
        self.logger.dump(self.num_timesteps)

        if continue_training and self.callback is not None:
            continue_training = self._on_event()
        return continue_training
