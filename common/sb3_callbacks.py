"""Stable-Baselines3 callbacks shared with the continuous-time benchmarks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import sync_envs_normalization

from evaluations.sustained_capture import (
    CaptureEpisodeResult,
    SustainedCaptureSpec,
    SustainedCaptureTracker,
    capture_selection_rank,
)
from common.mastery_curriculum import MasteryCurriculum


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
        log_prefix: str = "eval",
        **kwargs: Any,
    ) -> None:
        super().__init__(eval_env, **kwargs)
        self.capture_spec = capture_spec
        self.reset_seed = None if reset_seed is None else int(reset_seed)
        self.log_prefix = str(log_prefix).strip("/")
        if not self.log_prefix:
            raise ValueError("log_prefix must be non-empty")
        self.best_capture_success_rate = -np.inf
        self.best_capture_duration = -np.inf
        self.last_capture_success_rate: Optional[float] = None
        self.last_capture_duration: Optional[float] = None
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
        self.last_capture_success_rate = float(capture_rate)
        self.last_capture_duration = float(mean_capture_duration)

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
                f"{self.log_prefix} num_timesteps={self.num_timesteps}, "
                f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
            )
            print(
                f"Episode length: {mean_length:.2f} +/- {std_length:.2f}; "
                f"strict capture={capture_rate:.3f}, "
                f"mean max duration={mean_capture_duration:.3f}s"
            )

        self.logger.record(f"{self.log_prefix}/mean_reward", mean_reward)
        self.logger.record(
            f"{self.log_prefix}/mean_ep_length", mean_length
        )
        self.logger.record(
            f"{self.log_prefix}/strict_capture_success_rate", capture_rate
        )
        self.logger.record(
            f"{self.log_prefix}/strict_capture_mean_max_duration",
            mean_capture_duration,
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
            f"{self.log_prefix}/best_strict_capture_success_rate",
            self.best_capture_success_rate,
        )
        self.logger.record(
            f"{self.log_prefix}/best_strict_capture_mean_max_duration",
            self.best_capture_duration,
        )

        # Child callbacks consume this evaluation result and may add metrics
        # (for example the newly selected curriculum stage), so run them before
        # flushing the logger.
        if continue_training and self.callback is not None:
            continue_training = self._on_event()
        self.logger.dump(self.num_timesteps)
        return continue_training


class MasteryCurriculumCallback(BaseCallback):
    """Advance and log a curriculum after strict-capture evaluation.

    ``curriculum/probe_stage`` is the evaluated level; ``curriculum/stage`` and
    optional physical descriptors are the selected level after any transition.
    """

    def __init__(
        self,
        set_stage: Callable[[int], None],
        num_stages: int,
        success_threshold: float = 0.8,
        consecutive_evals: int = 1,
        verbose: int = 0,
        get_curriculum_metrics: Optional[
            Callable[[], Mapping[str, float]]
        ] = None,
    ) -> None:
        super().__init__(verbose)
        if not callable(set_stage):
            raise TypeError("set_stage must be callable")
        if (
            get_curriculum_metrics is not None
            and not callable(get_curriculum_metrics)
        ):
            raise TypeError("get_curriculum_metrics must be callable")
        self.set_stage = set_stage
        self.get_curriculum_metrics = get_curriculum_metrics
        self.curriculum = MasteryCurriculum(
            num_stages=num_stages,
            success_threshold=success_threshold,
            consecutive_evals=consecutive_evals,
        )

    @property
    def stage(self) -> int:
        return self.curriculum.stage

    @property
    def num_stages(self) -> int:
        return self.curriculum.num_stages

    def state_dict(self) -> dict[str, Any]:
        return self.curriculum.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.curriculum.load_state_dict(state)
        self.set_stage(self.stage)

    def _init_callback(self) -> None:
        self._record_progress(
            probe_stage=self.stage,
            success_rate=None,
            advanced=False,
        )

    def _record_progress(
        self,
        *,
        probe_stage: int,
        success_rate: Optional[float],
        advanced: bool,
    ) -> None:
        if self.get_curriculum_metrics is not None:
            for name, value in self.get_curriculum_metrics().items():
                self.logger.record(f"curriculum/{name}", float(value))

        progress = (
            self.stage / (self.num_stages - 1)
            if self.num_stages > 1
            else 1.0
        )
        self.logger.record("curriculum/probe_stage", int(probe_stage))
        self.logger.record("curriculum/stage", self.stage)
        self.logger.record("curriculum/num_stages", self.num_stages)
        self.logger.record("curriculum/progress", float(progress))
        self.logger.record(
            "curriculum/complete",
            float(self.curriculum.at_final_stage),
        )
        self.logger.record("curriculum/advanced", float(advanced))
        self.logger.record(
            "curriculum/probe_success_rate",
            np.nan if success_rate is None else float(success_rate),
        )
        self.logger.record(
            "curriculum/probe_passed",
            np.nan
            if success_rate is None
            else float(success_rate >= self.curriculum.success_threshold),
        )
        self.logger.record(
            "curriculum/consecutive_passes",
            self.curriculum.consecutive_passes,
        )
        self.logger.record(
            "curriculum/required_consecutive_evals",
            self.curriculum.consecutive_evals,
        )
        self.logger.record(
            "curriculum/success_threshold",
            self.curriculum.success_threshold,
        )

    def _on_step(self) -> bool:
        parent = getattr(self, "parent", None)
        success_rate = getattr(parent, "last_capture_success_rate", None)
        if success_rate is None:
            return True

        rate = float(success_rate)
        probe_stage = self.stage
        advanced = self.curriculum.observe(rate)
        if advanced:
            self.set_stage(self.stage)
            if self.verbose > 0:
                print(
                    f"[curriculum] probe success={rate:.3f}; "
                    f"advanced to stage {self.stage}/"
                    f"{self.curriculum.num_stages - 1}",
                    flush=True,
                )

        self._record_progress(
            probe_stage=probe_stage,
            success_rate=rate,
            advanced=advanced,
        )
        return True


class CurriculumFractionCallback(BaseCallback):
    """Drive a dm_control reset curriculum from SB3 training progress.

    Each step this pushes ``fraction = min(1, num_timesteps / total_steps)`` to
    every training env via ``env_method("set_curriculum_fraction", ...)``.
    Keying off ``num_timesteps`` — restored by SB3 on a resumed ``learn`` — keeps
    the schedule continuous across training chunks.  ``total_steps <= 0`` pins
    the fraction at 1 (curriculum complete).  Pushes are de-duplicated at
    millesimal resolution so the vectorized method call fires only when the
    schedule actually advances. Applied progress and optional reset-band
    descriptors are logged under ``curriculum/`` on every callback step.
    """

    def __init__(
        self,
        total_steps: int,
        verbose: int = 0,
        get_curriculum_metrics: Optional[
            Callable[[], Mapping[str, float]]
        ] = None,
    ) -> None:
        super().__init__(verbose)
        self.total_steps = int(total_steps)
        if (
            get_curriculum_metrics is not None
            and not callable(get_curriculum_metrics)
        ):
            raise TypeError("get_curriculum_metrics must be callable")
        self.get_curriculum_metrics = get_curriculum_metrics
        self._last_pushed: Optional[float] = None

    def _fraction(self) -> float:
        if self.total_steps <= 0:
            return 1.0
        return float(min(1.0, max(0.0, self.num_timesteps / self.total_steps)))

    def _apply(self) -> None:
        frac = round(self._fraction(), 3)
        if frac != self._last_pushed:
            self._last_pushed = frac
            self.training_env.env_method("set_curriculum_fraction", frac)
        if self.get_curriculum_metrics is not None:
            for name, value in self.get_curriculum_metrics().items():
                self.logger.record(f"curriculum/{name}", float(value))
        self.logger.record("curriculum/fraction", frac)
        self.logger.record("curriculum/progress", frac)
        self.logger.record("curriculum/complete", float(frac >= 1.0))

    def _on_training_start(self) -> None:
        self._apply()

    def _on_step(self) -> bool:
        self._apply()
        return True
