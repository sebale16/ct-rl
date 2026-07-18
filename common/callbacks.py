# common/callbacks.py

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, List, Union, TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm
from common.logger import dump

# from common.logger import get_logger
from evaluations.evaluation_helpers import evaluate_policy_per_episode

if TYPE_CHECKING:
    from algorithms.base import BaseAlgorithm


LocalsDict = dict[str, Any]
GlobalsDict = dict[str, Any]

MaybeCallback = Union[
    "BaseCallback",
    List["BaseCallback"],
    Callable[[LocalsDict, GlobalsDict], bool],
    None,
]


class BaseCallback(ABC):
    """
    Minimal base callback API for continuous-time RL training loops.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__()
        self.verbose = verbose
        # Number of time the callback was called
        self.n_calls: int = 0
        # n_envs * n times env.step() was called
        self.num_timesteps: int = 0
        self.algorithm: Optional[BaseAlgorithm] = None
        self.locals: LocalsDict = {}
        self.globals: GlobalsDict = {}

    def init_callback(self, algorithm: Any) -> None:
        """
        Called once before training. Stores a reference to the algorithm.
        """
        self.algorithm = algorithm
        self._init_callback()

    @property
    def logger(self) -> Any:
        """
        Getter for the logger object.
        """
        # In our implementation, the logger is part of the algorithm.
        return self.algorithm.logger

    def _init_callback(self) -> None:
        pass

    def on_training_start(self, locals_: LocalsDict, globals_: GlobalsDict) -> None:
        self.locals = locals_
        self.globals = globals_
        # Update num_timesteps in case training was done before
        self.num_timesteps = self.algorithm.num_timesteps
        self._on_training_start()

    def _on_training_start(self) -> None:
        pass

    def on_step(self) -> bool:
        """
        Called after every environment step (or aggregated time-step).
        """
        self.n_calls += 1
        self.num_timesteps = self.algorithm.num_timesteps
        return self._on_step()

    @abstractmethod
    def _on_step(self) -> bool:
        """
        Return False to request early stopping.
        """
        return True

    def on_training_end(self) -> None:
        self._on_training_end()

    def _on_training_end(self) -> None:
        pass

    def on_rollout_start(self) -> None:
        self._on_rollout_start()

    def _on_rollout_start(self) -> None:
        pass

    def on_rollout_end(self) -> None:
        self._on_rollout_end()

    def _on_rollout_end(self) -> None:
        pass

    def update_locals(self, locals_: LocalsDict) -> None:
        self.locals.update(locals_)
        self.update_child_locals(locals_)

    def update_child_locals(self, locals_: LocalsDict) -> None:
        pass


class CallbackList(BaseCallback):
    """
    Chain several callbacks together.
    """

    def __init__(self, callbacks: List[BaseCallback]) -> None:
        super().__init__()
        self.callbacks = list(callbacks)

    def _init_callback(self) -> None:
        for cb in self.callbacks:
            cb.init_callback(self.algorithm)

    def _on_training_start(self) -> None:
        for cb in self.callbacks:
            cb.on_training_start(self.locals, self.globals)

    def _on_rollout_start(self) -> None:
        for cb in self.callbacks:
            cb.on_rollout_start()

    def _on_step(self) -> bool:
        cont = True
        for cb in self.callbacks:
            cont = cb.on_step() and cont
        return cont

    def _on_rollout_end(self) -> None:
        for cb in self.callbacks:
            cb.on_rollout_end()

    def _on_training_end(self) -> None:
        for cb in self.callbacks:
            cb.on_training_end()


class ConvertCallback(BaseCallback):
    """
    Wraps a simple function callback(locals, globals) -> bool into BaseCallback.
    """

    def __init__(
        self,
        callback: Optional[Callable[[LocalsDict, GlobalsDict], bool]],
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.callback = callback

    def _on_step(self) -> bool:
        if self.callback is None:
            return True
        return bool(self.callback(self.locals, self.globals))


class EventCallback(BaseCallback):
    """
    Wraps a child callback and allows derived callbacks to "trigger an event"
    and run that child callback
    """

    def __init__(self, callback: Optional[BaseCallback] = None, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.callback = callback

    def _init_callback(self) -> None:
        if self.callback is not None:
            self.callback.parent = self
            self.callback.init_callback(self.algorithm)

    def update_child_locals(self, locals_: LocalsDict) -> None:
        if self.callback is not None:
            self.callback.update_locals(locals_)

    def _on_event(self) -> bool:
        if self.callback is None:
            return True
        return self.callback.on_step()

    def _on_step(self) -> bool:
        # default: do nothing; subclasses decide when to trigger _on_event()
        return True


class EveryNTimesteps(EventCallback):
    """
    Trigger a child callback every `n_steps` timesteps.
    """

    def __init__(self, n_steps: int, callback: BaseCallback, verbose: int = 0):
        super().__init__(callback=callback, verbose=verbose)
        assert n_steps > 0
        self.n_steps = int(n_steps)
        self._last_trigger_timesteps = 0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_trigger_timesteps) >= self.n_steps:
            self._last_trigger_timesteps = self.num_timesteps
            return self._on_event()
        return True


class LogEveryNTimesteps(EveryNTimesteps):
    """
    Log data every ``n_steps`` timesteps.
    Requires your training loop to call `callback.update_locals(locals())`

    :param n_steps: Number of timesteps between two trigger.
    """

    def __init__(self, n_steps: int, verbose: int = 0):
        # ConvertCallback will call _log_data(locals, globals)
        super().__init__(
            n_steps=n_steps, callback=ConvertCallback(self._log_data), verbose=verbose
        )

    def _log_data(self, _locals: LocalsDict, _globals: GlobalsDict) -> bool:
        """
        Called when the event triggers.
        """
        dump(step=self.num_timesteps)
        return True


class CheckpointCallback(BaseCallback):
    """
    Callback for saving a model every `save_freq` steps
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self._last_save_timesteps = 0

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_save_timesteps) >= self.save_freq:
            self._last_save_timesteps = self.num_timesteps
            path = os.path.join(
                self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pth"
            )
            self.algorithm.save(path)  # type: ignore[union-attr]
            if self.verbose > 0:
                print(f"Saving model checkpoint to {path}")
        return True


class WallClockCheckpointCallback(BaseCallback):
    """
    Save a full, resumable training checkpoint when the job approaches its wall
    time (or receives a termination signal), then stop the loop gracefully so a
    resubmission chain can pick up exactly where it left off.

    A checkpoint is triggered when either:
      - elapsed wall-clock time since training start exceeds ``max_seconds``, or
      - the process receives SIGTERM / SIGUSR1 (Slurm's pre-timeout signal, if
        the job is submitted with e.g. ``--signal=B:TERM@300``).

    On trigger it calls :func:`common.checkpoint.save_checkpoint` (model + replay
    buffer + optimizers + counters + entropy temperature + RNG + caller extra)
    and returns ``False`` so ``learn`` exits cleanly. ``extra_state_fn`` supplies
    the dict of auxiliary state to persist (used to carry the EvalCallback's
    best-reward and eval history across the resume).

    ``stopped`` is set True iff a checkpoint was written because of this callback,
    letting the runner distinguish "paused for wall time" from "finished".
    """

    def __init__(
        self,
        ckpt_dir: str,
        max_seconds: float,
        extra_state_fn: Optional[Callable[[], dict]] = None,
        catch_signals: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.ckpt_dir = ckpt_dir
        self.max_seconds = float(max_seconds)
        self.extra_state_fn = extra_state_fn
        self.catch_signals = bool(catch_signals)
        self.stopped = False
        self._start_time = 0.0
        self._signal_received = False

    def _on_training_start(self) -> None:
        import time as _time

        self._start_time = _time.monotonic()
        if self.catch_signals:
            import signal

            def _handler(signum, frame):
                self._signal_received = True

            for sig in (signal.SIGTERM, signal.SIGUSR1):
                try:
                    signal.signal(sig, _handler)
                except (ValueError, OSError):
                    # Not in the main thread, or signal unavailable; time budget
                    # remains as the trigger.
                    pass

    def _should_checkpoint(self) -> bool:
        import time as _time

        if self._signal_received:
            return True
        return (_time.monotonic() - self._start_time) >= self.max_seconds

    def _on_step(self) -> bool:
        if self.stopped:
            return False
        if not self._should_checkpoint():
            return True

        from common.checkpoint import save_checkpoint

        extra = {}
        if self.extra_state_fn is not None:
            try:
                extra = self.extra_state_fn() or {}
            except Exception as e:  # never let extra-collection abort the save
                if self.verbose > 0:
                    print(f"[WallClockCheckpoint] extra_state_fn failed: {e}")
                extra = {}

        reason = "signal" if self._signal_received else "wall-time budget"
        if self.verbose > 0:
            print(
                f"\n[WallClockCheckpoint] {reason} reached at "
                f"{self.num_timesteps} steps; saving checkpoint to {self.ckpt_dir}",
                flush=True,
            )
        save_checkpoint(self.algorithm, self.ckpt_dir, extra)
        if self.verbose > 0:
            print("[WallClockCheckpoint] checkpoint written; stopping.", flush=True)
        self.stopped = True
        return False


class ProgressBarCallback(BaseCallback):
    """
    Display a progress bar using tqdm.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.pbar = None
        self._last_num_timesteps = 0

    def _on_training_start(self) -> None:
        total = int(getattr(self.algorithm, "_total_timesteps", 0))
        self.pbar = tqdm(total=total)
        self._last_num_timesteps = int(getattr(self.algorithm, "num_timesteps", 0))

    def _on_step(self) -> bool:
        if self.pbar is None:
            return True
        cur = int(getattr(self.algorithm, "num_timesteps", self.num_timesteps))
        delta = max(cur - self._last_num_timesteps, 0)
        if delta > 0:
            self.pbar.update(delta)
        self._last_num_timesteps = cur
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


class EvalCallback(EventCallback):
    """
    Periodically evaluate on a separate eval env (single or VecContinuousEnv),
    log eval metrics, and optionally save the best model.

    Logs:
      - eval/mean_reward, eval/std_reward
      - eval/mean_ep_length, eval/std_ep_length
      - eval/best_mean_reward
    """

    def __init__(
        self,
        eval_env: Any,
        *,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        deterministic: bool = True,
        reset_seed: Optional[int] = None,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        verbose: int = 1,
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
    ):
        super().__init__(callback=callback_after_eval, verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.deterministic = bool(deterministic)
        self.reset_seed = None if reset_seed is None else int(reset_seed)

        self.log_path = log_path
        self.best_model_save_path = best_model_save_path
        self.callback_on_new_best = callback_on_new_best

        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self._last_eval_timesteps = 0

        self.evaluations_timesteps: List[int] = []
        self.evaluations_results: List[np.ndarray] = []
        self.evaluations_lengths: List[np.ndarray] = []

    def _init_callback(self) -> None:
        super()._init_callback()  # init callback_after_eval
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(self.log_path, exist_ok=True)

        if self.callback_on_new_best is not None:
            self.callback_on_new_best.parent = self
            self.callback_on_new_best.init_callback(self.algorithm)

    def _log_eval(self, mean_reward: float, std_reward: float, mean_len: float) -> None:
        self.logger.record("eval/mean_reward", float(mean_reward))
        self.logger.record("eval/std_reward", float(std_reward))
        self.logger.record("eval/mean_ep_length", float(mean_len))
        self.logger.record(
            "time/total_timesteps", int(self.num_timesteps), exclude="tensorboard"
        )
        dump(step=self.num_timesteps)

    def _save_evals(self, rewards: np.ndarray, lengths: np.ndarray) -> None:
        if self.log_path is None:
            return
        self.evaluations_timesteps.append(int(self.num_timesteps))
        self.evaluations_results.append(np.asarray(rewards, dtype=float))
        self.evaluations_lengths.append(np.asarray(lengths, dtype=int))
        np.savez(
            os.path.join(self.log_path, "evaluations.npz"),
            timesteps=np.asarray(self.evaluations_timesteps, dtype=int),
            results=np.asarray(self.evaluations_results, dtype=object),
            ep_lengths=np.asarray(self.evaluations_lengths, dtype=object),
        )

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True

        if (self.num_timesteps - self._last_eval_timesteps) < self.eval_freq:
            return True

        self._last_eval_timesteps = self.num_timesteps

        rewards, lengths = evaluate_policy_per_episode(
            model=self.algorithm.model,
            env=self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
            reset_seed=self.reset_seed,
        )

        rewards = np.asarray(rewards, dtype=float)
        lengths = np.asarray(lengths, dtype=int)

        mean_reward = float(np.mean(rewards)) if rewards.size else -np.inf
        std_reward = float(np.std(rewards)) if rewards.size else 0.0
        mean_len = float(np.mean(lengths)) if lengths.size else 0.0

        self.last_mean_reward = mean_reward
        self._save_evals(rewards, lengths)
        self._log_eval(mean_reward, std_reward, mean_len)

        continue_training = True

        # New best model save
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            if self.best_model_save_path is not None:
                path = os.path.join(self.best_model_save_path, "best_model.pth")
                self.algorithm.save(path)
                if self.verbose > 0:
                    print(f"New best mean reward={mean_reward:.3f}. Saving to {path}")

            if self.callback_on_new_best is not None:
                self.callback_on_new_best.update_locals(self.locals)
                continue_training = self.callback_on_new_best.on_step()

        # After-eval callback
        if continue_training:
            continue_training = self._on_event()

        return continue_training


class StopTrainingOnRewardThreshold(BaseCallback):
    """
    Use with EvalCallback, typically as callback_on_new_best.
    Stops when parent.best_mean_reward >= reward_threshold.
    """

    def __init__(self, reward_threshold: float, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.reward_threshold = float(reward_threshold)

    def _on_step(self) -> bool:
        assert (
            self.parent is not None
        ), "StopTrainingOnRewardThreshold must be used as a child callback of EvalCallback"
        best = float(getattr(self.parent, "best_mean_reward", -np.inf))
        if best >= self.reward_threshold:
            if self.verbose > 0:
                print(
                    f"Stopping training because best_mean_reward={best:.3f} >= {self.reward_threshold:.3f}"
                )
            return False
        return True


class StopTrainingOnNoModelImprovement(BaseCallback):
    """
    Use with EvalCallback, typically as callback_after_eval.
    Stops if there is no new best for `max_no_improvement_evals` evals (after `min_evals` evals).
    """

    def __init__(
        self, max_no_improvement_evals: int, min_evals: int = 0, verbose: int = 0
    ):
        super().__init__(verbose=verbose)
        self.max_no_improvement_evals = int(max_no_improvement_evals)
        self.min_evals = int(min_evals)

        self._evals = 0
        self._best_so_far = -np.inf
        self._no_improve_count = 0

    def _on_step(self) -> bool:
        assert (
            self.parent is not None
        ), "StopTrainingOnNoModelImprovement must be used as a child callback of EvalCallback"
        self._evals += 1

        best = float(getattr(self.parent, "best_mean_reward", -np.inf))
        if best > self._best_so_far:
            self._best_so_far = best
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1

        if (
            self._evals >= self.min_evals
            and self._no_improve_count >= self.max_no_improvement_evals
        ):
            if self.verbose > 0:
                print(
                    f"Stopping training: no new best for {self._no_improve_count} evals "
                    f"(best={self._best_so_far:.3f})."
                )
            return False

        return True


class StopTrainingOnMaxEpisodes(BaseCallback):
    """
    Optional. Counts completed episodes by reading `dones` from callback.locals.
    Requires: your learn loop calls callback.update_locals(locals()) before callback.on_step().
    """

    def __init__(self, max_episodes: int, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.max_episodes = int(max_episodes)
        self._episode_count = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("done", None)
        if dones is None:
            dones = self.locals.get("dones", None)
        if dones is None:
            return True

        dones_arr = np.asarray(dones, dtype=bool).reshape(-1)
        self._episode_count += int(dones_arr.sum())

        if self._episode_count >= self.max_episodes:
            if self.verbose > 0:
                print(f"Stopping training: reached max_episodes={self.max_episodes}")
            return False
        return True


def convert_callback(callback: MaybeCallback) -> "BaseCallback":
    """
    Helper to implicitly convert a different type (such as list or callable) to callback type
      - None -> empty CallbackList
      - list -> CallbackList(list)
      - callable -> ConvertCallback(callable)
      - BaseCallback -> stay the same
    """
    if callback is None:
        return CallbackList([])
    if isinstance(callback, list):
        return CallbackList(callback)
    if callable(callback):
        return ConvertCallback(callback)
    return callback
