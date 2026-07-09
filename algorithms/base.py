# algorithms/base.py
from __future__ import annotations

import io
import pathlib
import time
from collections import deque
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, Union, ClassVar

import torch as th
from torch.optim import Optimizer

from common.callbacks import BaseCallback
from environment.base import ContinuousEnv
from environment.vec_env import VecContinuousEnv
from models.base import Model
from models import ActorQCriticModel, ActorVCriticModel, CoupledVqModel
from common.schedules import Schedule, make_schedule
from common.logger import dump, logger, safe_mean
from common.utils import get_device, set_seed


class BaseAlgorithm(ABC):
    """
    Algorithm base class.

    Handles:
      - device & seeding
      - LR schedule based on progress_remaining in [1, 0]
      - global timestep counter

    Concrete algorithms only need to:
      - set up their models, optimizers
      - implement learn() in off/on-policy subclasses
    """

    _MODEL_REGISTRY: ClassVar[Dict[str, Type[Model]]] = {
        "ActorQCriticModel": ActorQCriticModel,
        "ActorVCriticModel": ActorVCriticModel,
        "CoupledVqModel": CoupledVqModel,
    }

    def __init__(
        self,
        env: Union[ContinuousEnv, VecContinuousEnv],
        model: Union[Model, str, Type[Model]],
        model_kwargs: Optional[Dict[str, Any]] = None,
        device: Union[str, th.device] = "auto",
        seed: Optional[int] = None,
        learning_rate: Union[float, Schedule] = 3e-4,
    ) -> None:
        self.env = env
        self.n_envs = getattr(env, "num_envs", 1)
        self.is_vec_env = isinstance(env, VecContinuousEnv)
        self.dt_default = (
            self.env.dt_default if hasattr(self.env, "dt_default") else 1.0
        )

        self.time_rescale = 1.0 / self.dt_default

        self.model_kwargs = model_kwargs or {}
        self.model: Model
        self.device: th.device = get_device(device)
        self.seed = seed

        self._setup_model(model, self.model_kwargs)

        # Configure logger
        self.logger = logger

        if seed is not None:
            set_seed(seed)
            # Gymnasium-style API
            try:
                self.env.reset(seed=seed)
            except TypeError:
                # older envs may not accept seed kwarg
                pass

        # Schedule: progress_remaining -> lr
        self.lr_schedule: Schedule = make_schedule(learning_rate)

        # Training progress
        self.num_timesteps: int = 0
        self._total_timesteps: int = 0
        self._progress_remaining: float = 1.0

        # Algorithms will populate this with optimizers to be scheduled
        self.optimizers: List[Optimizer] = []

        # For logging
        self.ep_info_buffer: deque[Dict[str, float]] = deque(maxlen=100)

    def _setup_model(
        self, model: Union[str, Type[Model]], model_kwargs: Dict[str, Any]
    ) -> None:
        """
        Instantiate the model.
        """
        if isinstance(model, Model):
            self.model = model
        else:
            if model_kwargs is None:
                model_kwargs = {}
            if isinstance(model, str):
                model_class = self._MODEL_REGISTRY[model]
            else:
                model_class = model
            if model_class is not None:
                try:
                    self.model = model_class(
                        observation_space=self.env.observation_space,
                        action_space=self.env.action_space,
                        **model_kwargs,
                    )
                except TypeError:
                    raise ValueError("Model class or model arguments are invalid.")

        self.model.to(device=self.device)

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback],
    ) -> Tuple[int, Optional[BaseCallback]]:
        """
        Shared setup for the learn method.
        """
        self._total_timesteps = total_timesteps
        # When resuming from a checkpoint the timestep counter is restored by
        # common.checkpoint.load_checkpoint and must survive setup; otherwise
        # start a fresh run from zero.
        if not getattr(self, "_resumed_from_checkpoint", False):
            self.num_timesteps = 0
        self.start_time = time.time()
        if callback:
            callback.init_callback(self)

        return total_timesteps, callback

    # ------------------------- Progress / LR -------------------------

    def _update_progress_remaining(self) -> None:
        if self._total_timesteps <= 0:
            self._progress_remaining = 1.0
        else:
            frac = float(self.num_timesteps) / float(self._total_timesteps)
            self._progress_remaining = max(0.0, 1.0 - frac)

    def _update_learning_rate(self) -> None:
        if not self.optimizers:
            return

        current_lr = self.lr_schedule(self._progress_remaining)
        for opt in self.optimizers:
            for pg in opt.param_groups:
                pg["lr"] = current_lr
        self.logger.record("train/learning_rate", current_lr)

    def _update_info_buffer(self, infos):
        """
        Update the episode info buffer with stats from the environment.
        """
        if infos is None:
            return
        if isinstance(infos, (list, tuple)):
            # Vectorized env case
            for info in infos:
                if isinstance(info, dict) and "episode" in info:
                    self.ep_info_buffer.append(info["episode"])
        else:
            # Single env case where infos is a dict
            if "episode" in infos:
                self.ep_info_buffer.append(infos["episode"])

    def _log_stats(self, log_interval: int) -> None:
        """
        Log training statistics (FPS, rewards, etc).
        """
        if self.num_timesteps > 0 and self.num_timesteps % log_interval == 0:
            fps = int(self.num_timesteps / (time.time() - self.start_time))
            self.logger.record("time/fps", fps)
            self.logger.record(
                "time/time_elapsed",
                int(time.time() - self.start_time),
                exclude="tensorboard",
            )
            self.logger.record(
                "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
            )
            if len(self.ep_info_buffer) > 0:
                self.logger.record(
                    "rollout/ep_rew_mean",
                    safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
                )
                self.logger.record(
                    "rollout/ep_len_mean",
                    safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
                )
            dump(step=self.num_timesteps)

    # ---------------------------- API ----------------------------

    @abstractmethod
    def learn(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 100,
    ) -> "BaseAlgorithm":
        """
        The main training loop. This is implemented in subclasses.
        """
        raise NotImplementedError

    def save(self, path: Union[str, pathlib.Path, io.BufferedIOBase]) -> None:
        """
        Save the model and replay buffer.
        """
        # Delegate saving to the model
        self.model.save(path)

    def load(
        self, path: Union[str, pathlib.Path, io.BufferedIOBase], strict: bool = True
    ) -> "BaseAlgorithm":
        """
        Load the model and replay buffer.
        """
        # Delegate loading to the model
        self.model.load_state(path, strict=strict)

        return self
