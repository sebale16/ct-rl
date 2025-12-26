# environment/monitor.py
from __future__ import annotations

import time
from collections import deque
from typing import Dict, Optional, SupportsFloat, Any, Tuple

import numpy as np
from gymnasium.core import Wrapper

from common.logger import record_mean, get_logger
from .base import ContinuousEnv


class Monitor(Wrapper):
    """
    A monitor wrapper for continuous-time environments.
    It keeps track of episode returns and lengths.
    """

    def __init__(
        self,
        env: ContinuousEnv,
        filename: Optional[str] = None,
        info_keywords: Tuple[str, ...] = (),
    ):
        super().__init__(env)
        self.dt_default = self.env.dt_default
        self.t_start = time.time()
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_count = 0
        self.needs_reset = True

        self.info_keywords = info_keywords
        self.rewards = []

    def step(
        self, action: Any
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Step the environment and log episode info. This is for discrete-time compatibility.
        """
        raise NotImplementedError("Use step_dt for continuous-time environments.")

    def step_dt(self, action: np.ndarray) -> tuple:
        if self.needs_reset:
            raise RuntimeError("Call reset before using step_dt")

        obs_t, t, _, reward, next_obs, next_t, terminated, truncated, info = (
            self.env.step_dt(action)
        )
        self.rewards.append(reward)

        done = terminated or truncated
        if done:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {
                "r": round(ep_rew, 6),
                "l": ep_len,
                "t": round(time.time() - self.t_start, 6),
            }
            info["episode"] = ep_info

        return obs_t, t, _, reward, next_obs, next_t, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        self.needs_reset = False
        self.rewards = []
        self.episode_count += 1
        return self.env.reset(**kwargs)
