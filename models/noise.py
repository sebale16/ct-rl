# models/noise.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class ActionNoise(ABC):
    """
    Base class for action noise used in DDPG/TD3-style algorithms.
    """

    @abstractmethod
    def __call__(self) -> np.ndarray:
        pass

    def reset(self) -> None:
        """
        Reset any internal state (for stateful noise like OU).
        """
        return


class GaussianActionNoise(ActionNoise):
    """
    Simple i.i.d. Gaussian noise: N(mu, sigma^2).
    """

    def __init__(self, mean: np.ndarray, sigma: np.ndarray) -> None:
        self.mean = np.array(mean, dtype=float)
        self.sigma = np.array(sigma, dtype=float)

    def __call__(self) -> np.ndarray:
        return np.random.normal(self.mean, self.sigma)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"GaussianActionNoise(mean={self.mean}, sigma={self.sigma})"


class OrnsteinUhlenbeckActionNoise(ActionNoise):
    """
    OU noise as in the original DDPG paper.
    dx_t = theta * (mu - x_t) dt + sigma dW_t
    """

    def __init__(
        self,
        mean: np.ndarray,
        sigma: np.ndarray,
        theta: float = 0.15,
        dt: float = 1e-2,
        x0: Optional[np.ndarray] = None,
    ) -> None:
        self.mean = np.array(mean, dtype=float)
        self.sigma = np.array(sigma, dtype=float)
        self.theta = float(theta)
        self.dt = float(dt)
        self.x_prev = (
            np.zeros_like(self.mean) if x0 is None else np.array(x0, dtype=float)
        )

    def __call__(self) -> np.ndarray:
        noise = np.random.normal(size=self.mean.shape)
        dx = (
            self.theta * (self.mean - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * noise
        )
        self.x_prev = self.x_prev + dx
        return self.x_prev

    def reset(self) -> None:
        self.x_prev = np.zeros_like(self.mean)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"OrnsteinUhlenbeckActionNoise(mean={self.mean}, sigma={self.sigma}, "
            f"theta={self.theta}, dt={self.dt})"
        )


class VectorizedActionNoise(ActionNoise):
    def __init__(self, base_noise: ActionNoise, n_envs: int):
        self.noises = [
            base_noise.__class__(**base_noise.__dict__) for _ in range(n_envs)
        ]
        self.n_envs = n_envs

    def __call__(self) -> np.ndarray:
        return np.stack([n() for n in self.noises], axis=0)

    def reset(self, indices=None) -> None:
        if indices is None:
            for n in self.noises:
                n.reset()
        else:
            for i in indices:
                self.noises[i].reset()
