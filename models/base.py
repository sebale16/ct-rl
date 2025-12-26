# models/base.py
from __future__ import annotations

from abc import abstractmethod
from typing import Union, Tuple, Optional
from typing import Union, Tuple, Optional, Iterable

import torch as th
from gymnasium import spaces
import numpy as np


class Model:
    """
    Model class include a set of neural networks such as actor, critics and/or their target networks
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[str, th.device] = "auto",
    ) -> None:
        if device == "auto":
            device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.device: th.device = th.device(device)
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def act(
        self,
        obs: Union[th.Tensor, np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        """
        Returns actions and optional log probabilities.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model parameters to disk.
        """
        raise NotImplementedError

    @abstractmethod
    def load_state(self, path: str, strict: bool = True) -> None:
        """
        Load parameters from a file into this instance.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self) -> Iterable[th.nn.Parameter]:
        """
        Returns an iterator over all model parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def to(self, device: Union[str, th.device]) -> None:
        """
        Move the model to the specified device.
        """
        raise NotImplementedError
