# common/torch_layers.py

from __future__ import annotations

from typing import List, Optional, Union, Dict, Type

import torch as th
import torch.nn as nn
import gymnasium as gym

from gymnasium import spaces
from .utils import get_flattened_obs_dim, get_device


class BaseFeaturesExtractor(nn.Module):
    """
    Base class for feature extractors.

    :param observation_space: Gymnasium observation space
    :param features_dim: Number of output features
    """

    def __init__(self, observation_space: gym.Space, features_dim: int) -> None:
        super().__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = int(features_dim)

    @property
    def features_dim(self) -> int:
        return self._features_dim


class FlattenExtractor(BaseFeaturesExtractor):
    """
    Simple feature extractor that just flattens the input.
    """

    def __init__(self, observation_space: gym.Space) -> None:
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.flatten(observations)


def create_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    output_activation: Optional[Type[nn.Module]] = None,
) -> nn.Sequential:
    """
    Build a standard MLP as nn.Sequential.

    :param input_dim: Input dimension
    :param output_dim: Output dimension (last linear layer)
    :param hidden_dims: List of hidden layer sizes
    :param activation_fn: Activation for hidden dimensions
    :param output_activation: Optional activation after final layer
    """
    layers: List[nn.Module] = []
    last_dim = input_dim

    for h in hidden_dims:
        layers.append(nn.Linear(last_dim, h))
        layers.append(activation_fn())
        last_dim = h

    if output_dim > 0:
        layers.append(nn.Linear(last_dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation())

    return nn.Sequential(*layers)


class MlpExtractor(nn.Module):
    """
    Minimal actor/critic MLP extractor.

    `net_arch` can be:
      - List[int]: same architecture for policy and value.
      - Dict: {"pi": [...], "vf": [...]} for different policy/value nets.
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module] = nn.ReLU,
        device: Union[str, th.device] = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)

        if isinstance(net_arch, dict):
            pi_layers = net_arch.get("pi", [])
            vf_layers = net_arch.get("vf", [])
        else:
            pi_layers = vf_layers = net_arch

        # Build policy MLP
        policy_modules: List[nn.Module] = []
        last_dim_pi = feature_dim
        for h in pi_layers:
            policy_modules.append(nn.Linear(last_dim_pi, h))
            policy_modules.append(activation_fn())
            last_dim_pi = h

        # Build value MLP
        value_modules: List[nn.Module] = []
        last_dim_vf = feature_dim
        for h in vf_layers:
            value_modules.append(nn.Linear(last_dim_vf, h))
            value_modules.append(activation_fn())
            last_dim_vf = h

        self.latent_dim_pi = last_dim_pi
        self.latent_dim_vf = last_dim_vf

        self.policy_net = nn.Sequential(*policy_modules).to(device)
        self.value_net = nn.Sequential(*value_modules).to(device)

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)
