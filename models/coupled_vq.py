# models/coupled_vq.py
from __future__ import annotations

from itertools import chain
from typing import Iterable, Sequence, Type, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from models.base import Model
from models.actor import StochasticActor
from common.torch_layers import create_mlp, get_flattened_obs_dim


def _get_action_dim(action_space: spaces.Space) -> int:
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    raise NotImplementedError("Only Box action spaces are currently supported.")


class CoupledVqModel(Model):
    """
    Value function and q-function (advantage-rate fct) coupled model: (V, q, π).

    - V(s): state value
    - q(s, a): small-q (rate)
    - π(a|s): stochastic policy
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        v_net_arch: Sequence[int],
        q_net_arch: Sequence[int],
        pi_net_arch: Sequence[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        log_std_init: float = -0.5,
        device: str = "auto",
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )

        obs_dim = get_flattened_obs_dim(observation_space)
        action_dim = _get_action_dim(action_space)

        # V(s)
        self.v_net = create_mlp(
            input_dim=obs_dim,
            output_dim=1,
            hidden_dims=list(v_net_arch),
            activation_fn=activation_fn,
        )

        # q(s, a)
        self.q_net = create_mlp(
            input_dim=obs_dim + action_dim,
            output_dim=1,
            hidden_dims=list(q_net_arch),
            activation_fn=activation_fn,
        )

        # π(a|s)
        self.actor = StochasticActor(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=pi_net_arch,
            activation_fn=activation_fn,
            log_std_init=log_std_init,
            device=self.device,
        )

        self.to(self.device)

    def to(self, device: Union[str, th.device]) -> None:
        """
        Move the model to the specified device.
        """
        self.device = device
        self.v_net.to(self.device)
        self.q_net.to(self.device)
        self.actor.to(self.device)

    def save(self, path: str) -> None:
        """
        Save model parameters to a file.
        """
        state_dict = {
            "v_net": self.v_net.state_dict(),
            "q_net": self.q_net.state_dict(),
            "actor": self.actor.state_dict(),
        }
        th.save(state_dict, path)

    def load_state(self, path: str, strict: bool = True) -> None:
        """
        Load parameters from a file into this instance.
        """
        state_dict = th.load(path, map_location=self.device)
        self.v_net.load_state_dict(state_dict["v_net"], strict=strict)
        self.q_net.load_state_dict(state_dict["q_net"], strict=strict)
        self.actor.load_state_dict(state_dict["actor"], strict=strict)

    # ------------------------ Helpers ------------------------

    def _process_obs(self, obs: th.Tensor | np.ndarray) -> th.Tensor:
        if not isinstance(obs, th.Tensor):
            obs = th.as_tensor(obs, dtype=th.float32, device=self.device)
        else:
            obs = obs.to(self.device)
        return obs.view(obs.shape[0], -1)

    def _process_act(self, act: th.Tensor | np.ndarray) -> th.Tensor:
        if not isinstance(act, th.Tensor):
            act = th.as_tensor(act, dtype=th.float32, device=self.device)
        else:
            act = act.to(self.device)
        return act.view(act.shape[0], -1)

    # ------------------------ Public API ------------------------

    def value(self, obs: Union[th.Tensor, np.ndarray]) -> th.Tensor:
        obs_flat = self._process_obs(obs)
        return self.v_net(obs_flat)

    def rate(
        self, obs: Union[th.Tensor, np.ndarray], act: Union[th.Tensor, np.ndarray]
    ) -> th.Tensor:
        obs_flat = self._process_obs(obs)
        act_flat = self._process_act(act)
        x = th.cat([obs_flat, act_flat], dim=-1)
        return self.q_net(x)

    def act(
        self, obs: Union[th.Tensor, np.ndarray], deterministic: bool = False
    ) -> tuple[th.Tensor, th.Tensor | None]:
        """
        Returns actions and optional log probabilities.
        """
        actions, log_prob, _ = self.actor(obs, deterministic=deterministic)
        return actions, log_prob

    # ------------------------ Parameter groups ------------------------

    @property
    def value_parameters(self) -> Iterable[th.nn.Parameter]:
        return self.v_net.parameters()

    @property
    def rate_parameters(self) -> Iterable[th.nn.Parameter]:
        return self.q_net.parameters()

    @property
    def actor_parameters(self) -> Iterable[th.nn.Parameter]:
        return self.actor.parameters()

    @property
    def parameters(self) -> Iterable[th.nn.Parameter]:
        return chain(self.value_parameters, self.rate_parameters, self.actor_parameters)
