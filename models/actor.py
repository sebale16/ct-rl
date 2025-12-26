# models/actor.py
from __future__ import annotations

import math
from typing import Optional, Sequence, Type

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from models.distribution import (
    DiagGaussianDistribution,
    SquashedDiagGaussianDistribution,
)
from common.torch_layers import create_mlp, get_flattened_obs_dim


def squashed_gaussian_log_prob_from_env_actions(
    *,
    actions_env: th.Tensor,
    mean: th.Tensor,
    log_std: th.Tensor,
    dist: SquashedDiagGaussianDistribution,
    action_low: th.Tensor,
    action_high: th.Tensor,
    eps: float = 1e-6,
) -> th.Tensor:
    """
    Log prob for tanh-squashed Gaussian, where `actions_env` are in env Box scale.
    Reuses your distribution's internal `_log_prob_from_pre_tanh(...)`.
    """
    denom = (action_high - action_low).clamp(min=1e-12)
    squashed = 2.0 * (actions_env - action_low) / denom - 1.0
    squashed = squashed.clamp(min=-1.0 + eps, max=1.0 - eps)

    pre_tanh = (
        th.atanh(squashed)
        if hasattr(th, "atanh")
        else 0.5 * (th.log1p(squashed) - th.log1p(-squashed))
    )
    return dist._log_prob_from_pre_tanh(pre_tanh, squashed, mean, log_std)


class StochasticActor(nn.Module):
    """
    Stochastic Gaussian actor for continuous actions.

    By default it uses a tanh-squashed Gaussian and then rescales to the
    action_space bounds (Box). It returns both actions and log-probabilities.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: Sequence[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        log_std_init: float = -0.5,
        squash_output: bool = True,
        device: str = "auto",
    ) -> None:
        super().__init__()
        self.device = (
            th.device("cuda" if th.cuda.is_available() else "cpu")
            if device == "auto"
            else th.device(device)
        )

        assert isinstance(
            action_space, spaces.Box
        ), "StochasticActor currently supports Box action spaces only."
        self.obs_space = observation_space
        self.action_space = action_space
        self.squash_output = squash_output

        obs_dim = get_flattened_obs_dim(observation_space)
        action_dim = int(np.prod(action_space.shape))

        # Body MLP
        hidden_dims = list(net_arch)
        self.body = create_mlp(
            input_dim=obs_dim,
            output_dim=net_arch[-1] if net_arch else obs_dim,
            hidden_dims=hidden_dims,
            activation_fn=activation_fn,
            output_activation=None,
        )

        if len(net_arch) == 0:
            self.net = nn.Identity()
            last_layer_dim = obs_dim
        else:
            hidden_dims = list(net_arch[:-1])
            last_layer_dim = net_arch[-1]
            self.net = create_mlp(
                input_dim=obs_dim,
                output_dim=last_layer_dim,
                hidden_dims=hidden_dims,
                activation_fn=activation_fn,
            )

        self.mu = nn.Linear(last_layer_dim, action_dim)
        self.log_std = nn.Parameter(th.ones(action_dim) * log_std_init)

        # Distribution in [-1, 1]
        self.dist = (
            SquashedDiagGaussianDistribution(action_dim)
            if squash_output
            else DiagGaussianDistribution(action_dim)
        )

        # Store Box bounds as buffers for rescaling
        low = th.as_tensor(action_space.low, dtype=th.float32, device=self.device)
        high = th.as_tensor(action_space.high, dtype=th.float32, device=self.device)
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)

        self.to(self.device)

    def _scale_action(self, action: th.Tensor) -> th.Tensor:
        """
        Map actions in [-1, 1] to Box(low, high).
        For unsquashed Gaussian, this is still a linear rescale, but
        the log-prob is only corrected up to an additive constant.
        """
        # action_low/high: shape [..., action_dim] or broadcastable
        return (
            self.action_low
            + (self.action_high - self.action_low) * (action + 1.0) / 2.0
        )

    def _process_obs(self, obs: th.Tensor | np.ndarray) -> th.Tensor:
        if not isinstance(obs, th.Tensor):
            obs = th.as_tensor(obs, dtype=th.float32, device=self.device)
        else:
            obs = obs.to(self.device)
        return obs.view(obs.shape[0], -1)

    def log_prob(
        self, obs: th.Tensor | np.ndarray, actions: th.Tensor | np.ndarray
    ) -> th.Tensor:
        obs_flat = self._process_obs(obs)
        features = self.body(obs_flat)
        mean = self.mu(features)
        log_std = self.log_std.expand_as(mean)
        # log_std = th.clamp(log_std, -20, 2)  # TODO: Clamp log_std for stability
        act = (
            actions
            if isinstance(actions, th.Tensor)
            else th.as_tensor(actions, device=mean.device, dtype=mean.dtype)
        )
        act = act.view(mean.shape[0], -1)

        if self.squash_output and isinstance(
            self.dist, SquashedDiagGaussianDistribution
        ):
            return squashed_gaussian_log_prob_from_env_actions(
                actions_env=act,
                mean=mean,
                log_std=log_std,
                dist=self.dist,
                action_low=self.action_low,
                action_high=self.action_high,
            )

        return self.dist.log_prob(act, mean, log_std)

    def forward(
        self, obs: th.Tensor | np.ndarray, deterministic: bool = False
    ) -> tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        :param obs: batch of observations
        :param deterministic: if True, use the mean action;
                              otherwise, sample from the policy.
        :return: (actions, log_prob, entropy_estimate_or_None)
        """
        obs_flat = self._process_obs(obs)
        features = self.body(obs_flat)
        mean = self.mu(features)
        log_std = self.log_std.expand_as(mean)

        if deterministic:
            # Use the mean action, and treat log_prob as that of the corresponding Gaussian.
            if isinstance(self.dist, SquashedDiagGaussianDistribution):
                pre_tanh = mean
                squashed = th.tanh(pre_tanh)
                actions = self._scale_action(squashed)
                log_prob = self.dist._log_prob_from_pre_tanh(
                    pre_tanh, squashed, mean, log_std
                )
            else:
                actions = self._scale_action(mean) if self.squash_output else mean
                log_prob = self.dist.log_prob(actions, mean, log_std)
        else:
            # raw_actions in [-1, 1] if squashed
            raw_actions, log_prob = self.dist.sample(mean, log_std)
            actions = (
                self._scale_action(raw_actions) if self.squash_output else raw_actions
            )

        # We don't provide a separate closed-form entropy for the squashed case.
        entropy_est = None
        return actions, log_prob, entropy_est


class DeterministicActor(nn.Module):
    """
    Deterministic actor for DDPG/TD3-style algorithms.
    Exploration is added externally via ActionNoise or StateDependentNoiseDistribution.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: Sequence[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = True,
        device: str = "auto",
    ) -> None:
        super().__init__()
        self.device = (
            th.device("cuda" if th.cuda.is_available() else "cpu")
            if device == "auto"
            else th.device(device)
        )

        assert isinstance(
            action_space, spaces.Box
        ), "DeterministicActor currently supports Box action spaces only."
        self.obs_space = observation_space
        self.action_space = action_space
        self.squash_output = squash_output

        obs_dim = get_flattened_obs_dim(observation_space)
        action_dim = int(np.prod(action_space.shape))

        self.net = create_mlp(
            input_dim=obs_dim,
            output_dim=action_dim,
            hidden_dims=list(net_arch),
            activation_fn=activation_fn,
            output_activation=nn.Tanh if squash_output else None,
        )

        low = th.as_tensor(action_space.low, dtype=th.float32, device=self.device)
        high = th.as_tensor(action_space.high, dtype=th.float32, device=self.device)
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)

        self.to(self.device)

    def _scale_action(self, raw_action: th.Tensor) -> th.Tensor:
        return (
            self.action_low
            + (self.action_high - self.action_low) * (raw_action + 1.0) / 2.0
        )

    def _process_obs(self, obs: th.Tensor | np.ndarray) -> th.Tensor:
        if not isinstance(obs, th.Tensor):
            obs = th.as_tensor(obs, dtype=th.float32, device=self.device)
        else:
            obs = obs.to(self.device)
        return obs.view(obs.shape[0], -1)

    def forward(self, obs: th.Tensor | np.ndarray) -> th.Tensor:
        obs_flat = self._process_obs(obs)
        raw_action = self.net(obs_flat)
        return self._scale_action(raw_action)
