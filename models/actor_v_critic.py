# models/actor_v_critic.py
from __future__ import annotations

from itertools import chain
from typing import Iterable, Optional, Sequence, Type, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from models.base import Model
from models.actor import StochasticActor
from models.actor import squashed_gaussian_log_prob_from_env_actions
from common.torch_layers import create_mlp, get_flattened_obs_dim
from models.distribution import SquashedDiagGaussianDistribution


class ActorVCriticModel(Model):
    """
    Actor-Critic Model for policy gradient algorithms: (V, π).
    Critic only evaluate Value function V rather than Q or q functions.

    When `feature_extractor` is provided, V and π share the same features.
    Otherwise they have separate networks from raw observations.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        v_net_arch: Sequence[int],
        pi_net_arch: Sequence[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        feature_extractor: Optional[nn.Module] = None,
        features_dim: Optional[int] = None,
        log_std_init: float = -0.5,
        device: str = "auto",
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        self.activation_fn = activation_fn

        obs_dim = get_flattened_obs_dim(observation_space)
        action_dim = int(np.prod(action_space.shape))

        self.use_shared_features = feature_extractor is not None

        # Move feature extractor to device
        self.feature_extractor = (
            feature_extractor.to(self.device) if self.use_shared_features else None
        )

        if self.use_shared_features:
            # Shared feature extractor -> features_dim must be known
            if features_dim is None:
                if hasattr(feature_extractor, "features_dim"):
                    features_dim = int(feature_extractor.features_dim)  # type: ignore[arg-type]
                else:
                    raise ValueError(
                        "When using shared feature_extractor, you must provide features_dim "
                        "or use a BaseFeaturesExtractor with a 'features_dim' attribute."
                    )
            self.features_dim = int(features_dim)

            # V head: features -> 1
            self.v_head = create_mlp(
                input_dim=self.features_dim,
                output_dim=1,
                hidden_dims=list(v_net_arch),
                activation_fn=activation_fn,
            )

            # Policy body: features -> latent
            if len(pi_net_arch) == 0:
                self.policy_body = nn.Identity()
                last_dim = self.features_dim
            else:
                hidden_dims = list(pi_net_arch[:-1])
                last_dim = pi_net_arch[-1]
                self.policy_body = create_mlp(
                    input_dim=self.features_dim,
                    output_dim=last_dim,
                    hidden_dims=hidden_dims,
                    activation_fn=activation_fn,
                )

            self.policy_mu = nn.Linear(last_dim, action_dim)
            self.policy_log_std = th.nn.Parameter(
                th.ones(action_dim).to(self.device) * log_std_init
            )

            self.squashed_dist = SquashedDiagGaussianDistribution(action_dim)
            self.action_low = th.as_tensor(
                action_space.low, dtype=th.float32, device=self.device
            )
            self.action_high = th.as_tensor(
                action_space.high, dtype=th.float32, device=self.device
            )
        else:
            # Separate networks directly from raw observations
            self.v_net = create_mlp(
                input_dim=obs_dim,
                output_dim=1,
                hidden_dims=list(v_net_arch),
                activation_fn=activation_fn,
            )
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
        if self.use_shared_features:
            if self.feature_extractor:
                self.feature_extractor.to(self.device)
            self.v_head.to(self.device)
            self.policy_body.to(self.device)
            self.policy_mu.to(self.device)
        else:
            self.v_net.to(self.device)
            self.actor.to(self.device)

    def save(self, path: str) -> None:
        """
        Save model parameters to a file.
        """
        state_dict = {}
        if self.use_shared_features:
            if self.feature_extractor:
                state_dict["feature_extractor"] = self.feature_extractor.state_dict()
            state_dict["v_head"] = self.v_head.state_dict()
            state_dict["policy_body"] = self.policy_body.state_dict()
            state_dict["policy_mu"] = self.policy_mu.state_dict()
            state_dict["policy_log_std"] = self.policy_log_std
        else:
            state_dict["v_net"] = self.v_net.state_dict()
            state_dict["actor"] = self.actor.state_dict()
        th.save(state_dict, path)

    def load_state(self, path: str, strict: bool = True) -> None:
        """
        Load parameters from a file into this instance.
        """
        state_dict = th.load(path, map_location=self.device)
        if self.use_shared_features:
            if self.feature_extractor and "feature_extractor" in state_dict:
                self.feature_extractor.load_state_dict(
                    state_dict["feature_extractor"], strict=strict
                )
            self.v_head.load_state_dict(state_dict["v_head"], strict=strict)
            self.policy_body.load_state_dict(state_dict["policy_body"], strict=strict)
            self.policy_mu.load_state_dict(state_dict["policy_mu"], strict=strict)
            if "policy_log_std" in state_dict:
                with th.no_grad():
                    self.policy_log_std.copy_(state_dict["policy_log_std"])
        else:
            self.v_net.load_state_dict(state_dict["v_net"], strict=strict)
            self.actor.load_state_dict(state_dict["actor"], strict=strict)

    # ------------------------ Helpers ------------------------

    def _process_obs(self, obs: th.Tensor | np.ndarray) -> th.Tensor:
        if not isinstance(obs, th.Tensor):
            obs = th.as_tensor(obs, dtype=th.float32, device=self.device)
        else:
            obs = obs.to(self.device)
        return obs

    def _scale_action(self, raw_action: th.Tensor) -> th.Tensor:
        return (
            self.action_low
            + (self.action_high - self.action_low) * (raw_action + 1.0) / 2.0
        )

    # ------------------------ Public API ------------------------
    def value(self, obs: Union[th.Tensor, np.ndarray]) -> th.Tensor:
        obs_t = self._process_obs(obs)
        if self.use_shared_features:
            features = self.feature_extractor(obs_t)  # type: ignore[operator]
            return self.v_head(features)
        else:
            obs_flat = obs_t.view(obs_t.shape[0], -1)
            return self.v_net(obs_flat)

    def act(
        self, obs: Union[th.Tensor, np.ndarray], deterministic: bool = False
    ) -> tuple[th.Tensor, th.Tensor | None]:
        """
        Returns actions and optional log probabilities.
        """
        obs_t = self._process_obs(obs)
        if self.use_shared_features:
            features = self.feature_extractor(obs_t)  # type: ignore[operator]
            body_out = self.policy_body(features)
            mean = self.policy_mu(body_out)
            log_std = self.policy_log_std.expand_as(mean)

            if deterministic:
                pre_tanh = mean
                squashed = th.tanh(pre_tanh)
                actions = self._scale_action(squashed)
                log_prob = self.squashed_dist._log_prob_from_pre_tanh(
                    pre_tanh, squashed, mean, log_std
                )
            else:
                raw_actions, log_prob = self.squashed_dist.sample(mean, log_std)
                actions = self._scale_action(raw_actions)

            return actions, log_prob
        else:
            actions, log_prob, _ = self.actor(obs_t, deterministic=deterministic)
            return actions, log_prob

    def log_prob(self, obs, actions) -> th.Tensor:
        if not self.use_shared_features:
            return self.actor.log_prob(obs, actions)

        obs_t = self._process_obs(obs)

        actions_t = (
            actions
            if isinstance(actions, th.Tensor)
            else th.as_tensor(actions, dtype=th.float32, device=self.device)
        )
        if actions_t.dim() == 1:
            actions_t = actions_t.unsqueeze(0)
        actions_t = actions_t.to(self.device)

        features = self.feature_extractor(obs_t)  # type: ignore[operator]
        h = self.policy_body(features)
        mean = self.policy_mu(h)
        log_std = self.policy_log_std.expand_as(mean)

        return squashed_gaussian_log_prob_from_env_actions(
            actions_env=actions_t,
            mean=mean,
            log_std=log_std,
            dist=self.squashed_dist,
            action_low=self.action_low,
            action_high=self.action_high,
        )

    # ------------------------ Parameter groups ------------------------

    @property
    def value_parameters(self) -> Iterable[th.nn.Parameter]:
        if self.use_shared_features:
            params = list(self.v_head.parameters())
            if self.feature_extractor is not None:
                params += list(self.feature_extractor.parameters())
            return params
        return self.v_net.parameters()

    @property
    def actor_parameters(self) -> Iterable[th.nn.Parameter]:
        if self.use_shared_features:
            return (
                list(self.policy_body.parameters())
                + list(self.policy_mu.parameters())
                + [self.policy_log_std]
            )
        return self.actor.parameters()

    @property
    def parameters(self) -> Iterable[th.nn.Parameter]:
        return chain(self.value_parameters, self.actor_parameters)
