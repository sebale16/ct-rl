# models/actor_q_critic.py
from __future__ import annotations

from copy import deepcopy
from itertools import chain
from typing import Iterable, Optional, Sequence, Type, Dict, Any, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from models.base import Model
from models.actor import (
    DeterministicActor,
    StochasticActor,
    encode_periodic_observations,
    validate_periodic_obs_indices,
)
from common.torch_layers import create_mlp, get_flattened_obs_dim


def _get_action_dim(action_space: spaces.Space) -> int:
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    raise NotImplementedError("Only Box action spaces are currently supported.")


class ActorQCriticModel(Model):
    """
    Actor-Critic models where Critics is the "continuous-time" Q-function: Q = V + q.
    Here Q doesn't represent a physical quantity but instead bears a numerical and function approximation meaning only.
    V and q have different physical unit and meaning. Here V stands for value function
    On the other hand, q has an unit of velocity, rate of advantage or value change in action direction
    Nonetheless, under optimal policy π: q(s, π:(s)) is likely zero, allowing Q to effectively uncover
    both V and q numerically. Consequently, Q still holds great values as an function approximation scheme

    This model can be used for off-policy algorithms like SAC / DDPG / TD3: (Q-ensemble, π).

    - Q ensemble: list of critics {Q_i(s,a)} and their target networks.
    - π: stochastic (SAC) or deterministic (DDPG/TD3) actor.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        q_net_arch: Sequence[int],
        pi_net_arch: Sequence[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        log_std_init: float = -0.5,
        n_critics: int = 2,
        v_net_arch: Optional[Sequence[int]] = None,
        periodic_obs_indices: Optional[Sequence[int]] = None,
        deterministic_policy: bool = False,
        use_actor_target: bool = False,
        device: str = "auto",
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        self.n_critics = int(n_critics)
        self.deterministic_policy = bool(deterministic_policy)
        self.use_actor_target = use_actor_target

        raw_obs_dim = get_flattened_obs_dim(observation_space)
        self.periodic_obs_indices = validate_periodic_obs_indices(
            raw_obs_dim, periodic_obs_indices
        )
        obs_dim = raw_obs_dim + len(self.periodic_obs_indices)
        action_dim = _get_action_dim(action_space)

        # Critics Q_i(s,a)
        q_in_dim = obs_dim + action_dim
        self.q_nets = nn.ModuleList()
        self.q_target_nets = nn.ModuleList()
        for _ in range(self.n_critics):
            q_net = create_mlp(
                input_dim=q_in_dim,
                output_dim=1,
                hidden_dims=list(q_net_arch),
                activation_fn=activation_fn,
            )
            q_target = deepcopy(q_net)
            self.q_nets.append(q_net)
            self.q_target_nets.append(q_target)

        # Actor
        if deterministic_policy:
            self.actor = DeterministicActor(
                observation_space=observation_space,
                action_space=action_space,
                net_arch=pi_net_arch,
                activation_fn=activation_fn,
                periodic_obs_indices=self.periodic_obs_indices,
                squash_output=True,
                device=self.device,
            )
        else:
            self.actor = StochasticActor(
                observation_space=observation_space,
                action_space=action_space,
                net_arch=pi_net_arch,
                activation_fn=activation_fn,
                log_std_init=log_std_init,
                periodic_obs_indices=self.periodic_obs_indices,
                squash_output=True,
                device=self.device,
            )

        # Build the optional value head after the shared actor/critic pieces.
        # This keeps their seeded initialization identical between the plain-MF
        # and MF+V architecture-control arms.
        self.has_v_head = v_net_arch is not None
        if self.has_v_head:
            self.v_net = create_mlp(
                input_dim=obs_dim,
                output_dim=1,
                hidden_dims=list(v_net_arch),
                activation_fn=activation_fn,
            )
            self.v_target_net = deepcopy(self.v_net)
            for p in self.v_target_net.parameters():
                p.requires_grad = False

        self.actor_target = None
        if self.use_actor_target:
            if not self.deterministic_policy:
                raise ValueError(
                    "Actor target network is only supported for deterministic policies."
                )
            self.actor_target = deepcopy(self.actor)
            for param in self.actor_target.parameters():
                param.requires_grad = False

        self.to(self.device)

    def save(self, path: str) -> None:
        """
        Save model parameters to a file.
        This is a custom implementation to save actor and critic separately.
        """
        state_dict = {
            "actor": self.actor.state_dict(),
            "critics": [q.state_dict() for q in self.q_nets],
            "critic_targets": [q.state_dict() for q in self.q_target_nets],
        }
        if self.actor_target:
            state_dict["actor_target"] = self.actor_target.state_dict()
        if self.has_v_head:
            state_dict["v_net"] = self.v_net.state_dict()
            state_dict["v_target_net"] = self.v_target_net.state_dict()
        th.save(state_dict, path)

    def load_state(self, path: str, strict: bool = True) -> None:
        """
        Load parameters from a file into this instance.
        """
        state_dict = th.load(path, map_location=self.device)
        self.actor.load_state_dict(state_dict["actor"], strict=strict)
        for q_net, q_state in zip(self.q_nets, state_dict["critics"]):
            q_net.load_state_dict(q_state, strict=strict)
        for q_target_net, q_target_state in zip(
            self.q_target_nets, state_dict["critic_targets"]
        ):
            q_target_net.load_state_dict(q_target_state, strict=strict)
        if self.has_v_head and "v_net" in state_dict:
            self.v_net.load_state_dict(state_dict["v_net"], strict=strict)
            self.v_target_net.load_state_dict(state_dict["v_target_net"], strict=strict)

    def to(self, device: Union[str, th.device]) -> None:
        """
        Move the model to the specified device.
        """
        self.device = device
        self.q_nets.to(self.device)
        self.q_target_nets.to(self.device)
        self.actor.to(self.device)
        if self.actor_target:
            self.actor_target.to(self.device)
        if getattr(self, "has_v_head", False):
            self.v_net.to(self.device)
            self.v_target_net.to(self.device)

    # ---- Helpers ----

    def _process_obs(self, obs: th.Tensor | np.ndarray) -> th.Tensor:
        if not isinstance(obs, th.Tensor):
            obs = th.as_tensor(obs, dtype=th.float32, device=self.device)
        else:
            obs = obs.to(self.device)
        obs = obs.view(obs.shape[0], -1)
        return encode_periodic_observations(obs, self.periodic_obs_indices)

    def _process_act(self, act: th.Tensor | np.ndarray) -> th.Tensor:
        if not isinstance(act, th.Tensor):
            act = th.as_tensor(act, dtype=th.float32, device=self.device)
        else:
            act = act.to(self.device)
        return act.view(act.shape[0], -1)

    # ------------------------ Q-evaluation ------------------------

    def q_values(
        self, obs: Union[th.Tensor, np.ndarray], act: Union[th.Tensor, np.ndarray]
    ) -> list[th.Tensor]:
        obs_flat = self._process_obs(obs)
        act_flat = self._process_act(act)
        x = th.cat([obs_flat, act_flat], dim=-1)
        return [q_net(x) for q_net in self.q_nets]

    def target_q_values(
        self, obs: Union[th.Tensor, np.ndarray], act: Union[th.Tensor, np.ndarray]
    ) -> list[th.Tensor]:
        obs_flat = self._process_obs(obs)
        act_flat = self._process_act(act)
        x = th.cat([obs_flat, act_flat], dim=-1)
        return [q_net(x) for q_net in self.q_target_nets]

    def min_q(
        self, obs: Union[th.Tensor, np.ndarray], act: Union[th.Tensor, np.ndarray]
    ) -> th.Tensor:
        qs = self.q_values(obs, act)
        stacked = th.stack(qs, dim=0)  # [n_critics, batch, 1]
        return stacked.min(dim=0).values

    def target_min_q(
        self, obs: Union[th.Tensor, np.ndarray], act: Union[th.Tensor, np.ndarray]
    ) -> th.Tensor:
        qs = self.target_q_values(obs, act)
        stacked = th.stack(qs, dim=0)
        return stacked.min(dim=0).values

    # ------------------------ Value head (optional) ------------------------

    def value(self, obs: Union[th.Tensor, np.ndarray]) -> th.Tensor:
        """Scalar state value V(s) from the online value head. Requires the
        model to have been built with ``v_net_arch`` (``has_v_head``)."""
        return self.v_net(self._process_obs(obs))

    def target_value(self, obs: Union[th.Tensor, np.ndarray]) -> th.Tensor:
        """Scalar state value V(s) from the lagged target value head."""
        return self.v_target_net(self._process_obs(obs))

    # ------------------------ Actor ------------------------

    def act(
        self,
        obs: Union[th.Tensor, np.ndarray],
        deterministic: bool = False,
    ) -> tuple[th.Tensor, th.Tensor | None]:
        """
        Returns actions and optional log probabilities.
        For a deterministic policy, log_prob is None.
        """
        if self.deterministic_policy:
            actions = self.actor(obs)
            return actions, None
        else:
            actions, log_prob, _ = self.actor(obs, deterministic=deterministic)
            return actions, log_prob

    def act_target(self, obs: Union[th.Tensor, np.ndarray]) -> th.Tensor:
        if self.actor_target is None:
            raise AttributeError(
                "actor_target is not available. Set use_actor_target=True during model initialization."
            )
        return self.actor_target(obs)

    # ------------------------ Target updates ------------------------

    def soft_update_targets(self, tau: float, update_actor: bool = False) -> None:
        """
        Polyak averaging for target critics.
        If update_actor is True, also update the actor's target network if it exists.
        """
        for q, q_target in zip(self.q_nets, self.q_target_nets):
            for param, target_param in zip(q.parameters(), q_target.parameters()):
                target_param.data.mul_(1.0 - tau)
                target_param.data.add_(tau * param.data)

        if getattr(self, "has_v_head", False):
            for param, target_param in zip(
                self.v_net.parameters(), self.v_target_net.parameters()
            ):
                target_param.data.mul_(1.0 - tau)
                target_param.data.add_(tau * param.data)

        if update_actor and self.actor_target is not None:
            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.mul_(1.0 - tau)
                target_param.data.add_(tau * param.data)

    # ------------------------ Parameter groups ------------------------

    @property
    def critic_parameters(self) -> Iterable[th.nn.Parameter]:
        return [p for q in self.q_nets for p in q.parameters()]

    @property
    def value_parameters(self) -> Iterable[th.nn.Parameter]:
        return list(self.v_net.parameters()) if getattr(self, "has_v_head", False) else []

    @property
    def actor_parameters(self) -> Iterable[th.nn.Parameter]:
        return list(self.actor.parameters())

    @property
    def parameters(self) -> Iterable[th.nn.Parameter]:
        return chain(self.actor_parameters, self.critic_parameters)
