# common/sb3_demo.py
"""Demonstration-seeded exploration for SB3's off-policy algorithms.

Mirrors ``algorithms.ct_sac.CTSAC``/``algorithms.ct_td3.CTTD3``'s
``demonstration_policy``/``demonstration_steps`` replay-buffer warm start
(see ``common.demonstration``) for Stable-Baselines3's SAC/TD3: for the
first ``demonstration_steps`` calls to ``_sample_action`` (default: matching
``learning_starts``), defer to ``demonstration_policy(obs)`` instead of
SB3's own random-before-``learning_starts`` / trained-policy-after
exploration.

Unlike the CT algorithms, SB3's constructors take a closed, library-defined
kwarg list, so there is no ``demonstration_policy=...`` constructor
parameter to add. Instead, ``demo_seeded_class`` builds a small subclass on
demand and the caller sets ``model.demonstration_policy`` /
``model.demonstration_steps`` as plain attributes after construction (or
after ``AlgoClass.load(...)`` on resume) -- the mixin reads them with
``getattr`` defaults, so an instance with neither set behaves exactly like
the unmodified algorithm.
"""

from __future__ import annotations

from typing import Type

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm


class DemoSeededOffPolicyMixin:
    """Mixed into an SB3 off-policy algorithm class; see module docstring."""

    demonstration_policy = None
    demonstration_steps = 0

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: ActionNoise | None = None,
        n_envs: int = 1,
    ):
        policy = self.demonstration_policy
        if policy is None or self.num_timesteps >= self.demonstration_steps:
            return super()._sample_action(learning_starts, action_noise, n_envs)

        assert self._last_obs is not None, "self._last_obs was not set"
        obs = np.asarray(self._last_obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        unscaled_action = np.stack(
            [np.asarray(policy(obs[i]), dtype=np.float32) for i in range(obs.shape[0])],
            axis=0,
        )
        unscaled_action = np.clip(
            unscaled_action, self.action_space.low, self.action_space.high
        )

        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip (unreachable for the
            # continuous-control envs this is used with, kept for parity with
            # OffPolicyAlgorithm._sample_action's own fallback).
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action


_DEMO_SEEDED_CLASS_CACHE: dict[Type[OffPolicyAlgorithm], Type[OffPolicyAlgorithm]] = {}


def demo_seeded_class(algo_class: Type[OffPolicyAlgorithm]) -> Type[OffPolicyAlgorithm]:
    """Return a ``DemoSeededOffPolicyMixin``-mixed subclass of ``algo_class``.

    Cached per base class so repeated calls (e.g. one per training run in a
    sweep) don't accumulate duplicate classes.
    """
    cached = _DEMO_SEEDED_CLASS_CACHE.get(algo_class)
    if cached is None:
        cached = type(
            f"DemoSeeded{algo_class.__name__}",
            (DemoSeededOffPolicyMixin, algo_class),
            {},
        )
        _DEMO_SEEDED_CLASS_CACHE[algo_class] = cached
    return cached
