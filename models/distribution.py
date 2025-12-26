# models/distribution.py
from __future__ import annotations

import math
from typing import Optional

import torch as th


class DiagGaussianDistribution:
    """
    Diagonal Gaussian distribution for continuous actions.
    Stateless: methods take mean/log_std explicitly.
    """

    def __init__(self, action_dim: int, epsilon: float = 1e-6) -> None:
        self.action_dim = int(action_dim)
        self.epsilon = float(epsilon)

    def sample(
        self, mean: th.Tensor, log_std: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor]:
        """
        Reparameterized sample: a = mean + std * eps, eps ~ N(0,I).

        Returns (actions, log_prob) with shape:
          actions: [batch, action_dim]
          log_prob: [batch, 1]
        """
        std = th.exp(log_std)
        noise = th.randn_like(mean).to(mean.device)
        actions = mean + std * noise
        log_prob = self.log_prob(actions, mean, log_std)
        return actions, log_prob

    def log_prob(
        self, actions: th.Tensor, mean: th.Tensor, log_std: th.Tensor
    ) -> th.Tensor:
        """
        Log-density of diagonal Gaussian.
        """
        var = th.exp(2.0 * log_std)
        # shape [batch, action_dim]
        log_prob = -0.5 * (
            (actions - mean) ** 2 / (var + self.epsilon)
            + 2.0 * log_std
            + math.log(2.0 * math.pi)
        )
        # sum over action dims -> [batch, 1]
        return log_prob.sum(dim=-1, keepdim=True)

    def entropy(self, log_std: th.Tensor) -> th.Tensor:
        """
        Entropy of diagonal Gaussian.
        """
        entropy_per_dim = 0.5 + 0.5 * math.log(2.0 * math.pi) + log_std
        return entropy_per_dim.sum(dim=-1, keepdim=True)


class SquashedDiagGaussianDistribution(DiagGaussianDistribution):
    """
    tanh-squashed diagonal Gaussian as used in SAC.

    All log-probs are computed in pre-tanh space and corrected by
    the log-Jacobian of tanh.
    """

    def __init__(self, action_dim: int, epsilon: float = 1e-6) -> None:
        super().__init__(action_dim, epsilon=epsilon)

    def sample(
        self, mean: th.Tensor, log_std: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor]:
        std = th.exp(log_std)
        noise = th.randn_like(mean).to(mean.device)
        pre_tanh = mean + std * noise
        actions = th.tanh(pre_tanh)
        log_prob = self._log_prob_from_pre_tanh(pre_tanh, actions, mean, log_std)
        return actions, log_prob

    def log_prob(
        self, actions: th.Tensor, mean: th.Tensor, log_std: th.Tensor
    ) -> th.Tensor:
        # Invert tanh with atanh for given actions.
        eps = self.epsilon
        clipped = th.clamp(actions, -1.0 + eps, 1.0 - eps)
        pre_tanh = 0.5 * (th.log1p(clipped) - th.log1p(-clipped))
        return self._log_prob_from_pre_tanh(pre_tanh, clipped, mean, log_std)

    def _log_prob_from_pre_tanh(
        self,
        pre_tanh: th.Tensor,
        actions: th.Tensor,
        mean: th.Tensor,
        log_std: th.Tensor,
    ) -> th.Tensor:
        # Gaussian log-prob in pre-tanh space
        log_prob_gauss = super().log_prob(pre_tanh, mean, log_std)
        # log |det d tanh / du| = sum log(1 - tanh(u)^2)
        log_det_jac = th.log(1.0 - actions.pow(2) + self.epsilon).sum(
            dim=-1, keepdim=True
        )
        return log_prob_gauss - log_det_jac

    # No closed-form entropy; typically estimated from -log_prob.
    def entropy(self, log_std: th.Tensor) -> th.Tensor:
        raise NotImplementedError(
            "Entropy for squashed Gaussian is not implemented; "
            "use -log_prob estimates if needed."
        )


class StateDependentNoiseDistribution:
    """
    Minimal state-dependent noise helper, similar to SB3's.

    Intended for deterministic policies (DDPG/TD3-style) where we add
    exploratory noise that depends on a learned feature vector of the state.

    Here we implement a simple variant:

      a = tanh(mean_actions + sigma * eps),   eps ~ N(0, I)

    Log-prob/entropy are not defined (not used in deterministic algorithms).
    """

    def __init__(
        self, action_dim: int, noise_std: float = 0.1, squash: bool = True
    ) -> None:
        self.action_dim = int(action_dim)
        self.noise_std = float(noise_std)
        self.squash = bool(squash)

    def sample(self, mean_actions: th.Tensor) -> th.Tensor:
        """
        Add Gaussian noise to mean_actions (Torch tensor) and optionally tanh-squash.
        """
        noise = th.randn_like(mean_actions).to(mean_actions.device) * self.noise_std
        actions = mean_actions + noise
        if self.squash:
            actions = th.tanh(actions)
        return actions

    def log_prob(self, *args, **kwargs) -> th.Tensor:
        raise NotImplementedError(
            "StateDependentNoiseDistribution is intended for deterministic "
            "exploration; log_prob is not defined."
        )

    def entropy(self, *args, **kwargs) -> th.Tensor:
        raise NotImplementedError(
            "StateDependentNoiseDistribution is intended for deterministic "
            "exploration; entropy is not defined."
        )
