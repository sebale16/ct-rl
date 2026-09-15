"""Oracle-only HJB fitted value flow and analytic Acrobot control.

This is the value-based Stage A experiment, not the existing Q-based CT-SAC
trainer. Only the return-value network is trained. There is no IDA-PBC layer.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from torch import nn

from models.acrobot_oracle import AcrobotOracle, UprightReward


@dataclass(frozen=True)
class ValueFlowConfig:
    discount_rate: float = 0.1
    value_step: float = 0.02
    learning_rate: float = 3e-4
    target_rate: float = 0.01
    target_interval: int = 1
    temperature: float = 0.0
    auto_temperature: bool = False
    temperature_learning_rate: float = 3e-4
    target_entropy: float = -1.0
    temperature_min: float = 1e-4
    temperature_max: float = 10.0
    quadrature_points: int = 128
    hidden_width: int = 64
    momentum_scale: float = 10.0
    grad_clip: float = 10.0

    def __post_init__(self):
        for key in ("discount_rate", "value_step", "learning_rate", "momentum_scale", "grad_clip"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not math.isfinite(self.target_rate) or not 0 < self.target_rate <= 1:
            raise ValueError("target_rate must be in (0,1]")
        if self.target_interval < 1:
            raise ValueError("target_interval must be at least one")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if not math.isfinite(self.temperature_learning_rate) or self.temperature_learning_rate <= 0:
            raise ValueError("temperature_learning_rate must be finite and positive")
        if not math.isfinite(self.target_entropy) or self.target_entropy >= math.log(2):
            raise ValueError("target_entropy must be finite and below log(2), for density on [-1,1]")
        if not (math.isfinite(self.temperature_min) and math.isfinite(self.temperature_max)
                and 0 < self.temperature_min < self.temperature_max):
            raise ValueError("temperature bounds must be finite with 0 < min < max")
        if self.auto_temperature and not self.temperature_min <= self.temperature <= self.temperature_max:
            raise ValueError("automatic temperature requires a positive initial temperature within its bounds")
        if self.hidden_width < 1 or self.quadrature_points < 8:
            raise ValueError("hidden_width must be positive and quadrature_points at least 8")


class AcrobotValue(nn.Module):
    def __init__(self, config: ValueFlowConfig):
        super().__init__()
        self.momentum_scale = config.momentum_scale
        self.anchor = config.temperature == 0
        self.net = nn.Sequential(nn.Linear(6, config.hidden_width), nn.Tanh(),
                                 nn.Linear(config.hidden_width, config.hidden_width), nn.Tanh(),
                                 nn.Linear(config.hidden_width, 1))
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def _features(self, z):
        q, p = z[..., :2], z[..., 2:] / self.momentum_scale
        return torch.cat((q.sin(), q.cos(), p), dim=-1)

    def forward(self, z):
        # The nominal plant and task are invariant under reflection about
        # upright with momentum reversal. Enforce that symmetry, not an LQR
        # shape, so the value-gradient policy has zero torque at upright rest.
        def symmetric_value(states):
            features = self._features(states)
            reflection = features.new_tensor([-1., -1., 1., 1., -1., -1.])
            return 0.5 * (self.net(features) + self.net(features * reflection)).squeeze(-1)

        value = symmetric_value(z)
        if self.anchor:
            goal = z.new_tensor([math.pi, 0., 0., 0.])
            value = value - symmetric_value(goal)
        return value


def bounded_action_score(eta, effort_weight, torque_limit):
    """chi(eta), including saturation; derivative is the maximizing torque."""
    u = (eta / effort_weight).clamp(-torque_limit, torque_limit)
    return eta * u - 0.5 * effort_weight * u.square()


class AcrobotPHValue:
    def __init__(self, oracle=None, reward=None, config=None, *, device="cpu"):
        self.oracle = oracle or AcrobotOracle()
        self.reward = reward or UprightReward()
        self.config = config or ValueFlowConfig()
        self.device = torch.device(device)
        self.value = AcrobotValue(self.config).to(self.device)
        self.target = deepcopy(self.value).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.value.parameters(), lr=self.config.learning_rate)
        self.log_temperature = None
        self.temperature_optimizer = None
        if self.config.auto_temperature:
            self.log_temperature = torch.tensor(math.log(self.config.temperature), device=self.device, requires_grad=True)
            self.temperature_optimizer = torch.optim.Adam([self.log_temperature], lr=self.config.temperature_learning_rate)
        nodes, weights = np.polynomial.legendre.leggauss(self.config.quadrature_points)
        self.nodes = torch.tensor(nodes, dtype=torch.float64, device=self.device)
        self.log_weights = torch.tensor(weights, dtype=torch.float64, device=self.device).log()
        self.updates = 0
        self.metadata = {}

    @property
    def temperature(self):
        return self.config.temperature if self.log_temperature is None else float(self.log_temperature.detach().exp())

    def value_gradient(self, z, *, target=False):
        # Acting needs derivatives even inside a caller's no_grad context.
        with torch.enable_grad():
            states = torch.as_tensor(z, dtype=torch.float32, device=self.device).detach().requires_grad_(True)
            values = (self.target if target else self.value)(states)
            gradient = torch.autograd.grad(values.sum(), states)[0]
        return states.detach(), values.detach(), gradient.detach()

    def soft_action_score(self, eta):
        """alpha log integral exp(g(u_max*a)/alpha) da, via Gauss-Legendre.

        Entropy is relative to da on [-1,1], not a categorical node entropy.
        Float64 log-sum-exp avoids exponent overflow. Resolution is configurable.
        """
        alpha = self.temperature
        if alpha == 0:
            return bounded_action_score(eta, self.reward.effort_weight, self.oracle.torque_limit)
        u = self.oracle.torque_limit * self.nodes
        scores = eta.double().unsqueeze(-1) * u - 0.5 * self.reward.effort_weight * u.square()
        return (alpha * torch.logsumexp(scores / alpha + self.log_weights, dim=-1)).to(eta.dtype)

    def policy_entropy(self, eta):
        """Differential entropy relative to normalized action measure da.

        Evaluate H = log Z - E[g/alpha] using the same quadrature as the
        soft HJB operator, with a score shift to avoid cancellation. This is
        not categorical entropy of the quadrature-node probabilities.
        """
        alpha = self.temperature
        if alpha <= 0:
            raise ValueError("policy entropy requires positive temperature")
        u = self.oracle.torque_limit * self.nodes
        scores = eta.detach().double().unsqueeze(-1) * u - 0.5 * self.reward.effort_weight * u.square()
        scaled = (scores - scores.max(dim=-1, keepdim=True).values) / alpha
        log_mass = scaled + self.log_weights
        entropy = torch.logsumexp(log_mass, dim=-1) - (log_mass.softmax(dim=-1) * scaled).sum(-1)
        return entropy

    def update_temperature(self, z, *, terminal_mask=None):
        """SAC-style log-temperature update, excluding absorbing failure states."""
        if self.temperature_optimizer is None:
            return {}
        states = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        if terminal_mask is not None:
            states = states[~torch.as_tensor(terminal_mask, dtype=torch.bool, device=self.device)]
        if not len(states):
            return {"temperature": self.temperature, "temperature_update_skipped": True}
        _, _, gradient = self.value_gradient(states)
        entropy = self.policy_entropy(gradient[..., 3]).mean()
        # Gradient descent increases alpha when H < H_target. The policy
        # entropy is detached: this step changes only log(alpha).
        loss = self.log_temperature * (entropy - self.config.target_entropy).detach()
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite temperature loss")
        self.temperature_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.temperature_optimizer.step()
        with torch.no_grad():
            self.log_temperature.clamp_(math.log(self.config.temperature_min), math.log(self.config.temperature_max))
        return {"temperature": self.temperature, "policy_entropy": float(entropy),
                "temperature_loss": float(loss.detach()),
                "entropy_error": float(entropy - self.config.target_entropy)}

    def operator(self, z, *, target=True):
        states, values, gradient = self.value_gradient(z, target=target)
        drift_term = (gradient * self.oracle.drift(states)).sum(-1)
        eta = gradient[..., 3]  # G^T grad V; physical elbow torque has gain one.
        operator = (-self.reward.state_cost(states, self.oracle) + drift_term
                    - self.config.discount_rate * values + self.soft_action_score(eta))
        return values, operator, eta

    def fitted_targets(self, z, *, terminal_mask=None, terminal_value=None):
        """Detached HJB labels, with absorbing boundary values where requested."""
        values, operator, eta = self.operator(z)
        target = values + self.config.value_step * operator
        if terminal_mask is not None:
            if terminal_value is None or not math.isfinite(terminal_value):
                raise ValueError("terminal_mask requires a finite terminal_value")
            mask = torch.as_tensor(terminal_mask, dtype=torch.bool, device=self.device)
            target = torch.where(mask, target.new_full((), terminal_value), target)
        return target, operator, eta

    def fit_labels(self, z, labels, *, refresh_target=True):
        """One regression step; a fixed-label diagnostic can freeze the target."""
        target = torch.as_tensor(labels, dtype=torch.float32, device=self.device).detach()
        prediction = self.value(torch.as_tensor(z, dtype=torch.float32, device=self.device))
        if prediction.shape != target.shape:
            raise ValueError("labels must have the same shape as value predictions")
        loss = (prediction - target).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite oracle value-flow loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.value.parameters(), self.config.grad_clip, error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        # The Polyak step every target_interval updates rather than every one.
        # Labels are built from the target, so they are exactly constant between
        # moves; the effective lag stretches to target_interval / target_rate
        # updates without the abruptness of a hard copy.
        if refresh_target and self.updates % self.config.target_interval == 0:
            with torch.no_grad():
                for slow, fast in zip(self.target.parameters(), self.value.parameters()):
                    slow.lerp_(fast, self.config.target_rate)
        return {"value_loss": float(loss.detach()), "gradient_norm": float(norm)}

    def update(self, z, *, terminal_mask=None, terminal_value=None):
        target, operator, eta = self.fitted_targets(z, terminal_mask=terminal_mask, terminal_value=terminal_value)
        metrics = self.fit_labels(z, target)
        temperature_metrics = self.update_temperature(z, terminal_mask=terminal_mask)
        return {**metrics, **temperature_metrics, "hjb_rms": float(operator.square().mean().sqrt()),
                "eta_abs_mean": float(eta.abs().mean())}

    def act(self, z, *, deterministic=True, rng=None):
        """Return normalized actions. Deterministic deployment uses the mode."""
        _, _, gradient = self.value_gradient(z)
        eta = gradient[..., 3].cpu().numpy()
        mean = eta / (self.reward.effort_weight * self.oracle.torque_limit)
        if deterministic or self.temperature == 0:
            return np.clip(mean, -1., 1.)[..., None]
        from scipy.stats import truncnorm
        scale = math.sqrt(self.temperature / (self.reward.effort_weight * self.oracle.torque_limit**2))
        a = truncnorm.rvs((-1 - mean) / scale, (1 - mean) / scale,
                         loc=mean, scale=scale, random_state=rng)
        return np.asarray(a)[..., None]

    def save(self, path, *, metadata=None):
        torch.save({"format": "acrobot_ph_value_stage_a_v1", "oracle": asdict(self.oracle),
                    "reward": asdict(self.reward), "config": asdict(self.config),
                    "value": self.value.state_dict(), "target": self.target.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "updates": self.updates,
                    "log_temperature": None if self.log_temperature is None else self.log_temperature.detach(),
                    "temperature_optimizer": None if self.temperature_optimizer is None else self.temperature_optimizer.state_dict(),
                    "metadata": self.metadata if metadata is None else metadata}, path)

    @classmethod
    def load(cls, path, *, device="cpu"):
        state = torch.load(path, map_location=device, weights_only=True)
        if state["format"] != "acrobot_ph_value_stage_a_v1":
            raise ValueError("not an oracle Acrobot value-flow checkpoint")
        result = cls(AcrobotOracle(**state["oracle"]), UprightReward(**state["reward"]),
                     ValueFlowConfig(**state["config"]), device=device)
        result.value.load_state_dict(state["value"])
        result.target.load_state_dict(state["target"])
        result.optimizer.load_state_dict(state["optimizer"])
        if result.log_temperature is not None:
            with torch.no_grad():
                result.log_temperature.copy_(state["log_temperature"])
            result.temperature_optimizer.load_state_dict(state["temperature_optimizer"])
        result.updates = state["updates"]
        result.metadata = state.get("metadata", {})
        return result
