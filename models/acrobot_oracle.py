"""Fixed Acrobot mechanics in z=(q1, q2, p1, p2), q1 measured from down.

No fitted dynamics or trainable physical parameters. Torque is in N m.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch


@dataclass(frozen=True)
class AcrobotOracle:
    a1: float = 1.333
    a2: float = 1.330
    a3: float = 1.0
    b1: float = 14.7
    b2: float = 9.8
    damping: float = 0.0
    torque_limit: float = 20.0

    def __post_init__(self):
        for name in ("a1", "a2", "a3", "b1", "b2", "torque_limit"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.a1 * self.a2 <= self.a3**2:
            raise ValueError("mass matrix must be positive definite for every angle")
        if not math.isfinite(self.damping) or self.damping < 0:
            raise ValueError("damping must be finite and nonnegative")

    def mass(self, q: torch.Tensor) -> torch.Tensor:
        c = self.a3 * q[..., 1].cos()
        m11 = self.a1 + self.a2 + 2 * c
        m12 = self.a2 + c
        m22 = torch.full_like(c, self.a2)
        return torch.stack((m11, m12, m12, m22), dim=-1).reshape(*q.shape[:-1], 2, 2)

    def canonical(self, qv: torch.Tensor) -> torch.Tensor:
        q, v = qv[..., :2], qv[..., 2:]
        p = (self.mass(q) @ v.unsqueeze(-1)).squeeze(-1)
        return torch.cat((q, p), dim=-1)

    def velocity(self, z: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve(self.mass(z[..., :2]), z[..., 2:].unsqueeze(-1)).squeeze(-1)

    def energy(self, z: torch.Tensor) -> torch.Tensor:
        q = z[..., :2]
        return (0.5 * (z[..., 2:] * self.velocity(z)).sum(-1)
                - self.b1 * q[..., 0].cos() - self.b2 * q.sum(-1).cos())

    def drift(self, z: torch.Tensor, torque: torch.Tensor | float = 0.0) -> torch.Tensor:
        """Canonical vector field; partial_q E is evaluated at fixed p."""
        q, v = z[..., :2], self.velocity(z)
        g2 = self.b2 * q.sum(-1).sin()
        g1 = self.b1 * q[..., 0].sin() + g2
        # -1/2 v^T (partial_q2 M) v, the kinetic part of partial_q2 E.
        kinetic_q2 = self.a3 * q[..., 1].sin() * (v[..., 0]**2 + v[..., 0] * v[..., 1])
        torque = torch.as_tensor(torque, dtype=z.dtype, device=z.device)
        if torque.ndim == z.ndim and torque.shape[-1] == 1:
            torque = torque.squeeze(-1)
        dp1 = -g1 - self.damping * v[..., 0]
        dp2 = -g2 - kinetic_q2 - self.damping * v[..., 1] + torque
        return torch.cat((v, torch.stack((dp1, dp2), dim=-1)), dim=-1)


@dataclass(frozen=True)
class UprightReward:
    angle1_weight: float = 10.0
    angle2_weight: float = 5.0
    velocity1_weight: float = 1.0
    velocity2_weight: float = 1.0
    velocity_scale: float = 4.5844
    effort_weight: float = 0.01
    state_cost_transform: str = "identity"
    log_epsilon: float | None = None
    log_cost_bound: float | None = None

    def __post_init__(self):
        for name in ("angle1_weight", "angle2_weight", "velocity1_weight", "velocity2_weight",
                     "velocity_scale", "effort_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.state_cost_transform not in ("identity", "log"):
            raise ValueError("state_cost_transform must be identity or log")
        if self.state_cost_transform == "log":
            for name in ("log_epsilon", "log_cost_bound"):
                value = getattr(self, name)
                if value is None or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"log state cost requires finite positive {name}")
            if not math.isfinite(self.log_cost_bound / self.log_epsilon):
                raise ValueError("log cost bound / epsilon must be finite")
        elif self.log_epsilon is not None or self.log_cost_bound is not None:
            raise ValueError("log parameters require state_cost_transform=log")

    def scaled(self, factor: float):
        """Scale the entire reward, including the controller's effort weight."""
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("reward scale must be finite and positive")
        parameters = {name: getattr(self, name) * factor for name in (
            "angle1_weight", "angle2_weight", "velocity1_weight", "velocity2_weight", "effort_weight")}
        if self.state_cost_transform == "log":
            parameters.update(log_epsilon=self.log_epsilon * factor, log_cost_bound=self.log_cost_bound * factor)
        return replace(self, **parameters)

    def with_log_state_cost(self, cost_bound, reference_angle_deg=5.):
        """Choose epsilon from a shoulder-only deviation at rest, in reward units."""
        if not math.isfinite(reference_angle_deg) or not 0 < reference_angle_deg <= 180:
            raise ValueError("log reference angle must be in (0,180] degrees")
        epsilon = 2 * self.angle1_weight * math.sin(math.radians(reference_angle_deg) / 2)**2
        return replace(self, state_cost_transform="log", log_epsilon=epsilon, log_cost_bound=cost_bound)

    def transform_state_cost(self, cost):
        """Same scalar/tensor mapping for MuJoCo rewards and differentiable HJB."""
        if self.state_cost_transform == "identity":
            return cost
        coefficient = self.log_cost_bound / math.log1p(self.log_cost_bound / self.log_epsilon)
        log_cost = torch.log1p(cost / self.log_epsilon) if torch.is_tensor(cost) else math.log1p(cost / self.log_epsilon)
        return coefficient * log_cost

    def base_state_cost_bound(self, *, velocity_limit, elbow_limit, shoulder_limit):
        """State-cost bound before the optional transform; excludes effort."""
        shoulder = (2 * self.angle1_weight if shoulder_limit is None else
                    self.angle1_weight * (1 - math.cos(min(shoulder_limit, math.pi))))
        elbow = self.angle2_weight * (1 - math.cos(min(elbow_limit, math.pi)))
        return shoulder + elbow + (self.velocity1_weight + self.velocity2_weight) * (velocity_limit / self.velocity_scale)**2

    def cost_bound(self, oracle: AcrobotOracle, *, velocity_limit, elbow_limit, shoulder_limit):
        """Upper bound on ordinary cost inside the declared state/action limits."""
        state_bound = self.base_state_cost_bound(velocity_limit=velocity_limit, elbow_limit=elbow_limit,
                                                shoulder_limit=shoulder_limit)
        return self.transform_state_cost(state_bound) + 0.5 * self.effort_weight * oracle.torque_limit**2

    def state_cost(self, z: torch.Tensor, oracle: AcrobotOracle) -> torch.Tensor:
        v = oracle.velocity(z) / self.velocity_scale
        # Equivalent to 1+cos(q1), with better accuracy near q1=pi.
        cost = (2 * self.angle1_weight * ((z[..., 0] - math.pi) / 2).sin().square()
                + 2 * self.angle2_weight * (z[..., 1] / 2).sin().square()
                + self.velocity1_weight * v[..., 0].square()
                + self.velocity2_weight * v[..., 1].square())
        return self.transform_state_cost(cost)

    def rate(self, z: torch.Tensor, torque: torch.Tensor | float,
             oracle: AcrobotOracle) -> torch.Tensor:
        u = torch.as_tensor(torque, dtype=z.dtype, device=z.device)
        if u.ndim == z.ndim and u.shape[-1] == 1:
            u = u.squeeze(-1)
        return -self.state_cost(z, oracle) - 0.5 * self.effort_weight * u.square()
