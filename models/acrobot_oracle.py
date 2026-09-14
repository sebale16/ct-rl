"""Fixed Acrobot mechanics in z=(q1, q2, p1, p2), q1 measured from down.

No fitted dynamics or trainable physical parameters. Torque is in N m.
"""
from __future__ import annotations

from dataclasses import dataclass
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

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    def state_cost(self, z: torch.Tensor, oracle: AcrobotOracle) -> torch.Tensor:
        v = oracle.velocity(z) / self.velocity_scale
        # Equivalent to 1+cos(q1), with better accuracy near q1=pi.
        return (2 * self.angle1_weight * ((z[..., 0] - math.pi) / 2).sin().square()
                + 2 * self.angle2_weight * (z[..., 1] / 2).sin().square()
                + self.velocity1_weight * v[..., 0].square()
                + self.velocity2_weight * v[..., 1].square())

    def rate(self, z: torch.Tensor, torque: torch.Tensor | float,
             oracle: AcrobotOracle) -> torch.Tensor:
        u = torch.as_tensor(torque, dtype=z.dtype, device=z.device)
        if u.ndim == z.ndim and u.shape[-1] == 1:
            u = u.squeeze(-1)
        return -self.state_cost(z, oracle) - 0.5 * self.effort_weight * u.square()
