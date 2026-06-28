# models/port_hamiltonian.py
"""Port-Hamiltonian dynamics model for the model-based generator in CT-SAC.

This supplies the drift ``b(x, a)`` (the "b term" of the controlled generator,
Eq. 6 of the Hamiltonian-Flow paper) and an optional diffusion ``sigma(x)`` so
that the generator ``(L^a V) = b . grad V + 1/2 Tr(sigma sigma^T Hess V)`` can be
evaluated analytically, without sampling a successor state.

Two modes:

- ``"mujoco"``: the drift is provided by a callable (e.g.
  ``DMCContinuousEnv.dynamics_terms``), i.e. the simulator's exact
  observation-space drift. No learnable parameters. Used as the validation
  oracle (Milestone M0).

- ``"phast"``: a learned, structure-preserving drift
  ``b(x, a) = (J - R) grad H(x) + G_a a`` with ``J`` skew-symmetric, ``R`` a
  positive-semidefinite (low-rank Householder) dissipation, ``H`` a scalar energy
  network, and ``G_a`` a linear actuation (port) map. This is the UNKNOWN-regime
  PHAST model on the observation treated as a generalized phase state. The full
  state-dependent damping ``D(q)`` and the Strang integrator are deferred (see
  ``docs/port_hamiltonian_ct_sac.md``).

The diffusion is ``None`` when ``human_input_intensity == 0`` (the v1 default,
``sigma = 0``); the fluctuation-dissipation form ``sigma sigma^T = 2 T D(q)`` is a
later milestone (M2).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import torch as th
from torch import nn


class PortHamiltonianModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        mode: str = "mujoco",
        drift_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        hidden: Sequence[int] = (256, 256),
        dissipation_rank: int = 4,
        human_input_intensity: float = 0.0,
        device: str | th.device = "cpu",
    ) -> None:
        super().__init__()
        self.mode = str(mode)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.human_input_intensity = float(human_input_intensity)
        self.device = th.device(device)
        self._drift_fn = drift_fn

        if self.mode == "mujoco":
            if drift_fn is None:
                raise ValueError(
                    "mode='mujoco' requires drift_fn (e.g. env.dynamics_terms)."
                )
        elif self.mode == "phast":
            d = self.obs_dim
            # Scalar energy H_theta(x)
            layers: list[nn.Module] = []
            last = d
            for h in hidden:
                layers += [nn.Linear(last, h), nn.SiLU()]
                last = h
            layers += [nn.Linear(last, 1)]
            self.energy = nn.Sequential(*layers)
            # Skew-symmetric J = A - A^T
            self._J_raw = nn.Parameter(0.01 * th.randn(d, d))
            # PSD dissipation R = softplus(d0) I + L L^T  (constant; PHAST Eq. 10 low-rank)
            self._d0 = nn.Parameter(th.tensor(-2.0))
            self._L = nn.Parameter(0.01 * th.randn(d, int(dissipation_rank)))
            # Agent port G_a: action -> state drift
            self.G_a = nn.Linear(self.action_dim, d, bias=False)
        else:
            raise ValueError(f"Unknown mode '{self.mode}'.")

        self.to(self.device)

    # ------------------------ structure helpers (phast) ------------------------

    def _J(self) -> th.Tensor:
        return self._J_raw - self._J_raw.t()

    def _R(self) -> th.Tensor:
        d0 = th.nn.functional.softplus(self._d0)
        eye = th.eye(self.obs_dim, device=self._L.device)
        return d0 * eye + self._L @ self._L.t()

    def _grad_H(self, x: th.Tensor) -> th.Tensor:
        xin = x.clone().requires_grad_(True)
        H = self.energy(xin).sum()
        (gH,) = th.autograd.grad(H, xin, create_graph=True)
        return gH

    # ------------------------ public API ------------------------

    def drift(self, obs, action) -> th.Tensor:
        """Return the drift b(obs, action) of shape (B, obs_dim) on ``self.device``."""
        if self.mode == "mujoco":
            b = self._drift_fn(obs, action)  # numpy (B, d)
            return th.as_tensor(np.asarray(b), dtype=th.float32, device=self.device)

        x = th.as_tensor(obs, dtype=th.float32, device=self.device)
        a = th.as_tensor(action, dtype=th.float32, device=self.device)
        gH = self._grad_H(x)  # (B, d)
        JR = self._J() - self._R()  # (d, d)
        return gH @ JR.t() + self.G_a(a)

    def diffusion(self, obs) -> Optional[th.Tensor]:
        """Return sigma(obs) of shape (B, obs_dim, k), or None if sigma == 0."""
        if self.human_input_intensity <= 0.0:
            return None
        batch = obs.shape[0] if hasattr(obs, "shape") else len(obs)
        scale = float(self.human_input_intensity) ** 0.5
        eye = th.eye(self.obs_dim, device=self.device) * scale
        return eye.unsqueeze(0).expand(batch, -1, -1)

    def fit_step(self, obs, action, next_obs, dt, optimizer) -> float:
        """One supervised PHAST data step: minimize ||(x + b*dt) - x'||^2 (mode='phast')."""
        assert self.mode == "phast", "fit_step only applies to mode='phast'."
        x = th.as_tensor(obs, dtype=th.float32, device=self.device)
        a = th.as_tensor(action, dtype=th.float32, device=self.device)
        xp = th.as_tensor(next_obs, dtype=th.float32, device=self.device)
        dt_t = th.as_tensor(dt, dtype=th.float32, device=self.device).reshape(-1, 1)
        pred = x + self.drift(x, a) * dt_t
        loss = ((pred - xp) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss.detach())
