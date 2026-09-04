"""Xin-Kaneda swing-up, latching to an SOS-certified saturated controller.

Same one-way-latch shape as :class:`~controllers.acrobot_gated_lyapunov.
XKLQRSwitchedController`, but the switching region and the local law are
whatever :mod:`evaluations.acrobot_sos_roa` actually certified -- a quadratic
Lyapunov sublevel set ``B_rho = {e : e^T P e <= rho}`` (Lai et al.'s equation
17 / the local-LQR-in-a-generous-boundary region this repository uses
elsewhere are NOT the same set) and a cubic saturated feedback ``u(e)``,
degree 1 through 3 in the four upright-error coordinates -- instead of the
region-17 test and the plain linear ``tau = -Ke``. Both come from
``results/acrobot_sos_roa_tau20_certificate.json``
(``.claude_scratch/extract_sos_certificate.py``, run inside the Drake+MOSEK
apptainer container this project's venv doesn't have), loaded here as plain
numpy/JSON so nothing downstream needs Drake.

The certificate itself was found by a nondeterministic, incompletely-
converged alternation (see docs/acrobot_sos_roa.md's investigation) -- it is
a real, solver-verified certificate for its own rho, but not demonstrably the
largest rho the method could reach, and it certifies the degree-3 Taylor
model of the dynamics, not the exact plant. Its physical size is small: at
tau_max=20 the certified ellipsoid's single-axis bounds are sub-degree in
angle and a few hundredths of a rad/s in rate (see the session record).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .acrobot_gated_lyapunov import UPRIGHT_STATE, upright_error
from .xin_kaneda import AcrobotParams, XinKanedaController

DEFAULT_CERTIFICATE_PATH = (
    Path(__file__).resolve().parent.parent
    / "results"
    / "acrobot_sos_roa_tau20_certificate.json"
)


@dataclass(frozen=True)
class SOSCertificate:
    """A loaded ``(P, rho, u(e))`` triple: region membership plus the law
    certified to decrease inside it.

    ``controller_terms`` is ``[(exponents, coefficient), ...]`` for the cubic
    polynomial ``u(e) = sum coefficient * prod(e_i ** exponent_i)``, in the
    physical torque units :func:`evaluations.acrobot_sos_roa.saturated_system`
    was built with (N*m, the same convention
    :class:`~controllers.xin_kaneda.XinKanedaController` clips and normalizes
    by ``gear``, not the normalized ``[-1, 1]`` action).
    """

    tau_max: float
    rho: float
    P: np.ndarray
    controller_terms: Tuple[Tuple[Tuple[int, ...], float], ...]
    lqr_level: Optional[float] = None
    converged: bool = False
    passes: Optional[int] = None

    def __post_init__(self) -> None:
        p = np.asarray(self.P, dtype=np.float64)
        if p.shape != (4, 4):
            raise ValueError(f"P must have shape (4, 4), got {p.shape}")
        if not np.allclose(p, p.T, atol=1e-9):
            raise ValueError("P must be symmetric")
        if np.any(np.linalg.eigvalsh(p) <= 0):
            raise ValueError("P must be positive definite")
        object.__setattr__(self, "P", p)
        if not np.isfinite(self.rho) or self.rho <= 0.0:
            raise ValueError(f"rho must be finite and positive, got {self.rho}")

    def residual(self, state: np.ndarray) -> float:
        """``e^T P e``; inside the certified region iff this is ``<= rho``."""
        error = upright_error(state)
        return float(error @ self.P @ error)

    def contains(self, state: np.ndarray) -> bool:
        return self.residual(state) <= self.rho

    def residual_batch(self, states: np.ndarray) -> np.ndarray:
        values = np.asarray(states, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValueError(f"expected states of shape (N, 4), got {values.shape}")
        errors = values - UPRIGHT_STATE
        errors[:, :2] = np.arctan2(np.sin(errors[:, :2]), np.cos(errors[:, :2]))
        return np.einsum("ni,ij,nj->n", errors, self.P, errors)

    def contains_batch(self, states: np.ndarray) -> np.ndarray:
        return self.residual_batch(states) <= self.rho

    def command(self, state: np.ndarray) -> float:
        """Physical torque (N*m) from the certified cubic ``u(e)``."""
        error = upright_error(state)
        total = 0.0
        for exponents, coefficient in self.controller_terms:
            term = coefficient
            for value, power in zip(error, exponents):
                if power:
                    term *= value**power
            total += term
        return float(total)

    def command_batch(self, states: np.ndarray) -> np.ndarray:
        values = np.asarray(states, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValueError(f"expected states of shape (N, 4), got {values.shape}")
        errors = values - UPRIGHT_STATE
        errors[:, :2] = np.arctan2(np.sin(errors[:, :2]), np.cos(errors[:, :2]))
        total = np.zeros(values.shape[0])
        for exponents, coefficient in self.controller_terms:
            term = np.full(values.shape[0], coefficient)
            for i, power in enumerate(exponents):
                if power:
                    term = term * errors[:, i] ** power
            total += term
        return total

    def sample_uniform(self, rng: np.random.RandomState) -> np.ndarray:
        """A state drawn uniformly (by volume) from the solid ellipsoid
        ``B_rho``, mapped back onto ``[q1, q2, qd1, qd2]`` about upright.

        Standard construction: a direction uniform on the unit sphere times a
        radius ``u**(1/n)`` (``n=4`` here) is uniform in the unit ball; the
        ellipsoid's own "square root" ``P^{-1/2}`` (eigendecomposition, since
        ``P`` is symmetric positive definite) carries that ball onto
        ``{e : e^T P e <= rho}``.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.P)
        inv_sqrt_p = eigenvectors @ np.diag(eigenvalues**-0.5) @ eigenvectors.T
        direction = rng.normal(size=4)
        direction /= np.linalg.norm(direction)
        radius = rng.uniform() ** 0.25
        unit_ball_point = radius * direction
        error = np.sqrt(self.rho) * (inv_sqrt_p @ unit_ball_point)
        return UPRIGHT_STATE + error


def load_certificate(path=DEFAULT_CERTIFICATE_PATH) -> SOSCertificate:
    with open(path) as f:
        data = json.load(f)
    return SOSCertificate(
        tau_max=float(data["tau_max"]),
        rho=float(data["rho"]),
        P=np.asarray(data["P"], dtype=np.float64),
        controller_terms=tuple(
            (tuple(int(p) for p in exponents), float(coefficient))
            for exponents, coefficient in data["controller_terms"]
        ),
        lqr_level=data.get("lqr_level"),
        converged=bool(data.get("converged", False)),
        passes=data.get("passes"),
    )


class XKSOSSwitchedController:
    """Xin-Kaneda swing-up, latching one-way to the SOS-certified law.

    Same shape as :class:`~controllers.acrobot_gated_lyapunov.
    XKLQRSwitchedController`: swing up under the exact Xin-Kaneda law, and on
    first entry to the certificate's own region latch permanently to its
    certified cubic ``u(e)`` instead of the linear ``tau = -Ke`` -- using a
    *different* law than the region was proven against would silently drop
    the certification's guarantee.
    """

    SWING_UP = 1
    BALANCE = 2

    def __init__(
        self,
        params: AcrobotParams,
        gains,
        certificate: Optional[SOSCertificate] = None,
        *,
        torque_limit: Optional[float] = None,
    ) -> None:
        self.params = params
        self.certificate = certificate or load_certificate()
        self.torque_limit = (
            float(params.gear) if torque_limit is None else float(torque_limit)
        )
        if not np.isfinite(self.torque_limit) or self.torque_limit <= 0.0:
            raise ValueError(
                f"torque_limit must be finite and > 0, got {self.torque_limit}"
            )
        self.swing_up = XinKanedaController(
            params, gains, torque_limit=self.torque_limit
        )
        self.reset()

    def reset(self) -> None:
        self.swing_up.reset()
        self.stage = self.SWING_UP
        self.switch_step: Optional[int] = None
        self.last_torque = 0.0
        self.last_commanded_torque = 0.0
        self.saturated_steps = 0
        self.steps = 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        state = np.asarray(obs, dtype=np.float64).reshape(-1)
        if self.stage == self.SWING_UP and self.certificate.contains(state):
            self.stage = self.BALANCE
            self.switch_step = self.steps
        self.steps += 1

        if self.stage == self.SWING_UP:
            action = self.swing_up(obs)
            self.last_torque = self.swing_up.last_torque
            self.last_commanded_torque = self.swing_up.last_commanded_torque
            return action

        commanded = self.certificate.command(state)
        applied = float(np.clip(commanded, -self.torque_limit, self.torque_limit))
        if abs(commanded) > self.torque_limit:
            self.saturated_steps += 1
        self.last_commanded_torque = commanded
        self.last_torque = applied
        return np.array([applied / self.params.gear], dtype=np.float64)

    def actions(self, obs: np.ndarray) -> np.ndarray:
        """Normalized commands for a batch of observations, ``(N, 4) -> (N, 1)``.

        Stateless and vectorized, like
        :meth:`~controllers.acrobot_gated_lyapunov.XKLQRSwitchedController.
        actions` -- see that method's docstring for why the imitation loss
        needs this, and why membership is re-tested per row rather than
        latched.
        """
        values = np.asarray(obs, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValueError(
                f"expected observations of shape (N, 4), got {values.shape}"
            )

        commands = self.swing_up.actions(values).reshape(-1)
        inside = self.certificate.contains_batch(values)
        if inside.any():
            commanded = self.certificate.command_batch(values[inside])
            applied = np.clip(commanded, -self.torque_limit, self.torque_limit)
            commands[inside] = applied / self.params.gear
        return commands.reshape(-1, 1)
