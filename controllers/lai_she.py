"""Lai--She three-stage nonsmooth-Lyapunov controller for the Acrobot.

This implements the controller in Lai, She, Yang & Wu, *Stability Analysis
and Control Law Design for Acrobots*, ICRA 2006, equations (20), (26), and
(36).  The paper's generalized coordinates have the straight upright at
``x=(0, 0, 0, 0)`` and the straight hanging pose at ``x=(pi, 0, 0, 0)``.

The ``acrobot-swingup-xk`` model used by this repository has the same published
mechanical parameters, but its first link is measured from the horizontal and
both joint senses are reflected.  Therefore its raw state ``q`` maps to the
2006 paper's state as

    x1 = pi/2 - q1,  x2 = -q2,  xdot1 = -qdot1,  xdot2 = -qdot2.

The generalized elbow torque is reflected too: ``tau_q = -tau_x``.

Reproducibility note
--------------------
The paper publishes the plant and scalar controller values, the fuzzy rule
table, and the attractive-set bounds.  It does not publish the fuzzy membership
ranges or the LQR ``Q`` and ``R`` matrices.  :class:`Design` consequently
keeps those two implementation choices explicit.  The defaults use symmetric
triangular fuzzy sets on normalized energy/power and a conventional diagonal
LQR cost; none is presented as a value reported by the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.linalg import solve_continuous_are


def wrap(angle):
    """Wrap an angle to ``[-pi, pi)`` (scalar or array)."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class AcrobotParams:
    """Physical parameters from Section IV of Lai et al. (2006)."""

    m1: float = 1.0
    m2: float = 1.0
    i1: float = 0.083
    i2: float = 0.33
    l1: float = 1.0
    l2: float = 2.0
    lc1: float = 0.5
    lc2: float = 1.0
    gravity: float = 9.8
    gear: float = 80.0

    def __post_init__(self) -> None:
        for name in (
            "m1", "m2", "i1", "i2", "l1", "l2", "lc1", "lc2",
            "gravity", "gear",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {value}")
        if self.lc1 > self.l1 or self.lc2 > self.l2:
            raise ValueError("a link center of mass cannot lie beyond its link")
        if self.a1 * self.a2 <= self.a3**2:
            raise ValueError("the inertia matrix must be positive definite")

    @property
    def a1(self) -> float:
        return self.m1 * self.lc1**2 + self.i1 + self.m2 * self.l1**2

    @property
    def a2(self) -> float:
        return self.m2 * self.lc2**2 + self.i2

    @property
    def a3(self) -> float:
        return self.m2 * self.l1 * self.lc2

    @property
    def b1(self) -> float:
        return (self.m1 * self.lc1 + self.m2 * self.l1) * self.gravity

    @property
    def b2(self) -> float:
        return self.m2 * self.lc2 * self.gravity

    @property
    def energy_top(self) -> float:
        return self.b1 + self.b2

    @property
    def energy_span(self) -> float:
        return 2.0 * self.energy_top

    def mass_matrix(self, x2: float) -> np.ndarray:
        cosine = np.cos(float(x2))
        m12 = self.a2 + self.a3 * cosine
        return np.array(
            [[self.a1 + self.a2 + 2.0 * self.a3 * cosine, m12],
             [m12, self.a2]],
            dtype=np.float64,
        )

    def coriolis(self, state: np.ndarray) -> np.ndarray:
        _, x2, x3, x4 = np.asarray(state, dtype=np.float64)
        coefficient = self.a3 * np.sin(x2)
        return coefficient * np.array(
            [-(2.0 * x3 * x4 + x4**2), x3**2], dtype=np.float64
        )

    def gravity_vector(self, state: np.ndarray) -> np.ndarray:
        x1, x2 = np.asarray(state, dtype=np.float64)[:2]
        return np.array(
            [
                -self.b1 * np.sin(x1) - self.b2 * np.sin(x1 + x2),
                -self.b2 * np.sin(x1 + x2),
            ],
            dtype=np.float64,
        )

    def energy(self, state: np.ndarray) -> float:
        values = np.asarray(state, dtype=np.float64)
        x1, x2, x3, x4 = values
        velocity = np.array([x3, x4])
        kinetic = 0.5 * float(velocity @ self.mass_matrix(x2) @ velocity)
        potential = self.b1 * np.cos(x1) + self.b2 * np.cos(x1 + x2)
        return kinetic + potential

    def drift_and_input(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``f(x), b(x)`` in equations (10)--(12)."""
        values = np.asarray(state, dtype=np.float64)
        mass = self.mass_matrix(values[1])
        bias = self.coriolis(values) + self.gravity_vector(values)
        acceleration_drift = np.linalg.solve(mass, -bias)
        acceleration_input = np.linalg.solve(mass, np.array([0.0, 1.0]))
        return (
            np.array([values[2], values[3], *acceleration_drift]),
            np.array([0.0, 0.0, *acceleration_input]),
        )

    @classmethod
    def from_physics(cls, physics) -> "AcrobotParams":
        """Read the paper-parameter plant back from a two-link MuJoCo model."""
        model = physics.model
        mass = np.asarray(model.body_mass, dtype=np.float64)
        ipos = np.asarray(model.body_ipos, dtype=np.float64)
        bpos = np.asarray(model.body_pos, dtype=np.float64)
        if int(model.nbody) != 3 or int(model.nq) != 2 or int(model.nv) != 2:
            raise ValueError("expected a two-link Acrobot MuJoCo model")
        link_axis = int(np.argmax(np.abs(bpos[2])))
        hinge_axis = int(np.argmax(np.abs(np.asarray(model.jnt_axis)[0])))
        inertia = np.asarray(model.body_inertia, dtype=np.float64)[:, hinge_axis]
        return cls(
            m1=float(mass[1]),
            m2=float(mass[2]),
            i1=float(inertia[1]),
            i2=float(inertia[2]),
            l1=float(abs(bpos[2, link_axis])),
            # MuJoCo does not store a massless link endpoint; the paper's l2 is
            # independent of the dynamics and is known from its plant table.
            l2=2.0,
            lc1=float(abs(ipos[1, link_axis])),
            lc2=float(abs(ipos[2, link_axis])),
            gravity=float(-np.asarray(model.opt.gravity)[2]),
            gear=float(abs(np.asarray(model.actuator_gear)[0, 0])),
        )


PAPER_PARAMS = AcrobotParams()


@dataclass(frozen=True)
class Design:
    """Published scalar design values plus explicit omitted design choices."""

    beta1: float = np.pi / 6.0
    beta2: float = np.pi / 6.0
    energy_tolerance: float = 1.2
    kp1: float = 1.0
    kd1: float = 1.0
    ke1: float = 0.2
    lambda1: float = 38.0
    phi1: float = 10.0
    zeta: float = -2.0
    kp2: float = 1.0
    kd2: float = 1.0
    phi2: float = 5.0
    lambda_alpha: float = 0.5
    # Not reported in the paper: normalizers for the five fuzzy sets.
    fuzzy_energy_scale: Optional[float] = None
    fuzzy_power_scale: float = 10.0
    # Not reported in the paper: LQR state and input cost.
    lqr_q: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    lqr_r: float = 1.0

    def __post_init__(self) -> None:
        positive = (
            "beta1", "beta2", "energy_tolerance", "kp1", "kd1", "ke1",
            "lambda1", "phi1", "kp2", "kd2", "phi2", "lambda_alpha",
            "fuzzy_power_scale", "lqr_r",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        if not np.isfinite(self.zeta) or self.zeta >= 0.0:
            raise ValueError("zeta must be finite and < 0")
        if self.fuzzy_energy_scale is not None and self.fuzzy_energy_scale <= 0:
            raise ValueError("fuzzy_energy_scale must be > 0 when specified")
        if len(self.lqr_q) != 4 or any(value <= 0 for value in self.lqr_q):
            raise ValueError("lqr_q must contain four positive weights")


def xk_to_paper(obs: np.ndarray) -> np.ndarray:
    """Transform the repository's horizontal-frame state to Lai et al.'s frame."""
    values = np.asarray(obs, dtype=np.float64).reshape(-1)
    if values.shape != (4,):
        raise ValueError(f"expected [q1, q2, qdot1, qdot2], got {values.shape}")
    return np.array(
        [0.5 * np.pi - values[0], -values[1], -values[2], -values[3]],
        dtype=np.float64,
    )


def paper_to_xk(state: np.ndarray) -> np.ndarray:
    """Inverse of :func:`xk_to_paper`."""
    values = np.asarray(state, dtype=np.float64).reshape(-1)
    if values.shape != (4,):
        raise ValueError(f"expected [x1, x2, x3, x4], got {values.shape}")
    return np.array(
        [0.5 * np.pi - values[0], -values[1], -values[2], -values[3]],
        dtype=np.float64,
    )


def _linear_model(params: AcrobotParams) -> tuple[np.ndarray, np.ndarray]:
    """Linearize equations (6)--(9) at the upright equilibrium."""
    equilibrium = np.zeros(4, dtype=np.float64)
    step = 1e-6
    a = np.empty((4, 4), dtype=np.float64)
    for column in range(4):
        offset = np.zeros(4)
        offset[column] = step
        forward, _ = params.drift_and_input(equilibrium + offset)
        backward, _ = params.drift_and_input(equilibrium - offset)
        a[:, column] = (forward - backward) / (2.0 * step)
    _, b = params.drift_and_input(equilibrium)
    return a, b[:, None]


def lqr_solution(
    params: AcrobotParams, design: Design
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``A, B, F, P`` for equations (33)--(38)."""
    a, b = _linear_model(params)
    q = np.diag(np.asarray(design.lqr_q, dtype=np.float64))
    r = np.array([[design.lqr_r]], dtype=np.float64)
    p = solve_continuous_are(a, b, q, r)
    f = np.linalg.solve(r, b.T @ p)
    return a, b, f, p


def _memberships(value: float) -> np.ndarray:
    """Five symmetric triangular sets NB, NM, ZR, PM, PB on ``[-1, 1]``."""
    centers = np.linspace(-1.0, 1.0, 5)
    value = float(np.clip(value, -1.0, 1.0))
    weights = np.maximum(0.0, 1.0 - np.abs(value - centers) / 0.5)
    total = float(weights.sum())
    if total == 0.0:  # defensive; clipping and shoulder sets make this unreachable
        weights[int(np.argmin(np.abs(centers - value)))] = 1.0
        total = 1.0
    return weights / total


# Table I, encoded as indices into NB=-1, NM=-.5, ZR=0, PM=.5, PB=1.
_FUZZY_RULES = np.array(
    [
        [0, 0, 0, 1, 2],
        [0, 0, 1, 2, 3],
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 4],
        [2, 3, 4, 4, 4],
    ],
    dtype=np.int64,
)
_FUZZY_OUTPUTS = np.linspace(-1.0, 1.0, 5)


def fuzzy_adjustment(
    energy_error: float,
    power: float,
    params: AcrobotParams,
    design: Design,
) -> float:
    """Table-I fuzzy output ``r`` using product inference and centroid output."""
    energy_scale = design.fuzzy_energy_scale or params.energy_span
    energy_membership = _memberships(float(energy_error) / energy_scale)
    power_membership = _memberships(float(power) / design.fuzzy_power_scale)
    firing = power_membership[:, None] * energy_membership[None, :]
    result = float(np.sum(firing * _FUZZY_OUTPUTS[_FUZZY_RULES]) / np.sum(firing))
    # Equation (30) requires the strict inequality -1 < r < 1.
    return float(np.clip(result, -1.0 + 1e-6, 1.0 - 1e-6))


class LaiSheController:
    """Stateful minimum-switching implementation of the three control laws."""

    STAGE_1 = 1
    STAGE_2 = 2
    STAGE_3 = 3

    def __init__(
        self,
        params: AcrobotParams = PAPER_PARAMS,
        design: Design = Design(),
        *,
        frame: str = "xk",
        torque_limit: Optional[float] = None,
    ) -> None:
        if frame not in ("paper", "xk"):
            raise ValueError("frame must be 'paper' or 'xk'")
        self.params = params
        self.design = design
        self.frame = frame
        self.torque_limit = params.gear if torque_limit is None else float(torque_limit)
        if not np.isfinite(self.torque_limit) or self.torque_limit <= 0.0:
            raise ValueError("torque_limit must be finite and > 0")
        if self.torque_limit > params.gear * (1.0 + 1e-12):
            raise ValueError("torque_limit cannot exceed the plant gear")
        self.a, self.b, self.lqr_gain, self.p = lqr_solution(params, design)
        self.reset()

    def reset(self) -> None:
        self.stage = self.STAGE_1
        self.delta1: Optional[float] = None
        self.last_torque = 0.0
        self.last_commanded_torque = 0.0
        self.last_energy_error = 0.0
        self.last_fuzzy_adjustment = 0.0
        self.last_denominator = np.nan
        self.switch_log: list[tuple[int, int]] = []
        self.steps = 0
        self.saturated_steps = 0

    def paper_state(self, obs: np.ndarray) -> np.ndarray:
        return (
            np.asarray(obs, dtype=np.float64).reshape(-1)
            if self.frame == "paper"
            else xk_to_paper(obs)
        )

    def _terms(self, state: np.ndarray):
        drift, input_vector = self.params.drift_and_input(state)
        f_eta = float(drift[3])
        b_eta = float(input_vector[3])
        energy_error = self.params.energy(state) - self.params.energy_top
        return f_eta, b_eta, energy_error

    def j1(self, state: np.ndarray) -> float:
        _, _, energy_error = self._terms(state)
        d = self.design
        delta = 0.0 if self.delta1 is None else self.delta1
        return float(
            0.5
            * (d.kp1 * state[1] ** 2 + d.kd1 * state[3] ** 2
               + d.ke1 * energy_error**2)
            + delta
        )

    def j2(self, state: np.ndarray) -> float:
        d = self.design
        delta = 0.0 if self.delta1 is None else self.delta1
        return float(0.5 * (d.kp2 * state[1] ** 2 + d.kd2 * state[3] ** 2) + delta)

    def lqr_state(self, state: np.ndarray) -> np.ndarray:
        values = np.asarray(state, dtype=np.float64).copy()
        values[0] = wrap(values[0])
        values[1] = wrap(values[1])
        return values

    def j3(self, state: np.ndarray) -> float:
        error = self.lqr_state(state)
        return float(error @ self.p @ error)

    def in_attractive_area(self, state: np.ndarray) -> bool:
        d = self.design
        return bool(
            abs(float(wrap(state[0]))) <= d.beta1
            and abs(float(wrap(state[0] + state[1]))) <= d.beta2
            and abs(self.params.energy(state) - self.params.energy_top)
            <= d.energy_tolerance
        )

    def torque_c1(self, state: np.ndarray) -> float:
        f_eta, b_eta, energy_error = self._terms(state)
        d = self.design
        denominator = d.kd1 * b_eta + d.ke1 * energy_error
        self.last_denominator = denominator
        if abs(denominator) < 1e-10:
            raise FloatingPointError("C1 reached its singular surface")
        numerator = (
            d.kp1 * state[1]
            + d.kd1 * f_eta
            + d.lambda1 * np.clip(state[3] / d.phi1, -1.0, 1.0)
        )
        return float(-numerator / denominator)

    def torque_c2(self, state: np.ndarray) -> float:
        f_eta, b_eta, energy_error = self._terms(state)
        d = self.design
        torque_tilde = -(d.kp2 * state[1] + d.kd2 * f_eta) / (d.kd2 * b_eta)
        power = state[3] * torque_tilde
        adjustment = fuzzy_adjustment(energy_error, power, self.params, d)
        lambda2 = d.lambda_alpha * (1.0 + adjustment)
        self.last_fuzzy_adjustment = adjustment
        return float(
            torque_tilde - lambda2 * np.clip(state[3] / d.phi2, -1.0, 1.0)
        )

    def torque_c3(self, state: np.ndarray) -> float:
        return float(-(self.lqr_gain @ self.lqr_state(state))[0])

    def _maybe_switch(self, state: np.ndarray) -> None:
        d = self.design
        if self.stage == self.STAGE_1:
            _, b_eta, energy_error = self._terms(state)
            denominator = d.kd1 * b_eta + d.ke1 * energy_error
            self.last_denominator = denominator
            # Equation (23) prints ``denominator <= zeta``.  Along the reported
            # swing-up the denominator approaches zero from below, however, so
            # that inequality is true at the initial hanging pose and false
            # near the singularity.  The figure's delayed C1->C2 switch and the
            # stated purpose (switch *before* the zero) require the crossing
            # ``denominator >= zeta``; we implement that operational condition.
            if self.j2(state) < self.j1(state) and denominator >= d.zeta:
                self.switch_log.append((self.STAGE_1, self.steps))
                self.stage = self.STAGE_2
        if self.stage == self.STAGE_2 and self.in_attractive_area(state):
            # Equation (55): choose Delta1 at the first attractive-set entry.
            self.delta1 = self.j3(state)
            self.switch_log.append((self.STAGE_2, self.steps))
            self.stage = self.STAGE_3

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        state = self.paper_state(obs)
        self._maybe_switch(state)
        if self.stage == self.STAGE_1:
            commanded = self.torque_c1(state)
        elif self.stage == self.STAGE_2:
            commanded = self.torque_c2(state)
        else:
            commanded = self.torque_c3(state)
        applied = float(np.clip(commanded, -self.torque_limit, self.torque_limit))
        self.last_commanded_torque = commanded
        # The xk plant's reflected joint sense requires tau_q = -tau_x.
        self.last_torque = applied if self.frame == "paper" else -applied
        self.last_energy_error = self.params.energy(state) - self.params.energy_top
        self.steps += 1
        if abs(commanded) > self.torque_limit:
            self.saturated_steps += 1
        normalized = self.last_torque / self.params.gear
        return np.array([normalized], dtype=np.float64)

    @property
    def saturation_fraction(self) -> float:
        return 0.0 if self.steps == 0 else self.saturated_steps / self.steps
