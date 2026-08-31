"""Lai--She unified WCLF controller for the Acrobot.

Implements the Acrobot specialization of Lai, She, Yang & Wu,
"Comprehensive Unified Control Strategy for Underactuated Two-Link
Manipulators", IEEE TSMC-B 39(2), 2009, equations (20), (25), (36), (41),
and (46).

The paper measures the shoulder from the upward vertical: upright is
``x=(0, 0, 0, 0)`` and hanging is ``x=(pi, 0, 0, 0)``.  The dedicated
``acrobot-swingup-wclf`` plant uses these coordinates directly.  An adapter for
the repository's horizontal-frame ``acrobot-swingup-xk`` plant remains
available and applies

    x = [pi/2 - q1, -q2, -qdot1, -qdot2],  tau_q = -tau_x.

Unlike the earlier Lai--She controller, this formulation contains no fuzzy
logic and no intermediate transition controller.  A single state-dependent
weak-control Lyapunov function (WCLF) governs swing-up, then a published LQR
gain takes over on first entry to the attractive area.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .acrobot_gated_lyapunov import AttractiveRegion, riccati_feedback


def wrap(angle):
    """Wrap an angle to ``[-pi, pi)`` (scalar or array)."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class AcrobotParams:
    """Acrobot parameters from Table II of Lai et al. (2009)."""

    m1: float = 1.0
    m2: float = 1.0
    i1: float = 8.33e-2
    i2: float = 0.33
    l1: float = 1.0
    l2: float = 2.0
    lc1: float = 0.5
    lc2: float = 1.0
    gravity: float = 9.8
    # The paper does not impose an actuator bound. ``gear`` is solely the
    # normalized MuJoCo action interface used by the executable plant.
    gear: float = 50.0

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
    def b2_gravity(self) -> float:
        return self.m2 * self.lc2 * self.gravity

    @property
    def energy_top(self) -> float:
        return self.b1 + self.b2_gravity

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

    def det_mass(self, x2: float) -> float:
        cosine = np.cos(float(x2))
        return float(self.a1 * self.a2 - self.a3**2 * cosine**2)

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
                -self.b1 * np.sin(x1)
                - self.b2_gravity * np.sin(x1 + x2),
                -self.b2_gravity * np.sin(x1 + x2),
            ],
            dtype=np.float64,
        )

    def energy(self, state: np.ndarray) -> float:
        values = np.asarray(state, dtype=np.float64)
        x1, x2, x3, x4 = values
        velocity = np.array([x3, x4])
        kinetic = 0.5 * float(velocity @ self.mass_matrix(x2) @ velocity)
        potential = self.b1 * np.cos(x1) + self.b2_gravity * np.cos(x1 + x2)
        return kinetic + potential

    def drift_and_input(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``f_a(x), b_a(x)`` in equations (13)--(16)."""
        values = np.asarray(state, dtype=np.float64)
        mass = self.mass_matrix(values[1])
        bias = self.coriolis(values) + self.gravity_vector(values)
        acceleration_drift = np.linalg.solve(mass, -bias)
        acceleration_input = np.linalg.solve(mass, np.array([0.0, 1.0]))
        return (
            np.array([values[2], values[3], *acceleration_drift]),
            np.array([0.0, 0.0, *acceleration_input]),
        )

    def elbow_input_gain(self, x2: float) -> float:
        """``b_2(x)=m_11/det(M)`` in equations (29)--(30)."""
        return float(self.mass_matrix(x2)[0, 0] / self.det_mass(x2))

    def elbow_input_gain_derivative(self, x2: float) -> float:
        """Derivative ``d b_2 / d x_2`` used to evaluate ``beta_dot``."""
        sine = np.sin(float(x2))
        cosine = np.cos(float(x2))
        m11 = self.a1 + self.a2 + 2.0 * self.a3 * cosine
        determinant = self.a1 * self.a2 - self.a3**2 * cosine**2
        dm11 = -2.0 * self.a3 * sine
        ddeterminant = 2.0 * self.a3**2 * cosine * sine
        return float(
            (dm11 * determinant - m11 * ddeterminant) / determinant**2
        )

    @classmethod
    def from_physics(cls, physics) -> "AcrobotParams":
        """Recover Table-II physical values from a two-link MuJoCo model."""
        model = physics.model
        if int(model.nbody) != 3 or int(model.nq) != 2 or int(model.nv) != 2:
            raise ValueError("expected a two-link Acrobot MuJoCo model")
        mass = np.asarray(model.body_mass, dtype=np.float64)
        ipos = np.asarray(model.body_ipos, dtype=np.float64)
        bpos = np.asarray(model.body_pos, dtype=np.float64)
        link_axis = int(np.argmax(np.abs(bpos[2])))
        hinge_axis = int(np.argmax(np.abs(np.asarray(model.jnt_axis)[0])))
        inertia = np.asarray(model.body_inertia, dtype=np.float64)[:, hinge_axis]
        return cls(
            m1=float(mass[1]),
            m2=float(mass[2]),
            i1=float(inertia[1]),
            i2=float(inertia[2]),
            l1=float(abs(bpos[2, link_axis])),
            l2=2.0,
            lc1=float(abs(ipos[1, link_axis])),
            lc2=float(abs(ipos[2, link_axis])),
            gravity=float(-np.asarray(model.opt.gravity)[2]),
            gear=float(abs(np.asarray(model.actuator_gear)[0, 0])),
        )


PAPER_PARAMS = AcrobotParams()


@dataclass(frozen=True)
class Design:
    """Published Acrobot design values from equations (73) and (75)."""

    alpha1: float = 0.5
    alpha2: float = 30.0
    eta: float = 25.0
    gamma0: float = 1.6
    energy_epsilon: float = 0.5
    energy_top: float = 24.5
    angle1_tolerance: float = np.pi / 6.0
    angle2_tolerance: float = np.pi / 6.0
    velocity1_weight: float = 1e-3
    velocity2_weight: float = 1e-3
    velocity_tolerance: float = 1e3
    energy_tolerance: float = 1.0
    lqr_q: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    lqr_r: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "alpha1", "alpha2", "eta", "gamma0", "energy_epsilon",
            "energy_top", "angle1_tolerance", "angle2_tolerance",
            "velocity1_weight", "velocity2_weight", "velocity_tolerance",
            "energy_tolerance", "lqr_r",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        if len(self.lqr_q) != 4 or any(value <= 0 for value in self.lqr_q):
            raise ValueError("lqr_q must contain four positive weights")
        if self.eta <= 2.0 * self.alpha1 * self.energy_top:
            raise ValueError("equation (36) requires eta > 2*alpha1*E0")

    def attractive_region(self) -> AttractiveRegion:
        """Equation (17)'s region carrying these tolerances.

        The region itself lives in :mod:`controllers.acrobot_gated_lyapunov`,
        which also builds the nonsmooth Lyapunov function on it; its conditions
        are even in their arguments, so the same object serves this paper frame
        and the repository's Xin--Kaneda frame unchanged.
        """
        return AttractiveRegion(
            angle_tolerance=self.angle1_tolerance,
            tip_tolerance=self.angle2_tolerance,
            energy_tolerance=self.energy_tolerance,
            velocity_weights=(self.velocity1_weight, self.velocity2_weight),
            velocity_tolerance=self.velocity_tolerance,
        )


def xk_to_paper(obs: np.ndarray) -> np.ndarray:
    """Map a horizontal-frame raw state into the 2009 paper's coordinates."""
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
    """Return ``A, B, F, P`` for equations (43)--(46)."""
    a, b = _linear_model(params)
    f, p = riccati_feedback(a, b, design.lqr_q, design.lqr_r)
    return a, b, f, p


# Equation (75).  Recomputing the CARE from the rounded Table-II inertia gives
# a gain within 0.2 N.m of these entries; use the explicitly published control
# law for reproduction rather than silently substituting the rounded rebuild.
PUBLISHED_LQR_GAIN = np.array(
    [[-260.559, -104.448, -112.604, -52.944]], dtype=np.float64
)


class LaiSheController:
    """Stateful WCLF swing-up/LQR balance controller."""

    SWING_UP = 1
    BALANCE = 2

    def __init__(
        self,
        params: AcrobotParams = PAPER_PARAMS,
        design: Design = Design(),
        *,
        frame: str = "paper",
        torque_limit: Optional[float] = None,
    ) -> None:
        if frame not in ("paper", "xk"):
            raise ValueError("frame must be 'paper' or 'xk'")
        if abs(params.energy_top - design.energy_top) > 1e-8:
            raise ValueError(
                "the plant and design disagree on E0: "
                f"{params.energy_top} versus {design.energy_top}"
            )
        self.params = params
        self.design = design
        self.frame = frame
        self.torque_limit = params.gear if torque_limit is None else float(torque_limit)
        if not np.isfinite(self.torque_limit) or self.torque_limit <= 0.0:
            raise ValueError("torque_limit must be finite and > 0")
        if self.torque_limit > params.gear * (1.0 + 1e-12):
            raise ValueError("torque_limit cannot exceed the plant gear")
        self.region = design.attractive_region()
        self.a, self.b, recomputed_gain, self.p = lqr_solution(params, design)
        self.recomputed_lqr_gain = recomputed_gain
        self.lqr_gain = PUBLISHED_LQR_GAIN.copy()
        self.reset()

    def reset(self) -> None:
        self.stage = self.SWING_UP
        self.switch_step: Optional[int] = None
        self.last_torque = 0.0
        self.last_commanded_torque = 0.0
        self.last_energy_error = -2.0 * self.params.energy_top
        self.last_beta = np.nan
        self.last_gamma = np.nan
        self.last_wclf = np.nan
        self.steps = 0
        self.saturated_steps = 0

    def paper_state(self, obs: np.ndarray) -> np.ndarray:
        values = np.asarray(obs, dtype=np.float64).reshape(-1)
        if values.shape != (4,):
            raise ValueError(f"expected a 4-vector state, got {values.shape}")
        return values if self.frame == "paper" else xk_to_paper(values)

    def beta(self, state: np.ndarray) -> float:
        return float(self.design.eta / self.params.elbow_input_gain(state[1]))

    def beta_dot(self, state: np.ndarray) -> float:
        input_gain = self.params.elbow_input_gain(state[1])
        input_gain_dot = (
            self.params.elbow_input_gain_derivative(state[1]) * state[3]
        )
        return float(-self.design.eta * input_gain_dot / input_gain**2)

    def gamma(self, state: np.ndarray) -> float:
        energy_epsilon = (
            self.params.energy(state)
            + self.design.energy_top
            + self.design.energy_epsilon
        )
        return float(self.design.gamma0 * energy_epsilon)

    def wclf(self, state: np.ndarray) -> float:
        energy_error = self.params.energy(state) - self.design.energy_top
        return float(
            0.5
            * (
                self.design.alpha1 * energy_error**2
                + self.design.alpha2 * state[1] ** 2
                + self.beta(state) * state[3] ** 2
            )
        )

    def swingup_torque(self, state: np.ndarray) -> float:
        drift, input_vector = self.params.drift_and_input(state)
        f2 = float(drift[3])
        b2 = float(input_vector[3])
        energy_error = self.params.energy(state) - self.design.energy_top
        beta = self.beta(state)
        beta_dot = self.beta_dot(state)
        gamma = self.gamma(state)
        denominator = self.design.alpha1 * energy_error + beta * b2
        if denominator <= 0.0 or not np.isfinite(denominator):
            raise FloatingPointError(
                "WCLF denominator must stay positive by equation (37), got "
                f"{denominator}"
            )
        numerator = (
            -self.design.alpha2 * state[1]
            - beta * f2
            - 0.5 * beta_dot * state[3]
            - gamma * state[3]
        )
        self.last_beta = beta
        self.last_gamma = gamma
        return float(numerator / denominator)

    def lqr_state(self, state: np.ndarray) -> np.ndarray:
        values = np.asarray(state, dtype=np.float64).copy()
        values[0] = wrap(values[0])
        values[1] = wrap(values[1])
        return values

    def balance_torque(self, state: np.ndarray) -> float:
        return float(-(self.lqr_gain @ self.lqr_state(state))[0])

    def in_attractive_area(self, state: np.ndarray) -> bool:
        """Equation (17), evaluated by the shared :class:`AttractiveRegion`."""
        return bool(
            self.region.residual_of(
                float(wrap(state[0])),
                float(wrap(state[0] + state[1])),
                self.params.energy(state) - self.design.energy_top,
                state[2:],
            )
            <= 1.0
        )

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        state = self.paper_state(obs)
        if self.stage == self.SWING_UP and self.in_attractive_area(state):
            self.stage = self.BALANCE
            self.switch_step = self.steps
        commanded = (
            self.swingup_torque(state)
            if self.stage == self.SWING_UP
            else self.balance_torque(state)
        )
        applied = float(np.clip(commanded, -self.torque_limit, self.torque_limit))
        paper_torque = applied
        plant_torque = paper_torque if self.frame == "paper" else -paper_torque
        self.last_commanded_torque = commanded
        self.last_torque = plant_torque
        self.last_energy_error = self.params.energy(state) - self.design.energy_top
        self.last_wclf = self.wclf(state)
        self.steps += 1
        if abs(commanded) > self.torque_limit:
            self.saturated_steps += 1
        return np.array([plant_torque / self.params.gear], dtype=np.float64)

    @property
    def saturation_fraction(self) -> float:
        return 0.0 if self.steps == 0 else self.saturated_steps / self.steps
