"""Xin–Kaneda energy-based swing-up control for the Acrobot.

The control law is a single expression that appears in two papers:

* Xin, X. and Kaneda, M. (2002), "The Swing up Control for the Acrobot based on
  Energy Control Approach", CDC 2002, eq. (14), with a Lyapunov gain ``k_E``.
* Xin, X. and Kaneda, M. (2007), "Analysis of the energy-based swing-up control
  of the Acrobot", Int. J. Robust Nonlinear Control 17:1503-1524, eq. (18),
  with ``k_E`` normalized to 1.

The law is invariant under a common scaling of ``(k_E, k_V, k_D, k_P)``, so only
the ratios matter and this module fixes ``k_E = 1`` throughout, following 2007.
The 2002 thresholds are still computed (``kd_min_theorem4``, ``eta_star``,
``xi_star``) because they are the published fixtures the tests check against and
because they show how much the 2002 bounds over-pay.

Coordinates
-----------
Everything here works in the papers' frame: ``q1`` is measured from the
horizontal, upright is ``q1 = pi/2`` and hanging ``q1 = -pi/2``.  The
``acrobot-swingup-xk`` plant is built in exactly those coordinates, so its
``qpos``/``qvel`` need no conversion and ``gear * ctrl`` is ``tau2`` outright.

The stock dm_control Acrobot instead measures the shoulder from the upward
vertical, upright ``qpos = (0, 0)`` and hanging ``(pi, 0)``.  That frame is a
reflection of this one,

    q1_paper = pi/2 - q1_stock,  q2_paper = -q2_stock,  qdot_paper = -qdot_stock

with the torque flipping to match.  ``M(q)`` is invariant under it (it depends
on ``cos q2`` only) and the mechanical energies differ by the constant
``2*(b1 + b2)`` from the height reference.  :func:`obs_to_paper` applies the map,
and :class:`XinKanedaController` will do so when constructed with
``frame="upward_vertical"``.

Dynamics
--------
``M(q) qddot + H(q, qdot) + G(q) = [0, tau2]``, with

    M   = [[a1 + a2 + 2 a3 cos q2, a2 + a3 cos q2], [a2 + a3 cos q2, a2]]
    H   = a3 sin q2 [-2 qd1 qd2 - qd2^2, qd1^2]
    G   = [b1 cos q1 + b2 cos(q1 + q2), b2 cos(q1 + q2)]
    E   = 1/2 qdot' M qdot + b1 sin q1 + b2 sin(q1 + q2)

and ``a1 = m1 lc1^2 + m2 l1^2 + I1``, ``a2 = m2 lc2^2 + I2``,
``a3 = m2 l1 lc2``, ``b1 = (m1 lc1 + m2 l1) g``, ``b2 = m2 lc2 g``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# Grid resolution for the one-dimensional extremal problems below.  They are all
# smooth and 2*pi-periodic in the elbow angle, so a dense sweep followed by a
# parabolic refinement is exact to well past the precision the published
# constants are quoted at.
_GRID = 200_001


def _refine(f, grid: np.ndarray, index: int, *, maximize: bool) -> float:
    """Parabolic refinement of an extremum bracketed at ``grid[index]``.

    Returns the extremal *value* of ``f``.  Minimization is handled by fitting
    the parabola to ``-f`` and evaluating ``f`` at the located abscissa.
    """
    if index <= 0 or index >= grid.size - 1:
        return float(f(grid[index]))
    sign = 1.0 if maximize else -1.0
    x0, x1, x2 = grid[index - 1], grid[index], grid[index + 1]
    y0, y1, y2 = sign * f(x0), sign * f(x1), sign * f(x2)
    denominator = y0 - 2.0 * y1 + y2
    if denominator == 0.0:
        return float(f(x1))
    offset = 0.5 * (y0 - y2) / denominator
    return float(f(x1 + offset * (x1 - x0)))


@dataclass(frozen=True)
class Gains:
    """Control gains of the law, with ``k_E`` normalized to 1."""

    k_v: float
    k_d: float
    k_p: float

    def __post_init__(self) -> None:
        for name in ("k_v", "k_d", "k_p"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {value}")


@dataclass(frozen=True)
class AcrobotParams:
    """Mechanical parameters in the papers' ``(a, b)`` grouping.

    ``gear`` is the actuator scaling of the plant the controller commands: the
    dm_control Acrobot takes ``ctrl`` in ``[-1, 1]`` and applies
    ``tau = gear * ctrl``.  It plays no part in the dynamics here and is carried
    only so the controller can emit a normalized command.
    """

    a1: float
    a2: float
    a3: float
    b1: float
    b2: float
    gear: float = 1.0
    gravity: float = 9.81

    def __post_init__(self) -> None:
        for name in ("a1", "a2", "a3", "b1", "b2", "gear", "gravity"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {value}")
        # Positive definiteness of M(q) for every q; the papers assume it and
        # every threshold below divides by det M.
        if self.a1 * self.a2 <= self.a3**2:
            raise ValueError(
                "a1*a2 must exceed a3^2 for M(q) to stay positive definite, got "
                f"a1*a2={self.a1 * self.a2}, a3^2={self.a3**2}"
            )

    # --- Model quantities -------------------------------------------------

    @property
    def energy_top(self) -> float:
        """``E_r``: energy at the upright equilibrium, eq. (12) of 2007."""
        return self.b1 + self.b2

    @property
    def energy_span(self) -> float:
        """``E_top - E_hang = 2 (b1 + b2)``: the energy a swing-up must supply."""
        return 2.0 * (self.b1 + self.b2)

    def m11(self, q2):
        """Upper-left entry of ``M``; a function of the elbow angle alone."""
        return self.a1 + self.a2 + 2.0 * self.a3 * np.cos(q2)

    def det_m(self, q2):
        """``det M(q) = a1 a2 - a3^2 cos^2 q2``, eq. (16) of 2007."""
        return self.a1 * self.a2 - self.a3**2 * np.cos(q2) ** 2

    def mass_matrix(self, q2: float) -> np.ndarray:
        c2 = np.cos(q2)
        off = self.a2 + self.a3 * c2
        return np.array(
            [[self.a1 + self.a2 + 2.0 * self.a3 * c2, off], [off, self.a2]],
            dtype=np.float64,
        )

    def bias(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        """``H + G``: Coriolis/centrifugal plus gravity, in paper coordinates."""
        q1, q2 = float(q[0]), float(q[1])
        d1, d2 = float(qdot[0]), float(qdot[1])
        s2 = np.sin(q2)
        h = np.array([s2 * (-2.0 * d1 * d2 - d2**2), s2 * d1**2]) * self.a3
        g = np.array(
            [
                self.b1 * np.cos(q1) + self.b2 * np.cos(q1 + q2),
                self.b2 * np.cos(q1 + q2),
            ]
        )
        return h + g

    def energy(self, q: np.ndarray, qdot: np.ndarray) -> float:
        """Total mechanical energy ``E``, eq. (6)/(7) of the papers."""
        q1, q2 = float(q[0]), float(q[1])
        v = np.asarray(qdot, dtype=np.float64)
        kinetic = 0.5 * float(v @ self.mass_matrix(q2) @ v)
        potential = self.b1 * np.sin(q1) + self.b2 * np.sin(q1 + q2)
        return kinetic + potential

    def potential_envelope(self, q2):
        """``F(q2) = sqrt(b1^2 + b2^2 + 2 b1 b2 cos q2)``, eq. (24) of 2007.

        The extreme value of the potential over the poses where the shoulder
        gravity torque vanishes, which is what makes condition (25) exact.
        """
        return np.sqrt(self.b1**2 + self.b2**2 + 2.0 * self.b1 * self.b2 * np.cos(q2))

    # --- Construction from a MuJoCo model ---------------------------------

    @classmethod
    def from_physics(cls, physics) -> "AcrobotParams":
        """Recover ``(a, b)`` from a dm_control Acrobot ``Physics``.

        Reads the two links' masses, centre-of-mass offsets, hinge-axis
        inertias, and the joint offset, so the result tracks any edit to the
        model XML (damping and gear changes included).
        """
        model = physics.model
        if int(model.nbody) != 3 or int(model.nq) != 2 or int(model.nv) != 2:
            raise ValueError(
                "expected a two-link Acrobot model (3 bodies, nq = nv = 2), got "
                f"nbody={int(model.nbody)}, nq={int(model.nq)}, nv={int(model.nv)}"
            )
        mass = np.asarray(model.body_mass, dtype=np.float64)
        ipos = np.asarray(model.body_ipos, dtype=np.float64)
        bpos = np.asarray(model.body_pos, dtype=np.float64)
        # Read the layout off the model instead of assuming one: the link
        # direction is the axis the elbow is offset along, and the relevant
        # principal inertia is the one about the hinge axis.  This keeps the
        # recovery valid both for a model laid out along +z (the dm_control
        # Acrobot) and one laid out along +x (the paper-coordinate plant).
        link_axis = int(np.argmax(np.abs(bpos[2])))
        hinge = np.abs(np.asarray(model.jnt_axis, dtype=np.float64)[0])
        inertia_axis = int(np.argmax(hinge))
        inertia = np.asarray(model.body_inertia, dtype=np.float64)[:, inertia_axis]
        m1, m2 = float(mass[1]), float(mass[2])
        lc1 = float(abs(ipos[1, link_axis]))
        lc2 = float(abs(ipos[2, link_axis]))
        l1 = float(abs(bpos[2, link_axis]))
        i1, i2 = float(inertia[1]), float(inertia[2])
        gravity = float(-np.asarray(model.opt.gravity, dtype=np.float64)[2])
        gear = float(np.asarray(model.actuator_gear, dtype=np.float64)[0, 0])
        return cls(
            a1=m1 * lc1**2 + m2 * l1**2 + i1,
            a2=m2 * lc2**2 + i2,
            a3=m2 * l1 * lc2,
            b1=(m1 * lc1 + m2 * l1) * gravity,
            b2=m2 * lc2 * gravity,
            gear=gear,
            gravity=gravity,
        )


# --- Published mechanical parameters -------------------------------------

# Xin & Kaneda's own simulation plant, used in both papers' figures:
# m1 = m2 = 1, l1 = 1, l2 = 2, lc1 = 0.5, lc2 = 1, I1 = 0.083, I2 = 0.33, g = 9.8.
PAPER_PARAMS = AcrobotParams(
    a1=1.0 * 0.5**2 + 1.0 * 1.0**2 + 0.083,
    a2=1.0 * 1.0**2 + 0.33,
    a3=1.0 * 1.0 * 1.0,
    b1=(1.0 * 0.5 + 1.0 * 1.0) * 9.8,
    b2=1.0 * 1.0 * 9.8,
    gravity=9.8,
)


# --- Gain conditions ------------------------------------------------------


def kd_min(params: AcrobotParams) -> float:
    """Condition (25) of 2007: the exact no-singularity threshold on ``k_D``.

    ``k_D > max_q2 (F(q2) + E_r) det M(q2) / M11(q2)`` is *necessary and
    sufficient* for the denominator of the control law to stay away from zero
    for every initial state and all future time (Proposition 1).  The 2002
    bound :func:`kd_min_theorem4` is sufficient only, and larger.
    """
    grid = np.linspace(0.0, 2.0 * np.pi, _GRID)

    def value(q2):
        return (
            (params.potential_envelope(q2) + params.energy_top)
            * params.det_m(q2)
            / params.m11(q2)
        )

    return _refine(value, grid, int(np.argmax(value(grid))), maximize=True)


def kd_min_theorem4(params: AcrobotParams) -> float:
    """Condition (13) of 2002 at ``k_E = 1``: ``2 E_top / rho*``.

    Kept for comparison; it is the sufficient bound Theorem 4 uses, obtained by
    replacing ``E_r - P(q)`` with the cruder ``2 E_r``.
    """
    return 2.0 * params.energy_top / rho_star(params)


def kp_min(params: AcrobotParams) -> float:
    """Condition (43) of 2007: ``(2/pi) min(b1^2, b2^2)``.

    Proposition 4's threshold.  It does not remove the closed-loop equilibria
    with ``q2 != 0``; it makes their number finite, and they are then shown to
    be unstable and hyperbolic, so the initial conditions converging to them
    form a set of Lebesgue measure zero.
    """
    return (2.0 / np.pi) * min(params.b1**2, params.b2**2)


def kp_min_exact(params: AcrobotParams) -> float:
    """Condition (51) of 2007: ``b1 b2 sup_{q2 != 0} Z(q2)``.

    The exact form of the requirement that :func:`kp_min` bounds in closed form
    via ``sup Z <= (2/pi) min(b1/b2, b2/b1)`` (their Appendix C).  ``Z`` is the
    same function 2002 calls ``eta``.
    """
    return params.b1 * params.b2 * eta_star(params)


def kp_boundary(params: AcrobotParams) -> float:
    """``2 b1 b2``: the threshold that both papers' strong results share.

    It is condition (57) of 2002 Theorem 4, condition (63) of 2007 Corollary 1,
    and the spectral boundary of 2007 Proposition 5 at the hanging equilibrium
    — the same number in all three roles.  Above it the closed loop has no
    equilibrium with ``q2 != 0``; below it the hanging equilibrium gains a third
    right-half-plane eigenvalue and the escape from hanging speeds up by two
    orders of magnitude.
    """
    return 2.0 * params.b1 * params.b2


def rho_star(params: AcrobotParams) -> float:
    """``rho* = min_q2 M11(q2) / det M(q2)``, eq. (11) of 2002."""
    grid = np.linspace(0.0, 2.0 * np.pi, _GRID)

    def value(q2):
        return params.m11(q2) / params.det_m(q2)

    return _refine(value, grid, int(np.argmin(value(grid))), maximize=False)


def _beta_delta(params: AcrobotParams, q2):
    beta = params.b1 / params.b2
    delta = np.sqrt(1.0 + beta**2 + 2.0 * beta * np.cos(q2))
    return beta, delta


def eta_star(params: AcrobotParams) -> float:
    """``eta* = sup_{q2 != 0} (delta - beta - 1) sin q2 / (delta q2)``.

    eq. (34)/(36) of 2002, identical to ``sup Z`` in eq. (50) of 2007.  The 2002
    paper takes the maximum over ``[pi, 3 pi / 2]``, where the numerator is
    positive; the sweep here covers a full period, which contains it.
    """
    grid = np.linspace(1e-6, 2.0 * np.pi - 1e-6, _GRID)

    def value(q2):
        beta, delta = _beta_delta(params, q2)
        return (delta - beta - 1.0) * np.sin(q2) / (delta * q2)

    return _refine(value, grid, int(np.argmax(value(grid))), maximize=True)


def xi_star(params: AcrobotParams) -> float:
    """``xi* = sup_{q2 != 0} (delta + beta + 1) sin q2 / (delta q2) = 2``.

    eq. (55)/(56) of 2002, eq. (65)/(66) of 2007.  The supremum is the limit at
    ``q2 -> 0``, where ``delta -> 1 + beta``, giving exactly 2 for every plant;
    this is why :func:`kp_boundary` is always ``2 b1 b2`` and why ``eta*`` never
    binds in the 2002 Theorem-4 condition.
    """
    grid = np.linspace(1e-9, 2.0 * np.pi - 1e-9, _GRID)

    def value(q2):
        beta, delta = _beta_delta(params, q2)
        return (delta + beta + 1.0) * np.sin(q2) / (delta * q2)

    return max(2.0, float(np.max(value(grid))))


def alpha(params: AcrobotParams) -> float:
    """Condition (31) of 2002: the mechanical non-degeneracy quantity.

    Theorem 2 of 2002 needs ``alpha != 0``, which the paper calls a mild
    condition on the mechanical parameters.  Reported so a caller can confirm it
    for whatever plant is in use.

    2002 writes this in its ``theta`` grouping, where ``theta4 = b1 / g`` and
    ``theta5 = b2 / g`` are mass moments rather than gravity torques; the
    gravity factors are divided out here so the value matches the published one.
    """
    a1, a2, a3 = params.a1, params.a2, params.a3
    t4, t5 = params.b1 / params.gravity, params.b2 / params.gravity
    alpha0 = (2.0 * a3 * t4 + a1 * t5) / (3.0 * a3 * t5)
    alpha1 = a2 * t4 - a3 * t5 + a3 * t4 - a1 * t5
    alpha2 = a2 * t4 - a3 * t5 - a3 * t4 + a1 * t5
    if alpha0 > 1.0:
        return float(alpha1**2 + alpha2**2)
    if alpha0 == 1.0:
        return float(alpha2)
    return float(
        2.0 * a2 * t4
        + 2.0 * a3 * t5
        - (a3 * t4 - a1 * t5) * alpha0
        - 3.0 * a3 * t5 * alpha0**2
    )


def assert_admissible(params: AcrobotParams, gains: Gains) -> None:
    """Raise unless the gains satisfy the 2007 conditions for this plant."""
    if alpha(params) == 0.0:
        raise ValueError(
            "mechanical condition (31) fails for this plant: alpha = 0"
        )
    floor_d = kd_min(params)
    if gains.k_d <= floor_d:
        raise ValueError(
            f"k_D must exceed condition (25) = {floor_d:.6f}, got {gains.k_d}"
        )
    floor_p = kp_min(params)
    if gains.k_p <= floor_p:
        raise ValueError(
            f"k_P must exceed condition (43) = {floor_p:.6f}, got {gains.k_p}"
        )


def homoclinic_speed(params: AcrobotParams) -> float:
    """Peak ``|qdot1|`` on the homoclinic orbit, at the hanging pose.

    From eq. (32) of 2007 at ``sin q1 = -1``:
    ``sqrt(4 (b1 + b2) / (a1 + a2 + 2 a3))``.
    """
    return float(
        np.sqrt(
            4.0
            * (params.b1 + params.b2)
            / (params.a1 + params.a2 + 2.0 * params.a3)
        )
    )


def asymptotic_torque_bound(params: AcrobotParams) -> float:
    """Bound (69) of 2007 on ``|tau2|`` as the orbit is approached.

    ``(a2 b1 + a3 b1 - a1 b2 - a3 b2) / (a1 + a2 + 2 a3)``, free of the gains
    and of the initial state.
    """
    a1, a2, a3, b1, b2 = params.a1, params.a2, params.a3, params.b1, params.b2
    return float(
        (a2 * b1 + a3 * b1 - a1 * b2 - a3 * b2) / (a1 + a2 + 2.0 * a3)
    )


# --- The control law ------------------------------------------------------


def torque(params: AcrobotParams, gains: Gains, state: np.ndarray) -> float:
    """Law (18) of 2007 / (14) of 2002, in paper coordinates.

    ``state`` is ``[q1, q2, qdot1, qdot2]``.  Returns the elbow torque.
    """
    q = np.asarray(state[:2], dtype=np.float64)
    qdot = np.asarray(state[2:], dtype=np.float64)
    mass = params.mass_matrix(q[1])
    bias = params.bias(q, qdot)
    det = mass[0, 0] * mass[1, 1] - mass[0, 1] * mass[1, 0]
    energy_error = params.energy(q, qdot) - params.energy_top
    denominator = gains.k_d * mass[0, 0] + energy_error * det
    if not np.isfinite(denominator) or abs(denominator) < 1e-9:
        raise FloatingPointError(
            "Xin-Kaneda control law hit its singularity: k_D M11 + (E - E_r) "
            f"det M = {denominator}. Condition (25) is what rules this out; "
            f"k_D = {gains.k_d}, its floor is {kd_min(params):.6f}."
        )
    numerator = (gains.k_v * qdot[1] + gains.k_p * q[1]) * det + gains.k_d * (
        mass[1, 0] * bias[0] - mass[0, 0] * bias[1]
    )
    return float(-numerator / denominator)


def torque_batch(
    params: AcrobotParams, gains: Gains, states: np.ndarray
) -> np.ndarray:
    """:func:`torque` evaluated over a batch, ``states`` of shape ``(N, 4)``.

    Rows sitting on the law's singularity come back as ``nan`` rather than
    raising, so a caller sweeping states it did not choose -- a replay buffer,
    say -- can drop those rows instead of losing the whole batch.  Whenever a
    row is admissible the value matches :func:`torque` to floating-point.
    """
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(
            f"expected states of shape (N, 4) [q1, q2, qd1, qd2], got {values.shape}"
        )
    q1, q2, d1, d2 = values[:, 0], values[:, 1], values[:, 2], values[:, 3]
    c2, s2 = np.cos(q2), np.sin(q2)

    m11 = params.a1 + params.a2 + 2.0 * params.a3 * c2
    m12 = params.a2 + params.a3 * c2
    m22 = np.full_like(m11, params.a2)
    det = params.det_m(q2)

    bias1 = params.a3 * s2 * (-2.0 * d1 * d2 - d2**2) + (
        params.b1 * np.cos(q1) + params.b2 * np.cos(q1 + q2)
    )
    bias2 = params.a3 * s2 * d1**2 + params.b2 * np.cos(q1 + q2)

    kinetic = 0.5 * (m11 * d1**2 + 2.0 * m12 * d1 * d2 + m22 * d2**2)
    potential = params.b1 * np.sin(q1) + params.b2 * np.sin(q1 + q2)
    energy_error = kinetic + potential - params.energy_top

    denominator = gains.k_d * m11 + energy_error * det
    numerator = (gains.k_v * d2 + gains.k_p * q2) * det + gains.k_d * (
        m12 * bias1 - m11 * bias2
    )
    singular = ~np.isfinite(denominator) | (np.abs(denominator) < 1e-9)
    result = np.divide(
        -numerator,
        denominator,
        out=np.full_like(denominator, np.nan),
        where=~singular,
    )
    return np.where(singular, np.nan, result)


def lyapunov(params: AcrobotParams, gains: Gains, state: np.ndarray) -> float:
    """``V = 1/2 (E - E_r)^2 + 1/2 k_D qdot2^2 + 1/2 k_P q2^2``, eq. (11)."""
    q = np.asarray(state[:2], dtype=np.float64)
    qdot = np.asarray(state[2:], dtype=np.float64)
    energy_error = params.energy(q, qdot) - params.energy_top
    return float(
        0.5 * energy_error**2
        + 0.5 * gains.k_d * qdot[1] ** 2
        + 0.5 * gains.k_p * q[1] ** 2
    )


def closed_loop(
    params: AcrobotParams,
    gains: Gains,
    state: np.ndarray,
    *,
    damping: float = 0.0,
    torque_limit: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """Closed-loop drift in paper coordinates, and the applied torque.

    ``damping`` adds ``-damping * qdot`` as a passive joint force; it is zero in
    the papers, and any positive value voids their Lyapunov argument (the
    identity ``Edot = qdot2 tau2`` becomes ``Edot = qdot2 tau2 - damping |qdot|^2``).
    """
    commanded = torque(params, gains, state)
    applied = (
        commanded
        if torque_limit is None
        else float(np.clip(commanded, -torque_limit, torque_limit))
    )
    q = np.asarray(state[:2], dtype=np.float64)
    qdot = np.asarray(state[2:], dtype=np.float64)
    forcing = np.array([0.0, applied]) - params.bias(q, qdot) - damping * qdot
    qddot = np.linalg.solve(params.mass_matrix(q[1]), forcing)
    return np.concatenate([qdot, qddot]), commanded


def hanging_jacobian(params: AcrobotParams, gains: Gains) -> np.ndarray:
    """Jacobian of the closed loop at the hanging equilibrium ``(-pi/2, 0, 0, 0)``.

    Hanging is an exact equilibrium of the closed loop: the gravity torques
    vanish there and ``q2 = qdot2 = 0``, so the law commands zero.  Its spectrum
    is what 2007 Proposition 5 classifies.
    """
    base = np.array([-0.5 * np.pi, 0.0, 0.0, 0.0])
    step = 1e-5
    jacobian = np.zeros((4, 4), dtype=np.float64)
    for i in range(4):
        offset = np.zeros(4)
        offset[i] = step
        forward, _ = closed_loop(params, gains, base + offset)
        backward, _ = closed_loop(params, gains, base - offset)
        jacobian[:, i] = (forward - backward) / (2.0 * step)
    return jacobian


def hanging_regime(params: AcrobotParams, gains: Gains) -> dict:
    """Classify the hanging equilibrium per 2007 Proposition 5.

    Returns the number of right-half-plane eigenvalues, the dominant real part,
    the implied escape time constant, and which of the two papers' regimes the
    gains sit in.
    """
    eigenvalues = np.linalg.eigvals(hanging_jacobian(params, gains))
    dominant = float(np.max(eigenvalues.real))
    boundary = kp_boundary(params)
    if gains.k_p < boundary:
        regime = "prop4_fast"
    elif gains.k_p > boundary:
        regime = "corollary1_slow"
    else:
        regime = "boundary"
    return {
        "regime": regime,
        "n_unstable": int(np.sum(eigenvalues.real > 1e-9)),
        "max_real": dominant,
        "escape_time_constant": float("inf") if dominant <= 0.0 else 1.0 / dominant,
        "kp_boundary": float(boundary),
        "eigenvalues": eigenvalues,
    }


# --- Frame adapter --------------------------------------------------------


def obs_to_paper(obs: np.ndarray) -> np.ndarray:
    """Map an *upward-vertical* raw-state observation into paper coordinates.

    The stock dm_control Acrobot measures the shoulder from the upward vertical,
    so ``obs = [q1, q2, qdot1, qdot2]`` there becomes
    ``[pi/2 - q1, -q2, -qdot1, -qdot2]``, with the elbow torque flipping to
    match.  ``acrobot-swingup-xk`` is built directly in the paper's frame and
    needs no such map; this exists for the stock model.
    """
    values = np.asarray(obs, dtype=np.float64).reshape(-1)
    if values.shape != (4,):
        raise ValueError(f"expected a 4-vector [q1, q2, qd1, qd2], got {values.shape}")
    return np.array(
        [
            0.5 * np.pi - values[0],
            -values[1],
            -values[2],
            -values[3],
        ]
    )


# How a plant's raw state relates to the paper's coordinates.  ``paper`` is the
# identity, and is what ``acrobot-swingup-xk`` provides.
FRAMES = ("paper", "upward_vertical")


class XinKanedaController:
    """The energy-based swing-up law as an ``act(obs) -> action`` callable.

    Consumes dm_control raw-state observations and emits a normalized command in
    ``[-1, 1]``, so the evaluation harness can score this controller and a
    learned policy through one code path.

    ``frame`` says how the observation relates to the paper's coordinates:
    ``"paper"`` (the default, and what ``acrobot-swingup-xk`` provides) is the
    identity, while ``"upward_vertical"`` applies :func:`obs_to_paper` and flips
    the commanded torque, which is what the stock dm_control Acrobot needs.

    ``torque_limit`` describes the plant's actuator: the command is clipped to
    it before being normalized by ``params.gear``.  Passing ``None`` takes the
    limit from ``params.gear``, which is how the dm_control model expresses it.
    Clipping voids the Lyapunov argument, so :attr:`saturated_steps` tracks how
    often it bound.
    """

    def __init__(
        self,
        params: AcrobotParams,
        gains: Gains,
        *,
        torque_limit: Optional[float] = None,
        frame: str = "paper",
        check_admissible: bool = True,
    ) -> None:
        if frame not in FRAMES:
            raise ValueError(f"frame must be one of {FRAMES}, got {frame!r}")
        self.frame = frame
        if check_admissible:
            assert_admissible(params, gains)
        self.params = params
        self.gains = gains
        self.torque_limit = (
            float(params.gear) if torque_limit is None else float(torque_limit)
        )
        if not np.isfinite(self.torque_limit) or self.torque_limit <= 0.0:
            raise ValueError(
                f"torque_limit must be finite and > 0, got {self.torque_limit}"
            )
        # The plant applies tau = gear * ctrl with ctrl in [-1, 1], so a limit
        # above the gear could not be commanded and would silently be clipped
        # again by the actuator.
        if self.torque_limit > float(params.gear) * (1.0 + 1e-12):
            raise ValueError(
                f"torque_limit {self.torque_limit} exceeds what the plant can "
                f"apply (gear = {params.gear}); raise the model's gear instead"
            )
        self.reset()

    def reset(self) -> None:
        self.last_torque = 0.0
        self.last_commanded_torque = 0.0
        self.last_energy_error = 0.0
        self.last_lyapunov = 0.0
        self.saturated_steps = 0
        self.steps = 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        state = (
            np.asarray(obs, dtype=np.float64).reshape(-1)
            if self.frame == "paper"
            else obs_to_paper(obs)
        )
        commanded = torque(self.params, self.gains, state)
        applied = float(np.clip(commanded, -self.torque_limit, self.torque_limit))
        self.last_commanded_torque = commanded
        self.last_torque = applied
        self.last_energy_error = (
            self.params.energy(state[:2], state[2:]) - self.params.energy_top
        )
        self.last_lyapunov = lyapunov(self.params, self.gains, state)
        self.steps += 1
        if abs(commanded) > self.torque_limit:
            self.saturated_steps += 1
        # The plant takes a normalized command scaled by its gear.  In the
        # upward-vertical frame the reflection also flips the torque sign.
        sign = 1.0 if self.frame == "paper" else -1.0
        return np.array([sign * applied / self.params.gear], dtype=np.float64)

    def actions(self, obs: np.ndarray) -> np.ndarray:
        """Normalized commands for a batch of observations, ``(N, 4) -> (N, 1)``.

        The batched counterpart of :meth:`__call__`, for callers that score many
        states at once -- CT-SAC's imitation loss reads the law through here.
        Two differences follow from those states not being ones this controller
        drove: the saturation bookkeeping is left alone, since it counts what
        was actually commanded on a trajectory, and states on the law's
        singularity yield ``nan`` instead of raising, for the caller to mask.
        """
        values = np.asarray(obs, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValueError(
                f"expected observations of shape (N, 4), got {values.shape}"
            )
        if self.frame != "paper":
            values = np.column_stack(
                [
                    0.5 * np.pi - values[:, 0],
                    -values[:, 1],
                    -values[:, 2],
                    -values[:, 3],
                ]
            )
        commanded = torque_batch(self.params, self.gains, values)
        applied = np.clip(commanded, -self.torque_limit, self.torque_limit)
        sign = 1.0 if self.frame == "paper" else -1.0
        return (sign * applied / self.params.gear).reshape(-1, 1)

    @property
    def saturation_fraction(self) -> float:
        return 0.0 if self.steps == 0 else self.saturated_steps / self.steps
