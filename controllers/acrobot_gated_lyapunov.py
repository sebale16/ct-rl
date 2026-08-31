"""Gated Xin--Kaneda/LQR Lyapunov constructions for the Acrobot.

The Xin--Kaneda function is a global set-stabilization certificate: its zero
set is the target homoclinic orbit.  An LQR value is instead locally positive
definite about upright rest.  This module constructs hand-offs between the two
without changing any environment reward by merely importing it.

Two of them live here.  :class:`GatedLyapunov` blends the pieces on the
Xin--Kaneda switching test with each piece normalized on its own scale, and
carries the reward ridge described under *Important limitation* below.
:class:`NonsmoothLyapunov` is the published repair: the nonsmooth Lyapunov
function of Lai, Wu, She and Yang, which puts both pieces on one scale and
offsets the outer piece so that crossing the gate can only step the value down.

## Gated candidate

The 2007 paper declares the local-controller switching region through

    |e1| + |e2| + 0.1 |e3| + 0.1 |e4| < zeta,  zeta = 0.04,

where ``e = [q1 - pi/2, q2, qdot1, qdot2]``.  The gate uses a differentiable,
conservative approximation of that residual: each ``|w_i e_i|`` is replaced
by ``sqrt((w_i e_i)^2 + epsilon^2)``.  The smooth residual upper-bounds the
printed one, so the active gate remains inside the published switching region.
A quintic smootherstep changes from the Xin--Kaneda value outside the 0.04
boundary to the normalized LQR value inside a 0.02 boundary.

This is deliberately called a *candidate*.  Smoothly combining two Lyapunov
functions does not by itself prove that the result decreases under a blended
controller.  :meth:`GatedLyapunov.rate` includes the gate-gradient term needed
to test that property under any supplied state derivative.

## Nonsmooth Lyapunov function

Lai, Wu, She and Yang build one Lyapunov function for the whole motion space
out of a swing-up piece and a local Riccati piece, and require the switched
function to decrease across the switching surface.  Their equation (71) meets
that requirement by adding to the swing-up piece a constant ``Delta`` equal to
the largest local value the attractive region admits, so the outer piece
dominates the inner one everywhere the switch can happen.

The region is their equation (17) rather than the 2007 switching test: two
angle conditions and an energy band, with speed entering through the energy
alone.  ``docs/acrobot_xk_gated_lyapunov.md`` records the sweeps behind the
default tolerances, which tighten the printed angle box.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import solve_continuous_are

from .xin_kaneda import AcrobotParams, Gains, XinKanedaController, _refine


UPRIGHT_STATE = np.array([0.5 * np.pi, 0.0, 0.0, 0.0], dtype=np.float64)
LQR_SWITCH_WEIGHTS = np.array([1.0, 1.0, 0.1, 0.1], dtype=np.float64)

# Lai et al. (2009) equation (17) prints eps1 = eps2 = pi/6 and eps_E = 1 J.
# The angle tolerance is tightened here; ``docs/acrobot_xk_gated_lyapunov.md``
# records the sweep that motivates it.
LAI_ANGLE_TOLERANCE = np.pi / 30.0
LAI_ENERGY_TOLERANCE = 1.0


def _wrap(angle):
    """Map an angle to ``(-pi, pi]``."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def _energy_and_gradient(
    params: AcrobotParams, state: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Mechanical energy and its exact state gradient in paper coordinates."""
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (4,):
        raise ValueError(f"state must have shape (4,), got {state.shape}")
    q1, q2, d1, d2 = state
    velocity = state[2:]
    gradient = np.empty(4, dtype=np.float64)
    gradient[0] = params.b1 * np.cos(q1) + params.b2 * np.cos(q1 + q2)
    gradient[1] = (
        -params.a3 * np.sin(q2) * (d1**2 + d1 * d2)
        + params.b2 * np.cos(q1 + q2)
    )
    gradient[2:] = params.mass_matrix(q2) @ velocity
    return float(params.energy(state[:2], velocity)), gradient


def plant_drift_and_gain(
    params: AcrobotParams, state: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Split the undamped paper plant into ``xdot = f(x) + g(x) tau``."""
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (4,):
        raise ValueError(f"state must have shape (4,), got {state.shape}")
    mass = params.mass_matrix(state[1])
    drift = np.concatenate(
        [state[2:], np.linalg.solve(mass, -params.bias(state[:2], state[2:]))]
    )
    gain = np.concatenate(
        [np.zeros(2), np.linalg.solve(mass, np.array([0.0, 1.0]))]
    )
    return drift, gain


def upright_error(state: np.ndarray) -> np.ndarray:
    """Return the wrapped error to upright in the Xin--Kaneda paper frame."""
    value = np.asarray(state, dtype=np.float64)
    if value.shape != (4,):
        raise ValueError(f"state must have shape (4,), got {value.shape}")
    error = value - UPRIGHT_STATE
    error[:2] = _wrap(error[:2])
    return error


def lqr_switch_residual(
    state: np.ndarray, weights: np.ndarray = LQR_SWITCH_WEIGHTS
) -> float:
    """Equation (74)'s weighted-L1 residual to upright rest."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (4,) or np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("weights must contain four finite positive values")
    return float(weights @ np.abs(upright_error(state)))


def upright_linearization(params: AcrobotParams) -> Tuple[np.ndarray, np.ndarray]:
    """Continuous-time ``(A, B)`` at ``q = (pi/2, 0), qdot = 0``.

    The input is the physical elbow torque, not the normalized MuJoCo command.
    """
    mass = params.mass_matrix(0.0)
    gravity_stiffness = np.array(
        [[params.b1 + params.b2, params.b2], [params.b2, params.b2]],
        dtype=np.float64,
    )
    acceleration_position = np.linalg.solve(mass, gravity_stiffness)
    acceleration_input = np.linalg.solve(mass, np.array([[0.0], [1.0]]))
    a = np.block(
        [
            [np.zeros((2, 2)), np.eye(2)],
            [acceleration_position, np.zeros((2, 2))],
        ]
    )
    b = np.vstack([np.zeros((2, 1)), acceleration_input])
    return a, b


@dataclass(frozen=True)
class LQRDesign:
    """Design choices for the local Riccati value and its gate.

    ``Q = I`` and ``R = 0.5`` match the local design used for this same
    published Acrobot parameter set in Lai et al. (2009); Xin--Kaneda specify
    the switching test but not a unique Riccati cost.
    """

    q: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    r: float = 0.5
    switch_threshold: float = 0.04
    inner_switch_threshold: float = 0.02
    smooth_abs_epsilon: float = 1e-6
    switch_weights: tuple[float, float, float, float] = (1.0, 1.0, 0.1, 0.1)

    def __post_init__(self) -> None:
        if len(self.q) != 4 or any(not np.isfinite(x) or x <= 0 for x in self.q):
            raise ValueError("q must contain four finite positive values")
        if not np.isfinite(self.r) or self.r <= 0:
            raise ValueError("r must be finite and positive")
        if not np.isfinite(self.switch_threshold) or self.switch_threshold <= 0:
            raise ValueError("switch_threshold must be finite and positive")
        if (
            not np.isfinite(self.inner_switch_threshold)
            or not 0 < self.inner_switch_threshold < self.switch_threshold
        ):
            raise ValueError(
                "inner_switch_threshold must lie strictly between zero and "
                "switch_threshold"
            )
        if not np.isfinite(self.smooth_abs_epsilon) or self.smooth_abs_epsilon <= 0:
            raise ValueError("smooth_abs_epsilon must be finite and positive")
        if 4.0 * self.smooth_abs_epsilon >= self.inner_switch_threshold:
            raise ValueError(
                "four smooth_abs_epsilon terms must sum to less than "
                "inner_switch_threshold"
            )
        if len(self.switch_weights) != 4 or any(
            not np.isfinite(x) or x <= 0 for x in self.switch_weights
        ):
            raise ValueError("switch_weights must contain four finite positive values")


def riccati_feedback(
    a: np.ndarray,
    b: np.ndarray,
    q: tuple[float, float, float, float],
    r: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the CARE for ``(A, B)`` and return ``(K, P)``.

    ``K = R^-1 B^T P`` is the gain of equation (46), and ``P`` is symmetrized
    before it is returned so that quadratic forms built on it are exact.  Both
    this module and :mod:`controllers.lai_she` reach the Riccati step through
    here; they keep their own plants and linearizations, which differ in the
    inertia the two papers print.
    """
    weights = np.diag(np.asarray(q, dtype=np.float64))
    cost = np.array([[float(r)]], dtype=np.float64)
    p = solve_continuous_are(np.asarray(a), np.asarray(b), weights, cost)
    p = 0.5 * (p + p.T)
    return np.linalg.solve(cost, np.asarray(b).T @ p), p


def lqr_solution(
    params: AcrobotParams, design: LQRDesign = LQRDesign()
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(A, B, K, P)`` with the convention ``tau = -K e``."""
    a, b = upright_linearization(params)
    k, p = riccati_feedback(a, b, design.q, design.r)
    return a, b, k, p


def lqr_scale_on_switch_region(
    p: np.ndarray,
    switch_threshold: float = 0.04,
    weights: np.ndarray = LQR_SWITCH_WEIGHTS,
) -> float:
    """Maximum ``e.T P e`` over the paper's weighted-L1 switching region.

    A convex quadratic attains its maximum over the weighted-L1 polytope at a
    vertex ``e = +/- switch_threshold / weight_i``.  Dividing the local value
    by this scale therefore places every state in the paper region in ``[0, 1]``.
    """
    p = np.asarray(p, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if p.shape != (4, 4) or not np.all(np.isfinite(p)):
        raise ValueError("p must be a finite 4-by-4 matrix")
    if weights.shape != (4,) or np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("weights must contain four finite positive values")
    if not np.isfinite(switch_threshold) or switch_threshold <= 0:
        raise ValueError("switch_threshold must be finite and positive")
    eigenvalues = np.linalg.eigvalsh(0.5 * (p + p.T))
    if np.min(eigenvalues) <= 0:
        raise ValueError("p must be positive definite")

    return float(switch_threshold**2 * np.max(np.diag(p) / weights**2))


def _smootherstep(value: float) -> Tuple[float, float]:
    """Return quintic smootherstep and its derivative on the real line."""
    if value <= 0.0:
        return 0.0, 0.0
    if value >= 1.0:
        return 1.0, 0.0
    weight = value**3 * (value * (6.0 * value - 15.0) + 10.0)
    derivative = 30.0 * value**2 * (value - 1.0) ** 2
    return float(weight), float(derivative)


@dataclass(frozen=True)
class GatedLyapunov:
    """Normalized Xin--Kaneda/LQR candidate and its exact state gradient."""

    params: AcrobotParams
    gains: Gains
    design: LQRDesign = field(default_factory=LQRDesign)
    a: np.ndarray = field(init=False, repr=False)
    b: np.ndarray = field(init=False, repr=False)
    k: np.ndarray = field(init=False, repr=False)
    p: np.ndarray = field(init=False, repr=False)
    lqr_scale: float = field(init=False)
    xk_scale: float = field(init=False)

    def __post_init__(self) -> None:
        a, b, k, p = lqr_solution(self.params, self.design)
        weights = np.asarray(self.design.switch_weights, dtype=np.float64)
        lqr_scale = lqr_scale_on_switch_region(
            p, self.design.switch_threshold, weights
        )
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "p", p)
        object.__setattr__(self, "lqr_scale", lqr_scale)
        object.__setattr__(self, "xk_scale", 0.5 * self.params.energy_span**2)

    def lqr_value(self, state: np.ndarray) -> float:
        """Raw local Riccati value ``e.T P e``."""
        error = upright_error(state)
        return float(error @ self.p @ error)

    def lqr_value_normalized(self, state: np.ndarray) -> float:
        """Riccati value scaled to at most one in the paper's switch region."""
        return self.lqr_value(state) / self.lqr_scale

    def xk_value(self, state: np.ndarray) -> float:
        """Xin--Kaneda value normalized to one at hanging rest."""
        value, _ = self._xk_value_and_gradient(state)
        return value

    def gate(self, state: np.ndarray) -> float:
        """LQR membership: one inside 0.02 and zero outside the 0.04 region."""
        residual, _ = self._smooth_residual_and_gradient(state)
        inner = self.design.inner_switch_threshold
        outer = self.design.switch_threshold
        coordinate = (outer - residual) / (outer - inner)
        return _smootherstep(coordinate)[0]

    def value(self, state: np.ndarray) -> float:
        """Return ``(1-mu) V_XK_bar + mu V_LQR_bar``."""
        xk_value = self.xk_value(state)
        lqr_value = self.lqr_value_normalized(state)
        membership = self.gate(state)
        return float((1.0 - membership) * xk_value + membership * lqr_value)

    def value_and_gradient(self, state: np.ndarray) -> Tuple[float, np.ndarray]:
        """Return the gated value and its exact gradient with respect to state."""
        xk_value, xk_gradient = self._xk_value_and_gradient(state)
        error = upright_error(state)
        lqr_value = float(error @ self.p @ error) / self.lqr_scale
        lqr_gradient = 2.0 * (self.p @ error) / self.lqr_scale

        residual, residual_gradient = self._smooth_residual_and_gradient(state)
        inner = self.design.inner_switch_threshold
        outer = self.design.switch_threshold
        coordinate = (outer - residual) / (outer - inner)
        membership, smootherstep_derivative = _smootherstep(coordinate)
        membership_gradient = (
            -smootherstep_derivative / (outer - inner) * residual_gradient
        )
        gradient = (
            (1.0 - membership) * xk_gradient
            + membership * lqr_gradient
            + (lqr_value - xk_value) * membership_gradient
        )
        value = (1.0 - membership) * xk_value + membership * lqr_value
        return float(value), gradient

    def rate(self, state: np.ndarray, state_derivative: np.ndarray) -> float:
        """Directional derivative, including the derivative of the smooth gate."""
        derivative = np.asarray(state_derivative, dtype=np.float64)
        if derivative.shape != (4,):
            raise ValueError(
                f"state_derivative must have shape (4,), got {derivative.shape}"
            )
        _, gradient = self.value_and_gradient(state)
        return float(gradient @ derivative)

    def lqr_torque(self, state: np.ndarray) -> float:
        """Local feedback associated with the Riccati value: ``tau = -K e``."""
        return float(-(self.k @ upright_error(state))[0])

    def smooth_switch_residual(self, state: np.ndarray) -> float:
        """Differentiable upper bound on equation (74)'s exact residual."""
        return self._smooth_residual_and_gradient(state)[0]

    def _smooth_residual_and_gradient(
        self, state: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        error = upright_error(state)
        weights = np.asarray(self.design.switch_weights, dtype=np.float64)
        epsilon = self.design.smooth_abs_epsilon
        weighted = weights * error
        roots = np.sqrt(weighted**2 + epsilon**2)
        gradient = weights * weighted / roots
        return float(np.sum(roots)), gradient

    def _xk_value_and_gradient(
        self, state: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        raw_value, raw_gradient = _xk_raw_value_and_gradient(
            self.params, self.gains, state
        )
        return raw_value / self.xk_scale, raw_gradient / self.xk_scale


def _xk_raw_value_and_gradient(
    params: AcrobotParams, gains: Gains, state: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Unnormalized Xin--Kaneda value and its exact state gradient."""
    state = np.asarray(state, dtype=np.float64)
    energy, energy_gradient = _energy_and_gradient(params, state)
    _, q2, _, d2 = state
    energy_error = energy - params.energy_top
    value = (
        0.5 * energy_error**2
        + 0.5 * gains.k_d * d2**2
        + 0.5 * gains.k_p * q2**2
    )
    gradient = energy_error * energy_gradient
    gradient[1] += gains.k_p * q2
    gradient[3] += gains.k_d * d2
    return float(value), gradient


# --- Lai et al. nonsmooth Lyapunov function -------------------------------


@dataclass(frozen=True)
class AttractiveRegion:
    """Lai et al. (2009) equation (17): the attractive area ``Sigma_2``.

    Membership is four conditions on the upright error and the energy,

        |x1| <= eps_1,   |x1 + x2| <= eps_2,
        ||(w3 x3, w4 x4)|| <= eps_5,   |E - E_top| <= eps_E,

    both angles wrapped.  The velocity condition is vacuous at the paper's own
    weights, so speed enters only through the energy band -- which is what
    makes the region reachable by a swing-up that regulates energy.

    The conditions read the same in either coordinate frame.  The 2009 paper
    measures the shoulder from upright, ``x = [x1, x2, x3, x4]``, while this
    repository's Xin--Kaneda plant measures it from the horizontal; the two are
    related by ``x = -e`` with ``e`` the wrapped upright error, and every
    condition above is even in its argument.  :meth:`residual_of` therefore
    takes the four scalars directly and serves both frames, with
    :meth:`exact_residual` the convenience wrapper for the Xin--Kaneda frame.

    ``transition_fraction`` places the inner boundary, inside which the gate is
    fully local, at that fraction of the outer boundary.
    """

    angle_tolerance: float = LAI_ANGLE_TOLERANCE
    #: Tolerance on the second angle condition; ``None`` reuses the first.
    tip_tolerance: Optional[float] = None
    energy_tolerance: float = LAI_ENERGY_TOLERANCE
    velocity_weights: Tuple[float, float] = (1e-3, 1e-3)
    velocity_tolerance: float = 1e3
    transition_fraction: float = 0.5
    smooth_abs_epsilon: float = 1e-9
    norm_order: float = 8.0

    def __post_init__(self) -> None:
        for name in (
            "angle_tolerance",
            "energy_tolerance",
            "velocity_tolerance",
            "smooth_abs_epsilon",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.tip_tolerance is not None and (
            not np.isfinite(self.tip_tolerance) or self.tip_tolerance <= 0.0
        ):
            raise ValueError("tip_tolerance must be finite and positive")
        if len(self.velocity_weights) != 2 or any(
            not np.isfinite(w) or w <= 0.0 for w in self.velocity_weights
        ):
            raise ValueError("velocity_weights must contain two positive values")
        if not 0.0 < self.transition_fraction < 1.0:
            raise ValueError("transition_fraction must lie strictly in (0, 1)")
        if not np.isfinite(self.norm_order) or self.norm_order < 1.0:
            raise ValueError("norm_order must be finite and at least one")

    @property
    def effective_tip_tolerance(self) -> float:
        """``eps_2``, defaulting to ``eps_1`` as the paper takes it."""
        return (
            self.angle_tolerance if self.tip_tolerance is None else self.tip_tolerance
        )

    @property
    def scales(self) -> np.ndarray:
        """Denominators of the four normalized conditions."""
        return np.array(
            [
                self.angle_tolerance,
                self.effective_tip_tolerance,
                self.energy_tolerance,
                self.velocity_tolerance,
            ],
            dtype=np.float64,
        )

    def residual_of(
        self,
        shoulder_error: float,
        tip_error: float,
        energy_error: float,
        velocity: np.ndarray,
    ) -> float:
        """Largest of the four normalized violations; ``<= 1`` inside.

        The angle arguments must already be wrapped.  This is the single home
        of equation (17); both coordinate frames reach the region through it.
        """
        velocity = np.asarray(velocity, dtype=np.float64).reshape(-1)
        if velocity.shape != (2,):
            raise ValueError(f"velocity must have shape (2,), got {velocity.shape}")
        weighted = np.asarray(self.velocity_weights, dtype=np.float64) * velocity
        scales = self.scales
        return float(
            max(
                abs(shoulder_error) / scales[0],
                abs(tip_error) / scales[1],
                abs(energy_error) / scales[2],
                np.linalg.norm(weighted) / scales[3],
            )
        )

    def contains(self, params: AcrobotParams, state: np.ndarray) -> bool:
        """Exact membership test of equation (17) in the Xin--Kaneda frame."""
        return self.exact_residual(params, state) <= 1.0

    def exact_residual(self, params: AcrobotParams, state: np.ndarray) -> float:
        """:meth:`residual_of` evaluated on a Xin--Kaneda-frame state."""
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (4,):
            raise ValueError(f"state must have shape (4,), got {state.shape}")
        q1, q2 = float(state[0]), float(state[1])
        return self.residual_of(
            _wrap(q1 - 0.5 * np.pi),
            _wrap(q1 + q2 - 0.5 * np.pi),
            params.energy(state[:2], state[2:]) - params.energy_top,
            state[2:],
        )

    def smooth_residual_and_gradient(
        self, params: AcrobotParams, state: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Differentiable upper bound on :meth:`exact_residual`.

        Each ``|.|`` is smoothed, the velocity norm is smoothed the same way,
        and the maximum is replaced by a ``p``-norm.  Every replacement only
        ever raises the residual, so any state given nonzero local membership
        lies strictly inside the printed region.
        """
        energy, energy_gradient = _energy_and_gradient(params, state)
        state = np.asarray(state, dtype=np.float64)
        q1, q2 = float(state[0]), float(state[1])
        scales = self.scales
        epsilon = self.smooth_abs_epsilon

        signed = np.array(
            [
                _wrap(q1 - 0.5 * np.pi),
                _wrap(q1 + q2 - 0.5 * np.pi),
                energy - params.energy_top,
            ]
        )
        jacobian = np.zeros((3, 4), dtype=np.float64)
        jacobian[0, 0] = 1.0
        jacobian[1, 0] = 1.0
        jacobian[1, 1] = 1.0
        jacobian[2] = energy_gradient
        signed = signed / scales[:3]
        jacobian = jacobian / scales[:3, None]
        magnitudes = np.sqrt(signed**2 + epsilon**2)
        gradients = (signed / magnitudes)[:, None] * jacobian

        weights = np.asarray(self.velocity_weights, dtype=np.float64)
        weighted = weights * state[2:]
        speed = np.sqrt(weighted @ weighted + epsilon**2)
        velocity_gradient = np.zeros(4, dtype=np.float64)
        velocity_gradient[2:] = weights**2 * state[2:] / (speed * scales[3])

        magnitudes = np.append(magnitudes, speed / scales[3])
        gradients = np.vstack([gradients, velocity_gradient])

        order = self.norm_order
        residual = float(np.sum(magnitudes**order) ** (1.0 / order))
        if residual == 0.0:
            return 0.0, np.zeros(4, dtype=np.float64)
        gradient = residual ** (1.0 - order) * ((magnitudes ** (order - 1.0)) @ gradients)
        return residual, gradient


def max_local_value_on_region(
    params: AcrobotParams,
    p: np.ndarray,
    region: AttractiveRegion = AttractiveRegion(),
    *,
    angle_samples: int = 91,
    velocity_samples: int = 361,
) -> float:
    """``Delta``: the maximum of ``e.T P e`` over the attractive region.

    This is Lai et al.'s equation (71).  Their own choice is the crude bound
    ``sum |P_ij| x_i,max x_j,max`` of their equation (72), which at this
    module's default tolerances overshoots the true maximum by 5.7 and would
    put the offset almost six times above the whole range of the swing-up
    piece; the maximum itself is used instead.

    For each admissible pose the energy band caps the kinetic energy, so the
    velocities are confined to an ellipsoid and a convex quadratic attains its
    maximum on that ellipsoid's boundary.  The boundary is a circle after a
    Cholesky change of variables, and is searched on a grid with parabolic
    refinement; the poses themselves are gridded.
    """
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (4, 4) or not np.all(np.isfinite(p)):
        raise ValueError("p must be a finite 4-by-4 matrix")
    if angle_samples < 3 or velocity_samples < 3:
        raise ValueError("angle_samples and velocity_samples must exceed two")

    shoulders = np.linspace(
        -region.angle_tolerance, region.angle_tolerance, angle_samples
    )
    tips = np.linspace(
        -region.effective_tip_tolerance,
        region.effective_tip_tolerance,
        angle_samples,
    )
    angles = np.linspace(0.0, 2.0 * np.pi, velocity_samples)
    circle = np.stack([np.cos(angles), np.sin(angles)])
    best = 0.0
    for shoulder in shoulders:
        q1 = 0.5 * np.pi + shoulder
        for tip in tips:
            q2 = tip - shoulder
            potential = params.b1 * np.sin(q1) + params.b2 * np.sin(q1 + q2)
            kinetic = params.energy_top + region.energy_tolerance - potential
            if kinetic <= 0.0:
                continue
            factor = np.linalg.cholesky(params.mass_matrix(q2))
            radius = np.sqrt(2.0 * kinetic)
            pose = np.array([shoulder, q2])

            def values(directions, factor=factor, radius=radius, pose=pose):
                velocities = np.linalg.solve(factor.T, radius * directions)
                errors = np.vstack([np.tile(pose[:, None], velocities.shape[1]), velocities])
                return np.einsum("ij,ik,kj->j", errors, p, errors)

            def value(angle, values=values):
                direction = np.array([[np.cos(angle)], [np.sin(angle)]])
                return float(values(direction)[0])

            samples = values(circle)
            index = int(np.argmax(samples))
            candidate = max(
                float(samples[index]), _refine(value, angles, index, maximize=True)
            )
            best = max(best, candidate)
    return best


@dataclass(frozen=True)
class NonsmoothLyapunov:
    """Lai et al.'s nonsmooth Lyapunov function for the Acrobot.

    Lai, Wu, She and Yang, *Comprehensive Unified Control Strategy for
    Underactuated Two-Link Manipulators*, IEEE Trans. SMC-B 39(2), 2009, build
    one Lyapunov function for the whole motion space out of two pieces: a
    swing-up piece (their equation 20) that is the Xin--Kaneda function plus a
    constant ``Delta``, and a local Riccati piece ``e.T P e`` (equation 44) on
    the attractive area.  Their Definition 3 requires the switched function to
    decrease across the switching surface, and equation (71) secures that by
    setting ``Delta`` to the largest local value the region admits.

    Two things follow, and both differ from :class:`GatedLyapunov`.  The pieces
    share one scale, the Xin--Kaneda value at hanging rest, so their levels are
    comparable at all.  And the outer piece carries ``Delta``, so it dominates
    the inner piece everywhere in the region: entering the gate can only step
    the value down.  That removes the ridge the earlier construction leaves
    along the homoclinic orbit.

    The constant shifts the value, never its gradient, so a reward built on
    this function has the same shaping during swing-up as one built on the
    bare Xin--Kaneda value.
    """

    params: AcrobotParams
    gains: Gains
    region: AttractiveRegion = field(default_factory=AttractiveRegion)
    design: LQRDesign = field(default_factory=LQRDesign)
    #: Skip the search for ``Delta`` and use this value.  Constructing the
    #: offset costs a second or so, which is worth avoiding when one process
    #: builds many identical copies; a value below the region's true maximum
    #: reinstates the ridge, so pass one obtained from a matching search.
    delta_override: Optional[float] = None
    a: np.ndarray = field(init=False, repr=False)
    b: np.ndarray = field(init=False, repr=False)
    k: np.ndarray = field(init=False, repr=False)
    p: np.ndarray = field(init=False, repr=False)
    scale: float = field(init=False)
    delta: float = field(init=False)

    def __post_init__(self) -> None:
        a, b, k, p = lqr_solution(self.params, self.design)
        if self.delta_override is None:
            delta = max_local_value_on_region(self.params, p, self.region)
        else:
            delta = float(self.delta_override)
            if not np.isfinite(delta) or delta <= 0.0:
                raise ValueError("delta_override must be finite and positive")
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "p", p)
        object.__setattr__(self, "scale", 0.5 * self.params.energy_span**2)
        object.__setattr__(self, "delta", delta)

    @property
    def normalized_delta(self) -> float:
        """``Delta`` as a fraction of the Xin--Kaneda value at hanging rest."""
        return self.delta / self.scale

    def swing_up_value(self, state: np.ndarray) -> float:
        """Outer piece: the Xin--Kaneda value plus ``Delta``, normalized."""
        raw, _ = _xk_raw_value_and_gradient(self.params, self.gains, state)
        return (raw + self.delta) / self.scale

    def local_value(self, state: np.ndarray) -> float:
        """Inner piece: ``e.T P e`` on the same scale as the outer piece."""
        error = upright_error(state)
        return float(error @ self.p @ error) / self.scale

    def gate(self, state: np.ndarray) -> float:
        """Local membership: one inside the inner boundary, zero outside."""
        residual, _ = self.region.smooth_residual_and_gradient(self.params, state)
        return self._membership_and_derivative(residual)[0]

    def value(self, state: np.ndarray) -> float:
        """Return ``(1-mu) (V_XK + Delta) / s + mu (e.T P e) / s``."""
        return self.value_and_gradient(state)[0]

    def value_and_gradient(self, state: np.ndarray) -> Tuple[float, np.ndarray]:
        """Return the switched value and its exact gradient with respect to state."""
        raw, raw_gradient = _xk_raw_value_and_gradient(self.params, self.gains, state)
        outer = (raw + self.delta) / self.scale
        outer_gradient = raw_gradient / self.scale

        error = upright_error(state)
        inner = float(error @ self.p @ error) / self.scale
        inner_gradient = 2.0 * (self.p @ error) / self.scale

        residual, residual_gradient = self.region.smooth_residual_and_gradient(
            self.params, state
        )
        membership, derivative = self._membership_and_derivative(residual)
        membership_gradient = derivative * residual_gradient

        value = (1.0 - membership) * outer + membership * inner
        gradient = (
            (1.0 - membership) * outer_gradient
            + membership * inner_gradient
            + (inner - outer) * membership_gradient
        )
        return float(value), gradient

    def rate(self, state: np.ndarray, state_derivative: np.ndarray) -> float:
        """Directional derivative, including the derivative of the smooth gate."""
        derivative = np.asarray(state_derivative, dtype=np.float64)
        if derivative.shape != (4,):
            raise ValueError(
                f"state_derivative must have shape (4,), got {derivative.shape}"
            )
        _, gradient = self.value_and_gradient(state)
        return float(gradient @ derivative)

    def lqr_torque(self, state: np.ndarray) -> float:
        """Local feedback associated with the Riccati value: ``tau = -K e``."""
        return float(-(self.k @ upright_error(state))[0])

    def clf_margin(self, state: np.ndarray, torque_limit: float) -> float:
        """Smallest achievable rate of the local piece under a torque bound.

        Returns ``min_{|tau| <= limit} d/dt (e.T P e) / s``.  A negative value
        says some admissible torque decreases the local piece there, which is
        what a reward built on it needs; it is weaker than asking the linear
        feedback ``-K e`` to do the decreasing, and it is the property that
        survives when no balancing controller is ever switched in.
        """
        if not np.isfinite(torque_limit) or torque_limit <= 0.0:
            raise ValueError("torque_limit must be finite and positive")
        error = upright_error(state)
        gradient = 2.0 * (self.p @ error) / self.scale
        drift, gain = plant_drift_and_gain(self.params, state)
        return float(gradient @ drift - torque_limit * abs(gradient @ gain))

    def _membership_and_derivative(self, residual: float) -> Tuple[float, float]:
        """Smootherstep in the residual, and ``d mu / d residual``."""
        inner = self.region.transition_fraction
        coordinate = (1.0 - residual) / (1.0 - inner)
        membership, slope = _smootherstep(coordinate)
        return membership, -slope / (1.0 - inner)


class XKLQRSwitchedController:
    """Xin-Kaneda swing-up, latching one-way to the local LQR feedback.

    Same construction as :class:`NonsmoothLyapunov`'s two pieces, run as a
    controller rather than a reward: the exact Xin-Kaneda law drives the
    swing-up, and on first entry to :class:`AttractiveRegion` -- Lai et
    al.'s equation-(17) region -- control latches to the local Riccati
    feedback ``tau = -K e`` and never switches back (``benchmarks/render_
    acrobot_nslf.py``'s ``SwitchedController`` renders exactly this law; this
    is the reusable form, with the plain ``act(obs) -> action`` interface
    :class:`~controllers.xin_kaneda.XinKanedaController` uses, so it drops
    into anything that already accepts that controller -- including CT-SAC's
    ``demonstration_policy``).
    """

    SWING_UP = 1
    BALANCE = 2

    def __init__(
        self,
        params: AcrobotParams,
        gains: Gains,
        region: Optional[AttractiveRegion] = None,
        *,
        torque_limit: Optional[float] = None,
    ) -> None:
        self.params = params
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
        self.lyapunov = NonsmoothLyapunov(params, gains, region or AttractiveRegion())
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
        if self.stage == self.SWING_UP and self.lyapunov.region.contains(
            self.params, state
        ):
            self.stage = self.BALANCE
            self.switch_step = self.steps
        self.steps += 1

        if self.stage == self.SWING_UP:
            action = self.swing_up(obs)
            self.last_torque = self.swing_up.last_torque
            self.last_commanded_torque = self.swing_up.last_commanded_torque
            return action

        commanded = self.lyapunov.lqr_torque(state)
        applied = float(np.clip(commanded, -self.torque_limit, self.torque_limit))
        if abs(commanded) > self.torque_limit:
            self.saturated_steps += 1
        self.last_commanded_torque = commanded
        self.last_torque = applied
        return np.array([applied / self.params.gear], dtype=np.float64)
