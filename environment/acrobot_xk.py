"""Acrobot plant with Xin & Kaneda's geometry and their analysis's assumptions.

`Xin & Kaneda 2007 <https://doi.org/10.1002/rnc.1184>`_ derive and simulate
their swing-up results for one specific two-link robot. The plant here
reproduces that robot exactly:

    m1 = m2 = 1 kg, l1 = 1 m, l2 = 2 m, lc1 = 0.5 m, lc2 = 1 m,
    I1 = 0.083 kg m^2, I2 = 0.33 kg m^2, g = 9.8 m/s^2

Those values give the grouped parameters that their equations use,
``a = (1.333, 1.330, 1.000)`` and ``b = (14.7, 9.8)``, and so ``E_r = 24.5``.
Explicit ``<inertial>`` elements set the link inertias. ``I1`` and ``I2``
therefore land on the published values, and not on the inertia of a capsule of
that length.

Two further properties of their setting are carried over:

* **No dissipation.** ``Edot = qdot2 tau2`` is the engine of the whole
  derivation. With joint damping ``d`` it becomes
  ``Edot = qdot2 tau2 - d (qdot1^2 + qdot2^2)``. That leaves
  ``Vdot = -k_V qdot2^2 - (E - E_r) d |qdot|^2``, whose second term is
  *positive* throughout a swing-up. On the target set ``q2 = qdot2 = 0`` the
  injected power is exactly zero, and the shoulder still dissipates. The
  homoclinic orbit is then no invariant set of the damped closed loop, at any
  gains. ``damping`` therefore defaults to 0. A positive value stays reachable,
  so a run can measure the obstruction.
* **Bounded actuation and runaway termination.** The law asks for a little
  under 20 N*m at the paper's gains, so ``torque_limit`` defaults to 20 N*m.
  The episode terminates on any of three events. The unwrapped elbow winds to
  ``4 pi``. The elbow rate reaches ``4 pi`` rad/s. The shoulder rate reaches
  twice its peak on the target homoclinic orbit. These bounds keep the
  analytical swing-up trajectories, and stop the energetic runaways that appear
  during learning.

The task selects one of the four reward rates in
``docs/reward_shaping_for_acrobot_swingup.md`` with ``reward_kind``:

    r0(x) = -[(Etil / E_s)^2 + (q2 / q_s)^2 + (qdot2 / omega_s)^2]
    r1(x) = -V(x) / V_down
    r2(x, u) = [-V(x) - eta Vdot(x, u)] / V_down
    r3(x, u) = [-V(x) - eta Vdot(x, u) + lambda eta V(x)] / V_down

The optional ``reward_transform="log_reciprocal"`` maps the selected reward
family to ``log(1 / -r) = -log(-r)``. A small floor on ``-r`` keeps the
transform finite at the target, and for shaped rewards that cross zero. The raw
values stay available in the reward diagnostics under ``raw_r0`` through
``raw_r3``.

Historical runs keep these definitions through the default
``reward_base="lyapunov"``. The ``reward_base="r0"`` experiment family keeps
the same r1--r3 shaping structure, and replaces the leading
``-V(x) / V_down`` term alone with the normalized baseline reward:

    r1_r0(x) = r0(x)
    r2_r0(x, u) = r0(x) - eta Vdot(x, u) / V_down
    r3_r0(x, u) = r0(x) - eta Vdot(x, u) / V_down
                  + lambda eta V(x) / V_down

The derivative, and for r3 the discount correction, therefore still use the
original Xin--Kaneda ``V``. ``reward_base`` leaves both of them alone, and
never rereads them as a derivative or potential of ``r0``.

``reward_base="lai_she"`` replaces ``V`` with
:class:`controllers.acrobot_gated_lyapunov.NonsmoothLyapunov`. That is the
nonsmooth Lyapunov function of Lai, Wu, She and Yang. The task builds it from
its own physics through ``AcrobotParams.from_physics`` and its ``k_v``,
``k_d`` and ``k_p`` gains, already normalized on its own scale. ``V`` and
``Vdot`` in the r1--r3 formulas above become the value and the directional
derivative of that function. The r0 baseline and ``reward_transform`` stay as
they are. This base requires ``lyapunov_rate_source="actual"``. The
``xk_closed_loop`` surrogate assumes the global Xin--Kaneda swing-up law, and
loses its meaning once the LQR-gated local piece of the Lai--She function is
active.

By default ``Vdot`` is the true directional derivative under the action that
the plant applies. The counterfactual ``lyapunov_rate_source="xk_closed_loop"``
substitutes the identity from the exact Xin--Kaneda feedback law,
``Vdot_XK = -k_V qdot2^2``. Under a learned policy that identity is an
action-independent surrogate for the derivative of ``V`` along the policy. It
can reward elbow speed, so it stays an explicit experiment arm and never
replaces the physical derivative.

Here ``V_down = V(x_down) = E_s^2 / 2`` is the Lyapunov value at hanging rest.
The common linear scale preserves the shape and units of every Lyapunov term.
The state, rate, and torque caps make their ranges finite, with no clipping.
``eta`` is an explicit, non-negative parameter that ``r2`` and ``r3`` require.
``r3`` also requires the physical discount rate ``lambda``. On a cap crossing,
the task emits the finite lower envelope of the selected reward as the terminal
reward. It then freezes that value over the part of the episode that never
runs. That lower
envelope is the minimum reward attainable anywhere in the capped state and
action closure. CT-SAC sums the remaining-horizon return analytically, so a cap
is never preferable to any admissible ordinary transition. These rewards are
training signals alone. The analytical controller ignores them, and every
comparison uses the seven reward-independent metrics recomputed from state,
physical time, and torque.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional

import numpy as np

from dm_control.rl import control
from dm_control import mujoco
from dm_control.suite import base as suite_base
from dm_control.suite import common


# Xin & Kaneda's published mechanical parameters (their Section 7).
LINK1_LENGTH = 1.0
LINK2_LENGTH = 2.0
LINK1_COM = 0.5
LINK2_COM = 1.0
LINK1_MASS = 1.0
LINK2_MASS = 1.0
LINK1_INERTIA = 0.083
LINK2_INERTIA = 0.33
GRAVITY = 9.8

# Full reach of the chain, used to place the target site, the light and the floor.
REACH = LINK1_LENGTH + LINK2_LENGTH

# The paper-gain controller peaks at about 19.71 N*m on the release protocol.
DEFAULT_TORQUE_LIMIT = 20.0

# State bounds that stop physically unhelpful high-energy trajectories.
# The task multiplies the shoulder-rate threshold by the plant-derived omega_s
# after energy calibration. The elbow angle test reads the angle unwrapped, on
# purpose.
ELBOW_ANGLE_LIMIT = 4.0 * np.pi
ELBOW_RATE_LIMIT = 4.0 * np.pi
SHOULDER_RATE_SCALE_LIMIT = 2.0

TERMINATION_ELBOW_ANGLE = "elbow_angle_limit"
TERMINATION_ELBOW_RATE = "elbow_rate_limit"
TERMINATION_SHOULDER_RATE = "shoulder_rate_limit"
LOWER_BOUND_TERMINATION_REWARD_SOURCE = "reward_lower_bound"

# Reset for the swing-up arms. It starts near hanging, at the paper's own
# initial condition, a small angle off the downward equilibrium.
DEFAULT_ANGLE_NOISE = 0.05
DEFAULT_VELOCITY_NOISE = 0.01

# Xin & Kaneda's Section-7 gains for the Lyapunov rewards. k_D and k_P define
# V. Only the optional closed-loop Vdot surrogate reads k_V. Keep all three
# open to the caller, so that a reward study can vary either construction and
# leave the plant alone.
DEFAULT_LYAPUNOV_K_D = 35.8
DEFAULT_LYAPUNOV_K_P = 61.2
DEFAULT_LYAPUNOV_K_V = 66.3
LYAPUNOV_RATE_SOURCES = frozenset(("actual", "xk_closed_loop"))
REWARD_BASES = frozenset(("lyapunov", "r0", "lai_she"))
DEFAULT_REWARD_BASE = "lyapunov"
REWARD_KINDS = frozenset(("r0", "r1", "r2", "r3"))
REWARD_TRANSFORMS = frozenset(("identity", "log_reciprocal"))
DEFAULT_REWARD_TRANSFORM = "identity"
LOG_RECIPROCAL_REWARD_FLOOR = 1e-6

# Xin--Kaneda's grouped constants, derived from the physical constants above.
_A1 = LINK1_INERTIA + LINK1_MASS * LINK1_COM**2 + LINK2_MASS * LINK1_LENGTH**2
_A2 = LINK2_INERTIA + LINK2_MASS * LINK2_COM**2
_A3 = LINK2_MASS * LINK1_LENGTH * LINK2_COM
_B1 = (LINK1_MASS * LINK1_COM + LINK2_MASS * LINK1_LENGTH) * GRAVITY
_B2 = LINK2_MASS * LINK2_COM * GRAVITY
_ENERGY_SPAN = 2.0 * (_B1 + _B2)
_EXTENDED_M11 = _A1 + _A2 + 2.0 * _A3
_EXTENDED_M12 = _A2 + _A3
_OMEGA_S = np.sqrt(2.0 * _ENERGY_SPAN / _EXTENDED_M11)
_V_DOWN = 0.5 * _ENERGY_SPAN**2


@dataclass(frozen=True)
class PlantScales:
    """Link mass/length multipliers; COM scales with length, inertia with m*l².

    Gravity, damping and actuator capacity are independent of these factors.
    Unit factors reproduce the published plant exactly.
    """

    mass1: float = 1.0
    mass2: float = 1.0
    length1: float = 1.0
    length2: float = 1.0

    def __post_init__(self):
        for name in ("mass1", "mass2", "length1", "length2"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")
            object.__setattr__(self, name, value)

    @classmethod
    def coerce(cls, value):
        return value if isinstance(value, cls) else cls(**(value or {}))

    @property
    def grouped(self):
        m1, m2 = LINK1_MASS * self.mass1, LINK2_MASS * self.mass2
        l1 = LINK1_LENGTH * self.length1
        c1, c2 = LINK1_COM * self.length1, LINK2_COM * self.length2
        return (
            m1 * c1**2 + m2 * l1**2 + LINK1_INERTIA * self.mass1 * self.length1**2,
            m2 * c2**2 + LINK2_INERTIA * self.mass2 * self.length2**2,
            m2 * l1 * c2,
            (m1 * c1 + m2 * l1) * GRAVITY,
            m2 * c2 * GRAVITY,
        )


@lru_cache(maxsize=64)
def _elbow_acceleration_abs_bound(
    torque_limit: float,
    damping: float,
    shoulder_rate_limit: float,
    elbow_rate_limit: float,
    plant_scales: PlantScales = PlantScales(),
) -> float:
    """Bound ``|qddot2|`` over the capped state/action closure.

    At fixed ``q2`` an analytical step maximizes each of the velocity, damping,
    torque and gravity contributions to the second row of ``M^-1``. A smooth
    one-dimensional periodic envelope remains. The function sweeps a dense
    deterministic grid, then refines every sampled local maximum within bounds.
    That path avoids a costly five-dimensional optimizer and keeps a small
    outward margin.
    """
    from scipy.optimize import minimize_scalar

    a1, a2, a3, b1, b2 = plant_scales.grouped

    shoulder_rate = float(shoulder_rate_limit)
    elbow_rate = float(elbow_rate_limit)

    def envelope(q2):
        q2 = np.asarray(q2, dtype=np.float64)
        cosine = np.cos(q2)
        sine_abs = np.abs(np.sin(q2))
        m11 = a1 + a2 + 2.0 * a3 * cosine
        m12 = a2 + a3 * cosine
        determinant = a1 * a2 - (a3 * cosine) ** 2

        coriolis = a3 * sine_abs * (
            np.abs(m12)
            * (2.0 * shoulder_rate * elbow_rate + elbow_rate**2)
            + m11 * shoulder_rate**2
        )
        damping_force = damping * (
            np.abs(m12) * shoulder_rate + m11 * elbow_rate
        )
        first_gravity = m12 * b1
        second_gravity = (m12 - m11) * b2
        gravity = np.sqrt(
            np.maximum(
                0.0,
                first_gravity**2
                + second_gravity**2
                + 2.0
                * first_gravity
                * second_gravity
                * cosine,
            )
        )
        return (
            coriolis + damping_force + gravity + m11 * torque_limit
        ) / determinant

    grid = np.linspace(0.0, 2.0 * np.pi, 4097)
    values = envelope(grid)
    candidate_indices = np.flatnonzero(
        (values[1:-1] >= values[:-2])
        & (values[1:-1] >= values[2:])
    ) + 1
    candidates = [float(values[0]), float(values[-1])]
    for index in candidate_indices:
        result = minimize_scalar(
            lambda angle: -float(envelope(angle)),
            bounds=(float(grid[index - 1]), float(grid[index + 1])),
            method="bounded",
            options={"xatol": 1e-14},
        )
        candidates.append(-float(result.fun))

    # Round outward so floating-point/refinement error cannot make the failure
    # rate slightly more attractive than an admissible ordinary reward.
    return float(np.nextafter(max(candidates) * (1.0 + 1e-10), np.inf))


def _normalize_reward_transform(reward_transform: str) -> str:
    transform = str(reward_transform).strip().lower()
    if transform not in REWARD_TRANSFORMS:
        choices = ", ".join(sorted(REWARD_TRANSFORMS))
        raise ValueError(
            f"reward_transform must be one of {{{choices}}}, got "
            f"{reward_transform!r}"
        )
    return transform


def _transform_reward(reward: float, reward_transform: str) -> float:
    transform = _normalize_reward_transform(reward_transform)
    reward = float(reward)
    if transform == "identity":
        return reward
    reciprocal_argument = max(-reward, LOG_RECIPROCAL_REWARD_FLOOR)
    return float(-np.log(reciprocal_argument))


@lru_cache(maxsize=32)
def _lai_she_delta_over_scale(k_v: float, k_d: float, k_p: float) -> float:
    """``Delta / scale`` for a Lai-She function on this module's paper parameters.

    The raw-quadratic reward-rate lower bound below uses this value as a
    correction under ``reward_base='lai_she'``.

    The function needs no physics argument. This plant already *is* the paper's
    fixed geometry, and ``AcrobotParams.from_physics`` on it recovers exactly
    ``_A1`` through ``_B2``. ``gear`` plays no part in the value of
    ``NonsmoothLyapunov``. It enters the control output alone, which this
    function never asks for, so any placeholder works.

    The correction stays a valid and conservative bound everywhere, inside the
    LQR-gated region and outside it. The Definition-3 property of
    ``NonsmoothLyapunov`` builds ``Delta`` so that the local piece never exceeds
    the outer swing-up piece anywhere in the region. The test
    ``test_delta_dominates_the_local_value_everywhere_on_the_region`` covers
    that property. The swing-up piece alone, ``(raw_XK_V + Delta) / scale``,
    therefore bounds the switched value everywhere. ``raw_XK_V`` is exactly the
    quantity that the raw-quadratic bound below already bounds, through the same
    energy and unwrapped-elbow formula, so compare
    ``_xk_raw_value_and_gradient``. This one constant added to that existing
    bound is therefore a valid upper bound on the value of
    ``NonsmoothLyapunov`` everywhere in the caps closure.
    """
    from controllers.acrobot_gated_lyapunov import AttractiveRegion, NonsmoothLyapunov
    from controllers.xin_kaneda import AcrobotParams, Gains

    params = AcrobotParams(a1=_A1, a2=_A2, a3=_A3, b1=_B1, b2=_B2, gear=1.0, gravity=GRAVITY)
    gains = Gains(k_v=k_v, k_d=k_d, k_p=k_p)
    lai_she = NonsmoothLyapunov(params, gains, AttractiveRegion())
    return float(lai_she.delta / lai_she.scale)


@lru_cache(maxsize=128)
def _raw_reward_rate_lower_bound(
    reward_kind: str,
    *,
    plant_scales: PlantScales = PlantScales(),
    reward_base: str = DEFAULT_REWARD_BASE,
    k_d: float = DEFAULT_LYAPUNOV_K_D,
    k_p: float = DEFAULT_LYAPUNOV_K_P,
    k_v: float = DEFAULT_LYAPUNOV_K_V,
    eta: Optional[float] = None,
    discount_rate: Optional[float] = None,
    lyapunov_rate_source: str = "actual",
    torque_limit: float = DEFAULT_TORQUE_LIMIT,
    damping: float = 0.0,
    elbow_angle_limit: float = ELBOW_ANGLE_LIMIT,
    elbow_rate_limit: float = ELBOW_RATE_LIMIT,
    shoulder_rate_scale_limit: float = SHOULDER_RATE_SCALE_LIMIT,
) -> float:
    """Return the conservative reward-rate lower envelope used at termination.

    The calculation uses the closure of the nonterminal state/action limits.
    It is deliberately an envelope rather than a sampled endpoint minimum: a
    single state need not attain all individually worst terms simultaneously.
    Every reward kind's cap-crossing terminal reward is this envelope, so a
    cap can never be preferable to any admissible ordinary transition.
    """
    kind = str(reward_kind).strip().lower()
    if kind not in REWARD_KINDS:
        raise ValueError(f"unknown Acrobot-XK reward kind {reward_kind!r}")
    k_d = float(k_d)
    k_p = float(k_p)
    k_v = float(k_v)
    torque_limit = float(torque_limit)
    damping = float(damping)
    source = str(lyapunov_rate_source).strip().lower()
    base = str(reward_base).strip().lower()
    elbow_angle_limit = float(elbow_angle_limit)
    elbow_rate_limit = float(elbow_rate_limit)
    shoulder_rate_scale_limit = float(shoulder_rate_scale_limit)
    if not np.isfinite(k_d) or k_d <= 0.0:
        raise ValueError("k_d must be finite and > 0")
    if not np.isfinite(k_p) or k_p <= 0.0:
        raise ValueError("k_p must be finite and > 0")
    if not np.isfinite(k_v) or k_v <= 0.0:
        raise ValueError("k_v must be finite and > 0")
    if source not in LYAPUNOV_RATE_SOURCES:
        choices = ", ".join(sorted(LYAPUNOV_RATE_SOURCES))
        raise ValueError(
            f"lyapunov_rate_source must be one of {{{choices}}}, got "
            f"{lyapunov_rate_source!r}"
        )
    if base not in REWARD_BASES:
        choices = ", ".join(sorted(REWARD_BASES))
        raise ValueError(
            f"reward_base must be one of {{{choices}}}, got {reward_base!r}"
        )
    if kind == "r0" and base != DEFAULT_REWARD_BASE:
        raise ValueError(
            "reward_base is only meaningful for reward_kind='r1'--'r3'"
        )
    if kind == "r0" and source != "actual":
        raise ValueError(
            "lyapunov_rate_source is not meaningful for reward_kind='r0'"
        )
    if base == "lai_she" and kind in ("r2", "r3"):
        raise NotImplementedError(
            "reward_rate_lower_bound does not derive a conservative Vdot "
            "envelope for reward_base='lai_she' (the gate-derivative term "
            "in NonsmoothLyapunov.rate has no closed-form bound here yet); "
            "only reward_kind='r1' is supported for this reward_base"
        )
    if not np.isfinite(torque_limit) or torque_limit <= 0.0:
        raise ValueError("torque_limit must be finite and > 0")
    if not np.isfinite(damping) or damping < 0.0:
        raise ValueError("damping must be finite and >= 0")
    for name, value in (
        ("elbow_angle_limit", elbow_angle_limit),
        ("elbow_rate_limit", elbow_rate_limit),
        ("shoulder_rate_scale_limit", shoulder_rate_scale_limit),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and > 0")

    plant_scales = PlantScales.coerce(plant_scales)
    if base == "lai_she" and plant_scales != PlantScales():
        raise ValueError("lai_she reward bounds currently require the nominal plant")
    a1, a2, a3, b1, b2 = plant_scales.grouped
    energy_span = 2.0 * (b1 + b2)
    extended_m11 = a1 + a2 + 2.0 * a3
    extended_m12 = a2 + a3
    omega_s = np.sqrt(2.0 * energy_span / extended_m11)
    v_down = 0.5 * energy_span**2
    shoulder_rate = shoulder_rate_scale_limit * omega_s
    elbow_rate = elbow_rate_limit
    elbow_angle = elbow_angle_limit
    kinetic_max = 0.5 * (
        extended_m11 * shoulder_rate**2
        + 2.0 * extended_m12 * shoulder_rate * elbow_rate
        + a2 * elbow_rate**2
    )
    energy_error_abs = max(energy_span, kinetic_max)

    wrapped_elbow_abs = min(elbow_angle, np.pi)
    baseline_magnitude = (
        (energy_error_abs / energy_span) ** 2
        + (wrapped_elbow_abs / np.pi) ** 2
        + (elbow_rate / omega_s) ** 2
    )
    if kind == "r0" or (base == "r0" and kind == "r1"):
        return -float(baseline_magnitude)

    state_part_max = (
        0.5 * energy_error_abs**2 + 0.5 * k_p * elbow_angle**2
    )
    lyapunov_max = (
        state_part_max + 0.5 * k_d * elbow_rate**2
    ) / v_down
    if kind == "r1":
        if base == "lai_she":
            lyapunov_max += _lai_she_delta_over_scale(k_v, k_d, k_p)
        return -float(lyapunov_max)

    eta_value = float(eta) if eta is not None else float("nan")
    if not np.isfinite(eta_value) or eta_value < 0.0:
        raise ValueError(f"reward_kind={kind!r} requires finite eta >= 0")
    lambda_value = None
    if kind == "r3":
        lambda_value = (
            float(discount_rate)
            if discount_rate is not None
            else float("nan")
        )
        if not np.isfinite(lambda_value) or lambda_value < 0.0:
            raise ValueError(
                "reward_kind='r3' requires finite discount_rate >= 0"
            )
    if base == "r0" and source == "xk_closed_loop":
        # Both shaping additions are non-negative on the capped domain:
        # -eta Vdot_XK = eta k_V qdot2^2 and lambda eta V >= 0.
        return -float(baseline_magnitude)
    if source == "xk_closed_loop":
        coefficient = 1.0
        if kind == "r3":
            coefficient = 1.0 - lambda_value * eta_value
            if coefficient <= 0.0:
                raise ValueError(
                    "xk_closed_loop r3 requires discount_rate * eta < 1 so "
                    "the energy and elbow-angle terms retain negative reward "
                    "coefficients"
                )
        rate_coefficient = max(
            0.5 * coefficient * k_d - eta_value * k_v, 0.0
        )
        magnitude = (
            coefficient * state_part_max
            + rate_coefficient * elbow_rate**2
        ) / v_down
        return -float(magnitude)

    acceleration = _elbow_acceleration_abs_bound(
        torque_limit,
        damping,
        shoulder_rate,
        elbow_rate,
        plant_scales,
    )
    energy_rate_abs = (
        elbow_rate * torque_limit
        + damping * (shoulder_rate**2 + elbow_rate**2)
    )
    lyapunov_rate_abs = (
        energy_error_abs * energy_rate_abs
        + k_d * elbow_rate * acceleration
        + k_p * elbow_angle * elbow_rate
    ) / v_down

    if base == "r0":
        # The r3 correction lambda eta V is non-negative, so dropping it gives
        # a conservative lower envelope shared with the r2 construction.
        return -float(baseline_magnitude + eta_value * lyapunov_rate_abs)

    coefficient = 1.0
    if kind == "r3":
        coefficient = max(1.0 - lambda_value * eta_value, 0.0)
    return -float(coefficient * lyapunov_max + eta_value * lyapunov_rate_abs)


@lru_cache(maxsize=256)
def reward_rate_lower_bound(
    reward_kind: str,
    *,
    plant_scales: PlantScales = PlantScales(),
    reward_base: str = DEFAULT_REWARD_BASE,
    reward_transform: str = DEFAULT_REWARD_TRANSFORM,
    k_d: float = DEFAULT_LYAPUNOV_K_D,
    k_p: float = DEFAULT_LYAPUNOV_K_P,
    k_v: float = DEFAULT_LYAPUNOV_K_V,
    eta: Optional[float] = None,
    discount_rate: Optional[float] = None,
    lyapunov_rate_source: str = "actual",
    torque_limit: float = DEFAULT_TORQUE_LIMIT,
    damping: float = 0.0,
    elbow_angle_limit: float = ELBOW_ANGLE_LIMIT,
    elbow_rate_limit: float = ELBOW_RATE_LIMIT,
    shoulder_rate_scale_limit: float = SHOULDER_RATE_SCALE_LIMIT,
) -> float:
    """Return the selected reward's transformed terminal lower envelope."""
    raw_bound = _raw_reward_rate_lower_bound(
        reward_kind,
        plant_scales=plant_scales,
        reward_base=reward_base,
        k_d=k_d,
        k_p=k_p,
        k_v=k_v,
        eta=eta,
        discount_rate=discount_rate,
        lyapunov_rate_source=lyapunov_rate_source,
        torque_limit=torque_limit,
        damping=damping,
        elbow_angle_limit=elbow_angle_limit,
        elbow_rate_limit=elbow_rate_limit,
        shoulder_rate_scale_limit=shoulder_rate_scale_limit,
    )
    return _transform_reward(raw_bound, reward_transform)


# The reward-independent homoclinic tube from the experiment protocol.
HOMOCLINIC_ENERGY_TOLERANCE = 0.05
HOMOCLINIC_ANGLE_TOLERANCE = 0.025
HOMOCLINIC_RATE_TOLERANCE = 0.05

# The "release" reset. The chain is straight, starts at rest, and the shoulder
# is displaced from hanging. The paper's own initial condition belongs to this
# family, at a displacement of 0.1708 rad. The reward experiments share it as
# their evaluation distribution. See
# ``docs/reward_shaping_for_acrobot_swingup.md``. The displacement stays away
# from zero, because hanging is an equilibrium of the closed loop and a zero
# displacement never starts.
RELEASE_ANGLE_RANGE = (0.05, 0.5)

# The upright and hanging equilibria of eq. 10, and the paper's own initial
# condition, all directly in qpos because the model is built in its coordinates.
UPRIGHT_SHOULDER = 0.5 * np.pi
HANGING_SHOULDER = -0.5 * np.pi
PAPER_INITIAL_SHOULDER = -1.4

# The planar problem constrains the inertia about the hinge axis alone. The
# out-of-plane principal moment only has to keep the tensor admissible.
_MINOR_INERTIA = 0.001

# The layout of the model makes ``qpos`` *be* the paper's ``(q1, q2)``. That
# removes the coordinate transform:
#
#   * the links rest along +x, so ``q1 = 0`` is link 1 horizontal, as in eq. 10
#   * the hinges turn about -y, so a larger ``q1`` lifts the link toward +z,
#     upright is ``q1 = +pi/2``, and hanging is ``-pi/2``
#   * the shoulder sits at the world origin, so the gravitational potential of
#     MuJoCo equals ``P(q) = b1 sin q1 + b2 sin(q1 + q2)`` of eq. 7, with no
#     offset
#
# Four consequences follow, and the tests cover all four. ``mj_fullM`` equals
# ``M(q)``. ``qfrc_bias`` equals ``+(H + G)`` with no sign flip. The MuJoCo
# mechanical energy equals ``E`` outright. ``gear * ctrl`` is ``tau2`` directly.
# ``diaginertia`` runs in the order ``(about the rod, about y, about z)``, so
# the published moment lands on the hinge axis.
_MODEL_XML = """
<mujoco model="acrobot-xk">
  <include file="./common/visual.xml"/>
  <include file="./common/skybox.xml"/>
  <include file="./common/materials.xml"/>

  <option timestep="0.01" integrator="RK4" gravity="0 0 -{gravity}">
    <flag constraint="disable" energy="enable"/>
  </option>

  <!-- Large enough offscreen buffer for the swing-up render. -->
  <visual>
    <global offwidth="1280" offheight="960"/>
  </visual>

  <default>
    <joint type="hinge" axis="0 -1 0" damping="{damping}" limited="false"/>
    <geom type="capsule" mass="0"/>
  </default>

  <worldbody>
    <light name="light" pos="0 0 {light_height}"/>
    <geom name="floor" pos="0 0 {floor_height}" size="6 6 .2" type="plane"
          material="grid"/>
    <site name="target" type="sphere" pos="0 0 {reach}" size="0.2"
          material="target" group="3"/>
    <camera name="fixed" pos="0 -8 0" zaxis="0 -1 0"/>
    <body name="upper_arm" pos="0 0 0">
      <joint name="shoulder"/>
      <inertial pos="{lc1} 0 0" mass="{m1}" diaginertia="{minor1} {i1} {i1}"/>
      <geom name="upper_arm" fromto="0 0 0 {l1} 0 0" size="0.05" material="self"/>
      <body name="lower_arm" pos="{l1} 0 0">
        <joint name="elbow"/>
        <inertial pos="{lc2} 0 0" mass="{m2}" diaginertia="{minor2} {i2} {i2}"/>
        <geom name="lower_arm" fromto="0 0 0 {l2} 0 0" size="0.049"
              material="self"/>
        <site name="tip" pos="{l2} 0 0" size="0.01"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="elbow" joint="elbow" gear="{gear}" ctrllimited="true"
           ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


def _model_xml(damping: float, torque_limit: float,
               plant_scales: PlantScales = PlantScales()) -> bytes:
    """Xin-Kaneda's geometry as a MuJoCo model, at the given damping and gear."""
    if not np.isfinite(damping) or damping < 0.0:
        raise ValueError(f"damping must be finite and >= 0, got {damping}")
    if not np.isfinite(torque_limit) or torque_limit <= 0.0:
        raise ValueError(
            f"torque_limit must be finite and > 0, got {torque_limit}"
        )
    scales = PlantScales.coerce(plant_scales)
    reach = LINK1_LENGTH * scales.length1 + LINK2_LENGTH * scales.length2
    return _MODEL_XML.format(
        gravity=GRAVITY,
        damping=float(damping),
        gear=float(torque_limit),
        reach=reach,
        light_height=reach + 2.0,
        floor_height=-(reach + 3.0),
        l1=LINK1_LENGTH * scales.length1,
        l2=LINK2_LENGTH * scales.length2,
        lc1=LINK1_COM * scales.length1,
        lc2=LINK2_COM * scales.length2,
        m1=LINK1_MASS * scales.mass1,
        m2=LINK2_MASS * scales.mass2,
        i1=LINK1_INERTIA * scales.mass1 * scales.length1**2,
        i2=LINK2_INERTIA * scales.mass2 * scales.length2**2,
        minor1=_MINOR_INERTIA * scales.mass1 * scales.length1**2,
        minor2=_MINOR_INERTIA * scales.mass2 * scales.length2**2,
    ).encode("utf-8")


class BalanceXK(suite_base.Task):
    """Swing-up task with selectable ``r0`` through ``r3`` reward rates.

    Deliberately standalone: it shares no code with the ``v2 ... v6.1`` reward
    line, so its numbers cannot drift when those tasks are edited.
    """

    def __init__(
        self,
        *,
        random=None,
        angle_noise: float = DEFAULT_ANGLE_NOISE,
        velocity_noise: float = DEFAULT_VELOCITY_NOISE,
        uniform_start: bool = False,
        paper_start: bool = False,
        release_start: bool = False,
        release_angle_range: tuple = RELEASE_ANGLE_RANGE,
        roa_start_fraction: float = 0.0,
        reward_kind: str = "r0",
        reward_base: str = DEFAULT_REWARD_BASE,
        reward_transform: str = DEFAULT_REWARD_TRANSFORM,
        k_d: float = DEFAULT_LYAPUNOV_K_D,
        k_p: float = DEFAULT_LYAPUNOV_K_P,
        k_v: float = DEFAULT_LYAPUNOV_K_V,
        eta: Optional[float] = None,
        discount_rate: Optional[float] = None,
        lyapunov_rate_source: str = "actual",
        torque_limit: float = DEFAULT_TORQUE_LIMIT,
        damping: float = 0.0,
        elbow_angle_limit: float = ELBOW_ANGLE_LIMIT,
        elbow_rate_limit: float = ELBOW_RATE_LIMIT,
        shoulder_rate_scale_limit: float = SHOULDER_RATE_SCALE_LIMIT,
        cap_terminal_penalty: bool = True,
        plant_scales: PlantScales = PlantScales(),
    ) -> None:
        super().__init__(random=random)
        # True is the default. A state-cap crossing then freezes the reward at
        # the conservative reward-rate lower envelope, and CT-SAC sums that
        # frozen rate analytically over the part of the episode that never runs
        # (see DMCContinuousEnv._step_physics_unlocked). With False, a cap ends
        # the episode at the live reward computed for the terminal state. That
        # is an ordinary termination, with no absorbing continuation and no
        # bootstrap.
        self.cap_terminal_penalty = bool(cap_terminal_penalty)
        self.angle_noise = float(angle_noise)
        self.velocity_noise = float(velocity_noise)
        if not np.isfinite(self.angle_noise) or self.angle_noise < 0.0:
            raise ValueError("angle_noise must be finite and >= 0")
        if not np.isfinite(self.velocity_noise) or self.velocity_noise < 0.0:
            raise ValueError("velocity_noise must be finite and >= 0")
        self.uniform_start = bool(uniform_start)
        self.paper_start = bool(paper_start)
        self.release_start = bool(release_start)
        chosen = sum(
            (self.uniform_start, self.paper_start, self.release_start)
        )
        if chosen > 1:
            raise ValueError(
                "uniform_start, paper_start and release_start are mutually "
                "exclusive resets"
            )
        low, high = (float(v) for v in release_angle_range)
        if not (0.0 < low < high):
            raise ValueError(
                f"release_angle_range must satisfy 0 < low < high, got {low}, {high}"
            )
        self.release_angle_range = (low, high)
        self.roa_start_fraction = float(roa_start_fraction)
        if not (0.0 <= self.roa_start_fraction <= 1.0):
            raise ValueError(
                "roa_start_fraction must be in [0, 1], got "
                f"{self.roa_start_fraction}"
            )
        if self.roa_start_fraction > 0.0 and not self.release_start:
            raise ValueError(
                "roa_start_fraction > 0 only replaces some release_start "
                "draws; pass release_start=True too"
            )
        self._sos_certificate = None
        self.reward_kind = str(reward_kind).strip().lower()
        if self.reward_kind not in REWARD_KINDS:
            choices = ", ".join(sorted(REWARD_KINDS))
            raise ValueError(
                f"reward_kind must be one of {{{choices}}}, got {reward_kind!r}"
            )
        self.reward_base = str(reward_base).strip().lower()
        if self.reward_base not in REWARD_BASES:
            choices = ", ".join(sorted(REWARD_BASES))
            raise ValueError(
                f"reward_base must be one of {{{choices}}}, got {reward_base!r}"
            )
        if self.reward_kind == "r0" and self.reward_base != DEFAULT_REWARD_BASE:
            raise ValueError(
                "reward_base is only meaningful for reward_kind='r1'--'r3'"
            )
        self.reward_transform = _normalize_reward_transform(reward_transform)
        self.k_d = float(k_d)
        self.k_p = float(k_p)
        self.k_v = float(k_v)
        for name, value in (
            ("k_d", self.k_d),
            ("k_p", self.k_p),
            ("k_v", self.k_v),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {value}")
        self.lyapunov_rate_source = str(lyapunov_rate_source).strip().lower()
        if self.lyapunov_rate_source not in LYAPUNOV_RATE_SOURCES:
            choices = ", ".join(sorted(LYAPUNOV_RATE_SOURCES))
            raise ValueError(
                f"lyapunov_rate_source must be one of {{{choices}}}, got "
                f"{lyapunov_rate_source!r}"
            )
        if self.reward_kind == "r0" and self.lyapunov_rate_source != "actual":
            raise ValueError(
                "lyapunov_rate_source is not meaningful for reward_kind='r0'"
            )
        if self.reward_base == "lai_she" and self.lyapunov_rate_source != "actual":
            raise ValueError(
                "lyapunov_rate_source must be 'actual' when "
                "reward_base='lai_she': the 'xk_closed_loop' surrogate "
                "assumes the global Xin-Kaneda swing-up law and is not "
                "meaningful once the Lai-She function's LQR-gated local "
                "piece is active"
            )
        self.elbow_angle_limit = float(elbow_angle_limit)
        self.elbow_rate_limit = float(elbow_rate_limit)
        self.shoulder_rate_scale_limit = float(shoulder_rate_scale_limit)
        for name, value in (
            ("elbow_angle_limit", self.elbow_angle_limit),
            ("elbow_rate_limit", self.elbow_rate_limit),
            ("shoulder_rate_scale_limit", self.shoulder_rate_scale_limit),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {value}")
        if eta is None:
            self.eta = None
        else:
            self.eta = float(eta)
            if not np.isfinite(self.eta) or self.eta < 0.0:
                raise ValueError(f"eta must be finite and >= 0, got {eta}")
        derivative_rewards = {"r2", "r3"}
        if self.reward_kind in derivative_rewards and self.eta is None:
            raise ValueError(
                f"reward_kind={self.reward_kind!r} requires an explicit eta"
            )
        if self.reward_kind not in derivative_rewards and self.eta is not None:
            raise ValueError("eta is only valid for reward_kind='r2' or 'r3'")
        if discount_rate is None:
            self.discount_rate = None
        else:
            self.discount_rate = float(discount_rate)
            if not np.isfinite(self.discount_rate) or self.discount_rate < 0.0:
                raise ValueError(
                    "discount_rate must be finite and >= 0 s^-1, got "
                    f"{discount_rate}"
                )
        if self.reward_kind == "r3" and self.discount_rate is None:
            raise ValueError("reward_kind='r3' requires an explicit discount_rate")
        if self.reward_kind != "r3" and self.discount_rate is not None:
            raise ValueError("discount_rate is only valid when reward_kind='r3'")
        if (
            self.reward_kind == "r3"
            and self.reward_base == "lyapunov"
            and self.lyapunov_rate_source == "xk_closed_loop"
            and self.discount_rate * self.eta >= 1.0
        ):
            raise ValueError(
                "xk_closed_loop r3 requires discount_rate * eta < 1 so the "
                "energy and elbow-angle terms retain negative reward "
                "coefficients"
            )
        self.plant_scales = PlantScales.coerce(plant_scales)
        self.failure_reward_rate = reward_rate_lower_bound(
            self.reward_kind,
            plant_scales=self.plant_scales,
            reward_base=self.reward_base,
            reward_transform=self.reward_transform,
            k_d=self.k_d,
            k_p=self.k_p,
            k_v=self.k_v,
            eta=self.eta,
            discount_rate=self.discount_rate,
            lyapunov_rate_source=self.lyapunov_rate_source,
            torque_limit=torque_limit,
            damping=damping,
            elbow_angle_limit=self.elbow_angle_limit,
            elbow_rate_limit=self.elbow_rate_limit,
            shoulder_rate_scale_limit=self.shoulder_rate_scale_limit,
        )
        self.failure_reward_rate_source = LOWER_BOUND_TERMINATION_REWARD_SOURCE
        self._energy_hang: Optional[float] = None
        self._energy_span: Optional[float] = None
        self._rate_scale: Optional[float] = None
        self._last_reward_terms: Optional[Dict[str, float]] = None
        self._last_termination_reason: Optional[str] = None
        self._lai_she = None  # lazy: needs physics, built on first reward call

    # --- Mechanical energy ------------------------------------------------

    @staticmethod
    def _mass_matrix(physics) -> np.ndarray:
        import mujoco

        nv = int(physics.model.nv)
        mass_matrix = np.zeros((nv, nv), dtype=np.float64)
        mujoco.mj_fullM(physics.model.ptr, mass_matrix, physics.data.qM)
        return mass_matrix

    @classmethod
    def _mechanical_energy(cls, physics) -> float:
        """Kinetic plus gravitational potential, read from the model."""
        qvel = np.asarray(physics.data.qvel, dtype=np.float64)
        kinetic = 0.5 * float(qvel @ cls._mass_matrix(physics) @ qvel)
        potential = -float(
            np.asarray(physics.model.body_mass)
            @ (np.asarray(physics.data.xipos) @ np.asarray(physics.model.opt.gravity))
        )
        return kinetic + potential

    def _calibrate_energy(self, physics) -> None:
        """Measure hanging-rest and upright-rest energies. Clobbers the state."""
        physics.data.qvel[:] = 0.0
        physics.named.data.qpos[["shoulder", "elbow"]] = [UPRIGHT_SHOULDER, 0.0]
        physics.forward()
        energy_up = self._mechanical_energy(physics)
        # M11 with the elbow straight, where the homoclinic orbit lives. Read
        # here, the velocity scale of the reward stays a constant, and does not
        # depend on the pose.
        extended_m11 = float(self._mass_matrix(physics)[0, 0])
        physics.named.data.qpos[["shoulder", "elbow"]] = [HANGING_SHOULDER, 0.0]
        physics.forward()
        energy_hang = self._mechanical_energy(physics)
        span = energy_up - energy_hang
        if not np.isfinite(span) or span <= 0.0:
            raise RuntimeError(
                "Acrobot energy calibration failed: upright-rest energy must "
                f"exceed hanging-rest energy, got span {span}"
            )
        self._energy_hang = energy_hang
        self._energy_span = span
        # sqrt(2 E_s / M11(0)) is exactly the peak shoulder speed on the
        # homoclinic orbit, so r0 and the evaluation metrics share one scale.
        self._rate_scale = float(np.sqrt(2.0 * span / extended_m11))

    @property
    def energy_span(self) -> float:
        if self._energy_span is None:
            raise RuntimeError("energy references are not calibrated yet")
        return self._energy_span

    @property
    def energy_top(self) -> float:
        if self._energy_hang is None or self._energy_span is None:
            raise RuntimeError("energy references are not calibrated yet")
        return self._energy_hang + self._energy_span

    @property
    def lyapunov_scale(self) -> float:
        """Return ``V_down = V(hanging rest) = E_s^2 / 2``.

        At hanging rest the elbow-angle and elbow-rate terms vanish, while the
        energy error is ``-E_s``.  This scale therefore does not depend on the
        configurable Lyapunov gains and puts both ``r0`` and ``r1`` at ``-1``
        for the exact hanging-rest state.
        """
        return 0.5 * self.energy_span**2

    # --- Episode ----------------------------------------------------------

    def reseed(self, seed) -> None:
        """Make evaluation starts repeatable without rebuilding the task."""
        self._random = np.random.RandomState(seed)

    def initialize_episode(self, physics) -> None:
        self._calibrate_energy(physics)
        self._last_reward_terms = None
        self._last_termination_reason = None
        if self.uniform_start:
            physics.named.data.qpos[["shoulder", "elbow"]] = self.random.uniform(
                -np.pi, np.pi, 2
            )
            physics.named.data.qvel[["shoulder", "elbow"]] = 0.0
        elif self.release_start and (
            self.roa_start_fraction <= 0.0
            or self.random.uniform() >= self.roa_start_fraction
        ):
            # Straight chain, released from rest, shoulder displaced from
            # hanging by a magnitude drawn uniformly and a random sign.
            low, high = self.release_angle_range
            displacement = self.random.uniform(low, high)
            if self.random.uniform() < 0.5:
                displacement = -displacement
            physics.named.data.qpos[["shoulder", "elbow"]] = [
                HANGING_SHOULDER + displacement,
                0.0,
            ]
            physics.named.data.qvel[["shoulder", "elbow"]] = 0.0
        elif self.release_start:
            # roa_start_fraction's draw: a state sampled uniformly (by
            # volume) from the SOS-certified region of attraction itself,
            # angles AND rates -- not the release law's zero-velocity,
            # elbow-locked start. See controllers.acrobot_sos_switched.
            certificate = self._ensure_sos_certificate()
            state = certificate.sample_uniform(self.random)
            physics.named.data.qpos[["shoulder", "elbow"]] = state[:2]
            physics.named.data.qvel[["shoulder", "elbow"]] = state[2:]
        elif self.paper_start:
            physics.named.data.qpos[["shoulder", "elbow"]] = [
                PAPER_INITIAL_SHOULDER,
                0.0,
            ]
            physics.named.data.qvel[["shoulder", "elbow"]] = 0.0
        else:
            physics.named.data.qpos[["shoulder", "elbow"]] = [
                HANGING_SHOULDER,
                0.0,
            ] + self.random.uniform(-self.angle_noise, self.angle_noise, 2)
            physics.named.data.qvel[["shoulder", "elbow"]] = self.random.uniform(
                -self.velocity_noise, self.velocity_noise, 2
            )
        super().initialize_episode(physics)

    @property
    def last_termination_reason(self) -> Optional[str]:
        """Stable reason code for the most recent state-limit termination."""
        return self._last_termination_reason

    def get_termination(self, physics) -> Optional[float]:
        """Terminate energetic runaways, and return dm-control discount zero.

        The tests here read the post-step state. The method reads ``q2``
        directly from ``qpos`` and never wraps it, so a winding elbow can reach
        its limit. The terminal endpoint can overshoot by one control interval.
        The method applies no nonphysical state clipping.
        """
        rate_scale = self._rate_scale
        if rate_scale is None:
            raise RuntimeError(
                "Acrobot termination limits require an initialized episode"
            )
        qpos = np.asarray(physics.data.qpos, dtype=np.float64).reshape(-1)
        qvel = np.asarray(physics.data.qvel, dtype=np.float64).reshape(-1)
        reason: Optional[str] = None
        if abs(float(qpos[1])) >= self.elbow_angle_limit:
            reason = TERMINATION_ELBOW_ANGLE
        elif abs(float(qvel[1])) >= self.elbow_rate_limit:
            reason = TERMINATION_ELBOW_RATE
        elif (
            abs(float(qvel[0]))
            >= self.shoulder_rate_scale_limit * rate_scale
        ):
            reason = TERMINATION_SHOULDER_RATE
        self._last_termination_reason = reason
        return 0.0 if reason is not None else None

    def get_observation(self, physics):
        """Wrapped joint angles and rates.

        The stock Acrobot helpers of dm_control read the body z-axes, and so
        assume links along +z. This model lays the links along +x, so the
        method builds the observation from the joint angles directly. The
        swing-up arms use ``raw_state_obs``, which skips this method.
        """
        angles = np.asarray(physics.data.qpos, dtype=np.float64)
        obs = collections.OrderedDict()
        obs["orientations"] = np.concatenate([np.cos(angles), np.sin(angles)])
        obs["velocity"] = physics.velocity()
        return obs

    # --- Reward -----------------------------------------------------------

    def get_reward(self, physics) -> float:
        terms = self.xk_reward_terms(physics)
        self._last_reward_terms = terms
        return float(terms["reward"])

    @property
    def last_reward_terms(self) -> Optional[Dict[str, float]]:
        """Terms used for the most recently emitted transition reward."""
        if self._last_reward_terms is None:
            return None
        return dict(self._last_reward_terms)

    def _ensure_sos_certificate(self):
        """Load and cache the SOS-certified region of attraction.

        Needs no physics (unlike :meth:`_ensure_lai_she`): the certificate is
        a fixed numerical result loaded from
        ``results/acrobot_sos_roa_tau20_certificate.json``, already in this
        plant's own paper-frame coordinates.
        """
        if self._sos_certificate is None:
            from controllers.acrobot_sos_switched import load_certificate

            self._sos_certificate = load_certificate()
        return self._sos_certificate

    def _ensure_lai_she(self, physics):
        """Build and cache the Lai-She nonsmooth Lyapunov function.

        The build waits for first use, because it needs ``physics``. It
        recovers the mass, length and inertia of this plant through
        :meth:`AcrobotParams.from_physics`. Those parameters stay the same
        between resets, so the method builds the function once and reuses it
        for the lifetime of the task. It reads the task's own ``k_v``, ``k_d``
        and ``k_p`` gains, and never the paper-section defaults of this module.
        A caller therefore tunes it exactly as it tunes the other reward bases.
        """
        if self._lai_she is None:
            from controllers.acrobot_gated_lyapunov import (
                AttractiveRegion,
                NonsmoothLyapunov,
            )
            from controllers.xin_kaneda import AcrobotParams, Gains

            params = AcrobotParams.from_physics(physics)
            gains = Gains(k_v=self.k_v, k_d=self.k_d, k_p=self.k_p)
            self._lai_she = NonsmoothLyapunov(params, gains, AttractiveRegion())
        return self._lai_she

    def baseline_terms(self, physics) -> Dict[str, float]:
        """The ``r0`` baseline and its three normalized parts.

        The scales follow the doc. ``E_s`` is the swing-up energy span, and
        ``q_s = pi``. ``omega_s`` is the peak shoulder speed on the homoclinic
        orbit, ``sqrt(2 E_s / M11(0))``. That is the speed at which the extended
        pose carries the whole span as kinetic energy.
        """
        if self._energy_hang is None or self._energy_span is None:
            self._calibrate_energy(physics)
        qpos = np.asarray(physics.data.qpos, dtype=np.float64).reshape(-1)
        qvel = np.asarray(physics.data.qvel, dtype=np.float64).reshape(-1)
        energy_error = self._mechanical_energy(physics) - self.energy_top
        elbow = float(np.arctan2(np.sin(qpos[1]), np.cos(qpos[1])))
        omega_scale = self._rate_scale
        parts = (
            (energy_error / self.energy_span) ** 2,
            (elbow / np.pi) ** 2,
            (float(qvel[1]) / omega_scale) ** 2,
        )
        return {
            "reward": -float(sum(parts)),
            "energy_error": float(energy_error),
            "energy_error_norm": float(energy_error / self.energy_span),
            "elbow": elbow,
            "elbow_norm": float(elbow / np.pi),
            "elbow_rate": float(qvel[1]),
            "elbow_rate_norm": float(qvel[1] / omega_scale),
            "shoulder_rate": float(qvel[0]),
        }

    def xk_diagnostic_terms(self, physics) -> Dict[str, float]:
        """Reward-independent endpoint diagnostics for checkpoint selection."""
        baseline = self.baseline_terms(physics)
        inside = (
            abs(baseline["energy_error_norm"])
            <= HOMOCLINIC_ENERGY_TOLERANCE
            and abs(baseline["elbow_norm"])
            <= HOMOCLINIC_ANGLE_TOLERANCE
            and abs(baseline["elbow_rate_norm"])
            <= HOMOCLINIC_RATE_TOLERANCE
        )
        return {
            "energy_error_norm": baseline["energy_error_norm"],
            "elbow_norm": baseline["elbow_norm"],
            "elbow_rate_norm": baseline["elbow_rate_norm"],
            "in_homoclinic_tube": float(inside),
        }

    def xk_reward_terms(self, physics) -> Dict[str, float]:
        """Return all four reward rates at the live endpoint state.

        ``r0`` is the normalized, periodic distance that this task already
        ships. The Lyapunov rewards use the unwrapped shape coordinate of
        Xin--Kaneda, because their function penalizes elbow winding on ``R``.
        ``reward_base`` picks the r1 state term, either ``-V / V_down`` or
        ``r0``. r2 and r3 build on that choice. The derivative and the r3
        correction always keep the original ``V`` and its common ``V_down``
        scale. The actual derivative reads the generalized force that the plant
        applies, so nothing reads the normalized policy action as a physical
        torque. The Xin--Kaneda closed-loop value carries its own name. It is
        the optional action-independent surrogate ``-k_V qdot2^2``.
        """
        baseline = self.baseline_terms(physics)
        qpos = np.asarray(physics.data.qpos, dtype=np.float64).reshape(-1)
        qvel = np.asarray(physics.data.qvel, dtype=np.float64).reshape(-1)
        energy_error = baseline["energy_error"]

        mass = self._mass_matrix(physics)
        actuator_force = np.asarray(
            physics.data.qfrc_actuator, dtype=np.float64
        ).reshape(-1)
        passive_force = np.asarray(
            physics.data.qfrc_passive, dtype=np.float64
        ).reshape(-1)
        external_force = np.asarray(
            physics.data.qfrc_applied, dtype=np.float64
        ).reshape(-1)
        constraint_force = np.asarray(
            physics.data.qfrc_constraint, dtype=np.float64
        ).reshape(-1)
        bias_force = np.asarray(
            physics.data.qfrc_bias, dtype=np.float64
        ).reshape(-1)
        applied_force = (
            actuator_force
            + passive_force
            + external_force
            + constraint_force
        )
        qacc = np.linalg.solve(mass, applied_force - bias_force)
        energy_rate = float(qvel @ applied_force)

        elbow_unwrapped = float(qpos[1])
        elbow_rate = float(qvel[1])
        lyapunov_scale = self.lyapunov_scale
        if self.reward_base == "lai_she":
            # This plant already *is* the Xin-Kaneda paper's own coordinates.
            # The [qpos; qvel] of raw_state_obs needs no frame conversion, so
            # the state and its derivative come straight off physics.
            # NonsmoothLyapunov.value and .rate are already normalized on their
            # own scale. A multiplication back by lyapunov_scale here keeps
            # every later step byte-for-byte the same as the other reward
            # bases: normalization, selection, and raw_r1 through raw_r3.
            lai_she = self._ensure_lai_she(physics)
            state = np.array(
                [qpos[0], qpos[1], qvel[0], qvel[1]], dtype=np.float64
            )
            xdot = np.array(
                [qvel[0], qvel[1], float(qacc[0]), float(qacc[1])],
                dtype=np.float64,
            )
            lyapunov = float(lai_she.value(state)) * lyapunov_scale
            lyapunov_rate = float(lai_she.rate(state, xdot)) * lyapunov_scale
        else:
            lyapunov = float(
                0.5 * energy_error**2
                + 0.5 * self.k_d * elbow_rate**2
                + 0.5 * self.k_p * elbow_unwrapped**2
            )
            lyapunov_rate = float(
                energy_error * energy_rate
                + self.k_d * elbow_rate * float(qacc[1])
                + self.k_p * elbow_unwrapped * elbow_rate
            )
        xk_closed_loop_lyapunov_rate = -self.k_v * elbow_rate**2
        selected_lyapunov_rate = (
            lyapunov_rate
            if self.lyapunov_rate_source == "actual"
            else xk_closed_loop_lyapunov_rate
        )
        lyapunov_normalized = lyapunov / lyapunov_scale
        lyapunov_rate_normalized = lyapunov_rate / lyapunov_scale
        xk_closed_loop_lyapunov_rate_normalized = (
            xk_closed_loop_lyapunov_rate / lyapunov_scale
        )
        selected_lyapunov_rate_normalized = (
            selected_lyapunov_rate / lyapunov_scale
        )
        eta = float(self.eta or 0.0)
        discount_rate = float(self.discount_rate or 0.0)
        raw_r0 = float(baseline["reward"])
        lyapunov_reward = -lyapunov_normalized
        raw_r1 = raw_r0 if self.reward_base == "r0" else lyapunov_reward
        raw_r2 = raw_r1 - eta * selected_lyapunov_rate_normalized
        raw_r3 = raw_r2 + discount_rate * eta * lyapunov_normalized
        raw_rewards = {
            "r0": raw_r0,
            "r1": raw_r1,
            "r2": raw_r2,
            "r3": raw_r3,
        }
        rewards = {
            name: _transform_reward(value, self.reward_transform)
            for name, value in raw_rewards.items()
        }
        selected = {
            "r0": rewards["r0"],
            "r1": rewards["r1"],
            "r2": rewards["r2"],
            "r3": rewards["r3"],
        }[self.reward_kind]
        raw_selected = raw_rewards[self.reward_kind]
        return {
            "reward": float(selected),
            "raw_reward": float(raw_selected),
            "r0": rewards["r0"],
            "r1": rewards["r1"],
            "r2": rewards["r2"],
            "r3": rewards["r3"],
            "raw_r0": raw_r0,
            "raw_r1": raw_r1,
            "raw_r2": raw_r2,
            "raw_r3": raw_r3,
            "lyapunov_reward": lyapunov_reward,
            "lyapunov": lyapunov,
            "lyapunov_rate": lyapunov_rate,
            "xk_closed_loop_lyapunov_rate": xk_closed_loop_lyapunov_rate,
            "selected_lyapunov_rate": selected_lyapunov_rate,
            "lyapunov_scale": lyapunov_scale,
            "lyapunov_normalized": lyapunov_normalized,
            "lyapunov_rate_normalized": lyapunov_rate_normalized,
            "xk_closed_loop_lyapunov_rate_normalized": (
                xk_closed_loop_lyapunov_rate_normalized
            ),
            "selected_lyapunov_rate_normalized": (
                selected_lyapunov_rate_normalized
            ),
            "energy_rate": energy_rate,
            "elbow_acceleration": float(qacc[1]),
            "applied_torque": float(actuator_force[1]),
        }


def swingup_xk(
    *,
    time_limit: float = 60.0,
    random=None,
    environment_kwargs: Optional[Dict[str, Any]] = None,
    damping: float = 0.0,
    torque_limit: float = DEFAULT_TORQUE_LIMIT,
    angle_noise: float = DEFAULT_ANGLE_NOISE,
    velocity_noise: float = DEFAULT_VELOCITY_NOISE,
    uniform_start: bool = False,
    paper_start: bool = False,
    release_start: bool = False,
    release_angle_range: tuple = RELEASE_ANGLE_RANGE,
    roa_start_fraction: float = 0.0,
    reward_kind: str = "r0",
    reward_base: str = DEFAULT_REWARD_BASE,
    reward_transform: str = DEFAULT_REWARD_TRANSFORM,
    k_d: float = DEFAULT_LYAPUNOV_K_D,
    k_p: float = DEFAULT_LYAPUNOV_K_P,
    k_v: float = DEFAULT_LYAPUNOV_K_V,
    eta: Optional[float] = None,
    discount_rate: Optional[float] = None,
    lyapunov_rate_source: str = "actual",
    elbow_angle_limit: float = ELBOW_ANGLE_LIMIT,
    elbow_rate_limit: float = ELBOW_RATE_LIMIT,
    shoulder_rate_scale_limit: float = SHOULDER_RATE_SCALE_LIMIT,
    cap_terminal_penalty: bool = True,
    plant_scales: Optional[Dict[str, float]] = None,
):
    """Construct ``acrobot-swingup-xk``.

    This plant deviates from the stock model in two places, ``damping = 0`` and
    a configurable ``torque_limit``. The built model carries both values, so a
    caller can always make sure that it holds the plant it expects.

    ``uniform_start=True`` samples both joint angles over ``[-pi, pi)`` and
    releases them at exactly zero joint velocity. It excludes ``paper_start``
    and the near-hanging ``release_start``.

    ``roa_start_fraction`` only means anything alongside
    ``release_start=True``. It replaces that fraction of the ``release_start``
    draws with a state drawn uniformly by volume from the SOS-certified region
    of attraction itself -- angles and rates both, not the release law's
    zero-velocity start. See
    ``controllers.acrobot_sos_switched.SOSCertificate.sample_uniform``.

    ``reward_kind`` chooses the training reward. For r1 through r3,
    ``reward_base`` chooses between the historical ``-V / V_down`` state term
    and ``r0``. The r0 choice keeps the shaping terms built on the original
    ``V``. ``r2`` and ``r3`` also require an explicit ``eta`` shaping time
    scale, and ``r3`` requires the physical ``discount_rate`` that CT-SAC uses.
    ``lyapunov_rate_source`` selects the actual action-dependent derivative, or
    the counterfactual Xin--Kaneda closed-loop surrogate.
    ``reward_transform="log_reciprocal"`` applies ``log(1 / -r)`` with a finite
    numerical floor. The three state limits are instance kwargs.

    ``cap_terminal_penalty=True`` is the default. A cap crossing then emits the
    finite lower envelope of the selected reward as the terminal reward, and
    freezes that rate. The lower envelope is the minimum reward attainable
    anywhere in the capped state and action closure. CT-SAC sums the frozen rate
    over the part of the episode that never runs. With
    ``cap_terminal_penalty=False``, a cap crossing is an ordinary termination.
    The episode ends there at the live reward computed for that terminal state,
    with no penalty override and no analytical continuation.
    """
    plant_scales = PlantScales.coerce(plant_scales)
    physics = mujoco.Physics.from_xml_string(
        _model_xml(damping, torque_limit, plant_scales), common.ASSETS
    )
    task = BalanceXK(
        random=random,
        angle_noise=angle_noise,
        velocity_noise=velocity_noise,
        uniform_start=uniform_start,
        paper_start=paper_start,
        release_start=release_start,
        release_angle_range=release_angle_range,
        roa_start_fraction=roa_start_fraction,
        reward_kind=reward_kind,
        reward_base=reward_base,
        reward_transform=reward_transform,
        k_d=k_d,
        k_p=k_p,
        k_v=k_v,
        eta=eta,
        discount_rate=discount_rate,
        lyapunov_rate_source=lyapunov_rate_source,
        torque_limit=torque_limit,
        damping=damping,
        elbow_angle_limit=elbow_angle_limit,
        elbow_rate_limit=elbow_rate_limit,
        shoulder_rate_scale_limit=shoulder_rate_scale_limit,
        cap_terminal_penalty=cap_terminal_penalty,
        plant_scales=plant_scales,
    )
    return control.Environment(
        physics,
        task,
        time_limit=float(time_limit),
        **dict(environment_kwargs or {}),
    )
