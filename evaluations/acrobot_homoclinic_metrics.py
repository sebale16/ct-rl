"""Reward-independent metrics for Acrobot swing-up toward the homoclinic orbit.

Implements metrics 1-6 of ``docs/reward_shaping_for_acrobot_swingup.md`` as pure
functions of a recorded trajectory, so the analytical Xin-Kaneda controller and a
learned CT-SAC policy are scored by identical code and neither is credited for
the reward it was trained on.

Everything is computed from raw state ``[q1, q2, qdot1, qdot2]`` plus the applied
torque.  Nothing is read from ``info``, and nothing is shared with the
``v2 ... v6.1`` reward line.

Scales
------
The doc leaves the normalizations to be chosen; all three are derived rather
than tuned.

* ``E_s = E_top - E_down = 2 (b1 + b2)`` — the energy a swing-up must supply.
* ``q_s = pi``.
* ``omega_s = sqrt(4 (b1 + b2) / (a1 + a2 + 2 a3))`` — the peak shoulder speed on
  the homoclinic orbit itself, which is also the speed at which the full energy
  span is carried as kinetic energy in the extended pose.  It normalizes both
  the elbow rate in ``r0`` and the shoulder rate in ``d_Gamma``.

The homoclinic orbit
--------------------
Eq. 32 of 2007 is ``1/2 (a1+a2+2a3) qdot1^2 = (b1+b2)(1 - sin q1)``, that is

    qdot1 = +- omega_s * sqrt((1 - sin q1) / 2),   q2 = qdot2 = 0.

It passes through the upright pose ``q1 = pi/2`` at rest and reaches
``|qdot1| = omega_s`` at hanging.

Coordinates
-----------
Everything here is in the paper's frame, which is what ``acrobot-swingup-xk``
reports directly: ``q1`` is measured from the horizontal, upright is
``q1 = pi/2`` and hanging ``q1 = -pi/2``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from controllers.xin_kaneda import AcrobotParams, homoclinic_speed


# Resolution of the parameterization of Gamma used for the distance metric, and
# the trajectory block size that bounds the pairwise distance matrix.
_ORBIT_SAMPLES = 4001
_DISTANCE_BLOCK = 512


def _wrap(angle):
    """Map an angle to ``(-pi, pi]``."""
    return np.arctan2(np.sin(angle), np.cos(angle))


@dataclass(frozen=True)
class Scales:
    """Normalizations shared by the tube, the error, and the orbit distance."""

    energy: float
    angle: float
    rate: float

    @classmethod
    def from_params(cls, params: AcrobotParams) -> "Scales":
        return cls(
            energy=params.energy_span,
            angle=float(np.pi),
            rate=homoclinic_speed(params),
        )


@dataclass(frozen=True)
class TubeSpec:
    """The tolerance tube ``H`` around the homoclinic set, and its dwell time.

    Each bound is a fraction of the corresponding scale, and the tube is
    independent of the reward under test.  The elbow bound is half the other
    two: ``q_s = pi`` is a much larger scale than the swing-up energy span or
    the orbit's peak speed, so an equal fraction of it would be a far looser
    constraint on the pose than on the energy.
    """

    energy_tolerance: float = 0.05
    angle_tolerance: float = 0.025
    rate_tolerance: float = 0.05
    dwell_seconds: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "energy_tolerance",
            "angle_tolerance",
            "rate_tolerance",
            "dwell_seconds",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {value}")


@dataclass
class Trajectory:
    """One recorded episode.

    ``time`` and ``state`` hold the ``N + 1`` interval endpoints; ``torque`` and
    ``commanded_torque`` hold the ``N`` per-interval values, with ``torque`` the
    value the plant actually applied and ``commanded_torque`` what the policy
    asked for before any actuator limit.
    """

    time: np.ndarray
    state: np.ndarray
    torque: np.ndarray
    commanded_torque: np.ndarray
    torque_limit: float

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=np.float64).reshape(-1)
        self.state = np.asarray(self.state, dtype=np.float64)
        self.torque = np.asarray(self.torque, dtype=np.float64).reshape(-1)
        self.commanded_torque = np.asarray(
            self.commanded_torque, dtype=np.float64
        ).reshape(-1)
        n = self.time.size - 1
        if n < 1:
            raise ValueError("a trajectory needs at least one interval")
        if self.state.shape != (n + 1, 4):
            raise ValueError(
                f"state must have shape ({n + 1}, 4), got {self.state.shape}"
            )
        if self.torque.size != n or self.commanded_torque.size != n:
            raise ValueError(
                f"torque arrays must have {n} entries, got {self.torque.size} "
                f"and {self.commanded_torque.size}"
            )
        if np.any(np.diff(self.time) <= 0.0):
            raise ValueError("trajectory times must be strictly increasing")

    @property
    def dt(self) -> np.ndarray:
        return np.diff(self.time)

    @property
    def duration(self) -> float:
        return float(self.time[-1] - self.time[0])


# --- State-derived quantities --------------------------------------------


def energy_error(state: np.ndarray, params: AcrobotParams) -> np.ndarray:
    """``E - E_r`` for each row of ``state``, in the paper's coordinates."""
    values = np.atleast_2d(np.asarray(state, dtype=np.float64))
    q1, q2 = values[:, 0], values[:, 1]
    d1, d2 = values[:, 2], values[:, 3]
    c2 = np.cos(q2)
    m11 = params.a1 + params.a2 + 2.0 * params.a3 * c2
    m12 = params.a2 + params.a3 * c2
    kinetic = 0.5 * (m11 * d1**2 + 2.0 * m12 * d1 * d2 + params.a2 * d2**2)
    potential = params.b1 * np.sin(q1) + params.b2 * np.sin(q1 + q2)
    return kinetic + potential - params.energy_top


def normalized_error_components(
    state: np.ndarray, params: AcrobotParams, scales: Optional[Scales] = None
) -> np.ndarray:
    """The three normalized deviations ``(Etil/E_s, q2/q_s, qdot2/omega_s)``."""
    scales = scales or Scales.from_params(params)
    values = np.atleast_2d(np.asarray(state, dtype=np.float64))
    return np.stack(
        [
            energy_error(values, params) / scales.energy,
            _wrap(values[:, 1]) / scales.angle,
            values[:, 3] / scales.rate,
        ],
        axis=1,
    )


def squared_distance(
    state: np.ndarray, params: AcrobotParams, scales: Optional[Scales] = None
) -> np.ndarray:
    """``d^2(x) = -r0(x)``: the summed squares of the normalized deviations."""
    return np.sum(normalized_error_components(state, params, scales) ** 2, axis=1)


def inside_tube(
    state: np.ndarray,
    params: AcrobotParams,
    spec: TubeSpec,
    scales: Optional[Scales] = None,
) -> np.ndarray:
    """Boolean membership of ``H`` for each row of ``state``."""
    components = np.abs(normalized_error_components(state, params, scales))
    bounds = np.array(
        [spec.energy_tolerance, spec.angle_tolerance, spec.rate_tolerance]
    )
    return np.all(components <= bounds, axis=1)


def orbit_distance(
    state: np.ndarray, params: AcrobotParams, scales: Optional[Scales] = None
) -> np.ndarray:
    """``d_Gamma``: normalized distance from ``(q1, qdot1)`` to the orbit.

    Gamma is eq. 32 solved for the rate, ``qdot1 = +- omega_s sqrt((1 - sin q1)/2)``;
    both signed branches are covered by sweeping ``q1`` over a full turn twice.
    """
    scales = scales or Scales.from_params(params)
    values = np.atleast_2d(np.asarray(state, dtype=np.float64))
    # Writing the offset from upright as u = q1 - pi/2 turns (1 - sin q1)/2 into
    # sin^2(u/2), so the orbit is the smooth closed curve
    # (q1, qdot1) = (pi/2 + u, omega_s sin(u/2)) traced once as u runs over the
    # 4*pi period of sin(u/2).  That covers both signed branches and, unlike
    # sweeping q1 directly, samples evenly through the upright pose where the
    # rate has a square-root profile in q1.
    offset = np.linspace(0.0, 4.0 * np.pi, _ORBIT_SAMPLES)
    orbit_q1 = _wrap(0.5 * np.pi + offset)
    orbit_rate = scales.rate * np.sin(0.5 * offset)
    out = np.empty(values.shape[0], dtype=np.float64)
    for start in range(0, values.shape[0], _DISTANCE_BLOCK):
        block = values[start : start + _DISTANCE_BLOCK]
        angle_gap = _wrap(block[:, 0:1] - orbit_q1[None, :]) / scales.angle
        rate_gap = (block[:, 2:3] - orbit_rate[None, :]) / scales.rate
        out[start : start + block.shape[0]] = np.sqrt(
            np.min(angle_gap**2 + rate_gap**2, axis=1)
        )
    return out


def lqr_residual(state: np.ndarray) -> np.ndarray:
    """The switching function of eq. 74 of 2007.

    ``|x1| + |x2| + 0.1|x3| + 0.1|x4|`` with ``x = [q1 - pi/2, q2, qdot1, qdot2]``
    the error to the upright equilibrium of eq. 10, both angles wrapped.
    """
    values = np.atleast_2d(np.asarray(state, dtype=np.float64))
    return (
        np.abs(_wrap(values[:, 0] - 0.5 * np.pi))
        + np.abs(_wrap(values[:, 1]))
        + 0.1 * np.abs(values[:, 2])
        + 0.1 * np.abs(values[:, 3])
    )


# --- Time-domain reductions ----------------------------------------------


def _interval_inside(inside: np.ndarray) -> np.ndarray:
    """An interval counts as inside only if both its endpoints qualify.

    This is what keeps the interval straddling first entry from being credited
    to the captured state.
    """
    return inside[:-1] & inside[1:]


def capture_time(
    time: np.ndarray, inside: np.ndarray, dwell_seconds: float
) -> float:
    """``T_cap``: first ``t`` with ``x(s) in H`` for all ``s`` in ``[t, t+Delta]``.

    Returns the start of the first qualifying run, or ``inf`` if no run of
    physical length ``dwell_seconds`` exists.
    """
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    qualifying = _interval_inside(np.asarray(inside, dtype=bool).reshape(-1))
    run_start: Optional[float] = None
    for index, ok in enumerate(qualifying):
        if not ok:
            run_start = None
            continue
        if run_start is None:
            run_start = float(time[index])
        if time[index + 1] - run_start >= dwell_seconds - 1e-9:
            return run_start
    return float("inf")


def retention_fraction(
    time: np.ndarray, inside: np.ndarray, capture: float
) -> float:
    """``rho_H``: fraction of physical time inside ``H`` after capture."""
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    if not np.isfinite(capture) or capture >= time[-1]:
        return float("nan")
    qualifying = _interval_inside(np.asarray(inside, dtype=bool).reshape(-1))
    weights = _post_capture_weights(time, capture)
    total = float(np.sum(weights))
    if total <= 0.0:
        return float("nan")
    return float(np.sum(weights * qualifying) / total)


def _post_capture_weights(time: np.ndarray, capture: float) -> np.ndarray:
    """Physical duration of each interval that falls after ``capture``."""
    left = np.maximum(time[:-1], capture)
    right = np.maximum(time[1:], capture)
    return np.maximum(right - left, 0.0)


def time_weighted_rms(
    time: np.ndarray, values: np.ndarray, capture: float
) -> float:
    """Root of the post-capture time average of ``values`` (already squared).

    ``values`` is given per endpoint; each interval takes the mean of its two
    endpoints, which is the trapezoid rule the integral in the doc asks for.
    """
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(capture) or capture >= time[-1]:
        return float("nan")
    weights = _post_capture_weights(time, capture)
    total = float(np.sum(weights))
    if total <= 0.0:
        return float("nan")
    midpoints = 0.5 * (values[:-1] + values[1:])
    return float(np.sqrt(np.sum(weights * midpoints) / total))


def first_time_below(
    time: np.ndarray, values: np.ndarray, threshold: float
) -> float:
    """First endpoint time at which ``values`` drops below ``threshold``."""
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    hits = np.flatnonzero(values < threshold)
    return float(time[hits[0]]) if hits.size else float("inf")


def control_effort(
    time: np.ndarray, torque: np.ndarray, horizon: float
) -> float:
    """``J_u = integral of tau2^2`` up to ``horizon`` (``T_cap`` in the doc).

    A non-finite horizon integrates the whole episode, which is what makes the
    number comparable across episodes that never captured.
    """
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    torque = np.asarray(torque, dtype=np.float64).reshape(-1)
    limit = time[-1] if not np.isfinite(horizon) else min(horizon, time[-1])
    left = np.minimum(time[:-1], limit)
    right = np.minimum(time[1:], limit)
    return float(np.sum(np.maximum(right - left, 0.0) * torque**2))


def saturation_fraction(
    time: np.ndarray, commanded: np.ndarray, torque_limit: float
) -> float:
    """``rho_sat``: fraction of physical time with the command at the limit."""
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    commanded = np.asarray(commanded, dtype=np.float64).reshape(-1)
    dt = np.diff(time)
    at_limit = np.abs(commanded) >= torque_limit * (1.0 - 1e-9)
    total = float(np.sum(dt))
    return 0.0 if total <= 0.0 else float(np.sum(dt * at_limit) / total)


# --- Episode summary ------------------------------------------------------


@dataclass(frozen=True)
class EpisodeMetrics:
    """Metrics 1-6 for one episode."""

    duration: float
    captured: bool
    capture_time: float
    retention: float
    error_rms: float
    error_rms_energy: float
    error_rms_angle: float
    error_rms_rate: float
    orbit_distance_rms: float
    orbit_distance_final: float
    control_effort: float
    saturation: float
    lqr_time: float
    lqr_residual_min: float
    min_abs_energy_error: float
    final_abs_energy_error: float
    peak_shoulder_rate: float
    peak_commanded_torque: float

    def as_row(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


LQR_SWITCH_THRESHOLD = 0.04


def evaluate_episode(
    trajectory: Trajectory,
    params: AcrobotParams,
    spec: TubeSpec = TubeSpec(),
    *,
    lqr_threshold: float = LQR_SWITCH_THRESHOLD,
) -> EpisodeMetrics:
    """Reduce one trajectory to metrics 1-6."""
    scales = Scales.from_params(params)
    state, time = trajectory.state, trajectory.time
    components = normalized_error_components(state, params, scales)
    inside = inside_tube(state, params, spec, scales)
    capture = capture_time(time, inside, spec.dwell_seconds)
    distances = orbit_distance(state, params, scales)
    errors = np.abs(energy_error(state, params))
    # The switching set is tight enough that the residual dips below it only
    # briefly, so a zero-order-hold controller on a coarse control period can
    # pass through and never sample the crossing.  Reporting the closest
    # approach alongside the crossing time keeps the metric readable when that
    # happens, and makes the sampling dependence visible instead of silent.
    residual = lqr_residual(state)
    return EpisodeMetrics(
        duration=trajectory.duration,
        captured=bool(np.isfinite(capture)),
        capture_time=capture,
        retention=retention_fraction(time, inside, capture),
        error_rms=time_weighted_rms(
            time, np.sum(components**2, axis=1), capture
        ),
        error_rms_energy=time_weighted_rms(time, components[:, 0] ** 2, capture),
        error_rms_angle=time_weighted_rms(time, components[:, 1] ** 2, capture),
        error_rms_rate=time_weighted_rms(time, components[:, 2] ** 2, capture),
        orbit_distance_rms=time_weighted_rms(time, distances**2, capture),
        orbit_distance_final=float(distances[-1]),
        control_effort=control_effort(time, trajectory.torque, capture),
        saturation=saturation_fraction(
            time, trajectory.commanded_torque, trajectory.torque_limit
        ),
        lqr_time=first_time_below(time, residual, lqr_threshold),
        lqr_residual_min=float(np.min(residual)),
        min_abs_energy_error=float(np.min(errors)),
        final_abs_energy_error=float(errors[-1]),
        peak_shoulder_rate=float(np.max(np.abs(state[:, 2]))),
        peak_commanded_torque=float(np.max(np.abs(trajectory.commanded_torque))),
    )


@dataclass(frozen=True)
class AggregateMetrics:
    """Metrics 1 and 2 across a set of initial conditions."""

    episodes: int
    success_rate: float
    capture_times: np.ndarray = field(repr=False)

    @property
    def capture_quantiles(self) -> dict:
        finite = self.capture_times[np.isfinite(self.capture_times)]
        if finite.size == 0:
            return {"p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
        return {
            "p10": float(np.quantile(finite, 0.10)),
            "p50": float(np.quantile(finite, 0.50)),
            "p90": float(np.quantile(finite, 0.90)),
        }


def aggregate(results: Sequence[EpisodeMetrics]) -> AggregateMetrics:
    """``P(T_cap < T_max)`` and the distribution of ``T_cap``."""
    if not results:
        raise ValueError("aggregate needs at least one episode")
    times = np.array([r.capture_time for r in results], dtype=np.float64)
    return AggregateMetrics(
        episodes=len(results),
        success_rate=float(np.mean([r.captured for r in results])),
        capture_times=times,
    )


# --- Rollout driver -------------------------------------------------------


def rollout(
    env,
    act: Callable[[np.ndarray], np.ndarray],
    seed: int,
    *,
    torque_limit: Optional[float] = None,
) -> Trajectory:
    """Drive ``env`` with ``act`` for one episode and record the trajectory.

    ``env`` must be a :class:`~environment.dmc.DMCContinuousEnv` built with
    ``raw_state_obs=True``, so the observation is exactly ``[q1, q2, qd1, qd2]``.
    The applied torque is recovered as ``gear * ctrl``; the commanded torque is
    read from the controller when it publishes one, and otherwise equals the
    applied torque (a learned policy cannot ask for more than it commands).
    """
    gear = float(np.asarray(env._env.physics.model.actuator_gear)[0, 0])
    limit = gear if torque_limit is None else float(torque_limit)
    obs, _ = env.reset(seed=seed)
    reset = getattr(act, "reset", None)
    if callable(reset):
        reset()
    times = [float(env.cur_t)]
    states = [np.asarray(obs, dtype=np.float64).copy()]
    applied: list[float] = []
    commanded: list[float] = []
    while True:
        action = act(obs)
        _, _, action, _, obs, next_t, terminated, truncated = env.step_dt(action)[:8]
        # A pre-built time grid that runs out before the duration check clears
        # yields a final zero-length step, which carries no physical time and
        # would break the strictly-increasing invariant.
        if float(next_t) <= times[-1]:
            break
        times.append(float(next_t))
        states.append(np.asarray(obs, dtype=np.float64).copy())
        step_torque = gear * float(np.asarray(action).reshape(-1)[0])
        applied.append(step_torque)
        commanded.append(
            float(getattr(act, "last_commanded_torque", step_torque))
        )
        if terminated or truncated:
            break
    return Trajectory(
        time=np.array(times),
        state=np.array(states),
        torque=np.array(applied),
        commanded_torque=np.array(commanded),
        torque_limit=limit,
    )
