#!/usr/bin/env python
"""Plot the Lyapunov reward r1 = -V under the analytical controller.

Re-runs the shared evaluation protocol of
``docs/reward_shaping_for_acrobot_swingup.md`` -- the ``release`` reset, 32
starts, 20 s, 2 ms control period -- and records

    V(x) = 1/2 Etil^2 + 1/2 k_D qdot2^2 + 1/2 k_P q2^2,   r1 = -V

at every step, then draws the mean across starts with a +-1 standard deviation
band.  Every episode shares one time grid, so the band is a spread across
initial conditions at fixed time rather than an average over ragged runs.

V is drawn on a log axis, where the decay is legible over the three orders of
magnitude a linear axis compresses; r1 = -V is the same curve mirrored.

The Lyapunov levels at which the eq. 74 switching set becomes reachable are
computed by :func:`lqr_thresholds` but deliberately *not* drawn here: this
figure is generated at a 0.5 ms hold, where the control period contributes about
0.02 to that residual against a box of width 0.04, so the run cannot support a
read-off of when the switch becomes available.  Use a 0.1 ms hold for anything
that turns on the switching set.

For reference, the two levels are.  On the target set the shoulder turns back short of upright at
delta = sqrt(2|Etil| / E_r) and V ~ Etil^2 / 2 there, so setting delta = zeta
gives

    V* = zeta^4 E_r^2 / 8

(1.92e-4 at zeta = 0.04, E_r = 24.5).  This is a threshold on the *per-lap
closest approach*, not a sufficient condition for being inside the set at a
given instant: V vanishes on the whole homoclinic orbit, hanging included.

    MUJOCO_GL=disable python -m benchmarks.plot_acrobot_xk_baseline_r1
    MUJOCO_GL=disable python -m benchmarks.plot_acrobot_xk_baseline_r1 --validate

``--validate`` integrates the analytic closed loop from each start out to
``--validate-horizon`` and prints when V actually crosses V*.  It is a check on
the tail extrapolation, which underestimates: the decay rate itself decays as
the lap period grows near the saddle.
"""

import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "disable")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from controllers.xin_kaneda import AcrobotParams, Gains, XinKanedaController
from environment.dmc import DMCContinuousEnv
from evaluations.acrobot_homoclinic_metrics import (
    Scales,
    TubeSpec,
    energy_error,
    inside_tube,
)

# One series, so no categorical palette is in play: a single accent against
# recessive ink.  The band is the same hue at low alpha rather than a second
# color, since it is the same quantity's spread.
ACCENT = "#2b6cb0"
BAND = "#2b6cb0"
CAPTURE = "#b7791f"
TARGET = "#9b2c2c"
INK = "#1a202c"
MUTED = "#4a5568"


def lyapunov_series(state, params, gains):
    """``V(x)`` along a trajectory, from raw state in the paper's frame.

    ``q2`` enters **unwrapped**.  2007 §3 takes the underactuated ``q1`` in S^1
    but the actuated shape variable ``q2`` in R, and that is what makes V
    penalize winding: the elbow passes ``pi`` during the pump on this plant
    (max |q2| ~ 3.6 rad), and folding it to (-pi, pi] puts a spurious ~10% rise
    into an otherwise monotone V.  The vectorized form here agrees with
    ``controllers.xin_kaneda.lyapunov``.
    """
    values = np.atleast_2d(np.asarray(state, dtype=np.float64))
    return (
        0.5 * energy_error(values, params) ** 2
        + 0.5 * gains.k_d * values[:, 3] ** 2
        + 0.5 * gains.k_p * values[:, 1] ** 2
    )


def collect(args):
    """Roll out the protocol and return ``(time, V[n_starts, n_steps], capture)``."""
    gains = Gains(k_v=args.kv, k_d=args.kd, k_p=args.kp)
    spec = TubeSpec()
    curves, captures, time = [], [], None
    for seed in range(args.seed0, args.seed0 + args.starts):
        env = DMCContinuousEnv(
            "acrobot",
            "swingup-xk",
            seed=seed,
            raw_state_obs=True,
            dt=args.dt,
            physics_dt=args.dt,
            max_steps=int(round(args.duration / args.dt)) + 1,
            episode_duration=args.duration,
            task_kwargs=dict(release_start=True),
        )
        params = AcrobotParams.from_physics(env._env.physics)
        controller = XinKanedaController(params, gains)
        obs, _ = env.reset(seed=seed)
        times, states = [float(env.cur_t)], [np.asarray(obs, dtype=np.float64)]
        while True:
            result = env.step_dt(controller(obs))
            obs, next_t, terminated, truncated = result[4], result[5], result[6], result[7]
            if float(next_t) <= times[-1]:
                break
            times.append(float(next_t))
            states.append(np.asarray(obs, dtype=np.float64))
            if terminated or truncated:
                break
        states = np.asarray(states)
        curves.append(lyapunov_series(states, params, gains))
        scales = Scales.from_params(params)
        inside = inside_tube(states, params, spec, scales)
        run = 0.0
        capture = np.inf
        grid = np.asarray(times)
        for i in range(len(grid) - 1):
            if inside[i] and inside[i + 1]:
                run = run if run else grid[i]
                if grid[i + 1] - run >= spec.dwell_seconds - 1e-9:
                    capture = run
                    break
            else:
                run = 0.0
        captures.append(capture)
        if time is None:
            time = grid
        print(f"  seed {seed}: V(0) = {curves[-1][0]:9.2f}  capture {capture:6.2f} s",
              flush=True)
    length = min(len(c) for c in curves)
    values = np.stack([c[:length] for c in curves])
    # V is non-increasing under the exact law; a zero-order hold leaves a
    # residual that scales with the control period.  Guard the scale of it so a
    # sign or wrapping error cannot pass unnoticed.
    excursion = float(np.max(values - np.minimum.accumulate(values, axis=1)))
    tolerance = 10.0 * args.dt * values[:, 0].mean()
    print(f"  V excursion above running min: {excursion:.4f} (tolerance {tolerance:.3f})")
    if excursion > tolerance:
        raise RuntimeError(
            f"V rose by {excursion:.3f} above its running minimum, far beyond the "
            f"{tolerance:.3f} expected from a {args.dt} s hold; V should be "
            "non-increasing, so this indicates a sign or wrapping error."
        )
    return time[:length], values, np.asarray(captures), params


def _set_style():
    """The repo's paper style, inlined.

    ``evaluations.plot_helpers.set_paper_style`` is the house version but it
    imports seaborn, which is not in this environment; these are the rcParams it
    would set.
    """
    matplotlib.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "#EAEAF2",
            "axes.grid": True,
            "grid.color": "white",
            "grid.linewidth": 1.0,
            "axes.edgecolor": "#CCCCCC",
            "axes.linewidth": 1.0,
            "axes.axisbelow": True,
            "font.size": 11,
        }
    )


def validate(args, params):
    """Integrate the exact closed loop until V crosses V*, and report the mean.

    Uses continuous feedback rather than the
    plant's zero-order hold, since over hundreds of seconds the hold residual
    would otherwise dominate the tail.
    """
    from controllers.xin_kaneda import closed_loop

    gains = Gains(k_v=args.kv, k_d=args.kd, k_p=args.kp)
    target = lqr_thresholds(params, args.zeta)[0]
    step = args.dt
    crossings = []
    for seed in range(args.seed0, args.seed0 + args.validate_starts):
        env = DMCContinuousEnv(
            "acrobot", "swingup-xk", seed=seed, raw_state_obs=True,
            dt=step, physics_dt=step, max_steps=8, episode_duration=1.0,
            task_kwargs=dict(release_start=True),
        )
        state, _ = env.reset(seed=seed)
        state = np.asarray(state, dtype=np.float64)
        crossing = np.inf
        for index in range(int(args.validate_horizon / step)):
            value = (
                0.5 * (params.energy(state[:2], state[2:]) - params.energy_top) ** 2
                + 0.5 * gains.k_d * state[3] ** 2
                + 0.5 * gains.k_p * state[1] ** 2
            )
            if value < target:
                crossing = index * step
                break
            k1, _ = closed_loop(params, gains, state)
            k2, _ = closed_loop(params, gains, state + 0.5 * step * k1)
            k3, _ = closed_loop(params, gains, state + 0.5 * step * k2)
            k4, _ = closed_loop(params, gains, state + step * k3)
            state = state + (step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        crossings.append(crossing)
        print(f"  seed {seed}: V < V* at {crossing:.1f} s", flush=True)
    finite = np.array([c for c in crossings if np.isfinite(c)])
    mean = float(finite.mean()) if finite.size else float("nan")
    print(f"  measured crossing: mean {mean:.1f} s over {finite.size} starts")
    return mean


def _sci(value, digits=3):
    """Format as KaTeX-style scientific notation with ``digits`` significant figures."""
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10.0**exponent
    return rf"{mantissa:.{digits - 1}f}\times 10^{{{exponent}}}"


def lqr_thresholds(params, zeta):
    """Lyapunov levels at which the eq. 74 set becomes reachable, both branches.

    Which one applies depends on the sign the energy error approaches from, and
    they differ by a factor of ~360:

    * **pass-through** (Etil > 0): the shoulder crosses upright with speed
      ``qdot1 = sqrt(2 Etil / M11)``, and the residual costs only ``0.1 qdot1``.
      Setting that to zeta gives ``Etil = 50 M11 zeta^2`` and
      ``V = 1250 M11^2 zeta^4``.
    * **turning point** (Etil < 0): the shoulder stops short of upright at
      ``delta = sqrt(2|Etil| / E_r)`` and pays the full angle, giving
      ``V = zeta^4 E_r^2 / 8``.

    Measured closest approaches sit on whichever branch the run is on, and the
    ones that come near the box are pass-throughs, so that is the operative
    level; the turning-point value is the pessimistic bound.
    """
    m11 = params.a1 + params.a2 + 2.0 * params.a3
    return (
        1250.0 * m11**2 * zeta**4,
        zeta**4 * params.energy_top**2 / 8.0,
    )


def tail_fit(time, mean_value, window):
    """Exponential fit of the tail; returns ``(tau, intercept)`` for ``V ~ e^{-t/tau}``."""
    inside = (time >= window[0]) & (time <= window[1])
    slope, intercept = np.polyfit(time[inside], np.log(mean_value[inside]), 1)
    return -1.0 / slope, intercept


def draw(time, values, captures, params, args):
    _set_style()
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    tau, intercept = tail_fit(time, mean, (args.fit_from, time[-1]))
    # Reported, not drawn: the level itself is exact, but this figure's hold
    # cannot resolve the switching box it refers to (see the module docstring).
    horizon = (intercept - np.log(lqr_thresholds(params, args.zeta)[0])) * tau
    median_capture = float(np.median(captures[np.isfinite(captures)]))

    figure, ax = plt.subplots(figsize=(9.2, 5.4))
    floor = float(np.min(mean - std)) * 0.5
    ax.fill_between(
        time, np.maximum(mean - std, floor), mean + std,
        color=BAND, alpha=0.22, lw=0, zorder=2,
    )
    ax.plot(time, mean, color=ACCENT, lw=2.0, zorder=4)
    ax.axvline(median_capture, color=CAPTURE, lw=1.5, ls="--", zorder=3)

    ax.set_yscale("log")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(floor, mean.max() * 3.0)
    ax.set_ylabel(r"$V(x)$", color=INK)
    ax.set_xlabel("time (s)", color=INK)
    ax.set_title(
        f"Lyapunov function under the analytical controller, {values.shape[0]} starts "
        r"(mean $\pm$ 1 s.d.);  $r_1 = -V$",
        color=INK, fontsize=12, pad=10,
    )

    ax.annotate(
        f"median capture {median_capture:.1f} s",
        xy=(median_capture, mean.max() * 1.6), xytext=(6, 0),
        textcoords="offset points", color=CAPTURE, fontsize=9.5,
        va="top", ha="left",
    )
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color("#CCCCCC")

    figure.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.12)
    print(f"  tail tau = {tau:.2f} s, extrapolated V* at t = {horizon:.1f} s")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(f"wrote {args.output}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", default="results/acrobot_xk_baseline_r1.png")
    parser.add_argument("--starts", type=int, default=32)
    parser.add_argument("--seed0", type=int, default=20000)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--kv", type=float, default=66.3)
    parser.add_argument("--kd", type=float, default=35.8)
    parser.add_argument("--kp", type=float, default=61.2)
    parser.add_argument("--zeta", type=float, default=0.04)
    parser.add_argument(
        "--fit-from", type=float, default=8.0,
        help="start of the window used for the exponential tail fit",
    )
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validate-starts", type=int, default=8)
    parser.add_argument("--validate-horizon", type=float, default=700.0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    time, values, captures, params = collect(args)
    print(
        f"V(0) mean {values[:, 0].mean():.1f}, "
        f"V(T) mean {values[:, -1].mean():.4f}, "
        f"capture {np.isfinite(captures).sum()}/{len(captures)}"
    )
    if args.validate:
        validate(args, params)
    draw(time, values, captures, params, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
