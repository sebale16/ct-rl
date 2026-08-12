"""Evaluate the Xin-Kaneda swing-up controller on the conservative Acrobot.

Two modes:

``--sweep single``
    One gain/plant configuration over a range of seeds.

``--sweep frontier``
    The torque-versus-time frontier.  ``k_P`` is swept from just above the
    Proposition-4 floor (eq. 43) up past the ``2 b1 b2`` boundary, with damping
    on and off.  Below the boundary the hanging equilibrium has three unstable
    eigenvalues and the swing-up is fast but torque-hungry; above it the law
    asks for much less torque and the escape from hanging slows by orders of
    magnitude.  The sweep measures that trade on the plant.

Examples
--------
    MUJOCO_GL=disable python -m evaluations.eval_acrobot_xk \\
        --sweep single --kp 61.2 --kd 35.8 --kv 66.3 --start paper --dt 1e-4 \\
        --t-max 12 --seeds 20000:20001 --output results/acrobot_xk_paper.csv

    MUJOCO_GL=disable python -m evaluations.eval_acrobot_xk \\
        --sweep frontier --seeds 20000:20004 \\
        --output results/acrobot_xk_frontier.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

os.environ.setdefault("MUJOCO_GL", "disable")

import numpy as np

from controllers.xin_kaneda import (
    AcrobotParams,
    Gains,
    XinKanedaController,
    hanging_regime,
    kd_min,
    kp_boundary,
    kp_min,
)
from environment.dmc import DMCContinuousEnv
from evaluations.acrobot_homoclinic_metrics import (
    LQR_SWITCH_THRESHOLD,
    TubeSpec,
    aggregate,
    evaluate_episode,
    rollout,
)


SCHEMA_VERSION = 1

OUTPUT_FIELDS = (
    "schema_version",
    "seed",
    "start",
    "damping",
    "torque_limit",
    "k_v",
    "k_d",
    "k_p",
    "kp_floor",
    "kp_boundary",
    "kd_floor",
    "regime",
    "n_unstable",
    "escape_time_constant",
    "t_max",
    "dt",
    "physics_dt",
    "tube_energy",
    "tube_angle",
    "tube_rate",
    "dwell_seconds",
    "lqr_threshold",
    "duration",
    "captured",
    "capture_time",
    "retention",
    "error_rms",
    "error_rms_energy",
    "error_rms_angle",
    "error_rms_rate",
    "orbit_distance_rms",
    "orbit_distance_final",
    "control_effort",
    "saturation",
    "lqr_time",
    "lqr_residual_min",
    "min_abs_energy_error",
    "final_abs_energy_error",
    "peak_shoulder_rate",
    "peak_commanded_torque",
)

# The release-protocol peak demand is about 19.71 N*m at the paper's gains.
DEFAULT_TORQUE_LIMIT = 20.0

# The plant XML's own integration step.
MODEL_TIMESTEP = 0.01

# Xin-Kaneda's own Section 7 gains; k_D clears the eq. 25 floor of 35.741.
DEFAULT_K_V = 66.3
DEFAULT_K_D = 35.8


@dataclass(frozen=True)
class Arm:
    """One point of the evaluation matrix."""

    k_v: float
    k_d: float
    k_p: float
    torque_limit: float
    damping: float
    t_max: float
    start: str

    @property
    def gains(self) -> Gains:
        return Gains(k_v=self.k_v, k_d=self.k_d, k_p=self.k_p)


def _parse_seeds(text: str) -> List[int]:
    if ":" in text:
        start, stop = text.split(":", 1)
        return list(range(int(start), int(stop)))
    return [int(part) for part in text.split(",") if part.strip()]


def build_env(
    arm: Arm, seed: int, dt: float, physics_dt: Optional[float] = None
) -> DMCContinuousEnv:
    """Build the plant for one arm; ``max_steps`` is set from ``t_max``.

    The wrapper realizes a control period as a whole number of physics steps
    (``nsub = max(1, round(dt / physics_dt))``), so asking for a period finer
    than the model's own timestep is otherwise a silent no-op.  Defaulting the
    physics step to ``min(dt, MODEL_TIMESTEP)`` makes a fine ``dt`` take effect,
    which matters for the LQR switching test: its residual dips below the
    threshold only briefly, so a coarse period can step over the crossing.
    """
    step = min(dt, MODEL_TIMESTEP) if physics_dt is None else float(physics_dt)
    return DMCContinuousEnv(
        "acrobot",
        "swingup-xk",
        seed=seed,
        raw_state_obs=True,
        dt=dt,
        physics_dt=step,
        max_steps=int(round(arm.t_max / dt)) + 1,
        episode_duration=arm.t_max,
        task_kwargs=dict(
            damping=arm.damping,
            torque_limit=arm.torque_limit,
            paper_start=(arm.start == "paper"),
            uniform_start=(arm.start == "uniform"),
            release_start=(arm.start == "release"),
        ),
    )


def run_arm(
    arm: Arm,
    seeds: Sequence[int],
    *,
    dt: float,
    spec: TubeSpec,
    lqr_threshold: float,
    physics_dt: Optional[float] = None,
    verbose: bool = True,
) -> List[dict]:
    """Roll out one arm over ``seeds`` and return one CSV row per episode."""
    rows: List[dict] = []
    episodes = []
    regime: Optional[dict] = None
    params: Optional[AcrobotParams] = None
    for seed in seeds:
        env = build_env(arm, seed, dt, physics_dt)
        if params is None:
            params = AcrobotParams.from_physics(env._env.physics)
            regime = hanging_regime(params, arm.gains)
        controller = XinKanedaController(params, arm.gains)
        metrics = evaluate_episode(
            rollout(env, controller, seed),
            params,
            spec,
            lqr_threshold=lqr_threshold,
        )
        episodes.append(metrics)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "seed": seed,
                "start": arm.start,
                "damping": arm.damping,
                "torque_limit": arm.torque_limit,
                "k_v": arm.k_v,
                "k_d": arm.k_d,
                "k_p": arm.k_p,
                "kp_floor": kp_min(params),
                "kp_boundary": kp_boundary(params),
                "kd_floor": kd_min(params),
                "regime": regime["regime"],
                "n_unstable": regime["n_unstable"],
                "escape_time_constant": regime["escape_time_constant"],
                "t_max": arm.t_max,
                "dt": dt,
                "physics_dt": float(env.physics_dt),
                "tube_energy": spec.energy_tolerance,
                "tube_angle": spec.angle_tolerance,
                "tube_rate": spec.rate_tolerance,
                "dwell_seconds": spec.dwell_seconds,
                "lqr_threshold": lqr_threshold,
                **metrics.as_row(),
            }
        )
    if verbose:
        summary = aggregate(episodes)
        quantiles = summary.capture_quantiles
        print(
            f"  kP={arm.k_p:8.3f} lim={arm.torque_limit:5.1f} "
            f"damp={arm.damping:4.2f} [{regime['regime']:>15}] "
            f"P(cap)={summary.success_rate:4.2f} "
            f"T_cap(p50)={quantiles['p50']:8.2f} "
            f"peak|tau|={np.max([e.peak_commanded_torque for e in episodes]):7.3f} "
            f"sat={np.mean([e.saturation for e in episodes]):5.3f} "
            f"min|Etil|={np.min([e.min_abs_energy_error for e in episodes]):7.3f}",
            flush=True,
        )
    return rows


def frontier_arms(
    *,
    k_v: float,
    k_d: float,
    t_max_fast: float,
    t_max_slow: float,
    start: str,
    kp_values: Optional[Iterable[float]] = None,
) -> List[Arm]:
    """The default frontier matrix: ``k_P`` x actuator x damping.

    The ``k_P`` grid runs from just above the eq. 43 floor (61.141 on this
    plant) across the ``2 b1 b2`` boundary (288.12), which is where the hanging
    equilibrium changes spectral type.  Arms above it need the long horizon.
    """
    if kp_values is None:
        kp_values = (62.0, 80.0, 120.0, 180.0, 288.12, 400.0, 600.0)
    arms: List[Arm] = []
    for k_p in kp_values:
        t_max = t_max_fast if k_p < 288.12 else t_max_slow
        for torque_limit in (DEFAULT_TORQUE_LIMIT,):
            for damping in (0.0, 0.05):
                arms.append(
                    Arm(
                        k_v=k_v,
                        k_d=k_d,
                        k_p=k_p,
                        torque_limit=torque_limit,
                        damping=damping,
                        t_max=t_max,
                        start=start,
                    )
                )
    return arms


def summarize_frontier(rows: Sequence[dict], energy_span: float) -> List[str]:
    """State, per plant, the cheapest ``k_P`` that reaches the orbit.

    The frontier's point: torque demand falls monotonically as ``k_P`` rises
    toward ``2 b1 b2`` while the escape from hanging slows, so the question is
    whether the two ends overlap anywhere inside a given actuator budget.
    """
    lines: List[str] = []
    plants = sorted({(row["torque_limit"], row["damping"]) for row in rows})
    for torque_limit, damping in plants:
        subset = [
            row
            for row in rows
            if row["torque_limit"] == torque_limit and row["damping"] == damping
        ]
        captured = sorted(
            {row["k_p"] for row in subset if row["captured"]},
        )
        peak = max(row["peak_commanded_torque"] for row in subset)
        best = min(row["min_abs_energy_error"] for row in subset)
        header = f"limit={torque_limit:5.1f} damping={damping:4.2f}:"
        if captured:
            rows_at_best = [
                row
                for row in subset
                if row["k_p"] == captured[0] and row["captured"]
            ]
            median = float(
                np.median([row["capture_time"] for row in rows_at_best])
            )
            lines.append(
                f"  {header} captures from k_P = {captured[0]:.3f} "
                f"(T_cap median {median:.1f} s); "
                f"largest demand over the sweep {peak:.2f} N*m"
            )
        else:
            lines.append(
                f"  {header} never captures; largest demand {peak:.2f} N*m, "
                f"closest approach |Etil| = {best:.3f} of a "
                f"{energy_span:.2f} J span"
            )
    return lines


def write_rows(rows: Sequence[dict], path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in OUTPUT_FIELDS})


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sweep", choices=("single", "frontier"), default="single")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="20000:20012")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument(
        "--physics-dt",
        type=float,
        default=None,
        help="integration step; defaults to min(--dt, 0.01)",
    )
    parser.add_argument("--kv", type=float, default=DEFAULT_K_V)
    parser.add_argument("--kd", type=float, default=DEFAULT_K_D)
    parser.add_argument("--kp", type=float, default=61.2)
    parser.add_argument(
        "--torque-limit", type=float, default=DEFAULT_TORQUE_LIMIT
    )
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--t-max", type=float, default=120.0)
    parser.add_argument("--t-max-fast", type=float, default=120.0)
    parser.add_argument("--t-max-slow", type=float, default=900.0)
    parser.add_argument(
        "--kp-values",
        default=None,
        help="comma-separated k_P grid for the frontier sweep",
    )
    parser.add_argument(
        "--start",
        choices=("hanging", "paper", "uniform", "release"),
        default="paper",
        help=(
            "paper reproduces the 2007 initial condition exactly (0.17 rad off "
            "hanging) and is therefore deterministic, so seeds repeat one "
            "trajectory -- use hanging or uniform for the capture-rate and "
            "capture-time distributions, which need distinct initial conditions"
        ),
    )
    parser.add_argument("--tube-energy", type=float, default=0.05)
    parser.add_argument("--tube-angle", type=float, default=0.025)
    parser.add_argument("--tube-rate", type=float, default=0.05)
    parser.add_argument("--dwell", type=float, default=1.0)
    parser.add_argument(
        "--lqr-threshold", type=float, default=LQR_SWITCH_THRESHOLD
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    seeds = _parse_seeds(args.seeds)
    spec = TubeSpec(
        energy_tolerance=args.tube_energy,
        angle_tolerance=args.tube_angle,
        rate_tolerance=args.tube_rate,
        dwell_seconds=args.dwell,
    )
    if args.sweep == "single":
        arms = [
            Arm(
                k_v=args.kv,
                k_d=args.kd,
                k_p=args.kp,
                torque_limit=args.torque_limit,
                damping=args.damping,
                t_max=args.t_max,
                start=args.start,
            )
        ]
    else:
        kp_values = (
            None
            if args.kp_values is None
            else [float(v) for v in args.kp_values.split(",")]
        )
        arms = frontier_arms(
            k_v=args.kv,
            k_d=args.kd,
            t_max_fast=args.t_max_fast,
            t_max_slow=args.t_max_slow,
            start=args.start,
            kp_values=kp_values,
        )
    print(
        f"{len(arms)} arm(s) x {len(seeds)} seed(s), start={args.start}, "
        f"dt={args.dt}",
        flush=True,
    )
    rows: List[dict] = []
    for arm in arms:
        rows.extend(
            run_arm(
                arm,
                seeds,
                dt=args.dt,
                spec=spec,
                lqr_threshold=args.lqr_threshold,
                physics_dt=args.physics_dt,
            )
        )
    if args.sweep == "frontier":
        print("\nfrontier:")
        span = AcrobotParams.from_physics(
            build_env(arms[0], seeds[0], args.dt, args.physics_dt)._env.physics
        ).energy_span
        for line in summarize_frontier(rows, span):
            print(line)
    write_rows(rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
