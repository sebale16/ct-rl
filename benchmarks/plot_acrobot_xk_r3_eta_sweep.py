#!/usr/bin/env python
"""Sweep ``eta`` in Acrobot-XK ``r2``/``r3`` along analytical-controller episodes.

The analytical Xin--Kaneda controller does not depend on ``eta`` or on the
discount rate.  Consequently, this script rolls out the 32 fixed release
starts once, records ``r0``, the normalized Lyapunov terms, and the raw
state emitted by the environment, and evaluates two reward families offline,
matching ``environment/acrobot_xk.py:xk_reward_terms`` exactly for either
``--reward-base``:

    reward_base="lyapunov" (the environment default):
        r2(eta)         = -Vbar - eta * Vdotbar
        r3(eta, lambda) = -Vbar + eta * (lambda * Vbar - Vdotbar)

    reward_base="r0" (r1 substitutes r0, the normalized periodic distance,
    for -Vbar; the derivative and discount correction still use the
    original Xin--Kaneda V, unaffected by the substitution):
        r2(eta)         = r0 - eta * Vdotbar
        r3(eta, lambda) = r0 - eta * Vdotbar + lambda * eta * Vbar

``r3`` is swept at both CT-SAC discount horizons in the benchmark matrix
(lambda=0.5 /s, 2 s and lambda=0.1 /s, 10 s); ``r2`` has no discount-rate term
in the implementation, so it is swept once.

Every reward above is then reformulated as ``ln(1 / -reward)``, so that
"better" (less negative, closer to 0) reads as "larger" and the near-zero
behaviour that a symlog plot used to approximate gets an exact, monotone
stretch instead.  The transform is undefined once ``reward >= 0``, which
happens routinely at larger eta (the ``-eta * Vdotbar`` term can overshoot
past zero), so before transforming, reward is clipped from above to a
ceiling: the largest ``r1`` (``r0`` or ``-Vbar``) ever reaches anywhere
along the fixed, eta-independent rollout (``reward_ceiling``).  ``r1`` is
the pure state-proximity term the reward is built from, so its own maximum
is as close to the target as this trajectory ever gets -- the same sense in
which the eq. 74 LQR switching residual (``|x|_zeta < 0.04``,
``evaluations/acrobot_homoclinic_metrics.py``) is never satisfied on these
rollouts either (see ``compute_region_masks``).  The ceiling deliberately
excludes the ``-eta * Vdotbar`` shaping term: using the full eta-dependent
reward as the ceiling would make it collapse toward 0 as eta grows,
saturating almost every sample and making "sustained settling" spuriously
instant -- an artifact of the ceiling, not genuine convergence.

Since the trajectory never enters the LQR region, no eta should be able to
make the reward claim it did: a sample only counts as genuinely entering it
if raw reward clears the ceiling by more than ``VIOLATION_TOLERANCE``, which
filters the floating-point-scale (~1e-8-1e-6) same-sample knife edge every
eta > 0 trips exactly at the ceiling-defining instant, without also passing
substantial, growing overshoot elsewhere.  ``never_enters_lqr_level`` is
that per-eta boolean, and it is *not* satisfied by every eta or reward: past
a reward- and base-dependent point, "sustained settling" would otherwise
collapse toward 0 s purely because reward sits above the ceiling for nearly
the whole episode, not because the trajectory converges faster.  "Best eta"
is chosen only among eta with ``never_enters_lqr_level`` True; other eta are
still swept and plotted (hatched) for context.

"Converged" means the transformed reward remains at or above
``ln(1/--settling-tolerance)`` (equivalently: raw reward within
``--settling-tolerance`` of 0 from below, or clipped) for every subsequent
sample through the end of the episode.  This sustained-settling definition
rejects transient crossings.

For each eta, the script also splits *transformed* reward samples by two
state-space regions: the LQR switching set above and the looser homoclinic
tube (``TubeSpec`` defaults, a superset of the LQR set on the trajectories
observed here).  The eta chosen as "best" is, among the eligible eta above,
the fastest-settling one where the mean transformed reward inside the LQR
set (or its edge proxy) exceeds the mean over the whole tube -- which, given
the clipping above, holds by construction whenever the LQR set is never
entered.

Example
-------
    MUJOCO_GL=disable python -m benchmarks.plot_acrobot_xk_r3_eta_sweep
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

os.environ.setdefault("MUJOCO_GL", "disable")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from controllers.xin_kaneda import AcrobotParams, Gains, XinKanedaController
from environment.dmc import DMCContinuousEnv
from evaluations.acrobot_homoclinic_metrics import (
    LQR_SWITCH_THRESHOLD,
    Scales,
    TubeSpec,
    inside_tube,
    lqr_residual,
)


DEFAULT_DISCOUNT_RATES = (0.1, 0.5)
INK = "#1a202c"
MUTED = "#4a5568"
GRID = "#ffffff"
PANEL = "#eaeaf2"
ACCENT = "#b83280"
LQR_COLOR = "#2b6cb0"
TUBE_COLOR = "#805ad5"
GOOD_BAND = "#38a169"


REWARD_BASES = ("lyapunov", "r0")
DEFAULT_REWARD_BASE = "lyapunov"  # matches environment.acrobot_xk.DEFAULT_REWARD_BASE


@dataclass(frozen=True)
class RecordedTerms:
    """Endpoint reward terms and raw state shared by every eta and lambda."""

    time: np.ndarray
    r0: np.ndarray
    lyapunov_normalized: np.ndarray
    lyapunov_rate_normalized: np.ndarray
    state: np.ndarray  # (episodes, len(time), 4): [q1, q2, qdot1, qdot2], paper frame
    params: AcrobotParams


def r1_values(terms: RecordedTerms, reward_base: str) -> np.ndarray:
    """The r1 state term: ``r0`` or ``-Vbar``, per ``reward_base``."""
    if reward_base == "r0":
        return terms.r0
    return -terms.lyapunov_normalized


def r2_values(terms: RecordedTerms, eta: float, reward_base: str) -> np.ndarray:
    """Evaluate the implemented normalized ``r2`` (no discount-rate term)."""
    eta = float(eta)
    return r1_values(terms, reward_base) - eta * terms.lyapunov_rate_normalized


def r3_values(
    terms: RecordedTerms, eta: float, discount_rate: float, reward_base: str
) -> np.ndarray:
    """Evaluate the implemented normalized ``r3`` on recorded terms.

    The discount-rate correction always retains the original Lyapunov
    ``Vbar``, regardless of ``reward_base`` -- only r1's leading term
    switches (``environment/acrobot_xk.py:xk_reward_terms``).
    """
    eta = float(eta)
    discount_rate = float(discount_rate)
    return (
        r2_values(terms, eta, reward_base)
        + discount_rate * eta * terms.lyapunov_normalized
    )


def reward_values(
    kind: str,
    terms: RecordedTerms,
    eta: float,
    discount_rate: Optional[float],
    reward_base: str = DEFAULT_REWARD_BASE,
) -> np.ndarray:
    if kind == "r2":
        return r2_values(terms, eta, reward_base)
    if kind == "r3":
        if discount_rate is None:
            raise ValueError("r3 requires a discount_rate")
        return r3_values(terms, eta, discount_rate, reward_base)
    raise ValueError(f"unknown reward kind {kind!r}")


def _build_env(args: argparse.Namespace, seed: int) -> DMCContinuousEnv:
    return DMCContinuousEnv(
        "acrobot",
        "swingup-xk",
        seed=seed,
        raw_state_obs=True,
        time_sampling="uniform",
        dt=args.dt,
        physics_dt=args.dt,
        max_steps=int(round(args.duration / args.dt)),
        episode_duration=args.duration,
        task_kwargs={
            "release_start": True,
            "damping": 0.0,
            "torque_limit": args.torque_limit,
            # These values only make the task publish its r3 decomposition.
            # Vbar and Vdotbar themselves are independent of eta and lambda.
            "reward_kind": "r3",
            "eta": 0.0,
            "discount_rate": DEFAULT_DISCOUNT_RATES[0],
            "k_d": args.kd,
            "k_p": args.kp,
        },
    )


def collect_terms(args: argparse.Namespace) -> RecordedTerms:
    """Run the fixed analytical protocol once and collect ``Vbar,Vdotbar,state``."""
    gains = Gains(k_v=args.kv, k_d=args.kd, k_p=args.kp)
    all_time: list[np.ndarray] = []
    all_r0: list[np.ndarray] = []
    all_v: list[np.ndarray] = []
    all_vdot: list[np.ndarray] = []
    all_state: list[np.ndarray] = []
    params: Optional[AcrobotParams] = None

    for episode, seed in enumerate(range(args.seed0, args.seed0 + args.starts), 1):
        env = _build_env(args, seed)
        try:
            episode_params = AcrobotParams.from_physics(env._env.physics)
            if params is None:
                params = episode_params
            controller = XinKanedaController(episode_params, gains)
            obs, _ = env.reset(seed=seed)

            # Release starts are at rest, so Vdot(0)=0 for any applied torque.
            initial = env._env.task.xk_reward_terms(env._env.physics)
            times = [0.0]
            r0_values = [float(initial["r0"])]
            values = [float(initial["lyapunov_normalized"])]
            rates = [float(initial["lyapunov_rate_normalized"])]
            states = [np.asarray(obs, dtype=np.float64).reshape(4).copy()]
            termination_reason = None

            while True:
                result = env.step_dt(controller(obs))
                obs = result[4]
                next_t = float(result[5])
                terminated, truncated, info = result[6], result[7], result[8]
                if next_t <= times[-1]:
                    break
                times.append(next_t)
                r0_values.append(float(info["acrobot_xk_r0"]))
                values.append(float(info["acrobot_xk_lyapunov_normalized"]))
                rates.append(float(info["acrobot_xk_lyapunov_rate_normalized"]))
                states.append(np.asarray(obs, dtype=np.float64).reshape(4).copy())
                if terminated:
                    termination_reason = info.get(
                        "acrobot_xk_termination_reason", "unknown cap"
                    )
                    break
                if truncated:
                    break

            if termination_reason is not None:
                raise RuntimeError(
                    f"analytical episode seed {seed} hit {termination_reason} at "
                    f"t={times[-1]:.6f} s; the fixed protocol should fit inside "
                    "the state caps"
                )
            expected_steps = int(round(args.duration / args.dt))
            if len(times) != expected_steps + 1:
                raise RuntimeError(
                    f"seed {seed} produced {len(times) - 1} transitions; "
                    f"expected {expected_steps}"
                )

            all_time.append(np.asarray(times, dtype=np.float64))
            all_r0.append(np.asarray(r0_values, dtype=np.float64))
            all_v.append(np.asarray(values, dtype=np.float64))
            all_vdot.append(np.asarray(rates, dtype=np.float64))
            all_state.append(np.stack(states, axis=0))
            if controller.saturated_steps:
                raise RuntimeError(
                    f"analytical controller saturated on seed {seed}: "
                    f"{controller.saturated_steps}/{controller.steps} steps"
                )
            print(
                f"  [{episode:02d}/{args.starts}] seed {seed}: "
                f"Vbar(0)={values[0]:.4f}, Vbar(T)={values[-1]:.6f}",
                flush=True,
            )
        finally:
            env.close()

    assert params is not None
    reference_time = all_time[0]
    for seed, time in zip(range(args.seed0, args.seed0 + args.starts), all_time):
        if not np.array_equal(time, reference_time):
            raise RuntimeError(f"seed {seed} did not use the shared uniform time grid")

    r0 = np.stack(all_r0)
    values = np.stack(all_v)
    rates = np.stack(all_vdot)
    state = np.stack(all_state)
    if abs(float(rates[:, 0].max())) > 1e-12:
        raise RuntimeError("release-start Vdot(0) should be zero")
    return RecordedTerms(
        time=reference_time,
        r0=r0,
        lyapunov_normalized=values,
        lyapunov_rate_normalized=rates,
        state=state,
        params=params,
    )


@dataclass(frozen=True)
class RegionMasks:
    """Per-sample state-space region membership, shared across eta and kind."""

    lqr: np.ndarray  # (episodes, len(time)) bool: inside the LQR switching set
    tube: np.ndarray  # (episodes, len(time)) bool: inside the homoclinic tube
    edge_flat_index: int  # flat index of the sample closest to the LQR set,
    # i.e. the smallest eq. 74 residual recorded -- used as an edge proxy for
    # the LQR set on rollouts where it is never actually entered.


def compute_region_masks(
    terms: RecordedTerms, lqr_threshold: float, tube_spec: TubeSpec
) -> RegionMasks:
    """Classify every recorded sample by LQR-set / homoclinic-tube membership."""
    shape = terms.state.shape[:2]
    flat_state = terms.state.reshape(-1, 4)
    scales = Scales.from_params(terms.params)
    residual = lqr_residual(flat_state)
    lqr = residual < lqr_threshold
    tube = inside_tube(flat_state, terms.params, tube_spec, scales)
    return RegionMasks(
        lqr=lqr.reshape(shape),
        tube=tube.reshape(shape),
        edge_flat_index=int(np.argmin(residual)),
    )


def sustained_settling_times(
    time: np.ndarray, transformed: np.ndarray, threshold: float
) -> np.ndarray:
    """Earliest time after which the transformed reward never drops below
    ``threshold`` (i.e. raw reward never leaves the tolerance band from
    below, or is clipped) for the remainder of the episode."""
    time = np.asarray(time, dtype=np.float64)
    transformed = np.asarray(transformed, dtype=np.float64)
    if transformed.ndim != 2 or transformed.shape[1] != time.size:
        raise ValueError("transformed must have shape (episodes, len(time))")

    result = np.full(transformed.shape[0], np.inf, dtype=np.float64)
    for episode, curve in enumerate(transformed):
        violations = np.flatnonzero(curve < threshold)
        if violations.size == 0:
            result[episode] = float(time[0])
        elif violations[-1] < time.size - 1:
            result[episode] = float(time[violations[-1] + 1])
    return result


def _censored_quantile(
    settling: np.ndarray, quantile: float, duration: float, dt: float
) -> float:
    values = np.where(np.isfinite(settling), settling, duration + dt)
    return float(np.quantile(values, quantile))


CEILING_EPSILON = 1e-6  # keeps ln(1/-reward) finite even if r1's own max is
# ever >= 0 (a possibility in principle, not observed here).

# A raw reward sample counts as "entering the LQR region" only once it clears
# the ceiling by more than this.  The ceiling is realized exactly by some
# sample (r1's own global max), so *any* eta > 0 perturbs reward at that same
# sample by -eta*Vdotbar there; if Vdotbar happens to be negative at that one
# instant, any eta > 0 trips a same-sample violation by a floating-point-
# scale amount (~1e-8 to 1e-6 on these rollouts) that has nothing to do with
# genuine, growing overshoot elsewhere in the episode.  This tolerance
# separates that measure-zero knife edge from real exceedance.
VIOLATION_TOLERANCE = 1e-3


def reward_ceiling(terms: RecordedTerms, reward_base: str) -> float:
    """The largest ``r1`` ever reaches along the (eta-independent) rollout.

    ``r1`` (``r0`` or ``-Vbar``) is the pure state-proximity term the reward
    is built from; its own maximum is as close to the target as the fixed
    analytical trajectory ever gets, in the same sense the eq. 74 LQR
    switching residual is never satisfied on these rollouts (see
    ``compute_region_masks``) -- both describe "as close as this trajectory
    gets," using each metric's own weighting.  Reward is clipped to this
    ceiling before the ln(1/-reward) transform; it is also the "never enters
    the LQR region" reference: since the trajectory itself never gets there,
    no eta should be able to make the reward claim otherwise (see
    ``VIOLATION_TOLERANCE`` and ``never_enters_lqr_level`` in ``analyze``).
    It does not vary with eta or reward kind -- one value per reward base.
    """
    r1 = r1_values(terms, reward_base)
    return float(r1.max())


def transform_reward(
    reward: np.ndarray, ceiling: float, violation_tolerance: float = VIOLATION_TOLERANCE
) -> tuple[np.ndarray, float, bool]:
    """``ln(1/-reward)``, reward first clipped to ``ceiling``.

    Returns ``(transformed, exceedance_fraction, never_enters_lqr_level)``.
    ``exceedance_fraction`` is measured against the raw ceiling (samples
    that had to be clipped at all); ``never_enters_lqr_level`` additionally
    requires clearing ``violation_tolerance`` to ignore the same-sample
    knife-edge described above.
    """
    clip_value = min(ceiling, -CEILING_EPSILON)
    clipped = np.minimum(reward, clip_value)
    transformed = -np.log(-clipped)
    exceedance_fraction = float(np.mean(reward > clip_value))
    never_enters_lqr_level = bool(not np.any(reward > ceiling + violation_tolerance))
    return transformed, exceedance_fraction, never_enters_lqr_level


def analyze(
    terms: RecordedTerms,
    masks: RegionMasks,
    kind: str,
    etas: Iterable[float],
    discount_rate: Optional[float],
    tolerance: float,
    duration: float,
    dt: float,
    reward_base: str,
    ceiling: float,
    violation_tolerance: float = VIOLATION_TOLERANCE,
) -> list[dict[str, float]]:
    """Return one convergence-and-region-constraint summary row per eta."""
    rows: list[dict[str, float]] = []
    tail = terms.time >= max(0.0, duration - 2.0)
    settle_threshold = -np.log(tolerance)
    lqr_ceiling_transformed = float(-np.log(-min(ceiling, -CEILING_EPSILON)))
    for eta in etas:
        reward = reward_values(kind, terms, eta, discount_rate, reward_base)
        transformed, exceedance_fraction, never_enters = transform_reward(
            reward, ceiling, violation_tolerance
        )
        settling = sustained_settling_times(terms.time, transformed, settle_threshold)
        tube_values = transformed[masks.tube]
        tube_mean = float(np.mean(tube_values)) if tube_values.size else float("nan")
        lqr_values = transformed[masks.lqr]
        lqr_mean = (
            float(np.mean(lqr_values)) if lqr_values.size else lqr_ceiling_transformed
        )
        constraint_satisfied = (
            np.isfinite(lqr_mean) and np.isfinite(tube_mean) and lqr_mean > tube_mean
        )
        rows.append(
            {
                "kind": kind,
                "reward_base": reward_base,
                "discount_rate": float("nan") if discount_rate is None else float(discount_rate),
                "discount_horizon": (
                    float("nan") if discount_rate is None else 1.0 / float(discount_rate)
                ),
                "eta": float(eta),
                "settling_tolerance": float(tolerance),
                "settle_threshold": float(settle_threshold),
                "settled_fraction": float(np.mean(np.isfinite(settling))),
                "settling_p10": _censored_quantile(settling, 0.10, duration, dt),
                "settling_p50": _censored_quantile(settling, 0.50, duration, dt),
                "settling_p90": _censored_quantile(settling, 0.90, duration, dt),
                "exceedance_fraction": exceedance_fraction,
                "never_enters_lqr_level": never_enters,
                "mean_reward": float(
                    np.mean(np.trapezoid(transformed, terms.time, axis=1) / duration)
                ),
                "tail_reward": float(np.mean(transformed[:, tail])),
                "ceiling_raw": ceiling,
                "lqr_mean_reward": lqr_mean,
                "tube_mean_reward": tube_mean,
                "lqr_reward_exceeds_tube": bool(constraint_satisfied),
            }
        )
    return rows


def _best_row(rows: list[dict[str, float]]) -> dict[str, float]:
    """Fastest 90th-percentile settling among etas that never enter the LQR
    region (``never_enters_lqr_level``, beyond the floating-point-scale
    knife edge at the ceiling-defining sample -- see ``VIOLATION_TOLERANCE``)
    and whose LQR-set transformed reward exceeds the tube's; falls back to
    the unconstrained fastest eta over all rows if none qualify."""
    candidates = [row for row in rows if row["never_enters_lqr_level"]]
    pool = candidates if candidates else rows
    return min(
        pool,
        key=lambda row: (
            0 if row["lqr_reward_exceeds_tube"] else 1,
            row["settling_p90"],
            row["settling_p50"],
            -row["tail_reward"],
            row["eta"],
        ),
    )


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Index ranges (inclusive) of contiguous True runs in a 1-D bool array."""
    runs: list[tuple[int, int]] = []
    start: Optional[int] = None
    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _set_style() -> None:
    matplotlib.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": PANEL,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 1.0,
            "axes.edgecolor": "#cccccc",
            "axes.linewidth": 1.0,
            "axes.axisbelow": True,
            "font.size": 10.5,
        }
    )


def _base_tag(reward_base: str) -> str:
    return "r0 base" if reward_base == "r0" else r"$-\bar V$ base"


def _base_tag_plain(reward_base: str) -> str:
    return "r0 base" if reward_base == "r0" else "-Vbar base"


def _row_label(kind: str, discount_rate: Optional[float], reward_base: str) -> str:
    base = _base_tag(reward_base)
    if kind == "r2":
        return rf"$r_2$ ($\eta$-shaping only, no discount rate; {base})"
    return (
        rf"$r_3$: $\lambda={discount_rate:g}\,\mathrm{{s}}^{{-1}}$ "
        rf"($1/\lambda={1.0 / discount_rate:g}$ s; {base})"
    )


def _row_label_plain(kind: str, discount_rate: Optional[float], reward_base: str) -> str:
    base = _base_tag_plain(reward_base)
    if kind == "r2":
        return f"r2 (eta-shaping only, no discount rate; {base})"
    return (
        f"r3: lambda={discount_rate:g} /s (1/lambda={1.0 / discount_rate:g} s; {base})"
    )


def draw(
    terms: RecordedTerms,
    masks: RegionMasks,
    ceiling: float,
    panels: list[tuple[str, Optional[float], list[dict[str, float]]]],
    args: argparse.Namespace,
) -> None:
    _set_style()
    figure, axes = plt.subplots(
        len(panels),
        3,
        figsize=(18.0, 3.1 * len(panels) + 1.6),
        sharex="col",
        squeeze=False,
        gridspec_kw={"width_ratios": (1.35, 1.0, 1.0)},
    )
    cmap = matplotlib.colormaps["viridis"]
    norm = Normalize(vmin=0.0, vmax=args.eta_max)
    plot_stride = max(1, int(round(0.01 / args.dt)))
    plot_slice = slice(None, None, plot_stride)
    display_etas_base = np.linspace(0.0, args.eta_max, 11)
    settle_threshold = -np.log(args.settling_tolerance)

    for row_index, (kind, discount_rate, rows) in enumerate(panels):
        trajectory_ax, convergence_ax, region_ax = axes[row_index]
        best = _best_row(rows)
        best_eta = best["eta"]
        display_etas = np.unique(np.append(display_etas_base, best_eta))

        for eta in display_etas:
            reward = reward_values(kind, terms, eta, discount_rate, args.reward_base)
            transformed, _, _ = transform_reward(reward, ceiling, args.violation_tolerance)
            median = np.median(transformed, axis=0)
            is_best = np.isclose(eta, best_eta, atol=0.5 * args.eta_step)
            trajectory_ax.plot(
                terms.time[plot_slice],
                median[plot_slice],
                color=cmap(norm(float(eta))),
                lw=2.4 if is_best else 1.05,
                alpha=1.0 if is_best else 0.78,
                zorder=4 if is_best else 2,
            )
            if is_best:
                low, high = np.quantile(transformed, (0.10, 0.90), axis=0)
                trajectory_ax.fill_between(
                    terms.time[plot_slice],
                    low[plot_slice],
                    high[plot_slice],
                    color=cmap(norm(float(eta))),
                    alpha=0.13,
                    lw=0.0,
                    zorder=1,
                )

        trajectory_ax.axhline(
            settle_threshold,
            color=INK,
            lw=0.9,
            alpha=0.65,
            ls="--",
        )
        trajectory_ax.set_xlim(0.0, args.duration)
        trajectory_ax.set_ylabel(r"median $\ln(1/-\mathrm{reward})$")
        trajectory_ax.set_title(
            f"Reward through the episodes: {_row_label(kind, discount_rate, args.reward_base)}",
            color=INK,
            fontsize=11.0,
        )
        satisfied_tag = (
            "constraint met" if best["lqr_reward_exceeds_tube"] else "constraint UNMET"
        )
        trajectory_ax.text(
            0.985,
            0.05,
            rf"thick curve / band: best $\eta={best_eta:.2f}$ ({satisfied_tag}, 10--90%);"
            rf" dashed: settle threshold $\ln(1/{args.settling_tolerance:g})$",
            transform=trajectory_ax.transAxes,
            ha="right",
            va="bottom",
            color=MUTED,
            fontsize=8.6,
        )

        eta_values = np.asarray([item["eta"] for item in rows])
        p10 = np.asarray([item["settling_p10"] for item in rows])
        p50 = np.asarray([item["settling_p50"] for item in rows])
        p90 = np.asarray([item["settling_p90"] for item in rows])
        exceedance = 100.0 * np.asarray(
            [item["exceedance_fraction"] for item in rows]
        )
        satisfied_mask = np.asarray(
            [item["lqr_reward_exceeds_tube"] for item in rows], dtype=bool
        )
        excluded_mask = ~np.asarray(
            [item["never_enters_lqr_level"] for item in rows], dtype=bool
        )
        half_step = 0.5 * args.eta_step
        for start, end in _true_runs(satisfied_mask):
            convergence_ax.axvspan(
                max(0.0, eta_values[start] - half_step),
                min(args.eta_max, eta_values[end] + half_step),
                color=GOOD_BAND,
                alpha=0.09,
                lw=0.0,
                zorder=0,
            )
        for start, end in _true_runs(excluded_mask):
            convergence_ax.axvspan(
                max(0.0, eta_values[start] - half_step),
                min(args.eta_max, eta_values[end] + half_step),
                facecolor="none",
                edgecolor=MUTED,
                hatch="////",
                alpha=0.35,
                lw=0.0,
                zorder=0,
            )
        convergence_ax.fill_between(
            eta_values, p10, p90, color=ACCENT, alpha=0.16, lw=0.0
        )
        convergence_ax.plot(
            eta_values, p50, color=ACCENT, lw=2.0, label="median settling"
        )
        convergence_ax.plot(
            eta_values,
            p90,
            color="#702459",
            lw=1.35,
            ls="--",
            label="90th percentile",
        )
        convergence_ax.scatter(
            [best_eta],
            [best["settling_p90"]],
            color=ACCENT,
            edgecolor="white",
            linewidth=0.8,
            s=52,
            zorder=5,
        )
        convergence_ax.set_xlim(0.0, args.eta_max)
        finite_top = float(np.max(p90))
        convergence_ax.set_ylim(
            0.0,
            args.duration
            if finite_top > args.duration
            else max(1.0, 1.15 * finite_top),
        )
        convergence_ax.set_ylabel("sustained settling time (s)", color=ACCENT)
        convergence_ax.tick_params(axis="y", colors=ACCENT)
        convergence_ax.set_title(
            rf"Sustained $\ln(1/-\mathrm{{reward}})\geq\ln(1/{args.settling_tolerance:g})$ "
            "(median, 10--90%; green = constraint met;\n"
            r"hatched = enters the LQR region ($>$" + f"{args.violation_tolerance:g}"
            r" past ceiling), excluded from ranking)",
            color=INK,
            fontsize=10.0,
        )
        convergence_ax.text(
            0.02,
            0.04,
            rf"fastest, constrained: $\eta={best_eta:.2f}$, "
            rf"$T_{{90}}={best['settling_p90']:.3f}$ s",
            transform=convergence_ax.transAxes,
            ha="left",
            va="bottom",
            color=ACCENT,
            fontsize=8.4,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
        )
        convergence_ax.legend(loc="upper right", frameon=False, fontsize=8)

        exceedance_ax = convergence_ax.twinx()
        exceedance_ax.plot(
            eta_values,
            exceedance,
            color=MUTED,
            lw=1.15,
            ls="--",
            alpha=0.8,
        )
        exceedance_ax.set_ylim(bottom=0.0)
        exceedance_ax.set_ylabel(
            "samples clipped (%)\n(reward $\\geq$ ceiling)",
            color=MUTED,
            labelpad=10,
        )
        exceedance_ax.tick_params(axis="y", colors=MUTED)
        exceedance_ax.grid(False)

        lqr_means = np.asarray([item["lqr_mean_reward"] for item in rows])
        tube_means = np.asarray([item["tube_mean_reward"] for item in rows])
        for start, end in _true_runs(satisfied_mask):
            region_ax.axvspan(
                max(0.0, eta_values[start] - half_step),
                min(args.eta_max, eta_values[end] + half_step),
                color=GOOD_BAND,
                alpha=0.09,
                lw=0.0,
                zorder=0,
            )
        for start, end in _true_runs(excluded_mask):
            region_ax.axvspan(
                max(0.0, eta_values[start] - half_step),
                min(args.eta_max, eta_values[end] + half_step),
                facecolor="none",
                edgecolor=MUTED,
                hatch="////",
                alpha=0.35,
                lw=0.0,
                zorder=0,
            )
        edge_label = (
            "mean, LQR edge proxy (set never entered)"
            if not masks.lqr.any()
            else "mean, LQR set"
        )
        region_ax.plot(
            eta_values, lqr_means, color=LQR_COLOR, lw=2.0, label=edge_label
        )
        region_ax.plot(
            eta_values,
            tube_means,
            color=TUBE_COLOR,
            lw=1.6,
            ls="--",
            label="mean, tube (incl. LQR)",
        )
        region_ax.axvline(best_eta, color=ACCENT, lw=1.1, ls=":", alpha=0.85)
        region_ax.set_xlim(0.0, args.eta_max)
        region_ax.set_ylabel(r"mean $\ln(1/-\mathrm{reward})$", labelpad=8)
        region_ax.set_title(
            r"LQR-edge ceiling vs tube mean $\ln(1/-\mathrm{reward})$ ($\zeta="
            + f"{args.lqr_threshold:g}$)",
            color=INK,
            fontsize=10.6,
        )
        if not masks.lqr.any():
            region_ax.text(
                0.5,
                0.06,
                "LQR set never entered; the LQR curve is the max r1 (state\n"
                "proximity alone) ever reaches, used as the clipping ceiling",
                transform=region_ax.transAxes,
                ha="center",
                va="bottom",
                color=LQR_COLOR,
                fontsize=7.8,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": LQR_COLOR, "pad": 2.5},
            )
        region_ax.legend(loc="upper left", frameon=False, fontsize=8)

    axes[-1, 0].set_xlabel("episode time (s)")
    axes[-1, 1].set_xlabel(r"shaping time scale $\eta$ (s)")
    axes[-1, 2].set_xlabel(r"shaping time scale $\eta$ (s)")
    figure.suptitle(
        f"Acrobot r2/r3 eta sweep under the Xin–Kaneda analytical controller "
        f"({_base_tag_plain(args.reward_base)}, reward transformed by "
        r"$\ln(1/-\mathrm{reward})$" + ")\n"
        f"{args.starts} fixed release seeds, {args.duration:g} s, "
        f"{1e3 * args.dt:g} ms control/physics step, "
        f"{args.torque_limit:g} N·m gear; reward is clipped to the LQR-edge "
        "ceiling before the log; green band = LQR-edge exceeds tube mean",
        color=INK,
        fontsize=12.6,
        y=0.998,
    )
    figure.subplots_adjust(
        left=0.055, right=0.955, top=0.90, bottom=0.115, hspace=0.42, wspace=0.5
    )
    scalar = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar_ax = figure.add_axes((0.06, 0.035, 0.40, 0.016))
    colorbar = figure.colorbar(scalar, cax=colorbar_ax, orientation="horizontal")
    colorbar.set_label(r"displayed $\eta$ (s)")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi)
    plt.close(figure)
    print(f"wrote {output}")


def write_summary(
    panels: list[tuple[str, Optional[float], list[dict[str, float]]]],
    args: argparse.Namespace,
) -> Path:
    output = (
        Path(args.summary_output)
        if args.summary_output
        else Path(args.output).with_suffix(".csv")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "kind",
        "reward_base",
        "discount_rate",
        "discount_horizon",
        "eta",
        "settling_tolerance",
        "settle_threshold",
        "settled_fraction",
        "settling_p10",
        "settling_p50",
        "settling_p90",
        "exceedance_fraction",
        "never_enters_lqr_level",
        "mean_reward",
        "tail_reward",
        "ceiling_raw",
        "lqr_mean_reward",
        "tube_mean_reward",
        "lqr_reward_exceeds_tube",
    )
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for _, _, rows in panels:
            writer.writerows(rows)
    print(f"wrote {output}")
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--output", default="results/acrobot_xk_r3_eta_sweep.png"
    )
    parser.add_argument("--summary-output")
    parser.add_argument("--starts", type=int, default=32)
    parser.add_argument("--seed0", type=int, default=20000)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--torque-limit", type=float, default=20.0)
    parser.add_argument("--kv", type=float, default=66.3)
    parser.add_argument("--kd", type=float, default=35.8)
    parser.add_argument("--kp", type=float, default=61.2)
    parser.add_argument("--eta-step", type=float, default=0.01)
    parser.add_argument(
        "--eta-max",
        type=float,
        default=2.0,
        help="sweep eta over [0, eta-max]; widen if the fastest-settling "
        "eligible eta (never_enters_lqr_level) lands on the eta=eta-max "
        "boundary",
    )
    parser.add_argument("--settling-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--violation-tolerance",
        type=float,
        default=VIOLATION_TOLERANCE,
        help="how far raw reward may clear the r1 ceiling before an eta "
        "counts as entering the LQR region and is excluded from 'best eta' "
        "ranking -- filters the floating-point-scale same-sample knife edge "
        "every eta > 0 trips exactly at the ceiling-defining instant",
    )
    parser.add_argument(
        "--lqr-threshold",
        type=float,
        default=LQR_SWITCH_THRESHOLD,
        help="eq. 74 (2007) switching-set residual threshold zeta",
    )
    parser.add_argument(
        "--reward-base",
        choices=REWARD_BASES,
        default=DEFAULT_REWARD_BASE,
        help=(
            "r1's leading state term: '-Vbar' (environment default) or "
            "'r0', the normalized periodic distance (see "
            "environment/acrobot_xk.py:xk_reward_terms)"
        ),
    )
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)
    for name in (
        "duration",
        "dt",
        "torque_limit",
        "eta_step",
        "eta_max",
        "settling_tolerance",
        "lqr_threshold",
        "violation_tolerance",
    ):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and > 0")
    if args.starts <= 0:
        parser.error("--starts must be > 0")
    if args.eta_step > args.eta_max:
        parser.error("--eta-step must be <= --eta-max")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    terms = collect_terms(args)
    masks = compute_region_masks(terms, args.lqr_threshold, TubeSpec())
    ceiling = reward_ceiling(terms, args.reward_base)
    print(
        f"reward ceiling ({args.reward_base} base): max(r1)={ceiling:.6g}, "
        f"ln(1/-ceiling)={-np.log(-min(ceiling, -CEILING_EPSILON)):.4f}; "
        f"violation tolerance for 'never enters LQR region': "
        f"{args.violation_tolerance:.4g}"
    )
    etas = np.linspace(
        0.0, args.eta_max, int(round(args.eta_max / args.eta_step)) + 1
    )

    panel_specs: list[tuple[str, Optional[float]]] = [
        ("r3", DEFAULT_DISCOUNT_RATES[1]),  # lambda=0.5 /s, 2 s horizon
        ("r3", DEFAULT_DISCOUNT_RATES[0]),  # lambda=0.1 /s, 10 s horizon
        ("r2", None),  # eta-only shaping, no discount rate
    ]
    panels: list[tuple[str, Optional[float], list[dict[str, float]]]] = [
        (
            kind,
            discount_rate,
            analyze(
                terms,
                masks,
                kind,
                etas,
                discount_rate,
                args.settling_tolerance,
                args.duration,
                args.dt,
                args.reward_base,
                ceiling,
                args.violation_tolerance,
            ),
        )
        for kind, discount_rate in panel_specs
    ]

    write_summary(panels, args)
    draw(terms, masks, ceiling, panels, args)

    for kind, discount_rate, rows in panels:
        best = _best_row(rows)
        label = _row_label_plain(kind, discount_rate, args.reward_base)
        tag = "constraint met" if best["lqr_reward_exceeds_tube"] else "constraint UNMET"
        never_enters_tag = (
            "never enters LQR region"
            if best["never_enters_lqr_level"]
            else "WARNING: enters LQR region (no eligible eta found)"
        )
        print(
            f"{label}: fastest 90th-percentile sustained settling "
            f"eta={best['eta']:.2f} ({tag}; {never_enters_tag}), "
            f"T50={best['settling_p50']:.3f} s, T90={best['settling_p90']:.3f} s, "
            f"exceedance={100.0 * best['exceedance_fraction']:.2f}%, "
            f"mean ln(1/-reward) LQR-edge={best['lqr_mean_reward']:.4f} vs "
            f"tube={best['tube_mean_reward']:.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
