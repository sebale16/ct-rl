#!/usr/bin/env python
"""Sweep ``eta`` in Acrobot-XK ``r2``/``r3`` along analytical-controller episodes.

The analytical Xin--Kaneda controller does not depend on ``eta`` or on the
discount rate.  Consequently, this script rolls out the 32 fixed release
starts once, records the normalized Lyapunov terms and the raw state emitted
by the environment, and evaluates two reward families offline

    r2(eta)         = -Vbar - eta * Vdotbar
    r3(eta, lambda) = -Vbar + eta * (lambda * Vbar - Vdotbar)

``r3`` is swept at both CT-SAC discount horizons in the benchmark matrix
(lambda=0.5 /s, 2 s and lambda=0.1 /s, 10 s); ``r2`` has no discount-rate term
in the implementation, so it is swept once.

"Converged to zero" means that the absolute reward rate remains within
``--settling-tolerance`` for every subsequent sample through the end of the
episode.  This sustained-settling definition rejects transient zero crossings.
The positive-reward fraction is reported alongside it because increasing eta
can make the two terms cancel or make the reward positive without changing
the state trajectory.

For each eta, the script also splits reward samples by two state-space
regions defined in ``evaluations/acrobot_homoclinic_metrics.py``: the tight
LQR switching set of eq. 74 (``|x|_zeta < 0.04``) and the looser homoclinic
tube (``TubeSpec`` defaults, a superset of the LQR set on the trajectories
observed here).  The eta chosen as "best" is the fastest-settling eta among
those where the mean reward inside the LQR set exceeds the mean reward over
the whole tube -- i.e. reward keeps improving as the state tightens toward
the equilibrium, rather than plateauing or reversing before the LQR set is
reached.

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
DEFAULT_DISPLAY_ETAS = np.linspace(0.0, 1.0, 11)
INK = "#1a202c"
MUTED = "#4a5568"
GRID = "#ffffff"
PANEL = "#eaeaf2"
ACCENT = "#b83280"
LQR_COLOR = "#2b6cb0"
TUBE_COLOR = "#805ad5"
GOOD_BAND = "#38a169"


@dataclass(frozen=True)
class RecordedTerms:
    """Endpoint reward terms and raw state shared by every eta and lambda."""

    time: np.ndarray
    lyapunov_normalized: np.ndarray
    lyapunov_rate_normalized: np.ndarray
    state: np.ndarray  # (episodes, len(time), 4): [q1, q2, qdot1, qdot2], paper frame
    params: AcrobotParams


def r2_values(
    lyapunov_normalized: np.ndarray,
    lyapunov_rate_normalized: np.ndarray,
    eta: float,
) -> np.ndarray:
    """Evaluate the implemented normalized ``r2`` (no discount-rate term)."""
    eta = float(eta)
    return -lyapunov_normalized - eta * lyapunov_rate_normalized


def r3_values(
    lyapunov_normalized: np.ndarray,
    lyapunov_rate_normalized: np.ndarray,
    eta: float,
    discount_rate: float,
) -> np.ndarray:
    """Evaluate the implemented normalized ``r3`` on recorded terms."""
    eta = float(eta)
    discount_rate = float(discount_rate)
    return (
        -(1.0 - discount_rate * eta) * lyapunov_normalized
        - eta * lyapunov_rate_normalized
    )


def reward_values(
    kind: str,
    terms: RecordedTerms,
    eta: float,
    discount_rate: Optional[float],
) -> np.ndarray:
    if kind == "r2":
        return r2_values(terms.lyapunov_normalized, terms.lyapunov_rate_normalized, eta)
    if kind == "r3":
        if discount_rate is None:
            raise ValueError("r3 requires a discount_rate")
        return r3_values(
            terms.lyapunov_normalized,
            terms.lyapunov_rate_normalized,
            eta,
            discount_rate,
        )
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

    values = np.stack(all_v)
    rates = np.stack(all_vdot)
    state = np.stack(all_state)
    if abs(float(rates[:, 0].max())) > 1e-12:
        raise RuntimeError("release-start Vdot(0) should be zero")
    return RecordedTerms(
        time=reference_time,
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


def compute_region_masks(
    terms: RecordedTerms, lqr_threshold: float, tube_spec: TubeSpec
) -> RegionMasks:
    """Classify every recorded sample by LQR-set / homoclinic-tube membership."""
    shape = terms.state.shape[:2]
    flat_state = terms.state.reshape(-1, 4)
    scales = Scales.from_params(terms.params)
    lqr = lqr_residual(flat_state) < lqr_threshold
    tube = inside_tube(flat_state, terms.params, tube_spec, scales)
    return RegionMasks(lqr=lqr.reshape(shape), tube=tube.reshape(shape))


def sustained_settling_times(
    time: np.ndarray, rewards: np.ndarray, tolerance: float
) -> np.ndarray:
    """Earliest time after which ``|reward|`` never leaves the tolerance band."""
    time = np.asarray(time, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 2 or rewards.shape[1] != time.size:
        raise ValueError("rewards must have shape (episodes, len(time))")

    result = np.full(rewards.shape[0], np.inf, dtype=np.float64)
    for episode, curve in enumerate(rewards):
        violations = np.flatnonzero(np.abs(curve) > tolerance)
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


def _region_means(reward: np.ndarray, masks: RegionMasks) -> tuple[float, float]:
    """Mean reward inside the LQR set and inside the whole tube (superset)."""
    lqr_values = reward[masks.lqr]
    tube_values = reward[masks.tube]
    lqr_mean = float(np.mean(lqr_values)) if lqr_values.size else float("nan")
    tube_mean = float(np.mean(tube_values)) if tube_values.size else float("nan")
    return lqr_mean, tube_mean


def analyze(
    terms: RecordedTerms,
    masks: RegionMasks,
    kind: str,
    etas: Iterable[float],
    discount_rate: Optional[float],
    tolerance: float,
    duration: float,
    dt: float,
) -> list[dict[str, float]]:
    """Return one convergence-and-region-constraint summary row per eta."""
    rows: list[dict[str, float]] = []
    tail = terms.time >= max(0.0, duration - 2.0)
    for eta in etas:
        reward = reward_values(kind, terms, eta, discount_rate)
        settling = sustained_settling_times(terms.time, reward, tolerance)
        lqr_mean, tube_mean = _region_means(reward, masks)
        constraint_satisfied = (
            np.isfinite(lqr_mean) and np.isfinite(tube_mean) and lqr_mean > tube_mean
        )
        rows.append(
            {
                "kind": kind,
                "discount_rate": float("nan") if discount_rate is None else float(discount_rate),
                "discount_horizon": (
                    float("nan") if discount_rate is None else 1.0 / float(discount_rate)
                ),
                "eta": float(eta),
                "settling_tolerance": float(tolerance),
                "settled_fraction": float(np.mean(np.isfinite(settling))),
                "settling_p10": _censored_quantile(settling, 0.10, duration, dt),
                "settling_p50": _censored_quantile(settling, 0.50, duration, dt),
                "settling_p90": _censored_quantile(settling, 0.90, duration, dt),
                "positive_sample_fraction": float(np.mean(reward > 0.0)),
                "mean_absolute_reward": float(
                    np.mean(np.trapezoid(np.abs(reward), terms.time, axis=1) / duration)
                ),
                "tail_absolute_reward": float(np.mean(np.abs(reward[:, tail]))),
                "lqr_mean_reward": lqr_mean,
                "tube_mean_reward": tube_mean,
                "lqr_reward_exceeds_tube": bool(constraint_satisfied),
            }
        )
    return rows


def _best_row(rows: list[dict[str, float]]) -> dict[str, float]:
    """Fastest 90th-percentile settling among etas with LQR-set reward > tube
    reward; falls back to the unconstrained fastest eta if none qualify."""
    return min(
        rows,
        key=lambda row: (
            0 if row["lqr_reward_exceeds_tube"] else 1,
            row["settling_p90"],
            row["settling_p50"],
            row["tail_absolute_reward"],
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


def _row_label(kind: str, discount_rate: Optional[float]) -> str:
    if kind == "r2":
        return r"$r_2$ ($\eta$-shaping only, no discount rate)"
    return (
        rf"$r_3$: $\lambda={discount_rate:g}\,\mathrm{{s}}^{{-1}}$ "
        rf"($1/\lambda={1.0 / discount_rate:g}$ s)"
    )


def _row_label_plain(kind: str, discount_rate: Optional[float]) -> str:
    if kind == "r2":
        return "r2 (eta-shaping only, no discount rate)"
    return f"r3: lambda={discount_rate:g} /s (1/lambda={1.0 / discount_rate:g} s)"


def draw(
    terms: RecordedTerms,
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
    norm = Normalize(vmin=0.0, vmax=1.0)
    plot_stride = max(1, int(round(0.01 / args.dt)))
    plot_slice = slice(None, None, plot_stride)

    for row_index, (kind, discount_rate, rows) in enumerate(panels):
        trajectory_ax, convergence_ax, region_ax = axes[row_index]
        best = _best_row(rows)
        best_eta = best["eta"]
        display_etas = np.unique(np.append(DEFAULT_DISPLAY_ETAS, best_eta))

        for eta in display_etas:
            reward = reward_values(kind, terms, eta, discount_rate)
            median = np.median(reward, axis=0)
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
                low, high = np.quantile(reward, (0.10, 0.90), axis=0)
                trajectory_ax.fill_between(
                    terms.time[plot_slice],
                    low[plot_slice],
                    high[plot_slice],
                    color=cmap(norm(float(eta))),
                    alpha=0.13,
                    lw=0.0,
                    zorder=1,
                )

        trajectory_ax.axhline(0.0, color=INK, lw=0.9, alpha=0.65)
        trajectory_ax.axhspan(
            -args.settling_tolerance,
            args.settling_tolerance,
            color="white",
            alpha=0.45,
            zorder=0,
        )
        trajectory_ax.set_yscale(
            "symlog", linthresh=args.settling_tolerance, linscale=0.8
        )
        trajectory_ax.set_xlim(0.0, args.duration)
        trajectory_ax.set_ylabel("median reward rate")
        trajectory_ax.set_title(
            f"Reward through the episodes: {_row_label(kind, discount_rate)}",
            color=INK,
            fontsize=11.0,
        )
        satisfied_tag = (
            "constraint met" if best["lqr_reward_exceeds_tube"] else "constraint UNMET"
        )
        trajectory_ax.text(
            0.985,
            0.05,
            rf"thick curve / band: best $\eta={best_eta:.2f}$ ({satisfied_tag}, 10--90%)",
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
        positive = 100.0 * np.asarray(
            [item["positive_sample_fraction"] for item in rows]
        )
        satisfied_mask = np.asarray(
            [item["lqr_reward_exceeds_tube"] for item in rows], dtype=bool
        )
        half_step = 0.5 * args.eta_step
        for start, end in _true_runs(satisfied_mask):
            convergence_ax.axvspan(
                max(0.0, eta_values[start] - half_step),
                min(1.0, eta_values[end] + half_step),
                color=GOOD_BAND,
                alpha=0.09,
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
        convergence_ax.set_xlim(0.0, 1.0)
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
            rf"Stay within $|$reward$|\leq {args.settling_tolerance:g}$ "
            "(median, 10--90%; green = constraint met)",
            color=INK,
            fontsize=10.6,
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

        positive_ax = convergence_ax.twinx()
        positive_ax.plot(
            eta_values,
            positive,
            color=MUTED,
            lw=1.15,
            ls="--",
            alpha=0.8,
        )
        positive_ax.set_ylim(bottom=0.0)
        positive_ax.set_ylabel(
            r"samples with reward$>0$ (%)", color=MUTED, labelpad=10
        )
        positive_ax.tick_params(axis="y", colors=MUTED)
        positive_ax.grid(False)

        lqr_means = np.asarray([item["lqr_mean_reward"] for item in rows])
        tube_means = np.asarray([item["tube_mean_reward"] for item in rows])
        for start, end in _true_runs(satisfied_mask):
            region_ax.axvspan(
                max(0.0, eta_values[start] - half_step),
                min(1.0, eta_values[end] + half_step),
                color=GOOD_BAND,
                alpha=0.09,
                lw=0.0,
                zorder=0,
            )
        region_ax.plot(
            eta_values, lqr_means, color=LQR_COLOR, lw=2.0, label="mean reward, LQR set"
        )
        region_ax.plot(
            eta_values,
            tube_means,
            color=TUBE_COLOR,
            lw=1.6,
            ls="--",
            label="mean reward, tube (incl. LQR)",
        )
        region_ax.axvline(best_eta, color=ACCENT, lw=1.1, ls=":", alpha=0.85)
        region_ax.set_xlim(0.0, 1.0)
        region_ax.set_ylabel("mean reward", labelpad=8)
        region_ax.set_title(
            r"LQR-set vs tube mean reward ($\zeta="
            + f"{args.lqr_threshold:g}$)",
            color=INK,
            fontsize=10.6,
        )
        if np.all(np.isnan(lqr_means)):
            region_ax.text(
                0.5,
                0.5,
                "LQR set never entered in this rollout\n(min eq. 74 residual stays "
                "above $\\zeta$ for every episode) --\nconstraint is undefined at "
                "every $\\eta$, not just unmet",
                transform=region_ax.transAxes,
                ha="center",
                va="center",
                color=LQR_COLOR,
                fontsize=8.6,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": LQR_COLOR, "pad": 3},
            )
        region_ax.legend(loc="lower right", frameon=False, fontsize=8)

    axes[-1, 0].set_xlabel("episode time (s)")
    axes[-1, 1].set_xlabel(r"shaping time scale $\eta$ (s)")
    axes[-1, 2].set_xlabel(r"shaping time scale $\eta$ (s)")
    figure.suptitle(
        "Acrobot r2/r3 eta sweep under the Xin–Kaneda analytical controller\n"
        f"{args.starts} fixed release seeds, {args.duration:g} s, "
        f"{1e3 * args.dt:g} ms control/physics step, "
        f"{args.torque_limit:g} N·m gear; signed panels use a symmetric-log scale; "
        "green band = LQR-set mean reward exceeds tube mean reward",
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
        "discount_rate",
        "discount_horizon",
        "eta",
        "settling_tolerance",
        "settled_fraction",
        "settling_p10",
        "settling_p50",
        "settling_p90",
        "positive_sample_fraction",
        "mean_absolute_reward",
        "tail_absolute_reward",
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
    parser.add_argument("--settling-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--lqr-threshold",
        type=float,
        default=LQR_SWITCH_THRESHOLD,
        help="eq. 74 (2007) switching-set residual threshold zeta",
    )
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)
    for name in (
        "duration",
        "dt",
        "torque_limit",
        "eta_step",
        "settling_tolerance",
        "lqr_threshold",
    ):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and > 0")
    if args.starts <= 0:
        parser.error("--starts must be > 0")
    if args.eta_step > 1.0:
        parser.error("--eta-step must be <= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    terms = collect_terms(args)
    masks = compute_region_masks(terms, args.lqr_threshold, TubeSpec())
    etas = np.linspace(0.0, 1.0, int(round(1.0 / args.eta_step)) + 1)

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
            ),
        )
        for kind, discount_rate in panel_specs
    ]

    write_summary(panels, args)
    draw(terms, panels, args)

    for kind, discount_rate, rows in panels:
        best = _best_row(rows)
        label = _row_label_plain(kind, discount_rate)
        tag = "constraint met" if best["lqr_reward_exceeds_tube"] else "constraint UNMET"
        print(
            f"{label}: fastest 90th-percentile sustained settling "
            f"eta={best['eta']:.2f} ({tag}), T50={best['settling_p50']:.3f} s, "
            f"T90={best['settling_p90']:.3f} s, "
            f"P(reward>0)={100.0 * best['positive_sample_fraction']:.2f}%, "
            f"mean reward LQR={best['lqr_mean_reward']:.4f} vs "
            f"tube={best['tube_mean_reward']:.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
