"""Energy-efficiency and gait-consistency metrics for the cheetah-run task.

The default cheetah-run reward reads only forward torso speed and carries no
control/energy term, so two policies with equal return can differ severalfold
in how much energy they inject and in whether their motion is a periodic gait
or quasi-random flailing. This module measures those two axes directly from the
MuJoCo physics, independent of the observation layout the policy was trained on.

Energy (metric 1)
-----------------
The six leg motors are plain torque actuators with fixed unit gain, so the
generalized actuator torque on each actuated DOF is ``qfrc_actuator = gear*ctrl``
exactly, and it is state-independent across a control step. Under the zero-order
hold that RL applies (``ctrl`` constant over the control step), the mechanical
work an actuator does over one step is therefore

    W = qfrc_actuator * Δqpos            (exact, no substep integration)

with ``Δqpos`` the joint-angle change over that step. Per-joint positive work is
summed to the gross positive work E+, and the primary efficiency number is the
dimensionless cost of transport ``CoT = E+ / (M g d)`` (M total mass, d forward
distance). The Gym-style ``∫ Σ a² dt`` control cost is reported as a cheap,
kinematics-free effort proxy alongside it.

Gait (metric 3)
---------------
Over the steady-cruise window (after an acceleration warm-up), each leg-joint
and foot-height signal is analysed for periodicity: spectral entropy (→ a
0..1 "regularity"), dominant stride frequency, autocorrelation peak height,
stride-period coefficient of variation, a Poincaré return-map dispersion (spread
of the state at successive section crossings — tight = limit cycle), and a
front/back limb phase-locking value. All signal processing is numpy-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Physics access
# --------------------------------------------------------------------------- #
def _dmc_env(env: Any):
    """Unwrap gym wrappers (Monitor, ...) to the DMCContinuousEnv."""
    cur = env
    for _ in range(16):
        if cur is None:
            break
        if hasattr(cur, "_env") and hasattr(cur, "domain_name"):
            return cur
        cur = getattr(cur, "env", None)
    raise TypeError(
        "locomotion_metrics needs a DMCContinuousEnv (MuJoCo physics); "
        f"could not unwrap {type(env).__name__}."
    )


def _physics(env: Any):
    return _dmc_env(env)._env.physics


# --------------------------------------------------------------------------- #
# Rollout capture
# --------------------------------------------------------------------------- #
@dataclass
class CheetahRollout:
    """Per-step physics record of a single episode (arrays indexed by step)."""

    dt: np.ndarray          # (T,)   control-step duration [s]
    time: np.ndarray        # (T,)   cumulative episode time at step end [s]
    reward: np.ndarray      # (T,)   per-step reward
    qpos: np.ndarray        # (T, nq)
    qvel: np.ndarray        # (T, nv)
    qfrc_act: np.ndarray    # (T, nv) generalized actuator torque (gear*ctrl)
    action: np.ndarray      # (T, nu) policy action (raw, unclamped)
    speed: np.ndarray       # (T,)   torso_subtreelinvel[0] (reward speed)
    foot_z: np.ndarray      # (T, 2) global height of [bfoot, ffoot]
    qpos0: np.ndarray       # (nq,)  post-reset qpos (before first action)
    act_dof: np.ndarray     # (nu,)  actuated DOF indices
    gear: np.ndarray        # (nu,)  actuator gears
    mass: float             # total body mass [kg]
    g: float                # gravity magnitude [m/s^2]


def _model_constants(phys):
    model = phys.model
    trn_joint = np.asarray(model.actuator_trnid)[:, 0]
    act_dof = np.asarray(model.jnt_dofadr)[trn_joint].astype(int)
    gear = np.asarray(model.actuator_gear)[:, 0].astype(float)
    mass = float(np.asarray(model.body_mass).sum())
    g = float(abs(np.asarray(model.opt.gravity)[2]))
    return act_dof, gear, mass, g


def rollout_cheetah(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    env: Any,
    *,
    max_steps: Optional[int] = None,
) -> CheetahRollout:
    """Run one episode, recording the physics quantities the metrics need.

    ``policy_fn`` maps an observation (np.ndarray) to an action (np.ndarray).
    Pass ``lambda _obs: env.action_space.sample()`` for a random baseline.
    """
    phys = _physics(env)
    act_dof, gear, mass, g = _model_constants(phys)
    named = phys.named

    obs, _ = env.reset()
    obs = np.asarray(obs, dtype=np.float32)
    qpos0 = np.asarray(phys.data.qpos, dtype=float).copy()

    dt_l, time_l, rew_l = [], [], []
    qpos_l, qvel_l, qfrc_l, act_l, spd_l, foot_l = [], [], [], [], [], []

    done = False
    steps = 0
    while not done:
        action = np.asarray(policy_fn(obs), dtype=np.float32).reshape(-1)
        obs_t, t, _, reward, next_obs, next_t, terminated, truncated, info = (
            env.step_dt(action)
        )
        data = phys.data
        qpos_l.append(np.asarray(data.qpos, dtype=float).copy())
        qvel_l.append(np.asarray(data.qvel, dtype=float).copy())
        qfrc_l.append(np.asarray(data.qfrc_actuator, dtype=float).copy())
        act_l.append(action.astype(float))
        spd_l.append(float(phys.speed()))
        foot_l.append([
            float(named.data.xpos["bfoot"][2]),
            float(named.data.xpos["ffoot"][2]),
        ])
        rew_l.append(float(reward))
        dt_l.append(float(info.get("dt_used", next_t - t)))
        time_l.append(float(next_t))

        obs = np.asarray(next_obs, dtype=np.float32)
        done = bool(terminated or truncated)
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break

    return CheetahRollout(
        dt=np.asarray(dt_l),
        time=np.asarray(time_l),
        reward=np.asarray(rew_l),
        qpos=np.asarray(qpos_l),
        qvel=np.asarray(qvel_l),
        qfrc_act=np.asarray(qfrc_l),
        action=np.asarray(act_l),
        speed=np.asarray(spd_l),
        foot_z=np.asarray(foot_l),
        qpos0=qpos0,
        act_dof=act_dof,
        gear=gear,
        mass=mass,
        g=g,
    )


# --------------------------------------------------------------------------- #
# Energy metrics (metric 1)
# --------------------------------------------------------------------------- #
def energy_metrics(roll: CheetahRollout) -> Dict[str, float]:
    ad = roll.act_dof
    q = roll.qpos[:, ad]                                   # (T, nu)
    q_prev = np.vstack([roll.qpos0[ad][None, :], q[:-1]])  # ZOH previous angle
    dq = q - q_prev
    tau = roll.qfrc_act[:, ad]                             # gear*ctrl, exact
    work = tau * dq                                        # (T, nu) per-joint work [J]

    e_pos = float(np.clip(work, 0.0, None).sum())         # gross positive work
    e_neg = float(np.clip(work, None, 0.0).sum())         # braking (negative)
    e_net = float(work.sum())
    e_abs = float(np.abs(work).sum())

    duration = float(roll.dt.sum())
    distance = float(roll.qpos[-1, 0] - roll.qpos0[0])     # forward (rootx) travel
    ret = float(roll.reward.sum())

    a = np.clip(roll.action, -1.0, 1.0)
    control_cost = float((a ** 2).sum(axis=1) @ roll.dt)   # ∫ Σ a² dt

    mgd = roll.mass * roll.g * distance
    good = distance > 1e-4
    return {
        "return": ret,
        "distance_m": distance,
        "duration_s": duration,
        "mean_speed_mps": distance / duration if duration > 0 else np.nan,
        "work_pos_J": e_pos,
        "work_neg_J": e_neg,
        "work_net_J": e_net,
        "work_abs_J": e_abs,
        "mean_power_W": e_net / duration if duration > 0 else np.nan,
        "mean_abs_power_W": e_abs / duration if duration > 0 else np.nan,
        "control_cost": control_cost,
        # Cost of transport (dimensionless): energy per unit weight per distance.
        "cost_of_transport": e_pos / mgd if good else np.nan,
        "cost_of_transport_net": e_net / mgd if good else np.nan,
        # Reward earned per joule injected (higher = more efficient).
        "return_per_joule": ret / e_pos if e_pos > 1e-9 else np.nan,
        "return_per_joule_abs": ret / e_abs if e_abs > 1e-9 else np.nan,
    }


# --------------------------------------------------------------------------- #
# Signal-processing helpers (numpy-only)
# --------------------------------------------------------------------------- #
def _welch_psd(x: np.ndarray, fs: float, n_seg: int = 8):
    """One-sided power spectral density via Welch averaging (Hann window)."""
    n = len(x)
    seg = int(np.clip(n // n_seg, 64, n))
    if seg >= n:
        win = np.hanning(n)
        X = np.fft.rfft(x * win)
        psd = (np.abs(X) ** 2) / (fs * (win ** 2).sum())
        return np.fft.rfftfreq(n, 1.0 / fs), psd
    win = np.hanning(seg)
    norm = fs * (win ** 2).sum()
    step = max(1, seg // 2)
    psds = [
        (np.abs(np.fft.rfft(x[s:s + seg] * win)) ** 2) / norm
        for s in range(0, n - seg + 1, step)
    ]
    return np.fft.rfftfreq(seg, 1.0 / fs), np.mean(psds, axis=0)


_NAN_SPEC = {"spectral_entropy": np.nan, "dom_freq": np.nan,
             "peak_frac": np.nan, "band_frac": np.nan}


def _spectral_stats(
    x: np.ndarray, fs: float, fmin: float = 0.5, fmax: float = 8.0
) -> Dict[str, float]:
    """Spectral entropy, dominant frequency, peak- and stride-band power fractions.

    The analysis is restricted to the plausible stride band ``[fmin, fmax]`` so a
    near-stationary signal (all power at DC / very low frequency) is not scored as
    a periodic gait. ``band_frac`` is the fraction of total AC power that falls in
    the stride band — low for a policy that is drifting or standing rather than
    stepping. Entropy is normalized to [0, 1] over the in-band bins.
    """
    x = x - x.mean()
    if not np.any(np.abs(x) > 1e-9):
        return dict(_NAN_SPEC)
    f, p = _welch_psd(x, fs)
    ac_total = float(p[1:].sum())             # total power excluding DC
    fmax = min(fmax, 0.45 * fs)
    band = (f >= fmin) & (f <= fmax)
    fb, pb = f[band], p[band]
    if fb.size < 2 or pb.sum() <= 0 or ac_total <= 0:
        return dict(_NAN_SPEC)
    band_frac = float(pb.sum() / ac_total)
    pn = pb / pb.sum()
    entropy = float(-(pn * np.log(pn + 1e-12)).sum() / np.log(len(pn)))
    dom = float(fb[int(pb.argmax())])
    # Concentration of in-band power at the dominant (fundamental) peak. Tolerance
    # is at least ~1.5 spectral bins so a windowed tone's main lobe is not clipped.
    # (Harmonics of a low fundamental would span the whole stride band, so counting
    # them would make this saturate; the fundamental peak is the clean measure.)
    df = float(fb[1] - fb[0])
    peak_mask = np.abs(fb - dom) <= max(0.15 * dom, 1.5 * df)
    peak_frac = float(pb[peak_mask].sum() / pb.sum())
    return {"spectral_entropy": entropy, "dom_freq": dom,
            "peak_frac": peak_frac, "band_frac": band_frac}


def _autocorr_peak(x: np.ndarray, fs: float, fmin: float = 0.4, fmax: float = 8.0):
    """Height (∈ ~(-1, 1]) and lag of the dominant autocorrelation peak."""
    x = x - x.mean()
    n = len(x)
    if not np.any(np.abs(x) > 1e-9):
        return np.nan, np.nan
    ac = np.correlate(x, x, mode="full")[n - 1:]
    ac = ac / ac[0]
    lo, hi = int(fs / fmax), min(n - 1, int(fs / fmin))
    if hi <= lo + 1:
        return np.nan, np.nan
    k = int(np.argmax(ac[lo:hi])) + lo
    return float(ac[k]), float(k / fs)


def _analytic(x: np.ndarray) -> np.ndarray:
    """Analytic signal via FFT (numpy-only Hilbert transform)."""
    n = len(x)
    X = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(X * h)


def _phase(x: np.ndarray) -> np.ndarray:
    return np.unwrap(np.angle(_analytic(x - x.mean())))


def _detrend(x: np.ndarray) -> np.ndarray:
    """Remove the least-squares linear trend (kills slow drift before phase/AC)."""
    n = len(x)
    t = np.arange(n)
    a, b = np.polyfit(t, x, 1)
    return x - (a * t + b)


def _stride_stats(phi: np.ndarray, t: np.ndarray) -> Dict[str, float]:
    """Stride period and its coefficient of variation from an instantaneous phase."""
    if phi[-1] < phi[0]:
        phi = -phi
    if phi[-1] - phi[0] < 2 * np.pi:
        return {"stride_period_s": np.nan, "stride_cv": np.nan, "n_strides": 0}
    ks = np.arange(np.ceil(phi[0] / (2 * np.pi)), np.floor(phi[-1] / (2 * np.pi)) + 1)
    cross_t = np.interp(2 * np.pi * ks, phi, t)
    periods = np.diff(cross_t)
    if len(periods) < 2:
        return {"stride_period_s": np.nan, "stride_cv": np.nan, "n_strides": len(periods)}
    mean = float(periods.mean())
    return {
        "stride_period_s": mean,
        "stride_cv": float(periods.std() / mean) if mean > 0 else np.nan,
        "n_strides": int(len(periods)),
    }


def _plv(x: np.ndarray, y: np.ndarray) -> float:
    """Phase-locking value between two oscillators (∈ [0, 1])."""
    px = np.angle(_analytic(x - x.mean()))
    py = np.angle(_analytic(y - y.mean()))
    return float(np.abs(np.mean(np.exp(1j * (px - py)))))


def _poincare_dispersion(ref: np.ndarray, state: np.ndarray, norm: np.ndarray):
    """RMS spread (normalized) of the state at upward mean-crossings of ``ref``."""
    r = ref - ref.mean()
    idx = np.where((r[:-1] < 0) & (r[1:] >= 0))[0]
    if len(idx) < 3:
        return np.nan, int(len(idx))
    pts = []
    for i in idx:
        denom = r[i + 1] - r[i]
        frac = (-r[i] / denom) if denom != 0 else 0.0
        pts.append(state[i] + frac * (state[i + 1] - state[i]))
    pts = np.asarray(pts)
    scale = norm.copy()
    scale[scale < 1e-8] = 1.0
    pz = (pts - pts.mean(axis=0)) / scale
    return float(np.sqrt((pz ** 2).sum(axis=1)).mean()), int(len(idx))


# --------------------------------------------------------------------------- #
# Gait metrics (metric 3)
# --------------------------------------------------------------------------- #
_NAN_GAIT = {
    "gait_detected": 0.0, "autocorr_peak": np.nan, "stride_cv": np.nan,
    "poincare_dispersion": np.nan, "limb_plv": np.nan, "stride_freq_hz": np.nan,
    "stride_period_s": np.nan, "spectral_entropy_mean": np.nan,
    "peak_power_frac_mean": np.nan, "band_power_frac_mean": np.nan, "n_strides": 0,
}


def gait_metrics(
    roll: CheetahRollout,
    *,
    warmup_s: float = 2.0,
    fmin: float = 0.5,
    fmax: float = 8.0,
    min_strides: int = 3,
    min_band_frac: float = 0.2,
) -> Dict[str, float]:
    """Gait-consistency battery over the steady-cruise window.

    Returns ``gait_detected`` = 1.0 only when at least ``min_strides`` stride
    cycles are found *and* a real fraction (``min_band_frac``) of the limb-motion
    power lies in the stride band; otherwise the periodicity scores are NaN. The
    band-power gate is what separates a genuine oscillation from broadband jitter
    of a standing policy (whose analytic-phase still winds up spurious "strides").
    ``band_power_frac`` and ``n_strides`` are still reported so the absence of a
    gait is visible.
    """
    t = roll.time
    mask = t >= (t[0] + warmup_s) if len(t) else np.zeros(0, dtype=bool)
    tt = t[mask]
    if len(tt) < 64:
        return dict(_NAN_GAIT)

    # Resample to a uniform grid (control dt may be irregular) for clean FFTs.
    fs = 1.0 / float(np.median(np.diff(tt)))
    grid = np.arange(tt[0], tt[-1], 1.0 / fs)
    if len(grid) < 64:
        return dict(_NAN_GAIT)

    def rs(sig: np.ndarray) -> np.ndarray:
        return np.interp(grid, tt, sig[mask])

    ad = roll.act_dof                       # [bthigh, bshin, bfoot, fthigh, fshin, ffoot]
    joints = [rs(roll.qpos[:, j]) for j in ad]
    jvel = [rs(roll.qvel[:, j]) for j in ad]
    feet = [rs(roll.foot_z[:, 0]), rs(roll.foot_z[:, 1])]

    stats = [_spectral_stats(s, fs, fmin, fmax) for s in joints + feet]
    entropies = np.array([s["spectral_entropy"] for s in stats], dtype=float)
    peak_fracs = np.array([s["peak_frac"] for s in stats], dtype=float)
    dom_freqs = np.array([s["dom_freq"] for s in stats], dtype=float)
    band_fracs = np.array([s["band_frac"] for s in stats], dtype=float)

    # Reference oscillator = most spectrally concentrated leg joint (in-band).
    joint_pf = peak_fracs[:len(joints)]
    ref_i = int(np.nanargmax(joint_pf)) if np.any(np.isfinite(joint_pf)) else 0
    ref = _detrend(joints[ref_i])

    stride = _stride_stats(_phase(ref), grid)
    n_strides = int(stride["n_strides"])
    band_mean = float(np.nanmean(band_fracs))

    if n_strides < min_strides or band_mean < min_band_frac:
        out = dict(_NAN_GAIT)
        out["n_strides"] = n_strides
        out["band_power_frac_mean"] = band_mean
        return out

    ac_peak, _ = _autocorr_peak(ref, fs, fmin, fmax)
    state = np.column_stack(joints + jvel)
    disp, _ = _poincare_dispersion(ref, state, state.std(axis=0))
    plv = _plv(joints[0], joints[3])        # back thigh vs front thigh

    return {
        "gait_detected": 1.0,
        # Primary "is it a periodic gait" measures.
        "autocorr_peak": ac_peak,                       # 1 = perfectly periodic
        "stride_cv": stride["stride_cv"],               # 0 = identical strides
        "poincare_dispersion": disp,                    # 0 = tight limit cycle
        "limb_plv": plv,                                # 1 = front/back locked
        # Descriptive / spectral diagnostics.
        "stride_freq_hz": float(np.nanmedian(dom_freqs)),
        "stride_period_s": stride["stride_period_s"],
        "spectral_entropy_mean": float(np.nanmean(entropies)),   # 0 = pure tone
        "peak_power_frac_mean": float(np.nanmean(peak_fracs)),   # 1 = single freq
        "band_power_frac_mean": float(np.nanmean(band_fracs)),   # motion in stride band
        "n_strides": n_strides,
    }


# --------------------------------------------------------------------------- #
# Episode driver + aggregation
# --------------------------------------------------------------------------- #
def episode_metrics(roll: CheetahRollout, *, warmup_s: float = 2.0) -> Dict[str, float]:
    out = energy_metrics(roll)
    out.update(gait_metrics(roll, warmup_s=warmup_s))
    return out


def _aggregate(rows: List[Dict[str, float]]) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    keys = rows[0].keys() if rows else []
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        finite = vals[np.isfinite(vals)]
        agg[f"{k}_mean"] = float(finite.mean()) if finite.size else np.nan
        agg[f"{k}_std"] = float(finite.std()) if finite.size else np.nan
        agg[f"{k}_n"] = int(finite.size)
    return agg


def evaluate_locomotion(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    env: Any,
    *,
    n_episodes: int = 5,
    warmup_s: float = 2.0,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Roll out ``n_episodes`` and return per-episode metrics plus aggregates."""
    per_episode: List[Dict[str, float]] = []
    for _ in range(n_episodes):
        roll = rollout_cheetah(policy_fn, env, max_steps=max_steps)
        per_episode.append(episode_metrics(roll, warmup_s=warmup_s))
    return {"per_episode": per_episode, "aggregate": _aggregate(per_episode)}
