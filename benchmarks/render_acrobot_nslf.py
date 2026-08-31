#!/usr/bin/env python
"""Render the nonsmooth Lyapunov function and its Lai attractive-area switch.

Swing-up uses the Xin-Kaneda law, which is the same energy-and-posture function
the outer piece of the nonsmooth Lyapunov function is built from, and the
controller latches to the local Riccati feedback on first entry to Lai et al.'s
equation-(17) region rather than to the 2007 ``zeta = 0.04`` test.

Three panels: the mechanism, the shoulder phase portrait borrowed from
``render_acrobot_xk_swingup`` with the homoclinic orbit drawn as a reference,
and a trace of ``V(t)``.  The third panel is the point of the video.  The
transition band is shaded and the switch instant is marked, so the property the
offset ``Delta`` buys -- that crossing the gate never steps the value up -- is
visible rather than asserted.  ``Delta`` itself is drawn as the level the value
holds while the trajectory rides the orbit.

``--shoulder`` releases the straight chain from a chosen displacement, which
the plant's own near-hanging start cannot express because it perturbs both
joints at random.  The default enters the region around 10 s and balances with
under 1 N.m to spare; ``--shoulder -1.390796`` enters near 4 s at the cost of a
visible overshoot in the value while the local feedback settles.

Entry to the region is not by itself a successful balance.  Sweeping releases
under this pairing, 30 of 98 enter, and of those only 10 hold upright at the
plant's 64 N.m; the published linear gain asks for far more than that near the
boundary, and the value then rises steeply.  The construction is unharmed by
this -- the rise is the linear law overshooting, and the swing-up and the
transition band contribute none of it -- but a start has to be chosen with it
in mind, which is why the default is pinned rather than sampled.

Headless EGL software rendering; frames are piped to ffmpeg.

    MUJOCO_GL=egl python -m benchmarks.render_acrobot_nslf \\
        --duration 30 --output videos/acrobot_nslf/acrobot_nslf_lqr_switch.mp4
"""

import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import subprocess

import cv2
import numpy as np

from benchmarks.render_acrobot_xk_swingup import (
    BACKDROP,
    BAR,
    INK,
    LIVE,
    MUTED,
    ORBIT,
    PANEL,
    TRACE,
    PhasePanel,
    _advance,
    _wrap,
)
from controllers.acrobot_gated_lyapunov import AttractiveRegion, NonsmoothLyapunov
from controllers.xin_kaneda import AcrobotParams, Gains, XinKanedaController
from environment.dmc import DMCContinuousEnv
from evaluations.acrobot_homoclinic_metrics import Scales


GATE = (96, 108, 150)      # shaded transition band
SWITCH = (150, 120, 255)   # switch marker


class SwitchedController:
    """Xin-Kaneda swing-up, latching to the local feedback inside ``Sigma_2``.

    The switch is one way, as in Lai et al.: the region is entered once and the
    balance law keeps it.  ``last_torque`` is the physical elbow torque actually
    applied, after the plant's own actuator bound.
    """

    SWING_UP = 1
    BALANCE = 2

    def __init__(self, params, gains, lyapunov, *, torque_limit=None):
        self.params = params
        self.lyapunov = lyapunov
        self.torque_limit = (
            float(params.gear) if torque_limit is None else float(torque_limit)
        )
        self.swing_up = XinKanedaController(
            params, gains, torque_limit=self.torque_limit
        )
        self.reset()

    def reset(self):
        self.swing_up.reset()
        self.stage = self.SWING_UP
        self.switch_time = None
        self.last_torque = 0.0
        self.last_commanded_torque = 0.0
        self.saturated_after_switch = 0

    def __call__(self, obs, now):
        state = np.asarray(obs, dtype=np.float64).reshape(-1)
        if self.stage == self.SWING_UP and self.lyapunov.region.contains(
            self.params, state
        ):
            self.stage = self.BALANCE
            self.switch_time = float(now)
        if self.stage == self.SWING_UP:
            action = self.swing_up(obs)
            self.last_torque = self.swing_up.last_torque
            self.last_commanded_torque = self.swing_up.last_commanded_torque
            return action
        commanded = self.lyapunov.lqr_torque(state)
        applied = float(np.clip(commanded, -self.torque_limit, self.torque_limit))
        if abs(commanded) > self.torque_limit:
            self.saturated_after_switch += 1
        self.last_commanded_torque = commanded
        self.last_torque = applied
        return np.array([applied / self.params.gear], dtype=np.float64)


class LyapunovPanel:
    """``V(t)`` on fixed axes, with the gate shaded and the switch marked."""

    def __init__(self, lyapunov, duration, size=PANEL):
        self.size = size
        self.margin = 62
        self.duration = float(duration)
        self.offset = lyapunov.normalized_delta
        # Hanging rest is the largest value the outer piece takes, so the axis
        # is fixed by the construction rather than by the trajectory.
        self.ceiling = 1.06 * (1.0 + self.offset)
        self._background = self._draw_background()

    def _to_pixel(self, time_s, value):
        span = self.size - 2 * self.margin
        x = self.margin + np.asarray(time_s) / self.duration * span
        y = self.margin + (
            self.ceiling - np.clip(np.asarray(value), 0.0, self.ceiling)
        ) / self.ceiling * span
        return np.stack([x, y], axis=-1).astype(np.int32)

    def _draw_background(self):
        img = np.full((self.size, self.size, 3), BACKDROP, dtype=np.uint8)
        low, high = self.margin, self.size - self.margin
        cv2.rectangle(img, (low, low), (high, high), (54, 58, 66), 1, cv2.LINE_AA)
        font = cv2.FONT_HERSHEY_DUPLEX
        # The level the value holds on the orbit, which is the whole offset.
        level = int(self._to_pixel([0.0], [self.offset])[0][1])
        for x in range(low, high, 14):
            cv2.line(img, (x, level), (x + 7, level), ORBIT, 1, cv2.LINE_AA)
        cv2.putText(img, f"Delta = {self.offset:.4f}", (low + 8, level - 10),
                    font, 0.5, ORBIT, 1, cv2.LINE_AA)
        cv2.putText(img, "Lyapunov value V(t)", (low, low - 30), font, 0.62,
                    INK, 1, cv2.LINE_AA)
        cv2.putText(img, "shaded: transition band", (low, low - 10), font, 0.5,
                    GATE, 1, cv2.LINE_AA)
        cv2.putText(img, "0", (low - 16, high + 6), font, 0.44, MUTED, 1, cv2.LINE_AA)
        cv2.putText(img, "upright", (low - 4, high + 26), font, 0.44, MUTED, 1,
                    cv2.LINE_AA)
        cv2.putText(img, f"{self.ceiling:.2f}", (12, low + 6), font, 0.44, MUTED,
                    1, cv2.LINE_AA)
        cv2.putText(img, "V", (20, self.size // 2), font, 0.46, MUTED, 1, cv2.LINE_AA)
        cv2.putText(img, f"t / {self.duration:g} s",
                    (self.size // 2 - 40, high + 26), font, 0.46, MUTED, 1,
                    cv2.LINE_AA)
        return img

    def render(self, times, values, gates, switch_time):
        img = self._background.copy()
        low, high = self.margin, self.size - self.margin
        if len(times) > 1:
            # Shade every span on which the gate is partly or fully open.
            active = np.asarray(gates) > 0.0
            edges = np.flatnonzero(np.diff(active.astype(np.int8)))
            bounds = np.concatenate([[0], edges + 1, [len(active)]])
            for start, end in zip(bounds[:-1], bounds[1:]):
                if not active[start] or end - start < 1:
                    continue
                x0 = int(self._to_pixel([times[start]], [0.0])[0][0])
                x1 = int(self._to_pixel([times[end - 1]], [0.0])[0][0])
                overlay = img.copy()
                cv2.rectangle(overlay, (x0, low), (max(x1, x0 + 1), high), GATE, -1)
                cv2.addWeighted(overlay, 0.30, img, 0.70, 0.0, img)
            points = self._to_pixel(times, values)
            for start, end in zip(points[:-1], points[1:]):
                cv2.line(img, tuple(start), tuple(end), TRACE, 2, cv2.LINE_AA)
            cv2.circle(img, tuple(points[-1]), 6, LIVE, -1, cv2.LINE_AA)
        if switch_time is not None:
            x = int(self._to_pixel([switch_time], [0.0])[0][0])
            cv2.line(img, (x, low), (x, high), SWITCH, 1, cv2.LINE_AA)
            cv2.putText(img, "switch", (x + 6, low + 18), cv2.FONT_HERSHEY_DUPLEX,
                        0.46, SWITCH, 1, cv2.LINE_AA)
        return img


def status_bar(width, time_s, normalized_energy, elbow, torque, residual, value,
               gate, stage, rise):
    img = np.full((BAR, width, 3), BACKDROP, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    columns = (
        (26, [f"t = {time_s:6.2f} s",
              f"(E - E_r) / E_s = {normalized_energy:+.4f}"]),
        (330, [f"q2 = {elbow:+.4f} rad",
               f"tau2 = {torque:+7.2f} N.m"]),
        (660, [f"Sigma_2 residual = {residual:6.3f}   (enter at 1)",
               f"gate mu = {gate:.3f}"]),
        (1120, [f"V = {value:.4f}",
                f"largest rise so far = {rise:+.2e}"]),
    )
    for x, lines in columns:
        for row, text in enumerate(lines):
            colour = LIVE if (x == 660 and residual <= 1.0) else INK
            cv2.putText(img, text, (x, 34 + 34 * row), font, 0.58, colour, 1,
                        cv2.LINE_AA)
    if stage == SwitchedController.BALANCE:
        cv2.putText(img, "BALANCE", (1450, 44), font, 0.8, LIVE, 2, cv2.LINE_AA)
    return img


def build(args) -> int:
    duration = args.duration
    env = DMCContinuousEnv(
        "acrobot",
        "swingup-xk",
        seed=args.seed,
        raw_state_obs=True,
        dt=args.dt,
        physics_dt=args.dt,
        max_steps=int(round(duration / args.dt)) + 1,
        episode_duration=duration,
        task_kwargs=dict(
            damping=args.damping,
            angle_noise=args.angle_noise,
            paper_start=(args.start == "paper"),
            uniform_start=(args.start == "uniform"),
        ),
    )
    physics = env._env.physics
    params = AcrobotParams.from_physics(physics)
    gains = Gains(k_v=args.kv, k_d=args.kd, k_p=args.kp)
    region = AttractiveRegion(
        angle_tolerance=args.angle_tolerance,
        energy_tolerance=args.energy_tolerance,
    )
    lyapunov = NonsmoothLyapunov(params, gains, region)
    controller = SwitchedController(
        params, gains, lyapunov, torque_limit=args.torque_limit
    )
    scales = Scales.from_params(params)
    phase = PhasePanel(scales)
    trace = LyapunovPanel(lyapunov, duration)

    print(f"Delta = {lyapunov.delta:.2f}  ({lyapunov.normalized_delta:.4f} of the "
          f"value at hanging rest)")
    print(f"region: angles {region.angle_tolerance:.5f} rad, energy "
          f"{region.energy_tolerance:g} J; actuator {controller.torque_limit:g} N.m")

    stride = max(1, int(round(1.0 / (args.fps * args.slowdown * args.dt))))
    tail = int(round(args.phase_tail / args.dt))
    width = PANEL * 3

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{PANEL + BAR}", "-r", str(args.fps),
            "-i", "-",
            "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "20", args.output,
        ],
        stdin=subprocess.PIPE,
    )

    obs, _ = env.reset(seed=args.seed)
    if args.shoulder is not None:
        # The near-hanging start perturbs both joints at random, so a chosen
        # release is written in directly; raw_state_obs is exactly this vector.
        physics.named.data.qpos[["shoulder", "elbow"]] = [args.shoulder, 0.0]
        physics.named.data.qvel[["shoulder", "elbow"]] = 0.0
        physics.forward()
        obs = np.concatenate(
            [
                np.asarray(physics.named.data.qpos[["shoulder", "elbow"]]),
                np.asarray(physics.named.data.qvel[["shoulder", "elbow"]]),
            ]
        )
        print(f"released from q1 = {args.shoulder:.6f} rad, straight and at rest")
    q1_history: list[float] = []
    rate_history: list[float] = []
    times: list[float] = []
    values: list[float] = []
    gates: list[float] = []
    frames = 0
    held = 0
    largest_rise = 0.0
    previous_value = None
    best_residual = float("inf")
    step = 0
    while True:
        state = np.asarray(obs, dtype=np.float64).reshape(-1)
        now = float(env.cur_t)
        q1_history.append(float(state[0]))
        rate_history.append(float(state[2]))
        if len(q1_history) > tail:
            q1_history.pop(0)
            rate_history.pop(0)
        if step % stride == 0:
            value = lyapunov.value(state)
            if previous_value is not None:
                largest_rise = max(largest_rise, value - previous_value)
            previous_value = value
            residual = region.exact_residual(params, state)
            best_residual = min(best_residual, residual)
            times.append(now)
            values.append(value)
            gates.append(lyapunov.gate(state))
            scene = physics.render(height=PANEL, width=PANEL, camera_id="fixed")
            composite = np.vstack(
                [
                    np.hstack(
                        [
                            scene[:, :, ::-1],
                            phase.render(q1_history, rate_history),
                            trace.render(times, values, gates,
                                         controller.switch_time),
                        ]
                    ),
                    status_bar(
                        width,
                        now,
                        (params.energy(state[:2], state[2:]) - params.energy_top)
                        / params.energy_span,
                        float(_wrap(state[1])),
                        controller.last_torque,
                        residual,
                        value,
                        gates[-1],
                        controller.stage,
                        largest_rise,
                    ),
                ]
            )
            payload = np.ascontiguousarray(composite).tobytes()
            ffmpeg.stdin.write(payload)
            frames += 1
            if controller.switch_time is not None and held == 0:
                held = 1
                for _ in range(int(round(args.hold * args.fps))):
                    ffmpeg.stdin.write(payload)
                    frames += 1
        obs, terminated, truncated = _advance(env, controller(obs, now))
        step += 1
        if terminated or truncated:
            break

    ffmpeg.stdin.close()
    ffmpeg.wait()

    final = np.asarray(obs, dtype=np.float64).reshape(-1)
    error = np.concatenate(
        [_wrap(final[:2] - np.array([0.5 * np.pi, 0.0])), final[2:]]
    )
    print(f"wrote {args.output}: {frames} frames at {args.fps} fps")
    print(
        f"  entered Sigma_2: "
        f"{'%.3f s' % controller.switch_time if controller.switch_time is not None else 'never'}"
        f"   (closest approach {best_residual:.4f})"
    )
    print(f"  saturated steps after the switch: {controller.saturated_after_switch}")
    print(f"  largest rise in V over the episode: {largest_rise:+.3e}")
    print(f"  final |e| = {np.linalg.norm(error):.5f}, V = {lyapunov.value(final):.5f}")
    return 0


def _optional_float(text):
    """``none`` disables an option that otherwise takes a float."""
    return None if text.strip().lower() in ("none", "") else float(text)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--output", default="videos/acrobot_nslf/acrobot_nslf_lqr_switch.mp4"
    )
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--slowdown", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--start", choices=("hanging", "paper", "uniform"), default="hanging"
    )
    parser.add_argument(
        "--shoulder", type=_optional_float, default=-1.520796,
        help="release the straight chain from this shoulder angle, overriding "
             "--start; pass 'none' to sample from --start instead",
    )
    parser.add_argument(
        "--hold", type=float, default=2.0,
        help="seconds to freeze on entry to the attractive area",
    )
    parser.add_argument(
        "--phase-tail", type=float, default=3.0,
        help="seconds of trajectory kept in the phase portrait",
    )
    parser.add_argument("--angle-tolerance", type=float, default=np.pi / 30.0)
    parser.add_argument("--energy-tolerance", type=float, default=1.0)
    parser.add_argument(
        "--torque-limit", type=float, default=None,
        help="physical elbow bound; defaults to the plant's own gear",
    )
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--angle-noise", type=float, default=0.05)
    parser.add_argument("--kv", type=float, default=66.3)
    parser.add_argument("--kd", type=float, default=35.8)
    parser.add_argument("--kp", type=float, default=61.2)
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(build(parse_args()))
