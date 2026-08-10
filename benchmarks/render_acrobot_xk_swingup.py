#!/usr/bin/env python
"""Render the Xin-Kaneda controller swinging the Acrobot up from hanging.

Produces an MP4 with the mechanism on the left and the shoulder phase portrait
on the right.  The phase panel draws the homoclinic orbit ``Gamma`` (eq. 32 of
Xin & Kaneda 2007) as a fixed reference and traces ``(q1, qdot1)`` onto it, so
"reaching the orbit" is visible rather than asserted: the trajectory spirals
outward from hanging at rest and then rides the reference curve.

Headless EGL software rendering; frames are piped to ffmpeg, so no
``imageio-ffmpeg`` is needed.

    MUJOCO_GL=egl python -m benchmarks.render_acrobot_xk_swingup \\
        --duration 25 --output videos/acrobot_xk_swingup.mp4
"""

import argparse
import os
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np

from controllers.xin_kaneda import AcrobotParams, Gains, XinKanedaController, homoclinic_speed
from environment.dmc import DMCContinuousEnv
from evaluations.acrobot_homoclinic_metrics import (
    Scales,
    TubeSpec,
    energy_error,
    inside_tube,
    lqr_residual,
)


PANEL = 560
BAR = 96
TAIL_SECONDS = 3.0

# Frames are assembled in BGR, which is what ffmpeg is fed.
INK = (240, 238, 236)
MUTED = (164, 156, 150)
ORBIT = (255, 190, 120)   # light blue
TRACE = (92, 176, 255)    # amber
LIVE = (160, 235, 110)    # green
BACKDROP = (30, 26, 24)


def _wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


class PhasePanel:
    """Fixed-axes ``(q1, qdot1)`` plot with Gamma drawn as a reference."""

    def __init__(self, scales: Scales, size: int = PANEL):
        self.size = size
        self.rate_limit = 1.35 * scales.rate
        self.margin = 62
        # Gamma, from the smooth parameterization u = q1 - pi/2.
        offset = np.linspace(0.0, 4.0 * np.pi, 1441)
        self.orbit = (
            _wrap(0.5 * np.pi + offset),
            scales.rate * np.sin(0.5 * offset),
        )
        self._background = self._draw_background()

    def _to_pixel(self, q1, rate):
        span = self.size - 2 * self.margin
        x = self.margin + (_wrap(q1) + np.pi) / (2 * np.pi) * span
        y = self.margin + (self.rate_limit - rate) / (2 * self.rate_limit) * span
        return np.stack([x, y], axis=-1).astype(np.int32)

    def _draw_background(self):
        img = np.full((self.size, self.size, 3), BACKDROP, dtype=np.uint8)
        lo, hi = self.margin, self.size - self.margin
        cv2.rectangle(img, (lo, lo), (hi, hi), (54, 58, 66), 1, cv2.LINE_AA)
        # Zero-rate line and the upright abscissa.
        zero = self._to_pixel(np.array([-np.pi]), np.array([0.0]))[0]
        cv2.line(img, (lo, int(zero[1])), (hi, int(zero[1])), (54, 58, 66), 1, cv2.LINE_AA)
        up = self._to_pixel(np.array([0.5 * np.pi]), np.array([0.0]))[0]
        cv2.line(img, (int(up[0]), lo), (int(up[0]), hi), (54, 58, 66), 1, cv2.LINE_AA)
        # Gamma itself, as disconnected segments so the wrap does not draw a
        # chord straight across the panel.
        points = self._to_pixel(self.orbit[0], self.orbit[1])
        jump = np.abs(np.diff(self.orbit[0])) > np.pi
        for start, end in zip(points[:-1][~jump], points[1:][~jump]):
            cv2.line(img, tuple(start), tuple(end), ORBIT, 2, cv2.LINE_AA)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, "shoulder phase portrait", (lo, lo - 30), font, 0.62, INK, 1, cv2.LINE_AA)
        cv2.putText(img, "homoclinic orbit (eq. 32)", (lo, lo - 10), font, 0.5, ORBIT, 1, cv2.LINE_AA)
        cv2.putText(img, "trajectory", (lo + 260, lo - 10), font, 0.5, TRACE, 1, cv2.LINE_AA)
        cv2.putText(img, "-pi", (lo - 14, hi + 26), font, 0.46, MUTED, 1, cv2.LINE_AA)
        cv2.putText(img, "q1", (self.size // 2 - 10, hi + 26), font, 0.46, MUTED, 1, cv2.LINE_AA)
        cv2.putText(img, "pi", (hi - 12, hi + 26), font, 0.46, MUTED, 1, cv2.LINE_AA)
        cv2.putText(img, "up", (int(up[0]) - 10, lo - 2), font, 0.44, MUTED, 1, cv2.LINE_AA)
        cv2.putText(img, f"+{self.rate_limit:.1f}", (12, lo + 6), font, 0.44, MUTED, 1, cv2.LINE_AA)
        cv2.putText(img, "qdot1", (12, self.size // 2), font, 0.46, MUTED, 1, cv2.LINE_AA)
        cv2.putText(img, f"-{self.rate_limit:.1f}", (12, hi + 6), font, 0.44, MUTED, 1, cv2.LINE_AA)
        return img

    def render(self, q1_history, rate_history):
        img = self._background.copy()
        if len(q1_history) > 1:
            q1 = np.asarray(q1_history)
            rate = np.clip(np.asarray(rate_history), -self.rate_limit, self.rate_limit)
            points = self._to_pixel(q1, rate)
            jump = np.abs(np.diff(_wrap(q1))) > np.pi
            for start, end in zip(points[:-1][~jump], points[1:][~jump]):
                cv2.line(img, tuple(start), tuple(end), TRACE, 1, cv2.LINE_AA)
            cv2.circle(img, tuple(points[-1]), 6, LIVE, -1, cv2.LINE_AA)
        return img


def status_bar(
    width, time_s, normalized_energy, elbow, torque, captured, residual, threshold
):
    img = np.full((BAR, width, 3), BACKDROP, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    left = [
        f"t = {time_s:6.2f} s",
        f"(E - E_r) / E_s = {normalized_energy:+.4f}",
    ]
    middle = [
        f"q2 = {elbow:+.4f} rad",
        f"tau2 = {torque:+7.2f} N.m",
    ]
    for row, text in enumerate(left):
        cv2.putText(img, text, (26, 34 + 34 * row), font, 0.62, INK, 1, cv2.LINE_AA)
    for row, text in enumerate(middle):
        cv2.putText(img, text, (330, 34 + 34 * row), font, 0.62, INK, 1, cv2.LINE_AA)
    ready = residual < threshold
    cv2.putText(
        img,
        f"LQR residual = {residual:6.3f}   (switch at {threshold:g})",
        (660, 34),
        font,
        0.62,
        LIVE if ready else INK,
        1,
        cv2.LINE_AA,
    )
    banner = "LQR SWITCH" if ready else ("ON ORBIT" if captured else "")
    if banner:
        cv2.putText(img, banner, (660, 74), font, 0.8, LIVE, 2, cv2.LINE_AA)
    return img


def build(args) -> int:
    env = DMCContinuousEnv(
        "acrobot",
        "swingup-xk",
        seed=args.seed,
        raw_state_obs=True,
        dt=args.dt,
        physics_dt=args.dt,
        max_steps=int(round(args.duration / args.dt)) + 1,
        episode_duration=args.duration,
        task_kwargs=dict(
            damping=args.damping,
            angle_noise=args.angle_noise,
            paper_start=(args.start == "paper"),
            uniform_start=(args.start == "uniform"),
        ),
    )
    physics = env._env.physics
    params = AcrobotParams.from_physics(physics)
    scales = Scales.from_params(params)
    controller = XinKanedaController(
        params, Gains(k_v=args.kv, k_d=args.kd, k_p=args.kp)
    )
    spec = TubeSpec()
    panel = PhasePanel(scales)

    stride = max(1, int(round(1.0 / (args.fps * args.slowdown * args.dt))))
    tail = int(round(TAIL_SECONDS / args.dt))
    width = PANEL * 2

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
    q1_history: list[float] = []
    rate_history: list[float] = []
    frames = 0
    captured_at = None
    switched_at = None
    best_residual = float("inf")
    step = 0
    while True:
        q1_history.append(float(obs[0]))
        rate_history.append(float(obs[2]))
        if len(q1_history) > tail:
            q1_history.pop(0)
            rate_history.pop(0)
        if step % stride == 0:
            state = np.asarray(obs, dtype=np.float64)[None, :]
            inside = bool(inside_tube(state, params, spec, scales)[0])
            if inside and captured_at is None:
                captured_at = float(env.cur_t)
            residual = float(lqr_residual(state)[0])
            best_residual = min(best_residual, residual)
            scene = physics.render(
                height=PANEL, width=PANEL, camera_id="fixed"
            )[:, :, ::-1]
            composite = np.vstack(
                [
                    np.hstack([scene, panel.render(q1_history, rate_history)]),
                    status_bar(
                        width,
                        float(env.cur_t),
                        float(energy_error(state, params)[0]) / scales.energy,
                        float(_wrap(obs[1])),
                        controller.last_torque,
                        inside,
                        residual,
                        args.lqr_threshold,
                    ),
                ]
            )
            payload = np.ascontiguousarray(composite).tobytes()
            ffmpeg.stdin.write(payload)
            frames += 1
            # Hold on the switch: the residual dips under the threshold only
            # briefly, so at real-time playback the moment would be one frame.
            if residual < args.lqr_threshold and switched_at is None:
                switched_at = float(env.cur_t)
                for _ in range(int(round(args.hold * args.fps))):
                    ffmpeg.stdin.write(payload)
                    frames += 1
        obs, terminated, truncated = _advance(env, controller(obs))
        step += 1
        if terminated or truncated:
            break

    ffmpeg.stdin.close()
    ffmpeg.wait()
    final = np.asarray(obs, dtype=np.float64)[None, :]
    print(f"wrote {args.output}: {frames} frames at {args.fps} fps")
    print(
        f"  first entry to the tube: "
        f"{'%.2f s' % captured_at if captured_at is not None else 'never'}"
    )
    print(
        f"  first LQR switch (residual < {args.lqr_threshold:g}): "
        f"{'%.4f s' % switched_at if switched_at is not None else 'never'}"
        f"   (closest approach {best_residual:.5f})"
    )
    print(
        f"  final |E - E_r| / E_s = "
        f"{abs(float(energy_error(final, params)[0])) / scales.energy:.5f}"
    )
    return 0


def _advance(env, action):
    """Step the env and return ``(obs, terminated, truncated)``."""
    result = env.step_dt(action)
    return result[4], result[6], result[7]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", default="videos/acrobot_xk_swingup.mp4")
    parser.add_argument("--duration", type=float, default=25.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument(
        "--slowdown",
        type=float,
        default=1.0,
        help="playback slowdown; 2 renders twice as many frames per second",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--start",
        choices=("hanging", "paper", "uniform"),
        default="hanging",
        help="paper is the 2007 initial condition q1 = -1.4 with qdot = 0",
    )
    parser.add_argument("--lqr-threshold", type=float, default=0.04)
    parser.add_argument(
        "--hold",
        type=float,
        default=2.0,
        help="seconds to freeze on the first LQR switch",
    )
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--angle-noise", type=float, default=0.05)
    parser.add_argument("--kv", type=float, default=66.3)
    parser.add_argument("--kd", type=float, default=35.8)
    parser.add_argument("--kp", type=float, default=61.2)
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(build(parse_args()))
