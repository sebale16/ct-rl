#!/usr/bin/env python
"""Render the Lai--She 2009 WCLF Acrobot controller and its LQR switch.

The mechanism uses the paper's Table-II plant directly in its upward-vertical
coordinates.  A phase portrait and status bar expose the WCLF swing-up, the
published equation-(75) LQR switch, energy error, and physical torque.

    MUJOCO_GL=egl .venv/bin/python -m benchmarks.render_acrobot_lai_she
"""

import argparse
import os
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np

from controllers.lai_she import AcrobotParams, LaiSheController, wrap
from environment.dmc import DMCContinuousEnv


PANEL = 560
BAR = 106
TAIL_SECONDS = 7.0
INK = (240, 238, 236)
MUTED = (164, 156, 150)
TRACE = (92, 176, 255)
LIVE = (160, 235, 110)
TARGET = (255, 190, 120)
BACKDROP = (30, 26, 24)


class PhasePanel:
    def __init__(self, size: int = PANEL):
        self.size = size
        self.margin = 62
        self.rate_limit = 12.0
        self._background = self._draw_background()

    def _to_pixel(self, angle, rate):
        span = self.size - 2 * self.margin
        horizontal = self.margin + (wrap(angle) + np.pi) / (2 * np.pi) * span
        vertical = self.margin + (
            self.rate_limit - np.clip(rate, -self.rate_limit, self.rate_limit)
        ) / (2 * self.rate_limit) * span
        return np.stack([horizontal, vertical], axis=-1).astype(np.int32)

    def _draw_background(self):
        image = np.full((self.size, self.size, 3), BACKDROP, dtype=np.uint8)
        low, high = self.margin, self.size - self.margin
        cv2.rectangle(image, (low, low), (high, high), (54, 58, 66), 1, cv2.LINE_AA)
        zero = self._to_pixel(np.array([0.0]), np.array([0.0]))[0]
        cv2.line(image, (low, int(zero[1])), (high, int(zero[1])),
                 (54, 58, 66), 1, cv2.LINE_AA)
        cv2.line(image, (int(zero[0]), low), (int(zero[0]), high),
                 (54, 58, 66), 1, cv2.LINE_AA)
        cv2.circle(image, tuple(zero), 9, TARGET, 2, cv2.LINE_AA)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, "shoulder phase portrait", (low, low - 30), font,
                    0.62, INK, 1, cv2.LINE_AA)
        cv2.putText(image, "Lai et al. 2009 coordinates", (low, low - 9), font,
                    0.50, MUTED, 1, cv2.LINE_AA)
        cv2.putText(image, "upright", (int(zero[0]) + 12, int(zero[1]) - 10),
                    font, 0.46, TARGET, 1, cv2.LINE_AA)
        cv2.putText(image, "-pi", (low - 14, high + 26), font, 0.46,
                    MUTED, 1, cv2.LINE_AA)
        cv2.putText(image, "x1", (self.size // 2 - 10, high + 26), font,
                    0.46, MUTED, 1, cv2.LINE_AA)
        cv2.putText(image, "pi", (high - 12, high + 26), font, 0.46,
                    MUTED, 1, cv2.LINE_AA)
        cv2.putText(image, "+12", (14, low + 6), font, 0.44,
                    MUTED, 1, cv2.LINE_AA)
        cv2.putText(image, "x3", (20, self.size // 2), font, 0.46,
                    MUTED, 1, cv2.LINE_AA)
        cv2.putText(image, "-12", (14, high + 6), font, 0.44,
                    MUTED, 1, cv2.LINE_AA)
        return image

    def render(self, angles, rates):
        image = self._background.copy()
        if len(angles) > 1:
            angle_array = np.asarray(angles)
            points = self._to_pixel(angle_array, np.asarray(rates))
            jumps = np.abs(np.diff(wrap(angle_array))) > np.pi
            for start, end in zip(points[:-1][~jumps], points[1:][~jumps]):
                cv2.line(image, tuple(start), tuple(end), TRACE, 1, cv2.LINE_AA)
            cv2.circle(image, tuple(points[-1]), 6, LIVE, -1, cv2.LINE_AA)
        return image


def status_bar(width, time_s, state, controller):
    image = np.full((BAR, width, 3), BACKDROP, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    stage = "WCLF SWING-UP" if controller.stage == controller.SWING_UP else "LQR BALANCE"
    controller_detail = (
        f"beta = {controller.last_beta:6.3f}   gamma = {controller.last_gamma:6.3f}"
        if controller.stage == controller.SWING_UP
        else "published gain F (eq. 75)"
    )
    columns = (
        (24, [f"t = {time_s:6.2f} s", f"stage = {stage}"]),
        (400, [f"E - E0 = {controller.last_energy_error:+8.3f} J",
               f"x2 = {float(wrap(state[1])):+7.3f} rad"]),
        (760, [f"tau2 = {controller.last_torque:+8.2f} N.m",
               controller_detail]),
    )
    for x, lines in columns:
        for row, line in enumerate(lines):
            cv2.putText(image, line, (x, 36 + 38 * row), font, 0.56,
                        LIVE if (x == 24 and row == 1) else INK, 1, cv2.LINE_AA)
    return image


def build(args) -> int:
    env = DMCContinuousEnv(
        "acrobot",
        "swingup-wclf",
        seed=0,
        raw_state_obs=True,
        dt=args.dt,
        physics_dt=args.dt,
        max_steps=int(round(args.duration / args.dt)) + 1,
        episode_duration=args.duration,
        task_kwargs=dict(
            initial_perturbation=args.initial_perturbation,
            torque_interface=args.torque_interface,
        ),
    )
    physics = env._env.physics
    params = AcrobotParams.from_physics(physics)
    controller = LaiSheController(params, frame="paper")
    panel = PhasePanel()
    stride = max(1, int(round(1.0 / (args.fps * args.dt))))
    tail = int(round(TAIL_SECONDS / args.dt))
    width = 2 * PANEL

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{width}x{PANEL + BAR}",
            "-r", str(args.fps), "-i", "-", "-an", "-vcodec", "libx264",
            "-pix_fmt", "yuv420p", "-crf", "20", args.output,
        ],
        stdin=subprocess.PIPE,
    )

    obs, _ = env.reset(seed=0)
    angles: list[float] = []
    rates: list[float] = []
    frames = 0
    step = 0
    switch_seen = False
    peak_torque = 0.0
    while True:
        state = np.asarray(obs, dtype=np.float64)
        angles.append(float(state[0]))
        rates.append(float(state[2]))
        if len(angles) > tail:
            angles.pop(0)
            rates.pop(0)
        if step % stride == 0:
            scene = physics.render(height=PANEL, width=PANEL, camera_id="fixed")[:, :, ::-1]
            composite = np.vstack(
                [np.hstack([scene, panel.render(angles, rates)]),
                 status_bar(width, float(env.cur_t), state, controller)]
            )
            payload = np.ascontiguousarray(composite).tobytes()
            ffmpeg.stdin.write(payload)
            frames += 1
            if controller.switch_step is not None and not switch_seen:
                switch_seen = True
                for _ in range(int(round(args.hold * args.fps))):
                    ffmpeg.stdin.write(payload)
                    frames += 1
        obs, terminated, truncated = _advance(env, controller(obs))
        peak_torque = max(peak_torque, abs(controller.last_commanded_torque))
        step += 1
        if terminated or truncated:
            break

    ffmpeg.stdin.close()
    return_code = ffmpeg.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit status {return_code}")
    final = np.asarray(obs, dtype=np.float64)
    switch_time = (
        "never" if controller.switch_step is None
        else f"{controller.switch_step * args.dt:.3f} s"
    )
    print(f"wrote {args.output}: {frames} frames at {args.fps} fps")
    print(f"  WCLF -> LQR switch: {switch_time}")
    print(f"  peak commanded torque: {peak_torque:.3f} N.m")
    print(f"  saturation fraction: {controller.saturation_fraction:.6f}")
    print(
        "  final: "
        f"x1={float(wrap(final[0])):+.4f}, x2={float(wrap(final[1])):+.4f}, "
        f"x3={final[2]:+.4f}, x4={final[3]:+.4f}, "
        f"E-E0={params.energy(final) - params.energy_top:+.5f} J"
    )
    return 0


def _advance(env, action):
    result = env.step_dt(action)
    return result[4], result[6], result[7]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--output",
        default="videos/acrobot_lai_she/acrobot_lai_she_lqr_switch.mp4",
    )
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument(
        "--initial-perturbation",
        type=float,
        default=0.2,
        help="documented offset from the paper's otherwise invariant x1=pi start",
    )
    parser.add_argument("--torque-interface", type=float, default=50.0)
    parser.add_argument("--hold", type=float, default=1.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(build(parse_args()))
