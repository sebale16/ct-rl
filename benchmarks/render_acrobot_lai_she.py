#!/usr/bin/env python
"""Render the Lai--She three-stage Acrobot controller.

The left panel is the MuJoCo plant and the right panel is the shoulder phase
portrait in the ICRA 2006 coordinates (upright ``x1=0``, hanging ``x1=pi``).
The status bar exposes the active controller, energy error, elbow state, and
physical torque.  A short freeze marks each one-way controller switch.

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

from controllers.lai_she import (
    AcrobotParams,
    Design,
    LaiSheController,
    wrap,
    xk_to_paper,
)
from environment.dmc import DMCContinuousEnv


PANEL = 560
BAR = 106
TAIL_SECONDS = 6.0
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
        zero_rate = self._to_pixel(np.array([0.0]), np.array([0.0]))[0]
        cv2.line(image, (low, int(zero_rate[1])), (high, int(zero_rate[1])),
                 (54, 58, 66), 1, cv2.LINE_AA)
        upright = self._to_pixel(np.array([0.0]), np.array([0.0]))[0]
        cv2.line(image, (int(upright[0]), low), (int(upright[0]), high),
                 (54, 58, 66), 1, cv2.LINE_AA)
        cv2.circle(image, tuple(upright), 9, TARGET, 2, cv2.LINE_AA)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, "shoulder phase portrait", (low, low - 30), font,
                    0.62, INK, 1, cv2.LINE_AA)
        cv2.putText(image, "ICRA 2006 coordinates", (low, low - 9), font,
                    0.50, MUTED, 1, cv2.LINE_AA)
        cv2.putText(image, "upright", (int(upright[0]) + 12, int(upright[1]) - 10),
                    font, 0.46, TARGET, 1, cv2.LINE_AA)
        cv2.putText(image, "-pi", (low - 14, high + 26), font, 0.46, MUTED, 1, cv2.LINE_AA)
        cv2.putText(image, "x1", (self.size // 2 - 10, high + 26), font,
                    0.46, MUTED, 1, cv2.LINE_AA)
        cv2.putText(image, "pi", (high - 12, high + 26), font, 0.46, MUTED, 1, cv2.LINE_AA)
        cv2.putText(image, "+12", (14, low + 6), font, 0.44, MUTED, 1, cv2.LINE_AA)
        cv2.putText(image, "x3", (20, self.size // 2), font, 0.46, MUTED, 1, cv2.LINE_AA)
        cv2.putText(image, "-12", (14, high + 6), font, 0.44, MUTED, 1, cv2.LINE_AA)
        return image

    def render(self, angles, rates):
        image = self._background.copy()
        if len(angles) > 1:
            angles_array = np.asarray(angles)
            points = self._to_pixel(angles_array, np.asarray(rates))
            jumps = np.abs(np.diff(wrap(angles_array))) > np.pi
            for start, end in zip(points[:-1][~jumps], points[1:][~jumps]):
                cv2.line(image, tuple(start), tuple(end), TRACE, 1, cv2.LINE_AA)
            cv2.circle(image, tuple(points[-1]), 6, LIVE, -1, cv2.LINE_AA)
        return image


def status_bar(width, time_s, state, controller):
    image = np.full((BAR, width, 3), BACKDROP, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    stage_names = {1: "C1  ENERGY + POSTURE", 2: "C2  SINGULARITY AVOIDANCE", 3: "C3  LQR BALANCE"}
    columns = (
        (24, [f"t = {time_s:6.2f} s", f"stage = {stage_names[controller.stage]}"]),
        (430, [f"E - E0 = {controller.last_energy_error:+8.3f} J",
               f"x2 = {float(wrap(state[1])):+7.3f} rad"]),
        (770, [f"tau2 = {-controller.last_torque:+8.2f} N.m",
               f"r = {controller.last_fuzzy_adjustment:+6.3f}"]),
    )
    for x, lines in columns:
        for row, line in enumerate(lines):
            cv2.putText(image, line, (x, 36 + 38 * row), font, 0.58,
                        LIVE if (x == 24 and row == 1) else INK, 1, cv2.LINE_AA)
    return image


def _advance(env, action):
    result = env.step_dt(action)
    return result[4], result[6], result[7]


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
            damping=0.0,
            torque_limit=args.torque_limit,
            angle_noise=args.angle_noise,
            velocity_noise=0.0,
            # The controller is derived without state caps; keep the shared RL
            # plant's safety terminations out of this analytical demonstration.
            elbow_angle_limit=10_000.0,
            elbow_rate_limit=10_000.0,
            shoulder_rate_scale_limit=10_000.0,
        ),
    )
    physics = env._env.physics
    params = AcrobotParams.from_physics(physics)
    controller = LaiSheController(
        params,
        Design(fuzzy_power_scale=args.fuzzy_power_scale),
        frame="xk",
    )
    panel = PhasePanel()
    stride = max(1, int(round(1.0 / (args.fps * args.dt))))
    tail = int(round(TAIL_SECONDS / args.dt))
    width = 2 * PANEL

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{width}x{PANEL + BAR}",
            "-r", str(args.fps), "-i", "-", "-an", "-vcodec", "libx264",
            "-pix_fmt", "yuv420p", "-crf", "20", args.output,
        ],
        stdin=subprocess.PIPE,
    )

    obs, _ = env.reset(seed=args.seed)
    angles: list[float] = []
    rates: list[float] = []
    frames = 0
    step = 0
    seen_switches = 0
    while True:
        state = xk_to_paper(obs)
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
            if len(controller.switch_log) > seen_switches:
                seen_switches = len(controller.switch_log)
                for _ in range(int(round(args.hold * args.fps))):
                    ffmpeg.stdin.write(payload)
                    frames += 1
        obs, terminated, truncated = _advance(env, controller(obs))
        step += 1
        if terminated or truncated:
            break

    ffmpeg.stdin.close()
    return_code = ffmpeg.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit status {return_code}")
    final = xk_to_paper(obs)
    print(f"wrote {args.output}: {frames} frames at {args.fps} fps")
    print("  switches: " + ", ".join(
        f"C{stage}->C{stage + 1} at {index * args.dt:.3f} s"
        for stage, index in controller.switch_log
    ))
    print(
        "  final: "
        f"x1={float(wrap(final[0])):+.4f}, x2={float(wrap(final[1])):+.4f}, "
        f"x3={final[2]:+.4f}, x4={final[3]:+.4f}, "
        f"E-E0={params.energy(final) - params.energy_top:+.5f} J"
    )
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--output", default="videos/acrobot_lai_she/acrobot_lai_she_lqr_switch.mp4"
    )
    parser.add_argument("--duration", type=float, default=27.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--angle-noise", type=float, default=0.05)
    parser.add_argument("--torque-limit", type=float, default=200.0)
    parser.add_argument("--fuzzy-power-scale", type=float, default=10.0)
    parser.add_argument("--hold", type=float, default=1.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(build(parse_args()))
