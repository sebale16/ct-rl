"""Render every tip-height/velocity curriculum stage side by side.

The Acrobot v4.3 and v6.1 tasks share the same reset ladder, so one Acrobot
panel represents both reward branches.  The second panel shows the serial
double-linked CartPole v2 ladder.  Each stage first holds the exact reset pose,
then runs a zero-action preview to make the initial velocity (or lack of it)
visible.

Run from the repository root:

    MUJOCO_GL=egl python -m benchmarks.render_tip_curriculum_stages
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

# dm_control selects its OpenGL backend at import time.  EGL works on headless
# render nodes while an explicitly supplied backend still takes precedence.
os.environ.setdefault("MUJOCO_GL", "egl")

from environment.dmc import DMCContinuousEnv  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "videos" / "tip_curriculum_stages.mp4"
BACKGROUND = (12, 19, 29)
TEXT = (245, 247, 250)
MUTED_TEXT = (184, 197, 211)
ACCENT = (55, 211, 229)
INITIAL_ACCENT = (255, 184, 77)


def _make_env(
    domain: str,
    task: str,
    *,
    seed: int,
    fps: int,
) -> DMCContinuousEnv:
    return DMCContinuousEnv(
        domain_name=domain,
        task_name=task,
        seed=seed,
        time_sampling="uniform",
        dt=1.0 / fps,
        physics_dt=0.002,
        episode_duration=10.0,
        task_kwargs={"curriculum": True},
    )


def _put_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 1,
) -> None:
    """Draw readable anti-aliased text on an RGB frame."""

    font = cv2.FONT_HERSHEY_SIMPLEX
    shadow_origin = (origin[0] + 1, origin[1] + 1)
    cv2.putText(
        frame,
        text,
        shadow_origin,
        font,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        origin,
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _shade_band(
    frame: np.ndarray,
    top: int,
    bottom: int,
    *,
    opacity: float,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, top),
        (frame.shape[1], bottom),
        BACKGROUND,
        thickness=-1,
    )
    frame[:] = cv2.addWeighted(overlay, opacity, frame, 1.0 - opacity, 0.0)


def _metric_line(name: str, metrics: dict[str, float]) -> str:
    return (
        f"{name}   tip h={metrics['start_tip_height']:+.2f} m   "
        f"potential={metrics['start_potential_energy_norm']:.3f}   "
        f"start speed={metrics['start_tip_speed']:.2f} m/s"
    )


def _annotate(
    frame: np.ndarray,
    *,
    stage: int,
    num_stages: int,
    phase: str,
    acrobot_metrics: dict[str, float],
    cartpole_metrics: dict[str, float],
    panel_width: int,
) -> np.ndarray:
    result = np.ascontiguousarray(frame.copy())
    height, width = result.shape[:2]
    _shade_band(result, 0, 62, opacity=0.88)
    _shade_band(result, height - 91, height, opacity=0.88)

    initial_stage = stage == 0
    final_stage = stage == num_stages - 1
    if initial_stage:
        stage_kind = "UPRIGHT REST"
        subtitle = "Exact upright vertical pose at zero starting velocity"
    elif final_stage:
        stage_kind = "FINAL HANGING"
        subtitle = "Exact hanging state at zero velocity"
    else:
        stage_kind = "REST START"
        subtitle = "Zero starting velocity; mastery unlocks a lower tip height"
    accent = INITIAL_ACCENT if initial_stage else ACCENT
    _put_text(
        result,
        f"TIP CURRICULUM  |  STAGE {stage + 1}/{num_stages}  |  {stage_kind}",
        (17, 27),
        scale=0.63,
        color=accent,
        thickness=2,
    )
    _put_text(
        result,
        subtitle,
        (17, 51),
        scale=0.43,
        color=MUTED_TEXT,
    )
    _put_text(
        result,
        phase,
        (width - 307, 31),
        scale=0.48,
        color=TEXT,
        thickness=2,
    )

    _put_text(
        result,
        "ACROBOT v4.3 / v6.1",
        (16, 87),
        scale=0.49,
        color=TEXT,
        thickness=2,
    )
    _put_text(
        result,
        "two-link arm; elbow extended at reset",
        (16, 106),
        scale=0.34,
        color=MUTED_TEXT,
    )
    _put_text(
        result,
        "DOUBLE-LINKED CARTPOLE v2",
        (panel_width + 16, 87),
        scale=0.49,
        color=TEXT,
        thickness=2,
    )
    _put_text(
        result,
        "serial two-link chain; relative elbow = 0 deg at reset",
        (panel_width + 16, 106),
        scale=0.34,
        color=MUTED_TEXT,
    )
    cv2.line(
        result,
        (panel_width, 63),
        (panel_width, height - 92),
        (220, 226, 232),
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    _put_text(
        result,
        _metric_line("Acrobot", acrobot_metrics),
        (17, height - 61),
        scale=0.39,
        color=TEXT,
    )
    _put_text(
        result,
        _metric_line("CartPole", cartpole_metrics),
        (panel_width + 17, height - 61),
        scale=0.39,
        color=TEXT,
    )
    _put_text(
        result,
        "potential = normalized gravitational potential of the selected reset",
        (17, height - 37),
        scale=0.36,
        color=MUTED_TEXT,
    )

    progress = float(acrobot_metrics["progress"])
    bar_left, bar_right = 17, width - 17
    bar_top, bar_bottom = height - 22, height - 10
    cv2.rectangle(
        result,
        (bar_left, bar_top),
        (bar_right, bar_bottom),
        (73, 87, 104),
        thickness=-1,
    )
    fill_right = bar_left + int(round(progress * (bar_right - bar_left)))
    if fill_right > bar_left:
        cv2.rectangle(
            result,
            (bar_left, bar_top),
            (fill_right, bar_bottom),
            accent,
            thickness=-1,
        )
    _put_text(
        result,
        f"mastery progress {progress:.0%}",
        (width - 190, height - 32),
        scale=0.34,
        color=MUTED_TEXT,
    )
    return result


def _transcode_h264(intermediate: Path, output: Path) -> None:
    """Convert OpenCV's broadly available MP4V output to browser-safe H.264."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        intermediate.replace(output)
        print("ffmpeg not found; kept OpenCV MP4V encoding.", flush=True)
        return
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(intermediate),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    intermediate.unlink()


def render_demo(
    output: Path,
    *,
    panel_width: int = 640,
    height: int = 480,
    fps: int = 50,
    hold_seconds: float = 0.6,
    motion_seconds: float = 1.4,
    duration_scale: float = 1.0,
    reset_seed: int = 17,
) -> None:
    if panel_width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("panel_width, height, and fps must be positive")
    if panel_width % 2 or height % 2:
        raise ValueError("panel_width and height must be even")
    if hold_seconds <= 0.0 or motion_seconds <= 0.0:
        raise ValueError("hold_seconds and motion_seconds must be positive")
    if duration_scale <= 0.0:
        raise ValueError("duration_scale must be positive")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    intermediate = output.with_name(f".{output.stem}.opencv{output.suffix}")
    if intermediate.exists():
        intermediate.unlink()

    acrobot = _make_env(
        "acrobot",
        "swingup-v4.3",
        seed=reset_seed,
        fps=fps,
    )
    cartpole = _make_env(
        "cartpole",
        "two_poles-v2",
        seed=reset_seed,
        fps=fps,
    )
    envs = (acrobot, cartpole)
    writer = None
    frame_count = 0
    try:
        stage_counts = {env.num_curriculum_stages for env in envs}
        if None in stage_counts or len(stage_counts) != 1:
            raise RuntimeError(
                f"curriculum stage counts do not match: {stage_counts}"
            )
        num_stages = int(next(iter(stage_counts)))
        hold_frames = max(
            1, int(round(hold_seconds * duration_scale * fps))
        )
        motion_frames = max(
            1, int(round(motion_seconds * duration_scale * fps))
        )

        writer = cv2.VideoWriter(
            str(intermediate),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (2 * panel_width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(
                "OpenCV could not open the MP4 writer; check FFmpeg support"
            )

        zero_actions = tuple(
            np.zeros(env.action_space.shape, dtype=np.float32) for env in envs
        )
        for stage in range(num_stages):
            for env in envs:
                env.set_curriculum_stage(stage)
                # Reusing the seed keeps the sampled mirror side consistent
                # between levels, while stage 6 remains canonical hanging.
                env.reset(seed=reset_seed)

            acrobot_metrics = acrobot.curriculum_log_metrics()
            cartpole_metrics = cartpole.curriculum_log_metrics()
            for stage_frame in range(hold_frames + motion_frames):
                holding = stage_frame < hold_frames
                if not holding:
                    for env, action in zip(envs, zero_actions):
                        _, _, terminated, truncated, _ = env.step(action)
                        if terminated or truncated:
                            raise RuntimeError(
                                "demo environment ended inside a stage preview"
                            )

                rendered = [
                    env.render(
                        mode="rgb_array",
                        width=panel_width,
                        height=height,
                        camera_id=0,
                    )
                    for env in envs
                ]
                combined = np.hstack(rendered)
                phase = (
                    "EXACT RESET  |  PAUSED"
                    if holding
                    else "ZERO-ACTION RELEASE (NO POLICY)"
                )
                combined = _annotate(
                    combined,
                    stage=stage,
                    num_stages=num_stages,
                    phase=phase,
                    acrobot_metrics=acrobot_metrics,
                    cartpole_metrics=cartpole_metrics,
                    panel_width=panel_width,
                )
                writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                frame_count += 1
    finally:
        if writer is not None:
            writer.release()
        for env in envs:
            env.close()

    _transcode_h264(intermediate, output)
    print(
        f"Wrote {output} ({frame_count} frames, "
        f"{frame_count / fps:.2f}s at {fps} fps).",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--hold-seconds", type=float, default=0.6)
    parser.add_argument("--motion-seconds", type=float, default=1.4)
    parser.add_argument(
        "--duration-scale",
        type=float,
        default=1.0,
        help="Scale both phase durations; values below 1 are useful for smoke tests.",
    )
    parser.add_argument("--reset-seed", type=int, default=17)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render_demo(
        args.output,
        panel_width=args.panel_width,
        height=args.height,
        fps=args.fps,
        hold_seconds=args.hold_seconds,
        motion_seconds=args.motion_seconds,
        duration_scale=args.duration_scale,
        reset_seed=args.reset_seed,
    )
