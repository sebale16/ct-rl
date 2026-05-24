# evaluations/stability_metrics.py
"""Stability / fall metrics for DMC humanoid evaluation.

The aggregate DMC return conflates "stood still without falling" with
"actually walked." For humanoid this is misleading — a policy that
balances upright but never steps forward can score similarly to one
that locomotes. These helpers expose per-episode survival and fall
metrics derived from the underlying MuJoCo physics state.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np


# dm_control humanoid's target stand height is 1.4 m. A head height below
# ~1.0 m corresponds to the agent stooping or sitting; below ~0.8 m it is
# essentially lying down. We use 1.0 m as the default fall threshold.
DEFAULT_HEAD_HEIGHT_THRESHOLD = 1.0


def get_humanoid_head_height(env) -> Optional[float]:
    """Return current head height for a DMC humanoid env, else None."""
    inner = getattr(env, "_env", None)
    if inner is None:
        return None
    phys = getattr(inner, "physics", None)
    if phys is None or not hasattr(phys, "head_height"):
        return None
    try:
        return float(phys.head_height())
    except Exception:
        return None


def make_probe_fn(env_id: str) -> Optional[Callable[[Any], Optional[float]]]:
    """Return a probe callable for the given env, or None if unsupported."""
    if env_id.startswith("humanoid"):
        return get_humanoid_head_height
    return None


def compute_episode_stability(
    *,
    heights: List[Optional[float]],
    timestamps: List[float],
    threshold: float = DEFAULT_HEAD_HEIGHT_THRESHOLD,
) -> Dict[str, Any]:
    """Per-episode survival/fall metrics from per-step head heights.

    - fell: True iff head height drops below `threshold` at any step.
    - survival_fraction: time-weighted fraction of episode with head height
                         at or above `threshold`.
    - time_to_first_fall: timestamp of first below-threshold step, else None.
    - min_head_height / mean_head_height: summary stats.
    """
    clean = [(t, h) for t, h in zip(timestamps, heights) if h is not None]
    if not clean:
        return {
            "fell": False,
            "survival_fraction": float("nan"),
            "time_to_first_fall": None,
            "min_head_height": float("nan"),
            "mean_head_height": float("nan"),
        }

    t = np.asarray([p[0] for p in clean], dtype=float)
    h = np.asarray([p[1] for p in clean], dtype=float)
    below = h < threshold
    fell = bool(below.any())

    if len(t) > 1:
        dts = np.diff(t, prepend=t[0] - (t[1] - t[0]))
        dts = np.clip(dts, 0.0, None)
        total = float(dts.sum())
        survival_fraction = (
            float(dts[~below].sum() / total) if total > 0 else float((~below).mean())
        )
    else:
        survival_fraction = float((~below).mean())

    time_to_first_fall = float(t[int(np.argmax(below))]) if fell else None

    return {
        "fell": fell,
        "survival_fraction": survival_fraction,
        "time_to_first_fall": time_to_first_fall,
        "min_head_height": float(h.min()),
        "mean_head_height": float(h.mean()),
    }


def aggregate_stability(per_episode: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-episode metrics into population-level stats."""
    if not per_episode:
        return {
            "n_episodes": 0,
            "fall_rate": float("nan"),
            "mean_survival_fraction": float("nan"),
            "mean_time_to_first_fall": float("nan"),
            "mean_min_head_height": float("nan"),
            "mean_mean_head_height": float("nan"),
        }

    def _nanmean(vals: List[float]) -> float:
        arr = np.asarray([v for v in vals if v is not None and not np.isnan(v)])
        return float(arr.mean()) if arr.size else float("nan")

    fall_rate = float(np.mean([1.0 if m["fell"] else 0.0 for m in per_episode]))
    ttf_vals = [
        m["time_to_first_fall"]
        for m in per_episode
        if m["time_to_first_fall"] is not None
    ]
    return {
        "n_episodes": len(per_episode),
        "fall_rate": fall_rate,
        "mean_survival_fraction": _nanmean(
            [m["survival_fraction"] for m in per_episode]
        ),
        "mean_time_to_first_fall": (
            float(np.mean(ttf_vals)) if ttf_vals else float("nan")
        ),
        "mean_min_head_height": _nanmean([m["min_head_height"] for m in per_episode]),
        "mean_mean_head_height": _nanmean([m["mean_head_height"] for m in per_episode]),
    }
