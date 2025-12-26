# common/schedules.py
from __future__ import annotations

from typing import Callable, Union

import math

Schedule = Callable[[float], float]
"""
Maps progress_remaining in [1, 0] -> learning rate.
progress_remaining = 1 at start of training, 0 at the end.
"""


def make_schedule(learning_rate: Union[float, Schedule]) -> Schedule:
    """
    If `learning_rate` is a float, returns a constant schedule.
    If callable, returns it as-is.
    """
    if callable(learning_rate):
        return learning_rate
    lr = float(learning_rate)

    def _constant_schedule(progress_remaining: float) -> float:
        return lr

    return _constant_schedule


def linear_schedule(initial_value: float) -> Schedule:
    """
    Linearly decays LR from initial_value at progress=1 to 0 at progress=0.
    """
    initial_value = float(initial_value)

    def _schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return _schedule


def cosine_schedule(initial_value: float, final_ratio: float = 0.0) -> Schedule:
    """
    Cosine LR schedule (optional extra):

        lr(p) = final + 0.5*(initial-final)*(1 + cos(pi * p))

    where p = progress_remaining in [1, 0].

    final_ratio controls final LR = initial_value * final_ratio.
    """
    initial_value = float(initial_value)
    final_value = initial_value * float(final_ratio)

    def _schedule(progress_remaining: float) -> float:
        cos_term = 0.5 * (1.0 + math.cos(math.pi * (1.0 - progress_remaining)))
        return final_value + (initial_value - final_value) * cos_term

    return _schedule
