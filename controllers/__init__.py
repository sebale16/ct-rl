"""Analytical (non-learned) controllers used as references for the RL arms."""

from .xin_kaneda import (
    AcrobotParams,
    Gains,
    XinKanedaController,
    hanging_jacobian,
    hanging_regime,
    homoclinic_speed,
    obs_to_paper,
)

__all__ = [
    "AcrobotParams",
    "Gains",
    "XinKanedaController",
    "hanging_jacobian",
    "hanging_regime",
    "homoclinic_speed",
    "obs_to_paper",
]
