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
from .acrobot_gated_lyapunov import (
    AttractiveRegion,
    GatedLyapunov,
    LQRDesign,
    NonsmoothLyapunov,
    XKLQRSwitchedController,
    lqr_switch_residual,
    max_local_value_on_region,
    plant_drift_and_gain,
    riccati_feedback,
    upright_error,
)
from .lai_she import (
    AcrobotParams as LaiSheAcrobotParams,
    Design as LaiSheDesign,
    LaiSheController,
    paper_to_xk as lai_she_paper_to_xk,
    xk_to_paper as xk_to_lai_she,
)

__all__ = [
    "AcrobotParams",
    "Gains",
    "XinKanedaController",
    "hanging_jacobian",
    "hanging_regime",
    "homoclinic_speed",
    "obs_to_paper",
    "AttractiveRegion",
    "GatedLyapunov",
    "LQRDesign",
    "NonsmoothLyapunov",
    "XKLQRSwitchedController",
    "lqr_switch_residual",
    "max_local_value_on_region",
    "plant_drift_and_gain",
    "riccati_feedback",
    "upright_error",
    "LaiSheAcrobotParams",
    "LaiSheDesign",
    "LaiSheController",
    "lai_she_paper_to_xk",
    "xk_to_lai_she",
]
