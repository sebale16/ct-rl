# common/demonstration.py
"""Shared ``demonstration_policy`` replay-buffer warm start.

Builds the analytical-controller callable that ``algorithms.ct_sac.CTSAC``,
``algorithms.ct_td3.CTTD3``, and (via ``common.sb3_demo``) SB3's SAC/TD3 all
use to seed early rollouts from states an actual swing-up law reaches,
instead of a uniform random walk's states. Originally lived in
``benchmarks/run_ct_rl.py`` as ``_build_demonstration_policy``, gated to
``algo='ct_sac'``; moved here once CTTD3 and the SB3 algorithms grew the
same warm start.
"""

from __future__ import annotations

#: Acrobot-XK is currently the only environment with an analytical
#: demonstration controller wired up.
ACROBOT_XK_ENV_ID = "acrobot-swingup-xk"


def build_demonstration_policy(
    *,
    algo: str,
    env_id: str,
    env_kwargs: dict,
    train_env,
    controller_name: str,
):
    """Build the ``demonstration_policy`` replay-buffer warm start.

    Shared by ``algorithms.ct_sac.CTSAC``, ``algorithms.ct_td3.CTTD3``, and
    SB3's SAC/TD3 (via ``common.sb3_demo``); the imitation-loss term built on
    the same object is CTSAC-only, everyone else only gets the seeding.

    Two sources, both acting on ``acrobot-swingup-xk``'s raw
    ``[q1, q2, qdot1, qdot2]`` observation (the same ``frame="paper"``
    convention the evaluation protocol uses):

    ``controller_name='xin_kaneda'`` -- the analytical Xin-Kaneda swing-up
    law from ``controllers/xin_kaneda.py``, unconditionally.

    ``controller_name='xk_lqr_switch'`` --
    :class:`controllers.acrobot_gated_lyapunov.XKLQRSwitchedController`: the
    same Xin-Kaneda swing-up law, latching one-way to the local LQR feedback
    on first entry to Lai et al.'s equation-(17) attractive region. The
    demonstrations this fills the replay buffer with therefore include the
    balance phase the pure Xin-Kaneda law never reaches on its own.

    ``controller_name='xk_sos_switch'`` --
    :class:`controllers.acrobot_sos_switched.XKSOSSwitchedController`: same
    one-way latch, but the region and the local law are whatever
    ``evaluations.acrobot_sos_roa``'s alternation actually certified (a
    quadratic-Lyapunov sublevel set and a cubic saturated feedback, loaded
    from ``results/acrobot_sos_roa_tau20_certificate.json`` -- a different,
    much smaller region than Lai et al.'s, and not the same law as
    ``xk_lqr_switch``'s plain linear ``-Ke``).

    Gains and torque limit come from the task's own
    ``k_v``/``k_d``/``k_p``/``torque_limit`` when the reward config sets
    them (matching the reward's Vdot term to the controller that generated
    the demonstrations), and from the paper's Section-7 defaults otherwise.
    """
    if algo not in ("ct_sac", "ct_td3", "sac", "td3"):
        raise ValueError(
            "demonstration_controller is only wired for algo in "
            f"('ct_sac', 'ct_td3', 'sac', 'td3'), got algo={algo!r}"
        )
    if controller_name not in ("xin_kaneda", "xk_lqr_switch", "xk_sos_switch"):
        raise ValueError(
            "demonstration_controller must be 'xin_kaneda', 'xk_lqr_switch' "
            f"or 'xk_sos_switch', got {controller_name!r}"
        )
    if env_id != ACROBOT_XK_ENV_ID:
        raise ValueError(
            f"demonstration_controller={controller_name!r} requires env_id="
            f"{ACROBOT_XK_ENV_ID!r}, got {env_id!r}"
        )

    current = train_env.envs[0] if hasattr(train_env, "envs") and train_env.envs else train_env
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "raw_state_obs"):
            break
        current = getattr(current, "env", None)
    if current is None or not getattr(current, "raw_state_obs", False):
        raise ValueError(
            f"demonstration_controller={controller_name!r} requires a "
            "single raw_state_obs=True acrobot-swingup-xk env."
        )

    from controllers.xin_kaneda import AcrobotParams, Gains, XinKanedaController
    from controllers.acrobot_gated_lyapunov import XKLQRSwitchedController
    from controllers.acrobot_sos_switched import XKSOSSwitchedController
    from environment.acrobot_xk import (
        DEFAULT_LYAPUNOV_K_D,
        DEFAULT_LYAPUNOV_K_P,
        DEFAULT_LYAPUNOV_K_V,
        DEFAULT_TORQUE_LIMIT,
    )

    task_kwargs = env_kwargs.get("task_kwargs", {}) or {}
    gains = Gains(
        k_v=float(task_kwargs.get("k_v", DEFAULT_LYAPUNOV_K_V)),
        k_d=float(task_kwargs.get("k_d", DEFAULT_LYAPUNOV_K_D)),
        k_p=float(task_kwargs.get("k_p", DEFAULT_LYAPUNOV_K_P)),
    )
    torque_limit = float(task_kwargs.get("torque_limit", DEFAULT_TORQUE_LIMIT))
    params = AcrobotParams.from_physics(current._env.physics)
    if controller_name == "xk_lqr_switch":
        return XKLQRSwitchedController(params, gains, torque_limit=torque_limit)
    if controller_name == "xk_sos_switch":
        return XKSOSSwitchedController(params, gains, torque_limit=torque_limit)
    return XinKanedaController(params, gains, torque_limit=torque_limit)
