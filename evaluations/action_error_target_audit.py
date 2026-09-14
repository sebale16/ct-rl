#!/usr/bin/env python
"""What does an action error do to the CT-SAC critic target, as a function of dt?

``docs/ct_sac_advantage_parameterization.md`` makes a quantitative claim about
where CT-SAC's continuous-time content goes.  Restated as something measurable:

  * the object CT-SAC's theory names is the advantage-RATE
    ``q_V(s,a) = r + (L^a V)(s) - beta V(s)``, which is O(1) and carries no dt;
  * the object it actually stores is ``Q``, and the target that trains it puts
    the rate back down by the target reference interval ``T``:
        y(s,a) = T r(s,a) + V(s) + T * (exp(-beta dt) V(s') - V(s)) / dt
    so the ACTION-dependence inside the target is ``T * range_a q_V``;
  * the critic's own approximation error does not shrink with ``T``, so the
    signal-to-error ratio falls linearly in ``T``.

This audit measures exactly those three quantities on trained critics, holding
everything else fixed, and reports them per (arm, seed):

    range_y      mean over states of  max_a y(s,a) - min_a y(s,a)
                 -- the action signal the critic is asked to store
    range_qv     range_y / T
                 -- the same signal expressed as a RATE.  The doc's claim is
                    that THIS is dt-invariant and O(1); range_y is not.
    range_q      mean over states of  max_a Q_phi(s,a) - min_a Q_phi(s,a)
                 -- what the trained critic actually expresses over the action
                    range (the actor's whole view of the action)
    rms_shape_err  RMS over (state, action) of the ACTION-SHAPE residual
                 (Q_phi - mean_a Q_phi) - (y - mean_a y)
                 -- the critic's error in representing the action dependence,
                    with the per-state offset (which the actor cannot see)
                    removed.  This is the noise floor the signal competes with.
    snr          range_y / rms_shape_err
    spearman     mean per-state rank correlation between Q_phi(.,a) and y(.,a)
                 over the action grid -- does the critic order actions right?
    delta_y      mean over states of  y(s, a*) - y(s, clip(a* + DELTA))
                 for the analytical controller action a* and a fixed torque
                 error DELTA -- "what an error in the action does to the target"
                 in the doc's own units, with its paired t statistic.

V(s) is read exactly as the trainer reads it (``CTSAC._state_value``): from the
V-head when the run has one, otherwise as the sampled soft expectation
``E_{a~pi}[min-Q_target - price(alpha) log pi]``.  For the sampled path we use
the policy MEAN action (``deterministic=True``) so V is a deterministic
function of the state and the Monte-Carlo noise of a 1-sample read cannot be
confused with the signal being measured; ``--stochastic-value`` switches to an
n-sample stochastic read for comparison.

State sets are physical (qpos, qvel) snapshots and therefore shared verbatim by
every arm, so a 1 ms and a 10 ms critic are compared at the SAME states:

    tube      states the analytical Xin-Kaneda controller visits inside the
              homoclinic tube (acrobot-XK only) -- the doc's setting
    onpolicy  states the seed's own trained policy visits

Usage::

    python -m evaluations.action_error_target_audit --env acrobot-swingup-xk \
        --out results/action_error_target_audit_acrobot
    python -m evaluations.action_error_target_audit --env cheetah-run \
        --out results/action_error_target_audit_cheetah
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch as th
from scipy import stats

os.environ.setdefault("MUJOCO_GL", "egl")

from algorithms.ct_sac import CTSAC
from benchmarks.run_ct_rl import (
    _pop_structured_model_kwargs,
    _select_structured_dof_layout,
    make_ct_env,
)
from common.utils import load_ct_hyperparams_from_table
from models import ActorQCriticModel
from models.port_hamiltonian import DOFLayout, PortHamiltonianModel

HYPERPARAMS_DIR = "benchmarks/hyperparams"
SAVED = "saved_models/ct_sac"

# --------------------------------------------------------------------------
# Arms.  ``run`` is the run-id directory a chain wrote under seed_<n>/.
# ``label`` is what the report groups by; ``T_nominal`` is only a cross-check
# against the value the algorithm reports for itself.
# --------------------------------------------------------------------------
ARMS = {
    "acrobot-swingup-xk": [
        dict(
            label="1ms/imit-held",
            mode="xk_r3_eta0p23_fixed1ms_h2s_temp0p01_xkdot_q2dot4pi_logrecip"
            "_xkklrev_xkdemo200k_tau1p25e3_anneal0p5span600k",
            run="dt_0_001_maxs_20000_mseimit_v1",
            seeds=(0, 1, 2, 3, 4, 5),
            T_nominal=0.001,
        ),
        dict(
            label="1ms/imit-annealed-to-0",
            mode="xk_r3_eta0p23_fixed1ms_h2s_temp0p01_xkdot_q2dot4pi_logrecip"
            "_xkklrev_xkdemo200k_tau1p25e3_anneal0span600k",
            run="dt_0_001_maxs_20000_mseimit_v1",
            seeds=(0, 1, 2, 3, 4, 5),
            T_nominal=0.001,
        ),
        dict(
            label="10ms/no-imit",
            mode="xk_r3_eta0p23_ctrl10ms_h2s_temp0p01_xkdot_q2dot4pi_logrecip"
            "_tau1p25e2",
            run="dt_0_01_maxs_2000_nokl10ms_v1",
            seeds=(0, 1, 2, 3, 4, 5),
            T_nominal=0.01,
        ),
        dict(
            label="10ms/no-imit+demo",
            mode="xk_r3_eta0p23_ctrl10ms_h2s_temp0p01_xkdot_q2dot4pi_logrecip"
            "_xkdemo20k_tau1p25e2",
            run="dt_0_01_maxs_2000_nokl10ms_v1",
            seeds=(0, 1, 2, 3, 4, 5),
            T_nominal=0.01,
        ),
        # A matched horizon pair: same imitation loss (MSE), same anneal, same
        # tau, differing in the discount rate -- 2 s against 10 s.  The target's
        # action-dependence splits into a reward term that does not depend on
        # the horizon and a value term whose local slope scales like
        # lambda/(lambda + mu), so a shorter horizon should shift the ordering
        # onto the exactly-computed reward term.
        dict(
            label="1ms/h2s (lam=0.5)",
            mode="xk_r3_eta0p23_fixed1ms_h2s_temp0p01_xkdot_q2dot4pi_logrecip"
            "_xkmse_xkdemo200k_tau1p25e3_anneal0span600k",
            run=None, seeds=None, T_nominal=0.001,
        ),
        dict(
            label="1ms/h10s (lam=0.1)",
            mode="xk_r3_eta0p26_fixed1ms_h10s_temp0p01_xkdot_q2dot4pi_logrecip"
            "_xkmse_xkdemo200k_tau1p25e3_anneal0span600k",
            run=None, seeds=None, T_nominal=0.001,
        ),
        # Trained with the model-based generator (oracle drift + V-head), so
        # their critic target never differenced V across sampled next states.
        dict(
            label="10ms/mbq-vhead+demo",
            mode="xk_r3_eta0p23_ctrl10ms_h2s_temp0p01_xkdot_q2dot4pi_logrecip"
            "_xkdemo20k_tau1p25e2_mbqvhead",
            run=None,
            seeds=None,
            T_nominal=0.01,
        ),
        dict(
            label="10ms/mbq-vhead+anneal0",
            mode="xk_r3_eta0p23_ctrl10ms_h2s_temp0p01_xkdot_q2dot4pi_logrecip"
            "_xkklrev_xkdemo20k_tau1p25e2_anneal0span60k_mbqvhead",
            run=None,
            seeds=None,
            T_nominal=0.01,
        ),
    ],
    # No 1 ms run exists for these -- every saved cheetah/cartpole/humanoid
    # chain is at the same control interval.  They are audited anyway so the
    # signal/noise ledger can be read in environments where CT-SAC does train
    # without an imitation term, at a dt where the doc predicts it should.
    # V-head modes are preferred here because V is then a clean network read and
    # the tuned temperature (absent from the older chains' checkpoints) does not
    # enter the value at all.
    "cheetah-run": [
        dict(
            label="10ms/mbq-vhead",
            mode="mbq_vhead_quad",
            run=None,
            seeds=None,
            T_nominal=0.01,
        ),
    ],
    "cartpole-swingup": [
        dict(
            label="10ms/mf-vhead",
            mode="final_mf_vhead",
            run=None,
            seeds=None,
            T_nominal=0.01,
        ),
        dict(label="10ms/mf", mode="final_mf", run=None, seeds=None, T_nominal=0.01),
    ],
    "humanoid-walk": [
        dict(
            label="25ms/oracle-vhead",
            mode="oracle_buf1m",
            run=None,
            seeds=None,
            T_nominal=0.025,
        ),
    ],
}

# Torque error swept in the doc: 5.8 N.m against a 20 N.m torque limit.
DELTA_NORMALIZED = 0.29


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------
def fix_interval(env_kwargs: dict) -> dict:
    """Force every step to take the nominal control interval.

    Cheetah, cartpole and humanoid train with ``time_sampling='irregular'``:
    each transition draws its own duration from [min_dt, max_dt], with a
    tail-heavy distribution.  That is a genuine feature of those rows, but it is
    not a duration this measurement can attribute an action signal to -- a
    2 ms step and a 30 ms step from the same state carry different amounts of
    it.  Pinning the interval to the row's nominal ``dt`` (which is also the
    target reference interval these rows use) puts every environment on the
    same footing as acrobot-XK's fixed-interval rows.  The loaded weights are
    untouched either way.
    """
    fixed = dict(env_kwargs)
    fixed["time_sampling"] = "uniform"
    fixed.pop("min_dt", None)
    fixed.pop("max_dt", None)
    fixed.pop("time_sampling_kwargs", None)
    return fixed


def build_algorithm(env_id: str, mode: str, seed: int = 0, fixed_interval: bool = True):
    """CTSAC + env for one hyperparameter row, mirroring run_ct_rl's wiring."""
    _, env_kwargs, model_kwargs, algo_kwargs, _ = load_ct_hyperparams_from_table(
        "ct_sac", env_id, mode, hyperparams_dir=HYPERPARAMS_DIR
    )
    env_kwargs.pop("n_envs", None)
    env_kwargs.pop("eval_n_envs", None)
    if fixed_interval:
        env_kwargs = fix_interval(env_kwargs)
    env = make_ct_env(env_id=env_id, seed=seed, env_kwargs=dict(env_kwargs))

    # Warm-start seeding and the imitation term act only on training-time
    # action sampling and the actor loss; neither touches the weights being
    # audited.  Drop them rather than rebuilding run_ct_rl's controller wiring,
    # which would otherwise be required just to satisfy the constructor.
    for key in (
        "demonstration_controller",
        "demonstration_steps",
        "imitation_coef",
        "imitation_coef_final",
        "imitation_anneal_steps",
        "imitation_decay_steps",
        "imitation_direction",
        "imitation_sigma",
        "imitation_loss_type",
    ):
        algo_kwargs.pop(key, None)

    contact_force = int(
        str(algo_kwargs.pop("dynamics_contact_force", "") or "").strip() or 0
    )
    structured_model_kwargs = _pop_structured_model_kwargs(algo_kwargs)
    if str(algo_kwargs.get("use_model_based_q", "")).strip().lower() in (
        "1", "true", "yes",
    ):
        # Mirror run_ct_rl's branch on dynamics_source.  An oracle row wants the
        # mujoco drift and no DOF layout at all; only the structured rows need a
        # layout, and only some domains have one.
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        intensity = float(algo_kwargs.get("human_input_intensity", 0.0) or 0.0)
        source = str(algo_kwargs.get("dynamics_source", "mujoco")).strip()
        if source == "mujoco":
            algo_kwargs["dynamics_model"] = PortHamiltonianModel(
                obs_dim, act_dim, mode="mujoco",
                drift_fn=env.dynamics_terms, human_input_intensity=intensity,
            )
        elif source == "phast":
            algo_kwargs["dynamics_model"] = PortHamiltonianModel(
                obs_dim, act_dim, mode="phast", human_input_intensity=intensity,
            )
        else:
            algo_kwargs["dynamics_model"] = PortHamiltonianModel(
                obs_dim, act_dim, mode="structured",
                human_input_intensity=intensity, contact_force=contact_force,
                dof_layout=_select_structured_dof_layout(env, obs_dim, DOFLayout),
                **structured_model_kwargs,
            )
    algo = CTSAC(
        env=env, model=ActorQCriticModel, model_kwargs=model_kwargs, seed=seed,
        **algo_kwargs,
    )
    return algo, env


def resolve_run_dir(env_id: str, arm: dict, seed: int) -> str | None:
    base = os.path.join(SAVED, env_id, arm["mode"], f"seed_{seed}")
    if not os.path.isdir(base):
        return None
    if arm.get("run"):
        d = os.path.join(base, arm["run"])
        return d if os.path.isdir(d) else None
    runs = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    return os.path.join(base, runs[-1]) if runs else None


def discover_seeds(env_id: str, arm: dict) -> list[int]:
    if arm.get("seeds"):
        return list(arm["seeds"])
    base = os.path.join(SAVED, env_id, arm["mode"])
    if not os.path.isdir(base):
        return []
    return sorted(
        int(d.split("_")[1]) for d in os.listdir(base) if d.startswith("seed_")
    )


def latest_step_checkpoint(run_dir: str) -> str | None:
    """Highest-numbered ``*_<steps>_steps.pth`` in a run directory."""
    best, best_n = None, -1
    for name in os.listdir(run_dir):
        if not name.endswith("_steps.pth"):
            continue
        try:
            n = int(name[: -len("_steps.pth")].rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if n > best_n:
            best, best_n = os.path.join(run_dir, name), n
    return best


def load_seed(algo: CTSAC, run_dir: str, which: str):
    """Load one seed's weights.

    Chains that ended cleanly leave ``checkpoint/model.pth`` plus a
    ``train_state.pt`` holding the tuned temperature; older ones left only the
    periodic ``*_steps.pth`` files, so fall back to the newest of those.  When
    the run has no V-head, alpha enters V through the entropy price and an
    unknown alpha is reported rather than silently replaced.

    Returns ``(alpha_tensor, path, alpha_value_or_None, steps_or_None)``.
    """
    best = os.path.join(run_dir, "best_model", "best_model.pth")
    if which == "final":
        path = os.path.join(run_dir, "checkpoint", "model.pth")
        if not os.path.exists(path):
            path = latest_step_checkpoint(run_dir) or os.path.join(
                run_dir, "final_model.pth"
            )
        if not os.path.exists(path) and os.path.exists(best):
            # Some older chains kept only the best-evaluation snapshot.  Fall
            # back to it rather than dropping the arm; every row records the
            # checkpoint path it actually read.
            path = best
    else:
        path = best
    if not path or not os.path.exists(path):
        return None
    algo.load(path)

    state_path = os.path.join(run_dir, "checkpoint", "train_state.pt")
    alpha, steps = None, None
    if os.path.exists(state_path):
        ts = th.load(state_path, map_location="cpu", weights_only=False)
        counters = ts.get("counters", {})
        alpha = counters.get("alpha")
        steps = counters.get("num_timesteps")
        algo._value_updates = max(
            int(counters.get("_value_updates", 0) or 0), algo._value_updates
        )
        if algo.log_alpha is not None and "log_alpha" in ts:
            algo.log_alpha.data.copy_(
                th.as_tensor(ts["log_alpha"]).to(algo.log_alpha.device)
            )
    if algo.use_value_head:
        # A trained V-head is what the target read; without the train_state the
        # warmup counter is zero and _state_value would silently fall back to
        # the sampled expectation instead.
        algo._value_updates = max(algo._value_updates, algo.value_warmup)
    alpha_t = (
        th.exp(algo.log_alpha.detach())
        if algo.log_alpha is not None
        else algo.alpha_tensor
    )
    return alpha_t, path, alpha, steps


# --------------------------------------------------------------------------
# physical state handling
# --------------------------------------------------------------------------
def physics_of(env):
    return env._env.physics


def snapshot(env):
    p = physics_of(env)
    return p.data.qpos.copy(), p.data.qvel.copy()


def restore(env, qpos, qvel):
    """Place the env at a physical state and make it steppable from there.

    The env is reset first and only then overwritten: a dm_control environment
    swallows the first ``step`` after construction or after a LAST timestep and
    silently re-randomizes instead, so stepping a never-reset env measures a
    random state rather than the one asked for.  Resetting also rewinds the
    uniform time grid, so the next step is handed the arm's own control
    interval, and refreshes both observation caches so the observation the
    critic is queried at is the one the env would report.
    """
    from environment.dmc import _flatten_obs

    env.reset()
    p = physics_of(env)
    with p.reset_context():
        p.data.qpos[:] = qpos
        p.data.qvel[:] = qvel
    env._step_index = 0
    env.cur_t = 0.0
    if env.raw_state_obs:
        obs = env._raw_obs()
    else:
        obs = _flatten_obs(env._env.task.get_observation(p))
    obs = np.asarray(obs, dtype=np.float32)
    env._last_obs = obs
    env._last_obs_dmc = obs
    return obs


def _check_interval(dt_used, expected, tol=1e-9):
    """Guard against silently measuring a different control interval."""
    if abs(dt_used - expected) > tol:
        raise RuntimeError(
            f"step advanced {dt_used:g}s, expected {expected:g}s -- the env is "
            "not stepping at the interval this measurement assumes"
        )


def one_step(env, qpos, qvel, action):
    """Apply ``action`` for exactly one control interval from (qpos, qvel).

    Returns ``(obs, reward, next_obs, terminated, dt_used)``; ``dt_used`` is the
    physical time the step actually advanced, which every caller checks against
    the interval it believes it is measuring.
    """
    obs = restore(env, qpos, qvel)
    _, _, _, reward, next_obs, _, term, _, _ = env.step_dt(
        np.asarray(action, dtype=np.float32)
    )
    return (
        obs,
        float(reward),
        np.asarray(next_obs, dtype=np.float32),
        bool(term),
        float(env.cur_t),
    )


# --------------------------------------------------------------------------
# state sets
# --------------------------------------------------------------------------
def tube_states(env_id: str, n: int, seeds=(20000, 20001, 20002, 20003)):
    """In-tube physical states visited by the analytical Xin-Kaneda controller.

    Uses the same protocol as ``.claude_scratch/advantage_n185.py`` -- a 1 ms
    probe env, the analytical controller, states kept only while the
    homoclinic-capture flag is set -- so the sample spans the tube rather than
    one trajectory through it.  The result is a list of (qpos, qvel) pairs and
    is therefore independent of any arm's control interval.
    """
    from controllers.xin_kaneda import AcrobotParams, Gains, XinKanedaController
    from environment.acrobot_xk import (
        DEFAULT_LYAPUNOV_K_D,
        DEFAULT_LYAPUNOV_K_P,
        DEFAULT_LYAPUNOV_K_V,
        DEFAULT_TORQUE_LIMIT,
    )
    from environment.dmc import DMCContinuousEnv

    _, env_kwargs, _, _, _ = load_ct_hyperparams_from_table(
        "ct_sac",
        env_id,
        ARMS[env_id][0]["mode"],
        hyperparams_dir=HYPERPARAMS_DIR,
    )
    task_kwargs = dict(env_kwargs.get("task_kwargs", {}) or {})
    task_kwargs.update(uniform_start=False, paper_start=False, release_start=True)
    gains = Gains(
        k_v=float(task_kwargs.get("k_v", DEFAULT_LYAPUNOV_K_V)),
        k_d=float(task_kwargs.get("k_d", DEFAULT_LYAPUNOV_K_D)),
        k_p=float(task_kwargs.get("k_p", DEFAULT_LYAPUNOV_K_P)),
    )
    torque_limit = float(task_kwargs.get("torque_limit", DEFAULT_TORQUE_LIMIT))
    probe = DMCContinuousEnv(
        domain_name="acrobot",
        task_name="swingup-xk",
        seed=seeds[0],
        raw_state_obs=True,
        time_sampling="uniform",
        dt=0.001,
        physics_dt=0.001,
        max_steps=20000,
        episode_duration=20.0,
        return_reward_increment=False,
        task_kwargs=task_kwargs,
    )
    ctrl = XinKanedaController(
        AcrobotParams.from_physics(probe._env.physics), gains,
        torque_limit=torque_limit,
    )
    pool = []
    for episode_seed in seeds:
        obs, _ = probe.reset(seed=episode_seed)
        obs = np.asarray(obs, dtype=np.float32)
        ctrl.reset()
        for _ in range(20000):
            act = np.asarray(ctrl.actions(obs[None, :]), np.float32).reshape(-1)
            _, _, _, _, next_obs, _, term, trunc, info = probe.step_dt(act)
            if float(info.get("acrobot_xk_homoclinic_capture", 0.0)) > 0.5:
                pool.append(snapshot(probe))
            if term or trunc:
                break
            obs = np.asarray(next_obs, dtype=np.float32)
    if not pool:
        raise RuntimeError("no in-tube states collected")
    idx = np.linspace(0, len(pool) - 1, min(n, len(pool))).astype(int)
    return [pool[i] for i in idx], (gains, torque_limit, task_kwargs)


def onpolicy_states(algo, env, n: int, seed: int, episodes: int = 4):
    """Physical states the loaded policy itself visits (deterministic actions).

    Pooled over several episodes and then evenly subsampled: a sample drawn from
    one trajectory understated the spread of the same quantity by half in
    the original n = 185 advantage measurement, so a single episode is not a
    safe state distribution here even when it supplies enough states.
    """
    pool = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        obs = np.asarray(obs, dtype=np.float32)
        for _ in range(env.max_steps or 2000):
            with th.no_grad():
                act, _ = algo.model.act(
                    th.as_tensor(obs[None, :], dtype=th.float32, device=algo.device),
                    deterministic=True,
                )
            act = act.cpu().numpy().reshape(-1)
            _, _, _, _, next_obs, _, term, trunc, _ = env.step_dt(act)
            pool.append(snapshot(env))
            if term or trunc:
                break
            obs = np.asarray(next_obs, dtype=np.float32)
    idx = np.linspace(0, len(pool) - 1, min(n, len(pool))).astype(int)
    return [pool[i] for i in idx]


# --------------------------------------------------------------------------
# the measurement
# --------------------------------------------------------------------------
def state_value(algo, obs_t, alpha_t, stochastic: bool):
    """V(s) exactly as CTSAC._state_value reads it, minus the MC noise.

    With a V-head this is the target value net.  Without one it is the sampled
    soft expectation; we take the policy MEAN action so repeated reads of the
    same state agree bit-for-bit and a 1-sample Monte-Carlo wobble cannot be
    mistaken for the action signal under test.
    """
    with th.no_grad():
        if algo._value_head_ready:
            return algo.model.target_value(obs_t)
        return algo._value_expectation(obs_t, alpha_t, deterministic=not stochastic)


def audit_states(algo, env, states, alpha_t, grid, controller, stochastic):
    """Per-state action sweep of the critic target and the learned critic."""
    T = float(algo.target_reference_dt)
    dt = float(env.dt)  # the requested control interval; env.dt_default is
    # dm_control's NATIVE control timestep and is 10 ms even for a 1 ms run
    dev = algo.device
    G = len(grid)
    act_dim = int(np.prod(env.action_space.shape))
    lo = np.asarray(env.action_space.low, dtype=np.float32)
    hi = np.asarray(env.action_space.high, dtype=np.float32)

    rows = []
    for qpos, qvel in states:
        actions = np.tile(
            np.zeros(act_dim, dtype=np.float32), (G, 1)
        )
        # Sweep the FIRST actuator over the grid; hold the rest at the policy
        # mean, so multi-actuator tasks still measure a one-dimensional slice
        # through the same action point the actor is sitting on.
        obs0 = restore(env, qpos, qvel)
        obs0_t = th.as_tensor(obs0[None, :], dtype=th.float32, device=dev)
        with th.no_grad():
            a_pi, _ = algo.model.act(obs0_t, deterministic=True)
        a_pi = a_pi.cpu().numpy().reshape(-1)
        actions[:] = a_pi
        actions[:, 0] = grid * (hi[0] - lo[0]) / 2.0 + (hi[0] + lo[0]) / 2.0

        next_obs = np.empty((G, obs0.shape[0]), dtype=np.float32)
        rewards = np.empty((G, 1), dtype=np.float32)
        dones = np.zeros((G, 1), dtype=np.float32)
        for j in range(G):
            _, r, no, term, used = one_step(env, qpos, qvel, actions[j])
            _check_interval(used, dt)
            next_obs[j] = no
            rewards[j] = r
            dones[j] = 1.0 if term else 0.0

        obs_rep = th.as_tensor(
            np.repeat(obs0[None, :], G, axis=0), dtype=th.float32, device=dev
        )
        act_t = th.as_tensor(actions, dtype=th.float32, device=dev)
        nobs_t = th.as_tensor(next_obs, dtype=th.float32, device=dev)
        rew_t = th.as_tensor(rewards, dtype=th.float32, device=dev)
        done_t = th.as_tensor(dones, dtype=th.float32, device=dev)
        dt_t = th.full((G, 1), dt, dtype=th.float32, device=dev)

        with th.no_grad():
            V_cur = state_value(algo, obs_rep, alpha_t, stochastic)
            V_next = state_value(algo, nobs_t, alpha_t, stochastic)
            y = algo._finite_difference_target_from_values(
                V_cur, V_next, rew_t, done_t, dt_t
            )
            q = th.stack(algo.model.q_values(obs_rep, act_t), 0).min(0).values

        y = y.cpu().numpy().ravel().astype(np.float64)
        q = q.cpu().numpy().ravel().astype(np.float64)
        V = float(V_cur[0, 0])

        # the deviated-action pair, in the doc's own terms
        a_star = a_pi.copy()
        if controller is not None:
            law = np.asarray(controller.actions(obs0[None, :]), np.float32).reshape(-1)
            # The law returns nan on its own singularity (documented in
            # XinKanedaController.actions); fall back to the policy there.
            if np.all(np.isfinite(law)):
                a_star[: law.shape[0]] = np.clip(law, lo[: law.shape[0]],
                                                 hi[: law.shape[0]])
        a_dev = a_star.copy()
        a_dev[0] = float(np.clip(a_dev[0] + DELTA_NORMALIZED, lo[0], hi[0]))
        pair_y = []
        for a in (a_star, a_dev):
            _, r, no, term, used = one_step(env, qpos, qvel, a)
            _check_interval(used, dt)
            with th.no_grad():
                v_c = state_value(
                    algo,
                    th.as_tensor(obs0[None, :], dtype=th.float32, device=dev),
                    alpha_t,
                    stochastic,
                )
                v_n = state_value(
                    algo,
                    th.as_tensor(no[None, :], dtype=th.float32, device=dev),
                    alpha_t,
                    stochastic,
                )
                yy = algo._finite_difference_target_from_values(
                    v_c,
                    v_n,
                    th.full((1, 1), r, dtype=th.float32, device=dev),
                    th.full((1, 1), 1.0 if term else 0.0, dtype=th.float32, device=dev),
                    th.full((1, 1), dt, dtype=th.float32, device=dev),
                )
            pair_y.append(float(yy[0, 0]))

        shape_err = (q - q.mean()) - (y - y.mean())
        rho = (
            float(stats.spearmanr(q, y)[0])
            if np.ptp(q) > 0 and np.ptp(y) > 0
            else np.nan
        )
        rows.append(
            dict(
                V=V,
                range_y=float(y.max() - y.min()),
                range_q=float(q.max() - q.min()),
                rms_shape_err=float(np.sqrt(np.mean(shape_err**2))),
                spearman=float(rho),
                delta_y=pair_y[0] - pair_y[1],
                abs_delta_y=abs(pair_y[0] - pair_y[1]),
                argmax_gap=float(abs(grid[int(np.argmax(q))] - grid[int(np.argmax(y))])),
            )
        )
    return rows, T, dt


def dt_sweep(algo, env_id, arm_mode, states, alpha_t, grid, controller, stochastic,
             intervals=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05)):
    """Same critic, same states, different control intervals.

    Comparing a 1 ms run's critic with a 10 ms run's critic confounds the
    interval with two separately trained networks.  This probe removes that: it
    holds ONE loaded V fixed and re-forms the critic target at a range of
    control intervals, stepping the physics for each interval from the same
    physical states.  With ``dt = T`` the target reduces to its nominal branch

        y_T(a) = T r(s,a) + exp(-beta T) V(s'_T(a))

    so the doc's claim becomes a slope: ``range_a y_T`` should be proportional
    to ``T`` while ``range_a y_T / T`` -- the advantage-rate range -- stays flat.
    """
    from environment.dmc import DMCContinuousEnv

    _, env_kwargs, _, _, _ = load_ct_hyperparams_from_table(
        "ct_sac", env_id, arm_mode, hyperparams_dir=HYPERPARAMS_DIR
    )
    env_kwargs.pop("n_envs", None)
    env_kwargs.pop("eval_n_envs", None)
    env_kwargs = fix_interval(env_kwargs)
    beta = float(algo.discount_rate)
    dev = algo.device
    G = len(grid)
    out = []
    for T in intervals:
        kwargs = dict(env_kwargs)
        kwargs["dt"] = T
        kwargs["physics_dt"] = min(float(kwargs.get("physics_dt", T) or T), T)
        kwargs["max_steps"] = int(round(float(kwargs.get("episode_duration", 20)) / T))
        domain, task = env_id.split("-", 1)
        probe = DMCContinuousEnv(domain_name=domain, task_name=task, seed=0, **kwargs)
        act_dim = int(np.prod(probe.action_space.shape))
        lo = np.asarray(probe.action_space.low, dtype=np.float32)
        hi = np.asarray(probe.action_space.high, dtype=np.float32)

        rng_y, dlt = [], []
        for qpos, qvel in states:
            obs0 = restore(probe, qpos, qvel)
            obs0_t = th.as_tensor(obs0[None, :], dtype=th.float32, device=dev)
            with th.no_grad():
                a_pi, _ = algo.model.act(obs0_t, deterministic=True)
            a_pi = a_pi.cpu().numpy().reshape(-1)
            actions = np.tile(a_pi.astype(np.float32), (G + 2, 1))
            actions[:G, 0] = grid * (hi[0] - lo[0]) / 2.0 + (hi[0] + lo[0]) / 2.0
            a_star = a_pi.copy()
            if controller is not None:
                law = np.asarray(
                    controller.actions(obs0[None, :]), np.float32
                ).reshape(-1)
                if np.all(np.isfinite(law)):
                    a_star[: law.shape[0]] = np.clip(
                        law, lo[: law.shape[0]], hi[: law.shape[0]]
                    )
            actions[G] = a_star
            actions[G + 1] = a_star
            actions[G + 1, 0] = float(
                np.clip(a_star[0] + DELTA_NORMALIZED, lo[0], hi[0])
            )

            nxt = np.empty((G + 2, obs0.shape[0]), dtype=np.float32)
            rew = np.empty((G + 2,), dtype=np.float64)
            for j in range(G + 2):
                _, r, no, _, used = one_step(probe, qpos, qvel, actions[j])
                _check_interval(used, T)
                nxt[j] = no
                rew[j] = r
            with th.no_grad():
                V_next = (
                    state_value(
                        algo,
                        th.as_tensor(nxt, dtype=th.float32, device=dev),
                        alpha_t,
                        stochastic,
                    )
                    .cpu()
                    .numpy()
                    .ravel()
                    .astype(np.float64)
                )
            reward_term = rew * T if algo.reward_is_rate else rew
            y = reward_term + np.exp(-beta * T) * V_next
            rng_y.append(y[:G].max() - y[:G].min())
            dlt.append(y[G] - y[G + 1])
        rng_y = np.asarray(rng_y)
        dlt = np.asarray(dlt)
        sem = dlt.std(ddof=1) / np.sqrt(len(dlt)) if len(dlt) > 1 else np.nan
        out.append(
            dict(
                T=T,
                range_y=float(rng_y.mean()),
                range_qv=float(rng_y.mean() / T),
                delta_y=float(dlt.mean()),
                delta_y_sem=float(sem),
                delta_y_t=float(dlt.mean() / sem) if sem else np.nan,
                delta_y_rate=float(dlt.mean() / T),
                n_states=len(dlt),
            )
        )
        print(
            f"    dt=T={T*1000:6.1f}ms  range_y={out[-1]['range_y']:.5g}  "
            f"range_qv={out[-1]['range_qv']:.4g}  "
            f"delta_y={out[-1]['delta_y']:+.5g} (t={out[-1]['delta_y_t']:+.2f})  "
            f"delta_rate={out[-1]['delta_y_rate']:+.4g}",
            flush=True,
        )
    return out


def true_action_values(env, controller, states, grid, horizon, beta, policy=None):
    """True Q^pi(s, a) on the action grid, by exact rollout.  No networks.

    The first control step takes the grid action, every later step follows a
    continuation policy, and the beta-discounted reward is accumulated over
    ``horizon`` physical seconds.

    ``policy`` chooses WHICH action-value function this is, and the distinction
    matters.  With the default analytical law it is ``Q^{pi_XK}`` -- a reference
    ordering, useful for "does the critic prefer actions a good controller
    prefers", but not what ``Q_phi`` approximates.  Passing the critic's own
    policy gives ``Q^{pi_theta}``, which IS what ``Q_phi`` approximates, so the
    residual against it is approximation error with the policy gap removed by
    construction.  Ranking against the controller alone cannot separate the two.

    This is the reference ordering that sec. 7 of
    sec. 7 of ``docs/ct_sac_advantage_parameterization.md`` measures against --
    and unlike the critic's own target it does not depend on the seed or the
    arm, so one evaluation per control interval serves every seed.

    Returns ``(n_states, len(grid))`` of returns; the per-state offset is
    irrelevant since only the ordering across actions is used.
    """
    dt = float(env.dt)
    steps = int(round(horizon / dt))
    lo = float(env.action_space.low[0])
    hi = float(env.action_space.high[0])
    out = np.empty((len(states), len(grid)), dtype=np.float64)
    for i, (qpos, qvel) in enumerate(states):
        for j, g in enumerate(grid):
            first = float(np.clip(g * (hi - lo) / 2.0 + (hi + lo) / 2.0, lo, hi))
            obs = restore(env, qpos, qvel)
            total = 0.0
            for k in range(steps):
                if policy is None:
                    act = np.asarray(
                        controller.actions(obs[None, :]), np.float32
                    ).reshape(-1)
                else:
                    act = policy(obs)
                if not np.all(np.isfinite(act)):
                    break
                if k == 0:
                    act = np.array([first], dtype=np.float32)
                _, _, _, reward, next_obs, _, term, trunc, _ = env.step_dt(act)
                total += float(reward) * dt * np.exp(-beta * k * dt)
                if term or trunc:
                    break
                obs = np.asarray(next_obs, dtype=np.float32)
            out[i, j] = total
    return out


def rank_against_truth(algo, env, states, alpha_t, grid, truth, stochastic):
    """Per-state Spearman of the learned critic, and of its target, vs truth."""
    dev = algo.device
    lo = np.asarray(env.action_space.low, dtype=np.float32)
    hi = np.asarray(env.action_space.high, dtype=np.float32)
    dt = float(env.dt)
    G = len(grid)
    rho_q, rho_y, rho_rew, rho_val, share = [], [], [], [], []
    for i, (qpos, qvel) in enumerate(states):
        obs0 = restore(env, qpos, qvel)
        obs0_t = th.as_tensor(obs0[None, :], dtype=th.float32, device=dev)
        with th.no_grad():
            a_pi, _ = algo.model.act(obs0_t, deterministic=True)
        actions = np.tile(a_pi.cpu().numpy().reshape(-1).astype(np.float32), (G, 1))
        actions[:, 0] = grid * (hi[0] - lo[0]) / 2.0 + (hi[0] + lo[0]) / 2.0

        nxt = np.empty((G, obs0.shape[0]), dtype=np.float32)
        rew = np.empty((G, 1), dtype=np.float32)
        dn = np.zeros((G, 1), dtype=np.float32)
        for j in range(G):
            _, r, no, term, used = one_step(env, qpos, qvel, actions[j])
            _check_interval(used, dt)
            nxt[j], rew[j], dn[j] = no, r, 1.0 if term else 0.0
        obs_rep = th.as_tensor(
            np.repeat(obs0[None, :], G, axis=0), dtype=th.float32, device=dev
        )
        with th.no_grad():
            V_cur = state_value(algo, obs_rep, alpha_t, stochastic)
            V_next = state_value(
                algo, th.as_tensor(nxt, dtype=th.float32, device=dev),
                alpha_t, stochastic,
            )
            y = algo._finite_difference_target_from_values(
                V_cur, V_next,
                th.as_tensor(rew, device=dev), th.as_tensor(dn, device=dev),
                th.full((G, 1), dt, dtype=th.float32, device=dev),
            ).cpu().numpy().ravel()
            q = (
                th.stack(
                    algo.model.q_values(
                        obs_rep, th.as_tensor(actions, dtype=th.float32, device=dev)
                    ),
                    0,
                )
                .min(0)
                .values.cpu()
                .numpy()
                .ravel()
            )
        # Split the target into the two terms whose weights the discount rate
        # trades off.  The reward term does not depend on the horizon at all;
        # the value term's action-dependence shrinks as the horizon shortens
        # (V ~ r/lambda).  So rho(reward term, truth) is the ordering the target
        # would inherit in the myopic limit -- measurable without retraining.
        with th.no_grad():
            y_rew = algo._target_reward_term(
                th.as_tensor(rew, device=dev)
            ).cpu().numpy().ravel()
        y_val = y - y_rew

        t = truth[i]
        if np.ptp(t) > 0:
            if np.ptp(q) > 0:
                rho_q.append(float(stats.spearmanr(q, t)[0]))
            if np.ptp(y) > 0:
                rho_y.append(float(stats.spearmanr(y, t)[0]))
            if np.ptp(y_rew) > 0:
                rho_rew.append(float(stats.spearmanr(y_rew, t)[0]))
            if np.ptp(y_val) > 0:
                rho_val.append(float(stats.spearmanr(y_val, t)[0]))
        if np.ptp(y_val) > 0:
            share.append(float(np.ptp(y_rew) / np.ptp(y_val)))
    n = len(rho_q)
    sem = np.std(rho_q, ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    sem_y = (
        np.std(rho_y, ddof=1) / np.sqrt(len(rho_y)) if len(rho_y) > 1 else float("nan")
    )
    def m(v):
        return float(np.mean(v)) if v else float("nan")

    def se(v):
        return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")

    return dict(
        n_states=n,
        rho_q_truth=m(rho_q), rho_q_truth_sem=float(sem),
        rho_y_truth=m(rho_y), rho_y_truth_sem=float(sem_y),
        rho_reward_truth=m(rho_rew), rho_reward_truth_sem=se(rho_rew),
        rho_value_truth=m(rho_val), rho_value_truth_sem=se(rho_val),
        reward_over_value_range=m(share),
    )


def attach_oracle_dynamics(algo, env):
    """Give any loaded critic the MuJoCo oracle drift, whether or not it trained
    with one.

    This is what makes the estimator comparison a controlled one: the same
    trained value function can be read through the finite-difference target and
    through the analytic generator, so the two targets differ ONLY in how the
    action's effect is estimated, never in the weights they are estimated from.
    """
    if getattr(algo, "dynamics_target_model", None) is not None:
        return
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    model = PortHamiltonianModel(
        obs_dim, act_dim, mode="mujoco",
        drift_fn=env.dynamics_terms, human_input_intensity=0.0,
    )
    algo.dynamics_model = model
    algo.dynamics_target_model = model


def _deterministic_value_shim(algo):
    """Force ``_value_expectation`` onto the policy mean for the duration of a
    measurement.

    Without a V-head the generator's ``grad V`` comes from a sampled soft
    expectation, and the batch that sweeps the action grid holds the SAME state
    in every row -- so independent sampling per row would inject exactly the
    per-action noise this probe exists to measure the absence of.  The policy
    mean is what ``state_value`` uses everywhere else in this audit, so the two
    targets stay comparable.
    """
    original = algo._value_expectation

    def deterministic(obs, alpha_tensor, deterministic=False):
        return original(obs, alpha_tensor, deterministic=True)

    algo._value_expectation = deterministic
    return original


def compare_target_estimators(algo, env, states, alpha_t, grid, truth, stochastic):
    """Rank the finite-difference and model-based targets against the same truth.

    Both are formed at the same states, over the same action grid, from the same
    loaded weights.  The only difference is the estimator: ``y_fd`` reads
    ``V_phi`` at one sampled next state per action; ``y_mb`` reads ``V_phi`` and
    ``grad V_phi`` once at the current state and lets the action enter through
    the exact drift.
    """
    dev = algo.device
    dt = float(env.dt)
    G = len(grid)
    lo = np.asarray(env.action_space.low, dtype=np.float32)
    hi = np.asarray(env.action_space.high, dtype=np.float32)
    attach_oracle_dynamics(algo, env)
    restore_value = None if algo._value_head_ready else _deterministic_value_shim(algo)
    rho_fd, rho_mb, affine, agree, ratio = [], [], [], [], []
    try:
        for i, (qpos, qvel) in enumerate(states):
            obs0 = restore(env, qpos, qvel)
            obs0_t = th.as_tensor(obs0[None, :], dtype=th.float32, device=dev)
            with th.no_grad():
                a_pi, _ = algo.model.act(obs0_t, deterministic=True)
            actions = np.tile(a_pi.cpu().numpy().reshape(-1).astype(np.float32), (G, 1))
            actions[:, 0] = grid * (hi[0] - lo[0]) / 2.0 + (hi[0] + lo[0]) / 2.0

            nxt = np.empty((G, obs0.shape[0]), dtype=np.float32)
            rew = np.empty((G, 1), dtype=np.float32)
            dn = np.zeros((G, 1), dtype=np.float32)
            for j in range(G):
                _, r, no, term, used = one_step(env, qpos, qvel, actions[j])
                _check_interval(used, dt)
                nxt[j], rew[j], dn[j] = no, r, 1.0 if term else 0.0

            obs_rep = th.as_tensor(
                np.repeat(obs0[None, :], G, axis=0), dtype=th.float32, device=dev
            )
            act_t = th.as_tensor(actions, dtype=th.float32, device=dev)
            nxt_t = th.as_tensor(nxt, dtype=th.float32, device=dev)
            rew_t = th.as_tensor(rew, device=dev)
            dn_t = th.as_tensor(dn, device=dev)
            dt_t = th.full((G, 1), dt, dtype=th.float32, device=dev)

            with th.no_grad():
                V_cur = state_value(algo, obs_rep, alpha_t, stochastic)
                V_next = state_value(algo, nxt_t, alpha_t, stochastic)
                y_fd = algo._finite_difference_target_from_values(
                    V_cur, V_next, rew_t, dn_t, dt_t
                ).cpu().numpy().ravel()
            # the generator path needs a graph for grad V, so no no_grad here
            y_mb = algo._model_based_target(
                obs_rep, act_t, nxt_t, rew_t, dn_t, dt_t, alpha_t
            ).detach().cpu().numpy().ravel()

            # Mechanism diagnostics.  If V_phi is piecewise linear (ReLU) and the
            # reachable next states all fall inside one linear region, then
            # V(s'_a) is EXACTLY affine in a -- the finite difference is already
            # a clean directional derivative, there is no per-action network
            # noise to remove, and the analytic generator has nothing to fix.
            # ``vnext_affine_resid`` measures that: the residual of a straight
            # line fit through V(s'_a) against a, relative to its own range.
            vn = V_next.cpu().numpy().ravel().astype(np.float64)
            av = actions[:, 0].astype(np.float64)
            if np.ptp(vn) > 0 and np.ptp(av) > 0:
                fit = np.polyval(np.polyfit(av, vn, 1), av)
                affine.append(float(np.abs(vn - fit).max() / np.ptp(vn)))
            if np.ptp(y_fd) > 0 and np.ptp(y_mb) > 0:
                agree.append(float(stats.spearmanr(y_fd, y_mb)[0]))
                ratio.append(float(np.ptp(y_mb) / np.ptp(y_fd)))

            t = truth[i]
            if np.ptp(t) <= 0:
                continue
            if np.ptp(y_fd) > 0:
                rho_fd.append(float(stats.spearmanr(y_fd, t)[0]))
            if np.ptp(y_mb) > 0:
                rho_mb.append(float(stats.spearmanr(y_mb, t)[0]))
    finally:
        if restore_value is not None:
            algo._value_expectation = restore_value

    def pack(v):
        a = np.asarray(v, dtype=np.float64)
        n = len(a)
        return (
            float(a.mean()) if n else float("nan"),
            float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
        )

    m_fd, s_fd = pack(rho_fd)
    m_mb, s_mb = pack(rho_mb)
    return dict(
        n_states=len(rho_fd),
        rho_fd_truth=m_fd, rho_fd_truth_sem=s_fd,
        rho_mb_truth=m_mb, rho_mb_truth_sem=s_mb,
        vnext_affine_resid=float(np.mean(affine)) if affine else float("nan"),
        vnext_affine_resid_max=float(np.max(affine)) if affine else float("nan"),
        fd_mb_agreement=float(np.mean(agree)) if agree else float("nan"),
        mb_over_fd_range=float(np.mean(ratio)) if ratio else float("nan"),
    )


def learned_policy_fn(algo):
    """``obs -> action`` for the loaded critic's own deterministic policy."""
    def act(obs):
        with th.no_grad():
            a, _ = algo.model.act(
                th.as_tensor(obs[None, :], dtype=th.float32, device=algo.device),
                deterministic=True,
            )
        return a.cpu().numpy().reshape(-1).astype(np.float32)
    return act


def rho_vs_interval(algo, env_id, arm_mode, states, alpha_t, grid, gains, torque_limit,
                    horizon, intervals, stochastic, truth_cache, on_policy=False,
                    seed_tag=0):
    """Does the ACTION ORDERING decay with the control interval, network fixed?

    Every other comparison in this audit sets a critic trained at 1 ms against a
    different critic trained at 10 ms, so "the interval caused it" is inference:
    the two differ in more than their interval.  This probe removes that.  One
    loaded critic is held fixed while the interval is swept, and at each interval
    BOTH sides are rebuilt for that interval -- the critic's target (formed as a
    run with that reference interval would form it) and the rollout ground truth
    (deviate for one step of that length, then follow the analytical law for
    ``horizon`` seconds).

    The value reads happen before the reference interval is overridden, so V is
    the network's own value at its own entropy price; only the target formula's
    ``T`` moves.  Truth depends on the environment and the law, not on the seed,
    so it is cached per (mode, interval) and shared across seeds.

    A falling rho as the interval shrinks is the causal claim.  A flat rho says
    the interval is not the driver and the arm-to-arm differences are something
    else.
    """
    from controllers.xin_kaneda import AcrobotParams, Gains, XinKanedaController
    from environment.dmc import DMCContinuousEnv

    _, env_kwargs, _, _, _ = load_ct_hyperparams_from_table(
        "ct_sac", env_id, arm_mode, hyperparams_dir=HYPERPARAMS_DIR
    )
    env_kwargs.pop("n_envs", None)
    env_kwargs.pop("eval_n_envs", None)
    env_kwargs = fix_interval(env_kwargs)
    dev = algo.device
    G = len(grid)
    beta = float(algo.discount_rate)
    out = []

    for T in intervals:
        kwargs = dict(env_kwargs)
        kwargs["dt"] = T
        kwargs["physics_dt"] = min(float(kwargs.get("physics_dt", T) or T), T)
        kwargs["max_steps"] = int(round(float(kwargs.get("episode_duration", 20)) / T))
        domain, task = env_id.split("-", 1)
        probe = DMCContinuousEnv(domain_name=domain, task_name=task, seed=0, **kwargs)
        ctrl = XinKanedaController(
            AcrobotParams.from_physics(physics_of(probe)), gains,
            torque_limit=torque_limit,
        )
        lo = np.asarray(probe.action_space.low, dtype=np.float32)
        hi = np.asarray(probe.action_space.high, dtype=np.float32)

        # Truth depends on the task (reward shaping), the discount rate and the
        # interval -- never on which arm's weights are loaded.  Arms sharing a
        # reward and a lambda therefore share one computation.
        key = (
            repr(sorted((env_kwargs.get("task_kwargs") or {}).items())),
            round(beta, 9),
            round(T, 9),
            # the learned continuation differs per seed, so it cannot be shared
            ("pi", arm_mode, seed_tag) if on_policy else "xk",
        )
        if key not in truth_cache:
            ctrl.reset()
            truth_cache[key] = true_action_values(
                probe, ctrl, states, grid, horizon, beta,
                policy=learned_policy_fn(algo) if on_policy else None,
            )
        truth = truth_cache[key]

        rho, ranges = [], []
        eps_ac, sig_ac, span_ac, d_eps, d_true, eps_abs = [], [], [], [], [], []
        rho_val_only = []
        for i, (qpos, qvel) in enumerate(states):
            obs0 = restore(probe, qpos, qvel)
            obs0_t = th.as_tensor(obs0[None, :], dtype=th.float32, device=dev)
            with th.no_grad():
                a_pi, _ = algo.model.act(obs0_t, deterministic=True)
            actions = np.tile(a_pi.cpu().numpy().reshape(-1).astype(np.float32), (G, 1))
            actions[:, 0] = grid * (hi[0] - lo[0]) / 2.0 + (hi[0] + lo[0]) / 2.0

            nxt = np.empty((G, obs0.shape[0]), dtype=np.float32)
            rew = np.empty((G, 1), dtype=np.float32)
            dn = np.zeros((G, 1), dtype=np.float32)
            for j in range(G):
                _, r, no, term, used = one_step(probe, qpos, qvel, actions[j])
                _check_interval(used, T)
                nxt[j], rew[j], dn[j] = no, r, 1.0 if term else 0.0

            obs_rep = th.as_tensor(
                np.repeat(obs0[None, :], G, axis=0), dtype=th.float32, device=dev
            )
            with th.no_grad():
                V_cur = state_value(algo, obs_rep, alpha_t, stochastic)
                V_next = state_value(
                    algo, th.as_tensor(nxt, dtype=th.float32, device=dev),
                    alpha_t, stochastic,
                )
            saved_T = algo.target_reference_dt
            try:
                algo.target_reference_dt = T
                with th.no_grad():
                    y = algo._finite_difference_target_from_values(
                        V_cur, V_next,
                        th.as_tensor(rew, device=dev), th.as_tensor(dn, device=dev),
                        th.full((G, 1), T, dtype=th.float32, device=dev),
                    ).cpu().numpy().ravel()
            finally:
                algo.target_reference_dt = saved_T

            # --- epsilon = V_phi - V^pi at the landing points -------------
            # The rollout truth already contains V^pi there:
            #   G(s,a) = dt r(s,a) + exp(-beta dt) V^pi(s'_a)
            # so V^pi falls out by undoing the first step.  Comparing it with
            # the network's own value at the same points gives the error whose
            # behaviour across separations decides the mechanism: if epsilon is
            # smooth over delta-s its differences scale with delta-s and the
            # signal-to-error ratio is dt-invariant; if epsilon has decorrelated
            # they are flat in dt while the signal grows, and the ratio falls
            # linearly as the interval shrinks.
            gamma_T = float(np.exp(-beta * T))
            rew_amount = rew.ravel().astype(np.float64) * T
            V_true = (truth[i] - rew_amount) / gamma_T
            V_phi = V_next.cpu().numpy().ravel().astype(np.float64)
            eps = V_phi - V_true

            # separations between landing points, and the paired differences
            span = float(np.linalg.norm(nxt[-1] - nxt[0]))
            eps_ac.append(float(np.std(eps)))          # action-direction error
            sig_ac.append(float(np.std(V_true)))       # action-direction signal
            span_ac.append(span)
            d_eps.append(float(abs(eps[-1] - eps[0])))
            d_true.append(float(abs(V_true[-1] - V_true[0])))
            eps_abs.append(float(np.mean(eps)))        # state-level offset

            # Separate the two things that could make rho rise with the
            # interval: the VALUE estimate genuinely tracking the truth better,
            # or the shared reward term T r(s,a) -- present in target and truth
            # alike -- simply taking a larger share as the interval grows.
            # Ranking V_phi against V^pi at the landing points removes the
            # reward entirely, so it isolates the value estimate.
            if np.ptp(V_phi) > 0 and np.ptp(V_true) > 0:
                rho_val_only.append(float(stats.spearmanr(V_phi, V_true)[0]))

            t = truth[i]
            if np.ptp(t) > 0 and np.ptp(y) > 0:
                rho.append(float(stats.spearmanr(y, t)[0]))
                ranges.append(float(np.ptp(y)))

        n = len(rho)
        sem = float(np.std(rho, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        mm = lambda v: float(np.mean(v)) if v else float("nan")  # noqa: E731
        row = dict(
            T=T, n_states=n,
            rho_target_truth=float(np.mean(rho)) if rho else float("nan"),
            rho_target_truth_sem=sem,
            range_y=float(np.mean(ranges)) if ranges else float("nan"),
            # separation between the extreme landing points, in state units
            span_ds=mm(span_ac),
            # action-direction signal and error at that separation
            sig_action=mm(sig_ac),
            eps_action=mm(eps_ac),
            snr_action=mm(sig_ac) / mm(eps_ac) if mm(eps_ac) else float("nan"),
            # the same as a paired structure function across the full sweep
            d_true=mm(d_true),
            d_eps=mm(d_eps),
            # overall level of epsilon (carries the policy gap, not just error)
            eps_offset_rms=float(np.sqrt(np.mean(np.square(eps_abs))))
            if eps_abs else float("nan"),
            # value ordering with the shared reward term removed
            rho_value_only=mm(rho_val_only),
            rho_value_only_sem=float(
                np.std(rho_val_only, ddof=1) / np.sqrt(len(rho_val_only))
            ) if len(rho_val_only) > 1 else float("nan"),
        )
        out.append(row)
        print(
            f"    dt=T={T*1000:6.1f}ms  rho={row['rho_target_truth']:+.3f}"
            f"+-{sem:.3f}  |ds|={row['span_ds']:.4g}  "
            f"signal={row['sig_action']:.4g}  eps={row['eps_action']:.4g}  "
            f"snr={row['snr_action']:.3g}  "
            f"dTrue={row['d_true']:.4g}  dEps={row['d_eps']:.4g}  "
            f"rho_valueonly={row['rho_value_only']:+.3f}",
            flush=True,
        )
    return out


def _flush_rows(rows, path):
    """Write a result table after every seed, not only at the end.

    These audits run under a wall clock and their expensive part (the rollout
    truth) is front-loaded, so a run that overshoots its limit used to lose
    every row it had already computed.  Writing incrementally means a killed
    job still leaves usable results on disk.
    """
    if not rows:
        return
    with open(path + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(path + ".json", "w"), indent=2)


def policy_return_comparison(env, controller, algo, states, horizon, beta):
    """Which continuation policy actually earns more from these states?

    Every rho in this audit is a rank correlation over the actions available at
    one state, so it discards level entirely: two policies can agree on which
    torque to apply now (rho = 1) while one earns far more over the horizon.
    "Is the analytical controller a good yardstick for this reward" is a
    question about level, and needs its own measurement -- this one.  No action
    grid: each state is simply rolled out twice, once under each policy.
    """
    dt = float(env.dt)
    steps = int(round(horizon / dt))
    pi = learned_policy_fn(algo)
    g_xk, g_pi = [], []
    for qpos, qvel in states:
        for which, bucket in (("xk", g_xk), ("pi", g_pi)):
            obs = restore(env, qpos, qvel)
            total = 0.0
            for k in range(steps):
                if which == "xk":
                    act = np.asarray(
                        controller.actions(obs[None, :]), np.float32
                    ).reshape(-1)
                else:
                    act = pi(obs)
                if not np.all(np.isfinite(act)):
                    break
                _, _, _, reward, next_obs, _, term, trunc, _ = env.step_dt(act)
                total += float(reward) * dt * np.exp(-beta * k * dt)
                if term or trunc:
                    break
                obs = np.asarray(next_obs, dtype=np.float32)
            bucket.append(total)
    a, b = np.asarray(g_xk), np.asarray(g_pi)
    d = a - b
    n = len(d)
    sem = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return dict(
        n_states=n,
        V_xin_kaneda=float(a.mean()),
        V_learned=float(b.mean()),
        xk_minus_learned=float(d.mean()),
        xk_minus_learned_sem=sem,
        xk_minus_learned_t=float(d.mean() / sem) if sem else float("nan"),
        xk_better_frac=float((d > 0).mean()),
    )


def summarize(rows, T):
    a = {k: np.array([r[k] for r in rows], dtype=np.float64) for k in rows[0]}
    d = a["delta_y"]
    n = len(d)
    sem = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    out = dict(
        n_states=n,
        T=T,
        V_mean=float(a["V"].mean()),
        V_std=float(a["V"].std(ddof=1)),
        V_range=float(a["V"].max() - a["V"].min()),
        range_y=float(a["range_y"].mean()),
        range_qv=float(a["range_y"].mean() / T),
        range_q=float(a["range_q"].mean()),
        rms_shape_err=float(np.sqrt(np.mean(a["rms_shape_err"] ** 2))),
        spearman=float(np.nanmean(a["spearman"])),
        argmax_gap=float(a["argmax_gap"].mean()),
        delta_y=float(d.mean()),
        # |A| separates "the effect is small" from "the effect cancels across
        # states"; the original n = 185 measurement found the latter, with
        # 75 of 185 states taking the opposite sign.
        abs_delta_y=float(np.abs(d).mean()),
        delta_y_negative_frac=float((d < 0).mean()),
        delta_y_sem=float(sem),
        delta_y_t=float(d.mean() / sem) if sem and np.isfinite(sem) else np.nan,
        delta_y_rate=float(d.mean() / T),
    )
    out["snr"] = out["range_y"] / out["rms_shape_err"] if out["rms_shape_err"] else np.nan
    out["signal_frac_of_V"] = (
        out["range_y"] / out["V_range"] if out["V_range"] else np.nan
    )
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="acrobot-swingup-xk", choices=sorted(ARMS))
    ap.add_argument("--n-states", type=int, default=120)
    ap.add_argument("--grid", type=int, default=41)
    ap.add_argument("--which", default="final", choices=("final", "best"))
    ap.add_argument("--dist", default="both", choices=("tube", "onpolicy", "both"))
    ap.add_argument("--stochastic-value", action="store_true")
    ap.add_argument("--seeds", default="", help="comma list overriding the arm's seeds")
    ap.add_argument(
        "--dt-sweep",
        default="",
        help="comma list of control intervals in seconds; re-forms the target "
        "for ONE loaded critic at each of them (removes the two-networks "
        "confound from the 1 ms vs 10 ms comparison)",
    )
    ap.add_argument(
        "--dt-sweep-seeds",
        default="0",
        help="seeds to run the dt sweep on (it is the expensive probe)",
    )
    ap.add_argument(
        "--truth-grid",
        type=int,
        default=0,
        help="if > 1, roll out this many actions per state to get the TRUE "
        "Q^pi ordering and report the learned critic's rank correlation "
        "against it -- the measurement sec. 7 of the advantage-parameterization "
        "note rests on.  Truth is seed-independent, so it is computed once per arm.",
    )
    ap.add_argument("--truth-horizon", type=float, default=2.0)
    ap.add_argument(
        "--compare-targets",
        action="store_true",
        help="with --truth-grid, also form the MODEL-BASED target from the same "
        "loaded weights and rank both estimators against truth -- isolates the "
        "estimator from the network",
    )
    ap.add_argument(
        "--rho-dt-sweep",
        default="",
        help="comma list of intervals: hold ONE critic fixed, rebuild both its "
        "target and the rollout truth at each interval, and rank them.  Isolates "
        "the interval from the network.",
    )
    ap.add_argument(
        "--on-policy-truth",
        action="store_true",
        help="roll the continuation under the critic's OWN policy, giving "
        "Q^{pi_theta} instead of the controller's Q^{pi_XK}.  Removes the "
        "policy gap from epsilon; costs one truth computation per seed.",
    )
    ap.add_argument(
        "--policy-return",
        action="store_true",
        help="also compare the discounted return the analytical controller and "
        "the learned policy actually earn from the same states -- a LEVEL "
        "comparison, which the rank correlations cannot see",
    )
    ap.add_argument(
        "--skip-arm",
        default="",
        help="comma list of substrings; arms whose label contains one are skipped",
    )
    ap.add_argument("--out", default="results/action_error_target_audit")
    args = ap.parse_args()

    env_id = args.env
    grid = np.linspace(-1.0, 1.0, args.grid)
    tgrid_rho = np.linspace(-1.0, 1.0, max(args.truth_grid, 9))

    shared_tube, controller_spec = None, None
    dists = ["tube", "onpolicy"] if args.dist == "both" else [args.dist]
    if "tube" in dists:
        if env_id != "acrobot-swingup-xk":
            print(f"[{env_id}] no analytical controller; tube states skipped")
            dists = [d for d in dists if d != "tube"]
        else:
            shared_tube, controller_spec = tube_states(env_id, args.n_states)
            print(f"tube states: {len(shared_tube)} (shared by every arm)", flush=True)

    rows, sweep_rows, truth_rows, rho_rows, pol_rows = [], [], [], [], []
    truth_cache = {}
    sweep_seeds = {int(s) for s in args.dt_sweep_seeds.split(",") if s.strip()}
    skip = [x for x in args.skip_arm.split(",") if x.strip()]
    for arm in ARMS[env_id]:
        if any(x in arm["label"] for x in skip):
            print(f"[skip] {arm['label']}: excluded by --skip-arm")
            continue
        seeds = (
            [int(s) for s in args.seeds.split(",") if s.strip()]
            if args.seeds
            else discover_seeds(env_id, arm)
        )
        if not seeds:
            print(f"[skip] {arm['label']}: no seeds on disk")
            continue
        try:
            algo, env = build_algorithm(env_id, arm["mode"])
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[skip] {arm['label']}: build failed: {exc}")
            continue
        T = float(algo.target_reference_dt)
        print(
            f"\n=== {arm['label']}  mode={arm['mode']}\n"
            f"    dt={env.dt:g}s  T={T:g}s  beta={algo.discount_rate:g}/s  "
            f"reward_is_rate={algo.reward_is_rate}  v_head={algo._value_head_ready}",
            flush=True,
        )
        controller = None
        if controller_spec is not None:
            from controllers.xin_kaneda import AcrobotParams, Gains, XinKanedaController

            gains, torque_limit, _ = controller_spec
            controller = XinKanedaController(
                AcrobotParams.from_physics(physics_of(env)),
                gains,
                torque_limit=torque_limit,
            )

        truth = None
        if args.truth_grid > 1 and shared_tube is not None and controller is not None:
            tgrid = np.linspace(-1.0, 1.0, args.truth_grid)
            print(
                f"    computing true Q^pi on {len(shared_tube)} states x "
                f"{args.truth_grid} actions over {args.truth_horizon}s "
                f"(seed-independent)",
                flush=True,
            )
            controller.reset()
            truth = true_action_values(
                env, controller, shared_tube, tgrid, args.truth_horizon,
                float(algo.discount_rate),
            )

        for seed in seeds:
            run_dir = resolve_run_dir(env_id, arm, seed)
            if run_dir is None:
                print(f"  seed {seed}: no run dir")
                continue
            loaded = load_seed(algo, run_dir, args.which)
            if loaded is None:
                print(f"  seed {seed}: no {args.which} checkpoint")
                continue
            alpha_t, ckpt, alpha_val, steps = loaded

            for dist in dists:
                if dist == "tube":
                    states = shared_tube
                    ctrl = controller
                    if ctrl is not None:
                        ctrl.reset()
                else:
                    states = onpolicy_states(algo, env, args.n_states, 90000 + seed)
                    ctrl = controller
                    if ctrl is not None:
                        ctrl.reset()
                stats_rows, T_used, dt_used = audit_states(
                    algo, env, states, alpha_t, grid, ctrl, args.stochastic_value
                )
                s = summarize(stats_rows, T_used)
                s.update(
                    env_id=env_id,
                    arm=arm["label"],
                    mode=arm["mode"],
                    seed=seed,
                    dist=dist,
                    which=args.which,
                    dt=dt_used,
                    alpha=alpha_val,
                    steps=steps,
                    checkpoint=os.path.relpath(ckpt),
                )
                rows.append(s)
                _flush_rows(rows, args.out)
                print(
                    f"  seed {seed:2d} [{dist:8s}] "
                    f"range_y={s['range_y']:.4g}  range_qv={s['range_qv']:.3g}  "
                    f"range_Q={s['range_q']:.4g}  err={s['rms_shape_err']:.4g}  "
                    f"snr={s['snr']:.3g}  rho={s['spearman']:+.2f}  "
                    f"dY={s['delta_y']:+.4g} (t={s['delta_y_t']:+.2f})  "
                    f"Vrange={s['V_range']:.3g}",
                    flush=True,
                )

            if truth is not None:
                controller.reset()
                tr = rank_against_truth(
                    algo, env, shared_tube, alpha_t, tgrid, truth,
                    args.stochastic_value,
                )
                tr.update(
                    env_id=env_id, arm=arm["label"], mode=arm["mode"], seed=seed,
                    dist="tube", which=args.which, T=float(algo.target_reference_dt),
                )
                if args.compare_targets:
                    controller.reset()
                    cmp = compare_target_estimators(
                        algo, env, shared_tube, alpha_t, tgrid, truth,
                        args.stochastic_value,
                    )
                    tr.update(cmp)
                    print(
                        f"  seed {seed:2d} [estimators]  "
                        f"finite-difference rho={cmp['rho_fd_truth']:+.3f}"
                        f"+-{cmp['rho_fd_truth_sem']:.3f}   "
                        f"model-based rho={cmp['rho_mb_truth']:+.3f}"
                        f"+-{cmp['rho_mb_truth_sem']:.3f}   "
                        f"| V(s'|a) affine resid={cmp['vnext_affine_resid']:.4f} "
                        f"(max {cmp['vnext_affine_resid_max']:.3f})  "
                        f"fd~mb={cmp['fd_mb_agreement']:+.3f}  "
                        f"mb/fd range={cmp['mb_over_fd_range']:.3f}",
                        flush=True,
                    )
                truth_rows.append(tr)
                _flush_rows(truth_rows, args.out + "_truthrank")
                print(
                    f"  seed {seed:2d} [vs TRUE Q^pi] "
                    f"rho(critic)={tr['rho_q_truth']:+.3f}"
                    f"+-{tr['rho_q_truth_sem']:.3f}   "
                    f"rho(target)={tr['rho_y_truth']:+.3f}"
                    f"+-{tr['rho_y_truth_sem']:.3f}\n"
                    f"                    decomposed: reward term "
                    f"{tr['rho_reward_truth']:+.3f}+-{tr['rho_reward_truth_sem']:.3f}   "
                    f"value term {tr['rho_value_truth']:+.3f}"
                    f"+-{tr['rho_value_truth_sem']:.3f}   "
                    f"reward/value range={tr['reward_over_value_range']:.3f}",
                    flush=True,
                )

            if args.policy_return and shared_tube is not None and controller is not None:
                controller.reset()
                pr = policy_return_comparison(
                    env, controller, algo, shared_tube, args.truth_horizon,
                    float(algo.discount_rate),
                )
                pr.update(env_id=env_id, arm=arm["label"], mode=arm["mode"],
                          seed=seed, T=float(algo.target_reference_dt))
                pol_rows.append(pr)
                _flush_rows(pol_rows, args.out + "_policyreturn")
                print(
                    f"  seed {seed:2d} [policy return over {args.truth_horizon}s] "
                    f"XK={pr['V_xin_kaneda']:+.4g}  learned={pr['V_learned']:+.4g}  "
                    f"XK-learned={pr['xk_minus_learned']:+.4g}"
                    f"+-{pr['xk_minus_learned_sem']:.4g} "
                    f"(|t|={abs(pr['xk_minus_learned_t']):.2f}, "
                    f"XK better in {pr['xk_better_frac']*100:.0f}% of states)",
                    flush=True,
                )

            if args.rho_dt_sweep and shared_tube is not None and controller_spec:
                print(f"  seed {seed:2d} [rho vs interval, one fixed critic]", flush=True)
                g_, tq_, _ = controller_spec
                for entry in rho_vs_interval(
                    algo, env_id, arm["mode"], shared_tube, alpha_t, tgrid_rho,
                    g_, tq_, args.truth_horizon,
                    [float(x) for x in args.rho_dt_sweep.split(",") if x.strip()],
                    args.stochastic_value, truth_cache,
                    on_policy=args.on_policy_truth, seed_tag=seed,
                ):
                    entry.update(env_id=env_id, arm=arm["label"], mode=arm["mode"],
                                 seed=seed, which=args.which,
                                 truth_policy="learned" if args.on_policy_truth
                                 else "xin_kaneda",
                                 trained_T=float(algo.target_reference_dt))
                    rho_rows.append(entry)
                _flush_rows(rho_rows, args.out + "_rhodt")

            if args.dt_sweep and seed in sweep_seeds and shared_tube is not None:
                print(f"  seed {seed:2d} [dt-sweep, one fixed critic]", flush=True)
                ctrl = controller
                if ctrl is not None:
                    ctrl.reset()
                for entry in dt_sweep(
                    algo,
                    env_id,
                    arm["mode"],
                    shared_tube,
                    alpha_t,
                    grid,
                    ctrl,
                    args.stochastic_value,
                    intervals=[float(x) for x in args.dt_sweep.split(",") if x.strip()],
                ):
                    entry.update(
                        env_id=env_id, arm=arm["label"], mode=arm["mode"], seed=seed,
                        dist="tube", which=args.which,
                    )
                    sweep_rows.append(entry)
                _flush_rows(sweep_rows, args.out + "_dtsweep")
        env.close() if hasattr(env, "close") else None

    if not rows:
        print("no rows produced")
        return 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = list(rows[0])
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(args.out + ".json", "w"), indent=2)

    print("\n" + "=" * 108)
    print(
        f"{'arm':24s} {'dist':9s} {'T':>7s} {'range_y':>10s} {'range_qv':>9s} "
        f"{'range_Q':>9s} {'err':>9s} {'snr':>7s} {'rho':>6s} {'delta_y':>10s} "
        f"{'|t|':>5s} {'V_range':>8s}"
    )
    for arm in ARMS[env_id]:
        for dist in dists:
            sel = [r for r in rows if r["arm"] == arm["label"] and r["dist"] == dist]
            if not sel:
                continue
            m = lambda k: float(np.nanmean([r[k] for r in sel]))  # noqa: E731
            print(
                f"{arm['label']:24s} {dist:9s} {m('T'):7.4f} {m('range_y'):10.5f} "
                f"{m('range_qv'):9.3f} {m('range_q'):9.5f} {m('rms_shape_err'):9.5f} "
                f"{m('snr'):7.3f} {m('spearman'):+6.2f} {m('delta_y'):+10.5f} "
                f"{abs(m('delta_y_t')):5.2f} {m('V_range'):8.3f}"
            )
    if rho_rows:
        with open(args.out + "_rhodt.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rho_rows[0]))
            w.writeheader()
            w.writerows(rho_rows)
        json.dump(rho_rows, open(args.out + "_rhodt.json", "w"), indent=2)
        print("\n" + "=" * 84)
        print("action ordering vs control interval -- ONE fixed critic per row-group")
        Ts = sorted({r["T"] for r in rho_rows})
        print(f"{'arm (trained at)':30s} " + "".join(f"{T*1000:>10.0f}ms" for T in Ts))
        for arm in ARMS[env_id]:
            sel = [r for r in rho_rows if r["arm"] == arm["label"]]
            if not sel:
                continue
            vals = []
            for T in Ts:
                v = [r["rho_target_truth"] for r in sel if r["T"] == T]
                vals.append(float(np.mean(v)) if v else float("nan"))
            print(f"{arm['label']:30s} " + "".join(f"{v:+12.3f}" for v in vals))
        print(f"wrote {args.out}_rhodt.csv / .json ({len(rho_rows)} rows)")

    if truth_rows:
        with open(args.out + "_truthrank.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(truth_rows[0]))
            w.writeheader()
            w.writerows(truth_rows)
        json.dump(truth_rows, open(args.out + "_truthrank.json", "w"), indent=2)
        print("\n" + "=" * 84)
        print("action ordering vs TRUE Q^pi (rollout ground truth, per-state Spearman)")
        print(f"{'arm':24s} {'T':>7s} {'rho(critic)':>14s} {'rho(target)':>14s}")
        for arm in ARMS[env_id]:
            sel = [r for r in truth_rows if r["arm"] == arm["label"]]
            if not sel:
                continue
            rq = np.mean([r["rho_q_truth"] for r in sel])
            ry = np.mean([r["rho_y_truth"] for r in sel])
            sq = np.std([r["rho_q_truth"] for r in sel], ddof=1) if len(sel) > 1 else 0.0
            sy = np.std([r["rho_y_truth"] for r in sel], ddof=1) if len(sel) > 1 else 0.0
            print(
                f"{arm['label']:24s} {sel[0]['T']:7.4f} {rq:+8.3f}+-{sq:<5.3f} "
                f"{ry:+8.3f}+-{sy:<5.3f}"
            )
        print(f"wrote {args.out}_truthrank.csv / .json ({len(truth_rows)} rows)")

    if sweep_rows:
        with open(args.out + "_dtsweep.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sweep_rows[0]))
            w.writeheader()
            w.writerows(sweep_rows)
        json.dump(sweep_rows, open(args.out + "_dtsweep.json", "w"), indent=2)
        print("\n" + "=" * 96)
        print("dt sweep -- ONE trained critic, target re-formed at each interval")
        print(
            f"{'arm':24s} {'seed':>4s} {'T (ms)':>7s} {'range_y':>10s} "
            f"{'range_qv':>9s} {'delta_y':>10s} {'|t|':>5s} {'delta/T':>9s}"
        )
        for r in sweep_rows:
            print(
                f"{r['arm']:24s} {r['seed']:4d} {r['T']*1000:7.1f} "
                f"{r['range_y']:10.5f} {r['range_qv']:9.3f} {r['delta_y']:+10.5f} "
                f"{abs(r['delta_y_t']):5.2f} {r['delta_y_rate']:+9.3f}"
            )
        print(f"wrote {args.out}_dtsweep.csv / .json ({len(sweep_rows)} rows)")

    print(f"\nwrote {args.out}.csv / .json ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
