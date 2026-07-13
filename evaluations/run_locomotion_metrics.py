"""Run energy-efficiency and gait-consistency metrics on a saved cheetah policy.

Rolls a policy out on cheetah-run and reports the metrics in
``evaluations.locomotion_metrics``. The metrics read the MuJoCo physics
directly, so the only thing that must match training is the observation layout
(``--raw-state-obs`` / ``--no-raw-state-obs``) and the actor architecture
(``--pi-arch``), which set the actor's input/hidden sizes.

Examples
--------
Model-based / structured / oracle policies (raw [qpos; qvel] observation)::

    python -m evaluations.run_locomotion_metrics \
        --checkpoint /path/to/model.pt --n-episodes 8 --out results/oracle_locomotion.csv

Model-free baseline trained on the cheetah task observation::

    python -m evaluations.run_locomotion_metrics \
        --checkpoint /path/to/top.pt --no-raw-state-obs --out results/top_locomotion.csv

Random baseline (no checkpoint)::

    python -m evaluations.run_locomotion_metrics --random --out results/random_locomotion.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch as th
from torch import nn

from environment.dmc import DMCContinuousEnv
from models.actor_q_critic import ActorQCriticModel
from evaluations.locomotion_metrics import evaluate_locomotion

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU, "gelu": nn.GELU}

# The energy/gait fields worth printing, grouped, with a label and format.
_ENERGY_FIELDS = [
    ("return", "return", "{:.1f}"),
    ("mean_speed_mps", "mean speed [m/s]", "{:.3f}"),
    ("distance_m", "distance [m]", "{:.2f}"),
    ("cost_of_transport", "cost of transport", "{:.4f}"),
    ("return_per_joule", "return / joule", "{:.4f}"),
    ("mean_power_W", "mean power [W]", "{:.2f}"),
    ("mean_abs_power_W", "mean |power| [W]", "{:.2f}"),
    ("work_pos_J", "positive work [J]", "{:.1f}"),
    ("control_cost", "control cost ∫Σa²dt", "{:.2f}"),
]
_GAIT_FIELDS = [
    ("gait_detected", "gait detected (frac)", "{:.2f}"),
    ("autocorr_peak", "periodicity (autocorr)", "{:.3f}"),
    ("stride_cv", "stride-period CV", "{:.3f}"),
    ("poincare_dispersion", "Poincaré dispersion", "{:.3f}"),
    ("limb_plv", "front/back phase lock", "{:.3f}"),
    ("stride_freq_hz", "stride freq [Hz]", "{:.3f}"),
    ("spectral_entropy_mean", "spectral entropy", "{:.3f}"),
    ("peak_power_frac_mean", "fundamental power frac", "{:.3f}"),
    ("band_power_frac_mean", "stride-band power frac", "{:.3f}"),
    ("n_strides", "strides / episode", "{:.1f}"),
]


def _parse_arch(s: str) -> List[int]:
    return [int(x) for x in str(s).split(",") if str(x).strip() != ""]


def load_policy(
    checkpoint: str,
    env: DMCContinuousEnv,
    *,
    pi_arch: List[int],
    activation: str = "relu",
    deterministic_policy: bool = False,
    log_std_init: float = -3.0,
    device: str = "cpu",
) -> Callable[[np.ndarray], np.ndarray]:
    """Build the actor and load only its weights (critics/v-head irrelevant here)."""
    act_fn = _ACTIVATIONS[activation.lower()]
    model = ActorQCriticModel(
        observation_space=env.observation_space,
        action_space=env.action_space,
        q_net_arch=pi_arch,          # unused for acting; kept constructible
        pi_net_arch=pi_arch,
        activation_fn=act_fn,
        log_std_init=log_std_init,
        n_critics=1,
        deterministic_policy=deterministic_policy,
        device=device,
    )
    state = th.load(checkpoint, map_location=device)
    if not (isinstance(state, dict) and "actor" in state):
        raise ValueError(
            f"{checkpoint}: expected a model checkpoint with an 'actor' entry "
            f"(saved by ActorQCriticModel.save / CTSAC.save)."
        )
    model.actor.load_state_dict(state["actor"], strict=True)
    model.actor.eval()

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        ot = th.as_tensor(obs, dtype=th.float32, device=device).unsqueeze(0)
        with th.no_grad():
            act, _ = model.act(ot, deterministic=True)
        return act.cpu().numpy()[0]

    return policy_fn


def _parse_kv(s: str) -> dict:
    """Parse a 'k=v;k=v' string (the CSV time_sampling_kwargs format) to floats."""
    out = {}
    for part in str(s).split(";"):
        part = part.strip()
        if not part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = float(v)
    return out


def build_env(env_id: str, *, raw_state_obs: bool, dt: float,
              episode_seconds: float, seed: int, time_sampling: str = "uniform",
              physics_dt: float = 0.002, min_dt: float = 0.002,
              max_dt: float = 0.03, max_steps: int = 2000,
              time_sampling_kwargs: str = "") -> DMCContinuousEnv:
    domain, task = env_id.split("-", 1)
    kw = dict(
        domain_name=domain, task_name=task, seed=seed,
        raw_state_obs=raw_state_obs, time_sampling=time_sampling, dt=dt,
        episode_duration=episode_seconds,
    )
    if time_sampling == "irregular":
        # Match the training time base: a heavy small-dt tail (time_sampling_kwargs,
        # e.g. cheetah's tail_p=0.99;tail_split=0.9) drives the mean control dt to
        # ~0.005, so a fixed-duration episode accumulates ~max_steps control steps.
        # The gait metrics resample to a uniform grid, so only the return scale (an
        # un-normalized per-step sum) and the ∫dt energy terms are affected.
        kw.update(physics_dt=physics_dt, min_dt=min_dt, max_dt=max_dt,
                  max_steps=max_steps)
        if time_sampling_kwargs:
            kw["time_sampling_kwargs"] = _parse_kv(time_sampling_kwargs)
    return DMCContinuousEnv(**kw)


def _fmt(agg: dict, key: str, fmt: str) -> str:
    m, s, n = agg.get(f"{key}_mean"), agg.get(f"{key}_std"), agg.get(f"{key}_n", 0)
    if m is None or not np.isfinite(m):
        return f"(n/a, {n} finite)"
    return f"{fmt.format(m)} ± {fmt.format(s)}"


def print_summary(agg: dict, title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    print("-- Energy efficiency --")
    for key, label, fmt in _ENERGY_FIELDS:
        print(f"  {label:<24} {_fmt(agg, key, fmt)}")
    print("-- Gait consistency --")
    for key, label, fmt in _GAIT_FIELDS:
        print(f"  {label:<24} {_fmt(agg, key, fmt)}")


def write_outputs(result: dict, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = result["per_episode"]
    if rows:
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    out_json = out_csv.with_suffix(".agg.json")
    with out_json.open("w") as f:
        json.dump(result["aggregate"], f, indent=2)
    print(f"\n[written] per-episode: {out_csv}\n[written] aggregate:   {out_json}")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a saved model (ActorQCriticModel/CTSAC .pt).")
    p.add_argument("--random", action="store_true",
                   help="Use a random policy instead of a checkpoint.")
    p.add_argument("--env-id", type=str, default="cheetah-run")
    p.add_argument("--raw-state-obs", dest="raw_state_obs", action="store_true",
                   default=True, help="Observation is [qpos; qvel] (default).")
    p.add_argument("--no-raw-state-obs", dest="raw_state_obs", action="store_false",
                   help="Observation is the cheetah task obs (model-free baselines).")
    p.add_argument("--pi-arch", type=str, default="400,300")
    p.add_argument("--activation", type=str, default="relu")
    p.add_argument("--deterministic-policy", action="store_true",
                   help="Actor is deterministic (DDPG/TD3); default assumes SAC.")
    p.add_argument("--log-std-init", type=float, default=-3.0)
    p.add_argument("--n-episodes", type=int, default=8)
    p.add_argument("--warmup-s", type=float, default=2.0,
                   help="Seconds of acceleration to drop before gait analysis.")
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--time-sampling", choices=["uniform", "irregular"],
                   default="uniform",
                   help="'irregular' matches the training time base "
                        "(fine jittered control dt, ~max_steps steps/episode) so "
                        "the return scale lines up with training-eval numbers.")
    p.add_argument("--physics-dt", type=float, default=0.002)
    p.add_argument("--min-dt", type=float, default=0.002)
    p.add_argument("--max-dt", type=float, default=0.03)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--time-sampling-kwargs", type=str, default="",
                   help="irregular-sampler params in 'k=v;k=v' form; match the "
                        "training row, e.g. cheetah's 'tail_p=0.99;tail_split=0.9'.")
    p.add_argument("--episode-seconds", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--label", type=str, default=None)
    args = p.parse_args(argv)

    if not args.random and not args.checkpoint:
        p.error("provide --checkpoint PATH or --random")

    env = build_env(args.env_id, raw_state_obs=args.raw_state_obs, dt=args.dt,
                    episode_seconds=args.episode_seconds, seed=args.seed,
                    time_sampling=args.time_sampling, physics_dt=args.physics_dt,
                    min_dt=args.min_dt, max_dt=args.max_dt, max_steps=args.max_steps,
                    time_sampling_kwargs=args.time_sampling_kwargs)

    if args.random:
        label = args.label or "random"
        policy_fn = lambda _o: env.action_space.sample()  # noqa: E731
    else:
        label = args.label or Path(args.checkpoint).stem
        policy_fn = load_policy(
            args.checkpoint, env,
            pi_arch=_parse_arch(args.pi_arch), activation=args.activation,
            deterministic_policy=args.deterministic_policy,
            log_std_init=args.log_std_init, device=args.device,
        )

    result = evaluate_locomotion(
        policy_fn, env, n_episodes=args.n_episodes, warmup_s=args.warmup_s
    )
    print_summary(result["aggregate"], f"{label}  ({args.env_id}, {args.n_episodes} episodes)")

    out_csv = Path(args.out) if args.out else Path("results") / f"{label}_locomotion.csv"
    write_outputs(result, out_csv)


if __name__ == "__main__":
    main()
