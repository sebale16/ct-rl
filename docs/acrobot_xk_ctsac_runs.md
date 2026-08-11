# CT-SAC runs on acrobot-swingup-xk

An inventory of what was run. Configurations and file locations only; no results.

## Common to every run

CT-SAC, `env_id = acrobot-swingup-xk`, seeds 0–5, 1M steps, one Frontera node per
launch at `OMP_NUM_THREADS=1`.

Environment: `time_sampling = uniform`, `dt = physics_dt = 0.0005`,
`max_steps = 40000`, `episode_duration = 20`, `raw_state_obs`, `damping = 0`,
`release_start`. Model: q and pi nets `[400, 300]`, ReLU, `log_std_init = -1`,
2 critics, `periodic_obs_indices = (0,)`. Algorithm: `buffer_size = 1e6`,
`lr = 3e-4`, `batch_size = 256`, `train_freq = 1`, `gradient_steps = 1`,
`learning_starts = 10000`, `tau = 0.04`, `target_entropy = auto`.
Logging: `log_interval = 2000`, `save_freq = 100000`, `eval_freq = 100000`.

Evaluation is the fixed protocol `run_ct_rl` forces for this `env_id`: mode
`xk_eval`, the 32 seeds 20000–20031, uniform `dt = 5e-4`, 20 s episodes.
`--eval_mode` and `--n_eval_episodes` are rejected/overridden for this env.

Discount horizon below is `dt_default / beta` with `beta = -log(gamma)` and
`dt_default = control_timestep() = 0.01`.

## Runs

| run_id | job | arms x seeds | cells to 1M | notes |
|---|---|---|---|---|
| `xk_uniform_v2` | 7894354 | 8 x 6 = 48 | 48/48 | the r0/r1/r2 reward arms |
| `xk_gamma_v1` | 7895458 | 8 x 6 = 48 | 48/48 | gamma x reward bounding |
| `xk_spin_v1` | 7895627 | 6 x 6 = 36 | 12/36 | the 24 `_pos` cells were cancelled at 656k–934k |
| `xk_spinraw_v1` | 7895630 | 4 x 6 = 24 | 24/24 | as `xk_spin_v1` without `reward_squash` |

A superseded first launch, `xk_uniform_v1` (job 7894351), was cancelled at 30k
steps and is not included anywhere.

## Arms

Columns are gamma, discount horizon, alpha, reward kind, eta, actuator gear
(`torque_limit`), and the task kwargs that differ from the base row.

### `xk_uniform_v2` — reward kinds

| mode | gamma | horizon (s) | alpha | reward | eta | gear |
|---|---|---|---|---|---|---|
| `xk_r0_fixed0p5ms` | 0.995 | 1.995 | auto | r0 | – | 64 |
| `xk_r1_fixed0p5ms` | 0.995 | 1.995 | auto | r1 | – | 64 |
| `xk_r2_eta0_fixed0p5ms` | 0.995 | 1.995 | auto | r2 | 0 | 64 |
| `xk_r2_eta0p01_fixed0p5ms` | 0.995 | 1.995 | auto | r2 | 0.01 | 64 |
| `xk_r2_eta0p03_fixed0p5ms` | 0.995 | 1.995 | auto | r2 | 0.03 | 64 |
| `xk_r2_eta0p1_fixed0p5ms` | 0.995 | 1.995 | auto | r2 | 0.1 | 64 |
| `xk_r2_eta0p3_fixed0p5ms` | 0.995 | 1.995 | auto | r2 | 0.3 | 64 |
| `xk_r2_eta1_fixed0p5ms` | 0.995 | 1.995 | auto | r2 | 1.0 | 64 |

The reward rates, all on r1's unwrapped elbow coordinate where applicable:

    r0(x) = -[(Etil/E_s)^2 + (q2/q_s)^2 + (qdot2/omega_s)^2]
    r1(x) = -V(x),  V = 1/2 Etil^2 + 1/2 k_D qdot2^2 + 1/2 k_P q2^2
    r2(x,u) = -V(x) - eta Vdot(x,u)

with `k_D = 35.8`, `k_P = 61.2`.

### `xk_gamma_v1` — discount horizon x reward bounding

All r1, gear 64. `_sq` sets `reward_squash = 1200`, replacing `V` with
`V / (1 + V/1200)`.

| mode | gamma | horizon (s) | reward_squash |
|---|---|---|---|
| `xk_r1_g0p995_raw` | 0.995 | 1.995 | – |
| `xk_r1_g0p995_sq` | 0.995 | 1.995 | 1200 |
| `xk_r1_g0p998_raw` | 0.998 | 4.995 | – |
| `xk_r1_g0p998_sq` | 0.998 | 4.995 | 1200 |
| `xk_r1_g0p9991_raw` | 0.9991 | 11.106 | – |
| `xk_r1_g0p9991_sq` | 0.9991 | 11.106 | 1200 |
| `xk_r1_g0p9995_raw` | 0.9995 | 19.995 | – |
| `xk_r1_g0p9995_sq` | 0.9995 | 19.995 | 1200 |

`xk_r1_g0p995_raw` is configuration-identical to `xk_r1_fixed0p5ms`.

### `xk_spin_v1` — elbow-spin termination at gear 20

All r1, gear 20, `reward_squash = 1200.5`, `spin_limit = 2*pi` (the episode ends
when `|q2| >= 2*pi`).

| mode | gamma | horizon (s) | alpha | reward_offset | spin_penalty |
|---|---|---|---|---|---|
| `xk_spin_g0p995_pos` | 0.995 | 1.995 | auto | 1200.5 | 0 |
| `xk_spin_g0p995_neg` | 0.995 | 1.995 | auto | – | 2.3e7 |
| `xk_spin_g0p995_pos_a0p05` | 0.995 | 1.995 | 0.05 | 1200.5 | 0 |
| `xk_spin_g0p9991_pos` | 0.9991 | 11.106 | auto | 1200.5 | 0 |
| `xk_spin_g0p9991_neg` | 0.9991 | 11.106 | auto | – | 2.3e7 |
| `xk_spin_g0p9991_pos_a0p05` | 0.9991 | 11.106 | 0.05 | 1200.5 | 0 |

Only the two `_neg` arms (12 cells) ran to 1M.

### `xk_spinraw_v1` — the same without reward bounding

All r1, gear 20, `spin_limit = 2*pi`, `spin_penalty = 4e7`, no `reward_squash`
and no `reward_offset`.

| mode | gamma | horizon (s) | alpha |
|---|---|---|---|
| `xk_spinraw_g0p995` | 0.995 | 1.995 | auto |
| `xk_spinraw_g0p995_a0p05` | 0.995 | 1.995 | 0.05 |
| `xk_spinraw_g0p9991` | 0.9991 | 11.106 | auto |
| `xk_spinraw_g0p9991_a0p05` | 0.9991 | 11.106 | 0.05 |

## Reference controller

`evaluations/eval_acrobot_xk.py` was run on the analytical Xin-Kaneda law at
`k_v = 66.3`, `k_d = 35.8`, `k_p = 61.2` in three configurations: the paper's own
initial condition at `dt = 1e-4` over 12 s; the RL protocol exactly
(`--start release`, seeds 20000–20031, `dt = 5e-4`, 20 s, gear 64); and that
protocol with the episode cap raised to 120 s.

## Files

- `logs/**/progress.csv` — training curves, one per cell, `log_interval = 2000`.
  Each cell directory also holds `progress.json`, `log.txt`, a TensorBoard events
  file and `eval/evaluations.npz`; those are not committed.
- `results/xk_ctsac_eval/` — per-policy metric battery for `xk_uniform_v2`,
  48 files.
- `results/xk_eval_all/` — the same battery for `xk_gamma_v1`,
  `xk_spinraw_v1` and the `xk_spin_v1` `_neg` arms, 84 files, named
  `<run_id>__<mode>_seed<n>.csv`.

Both batteries come from `evaluations/eval_acrobot_xk_ctsac.py` over each cell's
`final_model.pth`, on the 32-seed protocol at `t_max = 20`, `dt = 5e-4`,
`damping = 0`, `torque_limit = 64`, `dwell_seconds = 1.0`, tube tolerances
`(0.05, 0.025, 0.05)`.

## Reading the logged columns

- `rollout/*` values are means over the 2000-step logging window of signed
  per-step quantities. `acrobot_xk_elbow_norm` averages to ~0 for a freely
  spinning elbow, and `acrobot_xk_lyapunov_rate` telescopes, since the integral
  of Vdot over a window is the change in V.
- `rollout/acrobot_xk_homoclinic_capture` is tube occupancy over the window, not
  continuous dwell.
- `eval/strict_capture_mean_max_duration` is the mean over the 32 protocol
  episodes of each episode's longest continuous dwell, not the maximum.
- `time/fps` is inflated after `--resume`. A true rate needs row-to-row deltas
  within one chunk, excluding the first `learning_starts = 10000` steps, which
  run without gradient updates.
