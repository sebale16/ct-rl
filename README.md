# Continuous-Time Reinforcement Learning (CT-RL)

This repository contains implementations of CT-SAC and CT-TD3 and benchmarks for other Continuous-Time Reinforcement Learning (CT-RL) algorithms together standard Discrete-Time RL baselines. It includes environments for control tasks (DeepMind Control Suite) and financial trading.

## Installation & Setup

It is recommended to use a Conda environment to manage dependencies.

1. **Create and activate a Conda environment:**
   ```bash
   conda create -n ct-rl python=3.9
   conda activate ct-rl
   ```

2. **Install dependencies:**
   You can install the required packages using the provided `requirements.txt`.
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you need to generate a new requirements file based on imports, you can use `pipreqs`.*

## Training

The repository provides two main entry points for training: one for Continuous-Time algorithms and one for Discrete-Time algorithms.

### Continuous-Time RL
To train a continuous-time algorithm (e.g., CT-SAC), use the `benchmarks.run_ct_rl` module:

```bash
python -m benchmarks.run_ct_rl --algo ct_sac --env_id cheetah-run --log_root log_test --save_root save_test --total_timesteps 80000
```

**Parameters:**
*   `--algo`: The algorithm to use. Options: `ct_sac`, `ct_td3`, `cppo`, `q_learning`.
*   `--env_id`: The environment ID. Options: `cheetah-run`, `walker-run`, `humanoid-walk`, `quadruped-run`, `trading`.
*   `--log_root`: Directory to store TensorBoard logs and CSV metrics.
*   `--save_root`: Directory to save trained models.
*   `--total_timesteps`: Total number of timesteps to train.

### Discrete-Time RL
To train a standard discrete-time algorithm (e.g., SAC) using Stable-Baselines3, use the `benchmarks.run_discrete_rl` module:

```bash
python -m benchmarks.run_discrete_rl --algo sac --env_id cheetah-run --log_root log_test --save_root save_test --total_timesteps 80000
```

**Parameters:**
*   `--algo`: The algorithm to use. Options: `sac`, `td3`, `ppo`, `trpo`.
*   `--env_id`: Same options as above.

### Acrobot Stage A: oracle PH value control

Stage A learns capture and sustained upright balance from near-upright resets.
It uses the fixed Xin–Kaneda Acrobot mechanics, a smooth upright reward, and
HJB fitted value-flow updates. Only the return-value network is trained. The
elbow torque comes from its momentum gradient and is clipped to the physical
actuator bounds. This is a dedicated value-based experiment, separate from
the Q-based `ct_sac` trainer above. There is no learned dynamics model, LQR
initialization, or controller handoff.

```bash
MUJOCO_GL=disable python -m benchmarks.run_acrobot_stage_a \
  --mode stage_a_hard --output out/acrobot_stage_a/seed_0 --seed 0
```

Hyperparameters are read from `benchmarks/hyperparams/acrobot_ph_value.csv`,
using the same `mode`, `env_id`, `env_*`, `model_*`, `algo_*`, and `log_*`
column convention as the CT-SAC table. The rows `stage_a_hard` and
`stage_a_soft` select deterministic and entropy-regularized value control.
The row `stage_a_soft_auto` uses the same soft settings with automatic
temperature tuning enabled.
These are untuned starting presets. `--hyperparams-dir` selects another table
directory, and explicit CLI options override CSV values, for example
`--updates 20000 --learning-rate 0.0001`. Blank cells retain runner defaults.
The selected row, file hash, and effective settings are saved with each run.
Checkpoint evaluation uses the checkpoint's physical/model settings rather
than reloading today's training preset.

The defaults use a 20 N m torque limit, zero damping, a 10 ms control interval,
1 ms MuJoCo integration, 5 s episodes, and a 1 s hold criterion. Resets sample
each angle within 0.05 rad of upright and each velocity within 0.1 rad/s of
rest. Capture requires both angle errors below 0.1 rad and both velocities
below 0.25 rad/s. Episodes continue after capture. State-limit exits incur an
absorbing failure cost; time limits are evaluation truncations. These defaults
are an initial experiment configuration, not a validated stabilizing policy.

The deterministic formulation is the default. `--mode stage_a_soft` selects the
soft value operator and an analytic truncated-Gaussian training policy;
`--quadrature-points` controls integration accuracy. Evaluation always uses
the bounded deterministic mode, and records this choice. For deterministic
training, `--exploration-std` adds collection noise in normalized action units.
`--value-step` is the value-iteration time step, distinct from physical `--dt`.

Each run writes `config.json`, `training.jsonl`, `evaluations.jsonl`, `best.pt`,
and `last.pt`. Evaluation uses fixed held-out seeds and reports capture,
terminal retention, dwell times, effort, saturation, and state-limit failures.
Best-checkpoint selection prioritizes terminal retention, then terminal hold
duration, then return. Output directories must be empty to prevent accidental
replacement of another experiment.

```bash
MUJOCO_GL=disable python -m benchmarks.run_acrobot_stage_a \
  --checkpoint out/acrobot_stage_a/seed_0/best.pt \
  --output out/acrobot_stage_a/seed_0_eval --eval-episodes 32
```

For incoming swing-up resets, add `--incoming-states train.npz` and
`--incoming-eval-states heldout.npz`. Each archive must contain a finite
`states` array of shape `[N,4]` ordered `[q1,q2,v1,v2]`, and a scalar string
`frame` equal to `downward_vertical_qv` (upright shoulder at pi) or
`xin_kaneda_qv` (upright shoulder at pi/2). Velocities are retained. Split by
source trajectory before exporting: exact overlapping states are rejected,
but that check alone cannot detect neighboring samples from the same rollout.
Training mixes these incoming resets with local resets; evaluation reports
the two groups separately. Without archives, the runner tests local balance
only and makes no claim about capture from previously learned swing-ups.

The critic update shares CT-SAC's Bellman/HJB foundation. On nonterminal states
with a reward rate, the first-order oracle CT-SAC branch fits a separate
`Q(z,u)` to `V(z) + T [r(z,u) + grad(V)·f(z,u) - beta V(z)]` at replayed
actions. Stage A instead maximizes the expression inside brackets (or uses its
soft integral) and fits `V(z)` directly with a value-flow step `Delta_tau`.
The existing CT-SAC value head, when enabled, is fitted to the actor's soft
expectation of target Q. Stage A has no separate Q critic or learned actor.
CT-SAC's model-free branch estimates the discounted value change from actual
successor states; when the sample duration equals its reference interval, it
reduces to the ordinary soft Bellman backup. These are related constructions,
not identical finite-step learning updates.

```bash
MUJOCO_GL=disable python -m unittest tests.test_acrobot_stage_a -v
```

New Stage A runs terminate when the unwrapped shoulder reaches `pi/2` or
`3*pi/2`, with upright at `pi` in the downward-vertical frame. Resets and incoming
states outside that open interval are rejected. `--shoulder-limit` specifies
the allowed deviation from upright (default `pi/2`), also exposed as
`env_shoulder_limit` in the CSV. The absorbing failure cost accounts for the
smaller shoulder-angle cost bound. Joint speeds must stay strictly below
`2*pi` rad/s and the unwrapped elbow angle strictly between `-pi` and `pi`.
These are `--velocity-limit` and `--elbow-limit`, with matching CSV columns.
Existing checkpoints
without this field retain their original unrestricted shoulder domain when
evaluated or used to initialize a diagnostic.

Stage A presets use `--reward-scale auto` (`env_reward_scale=auto` in the CSV).
Before training, this calculates one fixed multiplier from the reward weights,
torque bound, and state limits. With current defaults the multiplier is
`1 / 25.7568575204 = 0.0388246120`. It scales all angle, speed, and torque costs,
including the effort coefficient in the analytic controller, and the absorbing
failure continuation. Ordinary reward rates are bounded by `[-1, 0]` inside the
allowed region, and the failure-value target is `-10` at discount rate `0.1`.
An ordinary 10 ms reward is approximately `[-0.01, 0]`; a failure step includes
the additional discounted absorbing continuation and may cross a threshold
slightly before termination is detected. No reward clipping is applied.

The same multiplier scales the initial online/target value outputs and the
configured soft temperature and its bounds. This preserves the initial analytic
policy and the relative weighting of reward and entropy. The entropy target and
optimizer learning rates are unchanged; numerical training trajectories need
not be invariant to reward scaling. `--reward-scale 1` disables normalization;
another positive number sets an explicit multiplier. Changing limits or raw
weights recalculates the automatic factor for a new run, never during training.
`config.json` and checkpoints record `reward_scale`, `unscaled_reward`, and the
effective scaled reward/value-flow settings. Checkpoint loading uses those
settings directly, with no second scaling. Standalone reward objects retain
their explicit coefficients; the runner applies the normalization when building
the experiment, including the fixed/moving diagnostic.

### Finite log state-cost reward

The `stage_a_hard_log`, `stage_a_soft_log`, and `stage_a_soft_auto_log` CSV rows
apply the finite transform to the combined angle and velocity cost:

$$\tilde\ell=C\frac{\log(1+\ell/\epsilon)}{\log(1+C/\epsilon)},\qquad
\tilde r=-\tilde\ell-\tfrac12w_u u^2.$$

Here $\ell$ is the normalized original state cost, $C\approx0.922350776$ is its
bound, and $\epsilon\approx0.001477394$ is its shoulder-only cost at a
$5^\circ$ deviation from upright, at rest with zero elbow angle. The transform
maps zero to zero and $C$ to $C$. It increases sensitivity to small state costs
while retaining a finite slope. The quadratic torque penalty, ordinary rate
bound `[-1, 0]`, and failure target `-10` remain unchanged with default limits.
The transform changes the control objective; improved balance requires an
experimental comparison with the original rows, which retain the identity map.

```bash
python -m benchmarks.run_acrobot_stage_a \
  --mode stage_a_soft_auto_log --output out/stage_a_soft_auto_log/seed0
```

Any row can also use `--state-cost-transform log`. The reference angle is
configurable through `--log-reference-angle-deg` (default `5`); it sets the
cost scale, not a capture tolerance. Matching CSV columns are
`env_state_cost_transform` and `env_log_reference_angle_deg`. The runner
calculates $C$ and $\epsilon$ once using the run's weights, limits, and reward
scale, and records both in configuration and checkpoints. The simulator and
HJB operator use the same transformed state cost; the analytic action rule
retains its quadratic effort coefficient. The fixed/moving target diagnostic
also accepts these presets (automatic temperature remains excluded there).

### Automatic temperature for soft Stage A

Temperature is fixed by default. To tune it toward a target policy entropy:

```bash
MUJOCO_GL=disable python -m benchmarks.run_acrobot_stage_a \
  --mode stage_a_soft_auto --output out/stage_a_auto/seed0
```

After each value update, a separate optimizer adjusts `log(temperature)` using
the analytic policy's entropy on nonterminal minibatch states. It increases
temperature when entropy is below the target and decreases it when above.
Entropy is differential entropy relative to normalized action `a` in `[-1,1]`;
negative values are valid, and targets must be below the maximum `log(2)`.
The configured bounds are `--temperature-min 0.0001` and `--temperature-max 10`
in unscaled reward units; they and the initial temperature are multiplied by
the run's reward scale. Logs report temperature in the effective scaled units.
Automatic tuning requires a positive initial temperature, so use the soft mode.

The updated temperature enters both the next soft HJB target and stochastic
action sampling. The deterministic evaluation mode at a given value gradient
is independent of temperature. Checkpoints preserve the learned temperature and
its optimizer; training logs include temperature, entropy, entropy error, and
temperature loss. The CSV exposes matching `algo_` columns; `stage_a_hard` and
`stage_a_soft` retain fixed temperature unless overridden. The adaptive preset
uses initial temperature `0.1` before reward scaling, target entropy `-1`, and
temperature learning rate `0.0003`. `--no-auto-temperature` disables tuning.

Entropy uses the same continuous-action quadrature as the soft HJB score. For
very concentrated policies, check sensitivity to `--quadrature-points`; a small
temperature floor alone does not guarantee accurate integration. Adaptive
temperature changes the soft objective during learning and is not a guarantee
of stabilization or a fix for value-learning instability.

### Stage A fixed versus moving target diagnostic

This diagnostic requires fixed temperature; automatic tuning is rejected to
keep the comparison focused on value-target feedback. It compares fitting labels calculated once with the usual bootstrapped HJB
update. Both arms start with identical value, target, and Adam states and use
identical minibatch indices. No dynamics model is learned. Start from the same
CSV presets as Stage A:

```bash
MUJOCO_GL=disable python -m benchmarks.check_acrobot_stage_a_targets \
  --mode stage_a_hard --output out/stage_a_targets/hard_seed0 --seed 0 \
  --train-states 4096 --heldout-states 2048 --collection-steps 4096 \
  --updates 10000 --eval-every 1000
```

Repeat with `--mode stage_a_soft` and independent `--seed`, `--dataset-seed`,
and `--batch-seed` values for additional replications. To branch from an existing
Stage A checkpoint, add `--checkpoint path/to/checkpoint.pt`. This initializes
both arms, including their optimizer histories; physical, reward, and value-flow
settings come from the checkpoint and override the CSV/CLI settings for those
components. `--updates` is the number of **additional** diagnostic updates.

Before fitting, the initial policy collects independent trajectories for training
and held-out data. Each fixed dataset contains half reset-distribution states and
half sampled visited states, preserving absorbing failure labels. Both arms then
train without new interaction; control-evaluation trajectories never enter the
datasets. `--collection-steps` is the number of decisions **per split**; increase
it if there are too few distinct visited states. `--collection-steps 0` runs a
reset-only diagnostic, which cannot diagnose replay or failure-boundary effects.
Incoming data require trajectory-disjoint `--incoming-states` and
`--incoming-eval-states` banks. `--buffer-size` has no effect on this diagnostic.

Outputs include:

- `dataset.npz`: fixed canonical states, terminal/source masks, and initial labels
  for both splits; `config.json` records effective settings, seeds, and hashes.
- `metrics.csv` and `probes.jsonl`: train/held-out label RMSE, error against the
  original labels, label drift, interior/boundary errors, online HJB residuals,
  value/state-gradient magnitudes, target disagreement, and action saturation.
- `training.jsonl`: minibatch losses and optimizer gradient norms.
- `fixed/` and `moving/`: checkpoints and per-state probe arrays at initialization
  and every evaluation interval, including the final update.
- `evaluations.jsonl`: the usual deterministic control metrics on fixed resets;
  `--skip-rollouts` omits these evaluations for a faster regression-only check.
- `summary.json`: completion status; `failures.json` records any nonfinite arm,
  while the other arm continues. The command exits nonzero after such a failure.

Falling training error with rising held-out error indicates overfitting to the
fixed assignment. If fixed-label fitting behaves well but moving-label fitting
deteriorates, target feedback is implicated under this frozen state distribution.
This does not establish the true value function, prove divergence, or reproduce
the changing replay distribution of online training. Moving-label RMSE measures
a changing assignment, so interpret it alongside label drift, fixed-state HJB
residuals, and control performance. HJB residual summaries exclude absorbing
terminal states; their boundary errors are reported separately. An empty boundary
subset has null error metrics and provides no evidence about boundary fitting.

```bash
MUJOCO_GL=disable python -m unittest tests.test_acrobot_stage_a tests.test_acrobot_stage_a_targets -v
```

## Reproducing Results & Reporting

This repository includes a comprehensive evaluation pipeline to generate plots, tables, and statistical tests.

### Large Assets & Pre-trained Models

The `logs`, `saved_models`, and `data/trading/processed_data` asset folders are too large to be included directly in the repository. They are hosted on SwissTransfer via 2 different links. Because these links expire every month due to SwissTransfer's policy, they will be updated here frequently for reproducibility:

Link1: [https://www.swisstransfer.com/d/6579e0c9-ee61-4bba-a010-933d22ff242b](https://www.swisstransfer.com/d/6579e0c9-ee61-4bba-a010-933d22ff242b)

Link2: [https://www.swisstransfer.com/d/146de302-80a7-4b20-920c-e89474e030c2](https://www.swisstransfer.com/d/146de302-80a7-4b20-920c-e89474e030c2)

Password: icml2026

Please download from these SwissTransfer links the following folders `logs`, `saved_models`, and `data/trading/processed_data` in order to get train logs, saved checkpoints, and trading processed features. After that, please proceed to the below steps for reproducing our reports.

### Performance Report
To generate the full performance report from trained logs:

```bash
python -m evaluations.performance_report
```

This script will read from your log directories and produce the following in `out/final_reports`:
*   **RL Evaluation Plots**: Learning curves comparing algorithms.
*   **Ablation Table**: Analysis of top performing hyperparameters.
*   **Significance Testing**: Statistical analysis using Welch t-test and paired t-test.
*   **Runtime Analysis**: Execution time statistics.

**Hyperparameters:**
Detailed hyperparameter tuning spaces for each of the 5 environments and 8 algorithms (one file per algorithm) can be found in:
`out/final_report/hyperparam_spaces`

### Evaluation on Regular Settings
To evaluate the performance of models trained under irregular time settings when deployed in a regular time setting:

```bash
python -m evaluations.evaluation_on_regular
```

## Trading Environment Data

If using the `trading` environment, you need to download and preprocess the data first.

1.  **Download Data**: Use `data/trading/download_data.py` to fetch data using the Alpaca API.
2.  **Process Data**: Use `data/trading/preprocess_data.py` to generate the feature sets required for the environment.

## Directory Structure

*   **`algorithms/`**: Implementations of all main continuous-time RL algorithms (CT-SAC, CT-TD3, CPPO, etc.).
*   **`benchmarks/`**: Contains the main training scripts (`run_ct_rl.py` and `run_discrete_rl.py`) and hyperparameter configurations.
*   **`common/`**: Shared utilities, including callbacks, replay buffers, and logger configurations.
*   **`data/`**: Logic for downloading financial data and processing it for the trading environment.
*   **`environment/`**: Implementations of the environments (DMC wrappers, Trading environment).
*   **`evaluations/`**: Scripts for generating reports, plots, and statistical analysis from saved logs and models.
*   **`models/`**: Definitions of neural networks (Stochastic policies, Value functions V, q, and Q=V+q) used by the algorithms.
*   **`tests/`**: Unit and integration tests.

## Media

Visualizations of agent performance (videos and images) are generated in `out/final_reports/media/images` or `out/final_reports/media/videos`.

### Sample Episodes

**Walker Sample Run**

The image below shows CT-SAC's performance over a single Walker episode, compared against continuous-time baselines

![Continuous-time RL Walker](sample_images/walker-run_ct_sac_vs_continuous_regular_last.png)

---

**Cheetah Sample Run**

The image below shows CT-SAC's performance over a single Cheetah episode, compared against discrete-time baselines

![Dicrete-time RL Cheetah](sample_images/cheetah-run_ct_sac_vs_discrete_regular_last.png)

---

**Trading Sample Run**

The image below shows CT-SAC's performance over a single Trading (2-weeks) episode, compared against SAC with $15,000 and $5,500 PnL respectively.

![Trading CT-SAC vs SAC](sample_images/trading_ct_sac_vs_sac.png)
