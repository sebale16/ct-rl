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
