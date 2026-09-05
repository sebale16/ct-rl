# Experiment 3: generalization across Acrobot configurations

This setup tests whether model-free CT-SAC meets the generalization objective
before introducing CPHT-SAC. It implements three baselines:

| Arm | Training | Actor and critic inputs | Evaluation |
|---|---|---|---|
| Frozen | Existing nominal checkpoint; no updates | Original four state coordinates | Nominal and unseen plants |
| Randomized | One policy across 32 fixed training plants | Four state coordinates | Same unseen plants |
| Conditioned | Same training distribution and budget | State plus four physical parameter factors | Same unseen plants |

`nominal` optionally trains a fresh nominal control under the same settings.
`specialist` trains a separate state-only policy for one configuration, to check
whether a shared-policy failure occurs on a plant that is individually solvable.
Specialists trained on test configurations are diagnostic controls, not held-out
generalization results.

## Fixed protocol

The checked-in [manifest](../benchmarks/configs/acrobot_experiment3.json) fixes
all parameter tuples, initial-state seeds, and the source hyperparameter row.
[The runner](../benchmarks/acrobot_experiment3.py) rejects a changed source row
until a new manifest is explicitly generated and reviewed.

- Vary each link's mass and length independently. COM distances scale with link
  length; inertias scale with mass times length squared. Gravity remains 9.8,
  damping remains zero, and actuator capacity remains 20 N m.
- Training: 32 parameter tuples drawn from `[0.8, 1.2]` times nominal values.
  Draw one uniformly at each episode reset and hold it fixed through the episode.
- Validation: 8 disjoint tuples within that range, reserved for development.
- Interpolation test: 16 additional disjoint tuples within the training range.
- Extrapolation test: 16 tuples with exactly one factor in `[0.6, 0.75]` or
  `[1.25, 1.4]`; the other factors remain in `[0.8, 1.2]`. This is a modest
  extrapolation test, not a test of arbitrary simultaneous parameter changes.
- Evaluation: 32 release-from-rest starts, seeds 20000–20031, for each plant.
  The existing working training mode uses shoulder displacement in `[0.05, 0.5]`,
  straight elbow, and zero velocity. Evaluation uses that same distribution.
- Control period 10 ms, physics period 1 ms, episode horizon 20 s. Training uses
  400,000 decisions (nominally 4,000 simulated seconds), including 20,000
  demonstration decisions (200 seconds). Actual simulated and demonstration
  seconds are recorded, including any shortened transitions.
- Use the existing successful demonstration-only `r3` mode: eta 0.23, discount
  rate 0.5 s⁻¹, log-reciprocal reward, initial temperature 0.01, target update
  tau 0.0125, and no ongoing imitation loss. Architecture and other settings are
  copied from that row for both trained arms.

Energy targets, reward normalization, shoulder-rate termination thresholds,
terminal reward lower bounds, and trajectory metrics use each plant's actual
mechanics. Reward gains remain fixed at the base mode's values. The
`xk_closed_loop` reward-rate term remains the existing action-independent
surrogate; it is not a stability guarantee for the learned policy.

Demonstrations use the live plant's dynamics, with k_D and k_P raised to 1% above
their plant-specific admissibility floors when necessary; k_V stays fixed.
Both trained arms have identical access to these demonstrations. The demonstrator
may use physical parameters even when the learned state-only actor cannot.
The fixed torque bound can still prevent demonstration capture on a changed
plant. The simulator remains conservative; this protocol does not vary damping.

The conditioned observation is
`[q1, q2, qdot1, qdot2, mass1-1, mass2-1, length1-1, length2-1]`, where the last
four entries are multipliers relative to nominal. They also enter the critics.
These four factors determine all varying mechanics under this scaling rule.
The state-only arm gets no configuration ID or parameter context. No history
encoder or online identification is added.

## Run the baselines

Run commands from the repository root using the existing virtual environment.
Each training output directory must be new. The commands below use seed 0;
repeat the trained arms with seeds 1–5 for the six-seed comparison.

```sh
# Baseline 1: evaluate an existing nominal policy with the matching base mode.
.venv/bin/python -m benchmarks.acrobot_experiment3 evaluate \
  --checkpoint /path/to/nominal/final_model.pth --training-seed 0 \
  --output results/experiment3/frozen_seed0.json

# Baseline 2: train one state-only policy on the training configurations.
.venv/bin/python -m benchmarks.acrobot_experiment3 train \
  --arm randomized --seed 0 --output saved_models/experiment3/randomized_seed0

# Baseline 3: train with physical parameters appended to the observation.
.venv/bin/python -m benchmarks.acrobot_experiment3 train \
  --arm conditioned --seed 0 --output saved_models/experiment3/conditioned_seed0

# Evaluate each final checkpoint with identical test configurations and starts.
.venv/bin/python -m benchmarks.acrobot_experiment3 evaluate \
  --run saved_models/experiment3/randomized_seed0 \
  --output results/experiment3/randomized_seed0.json
.venv/bin/python -m benchmarks.acrobot_experiment3 evaluate \
  --run saved_models/experiment3/conditioned_seed0 \
  --output results/experiment3/conditioned_seed0.json
```

For a checkpoint trained with another reward/hyperparameter row, create a separate
manifest and pass it to **all** arms with `--manifest`. The checkpoint loader
checks architecture strictly; it cannot infer a checkpoint's training conditions,
so select its actual base mode. To reproduce the original nominal policy under
the default budget, use `train --arm nominal`.

```sh
.venv/bin/python -m benchmarks.acrobot_experiment3 prepare \
  --base-mode YOUR_EXISTING_MODE --output /tmp/experiment3-other-mode.json

# Diagnostic control after identifying a difficult configuration.
.venv/bin/python -m benchmarks.acrobot_experiment3 train \
  --arm specialist --configuration extrapolation_000 --seed 0 \
  --output saved_models/experiment3/specialist_extrapolation_000_seed0
```

The default selection rule is the final checkpoint. Training writes intermediate
weights every 20,000 decisions, an episode log of sampled configurations and reset
seeds, and `run.json` with the manifest, training settings, and realized budgets.
These are weights checkpoints, not resumable optimizer/replay snapshots. Evaluation
never updates weights. Use `--splits validation` for development; reserve the
interpolation and extrapolation tests until the protocol is fixed.

Evaluation JSON contains every episode's reward-independent metrics 1–6,
termination reason, physical parameters, per-configuration summaries, pooled
split summaries, worst-configuration capture rate, checkpoint hash, and protocol.
Undefined/non-finite metrics are `null`; `captured=false` distinguishes failures.
Compare success, retention, capture time, and effort separately for interpolation
and extrapolation. Inspect termination and saturation rates before attributing
failure to insufficient policy structure. Aggregate across independent training
seeds; the 32 initial-state evaluations are not 32 independent training runs.
This setup does not compute metric 7 learning curves across configurations.

## Quick verification

`--smoke` explicitly uses tiny networks, 32 decisions, eight demonstration
decisions, and short episodes. These outputs are labeled smoke runs and do not
measure learning performance. The evaluator reads their architecture from
`run.json`.

```sh
.venv/bin/python -m benchmarks.acrobot_experiment3 train \
  --arm conditioned --smoke --device cpu --output /tmp/experiment3-smoke
.venv/bin/python -m benchmarks.acrobot_experiment3 evaluate \
  --run /tmp/experiment3-smoke --splits nominal interpolation extrapolation \
  --seeds 20000 --horizon 0.04 --device cpu --output /tmp/experiment3-smoke.json
MUJOCO_GL=disable .venv/bin/python -m unittest \
  tests.test_acrobot_experiment3 tests.test_acrobot_xk_rewards \
  tests.test_acrobot_xk_termination -q
```

Success of the simpler baselines would reduce the motivation for CPHT-SAC on this
parameter range. Failure motivates specialist controls and subsequent CPHT-SAC
comparisons; it does not establish necessity. This experiment measures homoclinic
capture and retention, not completed upright stabilization.
