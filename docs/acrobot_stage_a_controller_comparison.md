# Controlled comparison of Stage A value-gradient controllers

`benchmarks.compare_acrobot_stage_a_controllers` isolates fixed regression,
derivative-based target feedback, and changing state coverage. It measures the
controllers produced by those training conditions. It does not introduce a new
controller or alter the Stage A training algorithm.

| Arm | Training states | Labels |
|---|---|---|
| `fixed_labels` (A) | Fixed training dataset | Computed once from the initial target network |
| `moving_labels` (B) | Same dataset and minibatch indices as A | Recomputed with the evolving target network |
| `online` (C) | Fresh resets plus growing visited-state replay | Recomputed with the evolving target network |

All three arms clone the same value network, target network, optimizer state,
and update counter. All share their first minibatch/update. A freezes its target
network; B and C retain the normal Polyak update. A and B receive exactly the
same indices thereafter, even when C's policy and data change.

C starts with the visited-state portion of the fixed training dataset. From
update two onward it collects one decision per update with its current policy,
stores both transition endpoints, and samples half fresh reset states and half
replay states. This matches the original runner's collection and sampling rule
after a **matched initial replay dataset**; it intentionally does not reproduce
the original runner's nearly empty replay at startup. Only actual state-limit
failures receive absorbing boundary labels; episode truncations do not.

The comparison is restricted to the **continuing objective** used by the
audited checkpoints. Specify `--no-finite-horizon` for a fresh run; a loaded
continuing checkpoint supplies its recorded objective and physics. Finite-horizon
models and automatic temperature updates are rejected because they change the
experiment being isolated. Fixed-temperature soft mode is supported.

## Run

From the repository root, run three seeds (nine training arms total):

```sh
.venv/bin/python -m benchmarks.compare_acrobot_stage_a_controllers \
  --output out/stage_a_controller_comparison \
  --mode stage_a_hard --no-finite-horizon --seeds 0 1 2 \
  --updates 10000 --probe-every 100 --eval-every 1000
```

Each seed gets its own directory and independently seeded collection and batch
streams. Within a seed, initialization is shared across conditions. Evaluation
uses the same held-out reset seeds across all conditions and training seeds.
Training and held-out datasets come from distinct frozen-policy trajectories;
neither held-out data nor evaluation trajectories enter the online replay.

For a quick smoke run including physical controller evaluation:

```sh
.venv/bin/python -m benchmarks.compare_acrobot_stage_a_controllers \
  --output out/stage_a_controller_comparison_smoke \
  --no-finite-horizon --updates 3 --hidden-width 8 \
  --train-states 16 --heldout-states 12 --collection-steps 16 \
  --batch-size 8 --buffer-size 16 --probe-every 1 --eval-every 2 \
  --eval-episodes 1 --episode-seconds .02 --hold-seconds .01
```

`--checkpoint PATH` initializes all three networks and optimizer states from
that checkpoint; `--updates` then counts additional updates. Use `--seed` for a
single run directly in the requested output directory. Nonempty output
directories are never overwritten. `--skip-rollouts` skips evaluation but keeps
dataset collection, online training, and all value diagnostics.

## Measurements

Probes run before training, every `--probe-every` updates, and at the final
update. Physical rollouts run at initialization, every `--eval-every`, and at
the final update. Each run writes:

- `config.json` and `dataset.npz`: parameters, seeds, dataset/checkpoint hashes,
  fixed training/held-out states, initial labels, and the upright probe states.
- `metrics.csv`: train/held-out fitting error, label drift, HJB residual,
  gradients, boundary error, and torque saturation for each arm.
- `upright.jsonl`: Hessian eigenvalue extrema, local closed-loop poles,
  sampled-controller spectral radius, numerical equilibrium torque, and RMS
  magnitudes of each HJB target contribution.
- `ARM/upright_UPDATE.npz`: full Hessians, pole spectra, physical torque
  Jacobians, and per-state value/reward/drift/discount/action-score/label/torque
  arrays for both online and target networks. Axis slices cover the configured
  reset radii around upright.
- `ARM/checkpoint_UPDATE.pt`: restartable agent checkpoints at probe times.
- `training.jsonl`: losses and collection time for every update.
- `evaluations.jsonl`: per-episode deterministic controller results.
- `online_replay_final.npz`: final replay contents and failure masks.
- `summary.json`: completion status and first probe with indefinite curvature.

Upright is `[pi,0,0,0]` in the network's canonical coordinates. Hessians are
also converted to physical errors `[q1-pi,q2,v1,v2]`. Curvature calculations use
a float64 copy of the stored network; training remains float32. Nominal
upright torque is measured separately through the float32 deployment path.
The local sampled-controller map uses the recorded decision interval and an
exact matrix exponential of the linearized oracle under held torque.

The indefinite flag requires eigenvalues below -1e-8 and above +1e-8 in the
reported physical coordinates; raw eigenvalues are also saved. Indefinite
curvature at update zero means the network started with a saddle.
Later probe times only bound the onset; they are not exact event times. A
negative-definite Hessian alone does not establish stabilization. Continuous
poles must have negative real parts and the held-controller spectral radius
must be below one for the corresponding local linear stability test. These
tests describe the mathematical unsaturated local feedback, not finite-radius
robustness or float32 behavior when enormous gains destroy the numerical
equilibrium. Rollouts provide the separate physical check.

If A behaves poorly, regression and approximation deserve attention. If A
stays controlled and B deteriorates, evolving derivative targets are implicated.
If B stays controlled and C deteriorates, online collection and changing state
coverage are implicated. Frozen labels are not the optimal value, and fitting
them successfully is not a stabilization result. Follow a reproducible failure
with one intervention at a time; these comparisons do not prove a unique cause.
