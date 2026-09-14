# Stage A oracle PH value learning: pilot logs

Logs only. Model checkpoints (`best.pt`, `last.pt`) stay in the run directories
under `out/` and are deliberately not committed.

Per run: `config.json` (full provenance -- resolved hyperparameters, seeds,
environment, torch version), `evaluations.jsonl` (one record per evaluation),
`training.jsonl` (per-100-update `value_loss`, `hjb_rms`, `eta_abs_mean`,
`gradient_norm`). `job_stdout.txt` is the batch job's own summary.

Both jobs ran on the LS6 `development` queue, `stage_a_hard` unless noted,
three seeds per arm, at the checked-in preset otherwise.

## `dev_3437100` -- baseline, both presets

Six runs: `stage_a_hard` and `stage_a_soft`, seeds 0-2, at the CSV budget of
10k updates. Establishes the baseline and the wall-clock cost.

Every run peaks mid-training and then collapses. `saturation_fraction` rises
close to monotonically to 0.80-0.93 as `success` falls back to zero.
`retained_success` is 0.000 at every checkpoint of every run: capture is always
transient. `return` does not track `success` and is not a usable model-selection
signal here -- `stage_a_hard_seed0` has its best return at 1k, when success is
still zero, and a worse return at its success peak than at its start.

## `sweep_3437316` -- target rate and value step

Twelve runs, four arms, all `stage_a_hard`. Total value-time
`value_step * updates` is held at 200 across every arm, so the arms are
compared at equal progress along the value flow rather than equal update count.

| arm | target_rate | value_step | updates |
|---|---|---|---|
| `A_baseline` | 0.01 | 0.02 | 10000 |
| `B_tau0p0125` | 0.0125 | 0.02 | 10000 |
| `C_vstep0p01` | 0.01 | 0.01 | 20000 |
| `D_vstep0p005` | 0.01 | 0.005 | 40000 |

`B` sets the target rate to CT-SAC's convention on acrobot. CT-SAC's table
converges on a 0.80 s physical target lag from four different
`(dt, tau, train_freq)` routes; Stage A runs 10 ms with one update per env step,
so parity is `tau = E[dt] / 0.80 = 0.0125` against the default 0.01, which is a
0.997 s lag. This makes the target track faster, so it is an alignment for
comparability, not a fix. It collapses ~19% earlier than `A` (mean crossing
3767 vs 4633 updates) at indistinguishable peak success.

`C` and `D` test whether the collapse is an explicit-Euler step-size
instability in `dV/ds = H[V]`. It is not, and it is not operator-intrinsic
either. Mean update at which `|eta|` crosses the saturation threshold
`effort_weight * torque_limit = 0.2`:

| arm | crossing (updates) | ratio vs A | crossing (value-time) |
|---|---|---|---|
| `A_baseline` | 4633 | 1.00 | 92.7 |
| `B_tau0p0125` | 3767 | 0.81 | 75.3 |
| `C_vstep0p01` | 6367 | 1.37 | 63.7 |
| `D_vstep0p005` | 7200 | 1.55 | 36.0 |

Step-size instability would put the crossing at a fixed value-time, requiring
ratios of 2.0 and 4.0 in update count; the observed 1.37 and 1.55 are far short,
and the value-time crossing instead falls monotonically. `|eta|` at the end of
training is *higher* for the smaller steps (`D` ends at 2.03-3.01 against `A`'s
1.00-1.77) because `D` runs four times the updates. The divergence therefore
scales with the number of network updates -- the regression and the Polyak
target chase -- not with progress along the flow, which makes it a fitted
approximation instability rather than a numerical integration one.

`retained_success` is 0.000 across all 18 runs in both jobs.
`C_vstep0p01_seed0` is the single exception to the collapse: it ends at
`success` 0.500 at update 20000, peaking at its final evaluation, with the
lowest end-`|eta|` of the small-step arms. Its two siblings collapse normally,
so this is one seed of three, not an arm effect.

## `targets_3438282` -- fixed vs moving label diagnostic

`benchmarks.check_acrobot_stage_a_targets`. Six runs, both presets, seeds 0-2,
10k updates on each of two arms. Both arms are cloned from the same value,
target and Adam state and fed identical minibatches drawn from one frozen
dataset; the only difference is whether the regression labels move. Fixed
temperature throughout, which the diagnostic requires in order to isolate
value-target feedback.

This is the experiment `sweep_3437316` pointed at. That sweep showed the
divergence scales with the number of network updates rather than with progress
along the value flow, implicating the regression and the moving bootstrap
target rather than the Euler discretization. Freezing the labels removes the
bootstrap while changing nothing else.

Means over three seeds, training split, at update 10000:

| preset | arm | mean \|eta\| | max \|eta\| | saturation | label drift |
|---|---|---|---|---|---|
| hard | fixed | 0.016 | 3.54 | 0.009 | 0 |
| hard | moving | 2.009 | 24.65 | 0.836 | 67.5 |
| soft | fixed | 0.316 | 5.03 | 0.238 | 0 |
| soft | moving | 2.393 | 18.99 | 0.889 | 45.5 |

The moving bootstrap target is the cause. With labels frozen, mean `|eta|`
stays two orders of magnitude below the moving arm and well under the
saturation threshold `effort_weight * torque_limit = 0.2`; saturation is 0.9%
against 83.6%. Same states, same batches, same initialization, same optimizer
state. This also rules out the dataset, Adam, and the regression itself: a
fixed-label regression on these exact states stays bounded.

Two metrics are actively misleading here. `label_rmse` is *lower* in the
moving arm (12.5 against 18.6, hard) -- it fits its labels better while
diverging, because the labels are running toward it. `hjb_rms` is *higher* in
the fixed arm (154 against 102), as it must be, since frozen labels cannot
satisfy the HJB equation the moving labels are chasing. Neither separates the
two arms in the direction one would naively expect.

The fixed arm is not perfectly tame either: max `|eta|` reaches 3.54 and max
state-gradient norm 376, so a tail of states still develops large gradients
without any bootstrap. It just never spreads to the bulk. The soft preset's
fixed arm sits between the two (mean `|eta|` 0.316, saturation 0.238), so the
entropy term contributes some growth on its own, well short of the bootstrap.

Per-checkpoint `.pt` files and the per-probe `.npz` arrays (11.2 MB and
19.8 MB) stay in `out/`. `metrics.csv` carries the per-state probe summaries
(`eta_abs_mean`, `eta_abs_max`, `saturation_fraction`, gradient norms) for both
splits at every checkpoint, and `config.json` records the dataset seeds, so the
frozen dataset regenerates deterministically.
