# Performance-gated tip-height curriculum

Three task IDs share the same reset curriculum while retaining their original
reward functions:

- `acrobot-swingup-v4.3`: the v4.1 capture-pressure reward;
- `acrobot-swingup-v6.1`: the v6 AR-EAPO quadratic cost;
- `cartpole-two_poles-v2`: the stock dm_control serial two-pole CartPole
  swing-up reward.

The prior Acrobot v4.2/v6 curriculum widens an angle band as a function of
global training steps. That changes potential and kinetic energy only
indirectly, reaches uniform random angles rather than the hanging state, and
can make the task harder before the policy has learned the current reset.
Those historical IDs remain unchanged.

## Reset ladder

A level is specified by two physical quantities: world-frame tip height and
Cartesian tip speed. It selects a family of poses rather than one pose, because
every reset also draws a mirror side and a relative angle between the two
links.

1. The first level starts at 99.5% normalized tip height and at rest, a small
   mirrored displacement from vertical that requires active recovery.
2. Every level has exactly zero starting velocity; subsequent levels
   monotonically lower the tip.
3. The last level hangs at zero velocity and remains the training distribution
   after it is reached.

Default Acrobot levels are `(height, speed)` =
`(3.98, 0), (3.5, 0), (3.0, 0), (2.0, 0), (1.0, 0), (0.0, 0)`.
The serial two-pole CartPole uses `(2.98, 0), (2.5, 0), (2.0, 0),
(1.0, 0), (0.0, 0), (-1.0, 0)`. Heights and speeds are in metres and
metres/second. In both mechanisms the unfolded first pose is about 8.1 degrees
from vertical and 0.283 m from the stabilization point, outside the 0.2 m
capture radius.

Both mechanisms are two unit links on a pivot — the Acrobot shoulder at
`z = 2`, the cart at `z = 1` — with the second hinge relative, so
`z_tip = z_pivot + cos(q1) + cos(q1 + q2)`. Folding by `q2 = e` leaves the tip
on the bisector at `2 cos(e/2)` from the pivot, so the requested height needs

    q1 = arccos(offset / (2 cos(e/2))) - e/2,

for `offset` the height above the pivot, and a rigid rotation of the folded arm
delivers the requested tip speed at that shortened radius. Mirroring negates
both angles. No reset noise is added.

`e` is drawn uniformly from `±30` degrees, except where folding would hand the
agent a start it has already reached: a fold shortens the arm, so near the
stabilization point it moves the tip toward the goal rather than around it. The
spread at each level is therefore capped at the fold whose tip sits exactly on
the 0.2 m capture radius, which binds only at the near-upright first level and
narrows it to `±11.5` degrees. Deep folds cannot span the full height range,
so at the hanging level the drawn pose splays symmetrically about the vertical
with its tip up to 0.068 m above the lowest one; every other level keeps its
tip height exactly. The fixed evaluation reset is unaffected and remains the
exact hanging state.

## Mastery gate

Curriculum progress has no timestep schedule. At the normal evaluation
frequency, a separate deterministic probe evaluates the current level. In both
mechanisms an episode passes only if the tip stays within 0.2 m of the
stabilization point and below 0.2 m/s for five uninterrupted physical seconds
and remains captured through the episode endpoint. A five-second hold followed
by a fall does not pass. The default level threshold is an 80% pass rate over
the probe episodes. A passing probe advances exactly one level; failed probes
leave the level unchanged.

The changing probe is not used for checkpoint selection. The primary
evaluation environment is fixed at the final hanging-at-rest task, so
best-model scores remain comparable throughout training. Performance
curriculum state is also included in resumable CT checkpoints.

## Curriculum telemetry

Both the continuous-time and SB3 runners write the selected curriculum state
under `curriculum/` in TensorBoard, JSON, and CSV logs. Important curves are:

- `stage`, `num_stages`, `progress`, `complete`, and `advanced`;
- `probe_stage`, `probe_success_rate`, `probe_passed`, and
  `consecutive_passes`;
- `start_tip_height`, `start_tip_height_norm`,
  `start_potential_energy_norm`, `start_tip_speed`, and
  `start_elbow_spread`.

On an advancement event, `probe_stage` is the level that was just mastered,
while `stage` and the physical reset descriptors identify the newly selected
level. Normalized potential and tip speed remain separate fields so the log
schema describes the physical reset directly. `start_potential_energy_norm` is
the unfolded chain's potential, the level's reference pose; a fold raises the
inner link above it at the same tip height, by an amount `start_elbow_spread`
bounds. The historical angle-band
curricula instead report `fraction`, `progress`, and `angle_spread_rad`; a
sampled reset distribution does not have one deterministic energy value.

Render all six reset stages for Acrobot v4.3/v6.1 and the serial double-linked
CartPole v2 with:

```bash
MUJOCO_GL=egl python -m benchmarks.render_tip_curriculum_stages
```

This writes `videos/tip_curriculum_stages.mp4`. Each stage pauses on its exact
reset pose and then runs a labeled zero-action release; no trained policy is
used in the preview.

The gate can be tuned without changing task definitions:

```bash
python -m benchmarks.run_ct_rl \
  --algos ct_sac \
  --env_id acrobot-swingup-v6.1 \
  --mode final_mf \
  --curriculum_success_threshold 0.8 \
  --curriculum_consecutive_evals 1
```

The v6.1 branch also mirrors v6's fixed-temperature entropy ablations:
`fixed_a2p0`, `fixed_a0p5`, and `fixed_a0p1`. They are identical to the
v6 rows except for the environment ID and set `algo_alpha` to `2.0`, `0.5`,
and `0.1`, respectively.

All three new task IDs have the same algorithm coverage as Acrobot v4.2:
CT-SAC (`final_mf` and `final_oracle_rollout`) plus CT-TD3, PPO, and SAC
(`final_mf`). The new CartPole rows exactly mirror the existing
`cartpole-two_poles-curriculum` settings for each algorithm.
