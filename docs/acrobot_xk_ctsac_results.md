# CT-SAC on acrobot-swingup-xk: results

Runs on Frontera, `run_id` in each heading. Every number here is reproducible from
`benchmarks/hyperparams/ct_sac.csv` plus the committed CSVs under `results/`;
the training logs and checkpoints are not committed (`.gitignore` excludes
`logs/*` and `saved_models/*`).

## Summary

Across ~200 cells and four launches, **CT-SAC achieved sustained capture in
exactly one of ~1500 evaluation rows** — one protocol episode, one seed, one
checkpoint. The analytical Xin–Kaneda controller captures on 32/32 protocol
seeds under the identical battery. The gap is a stabilization failure, not a
swing-up failure: every episode reaches the correct energy and none holds the
pose.

## The reference: the analytical controller (`results/xk_baseline/`)

`evaluations/eval_acrobot_xk.py`, protocol seeds 20000–20031, dwell 1.0 s.

| cell | P(capture) | capture time | rho_sat | peak τ | notes |
|---|---|---|---|---|---|
| `paper.csv` | 1.00 | 6.87 s | 0 | 19.6 | paper IC, dt 1e-4 — positive control |
| `matched.csv` | **1.00** | median 11.06 s | **0** | 19.4 / 64 | the RL protocol exactly |
| `long.csv` | 1.00 | median 11.06 s | 0 | 19.4 | 20 s cap lifted to 120 s |

So the 1 s bar is reachable on this plant, in a 20 s episode, at under a third
of the available torque, with zero saturation. `long.csv` shows the 20 s cap was
never binding (all 32 seeds capture before 16.55 s) and that the controller keeps
tightening when given more time: energy error 0.0422 → 0.0193, LQR residual
0.398 → 0.148. Peak demand over all 32 seeds is 19.70 N·m, so **gear 20 is the
floor at which the reference solution still fits**, with 1.5 % margin.

## `xk_uniform_v2` — the eight reward arms (job 7894354)

Eight `_fixed0p5ms` arms (r0, r1, r2 at η ∈ {0, .01, .03, .1, .3, 1}) × 6 seeds ×
1M steps. Per-policy metric battery in `results/xk_ctsac_eval/`.

- **No arm captured.** Best continuous hold 0.34 s against the 1.0 s bar.
- **r2(η=0) is bit-identical to r1** on all six seeds (every non-timing column),
  confirming η plumbing and determinism.
- **11/48 cells diverged** (|V| > 1e4), rising with η: 1/6 at η=.01, 2/6 at .03,
  1/6 at .3, 2/6 at η=1. r0 and r1 never diverged.
- **`rho_sat` = exactly 0 for seven of eight arms** across 192 episodes each.
  Only η=0.3 saturates, and only on seed 2 (78 % of the time, all 32 episodes) —
  so torque saturation is a single-seed pathology, not an η-driven incentive.
- Every RL arm commands 58–64 of 64 peak against the controller's 19.4, and uses
  2.5–42× its control effort, without capturing.

## `xk_gamma_v1` — discount horizon × reward bounding (job 7895458)

The discount is dt-invariant: `beta = -log(gamma)`, `gamma_dt = exp(-beta·dt/dt_default)`
with `dt_default = control_timestep() = 0.01`, so the horizon is `dt_default/beta`
regardless of the sampling dt. Verified against the constructed `CTSAC` object and
by compounding `gamma_dt` over a second of simulation:

| γ | horizon | γ=0.995 was the shipped value |
|---|---|---|
| 0.995 | 1.995 s | |
| 0.998 | 4.995 s | |
| 0.9991 | 11.106 s | ≈ the controller's median capture time |
| 0.9995 | 19.995 s | ≈ the whole episode |

**The task takes 11.06 s and the shipped horizon was 2.0 s.** All 48 cells reached
1M. `xk_r1_g0p995_raw` reproduces `xk_r1_fixed0p5ms` bit-identically, so the grid
is anchored.

`reward_squash = V0` (see below) helps at 2 s and **hurts at every longer
horizon**: at 11.1 s the squashed arms park at V ≈ 4e9 with 5/6 seeds there at
once, against 1/6 for the unsquashed arm. Recovery from an excursion falls from
67 % (raw) to 17 % (squashed). The long horizon triggers the excursion; the
squash makes it permanent.

## `xk_spin_v1` / `xk_spinraw_v1` — spin termination at gear 20 (jobs 7895627, 7895630)

`spin_limit = 2π` ends the episode when the elbow winds a full turn; gear 64 → 20.

Neither helped. Best mean dwell fell to 0.054 s (gated) and 0.035 s (non-gated),
against 0.329 s for the plain unmodified arm. The 24 `reward_offset` (`_pos`)
cells were cancelled at ~700k after reaching **zero capture on every seed** with
α collapsed to 1e-23 and episodes dying at 890–1445 steps.

**The sign trap, and where my sizing of it was wrong.** With r1 = −s(V) ≤ 0,
terminating *avoids* future negative reward, so a spin limit is an incentive to
spin. Undiscounted, a spinner returns −9.8e5 against −2.35e7 for a passive
episode — quitting gains 2.26e7. But the agent optimises a *discounted* return
over 3990 steps, not 40000, so the offset's deterrent is worth only
`r/(1-gamma_dt)` ≈ 1e6, while `_neg`'s explicit penalty is 2.3e7 immediate and
undiscounted — **23× stronger**. The `_neg` arms duly survived 13k–27k steps
against `_pos`'s 890.

## The ceiling, diagnosed (`results/xk_ceiling/`)

16 checkpoints × 32 protocol episodes = 512 episodes. Exactly one captured:
`xk_r1_g0p995_sq` seed 3 @ 300k, protocol seed 20023, **1.796 s dwell**.

- **Energy is solved.** 32/32 episodes of that checkpoint reach |Ẽ| < 0.05 J at
  some instant; the minimum is 5.5e-7 J against a 49 J span.
- **The exit is a pose failure.** Through the dwell |Ẽ| never exceeds 0.9 % of
  tolerance while the elbow angle drifts monotonically 51 % → 99 % of tolerance;
  a rate excursion then crosses tolerance and ends it. The angle drift is the
  failure, the rate crossing is the trigger.
- **The policy coasts where the controller regulates.** Torque during the dwell:
  mean 0.4621 (policy) vs 0.4530 (controller) — nearly identical — but rms
  0.6839 vs **1.8924**. The controller's torque is mostly AC, the policy's mostly
  DC. The homoclinic orbit is unstable; holding it requires modulation.
- **The in-tube signal is 4 orders below the training signal.** V = 0.219 inside
  the tube against a reward range of 1200, i.e. 0.018 %, and 2733× smaller than
  the ~600-magnitude rewards that shape the first 300k steps.

### Reward conditioning, measured on that trajectory

Ratio = |typical reward| ÷ |in-tube reward|; lower means less dynamic range for
the critic to resolve "holding" from "drifting".

| reward | ratio |
|---|---|
| r0 | **69** |
| r1 raw | 94 |
| r1 squashed | 87 |
| r2 η=0.03 | 71 |
| r2 η=0.3 | 22 |
| r2 η=1 | **11** |
| r2 η=1, chain-rule squash | **9.7** |

Two findings worth carrying forward:

1. **r0 is the best-conditioned plain reward**, because its per-term
   normalization weights the shape coordinates far more than r1's physical gains
   do: angle/energy 243 vs 61, rate/energy 114 vs 36 — 4.0× and 3.2× more weight
   on exactly the coordinates that fail. Consistent with r0 being the only arm
   with any capture in `xk_uniform_v2` and the only one whose α stayed sane
   (0.066 vs 1e2–1e6).
2. **η improves in-tube conditioning monotonically** (94 → 11), because V is flat
   near the target while V̇ is not. The η sweep was aimed correctly and failed on
   an unrelated axis: η·V̇ is unbounded and broke the optimisation globally
   before the local benefit could be used.

The chain-rule form `−s(V) − η·s'(V)·V̇` keeps both: s'(V) = 0.9982 at the
pre-exit state, so 99.8 % of the shaping survives near the target, while the same
term is damped 7106× at V = 1e5. Note that squashing V and V̇ *independently*
would break the identity that the rate term is the time derivative of the
potential, which is what the whole Lyapunov argument rests on.

## Reward knobs added to the task

All default to off, so existing rows are bit-identical.

- `reward_squash = V0` — `r1 = -V/(1 + V/V0)`, bounded by V0, strictly
  increasing, tangent to −V at V = 0. `DEFAULT_REWARD_SQUASH_V0 = 1200.5`, which
  is V at hanging: the links are aligned and at rest there, so V is pure energy
  error, `½·span² = ½·49²`. (`q = 0` is the *horizontal* configuration in this
  model's coordinates; hanging is `q1 = -π/2`.)
  **Order preservation is not incentive preservation.** s' decays as (V0/V)², so
  far outside V0 the marginal cost of getting worse is ~0 — the squashed arms
  wound the elbow 649 revolutions where the unsquashed arms never exceeded half a
  turn. `spin_limit` exists because of that.
- `spin_limit`, `spin_penalty` — terminate at |q2| ≥ limit via dm_control's
  discount-0 path, with an optional terminal cost.
- `reward_offset` — added to r1. Inert for fixed-length episodes; decisive once
  the agent chooses when the episode ends. r1 = −V admits no finite offset that
  lifts it to ≥ 0, so this is only usable together with `reward_squash`.

`reward_squash` and `reward_offset` are rejected for r0 and r2: r2 carries a
second unbounded, signed `η·V̇` term that the squash does not touch, so accepting
the combination would imply a bound that is not there.

## Where this leaves the study

The reward-shaping question is not yet answered, because two confounds sat
upstream of it for the whole `xk_uniform_v2` sweep: a 2 s discount horizon on an
11 s task, and α running to 1e6 on every unbounded-reward arm from a shared
α = 1.0 start (r0, the one bounded reward, settled at 0.066 instead). Neither
spin termination nor reward bounding moved the ceiling; the best cell in the
study remains the unmodified `r1` at γ = 0.995.

The diagnosis points away from further reward variants and toward making the
stabilization phase learnable: raise `k_p`/`k_d` so in-tube differences are not
0.018 % of range, warm-start from the controller via `--init_weights` so the
budget is not spent re-solving the energy problem, or split swing-up from
stabilization outright.

## Measurement traps encountered

Recorded because several cost real time and each one produced a wrong
intermediate conclusion.

- `time/fps` is inflated after `--resume`; and fitting a rate across the first
  `learning_starts` steps overstates steady state badly (44 vs 28.6 steps/s here)
  because those steps run without gradient updates.
- A 2000-step logging gap that straddles a multiple of `eval_freq` contains a
  whole evaluation. Eval cost = straddling gap − plain gap.
- `rollout/*` info values are **means over the logging window of signed
  quantities**. `elbow_norm` reads ≈0 for a *freely spinning* elbow; energy error
  |mean| understates mean|·| (0.114 vs 0.303); `lyapunov_rate` telescopes, since
  ∫V̇ dt = V_end − V_start, understating instantaneous |V̇| by up to 170×.
- `rollout/acrobot_xk_homoclinic_capture` is tube *occupancy* over a 1 s window,
  not continuous dwell. A policy sweeping through the tube scores 0.455 with a
  0.12 s longest hold. Rank arms on `eval/strict_capture_*`, not on this.
- `eval/strict_capture_mean_max_duration` is the **mean over 32 episodes** of each
  episode's longest dwell, not the maximum. One episode at 1.0 s plus 31 at 0.049
  gives 0.078 — so a nonzero success rate with a sub-second mean is consistent,
  and taking the max of each column independently mixes different eval rows.
