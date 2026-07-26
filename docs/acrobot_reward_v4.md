# Acrobot swing-up reward v4: energy regulation with a velocity-gated hold

`acrobot-swingup-v4` (`environment/acrobot_v2.py::BalanceV4`) is derived from
the failure evidence of the three earlier reward attempts. Mechanism, MuJoCo
model, observations, and the repeatable near-down reset are identical to
v2/v3; only the reward changes.

## Evidence from v1–v3

**v1 — stock `acrobot-swingup`** (four-mode comparison,
`results/acrobot_four_mode.csv`): best return ever seen 43/1000, final means
6–14. The stock narrow Gaussian target reward carries no signal near the
hanging pose, so nothing learns.

**v2 — tip-distance progress + precise tail** (`0.8·(1 − d/4) + 0.2·precise`,
final matrix in `results/swingup_final_current.csv`): all 7 modes × 12 seeds
pinned in 664–683 at 1M steps. Zero variance across algorithms and seeds means
the reward itself created the attractor: a bent hover just below the target
collects ≈0.7/step forever, and any capture attempt first loses that income
while crossing the low-reward gap to the unstable goal. Videos:
`videos/videos_acrobot_v2/`.

**v3 — anti-fold `extension · mean-uprightness + precise tail`** (pilot
`results/acrobot_v3_pilot.csv` on the remote branch, commits `2f4c996` /
`e422448`): plateaus at ≈230–260 by 250–300k (per-step ≈0.12), oracle ≈
model-free, and the γ∈{0.999, 0.9995} horizon check never reached tip_z > 3 —
max tip_z over 320 eval episodes was 1.87, below the shoulder mount at z=2.
The task actuates only the elbow, and energy pumping is rhythmic elbow
bending; the `extension` factor zeroes the dense term exactly during that
motion, so pumping earns the same ≈0.1 as aimless swinging and the policy
never discovers swing-up. v3 also pays 1.0 for a fast spin through the top
pose, another parking surface that pumping-free policies simply never reached.

## Requirements

1. Dense signal from the hanging pose (v1).
2. No sustained income comparable to the goal rate anywhere off the goal;
   in particular none reachable without the capture skill (v2).
3. The dense term must pay for the transient pumping motion itself, elbow
   bends and all (v3).

## Definition

reward = 0.2·ramp + 0.8·hold, clipped to [0, 1].

Energy: E(q, q̇) = ½ q̇ᵀM(q)q̇ − Σᵢ mᵢ g⃗·x⃗ᵢ, computed from the MuJoCo model
(`mj_fullM` + body CoM heights). Ẽ = (E − E_hang)/(E_up − E_hang) with both
references measured at rest poses during `initialize_episode`, so Ẽ=0 at
hanging rest and Ẽ=1 at upright rest.

- ramp = tol(Ẽ, bounds=(1,1), margin=1, value_at_margin=0.1) · (1 + ū)/2,
  with ū the mean link uprightness from `physics.vertical()`. Any action that
  moves E toward E_up raises the first factor regardless of pose, so pumping
  is rewarded directly; energy overshoot (spinning) is discounted
  symmetrically. The (1+ū)/2 tilt halves parking on the Ẽ=1 manifold away
  from the top (e.g. holding Ẽ=1 as kinetic energy at the bottom).
- hold = precise · slow, where precise is the stock target tolerance
  (d ≤ 0.2, margin 1) and slow = tol(‖q̇‖, bounds=(0, 0.5), margin=2,
  value_at_margin=0.1). Sustained near-1 income exists only while balancing
  at the exact target.

At E ≈ E_up the passive dynamics pass through the upright pose arbitrarily
slowly (the homoclinic orbit), so a policy that has learned the ramp visits
the hold region at low speed by construction — capture is discoverable
without ever fighting the dense term. Slow top passes already collect ≈0.99
transiently, and slowing them further raises the collected fraction, giving a
smooth gradient from swing-through into balance.

## Audit (per-step rates, `BalanceV4` on the real model)

| state | v4 | ramp | hold | Ẽ | v3 | v2 |
|---|---|---|---|---|---|---|
| hanging rest | 0.010 | 0.050 | 0.000 | 0.00 | 0.000 | 0.000 |
| upright rest (goal) | 1.000 | 1.000 | 1.000 | 1.00 | 1.000 | 1.000 |
| fold-up static | 0.130 | 0.649 | 0.001 | 0.75 | 0.000 | 0.400 |
| bent hover, wobbling | 0.205 | 0.966 | 0.014 | 1.01 | 0.758 | 0.690 |
| slow pass near goal | 0.990 | 0.998 | 0.989 | 1.01 | 0.997 | 0.958 |
| fast spin at top | 0.138 | 0.666 | 0.006 | 1.42 | 1.000 | 1.000 |
| fast swing at bottom | 0.063 | 0.313 | 0.000 | 0.55 | 0.000 | 0.000 |

Worst sustainable off-goal income is ≈0.2 against 1.0 at the goal (v2 offered
0.69, and v3 offered 0.76–1.00 on surfaces it could not reach). A scripted
collocated pump (kick, then elbow torque against the shoulder swing, backing
off as Ẽ→1) reaches Ẽ=0.92 and tip_z=3.54 in 20 s with quarter-mean v4 reward
climbing 0.03→0.16 as energy rises; the same trajectory scored by v3 stays
flat near 0.1. `tests/test_env_acrobot_v2.py::TestAcrobotSwingupV4Reward`
locks all of this in: the zero-velocity reward slice has its only local
maximum at upright-extended, the parking states above stay below their
bounds, and the pump trace must correlate with Ẽ under v4 but not under v3.

## Wiring

- Env id `acrobot-swingup-v4` in `DMCContinuousEnv`; v4-only info keys
  `acrobot_energy_norm`, `acrobot_speed`, `acrobot_slow_gate`,
  `acrobot_hold`, and `acrobot_strict_capture` (v2/v3 schemas unchanged).
- `evaluations/evaluate_swingup_final.py` accepts the env id; the tip_z > 3
  criterion and folded-extension diagnostics carry over unchanged.
- Pilot rows in `benchmarks/hyperparams/ct_sac.csv`: `final_mf` and
  `final_oracle_rollout` for `acrobot-swingup-v4`, copied from the v3 pilot
  with γ=0.995 (≈2 s horizon at dt=0.01; the v3 γ sweep showed horizon alone
  is not binding, and multi-swing pumping plus capture spans a few seconds),
  plus the model-free horizon arms `mf_hz_g0998` / `mf_hz_g0999`.

## v4.1: capture pressure + uniform starts

The v4 pilots reach the top but pass through it with surplus energy (Ẽ > 1,
fast), so the hold term never triggers — swing-through at rate ≈0.08–0.19,
not capture at ≈1. v4.1 tightens the energy tolerance margin above Ẽ = 1
from 1.0 to 0.25, so surplus-energy passes lose their ramp income. It also
tightens the hold speed tolerance from bounds [0, 0.5], margin 2.0 to
bounds [0, 0.1], margin 0.5. The pumping ramp remains identical to v4 for
Ẽ ≤ 1, while a moving near-target state now loses hold income much sooner.
See `acrobot_reward_versions.md` for both tolerances.

**Hanging-start v4.1 failed, and the failure is instructive.** Held-out eval
(`results/acrobot_v41_v5_eval.csv`): CT-SAC never even reached the height
(max tip 2.02, height and hold occupancy 0 across all seeds), strictly worse
than v4 which at least found tip 4.0; the fixed-dt SB3 baselines reached the
height (frac tip>3 up to 0.70) but with hold occupancy ≈0.001 — reach, not
hold. The cause: from hanging, the only discovery path to the top runs
through the overshoot the margin now penalizes (a first successful pump
arrives fast, with Ẽ > 1). v4.1 removed its own ladder — the capture-pressured
reward has its maximum on the slow Ẽ = 1 manifold, but that region is
unreachable from hanging without the penalized overshoot. The best_model
gate (hold occupancy ≥ 0.05) then stayed empty, so no peak checkpoint was
even captured.

**Uniform random starts fix it** (`uniform_start=True`, the v4.1 default),
the same lever that made v5 learnable. Starting from uniform random joint
angles puts near-top, near-Ẽ = 1 states directly in the start distribution:
18 % of resets begin above the height, and averaged over the whole start
stream the hold reward is ≈0.07 — already above the 0.05 gate before any
learning. The hold is trained directly where v4.1 rewards it most, and its
value propagates outward to lower-energy starts, so discovery no longer
requires the penalized overshoot. Energy calibration is pose-independent and
composes with the reset unchanged; `uniform_start=False` restores the
near-hanging reset for from-hanging probes and A/B comparison.

**Twenty-second runway.** PPO and CT-SAC now use 20 physical-second v4.1
episodes for both training and evaluation, leaving time to stabilize after a
late top arrival and to satisfy the one-second strict capture criterion. The
total 1 M-step budget, $\gamma=0.995$, and uniform starts are unchanged. PPO
uses 2,000 fixed steps per episode; irregular CT-SAC uses a 5,000-step cap so
its heavy small-$\Delta t$ tail has ample headroom to reach the full physical
horizon under the configured sampler.

Because training now measures capture-from-anywhere, the true task (swing up
from hanging) is scored two ways. Post-training, `evaluations/eval_acrobot_v41_v5.py`
evaluates each checkpoint from both starts (`start` column). Its
`strict_capture_success_rate` and `strict_capture_mean_max_duration` columns
reuse the same one-second physical-time tracker as checkpoint selection;
`mean_hold_occ` remains a separate smooth reward diagnostic, not a formal
capture count. During training,
both `run_ct_rl.py --eval_hanging` and
`run_discrete_rl.py --eval_hanging` add a second eval track from the hanging
start alongside the uniform-start primary. Each logs `eval_hanging/*` and
saves its own `best_model_hanging/`, without disturbing `best_model/`.
For v4.1 PPO and CT-SAC, both checkpoint paths use the same strict event:
tip distance below 0.2 and joint-speed norm below 0.2 continuously for at
least one physical second. Selection maximizes the episode success rate,
then mean maximum residence duration; return is diagnostic only. Physical
residence uses `dt_used`, so CT-SAC's irregular steps and PPO's fixed steps
are scored in seconds under the same definition. The hanging track is the
honest true-task selection because it can satisfy that event only after a
genuine from-down capture. v5's
ceiling is a caution: uniform starts made it learnable but held-out height
occupancy tops out ≈0.12, so uniform-start v4.1 is expected to become
learnable but not automatically to sustain balance — v4.1's velocity-gated
hold is a stronger balance signal than v5's raw occupancy, which is the reason
to prefer it.

Wall-clock checkpoints persist both the primary and hanging evaluation
histories and their best strict ranks, so neither best-model path can regress
after a resumed training chunk.

## v5: unshaped height occupancy as the control arm

`acrobot-swingup-v5` (`BalanceV5`) pays reward 1 while the tip strictly
exceeds the Gym height (one link length above the pivot ⟺ tip_z > 3) and 0
otherwise, over a fixed-length episode with no termination. The return is
the physical time spent above the height. No dense term below the height, so
there is no parking surface, and maximal income is staying up — balancing
near the top is the implicit optimum without any velocity gate or target
shaping. It isolates whether v4's shaping is necessary: v4 runs log
`gym_height_success` continuously, so if v4-mf learns while v5 flatlines the
ramp was the necessary ingredient, and if v5 also learns the simpler task
wins.

Episodes start from uniform random joint angles at near-zero velocity
(`uniform_start=True`, default) rather than the near-hanging pose: 18.5 % of
uniform resets begin above the height, so the sparse income exists in the
replay data from the first episodes and value propagates outward to lower
starts, instead of exploration having to climb ~10 s uphill unrewarded
(nothing unshaped has ever exceeded tip_z 1.87 from hanging here). Resets
above the line are unstable inverted poses, so collecting their income
directly trains balance. `uniform_start=False` restores the near-hanging
reset for from-hanging probes.

v5 rows run 30 s episodes (`env_max_steps` 6000) with γ = 0.999 and 0.9995
(`mf_hz_g0999`, `mf_hz_g09995`).

Independent of v5, the wrapper distinguishes the two dm_control LAST
sources: genuine task termination (discount 0) maps to `terminated`, while
dm_control's internal step limit (discount 1) and the wrapper's own episode
duration map to `truncated`, so bootstrapping is only ever cut on true
terminal states.

## v6: quadratic cost (AR-EAPO reward)

### Evidence from v4.2

The launch6 batch (6 seeds, 1 M steps, 20 held-out episodes per checkpoint)
gives v4.2 a working uniform track and a dead hanging track:

| arm | uniform ret | hanging ret | hanging frac tip>3 | strict capture |
|---|---|---|---|---|
| ct_sac final_mf (final) | 264 ± 41 | 41.1 ± 0.8 | 0.000 | 0.00 |
| ct_sac final_oracle_rollout (final) | 583 ± 90 | 303 ± 221 | 0.350 | 0.00 |
| ppo final_mf (final) | 122 ± 41 | 82 ± 67 | 0.458 | 0.00 |
| sac final_mf (final) | 104 ± 24 | 20.6 ± 0.5 | 0.000 | 0.00 |

41.1 over 20 s is 0.010/step, and 0.010/step is exactly what v4.1's reward pays
at the hanging rest pose: `hold` is 0 there and `energy_close` sits at its
`value_at_margin` floor, so `(1 − hold_weight)·ramp` = 0.2·(0.1·0.5) = 0.01.
The model-free policy collects the floor and never leaves it. The convergence
check rules out undertraining — every CT-SAC seed plateaus with a negative
final-quarter slope (−4 to −13 per 100 k) — so widening the reset band pushed
the start energy down toward hanging faster than the capture value propagated
outward, and the curriculum lost the top rather than extending its basin.

### Definition

`acrobot-swingup-v6` (`BalanceV6`) replaces the ramp-and-hold blend with the
quadratic state-and-command cost of Choe et al. (2024), eq. 16:

    r(s, a) = −α[(s − g)ᵀ Q (s − g) + aᵀ R a]

over s = [θ₁, θ₂, θ̇₁, θ̇₂] with g = 0 the upright rest pose,
Q = diag(50, 50, 4, 2), R = 1, α = 0.001. Every configuration is separated
from the goal by a strictly monotone position cost, which is the property v4.1
lacks at hanging. Two deviations from the published form, both forced by this
mechanism:

* angle errors are wrapped into (−π, π] before squaring, since the raw
  difference is discontinuous at the branch cut and both resets sample it;
* R multiplies the normalized command a ∈ [−1, 1] rather than a torque in N·m,
  so `action_weight` carries the paper's R·τ_max².

The reward is a cost: ≤ 0, zero only at the upright rest pose with zero
command, never clipped. Nothing terminates, so the task is the continuing MDP
the paper's average-reward criterion assumes. `reward_offset` adds a constant
and leaves the optimum unchanged under both the discounted-soft and the
average-reward objective, since no state is absorbing and the time limit
truncates with bootstrapping; it exists only to put returns on the scale of the
[0, 1]-reward arms.

### Audit (per-step rates, `BalanceV6` on the real model)

| state | r | angle | velocity | action |
|---|---|---|---|---|
| upright rest | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| upright rest, full command | −0.0010 | 0.0000 | 0.0000 | 0.0010 |
| hanging rest | −0.4935 | 0.4935 | 0.0000 | 0.0000 |
| folded above the pivot (θ₂ = π) | −0.4935 | 0.4935 | 0.0000 | 0.0000 |
| upright, θ̇ = (5, 10) | −0.3000 | 0.0000 | 0.3000 | 0.0000 |
| hard pump, θ̇ = (16, 58) | −7.8754 | 0.1234 | 7.7520 | 0.0000 |

The last row is the term to watch, and reading it as "Q is mis-scaled" is wrong.
The swing-up must acquire E_up − E_hang = 39.24 J, but the cost of carrying that
energy is set by which generalized mode holds it, and M(hanging) is strongly
anisotropic (eigenvalues 0.072 and 2.964):

| mode carrying the full 39.24 J | θ̇ [rad/s] | velocity cost |
|---|---|---|
| whole arm together, (−1, −0.66) | (−4.5, −2.9) | 0.097 |
| elbow flapping against shoulder, (−0.33, 1) | (−10.3, 31.2) | 2.376 |

Same energy, 24× apart, against a hanging position cost of 0.4935. A coordinated
swing-up is ~5× cheaper per step than hanging still, so the optimum is well
placed and the weights actively prefer efficient pumping over flailing.

The exposure is the path, not the optimum. The acrobot is elbow-actuated, and
M⁻¹[0, 1] aligns with the expensive mode at 1.000 and with the cheap one at
0.275: raw elbow torque excites exactly the motion the reward charges most for,
and reaching the cheap mode requires resonant pumping that transfers energy
across. Cost of leaving the hanging pose therefore depends on skill already
held — 0.591/step (1.2× do-nothing) coordinated, 8.416/step (17.1×) at the
measured bang-bang thrash |θ̇| = (16.6, 58.4). Hanging at rest is a strict local
optimum, since position cost moves second-order under a small perturbation while
velocity cost turns on immediately. This is the tension Choe et al. name in
their intro (§I, point 1); their counterweights are MaxEnt at τ = 2 against
rewards of order 0.5 — entropy outweighing reward ≈ 4:1 — and the average-reward
criterion.

Two consequences for the arms here. The reverse curriculum is the direct
instrument: fraction 0 starts inside the basin where no barrier exists, and the
band widens outward from a policy that already pumps efficiently, which is the
regime v4.2 could not reach. Second, `algo_alpha=auto` targets entropy −1 and
pins no reward-to-entropy ratio, so it does not reproduce the paper's 4:1. The
`fixed_a2p0` / `fixed_a0p5` / `fixed_a0p1` ladder brackets it: the entropy target
of −1 nat makes α·|H| ≈ α, so those three sit at roughly 4:1, 1:1 and 0.2:1
against the 0.4935/step hanging-to-top reward gap, with 2.0 the paper's τ
transplanted directly. Each row is its env's `final_mf` with `algo_alpha` the
only field changed.

### Exploration diagnostics

The velocity cost multiplies two independent quantities — how much energy is in
motion, and how wastefully it is carried — so on its own it cannot separate a
collapsed policy from a well-coordinated one. An efficient pump holding the full
swing-up budget costs 0.0966, under a fifth of the hanging position cost, which
is indistinguishable from rest in a training curve. Two per-step terms factor it
apart, logged from the behavior policy's own rollouts under `rollout/` (the
Monitor's `info_keywords`), since exploration is a property of the stochastic
policy that deterministic evaluation cannot show:

* `energy_norm` = (E − E_hang)/span, the fraction of the 39.24 J swing-up budget
  currently held, counting kinetic and potential together. Pumping raises total
  energy, while within one swing the kinetic share trades against potential at
  swing frequency, so this is the slow variable the policy controls. 0 = hanging
  at rest, 1 = enough to sit at the top, >1 = overshoot. Shared with v4.x, so
  the runs stay comparable. `kinetic_norm` reports the kinetic share alone.
* `velocity_cost_per_joule` = (q̇ᵀWq̇)/KE, a generalized Rayleigh quotient of the
  pencil (W, M(q)) and therefore scale-free in q̇: it reads coordination with the
  amplitude divided out, flat across energy levels within a mode.
  `coordination_loss` normalizes it onto [0, 1] between the cheapest and dearest
  directions at the current pose, which the raw ratio needs because the bounds
  move with the elbow angle — [0.0025, 0.0606] extended, [0.0060, 0.0127] folded
  at θ₂ = 2.

| what the policy is doing | vel_cost | energy_norm | cost/J | coordination_loss |
|---|---|---|---|---|
| hanging still | 0.0000 | 0.00 | NaN | NaN |
| coordinated pump @ 25 % | 0.0242 | 0.25 | 0.0025 | 0.00 |
| coordinated pump @ 100 % | 0.0966 | 1.00 | 0.0025 | 0.00 |
| flailing @ 25 % | 0.5941 | 0.25 | 0.0606 | 1.00 |
| flailing @ 100 % | 2.3763 | 1.00 | 0.0606 | 1.00 |
| upright at rest | 0.0000 | 1.00 | NaN | NaN |

Both ratios are NaN at rest rather than 0. A scale-free direction reading has no
value when there is no motion, and any finite sentinel would enter the logged
running mean: zero sits below the ratio's own lower bound, so resting steps
would pull the mean toward "perfectly coordinated" — the misreading these terms
exist to prevent, and worst on a collapsed policy, which rests the most. A
policy resting 70 % of the time and flailing the rest would log
0.7·0 + 0.3·1 = 0.30 under a zero sentinel, against a true 1.00 whenever it
moves. The Monitor drops non-finite values, so the logged number means
coordination *while moving*; `energy_norm` and `kinetic_norm` stay finite and
carry whether it moves at all. `coordination_loss` is therefore never
interpretable alone — 0 and NaN are different states, and the energy terms
separate them.

The pair is what makes the α ladder readable, since all three of its failure
modes present as a return stuck near the hanging floor: α too low leaves
`energy_norm` ≈ 0 (collapsed to rest, raise it), α too high leaves
`coordination_loss` near 1 with energy going in (random flailing, lower it), and
a workable α shows `energy_norm` climbing while `coordination_loss` falls
(pumping, capture not yet timed — give it steps).

The discount horizon governs how heavily the transient is weighted, and matters
only while the policy is still in the expensive mode. At γ = 0.995 per
`dt_default` = 0.01 s, β = −ln γ / dt = 0.501/s and 1/β ≈ 2.0 s, so a 1.4 s
swing-up carries 1 − e^(−β·1.4) = 50 % of the discounted weight against 1.4/20 =
7 % under the average-reward criterion over the reset cycle. `mf_hz_g09995`
(γ = 0.9995, 1/β = 20.0 s = one full cycle) is the closest reachable
approximation without adding a gain estimate to the critic.

### Wiring

Two env ids over the one reward isolate the reset exactly as v4.1/v4.2 do:

* `acrobot-swingup-v6` keeps the v4.2 reverse curriculum, driven by
  `set_curriculum_fraction` from global training progress;
* `acrobot-swingup-v6-uniform` carries no band schedule at all — every episode
  draws both joints uniformly on [−π, π], so `has_curriculum` is False and the
  trainer attaches no curriculum callback. It rejects `curriculum=True` rather
  than ignoring it, while still accepting the `curriculum=False` the runner and
  the eval harness pass generically to pin eval starts.

Both run 20 s episodes (`env_max_steps` 5000) in three modes: `final_mf` and
`final_oracle_rollout` at γ = 0.995, matching the v4.2 rows field for field so
the comparison is reward-only, plus `mf_hz_g09995` at γ = 0.9995.

Both are in `STRICT_CAPTURE_ENV_IDS`, which is load-bearing rather than
cosmetic here: the reward is a cost, so ranking checkpoints by reward would
select whichever policy spends least time swinging. Selection stays on capture
rate then mean maximum residence, identical to v4.1/v4.2, and `best_model_gate`
is superseded as before. The hold-occupancy eval column reads 0 on v6, which
has no `hold` term; the height and strict-capture columns carry the comparison.
