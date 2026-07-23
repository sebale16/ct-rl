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
  `acrobot_energy_norm`, `acrobot_speed`, `acrobot_slow_gate`, `acrobot_hold`
  (v2/v3 schemas unchanged).
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
from 1.0 to 0.25 (identical to v4 for Ẽ ≤ 1), so surplus-energy passes lose
their ramp income and the policy is pushed to regulate Ẽ → 1, where top
passes are slow and the hold is enterable. See `acrobot_reward_versions.md`
for the exact piecewise margin.

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

Because training now measures capture-from-anywhere, the true task (swing up
from hanging) is scored two ways. Post-training, `evaluations/eval_acrobot_v41_v5.py`
evaluates each checkpoint from both starts (`start` column). During training,
`run_ct_rl.py --eval_hanging` adds a second eval track from the hanging start
alongside the uniform-start primary: it logs `eval_hanging/*` and saves its
own gated `best_model_hanging/`, without disturbing the primary `best_model/`.
Because that hanging track resets every eval episode from down, its hold
occupancy only rises on genuine from-hanging capture — the start-distribution
income that trivially satisfies the uniform gate is absent — so
`best_model_hanging/` is the honest true-task selection (it will simply stay
empty if no from-hanging capture emerges, which is itself the answer). v5's
ceiling is a caution: uniform starts made it learnable but held-out height
occupancy tops out ≈0.12, so uniform-start v4.1 is expected to become
learnable but not automatically to sustain balance — v4.1's velocity-gated
hold is a stronger balance signal than v5's raw occupancy, which is the reason
to prefer it.

(On a resumed run the hanging track's best-so-far is not restored from the
checkpoint the way the primary eval's is, so `best_model_hanging/` may be
re-selected from a fresh baseline after a resubmission; the primary
`best_model/` keeps full resume fidelity.)

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
