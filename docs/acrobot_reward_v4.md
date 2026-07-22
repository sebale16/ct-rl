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
  is not binding, and multi-swing pumping plus capture spans a few seconds).
