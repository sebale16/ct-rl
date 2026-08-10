# Xin–Kaneda energy-based swing-up on the dm_control Acrobot

An analytical reference for the acrobot reward experiments of
[reward shaping for acrobot swing-up](reward_shaping_for_acrobot_swingup.md).
Those experiments compare CT-SAC under rewards built from Xin–Kaneda's Lyapunov
function, so they are only interpretable against a measurement of what the
controller that function was designed for actually does on this plant.

Two papers are involved and they share one control law:

- **2002** — Xin & Kaneda, *The Swing up Control for the Acrobot based on Energy
  Control Approach*, CDC 2002. Law (14), Theorem 4.
- **2007** — Xin & Kaneda, *Analysis of the energy-based swing-up control of the
  Acrobot*, Int. J. Robust Nonlinear Control 17:1503–1524. Law (18).

The implementation is `controllers/xin_kaneda.py`, the plant is
`environment/acrobot_xk.py` (`acrobot-swingup-xk`), the metrics are
`evaluations/acrobot_homoclinic_metrics.py`, and the sweep is
`evaluations/eval_acrobot_xk.py`.

## 1. The law

With a Lyapunov candidate V = ½k_E Ẽ² + ½k_D q̇₂² + ½k_P q₂², Ẽ = E − E_r, the
identity Ė = q̇₂τ₂ gives V̇ = q̇₂(k_E Ẽ τ₂ + k_D q̈₂ + k_P q₂). Solving for the
τ₂ that makes the bracket −k_V q̇₂ yields

  τ₂ = −[(k_V q̇₂ + k_P q₂)·det M + k_D(M₂₁(H₁+G₁) − M₁₁(H₂+G₂))] / (k_D M₁₁ + Ẽ·det M)

and V̇ = −k_V q̇₂² ≤ 0. The law is invariant under a common scaling of
(k_E, k_V, k_D, k_P), so only ratios matter; the module fixes k_E = 1, following
2007. The 2002 thresholds are still computed, since they are published fixtures.

### Coordinates

The plant is built directly in the papers' frame rather than in one needing a
transform: its links rest along the horizontal at q = 0, its hinges turn so that
q₁ = π/2 is upright and −π/2 hanging, and the shoulder sits at the height
reference. Consequently `qpos` *is* (q₁, q₂), `gear·ctrl` is τ₂ with no sign
flip, `qfrc_bias` is +(H + G), and the MuJoCo mechanical energy is E with no
additive offset. Verified over 200 random states in `tests/test_xin_kaneda.py`
against `mj_fullM` (max error 1e-12), `qfrc_bias` (1e-11) and the energy (1e-11).

The stock dm_control acrobot instead measures the shoulder from the upward
vertical, a reflection of this frame:

  q₁ᵖ = π/2 − q₁ˢ,  q₂ᵖ = −q₂ˢ,  q̇ᵖ = −q̇ˢ,  τ₂ˢ = −τ₂ᵖ

`obs_to_paper` applies it, and `XinKanedaController(frame="upward_vertical")`
drives that plant. The test suite checks both frames separately.

### Plant parameters

The geometry is Xin–Kaneda's own (their Section 7): m₁ = m₂ = 1, l₁ = 1, l₂ = 2,
l_c₁ = 0.5, l_c₂ = 1, I₁ = 0.083, I₂ = 0.33, g = 9.8, with the link inertias set
through explicit `<inertial>` elements so they land on the published values.
Recovered back out of the built model by `AcrobotParams.from_physics`:

| a₁ | a₂ | a₃ | b₁ | b₂ | E_r | E_s | ω_s |
|---|---|---|---|---|---|---|---|
| 1.333 | 1.330 | 1.000 | 14.7 | 9.8 | 24.5 | 49.0 | 4.5844 |

## 2. Which conditions, and why it matters

The two papers differ only in what they are willing to prove, and that choice
turns out to set the swing-up speed.

**2002 Theorem 4** achieves convergence for every initial condition by making
the closed-loop equilibria with q₂ ≠ 0 *not exist*. That is condition (57),
k_P > max(η*, ξ*)·k_E·θ₄θ₅g². Since ξ(q₂) → 2 as q₂ → 0 for every plant, ξ* = 2
always, η* never binds, and the condition is exactly k_P > 2b₁b₂.

**2007 Proposition 4** lets those equilibria exist and proves each one unstable
instead, by a Lyapunov-level argument: at any such equilibrium, their (56) gives
P(q₁*+δ, q₂*)·cos δ = P(q₁*, q₂*) with P < 0, so V(q₁*+δ, q₂*, 0, 0) <
V(q₁*, q₂*, 0, 0), and V is non-increasing. Proposition 5 and their (72) then
confirm hyperbolicity, so every stable manifold has measure zero — the abstract's
"for all initial conditions with the exception of a set of Lebesgue measure
zero". The price drops the requirement to (43), k_P > (2/π)·min(b₁², b₂²).

The k_D condition is also weaker, and exact. 2002's (13), k_D > 2k_E E_top/ρ*,
is sufficient only; it bounds E_r − P(q) by the crude 2E_r. Proposition 1 of 2007
gives the *necessary and sufficient* no-singularity condition (25),

  k_D > max_{q₂∈[0,2π]} (F(q₂) + E_r)·det M(q₂)/M₁₁(q₂),  F(q₂) = √(b₁²+b₂²+2b₁b₂cos q₂)

using the true extremum of the potential on the manifold where the shoulder
gravity torque vanishes, and proves necessity by constructing an initial
condition that reaches the singularity when it fails.

| threshold | this plant |
|---|---|
| k_D, 2007 (25) — exact | 35.741 |
| k_D, 2002 (13) — sufficient | 57.122 |
| k_P, 2007 (43) | **61.141** |
| k_P, 2002 (57) = 2007 (63) = 2b₁b₂ | **288.12** |

The first two match the values the paper quotes for its own plant, which is what
pins the gain-condition module down.

This module implements the 2007 conditions. k_P stays free above the (43) floor,
so raising it passes continuously through 2b₁b₂ and recovers the 2002 Theorem-4
gain set as the high-k_P end.

### 2b₁b₂ is a spectral boundary, not just a proof artefact

Hanging is an exact equilibrium of the closed loop — the gravity torques vanish
there and q₂ = q̇₂ = 0, so the law commands zero. 2007 Proposition 5 classifies
its Jacobian by k_P against 2b₁b₂:

- k_P < 2b₁b₂ → one left-half-plane eigenvalue, **three** right
- k_P = 2b₁b₂ → one left, one at the origin, two right (not hyperbolic)
- k_P > 2b₁b₂ → two left, two right

Measured at k_D = 35.8, k_V = 66.3:

| k_P | RHP roots | max Re | escape time constant |
|---|---|---|---|
| 62 | 3 | 1.930 | 0.52 s |
| 120 | 3 | 1.567 | 0.64 s |
| 180 | 3 | 1.132 | 0.88 s |
| **288.12 = 2b₁b₂** | 2 | 0.047 | 21.4 s |
| 400 | 2 | 0.081 | 12.3 s |
| 600 | 2 | 0.044 | 22.7 s |

Removing the spurious equilibria absorbs them into the hanging equilibrium,
which is what flattens its escape rate: correctness and speed are the same knob,
and two orders of magnitude separate the two settings. The implementation
reproduces the 2007 paper's own published characteristic equation at hanging,
s⁴ + 0.036k_V s³ − 3.375s² + 0.190k_V s − 43.076 (their eq. 71), which is the
strongest single correctness check in the test suite.

Note that 2002's Figures 2–3 are Theorem-**3** gains (k_P = 22, violating (57) by
3.3×); the paper states it gives "only the simulation results related to
Theorem 3 due to the page limitation". There is no published Theorem-4
trajectory, so the slow escape is a property those figures do not exhibit.

## 3. The plant, and its two deviations

`acrobot-swingup-xk` carries Xin–Kaneda's geometry with two kwargs governing the
assumptions their analysis makes about it.

### Joint damping voids the theory

Ė = q̇₂τ₂ is the engine of the whole derivation. With joint damping d it becomes
Ė = q̇₂τ₂ − d(q̇₁²+q̇₂²), leaving

  V̇ = −k_V q̇₂² − k_E Ẽ·d(q̇₁²+q̇₂²)

whose second term is **positive** whenever Ẽ < 0, that is throughout a swing-up.
The invariance argument fails harder: on the target set q₂ ≡ q̇₂ ≡ 0 the injected
power τ₂q̇₂ is identically zero while the shoulder dissipates up to
0.05 × 4.584² = 1.05 W, so the homoclinic orbit is not an invariant set of the
damped closed loop for any gains. `damping` therefore defaults to 0, with 0.05
reachable so the obstruction is measured rather than asserted. §5.2 shows what
that costs in practice: damping does not always prevent *entering* the tube, but
it destroys retention.

### The actuator is an experimental axis

The law is derived without an input bound, and on this geometry it asks for
about 20 N·m at the paper's own gains and up to 36 N·m elsewhere in the k_P
sweep. `torque_limit` sets the actuator gear and defaults to 64 N·m, high enough
never to bind, so ρ_sat stays 0 for the analytical controller. It stays a kwarg
because a learned policy sees the gear as an action scaling — see §6.

## 4. Metrics

`evaluations/acrobot_homoclinic_metrics.py` implements metrics 1–6 of the reward
doc as pure functions of a recorded trajectory, computed entirely from raw state
`[q₁, q₂, q̇₁, q̇₂]` plus the applied torque. Nothing is read from `info` and
nothing is shared with the v2 … v6.1 reward line, so the same code scores the
analytical controller and a learned policy and neither is credited for the
reward it was trained on.

All three normalizations are derived rather than tuned:

- E_s = E_top − E_down = 2(b₁+b₂) = 49.0 J
- q_s = π
- ω_s = √(4(b₁+b₂)/(a₁+a₂+2a₃)) = 4.584377 rad/s

ω_s is the peak shoulder speed on the homoclinic orbit itself, and equals
√(2E_s/M₁₁(0)), the speed at which the whole energy span is carried as kinetic
energy in the extended pose. It normalizes both the elbow rate in r₀ and the
shoulder rate in d_Γ, and the task's r₀ reward uses the same constant, so
d²(x) = −r₀(x) holds exactly.

**The orbit in repo coordinates.** Eq. (24) of 2002 / (32) of 2007 is
½(a₁+a₂+2a₃)q̇₁² = (b₁+b₂)(1 − sin q₁ᵖ). Since sin q₁ᵖ = cos q₁ʳ and
(1 − cos q₁)/2 = sin²(q₁/2), the repo-frame form is the closed curve

  q̇₁ = ±ω_s·sin(q₁/2),  q₂ = q̇₂ = 0

through the upright pose at rest, reaching |q̇₁| = ω_s at hanging. d_Γ minimizes
the normalized distance over a parameterization of it; both signed branches are
covered by sweeping the parameter over the 4π period of sin(θ/2).

Defaults: tolerance tube ε_E = ε_ω = 0.05 and ε_q = 0.025 (the elbow bound is
half the others because q_s = π is a much larger scale than the energy span or
the orbit's peak speed), dwell Δ = 1 s, and the LQR
switching threshold ζ = 0.04 from eq. (74) of 2007.

## 5. Results

All runs use the paper's own gains, k_D = 35.8, k_P = 61.2, k_V = 66.3 unless a
sweep says otherwise, on the conservative plant with the ample gear. Capture is
the 1 s dwell inside the tube at ε_E = ε_ω = 0.05, ε_q = 0.025; the LQR test is
eq. 74 at ζ = 0.04.

**Control period.** The state error against continuous feedback is first order
in the hold and negligible in the integration step — an RK4 step of 2 ms already
gives ~1e-7 after 20 s, so the two are set equal throughout. The period is
therefore the only accuracy knob, and it is quoted with every table below,
because it moves the numbers: on the shared baseline the median capture is
12.53 s at a 2 ms hold and 11.06 s at 0.5 ms. Energy-driven metrics converge by
0.5 ms; the eq. 74 residual does not, since the hold contributes ~0.02 to it by
t = 20 s at that period against a box of width 0.04.

### 5.1 Reproducing the paper's own trajectory

From its initial condition (q₁ = −1.4, q₂ = 0, q̇ = 0), at a 0.1 ms hold,
`results/acrobot_xk_paper_reproduction.csv`:

| quantity | value | paper |
|---|---|---|
| T_LQR at ζ = 0.04 | **8.0024 s** | "the switch was taken about t = 8 s" |
| peak ·τ₂· | 19.563 N·m | Fig. 4 spans roughly ±20 N·m |
| T_cap | 6.870 s | — |
| d_Γ RMS after capture | 0.0019 | — |
| min ·Ẽ· | 5.1e−5 of a 49 J span | Fig. 4 shows E − E_r → 0 |
| ρ_sat | 0 | law derived without an input bound |

This is the only run in the note at 0.1 ms, and it needs to be: at 0.5 ms the
hold contributes about 0.02 to the switching residual, so the 8 s crossing is
not resolvable at the period the rest of the tables use.

An independent RK4 integration of the analytic model gives T_LQR = 7.776 s and
is step-size stable from h = 1e-3 to 1e-5, so the MuJoCo figure is not an
artefact of either integrator.

### 5.2 The k_P frontier

`results/acrobot_xk_frontier.csv` — k_P swept from just above the eq. 43 floor
(61.141) across the 2b₁b₂ boundary (288.12), from the randomized near-hanging
reset, 5 seeds, 10 ms hold, T_max = 120 s below the boundary and 600 s above.
The hold is coarse here because the high-k_P arms need hundreds of seconds; the
bias it carries is common to every arm, so the comparison across k_P holds, but
these T_cap values are not comparable with the 0.5 ms baseline of the reward
note.

| k_P | damping | P(cap) | T_cap p50 | peak ·τ₂· | e_RMS | d_Γ RMS | ρ_ℋ |
|---|---|---|---|---|---|---|---|
| 62 | 0 | 1.00 | 10.6 s | 19.31 | 0.028 | 0.035 | 0.997 |
| 80 | 0 | 1.00 | 25.9 s | 29.67 | 0.029 | 0.036 | 1.000 |
| 120 | 0 | 1.00 | 31.7 s | 36.07 | 0.033 | 0.038 | 1.000 |
| 180 | 0 | 1.00 | 30.7 s | 26.17 | 0.040 | 0.043 | 1.000 |
| **288.12 = 2b₁b₂** | 0 | 1.00 | 249.5 s | 13.33 | 0.035 | 0.038 | 0.998 |
| 400 | 0 | 1.00 | 283.3 s | 8.96 | 0.021 | 0.023 | 0.999 |
| 600 | 0 | 1.00 | 388.7 s | 3.51 | 0.021 | 0.026 | 1.000 |
| 120 | 0.05 | 0.20 | 9.8 s | 33.69 | 0.070 | 0.059 | **0.323** |
| all others | 0.05 | 0.00 | — | 3.70–34.39 | — | — | — |

### Reading

**Torque demand and capture time trade off across the boundary.** Peak demand
falls monotonically from k_P = 120 upward, 36.1 → 3.5 N·m, while T_cap rises
from 32 s to 389 s. The fast end is not the k_P floor: demand *rises* from 19.3
to 36.1 N·m going from k_P = 62 to 120, because very weak elbow regulation lets
the swing-up wander before settling. The cheapest capture in both senses sits
near the floor.

**Crossing 2b₁b₂ costs about an order of magnitude in time.** T_cap p50 goes
from 30.7 s at k_P = 180 to 249.5 s at the boundary — the Proposition-5 change
of spectral type at the hanging equilibrium, showing up as the escape phase
rather than the pumping phase.

**Damping's cost is retention, and how much of it shows depends on the
tolerance.** Only k_P = 120 captures at all under damping 0.05, on 1/5 seeds,
and its ρ_ℋ is 0.323 against 1.000 undamped with e_RMS twice as large. That is
the non-invariance of the orbit made quantitative — the trajectory can be pumped
into the tube but cannot stay in it, because the target set injects zero power
while the shoulder keeps dissipating. Note that this reading is tolerance
dependent: at the looser ε_q = 0.05 used earlier, k_P = 120 captured on 5/5 and
k_P = 180 on 1/5, so a bound loose enough to admit grazing passes made damping
look survivable. The theoretical obstruction of §3 is what is invariant; whether
it shows up as a capture failure or only as a retention failure depends on how
tightly the tube is drawn.

**The actuator never binds.** ρ_sat is 0 in every arm; the largest demand
anywhere in the sweep is 36.1 N·m against the 64 N·m gear.

### 5.3 Reaching the LQR switching set

T_LQR is a much slower event than T_cap, and the two should not be conflated.
Convergence to Γ is asymptotic, and the closest approach to upright on each
traversal is governed by the residual energy — but by *two* different relations,
depending on the sign Ẽ approaches from, and they differ by a factor of 362.

**Turning point** (Ẽ < 0). The shoulder stops short of upright at
δ ≈ √(2·Ẽ·/E_r) and pays the full angle, so the residual is δ and the level at
which the box becomes reachable is

  V\* = ζ⁴E_r²/8 = 1.92 × 10⁻⁴.

**Pass-through** (Ẽ > 0). The shoulder crosses upright with
q̇₁ = √(2Ẽ/M₁₁) and the residual costs only 0.1·q̇₁, giving

  V\* = 1250·M₁₁²ζ⁴ = 6.96 × 10⁻².

The runs that come near the box are pass-throughs, so the second is the
operative level and the first is a pessimistic bound. Measured closest
approaches on the release distribution at a 0.5 ms hold, over 20 s:

| seed | min residual | q₁ − π/2 | q̇₁ | Ẽ | V | branch |
|---|---|---|---|---|---|---|
| 20002 | 0.046 | +0.0001 | −0.413 | +0.398 | 0.080 | pass-through |
| 20004 | 0.061 | +0.0001 | −0.540 | +0.683 | 0.235 | pass-through |
| 20000 | 0.290 | +0.283 | +0.000 | −0.988 | 0.497 | turning point |
| 20003 | 0.364 | +0.359 | +0.000 | −1.562 | 1.253 | turning point |

The pass-through relation predicts Ẽ = 0.373 at the boundary against 0.398
measured for seed 20002, whose residual is correspondingly just outside at
0.046. For those runs 90% of the residual budget is the 0.1·q̇₁ term: they reach
the pose and arrive too fast.

On the turning-point branch the decay is slow enough to track directly. From the
randomized near-hanging reset, continuous feedback:

| t | running-min residual | ·Ẽ·/E_s | √(2·Ẽ·/E_r) |
|---|---|---|---|
| 30 s | 0.344 | 2.80e−2 | 0.335 |
| 60 s | 0.238 | 1.33e−2 | 0.231 |
| 120 s | 0.137 | 4.53e−3 | 0.135 |
| 240 s | 0.060 | 8.46e−4 | 0.058 |
| 480 s | 0.0148 | 5.4e−5 | 0.0148 |
| 960 s | 0.0015 | 1e−6 | 0.0015 |

The residual first drops below 0.04 at **303 s** from that reset, against 8 s
from the paper's own initial condition; the difference is escape and pumping
from a 0.005 rad displacement rather than 0.17, not the final approach. The
prediction matches the measurement to three digits once the elbow has settled,
which is what makes the relation usable as a diagnostic.

The switch is also control-period sensitive, and more so than anything else
measured here. The hold error is first order and lands almost entirely on the
phase: at t = 20 s it contributes 0.066 to the residual at a 10 ms hold, 0.021
at 0.5 ms and 0.005 at 0.1 ms, against a box of width 0.04. So a coarse period
does not merely fail to sample the crossing, it displaces the trajectory by more
than the box. `lqr_residual_min` is reported beside `lqr_time` for this reason,
and any run whose headline is T_LQR should use a 0.1 ms hold.

### 5.4 Videos

`videos/acrobot_xk_swingup.mp4` — the randomized near-hanging reset reaching Γ,
capture at 13.24 s, real time. `videos/acrobot_xk_lqr_switch.mp4` — the paper's
initial condition reaching ζ = 0.04 at 8.02 s, rendered at dt = 1e-4 with a
freeze on the crossing. Both put the mechanism beside the shoulder phase
portrait with Γ drawn as a fixed reference. Rendered by
`benchmarks/render_acrobot_xk_swingup.py`.

## 6. Using this plant for the CT-SAC arms

Three things carry over badly from the controller to a learned policy.

**The torque limit is an action rescaling, not a constraint toggle.** The plant
applies τ₂ = gear·ctrl with ctrl ∈ [−1, 1], so `torque_limit` moves the gear.
The controller is indifferent — it computes τ₂ in physical units and divides by
the gear on the way out — but for CT-SAC the gear rescales exploration noise,
the meaning of `log_std_init` and the target entropy, and any action cost. At
gear 64 an untrained policy applies ±64 N·m against a gravity torque scale of
b₂ = 9.8 N·m. Size the gear to the regime instead: from §5.2 the peak demand is
19–20 N·m near the k_P floor, rises to 36 N·m around k_P = 120, and falls to
3.5 N·m by k_P = 600. Whatever is chosen, the learned arm and the controller it
is compared against must share it, since an agent trained at one gear cannot be
evaluated at another.

**"Unbounded" is not reachable for a learned policy.** The actor is
tanh-squashed (`models/actor_q_critic.py`), so its output is in [−1, 1] by
construction and τ₂ is bounded by the gear regardless. Raising the gear only
moves where the bound sits.

**ρ_sat does not mean the same thing for both.** For the controller it measures
demand exceeding capacity, computed from the pre-clip command, and is 0
throughout §5 because the gear is ample. A learned policy has
no pre-clip command, so `Trajectory.commanded_torque` falls back to the applied
torque and ρ_sat records only how often the policy chose an extreme action. The
two should not be pooled into one column without a note.

One piece of plumbing is missing for training runs.
`load_ct_hyperparams_from_table` (`common/utils.py`) special-cases exactly one
`env_*` column, `time_sampling_kwargs`; every other one goes through
`_parse_scalar`. An `env_task_kwargs` cell therefore arrives at
`DMCContinuousEnv` as a string and raises
`ValueError: dictionary update sequence element #0 has length 1; 2 is required`.
Until `task_kwargs` is added alongside `time_sampling_kwargs` there, a
hyperparams row cannot set `damping` or `torque_limit`, and a trainer gets the
factory defaults (damping 0, 2 N·m). Also note that r₂ = −V − η·V̇ is linear in
τ₂ through q̈₂, so its magnitude scales with the gear while r₀ and r₁ do not; η
needs rescaling if the gear changes, or the three arms stop being comparable.

## 7. Reproducing

```bash
python -m unittest tests.test_xin_kaneda

# 5.1 the paper's own trajectory, at the fine step its switch needs
MUJOCO_GL=disable python -m evaluations.eval_acrobot_xk \
    --sweep single --kp 61.2 --kd 35.8 --kv 66.3 --start paper --dt 1e-4 \
    --t-max 12 --seeds 20000:20001 \
    --output results/acrobot_xk_paper_reproduction.csv

# 5.2 the k_P frontier
MUJOCO_GL=disable python -m evaluations.eval_acrobot_xk \
    --sweep frontier --start hanging --seeds 20000:20005 \
    --t-max-fast 120 --t-max-slow 600 --output results/acrobot_xk_frontier.csv

# the shared baseline for the reward experiments
MUJOCO_GL=disable python -m evaluations.eval_acrobot_xk \
    --sweep single --kp 61.2 --kd 35.8 --kv 66.3 --start release \
    --dt 0.002 --t-max 20 --seeds 20000:20032 \
    --output results/acrobot_xk_baseline_release20.csv

# videos
MUJOCO_GL=egl python -m benchmarks.render_acrobot_xk_swingup \
    --duration 30 --fps 50 --output videos/acrobot_xk_swingup.mp4
MUJOCO_GL=egl python -m benchmarks.render_acrobot_xk_swingup \
    --start paper --duration 10 --dt 1e-4 --hold 2.5 \
    --output videos/acrobot_xk_lqr_switch.mp4
```

`--start paper` is deterministic, so seeds repeat one trajectory there; the
capture-rate and capture-time distributions need `hanging`, `release` or
`uniform`. `--dt` finer than the model's 0.01 timestep only takes effect because
`build_env` defaults `physics_dt` to `min(dt, 0.01)`.
