# The Hamiltonian Recovery Report

Base: https://hackmd.io/@-YScJRgTQoiFn3RF3xJ3Fg/rkgsjiWQzg

*What `evaluations/hamiltonian_recovery.py` measures, why each metric grants the gauge freedom it does, and how to read a report. Written against the `model-based-generator-ct-sac` branch of the `ct-rl` repo.*

[TOC]

## Three kinds of accuracy

The audit separates three axes, and a model should not be described as accurate on the strength of one alone:

1. **Predictive accuracy** (`predictive_report`) — does the model predict accelerations and open-loop rollouts? One-step drift vs the realized rate, plus displacement-normalized open-loop error over recorded windows; the recorded next states *are* the simulator's rollout under identical actions, so this is a paired learned-vs-MuJoCo comparison.
2. **Physical recovery** (`recovery_report`) — does it recover the true mass, potential, Coriolis, damping, actuator, and contact terms? The gauge-fixed, term-by-term identifiability comparison that is the core of this document.
3. **Control-relevant accuracy** (`generator_report`, `quadrature_report`) — does model error corrupt the targets CT-SAC actually consumes? The generator projection error and the quadrature value-increment error, measured under both dataset and policy-sampled candidate actions.

The strongest evidence of an accurate learned model is agreement across all three: controlled prediction, physically meaningful recovery, and low error on the generator and quadrature targets used during policy learning.

The structured dynamics model (`models/port_hamiltonian.py`, `mode="structured"`) learns the *physical objects themselves* — a mass matrix $M(q)$, a potential $V(q)$, a damping $D$, an actuator port $G_a$, and optionally a contact port — and generates the Coriolis force from $\partial M/\partial q$ by autodiff. Because these live in the same generalized coordinates MuJoCo integrates, the term-by-term comparison of axis 2 is well-posed.

That comparison answers a question one-step prediction error cannot. A model can predict held-out transitions accurately while attributing forces to the wrong terms — the project's own history supplies the canonical example: with no contact channel in the model class, least squares absorbed ground reaction into a spurious $z$-dependence of $M(q)$, which *helped* prediction on the training distribution while destroying the Coriolis term generated from the same tensor. The RL critic consumes the model at policy-*sampled* actions and rolled-out states, off the training distribution, so attribution errors that are invisible to a prediction metric become control errors. PHAST draws the same distinction between rollout accuracy and identifiability. Axis 3 closes the remaining gap: it measures the model where CT-SAC reads it, so attribution errors and coverage gaps show up as target error directly.

## The pipeline

```
datasets: reference (OU, fixed seed) | policy (--checkpoint) | best_policy (--best_checkpoint)
        │  per dataset: last n_eval=800 states of n=4000 transitions
        ▼
ground_truth(env, obs)          # MuJoCo introspection per state (+ contact truth)
learned_terms(model, obs)       # the model's own objects at the same states
sanity_check_truth(...)         # extraction self-checks (SPD, energy identity, invariance)
        │
        ▼
recovery_report(truth, learned)     # gauge-fix, then the physical metrics below
predictive_report(model, ...)       # one-step + open-loop accuracy
generator_report / quadrature_report  # target error (needs the checkpoint V-head)
energy_balance_report(model, ...)   # forced power balance along the model drift
```

### Evaluation distributions

Physical recovery is evaluated **on visited states** — the model is unconstrained off the data manifold, so pointwise off-manifold term comparisons would be meaningless — but the audit is no longer limited to each checkpoint's *own* visited states. Every report is computed per evaluation distribution:

- **reference** — a fixed broad-exploration OU set (fixed seed, independent of the run), so cross-checkpoint comparisons share a distribution and genuine model change separates from policy-distribution change;
- **policy** — the current policy's gait distribution (`--checkpoint`), the audit that matters for the RL run;
- **best_policy** — a frozen best-policy distribution (`--best_checkpoint`), the forgetting probe: a model that has tracked the declining policy away from the peak gait shows up here;
- **candidate actions** — the generator and quadrature errors are additionally measured at the visited states crossed with policy-*sampled* actions, which is where the critic target actually reads the model.

The JSON output nests everything under `datasets.<name>.<axis>`, and the console prints the primary dataset in full plus a cross-distribution headline table.

## Ground truth: the MuJoCo extraction identities

`ground_truth` snapshots the live physics, then for each evaluation state sets $(q, v)$ and reads:

| Quantity | Extraction | Identity used |
|---|---|---|
| $M(q)$ | `mj_fullM(model, M, data.qM)` after `forward()` | dense mass matrix from the sparse factorization |
| $e_{\text{pot}}, e_{\text{kin}}$ | `data.energy` with `mjENBL_ENERGY` enabled | simulator's own energy bookkeeping |
| $g_{\text{pot}} = \nabla(V_{\text{grav}} + V_{\text{spring}})$ | `qfrc_bias(q,0) − qfrc_passive(q,0)` | at $v=0$ the bias force reduces to gravity and the passive force to joint springs; dampers contribute nothing at zero velocity |
| Coriolis $C(q,v)\,v$ | `qfrc_bias(q,v) − qfrc_bias(q,0)` | the bias force is $C(q,v)v + g(q)$; subtracting the $v=0$ evaluation isolates the velocity-dependent part |
| $G_a$ | `qfrc_actuator` columns under unit control, one actuator at a time, at $v=0$ | exact for cheetah's linear torque motors |
| damping diagonal | `model.dof_damping` | model constant |

In addition, per state: `qfrc_constraint` — the generalized constraint force, the contact-force truth (caveat: joint-limit forces are included); the active-contact flag (`ncon > 0`); and the per-geom contact activity over the geoms seen in contact (for the permutation-invariant matching below).

Root $x$ is set to 0 for every state (each extracted quantity is translation-invariant), and the physics state is restored afterward. The extraction supports the cheetah task observation (`obs = [qpos[1:] (8); qvel (9)]`, planar floating base) and any raw-state hinge/slide domain (`--raw_state_obs`, `obs = [qpos; qvel]`).

`sanity_check_truth` guards the extraction itself: $M$ must be SPD, MuJoCo's kinetic energy must equal $\tfrac12 v^\top M v$ to $10^{-6}$ relative error, and the gravity+spring torque must have a zero root-$x$ component. If the extraction conventions ever drift, these fail before any comparison is reported.

## Learned terms

`learned_terms` evaluates the model's internals at the same states, with the same conventions:

- $\hat M(q)$ via the model's own `_mass` (Cholesky, SPD by construction), and $\hat V(q)$.
- $\hat e_{\text{kin}} = \tfrac12 \dot q^\top \hat M \dot q$.
- $\nabla \hat V$, scattered onto the 9-wide config axis — the cyclic root-$x$ slot is structurally zero, matching the truth.
- Coriolis $\hat c = \dot M \dot q - \tfrac12 \dot q^\top (\partial M/\partial q) \dot q = C(q,\dot q)\dot q$, computed from the *same* autodiff Jacobian the drift uses at training time. The convention matches the truth extraction (left-hand-side force in $M\ddot q + C\dot q + g = \tau$), which is itself pinned by a finite-difference test in `tests/test_hamiltonian_recovery.py`.
- Base damping diagonal $\mathrm{softplus}(\log d)$ and the damping force $d\circ\dot q$, and $G_a$ from the port's weight matrix (scattered if the layout is sparse).
- `dM_mag`: the per-position-coordinate mean $|\partial M/\partial q|$ — the input to the $z$-probe below.
- **When the contact port is active** (`contact_force > 0`): the combined conservative gradient $\nabla\hat V - \sum_i k_i\,\varphi(g_i)\,J_{n,i}$, the full generalized contact force $\sum_i (J_{n,i}\lambda_i + J_{t,i}f_{t,i})$ and its power, the learned gaps at the eval states, the in-contact fraction, and the spring-to-$\nabla V$ magnitude ratio.

## Gauge freedoms, and the philosophy of granting them

Two exact invariances mean the raw learned numbers can never match the truth, and should never be asked to:

1. **Global scale.** The flow $\ddot q = M^{-1}(G_a a - \nabla V - D\dot q - C\dot q)$ is invariant under $(M, V, D, G_a) \to (cM, cV, cD, cG_a)$ — the Coriolis force scales with $M$ automatically. A trajectory-fit model therefore learns everything up to one arbitrary positive constant.
2. **Potential offset.** $V$ and $V + \text{const}$ generate the same forces.

`recovery_report` fixes the scale once, in closed form, on the mass matrices:

$$
c^* = \arg\min_c \sum_{\text{states}} \lVert c\hat M - M \rVert_F^2 = \frac{\langle \hat M, M\rangle}{\langle \hat M, \hat M\rangle},
$$

and applies this **single shared scalar to every learned term**. The sharing is deliberately part of the test: a physically consistent model needs one $c$ to fix all four objects simultaneously, so any per-term scale disagreement survives into the stricter metrics as error rather than being silently absorbed.

Each metric then grants exactly the freedom the physics grants, and no more. After $c^*$ is fixed, only the potential (and total energy) retain an offset; nothing retains a slope. Affine-fit $R^2$ values are therefore *shape diagnostics* (`*_shape_R2`), useful but never presented as strict recovery; the strict numbers are the scale-locked errors:

| Freedom granted | Metrics | Rationale |
|---|---|---|
| none beyond $c^*$ | all `*_nrmse` force errors, mass errors, kinetic $R^2$/NRMSE, `damping_rel_err`, $G_a$ Frobenius error | these quantities have no residual gauge |
| offset only | `potential_locked_nrmse`, `total_H_locked_nrmse` | $V$ has an offset gauge and nothing else |
| affine (slope + offset) | `potential_shape_R2`, `total_H_shape_R2` | shape diagnostics; the slope is reported (`potential_slope_ratio`) so flatness is visible rather than hidden |
| scale-free by construction | all Pearson correlations, per-actuator cosines | Pearson and cosine are invariant to positive scaling, so these measure pure shape/direction |

Every force term reports **both** a correlation and a normalized magnitude error, plus a tail statistic — correlation and cosine ignore force magnitude, so neither alone establishes recovery:

$$
\operatorname{NRMSE}(f) = \frac{\sqrt{\sum_i\lVert\hat f_i-f_i\rVert^2}}{\sqrt{\sum_i\lVert f_i\rVert^2}+\epsilon},
\qquad
\texttt{err\_p95} = \frac{p_{95}\big(\lVert\hat f_i - f_i\rVert\big)}{\mathrm{RMS}\big(\lVert f_i\rVert\big)} .
$$

The Coriolis, combined-conservative, and contact forces additionally carry `*_nrmse_strata`: the NRMSE split by speed tercile and by contact phase (flight/contact), so a model that fits slow gaits and misses fast ones is visible directly.

## The metrics

**Mass matrix.**

| Key | Definition | What a bad value means |
|---|---|---|
| `gauge_scale_c` | $c^*$ as above | nothing by itself — it is a gauge. Reported for context and for the potential-slope comparison |
| `mass_rel_frob_err` | $\text{mean}_i\, \lVert c^*\hat M_i - M_i\rVert_F / \lVert M_i \rVert_F$ | magnitudes of $M$ wrong (even with good shape) |
| `mass_diag_rel_err`, `mass_offdiag_rel_err` | the same error restricted to the diagonal / strict upper triangle | separates inertia scales from coupling structure |
| `mass_entry_corr`, `mass_uppertri_corr` | Pearson over all entries / over unique upper-triangular entries | mass *structure* wrong; the upper-tri variant removes the double-counting of symmetric entries |
| `mass_eig_rel_err`, `mass_cond_ratio` | summed eigenvalue error of $c^*\hat M$ vs $M$; mean condition-number ratio | the spectrum the solve $M^{-1}(\cdots)$ actually feels |
| `mass_inverse_response_nrmse` | NRMSE of $(c^*\hat M)^{-1}\tau$ vs $M^{-1}\tau$, $\tau = G\,a$ over the dataset's actions | the acceleration response to representative forces — the most control-adjacent mass read |

**Architectural / leakage checks** (not recovery metrics):

| Key | Definition | What it means |
|---|---|---|
| `mass_dMdz_ratio` | $\overline{\lvert\partial M/\partial z\rvert} \,/\, \overline{\lvert\partial M/\partial q_{j\ne z}\rvert}$; **true value 0** | the contact-leak probe: inertia is invariant under vertical translation, so any $z$-dependence of $M$ is ground reaction being absorbed as a contact proxy |
| `mass_dMdz_structural` | whether $z$ is excluded from the mass input (contact port active) | when true, a zero ratio is **guaranteed by construction** and is no longer evidence that the remaining mass matrix is correct — read it as an invariant check, not a result |

**Energies.**

| Key | Definition | What a bad value means |
|---|---|---|
| `potential_shape_R2`, `potential_slope_a`, `potential_slope_ratio` | affine fit $e_{\text{pot}} \approx a\hat V + b$; `slope_ratio` $= a/c^*$ | low $R^2$: wrong shape. ratio $\gg 1$: potential too flat (historically 7–9×, from ground reaction cancelling gravity in the data) |
| `potential_locked_nrmse` | NRMSE of $c^*\hat V$ vs $e_{\text{pot}}$ after removing the mean difference (offset gauge only) | the strict potential error: scale is locked to the shared gauge |
| `kinetic_R2`, `kinetic_nrmse` | scale locked to $c^*$, **no offset** ($T(\dot q{=}0)=0$) | see the reading guide below; the strictest energy read |
| `total_H_shape_R2`, `total_H_locked_nrmse` | affine fit / offset-only NRMSE of $c^*(\hat V + \hat e_{\text{kin}})$ onto $e_{\text{pot}} + e_{\text{kin}}$ | total energy landscape wrong |

**Forces** (each with `_corr`, `_nrmse`, `_err_p95`, and where noted `_nrmse_strata`):

| Key | Definition | What a bad value means |
|---|---|---|
| `gradV_force_*` | $c^*\nabla\hat V$ vs $\nabla(V_{\text{grav}}+V_{\text{spring}})$ | conservative force wrong — or, with the contact port active, gravity has migrated into the port (check the combined metric before concluding anything) |
| `gradV_combined_*` (strata) | $c^*(\nabla\hat V - \sum_i k_i\varphi(g_i)J_{n,i})$ vs the same truth (port models only) | with the port, $V$ and the gap-spring potentials are only identified **in sum**; this is the number to read instead of the raw one |
| `coriolis_force_*` (strata) | $c^*\hat C(q,\dot q)\dot q$ vs truth | the term that governs fast motion ($\propto \dot q^2$) is wrong; *negative* correlation means $\partial M/\partial q$ is being used as something else entirely (historically: a touchdown brake) |
| `damping_force_*` | $c^*(d\circ\dot q)$ vs $d_{\text{true}}\circ\dot q$ | the damping force as felt on the data distribution |
| `actuator_force_*` | $c^*\hat G a$ vs $G a$ over the dataset's actions | actuator response wrong where actions actually live |
| `contact_force_*` (strata) | $c^*\sum_i(J_{n,i}\lambda_i + J_{t,i}f_{t,i})$ vs `qfrc_constraint` | the port's generalized force does not match the true contact force (includes joint-limit forces — a caveat on the truth side) |
| `contact_power_*` | $c^*\sum_i(\lambda_i\dot g_i + f_{t,i}v_{t,i})$ vs $\dot q^\top$`qfrc_constraint` | contact power misattributed — dissipation/propulsion in the wrong place |

**Contact port state and geometry.**

| Key | Definition | What a bad value means |
|---|---|---|
| `contact_in_frac` | fraction of (state, contact) pairs with $g_i < 0$ | 0.00 on gait data = the port never engaged (dead port); ~1 at a fast gait with flight phases = port acting as an always-on field rather than a contact model |
| `contact_precision`, `contact_recall` | any-learned-contact vs the true `ncon > 0` flag, per state | the port fires in the wrong states (precision) or misses true contact (recall) |
| `contact_edge_offset_steps`, `contact_edge_count`, `contact_edge_matched_frac` | mean \|offset\| between true and predicted touchdown/liftoff edges (nearest match within 10 steps, episode boundaries skipped) | contact timing wrong — the port switches late or early relative to the physics |
| `contact_match_corr` | greedy permutation-invariant matching of learned per-contact activity to true per-geom contact activity (Pearson per matched pair) | learned contact points do not correspond to physical feet; the port fits contact *in aggregate* only |
| `contact_impulse_rel_err` | relative error of $\sum_t F\,\Delta t$ (per-DOF impulse over the eval window) | net momentum transfer wrong even if instantaneous forces look plausible |
| `contact_spring_ratio` | mean $\lVert\sum_i k_i\varphi(g_i)J_{n,i}\rVert / \lVert\nabla\hat V\rVert$ | how much of the conservative field lives in the port; large is legitimate, it just changes which grad-V number is meaningful |
| `contact_k`, `contact_c`, `contact_mu` | per-contact stiffness and compression damping at gauge scale $c^*$; friction $\mu$ raw (a force ratio, gauge-free) | absolute values are soft ($k$ trades against the learned gap scale); the meaningful read is the *trend across checkpoints of one run* — stiffness creep makes the drift stiffer over training and adds quadrature/BPTT noise at exactly the high-speed regime |
| `contact_in_frac_per`, `contact_gap_mean/min` | per-contact activity split and gap statistics on the eval states | which of the $K$ points actually act (silent points are fine — spare capacity), and how deep the working penetrations sit |

**Damping and actuators.**

| Key | Definition | What a bad value means |
|---|---|---|
| `damping_rel_err`, `damping_abs_err_per_dof`, `damping_rel_err_per_dof`, `damping_locked_R2`, `damping_learned/true` | $c^*\,\mathrm{softplus}(\log d)$ vs `dof_damping`, directly at the shared gauge — **no affine refit**: after $c^*$ the damping diagonal retains no slope or offset freedom | passive joint damping mis-learned. Caveat: 9 points, so any aggregate is coarse; read the per-DOF lists |
| `G_rel_frob_err`, `G_actuator_cosine` | $\lVert c^*\hat G - G\rVert_F/\lVert G\rVert_F$; per-actuator column cosines | actuator *authority* wrong (Frobenius) vs actuator *direction* wrong (cosines). Degraded cosines on a gait are a specific alarm: actions phase-lock with stance, so missing contact propulsion regresses onto $G_a$ |

### Reading guide: why kinetic $R^2$ can be low while total-$H$ $R^2$ is high

This combination appears in real audits and is usually a gauge nuisance rather than a physics failure. Three effects stack:

1. Kinetic $R^2$ is scale-locked to $c^*$ with no offset, while potential and total-$H$ get affine refits. Kinetic energy must vanish at $\dot q = 0$, so granting it an offset would be unphysical — the strictness is intentional.
2. On a steady gait, $e_{\text{kin}}$ has a large mean and small variance. $R^2$ divides squared error by that small variance, so a pure scale mismatch (learned kinetic tracking truth at 0.95 correlation but at 0.75× the $c^*$-implied scale) craters the metric while the shape is nearly perfect.
3. $c^*$ is fit on mass *entries* weighted equally; the kinetic form weights entries by which DOFs the gait excites. If magnitude errors concentrate on velocity-heavy entries, the effective kinetic scale diverges from $c^*$.

Meanwhile total-$H$ is variance-dominated by the potential (root-height bobbing through gravity) and enjoys the affine refit, so it mostly reflects potential shape. When in doubt, compare `mass_entry_corr` (shape) against `mass_rel_frob_err` and `kinetic_nrmse` (magnitude), and read `mass_eig_rel_err` for the spectrum.

## The predictive axis

`predictive_report` measures what the recovery metrics deliberately do not: transition prediction.

| Key | Definition |
|---|---|
| `accel_corr`, `accel_nrmse` | one-step drift's acceleration block vs the realized rate $(x'-x)/\Delta t$ |
| `drift_nrmse` | the same over the full observation drift |
| `rollout_rel_err_H4`, `rollout_rel_err_H8` | open-loop model roll (recorded actions and durations, training-time integration resolution) vs the recorded states, displacement-normalized, windows that cross no episode boundary |

The recorded next states are the simulator's own rollout under those actions, so the open-loop numbers are paired learned-vs-MuJoCo rollouts with identical action sequences.

## The control-relevant axis

These two reports measure the model **where CT-SAC uses it**, and both need the checkpoint's V-head (`--checkpoint` on a V-head mode).

**Generator projection error** (`generator_report`). CT-SAC's first-order target consumes the projection $b(x,a)^\top\nabla V_\psi(x)$, not every drift component equally. The report measures

$$
e_{\mathrm{gen}}(x,a) = \big(\hat b(x,a) - b_{\mathrm{MJ}}(x,a)\big)^\top \nabla V_\psi(x),
$$

with $b_{\mathrm{MJ}}$ the oracle drift, reporting RMSE, bias, correlation of the two projections, p95/p99 tails, and the RMSE per speed tercile and contact phase — under the dataset's actions (`data_actions`) and under policy-sampled candidate actions at the same states (`policy_actions`), with `err_vs_action_novelty_corr` the correlation between $|e_{\mathrm{gen}}|$ and the candidate action's distance from the replayed one. This is the direct measure of how learned-dynamics error enters the critic target, and where coverage gaps (candidate actions, faster-than-visited states) become visible.

**Quadrature label error** (`quadrature_report`). The sub-step quadrature target reads a value increment over the model's roll. From the same state and constant candidate action, the model is integrated over $\Delta t_{\text{default}}$ at the training-time resolution and MuJoCo over the same interval, and the two increments the label would read are compared:

$$
\Delta V_{\mathrm{model}} = V_\psi(\hat x) - V_\psi(x_0)
\quad\text{vs}\quad
\Delta V_{\mathrm{true}} = V_\psi(x^{\mathrm{MJ}}) - V_\psi(x_0),
$$

with RMSE, bias, correlation, the **sign-disagreement rate** (the fraction of labels whose value-change direction is wrong), p95, strata, and `endpoint_state_nrmse` (the same paired rollout read on the predictive axis). This is the closest available test of the learned model's contribution to the actual CT-SAC label.

## The forced energy balance

`energy_balance_report` checks the model's power bookkeeping along its own drift. A bare "energy must decrease" test is invalid for controlled cheetah — actuator work can raise the mechanical energy — so the residual accounts for every port:

$$
\frac{\mathrm dE}{\mathrm dt} = \dot q^\top G_a a \;-\; \dot q^\top D\dot q \;+\; \sum_i\big(\lambda_i\dot g_i + f_{t,i}v_{t,i}\big),
\qquad E = V + T,
$$

where the contact sum contains the conservative spring exchange (storage, sign-indefinite) plus the compression and friction dissipation ($\le 0$). `residual_nrmse` checks the identity itself (it must be numerics-small for any parameters — a large value means the audit and the model disagree about the flow). `passivity_violation_frac` counts states whose energy rise is **not** explained by actuator input plus spring release; an energy increase covered by positive actuator work is not a violation. The mean per-port powers are reported alongside.

## Case study: three audits, two confounds diagnosed

All three are rollout-fit (`fit_horizon` 4) cheetah models audited on their own policy's data. `contact_roll` is the old model class (contact expressed only through a velocity-jump-gated damping term, since removed from the codebase, 2M updates); `cforce_roll v1` added the explicit contact port but shipped with a gap init 25 smoothing-widths into the softplus tail (dead port, 500k); `v2` is the fixed init (600k). (These audits predate the metric renames: "Grad-V corr" is `gradV_force_corr`, the potential/total-$H$/damping $R^2$ rows are the old affine fits — today's `*_shape_R2` and, for damping, the since-replaced affine $R^2$.)

| Metric | contact_roll (2M) | cforce_roll v1 (500k) | cforce_roll v2 (600k) |
|---|---|---|---|
| Mass $\partial M/\partial z$ ratio | **6.001** | 0.000 | 0.000 |
| Mass entry corr | 0.967 | 0.927 | 0.936 |
| Coriolis corr | **0.106** | 0.652 | **0.660** |
| Grad-V corr | 0.850 | 0.823 | 0.830 |
| Kinetic $R^2$ | 0.841 | 0.536 | 0.523 |
| Total $H$ $R^2$ | 0.934 | 0.899 | 0.926 |
| Damping $R^2$ | 0.959 | 0.952 | 0.959 |
| $G_a$ Frobenius err | 0.576 | **0.849** | **0.382** |
| $G_a$ cosines | [0.96, 0.97, 0.94, 1.0, 0.83, 0.89] | [0.50, 0.85, 0.90, **0.37**, 0.53, 0.73] | [0.90, 0.95, 0.82, 0.99, 1.00, 0.98] |
| Contact in-frac / spring ratio | — | 0.00 / 0.00 | 0.87 / 0.08 |

**Confound 1 — contact leaking into $M$** (first column). The model class had no contact term, so least squares routed ground reaction through the only state-switched channel available: a $z$-dependent $M$ modulates every force through $M^{-1}$, and its gradient generates $\dot q^2$-forces that fire at touchdown. The probe reads 6× (true value 0), and the cost lands exactly where the theory says: the Coriolis force generated from the same contaminated tensor is near-uncorrelated with truth at 0.106 — the model could represent slow gaits and had no idea about fast ones. Closing the leak structurally (root $z$ removed from the mass input when the port is active) took Coriolis to 0.65 in a quarter of the training, **even while the port itself was dead** (middle column).

**Confound 2 — propulsion leaking into $G_a$** (middle column). With the port dead and $M$ closed, the remaining contact force needed a home. Its conservative part can hide in $V(z)$; its propulsive part cannot — $\nabla V$ has a structurally zero root-$x$ slot — so it regressed onto the actuator port, where actions phase-lock with stance. The signature is the cosine collapse (0.37–0.53 on three actuators). Fixing the gap init woke the port (in-frac 0.87, gentle springs at 8% of $\nabla V$), and $G_a$ snapped back to the best actuator recovery of any audit (Frobenius 0.382, cosines 0.90–1.00). This one matters directly for RL: the critic's quadrature target rolls the model with *candidate* actions, and a $G_a$ inflated with contact propulsion overstates what actions can do everywhere off the gait cycle.

The offline validation had already shown the port's one identifiability price: on always-in-contact data, gravity can migrate from $V$ into the gap springs (raw grad-V corr fell to −0.06 there while dynamics were unaffected) — which is precisely why `gradV_combined_corr` exists.

## Running it

The dynamics model of every CT-SAC checkpoint is saved automatically as a sidecar: `<ckpt>.pth` ↔ `<ckpt>.dynamics.pth`.

```bash
# audit an RL run's model: reference + on-policy distributions, all three axes
# (the checkpoint's V-head enables the generator/quadrature metrics)
python evaluations/hamiltonian_recovery.py \
    --dynamics_path <ckpt>.dynamics.pth \
    --checkpoint    <ckpt>.pth \
    --mode mbq_structured_quad_cforce_roll \
    --contact_force 4 \
    --out audit.json

# add the frozen best-policy distribution (forgetting probe)
#   ... --best_checkpoint <peak_ckpt>.pth

# offline: fit a fresh model on OU data first, then audit it
python evaluations/hamiltonian_recovery.py --fit_steps 8000 --fit_horizon 4 --contact_force 4
```

The JSON is nested: `{"primary": ..., "datasets": {<name>: {"physical_recovery": ..., "predictive": ..., "generator": ..., "quadrature": ..., "energy_balance": ..., "headline": ...}}}` — `headline` is the one-row cross-distribution summary the console table prints.

The architecture flag (`--contact_force`) must match how the run was configured; a mismatch fails loudly at `load_state_dict`. (Sidecars from the removed gated-damping era contain weights the current model class no longer has and cannot be loaded; their audits are recorded above.) `--mode` names the hyperparameter row used to size the policy networks when `--checkpoint` collects the data. Runs on CPU in a few minutes.

## Caveats and sharp edges

- **Per-checkpoint policy distributions still differ.** The fixed `reference` set makes cross-checkpoint comparison sound, but the `policy` rows of different checkpoints see different gaits — trends and signs are trustworthy there, second-decimal differences are noise. (Also: collection runs at uniform $dt = 0.01$ for clean extraction, while training uses irregular sampling.)
- **Physical recovery is evaluated on visited states.** A term can be perfectly recovered on the gait manifold and arbitrarily wrong off it; the generator/quadrature metrics under policy-sampled candidate actions are the probes for exactly that off-manifold read.
- **The contact-force truth is `qfrc_constraint`**, which folds in joint-limit forces; on gaits with saturated joints the contact-force NRMSE carries that contamination.
- **Kinetic $R^2$ is strict by design** (see the reading guide); a low value alongside high mass-entry correlation usually indicates scale attribution rather than wrong physics — `kinetic_nrmse` and `mass_eig_rel_err` disambiguate.
- **Truth extraction supports cheetah and raw-state hinge/slide domains**; the $G$ extraction relies on linear torque actuators, and free-joint (quaternion) domains are rejected.
- **Damping audit covers the diagonal base only**, and its aggregate rests on 9 points — read the per-DOF error lists.
- **Control-relevant metrics need the checkpoint's V-head**; on modes without one they are skipped, and their $\nabla V_\psi$ is the *current* head — an imperfect value function reweights, but does not invalidate, the projection error.
