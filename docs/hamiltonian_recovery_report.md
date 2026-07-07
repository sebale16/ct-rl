# The Hamiltonian Recovery Report

*What `evaluations/hamiltonian_recovery.py::recovery_report` measures, why each metric grants the gauge freedom it does, and how to read a report. Written against commit `0d38d6d` of the `ct-rl` repo (branch `model-based-generator-ct-sac`).*

[TOC]

## Why this exists: prediction and identifiability are different axes

The structured dynamics model (`models/port_hamiltonian.py`, `mode="structured"`) learns the *physical objects themselves* — a mass matrix $M(q)$, a potential $V(q)$, a damping $D$, an actuator port $G_a$, and optionally a contact port — and generates the Coriolis force from $\partial M/\partial q$ by autodiff. Because these live in the same generalized coordinates MuJoCo integrates, a term-by-term comparison against the simulator is well-posed.

That comparison answers a question one-step prediction error cannot. A model can predict held-out transitions accurately while attributing forces to the wrong terms — the project's own history supplies the canonical example: with no contact channel in the model class, least squares absorbed ground reaction into a spurious $z$-dependence of $M(q)$, which *helped* prediction on the training distribution while destroying the Coriolis term generated from the same tensor. The RL critic consumes the model at policy-*sampled* actions and rolled-out states, off the training distribution, so attribution errors that are invisible to a prediction metric become control errors. PHAST draws the same distinction between rollout accuracy and identifiability; this report measures the identifiability axis.

## The pipeline

```
collect(env, n=4000)            # OU exploration, or a saved policy (--checkpoint)
        │
        ▼  last n_eval=800 states
ground_truth(env, obs)          # MuJoCo introspection per state
learned_terms(model, obs)       # the model's own objects at the same states
sanity_check_truth(...)         # extraction self-checks (SPD, energy identity, invariance)
        │
        ▼
recovery_report(truth, learned) # gauge-fix, then the metric dict below
```

Everything is evaluated **on visited states only** — the model is unconstrained off the data manifold, so off-manifold comparisons would be meaningless. With `--checkpoint` the visited states are the trained policy's own gait distribution (the audit that matters for an RL run); without it they come from Ornstein–Uhlenbeck exploration.

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

Root $x$ is set to 0 for every state (each extracted quantity is translation-invariant), and the physics state is restored afterward. The extraction is cheetah-specific (asserted) — planar floating base, `obs = [qpos[1:] (8); qvel (9)]`.

`sanity_check_truth` guards the extraction itself: $M$ must be SPD, MuJoCo's kinetic energy must equal $\tfrac12 v^\top M v$ to $10^{-6}$ relative error, and the gravity+spring torque must have a zero root-$x$ component. If the extraction conventions ever drift, these fail before any comparison is reported.

## Learned terms

`learned_terms` evaluates the model's internals at the same states, with the same conventions:

- $\hat M(q)$ via the model's own `_mass` (Cholesky, SPD by construction), and $\hat V(q)$.
- $\hat e_{\text{kin}} = \tfrac12 \dot q^\top \hat M \dot q$.
- $\nabla \hat V$, scattered onto the 9-wide config axis — the cyclic root-$x$ slot is structurally zero, matching the truth.
- Coriolis $\hat c = \dot M \dot q - \tfrac12 \dot q^\top (\partial M/\partial q) \dot q = C(q,\dot q)\dot q$, computed from the *same* autodiff Jacobian the drift uses at training time. The convention matches the truth extraction (left-hand-side force in $M\ddot q + C\dot q + g = \tau$), which is itself pinned by a finite-difference test in `tests/test_hamiltonian_recovery.py`.
- Base damping diagonal $\mathrm{softplus}(\log d)$, and $G_a$ from the port's weight matrix (scattered if the layout is sparse).
- `dM_mag`: the per-position-coordinate mean $|\partial M/\partial q|$ — the input to the $z$-probe below.
- **When the contact port is active** (`contact_force > 0`): the combined conservative gradient $\nabla\hat V - \sum_i k_i\,\varphi(g_i)\,J_{n,i}$, the learned gaps at the eval states, the in-contact fraction, and the spring-to-$\nabla V$ magnitude ratio.

## Gauge freedoms, and the philosophy of granting them

Two exact invariances mean the raw learned numbers can never match the truth, and should never be asked to:

1. **Global scale.** The flow $\ddot q = M^{-1}(G_a a - \nabla V - D\dot q - C\dot q)$ is invariant under $(M, V, D, G_a) \to (cM, cV, cD, cG_a)$ — the Coriolis force scales with $M$ automatically. A trajectory-fit model therefore learns everything up to one arbitrary positive constant.
2. **Potential offset.** $V$ and $V + \text{const}$ generate the same forces.

`recovery_report` fixes the scale once, in closed form, on the mass matrices:

$$
c^* = \arg\min_c \sum_{\text{states}} \lVert c\hat M - M \rVert_F^2 = \frac{\langle \hat M, M\rangle}{\langle \hat M, \hat M\rangle},
$$

and applies this **single shared scalar to every learned term**. The sharing is deliberately part of the test: a physically consistent model needs one $c$ to fix all four objects simultaneously, so any per-term scale disagreement survives into the stricter metrics as error rather than being silently absorbed.

Each metric then grants exactly the freedom the physics grants, and no more:

| Freedom granted | Metrics | Rationale |
|---|---|---|
| none beyond $c^*$ | mass Frobenius error, kinetic $R^2$, $G_a$ Frobenius error | these quantities have no residual gauge |
| affine (slope + offset) | potential $R^2$, total-$H$ $R^2$, damping $R^2$ | $V$ has an offset gauge; slope is reported so flatness is visible rather than hidden |
| scale-free by construction | all Pearson correlations, per-actuator cosines | Pearson and cosine are invariant to positive scaling, so these measure pure shape/direction |

## The metrics

| Key | Definition | What a bad value means |
|---|---|---|
| `gauge_scale_c` | $c^*$ as above | nothing by itself — it is a gauge. Reported for context and for the potential-slope comparison |
| `mass_rel_frob_err` | $\text{mean}_i\, \lVert c^*\hat M_i - M_i\rVert_F / \lVert M_i \rVert_F$ | magnitudes of $M$ wrong (even with good shape) |
| `mass_entry_corr` | Pearson over all entries and states | mass *structure* wrong |
| `mass_dMdz_ratio` | $\overline{\lvert\partial M/\partial z\rvert} \,/\, \overline{\lvert\partial M/\partial q_{j\ne z}\rvert}$; **true value 0** | the contact-leak probe: inertia is invariant under vertical translation, so any $z$-dependence of $M$ is ground reaction being absorbed as a contact proxy |
| `potential_affine_R2`, `potential_slope_a` | affine fit $e_{\text{pot}} \approx a\hat V + b$ | low $R^2$: wrong shape. $a/c^* \gg 1$: potential too flat (historically 7–9×, from ground reaction cancelling gravity in the data) |
| `kinetic_R2` | $1 - \sum(c^*\hat e_{\text{kin}} - e_{\text{kin}})^2 / \sum(e_{\text{kin}} - \bar e_{\text{kin}})^2$ — scale locked to $c^*$, **no offset** | see the reading guide below; this is the strictest metric in the report |
| `total_H_affine_R2` | affine fit of $c^*(\hat V + \hat e_{\text{kin}})$ onto $e_{\text{pot}} + e_{\text{kin}}$ | total energy landscape wrong |
| `gradV_force_corr` | Pearson of $\nabla\hat V$ vs $\nabla(V_{\text{grav}}+V_{\text{spring}})$ | conservative force points the wrong way — or, with the contact port active, gravity has migrated into the port (check the combined metric before concluding anything) |
| `gradV_combined_corr` | Pearson of $\nabla\hat V - \sum_i k_i\varphi(g_i)J_{n,i}$ vs the same truth (port models only) | with the port, $V$ and the gap-spring potentials are only identified **in sum**; this is the number to read instead of the raw one |
| `contact_in_frac` | fraction of (state, contact) pairs with $g_i < 0$ | 0.00 on gait data = the port never engaged (dead port); ~1 at a fast gait with flight phases = port acting as an always-on field rather than a contact model |
| `contact_spring_ratio` | mean $\lVert\sum_i k_i\varphi(g_i)J_{n,i}\rVert / \lVert\nabla\hat V\rVert$ | how much of the conservative field lives in the port; large is legitimate, it just changes which grad-V number is meaningful |
| `contact_k`, `contact_c`, `contact_mu` | per-contact stiffness and compression damping at gauge scale $c^*$; friction $\mu$ raw (a force ratio, gauge-free) | absolute values are soft ($k$ trades against the learned gap scale); the meaningful read is the *trend across checkpoints of one run* — stiffness creep makes the drift stiffer over training and adds quadrature/BPTT noise at exactly the high-speed regime |
| `contact_in_frac_per`, `contact_gap_mean/min` | per-contact activity split and gap statistics on the eval states | which of the $K$ points actually act (silent points are fine — spare capacity), and how deep the working penetrations sit |
| `coriolis_force_corr` | Pearson of $\hat C(q,\dot q)\dot q$ vs truth | the term that governs fast motion ($\propto \dot q^2$) is wrong; *negative* values mean $\partial M/\partial q$ is being used as something else entirely (historically: a touchdown brake) |
| `damping_affine_R2`, `damping_learned/true` | affine fit of $c^*\,\mathrm{softplus}(\log d)$ onto `dof_damping`, across the 9 DOFs | passive joint damping mis-learned. Caveat: 9 points, so the $R^2$ is coarse; read the per-DOF lists too |
| `G_rel_frob_err`, `G_actuator_cosine` | $\lVert c^*\hat G - G\rVert_F/\lVert G\rVert_F$; per-actuator column cosines | actuator *authority* wrong (Frobenius) vs actuator *direction* wrong (cosines). Degraded cosines on a gait are a specific alarm: actions phase-lock with stance, so missing contact propulsion regresses onto $G_a$ |

### Reading guide: why kinetic $R^2$ can be low while total-$H$ $R^2$ is high

This combination appears in real audits and is usually a gauge nuisance rather than a physics failure. Three effects stack:

1. Kinetic $R^2$ is scale-locked to $c^*$ with no offset, while potential and total-$H$ get affine refits. Kinetic energy must vanish at $\dot q = 0$, so granting it an offset would be unphysical — the strictness is intentional.
2. On a steady gait, $e_{\text{kin}}$ has a large mean and small variance. $R^2$ divides squared error by that small variance, so a pure scale mismatch (learned kinetic tracking truth at 0.95 correlation but at 0.75× the $c^*$-implied scale) craters the metric while the shape is nearly perfect.
3. $c^*$ is fit on mass *entries* weighted equally; the kinetic form weights entries by which DOFs the gait excites. If magnitude errors concentrate on velocity-heavy entries, the effective kinetic scale diverges from $c^*$.

Meanwhile total-$H$ is variance-dominated by the potential (root-height bobbing through gravity) and enjoys the affine refit, so it mostly reflects potential shape. The report does not currently separate kinetic shape from kinetic scale; when in doubt, compare `mass_entry_corr` (shape) against `mass_rel_frob_err` (magnitude) as the proxy.

## Case study: three audits, two confounds diagnosed

All three are rollout-fit (`fit_horizon` 4) cheetah models audited on their own policy's data. `contact_roll` is the old model class (contact expressed only through a velocity-jump-gated damping term, since removed from the codebase, 2M updates); `cforce_roll v1` added the explicit contact port but shipped with a gap init 25 smoothing-widths into the softplus tail (dead port, 500k); `v2` is the fixed init (600k).

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
# audit an RL run's model on its own policy's state distribution
python evaluations/hamiltonian_recovery.py \
    --dynamics_path <ckpt>.dynamics.pth \
    --checkpoint    <ckpt>.pth \
    --mode mbq_structured_quad_cforce_roll \
    --contact_force 4 \
    --out audit.json

# offline: fit a fresh model on OU data first, then audit it
python evaluations/hamiltonian_recovery.py --fit_steps 8000 --fit_horizon 4 --contact_force 4
```

The architecture flag (`--contact_force`) must match how the run was configured; a mismatch fails loudly at `load_state_dict`. (Sidecars from the removed gated-damping era contain weights the current model class no longer has and cannot be loaded; their audits are recorded above.) `--mode` names the hyperparameter row used to size the policy networks when `--checkpoint` collects the data. Runs on CPU in a few minutes.

## Caveats and sharp edges

- **The eval distribution is the collecting policy's.** Audits of different checkpoints see different gaits, so cross-run comparisons are approximate — trends and signs are trustworthy, second-decimal differences are noise. (Also: collection runs at uniform $dt = 0.01$ for clean extraction, while training uses irregular sampling.)
- **This is an identifiability report.** It says nothing about open-loop rollout accuracy; that is a separate, complementary measurement.
- **Visited states only.** A term can be perfectly recovered on the gait manifold and arbitrarily wrong off it; the report cannot see the off-manifold behavior the critic's action-sampling probes.
- **Kinetic $R^2$ is strict by design** (see the reading guide); a low value alongside high mass-entry correlation usually indicates scale attribution rather than wrong physics.
- **Truth extraction is cheetah-specific**: planar floating base, linear torque actuators (the $G$ extraction relies on this), and the `obs = [qpos[1:]; qvel]` layout.
- **Damping audit covers the diagonal base only**, and its $R^2$ rests on 9 points.
