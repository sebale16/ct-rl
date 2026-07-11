---
title: Structured Dynamics for CT-SAC — Progress and Results
tags: [ct-rl, ct-sac, port-Hamiltonian, model-based, results, timeline]
robots: noindex
---

# Structured Dynamics for CT-SAC — Progress and Results

Base: https://hackmd.io/@-YScJRgTQoiFn3RF3xJ3Fg/rkgsjiWQzg

:::info
**Overview.** The experiment log of the model-based CT-SAC effort on cheetah-run: what was built and tested, in order; the result at each stage; and what is in flight now. The model and the algorithm it plugs into are derived in the companion note [Structured Port-Hamiltonian Dynamics for Model-Based CT-SAC](https://hackmd.io/@-YScJRgTQoiFn3RF3xJ3Fg/HJGxLGfXMg). The identifiability metrics quoted throughout are defined in [The Hamiltonian Recovery Report](https://hackmd.io/@-YScJRgTQoiFn3RF3xJ3Fg/S1LqGV1EMl).
:::

[TOC]

---

## 1. The program

CT-SAC's critic target estimates the generator by a finite difference over the *sampled* next state, an estimator whose variance grows as $\mathcal O(1/u)$ at small step sizes. The model-based variant evaluates the generator analytically, $b\cdot\nabla V$, from a dynamics drift $b(x,a)$ — the simulator's own drift for validation, a learned one for the real method. The work has run in that order: first validate the target machinery with the oracle drift, then earn the learned drift. Every stage is measured on two axes — end-to-end return, and term-by-term **identifiability** of the learned physics against the true cheetah dynamics (mass, potential, Coriolis, damping, actuator port, contact geometry), because a model can predict transitions well while attributing forces to the wrong terms, and the critic reads the model off-distribution where wrong attribution becomes wrong control.

### Run vocabulary

| label | drift | critic target | buffer |
|---|---|---|---|
| `top` | — (model-free) | sampled finite difference | 300k |
| `top_buf1m` | — (model-free) | sampled finite difference | 1M |
| `mbq_vhead` | oracle | first-order $b\cdot\nabla V$, V-head | 300k |
| `mbq_vhead_quad` | oracle | sub-step quadrature, V-head | 300k |
| `mbq_vhead_quad_buf1m` | oracle | sub-step quadrature, V-head | 1M |
| cforce (`mbq_structured_quad_cforce`) | learned structured + contact port, one-step fit | sub-step quadrature, V-head | 300k |
| cforce_roll (`…_cforce_roll`) | same, $H{=}4$ rollout fit | sub-step quadrature, V-head | 300k |
| cforce_buf1m, cforce_roll_buf1m | the two cforce modes | sub-step quadrature, V-head | 1M |

---

## 2. Timeline

### June 28–30 — the generator machinery, validated against the oracle

The model-based target landed with an oracle (simulator) drift for validation and a black-box learned port-Hamiltonian drift fit online from replay. Three findings shaped everything after. The first-order target $b\cdot\nabla V$ with a learned drift helps only while the increment $|b\,\Delta t|$ stays small — at the benchmark step size the linearization is the binding error. A higher-order predict-then-difference form lifted that limit on simple systems and collapsed on cheetah, because it reads the learned model at its own off-distribution predictions: model accuracy is the bottleneck, and that form was removed. And a 12-seed comparison at the physics-floor step size had the oracle generator winning the per-update variance measurements while losing on return. What survived into the current stack: a dedicated scalar **V-head** so the generator reads a clean, sample-free $V$ and $\nabla V$ (behind a value warmup), and the **sub-step quadrature target** — roll the model $m$ Euler sub-steps, read $V^{\text{tgt}}(\hat x_m) - V^{\text{tgt}}(x)$ off the head — validated on cartpole.

### June 30 – July 1 — a model that can be accurate on cheetah

Offline diagnosis: cheetah's hard-to-fit accelerations are the centrifugal/Coriolis term, quadratic in $\dot q$ (contacts were checked and ruled out as the driver of the *offline* fit gap). A black-box energy has to reconstruct that structure by brute force; the **structured model** — SPD mass $M(q)$, potential $V(q)$, diagonal damping, actuator port, Coriolis generated from $\partial M/\partial q$ — supplies it by construction. Offline it nearly doubled the one-step acceleration fit (corr 0.49 → 0.91 on white-noise exploration, 0.47 → 0.84 on OU) and kept the $H{=}8$ open-loop rollout bounded (rel-err 0.46, flat) where the black box diverged (1.37), most of the way to the oracle (≈ 0.35). An interim contact-gated damping channel improved smooth-exploration fits (0.46 → 0.67), but a damping force lies along the current velocity — it can neither hold a resting foot nor propel — and it was later removed outright.

### July 3 — the rollout fit, the stability guards, and the recovery audit

Three pieces in quick succession:

- **Multi-step rollout fit** ($H{=}4$, backprop through the roll): offline, monotone gains over the one-step fit on every held-out metric at matched budget (accel corr 0.47 → 0.57, $H{=}8$ open-loop rel-err 0.83 → 0.77), and — the operative fact for $10^6$-update runs — continued one-step training *degrades* held-out metrics after its early peak while the rollout fit keeps improving.
- **Guards.** The first rollout-fit benchmark run died at 260k steps: one bad prediction blew up the compounding roll ($\dot q^2$ Coriolis), and NaN reached the critic through the target. The fit now clamps rolled increments, clips gradients, and skips non-finite updates; a non-finite model-based target raises and terminates the run — a model-free fallback would contaminate the benchmark, so every run is a pure model-based result or an explicit failure. The guarded rerun survived but ran flat (returns 30–38), later understood as the same contact confound the port fixed.
- **The Hamiltonian recovery audit**: term-by-term comparison of the learned $M$, $V$, $D$, $G_a$, and generated Coriolis against MuJoCo's own quantities on visited states, up to the single global gauge scale. This became the forensic instrument for everything below.

### July 5–7 — the explicit contact-force port

The audits found why cheetah resisted. With no contact term in the model class, least squares routed ground reaction through a spurious root-height dependence of $M(q)$ — probe ratio 6.0 where the truth is 0 — and the Coriolis force generated from that same tensor collapsed to correlation 0.106: the model could represent slow gaits and had no idea about fast ones. The **contact port** (four learned point contacts: gap and offset networks whose gradients form the contact Jacobian, Hunt–Crossley normal force, regularized Coulomb friction) plus the structural fix (root height removed from the mass input) closed the leak: $\partial M/\partial z \equiv 0$ by construction, and Coriolis recovery reached 0.65 in a quarter of the training — even in the run whose port was dead. Dead, because v1 shipped with the gap init 25 smoothing-widths into the saturated softplus tail: zero engagement, the conservative part of contact hiding in $V(z)$, the propulsive part regressing onto $G_a$ (three actuator cosines at 0.37–0.53). The quiet-but-reachable re-init (2.5 widths) woke the port — in-contact fraction 0.87, gentle springs at 8% of $\nabla V$ — and the actuator recovery snapped back (Frobenius error 0.85 → 0.38, cosines 0.90–1.00). The gated-damping mechanism and its plumbing were then deleted.

### July 7–9 — end-to-end cforce runs: peaks, declines, and the buffer

The port lifted end-to-end returns 3–4× over the pre-port structured runs, to peaks of 330–357 — followed by long declines. The peak-vs-final audits cleared the rollout-fit model (its physics *improves* across the decline, §4), and the declines outlive the 300k replay buffer — the peak-gait data ages out, deleting any rebound attractor — which made the buffer the tested variable and added the buf1m rows. The completed 2×2 (fit mode × buffer) peaked at 295–357 in every cell, against ~800 for the single-seed model-free `top`: a cap no cell escaped.

### July 9–10 — ceiling isolation: the target construction beats the baseline

With every learned-model cell capping near 300, the remaining suspects were the target construction itself and the model's coverage where the target reads it. One run separates them: `mbq_vhead_quad_buf1m` — the exact cforce target with the oracle drift, i.e. zero model error. Verdict at 12 seeds, 1M steps: **mean 1203 ± 165 vs 929 ± 297 for `top`** (median 1255 vs 1058), zero collapses vs several, still climbing at 1M while `top` plateaus from 400k. The backup construction is exonerated outright — it *beats* the model-free baseline — so the cforce cap is the learned drift at the points the target reads: candidate actions and faster-than-visited gaits. Coverage, in one word. The single-seed recovery audits of the 2×2 (§4) say the same thing from the other side: three of the four cells hold or improve their physics on the visited states across the decline.

### July 11 — irregular-duration train/use mismatch and flow matching

The next audit isolated a numerical mismatch before the remaining contact and coverage questions. The historical learned-model loss used one explicit Euler prediction over each complete replay duration,
$x+b_\phi(x,a)\Delta t$, even though the critic's quadrature rolled the vector field through shorter sub-steps. In a squared endpoint loss, a 30 ms sample has $(30/2)^2=225$ times the drift-error weight of a 2 ms sample. On the two-tail cheetah data, the sparse 30 ms transitions consequently supplied about 94% of the total $\Delta t^2$ weight and trained the instantaneous drift toward a coarse 30 ms secant—the wrong object for the critic's internal rollout.

The fit now performs **finite-duration flow matching**. It retains every replay transition and its original duration, holds the recorded action fixed, and integrates the learned drift with differentiable internal steps no larger than the environment's 2 ms physics step before comparing the endpoint with $x'$. Thus 2, 10, and 30 ms samples take 1, 5, and 15 internal steps; none are rounded to 2 ms or discarded. Single-transition fitting, $H>1$ rollout fitting, offline recovery fits, and the quadrature critic target all call the same flow integrator.

An oracle numerical check on held-out cheetah states verifies the integration change itself: relative endpoint error fell from 0.499 to 0.078 at 10 ms and from 0.817 to 0.062 at 30 ms; at 2 ms the two paths coincide (0.047). These are solver checks with the true MuJoCo drift, not learned-policy results. The discriminating next result is therefore a fresh learned cforce run under the flow-matching objective; all earlier cforce scores in this document used the historical coarse-transition fit.

---

## 3. Scoreboard (cheetah-run)

| run | seeds | result |
|---|---|---|
| `top` (model-free) | 12 | 929 ± 297 at 1M, median 1058; plateau from 400k; several collapses |
| `top_buf1m` | — | **running** — attribution ablation (§5) |
| `mbq_vhead` (oracle, first-order) | few | 600–950 |
| `mbq_vhead_quad` | — | **running** — attribution ablation (§5) |
| `mbq_vhead_quad_buf1m` | 12 | **1203 ± 165 at 1M**, median 1255; zero collapses; still climbing |
| pre-port structured modes | 1 each | 3–4× below the cforce peaks; first rollout-fit run NaN-died at 260k, guarded rerun flat at 30–38 |
| cforce / cforce_roll (300k) | 1 each | peak 330 / 357 → final 192 / 236 |
| cforce_buf1m / cforce_roll_buf1m | 1 each | peak 296 / 295 → final 264 / 219 |

---

## 4. Recovery audits of the cforce 2×2 (single seed, peak vs final checkpoint)

Each run's saved dynamics model was audited on its own policy's state distribution at the best-eval ("peak") and last ("final") checkpoints. Gauge conventions, metric definitions, and reading guides are in [The Hamiltonian Recovery Report](https://hackmd.io/@-YScJRgTQoiFn3RF3xJ3Fg/S1LqGV1EMl). One seed per cell; the eval distribution is each checkpoint's own gait, so cross-run trends and signs are trustworthy and second-decimal differences are noise.

**Physics recovery** (gauge-fixed; ∇V-combined is the potential read that respects the $V$↔port gauge):

| run | ckpt (step) | reward | mass rel-err | mass corr | $\partial M/\partial z$ | potential $R^2$ | kinetic $R^2$ | total-$H$ $R^2$ | ∇V comb. | Coriolis | damping $R^2$ | $G_a$ err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cforce | peak (0.8M) | 329.9 | 0.26 | 0.96 | 0.00 | 0.53 | 0.76 | 0.83 | 0.70 | 0.85 | 0.14 | 0.60 |
| cforce | final (2.0M) | 192.1 | 0.28 | 0.96 | 0.00 | 0.84 | −0.36 | 0.83 | 0.18 | 0.06 | 0.42 | 0.92 |
| cforce_roll | peak (0.6M) | 357.2 | 0.62 | 0.78 | 0.00 | 0.73 | −3.18 | 0.65 | 0.79 | 0.62 | 0.42 | 0.70 |
| cforce_roll | final (1.24M) | 236.4 | 0.36 | 0.93 | 0.00 | 0.82 | 0.64 | 0.91 | 0.63 | 0.69 | 0.02 | 0.65 |
| cforce_buf1m | peak (1.53M) | 295.9 | 0.25 | 0.97 | 0.00 | 0.83 | 0.80 | 0.71 | 0.82 | 0.71 | 0.07 | 0.60 |
| cforce_buf1m | final (2.0M) | 263.5 | 0.54 | 0.83 | 0.00 | 0.89 | −1.68 | 0.80 | 0.82 | 0.75 | 0.01 | 0.73 |
| cforce_roll_buf1m | peak (1.32M) | 295.4 | 0.45 | 0.89 | 0.00 | 0.89 | 0.85 | 0.89 | 0.80 | −0.05 | 0.01 | 0.62 |
| cforce_roll_buf1m | final (2.0M) | 219.1 | 0.21 | 0.98 | 0.00 | 0.96 | 0.76 | 0.89 | 0.89 | 0.50 | 0.75 | 0.61 |

**Contact port and actuators:**

| run | ckpt | gauge $c^*$ | in-frac | spring/∇V | gap mean / min | $k$ range | $\mu$ range | $G_a$ cosines |
|---|---|---|---|---|---|---|---|---|
| cforce | peak | 28.1 | 0.40 | 0.44 | 0.23 / −0.47 | 19.7–50.0 | 1.34–1.42 | 0.76–0.98 |
| cforce | final | 6.5 | 0.36 | 0.03 | 0.03 / −0.13 | 6.5–8.7 | 1.01–1.29 | 0.57–0.90 |
| cforce_roll | peak | 13.3 | 0.37 | 0.11 | 0.28 / −0.53 | 7.9–15.6 | 0.90–1.03 | 0.76–0.94 |
| cforce_roll | final | 14.8 | 0.52 | 0.62 | 0.61 / −0.54 | 15.7–33.7 | 0.68–1.45 | 0.75–0.98 |
| cforce_buf1m | peak | 30.6 | 0.52 | 0.36 | 0.20 / −0.52 | 19.8–58.4 | 0.82–1.11 | 0.81–1.00 |
| cforce_buf1m | final | 14.6 | 0.50 | 0.19 | 0.11 / −0.75 | 11.2–20.6 | 0.93–1.20 | 0.69–0.95 |
| cforce_roll_buf1m | peak | 16.1 | 0.50 | 0.23 | 0.08 / −0.25 | 16.5–30.9 | 0.64–1.21 | 0.82–0.92 |
| cforce_roll_buf1m | final | 19.7 | 0.51 | 0.21 | 0.16 / −0.57 | 24.9–41.2 | 1.02–1.18 | 0.79–0.99 |

---

## 5. What is being tested now

**The attribution 2×2 (cheetah).** `mbq_vhead_quad_buf1m` beat `top`, but it differs from `top` in two variables at once — the target construction and the buffer size. Two rows change one variable each and are running now: `top_buf1m` (model-free with the 1M buffer: does the buffer alone rescue the model-free plateau?) and `mbq_vhead_quad` (oracle quadrature V-head target at the standard 300k: does the target alone beat `top`?). Together with the two finished corners they complete a 2×2 in (target, buffer) for paper-grade attribution of the win.

**The raw-state validation ladder (cartpole-swingup).** The ceiling isolation leaves one question cheetah cannot answer cheaply: can a *learned* model in the loop reach oracle-level control at all? A smooth, contact-free, low-DOF system isolates that question from contact and coverage. Cheetah was nearly the only DMC task observing raw generalized coordinates — pendulum, cartpole, and acrobot expose cos/sin-encoded angles, which break the structured model's kinematic block ($\mathrm d(\cos\theta)/\mathrm dt \neq \dot\theta$). Raw-state observation ($[q;\dot q]$ straight from the physics, any hinge/slide domain) now exists across domains, the generic DOF layout comes with it, the oracle drift is exact there (correlation 1.000 on cartpole), and the recovery audit generalizes to the same map. The three-way — `top` / `mbq_vhead_quad` / `mbq_structured_quad_roll`, raw observations, 500k steps, 1M buffer so nothing evicts — reads out cleanly: learned ≈ oracle pins the cheetah gap on contact/coverage; learned ≪ oracle convicts the in-loop coupling itself, debuggable at roughly a hundredth of the cost per experiment.
