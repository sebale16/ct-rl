---
title: Structured Dynamics on Cheetah — the Contact-Driven Instantiation
tags: [ct-rl, ct-sac, port-Hamiltonian, cheetah, contact, model-based]
robots: noindex
---

# Structured Dynamics on Cheetah — the Contact-Driven Instantiation

:::info
**Overview.** This note instantiates the structured port-Hamiltonian model of [Structured Port-Hamiltonian Dynamics for Model-Based CT-SAC](structured_dynamics_model.md) on the system it was built for: the planar dm_control cheetah (9 DOF, 6 actuators, locomotion by ground contact). Cheetah is where every hard feature of the general construction earns its place — the cyclic root coordinate, the generated Coriolis force that dominates at running speed, and the explicit contact port with its two structural leak-closures. The smooth minimal counterpart, where every term has a closed form, is the companion note [Structured Dynamics on Cartpole](structured_dynamics_cartpole.md); the audit machinery referenced throughout is the [Hamiltonian recovery report](hamiltonian_recovery_report.md).
:::

[TOC]

---

## 1. The system

A planar half-cheetah: a torso free to translate in the sagittal plane and pitch, with a three-joint back leg and a three-joint front leg. Total mass $14$ kg, torso $6.25$ kg.

```
        root z (slide, ↑)                 root y = pitch (hinge, ↺)
              ↑                        ┌──────────────────────────┐
              │      ═════════════════│         torso  6.25 kg    │═══► head
              │                       └───●──────────────────●───┘
   root x (slide, →)               bthigh ●                  ● fthigh
   cyclic: position dropped        k=240  │ gear 120   k=180 │ gear 90
   from the observation,           d=6.0  │            d=4.5 │
   velocity kept; the reward       bshin  ●                  ● fshin
   is its speed                    k=180  │ gear 90    k=120 │ gear 60
                                   d=4.5  │            d=3.0 │
                                   bfoot  ●                  ● ffoot
                                   k=120  ╱ gear 60    k=60  ╱ gear 30
                                   d=3.0 ▟             d=1.5▟
   ═══════════════════════════════════════════════════════════════════ ground
        ground reaction at the feet: normal support + friction —
        the dominant force of the task, and the reason for the contact port
```

| DOF (config index) | joint | actuated | spring $k$ | damping $d$ |
|---|---|---|---|---|
| 0 | root $x$ (slide) | no | — | 0 |
| 1 | root $z$ (slide) | no | — | 0 |
| 2 | root pitch (hinge) | no | — | 0 |
| 3–5 | back thigh / shin / foot | gear 120 / 90 / 60 | 240 / 180 / 120 | 6.0 / 4.5 / 3.0 |
| 6–8 | front thigh / shin / foot | gear 90 / 60 / 30 | 180 / 120 / 60 | 4.5 / 3.0 / 1.5 |

The three root DOFs are unactuated, unsprung, and undamped: the torso can be moved *only* by leg torques transmitted through ground contact. Propulsion is friction; support is the normal force. The leg joints also carry hard range limits — like cartpole's rail ends, limit impacts are a real force with no term in the model class, a minor leak-risk kept in mind when reading audits.

## 2. Observation and DOF layout

The task observation is already raw generalized coordinates — the reason cheetah could be the first system, and the property the raw-state mode retrofits onto other domains:

$$
x_{\text{obs}} = [\,\underbrace{q_{1..8}}_{\text{8 positions: root }z,\ \text{pitch, 6 joints}}\,;\ \underbrace{\dot q_{0..8}}_{\text{9 velocities, root }x\ \text{included}}\,].
$$

Root $x$ is the **cyclic coordinate** of main-note §2.6: its position is dropped for translation invariance (the physics cannot depend on where along the track the cheetah stands) while its velocity — the thing the task rewards — is kept. The layout therefore maps 8 observed positions onto config DOFs $1..8$ and holds the root-$x$ slot of every configuration gradient at zero.

| layout field | value |
|---|---|
| observed positions / velocities | $8$ / $9$ |
| cyclic config DOFs | root $x$ (index 0) |
| position-to-config map | observed $i \mapsto$ config $i{+}1$ |
| actuator map | dense $6 \to 9$ |
| mass-input restriction | root $z$ removed from the mass network when the contact port is active |
| contact tangent DOF | root $x$ (friction is what propels) |

## 3. The true terms the model must recover

Unlike cartpole, nothing here has a two-line closed form — which is the point of learning them — but each object's *structure* is known and is what the audits compare against:

- **Mass matrix** $M(q)$: dense $9\times9$, configuration-dependent through the leg geometry, and invariant to both root $x$ (cyclic) and root $z$ (translating the mechanism vertically cannot change inertia). That second invariance is not representational pedantry: it is exactly the axis along which ground reaction leaked into $M$ when contact had no channel (§5).
- **Potential** $V(q) = V_{\text{grav}}(z, \text{pitch}, q_{\text{legs}}) + \tfrac12\sum_j k_j q_j^2$: gravity on all bodies plus the six joint springs above (spring reference at zero).
- **Coriolis** $c(q,\dot q) \propto \dot q^2$, generated from $\partial M/\partial q$: negligible in slow motion and *dominant* at running speed — the term the whole structured construction exists to get right, and the first casualty of any contact leak into $M$.
- **Damping** $D$: the constant diagonal above — zero on the root, substantial on the legs. This is the function class the model's diagonal damping matches one-for-one.
- **Actuator port** $G_a$: six columns, one per motor, each a torque on a single joint scaled by its gear ($120$ down to $30$); zero rows on the three root DOFs. The model learns it as a dense $9\times6$ map, so the sparsity and the zero root rows are predictions the audit checks (per-actuator cosines), and their corruption is diagnostic (§5).
- **Ground reaction**: at up to two feet, a unilateral normal force plus friction — the largest force in the system during stance and the only source of forward propulsion. This is what the contact port of main-note §2.7 represents with $K$ learned point contacts: gap functions $g_i(q)$ whose gradients give the normal directions (root-$x$ component structurally zero — a vertical force cannot propel), tangential directions $e_x + \partial h_i/\partial q$ carrying the friction that can, Hunt–Crossley normal magnitudes, and regularized Coulomb friction.

## 4. What the model learns

| learned object | shape here | must converge to (up to the global gauge $c^*$) |
|---|---|---|
| Cholesky mass network | $7$ inputs (8 positions minus root $z$ when the port is on) $\to 45$ entries of $L$ | the true $M(q)$: dense leg coupling, no $z$- or $x$-dependence |
| potential network | $8$ positions $\to$ scalar | gravity + the six joint springs (offset gauge; shares the conservative role with the port springs — see below) |
| generated Coriolis | autodiff of the mass network | the true $C(q,\dot q)\dot q$, the fast-gait term |
| damping diagonal | $9$ parameters | $[0,0,0,6,4.5,3,4.5,3,1.5]$ |
| actuator port | dense $9\times6$ | gear-scaled single-joint columns, zero root rows |
| contact port | $K$ gap nets $g_i(q)$, offset nets $h_i(q)$, per-contact $k_i, c_i, \mu_i$ | foot-height-like gaps switching with stance; normal support; friction carrying propulsion |

Two structural decisions, both audit-driven (main-note §2.7), shape what "learning" means here:

1. **The mass network cannot see root $z$** when the port is active, so $\partial M/\partial z \equiv 0$ by construction and ground reaction has nowhere to go but the port. Cartpole shows the same invariance being *learned* voluntarily; cheetah needed it *enforced*, because with contact in the data the $M$-route is the better local minimum.
2. **Propulsion can only flow through friction.** The normal directions have a structurally zero root-$x$ component; $\nabla V$'s root-$x$ slot is zero by cyclicity. If the port is dead, the fit's only remaining forward-force channel is $G_a$ — which is how a silent port announces itself as corrupted actuator cosines.

And one identifiability gauge specific to contact: the port's gap springs are themselves a conservative field, so on data where feet are nearly always planted, $V$ and $\sum_i k_i\Phi(g_i)$ hold the gravity field jointly — only the sum is identified, and audits read the combined conservative gradient.

## 5. What cheetah taught the model class

Each structural feature above was forced by an audit finding on this system (full detail in the [recovery report](hamiltonian_recovery_report.md) and the progress note's timeline):

- **The mass-leak.** With no contact channel, the fitted $M$ carried a root-height dependence $6\times$ its joint-angle dependence (true value: $0$). A $z$-switched $M$ modulates every force through $M^{-1}$ and its gradient fires Coriolis-like $\dot q^2$ brakes at touchdown — a serviceable impact model whose price was the *real* Coriolis term (recovery correlation $0.106$), i.e., the model could represent slow gaits and not fast ones. Closing the leak structurally took Coriolis recovery to $0.65{+}$ in a quarter of the training.
- **The propulsion-leak.** A contact port initialized too deep in its force law's saturated tail never engaged; the displaced ground reaction split by structure — its conservative part into $V(z)$, its propulsive part onto $G_a$, where actions phase-lock with stance (three actuator cosines fell to $0.37$–$0.53$). Waking the port restored them to $0.90$–$1.00$. The critic reads $b(x,a)$ at *candidate* actions, so an actuator port inflated with contact propulsion overstates action authority everywhere off the gait cycle.
- **The gravity migration.** On always-in-contact data the recovered $\nabla V$ alone fell to correlation $-0.06$ while the dynamics were unaffected: gravity had moved into the gap springs, the $V$–port gauge above, and the reason the combined conservative gradient is the audited object.

The through-line, and the transferable lesson of this instantiation: least squares routes every unmodeled force into whichever learnable channel can imitate its switching pattern, and the audit's job is to catch the imitation before the critic consumes it.

## 6. Role in the program

Cheetah is the benchmark the model-based question is posed on: model-free baseline, oracle drift, and learned structured drift share one target construction, and the definitive comparisons live in the progress note. The headline standing of this instantiation: the physical recovery is the strongest the project has produced (mass correlation $\approx 0.96$, Coriolis $0.66$–$0.83$, damping and actuator structure recovered), the same target construction fed the *true* drift beats the model-free baseline outright — and the learned-drift runs still cap well below both, which localizes the open problem at the drift's accuracy *where the critic reads it* (candidate actions, faster-than-visited gaits) rather than on the visited distribution the audits certify. Separating that in-the-loop question from contact is what the [cartpole instantiation](structured_dynamics_cartpole.md) is for.
