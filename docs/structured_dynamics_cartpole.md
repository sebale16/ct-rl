---
title: Structured Dynamics on Cartpole — the Minimal Instantiation
tags: [ct-rl, ct-sac, port-Hamiltonian, cartpole, model-based]
robots: noindex
---

# Structured Dynamics on Cartpole — the Minimal Instantiation

:::info
**Overview.** This note instantiates the structured port-Hamiltonian model of [Structured Port-Hamiltonian Dynamics for Model-Based CT-SAC](structured_dynamics_model.md) on the smallest system that exercises it: the dm_control cartpole (2 DOF, one actuator, no contact). Every general object of the main note — the SPD mass matrix $M(q)$, the potential $V(q)$, the generated Coriolis force, the diagonal damping $D$, the actuator port $G_a$ — takes a two-line closed form here, so the system doubles as the worked example of what the model must learn and as the cheap rung of the validation ladder: it isolates *learned model in the RL loop* from *contact*, the two difficulties that cheetah poses at once. The contact-driven instantiation is the companion note [Structured Dynamics on Cheetah](structured_dynamics_cheetah.md).
:::

[TOC]

---

## 1. The system

```
                         ○   pole tip
                        /
                       /
                      /    θ : hinge DOF, unactuated, unlimited
                     /         θ = 0 upright, θ = π hanging
                    /          pole m_p = 0.1 kg, l = 0.5 m (COM), J = 0.0344 kg m²
               ┌───┴────┐
   F = 10u →  │  cart   │   m_c = 1.0 kg
               └──┬──┬──┘
   ═══╪═══════════╧══╧═══════════╪═══   x : slide DOF, actuated
    −1.8 m        0            +1.8 m      motor u ∈ [−1,1], gear 10 → F ∈ [−10, 10] N
                                           rail ends are hard joint limits
                    g = 9.81 ↓
```

Two degrees of freedom, one actuator, and the actuator is on the *wrong* DOF: the pole can only be influenced through the configuration-dependent mass coupling. That underactuation is what makes this toy informative — the control-relevant physics lives in exactly the objects the structured model learns.

Constants read from the simulator:

| quantity | value |
|---|---|
| cart mass $m_c$ | $1.0$ kg |
| pole mass $m_p$, COM distance $l$ | $0.1$ kg, $0.5$ m ($m_p l = 0.05$) |
| pole inertia about the hinge $J$ | $0.0344$ kg m² |
| potential scale $m_p g l$ | $0.4905$ J ($E_{\text{up}} - E_{\text{down}} = 2 m_p g l = 0.981$ J) |
| actuator | one force motor on the slider, gear $10$ |
| joint damping | $[5\times10^{-4},\ 2\times10^{-6}]$ — essentially conservative |
| slider range | $x \in [-1.8, 1.8]$ m, hard-limited; hinge unlimited |

The rail ends are a separate force class. MuJoCo's joint-limit constraint is stiff and dissipative, and early exploration does not reliably avoid it: in a matching OU audit, 40% of states lay within 0.2 m of a rail and 7.3% carried a constraint-force norm above 1. Leaving that force outside the model made it leak into the global mass, gravity potential, and damping. The current cartpole layout therefore gives the rail its own smooth unilateral spring energy and localized passive damping. In the interior this port vanishes exponentially, leaving the translation-invariant cartpole mechanics; at either stop it supplies the missing inward force and non-positive damping power.

## 2. Observation and DOF layout

The model requires observations that are raw generalized coordinates, because its position-drift block *is* the identity $\mathrm d q_{\text{obs}}/\mathrm dt = \dot q$. The native dm_control cartpole observation is trigonometric — $[x, \cos\theta, \sin\theta, \dot x, \dot\theta]$ — which breaks that identity ($\mathrm d(\cos\theta)/\mathrm dt = -\sin\theta\,\dot\theta \neq \dot\theta$), so the environment is run in a raw-state mode:

$$
x_{\text{obs}} = [\,\underbrace{x,\ \theta}_{q\ (2)}\,;\ \underbrace{\dot x,\ \dot\theta}_{\dot q\ (2)}\,].
$$

The DOF layout (main note §2.6) is then the trivial one, and the contrast with cheetah is the point of writing it out:

| layout field | cartpole | cheetah |
|---|---|---|
| observed positions / velocities | $2$ / $2$ | $8$ / $9$ |
| cyclic config DOFs | none | root $x$ (position dropped, velocity kept) |
| position-to-config map | identity | observed positions $\to$ config DOFs $1..8$ |
| actuator map | sparse: action $\to$ slider DOF 0 | dense $6 \to 9$ |
| mass-input restriction | cart $x$ excluded exactly | root $z$ removed when the contact port is active |
| potential input | $(\sin\theta,\cos\theta)$; cart $x$ excluded | observed positions |
| constraint port | known rail locations, learned positive spring/damping | $K$ learned point contacts |

## 3. The true terms the model must recover

Everything the main note's §2 defines abstractly has a closed form here. The mass matrix, measured from the simulator at three configurations, is

$$
M(q) \;=\; \begin{pmatrix} m_c + m_p & m_p l\cos\theta \\ m_p l\cos\theta & J \end{pmatrix}
\;=\; \begin{pmatrix} 1.1 & 0.05\cos\theta \\ 0.05\cos\theta & 0.0344 \end{pmatrix},
$$

with the two diagonal entries constant and the coupling passing through zero at $\theta = \pi/2$ exactly. Two structural facts follow. $M$ is independent of the cart position $x$ — the translation invariance that cheetah has in its root height — and the *only* configuration dependence is the off-diagonal $\cos\theta$. The potential is

$$
V(q) = m_p g l \cos\theta = 0.4905\,\cos\theta \ \text{J} \qquad (\text{+ arbitrary offset}),
$$

independent of $x$ for the same reason. The Coriolis force is generated, not learned, and here the generation of §2.4 can be followed by hand: the only nonzero entry of $\partial M/\partial q$ is

$$
\frac{\partial M}{\partial \theta} = \begin{pmatrix} 0 & -m_p l\sin\theta \\ -m_p l\sin\theta & 0 \end{pmatrix},
$$

and assembling $c(q,\dot q) = \dot M \dot q - \tfrac12 \dot q^\top \tfrac{\partial M}{\partial q}\dot q$ gives the single centrifugal term of the textbook cart-pole equations,

$$
c(q, \dot q) = \begin{pmatrix} -m_p l \sin\theta\,\dot\theta^2 \\ 0 \end{pmatrix}
\quad\Longleftrightarrow\quad
(m_c{+}m_p)\ddot x + m_p l\cos\theta\,\ddot\theta - m_p l\sin\theta\,\dot\theta^2 = F .
$$

The remaining ports are small: damping is the near-zero diagonal above (the system is conservative to measurement precision — a passive release swings for six seconds with $0.002\%$ energy drift), and the actuator port is the rank-one map

$$
G_a = \begin{pmatrix} 10 \\ 0 \end{pmatrix}.
$$

**Why underactuation makes this a good test.** The zero in $G_a$'s second row says the action exerts no direct pole torque; every bit of pole control flows through the coupling $m_p l\cos\theta$ inside $M$. A model that gets $M$'s off-diagonal wrong mispredicts precisely the action-to-pole channel the critic's candidate-action reads depend on — the same reason cheetah's audits treat the actuator cosines and the mass structure as control-relevant rather than cosmetic.

## 4. What the model learns

Instantiating the learned objects of main-note §2 (all networks two hidden layers unless stated):

| learned object | shape here | must converge to (up to gauge) |
|---|---|---|
| Cholesky mass network | $(\sin\theta,\cos\theta) \to 3$ entries of $L$, $M = LL^\top + \varepsilon I$ | the matrix above: constant diagonal, $0.05\cos\theta$ coupling, structurally no $x$-dependence |
| base potential network | $(\sin\theta,\cos\theta) \to$ scalar | $m_p g l\cos\theta$, structurally no $x$-dependence |
| rail-limit port | positive spring and localized damping at $x=\pm1.8$ | MuJoCo's unilateral joint-limit reaction |
| generated Coriolis | autodiff of the mass network | $[-m_p l\sin\theta\,\dot\theta^2;\ 0]$ |
| damping diagonal | $2$ parameters | $\approx 0$ |
| actuator port | sparse scalar scattered to slider DOF 0 | scale $10\,/\,c^*$ |
| contact port | **off** | — |

Gauge freedoms are as in the general note: one global scale $c^*$ shared by $M, V, D, G_a$ (the flow is invariant under joint rescaling), and the potential offset. Audit-reading specifics for this system:

- **The $x$-invariance of $M$ and the base potential is exact.** The mechanical networks never receive cart position; only the separate rail energy does. The audit's first-coordinate mass-gradient probe must therefore be exactly zero. A nonzero value now diagnoses wiring/checkpoint mismatch rather than something the optimizer might learn away.
- **The hinge representation is periodic.** Raw $\theta$ remains in the state so $\dot q$ kinematics are exact, while the learned mechanical objects see $(\sin\theta,\cos\theta)$ and cannot disagree between angles separated by $2\pi$.
- **The damping prior is near zero.** The base damping starts near the plant value and is regularized in raw-log space; rail damping remains a distinct positive parameter. Relative damping errors still need the absolute magnitude because the true coefficients are nearly zero.
- **The rail is audited separately.** Gravity/base-potential recovery excludes the explicit rail energy, while the learned rail force is compared with MuJoCo's generalized constraint force.
- **Success looks like:** mass entry correlation and coupling shape near $1$, potential $R^2$ near $1$ with the $\cos\theta$ shape, Coriolis correlation near $1$ (a single clean term, no contact to contaminate it), actuator cosine near $1$ — and the same model, rolled open-loop through the flow integrator, tracking held-out windows.

## 5. What cartpole exercises, and what it cannot

**Exercised:** the SPD Cholesky mass with a configuration-dependent coupling; the periodic, translation-invariant base potential; generated Coriolis; sparse underactuated $G_a$; a known-location unilateral limit port; finite-duration flow integration; the quadrature target; and the full four-learner CT-SAC loop with a learned drift.

**Deliberately absent:** unobserved cyclic coordinates and learned contact geometry. The rail port does not undermine the validation design: its location and force direction are known, while only positive stiffness/damping are learned. If a learned structured model in the loop matches the oracle drift here, the machinery of fitting-while-controlling is sound and cheetah's remaining gap is attributable to contact/coverage; if it falls short here, the coupling of fit cadence, warmup, replay, and target is broken somewhere cheap to debug.

The comparison that decides this is the three-way defined in the progress note — model-free baseline, oracle drift, and learned structured drift, identical target construction and replay settings, raw-state observations, a buffer large enough that nothing evicts. The physics itself was verified directly before committing runs to it: exact agreement of the observation with the simulator state, passive energy conservation to $0.002\%$ over six seconds, actuator work matching the mechanical-energy gain to $0.5\%$ through a scripted swing-up (an energy-pumping controller plus an LQR catch synthesized from the finite-differenced true linearization — swing-up in $2.8$ s), and the rail-limit dissipation noted in §1, found precisely because the work–energy ledger refused to balance until the controller stayed off the rails.

## 6. Guarded online coupling

The first 12-seed learned-model attempt used the historical coarse-transition objective and failed loudly in 10 seeds with a non-finite model-based target. The surviving model predicted reference transitions reasonably but recovered compensating, physically inconsistent terms: forbidden cart-position dependence in $M$, rough potential gradients, excessive damping, and the wrong actuator magnitude. Three changes now prevent that local cancellation from being published directly into the critic:

1. **Flow-consistent, duration-balanced fitting.** Every 2–30 ms replay interval is fully integrated at 2 ms internal resolution. Endpoint losses are weighted by $(2\text{ ms}/\Delta t)^2$, so the sparse 30 ms tail no longer supplies roughly 95% of the drift-error leverage. The configured $H=4$ rollout begins as $H=1$ for 5,000 dynamics updates, avoiding an immediate worst-case 60-step BPTT graph.
2. **Accepted target dynamics.** The optimizer mutates a live model, but the critic reads a frozen copy. A candidate is published only after finite recorded-duration and nominal-duration flows on an independent replay batch and a flow error below the configured threshold. The first accepted model is copied exactly; later accepted versions enter by EMA. A non-finite candidate is rolled back to the last accepted model and its Adam state is cleared.
3. **Staged takeover and diagnostics.** The cartpole run requires 10,000 accepted publications and a ready scalar value head before learned quadrature replaces the model-free warmup target. Failures identify the first non-finite internal drift/state step or value component; logs include flow error, publication/rejection/rollback counts, gradient norm, mass eigenvalue/condition diagnostics, rolled-state magnitude, and value-increment magnitude.

The critic never silently falls back after takeover: rejected live candidates leave the last accepted learned target in place, so the experiment remains purely learned-model-based.

## 7. Reward, for reading the returns

The swingup reward is a per-step product of four shaped factors — upright $=(\cos\theta+1)/2$, cart-centered ($\ge 0.5$), small control ($\ge 0.8$), small pole velocity ($\ge 0.5$) — with maximum $1$ per step, no bonus for holding, and no integrated energy cost. Return therefore measures, essentially, *time spent balanced and centered within the episode*: swinging up faster earns more because every step still swinging forfeits $\approx 1$, and effort enters only through the mild instantaneous small-control factor. A balanced, centered, quiet pole scores exactly $1.0$ per step.
