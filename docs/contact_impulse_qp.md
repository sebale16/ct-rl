---
title: Contact Impulse QP, from First Principles
tags: [ct-rl, contact, convex-optimization, quadratic-programming, model-based]
robots: noindex
---

# The Contact-Impulse Quadratic Program, from First Principles

:::info
**Purpose.** This note derives the contact optimization used by the structured dynamics model from one mechanical principle. A contact reaction does no work along any motion the contact permits, so the outgoing motion is the kinetic-metric projection of the freely predicted motion onto the admissible set — a minimum. The note builds that projection into the contact quadratic program the model solves, then adds the friction cone, the gated compliant form, and the fixed-budget solver.

The short version: **the outgoing motion is the admissible motion closest to the free motion in the kinetic-energy metric.** The impulse is the multiplier of non-penetration, the friction cone is the admissible-impulse set, and a requested outgoing velocity encodes inelastic impact, restitution, and penetration recovery. The solution's location identifies zero impulse, sticking, or friction saturation; a saturated residual distinguishes incipient slip from sliding. A fixed-budget ADMM solves it, and a small latent-impulse penalty conditions the solve.

This is a companion to [§2.7 and Appendix B.3 of the structured dynamics note](structured_dynamics_model.md#27-explicit-contact-port-k-learned-point-contacts). The implementation is `_constraint_contact_solve` in [`models/port_hamiltonian.py`](../models/port_hamiltonian.py), with focused tests in [`tests/test_model_based_generator.py`](../tests/test_model_based_generator.py).
:::

[TOC]

Sections 1–5 build the ideal contact problem from the least-constraint principle: the principle (§1), the impulse and the contact map (§2), the projection written as a quadratic program (§3), the friction cone (§4), and the complete ideal problem (§5). Section 6 reads off its optimality conditions, §7 gives the gated compliant form, §8 the solver, and §§9–11 its use, with §12 a summary. Worked numerical examples are collected in Appendix A and a minimal optimization vocabulary in Appendix B.

---

## 1. Contact as least constraint

At each internal step the solver has a predicted motion — the velocity the smooth mechanics and the current action produce with no ground contact — and must choose the ground impulse that corrects it:

```text
current state and action
          |
          v
predict motion from smooth mechanics and action
          |
          v
choose a unilateral, friction-bounded contact impulse
          |
          v
obtain the outgoing velocity
```

One mechanical postulate fixes that choice and shows why the answer is the solution of a minimization.

**The postulate.** A frictionless contact reaction does no work along any motion the contact permits. The ground stops a foot from descending and can push it back, with a force that neither adds energy to nor removes energy from motions that slide along the constraint. This is the statement that the contact is an ideal, passive constraint.

**The consequence is a projection.** A workless reaction changes the motion only in the directions the constraint blocks and leaves every permitted direction untouched. So the realized motion stays as close as possible to the freely predicted motion while respecting the constraint, measured in the kinetic-energy metric — the metric in which "distance" is the kinetic energy of the difference. Writing $\dot q_{\mathrm{free}}$ for the freely predicted generalized velocity, $\dot q^+$ for the realized one, and $M$ for the mass matrix,

$$
\boxed{
\dot q^+=\underset{\dot q\in\mathcal A}{\arg\min}\ \tfrac12(\dot q-\dot q_{\mathrm{free}})^\top M(\dot q-\dot q_{\mathrm{free}}),
}
$$

where $\mathcal A$ is the set of admissible outgoing motions. The outgoing motion is the $M$-metric projection of the free motion onto $\mathcal A$. This is the **principle of least constraint**, and the entire contact solver is one way of computing this projection.

**What admissibility means.** The admissible set collects the conditions a real contact imposes:

- **non-penetration** — a closing point may not keep moving into the ground, so its outgoing normal velocity is bounded below by a requested value $v_n^*$ (zero for a plain inelastic stop, positive for rebound or penetration recovery);
- **a one-sided reaction** — the ground pushes and never pulls, so the normal impulse is nonnegative;
- **a friction bound** — the tangential impulse a contact can carry is limited by the normal impulse through the Coulomb coefficient.

The first condition defines $\mathcal A$ in velocity space; the second and third describe the impulses allowed to enforce it. The rest of the note turns the boxed projection into a quadratic program: §2 introduces the impulse and the map from impulse to contact velocity, §3 writes the projection as that program, §4 gives the friction cone, and §5 states the complete ideal problem. Sections 6–8 add the optimality conditions, the gated compliant form, and the solver.

---

## 2. The impulse and the contact map

Computing the projection of §1 needs two things: the variable the contact chooses, and the linear map from that variable to the outgoing contact velocity. Let the robot have $n$ generalized velocities and $K$ learned planar contact points, each contributing a normal and a tangential coordinate.

| notation | meaning |
|---|---|
| superscript $+$ | outgoing or post-impulse quantity |
| subscript $\mathrm{free}$ | smooth-mechanics prediction before the learned ground impulse |
| subscripts $n,t$ | normal and tangential contact components |
| $[x]_+=\max(x,0)$ | positive part of a scalar |
| $\dot q\ (n)$ | generalized velocity |
| $M\ (n\times n)$ | mass matrix |
| $J\ (2K\times n)$ | contact Jacobian |
| $\Lambda\ (2K)$ | physical contact impulse |
| $v\ (2K)$ | contact-point velocity |
| $W\ (2K\times 2K)$ | contact-space inverse mass, or Delassus matrix |
| $h$ | contact response horizon |

The impulse is ordered interleaved by contact:

$$
\Lambda=
(\Lambda_{n,1},\Lambda_{t,1},\ldots,
 \Lambda_{n,K},\Lambda_{t,K})^\top.
$$

### 2.1 Impulse as the contact variable

An impact acts over a very short time and can contain a very large force. Its finite mechanical effect is the **impulse**,

$$
\Lambda=\int_{t}^{t+h} f_c(s)\,ds.
$$

Impulse has units of force times time, such as $\mathrm{N\,s}$. It directly changes momentum. Once the impulse is known, the model can expose an equivalent average force over the step by dividing by $h$.

### 2.2 The contact Jacobian

For contact point $i$, write its contact coordinates in normal–tangential order as

$$
r_i^c(q)=
\begin{bmatrix}
g_i(q)\\
p_{i,x}(q)
\end{bmatrix},
$$

where $p_i(q)=(p_{i,x},p_{i,z})$ is the point's position in the plane of motion and

$$
g_i(q)=p_{i,z}(q)-\varrho_i
$$

is its signed distance to a ground plane at $z=0$, with $\varrho_i$ the radius of the capsule the point belongs to. The signed-gap convention is

$$
g_i>0\ \text{above the ground},
\qquad
g_i=0\ \text{at the ground},
\qquad
g_i<0\ \text{in penetration}.
$$

Both coordinates come from the kinematic tree in closed form. Each contact point is reached from the root through an ordered chain of hinge angles $a_0,a_1,\ldots,a_m$ with a link offset $d_k=(d_{k,x},d_{k,z})$ per joint, all hinges sharing one axis, so rotations compose by addition:

$$
\Theta_{i,k}=\sum_{j\le k}q_{a_j},
$$

$$
p_{i,z}=z_0+q_z+\sum_k\left(-d_{k,x}\sin\Theta_{i,k}+d_{k,z}\cos\Theta_{i,k}\right),
$$

$$
p_{i,x}=q_x+\sum_k\left(d_{k,x}\cos\Theta_{i,k}+d_{k,z}\sin\Theta_{i,k}\right),
$$

where $z_0$ is the root frame's height offset and $q_x,q_z$ are the root translations.

Differentiating a rotation about a shared axis gives the classical revolute-joint column, a cross product of the axis with the lever arm from that joint's pivot to the point:

$$
\frac{\partial p_i}{\partial q_{a_j}}
=\hat\omega\times(p_i-p_j)
=\begin{bmatrix}
+(p_{i,z}-p_{j,z})\\
-(p_{i,x}-p_{j,x})
\end{bmatrix},
$$

together with the two root columns

$$
\frac{\partial p_{i,x}}{\partial q_x}=1,
\qquad
\frac{\partial p_{i,z}}{\partial q_z}=1.
$$

The lever arm $p_i-p_j$ is the sum of every link offset from joint $a_j$ outward, so a single pair of cumulative sums along the chain — forward over the angles, backward over the placed offsets — supplies the whole point Jacobian at once.

Collecting the two components gives the point Jacobian $J_{v,i}$, whose rows are the contact-frame directions for a ground plane of constant normal:

$$
J_{n,i}=\hat z^\top J_{v,i}=\frac{\partial g_i}{\partial q},
\qquad
J_{t,i}=\hat x^\top J_{v,i}.
$$

Two entries follow directly and fix the coupling to the root degrees of freedom: $J_{n,i}$ has a unit entry in the root-height column and vanishes in the root-horizontal column, since a point's height is independent of horizontal translation; $J_{t,i}$ has a unit entry in the root-horizontal column, since translating the root translates every point with it.

Stacking those rows over all contacts gives $J$, which maps generalized velocity to contact velocity:

$$
v=J\dot q.
$$

The transpose maps a contact-space impulse back to generalized momentum:

$$
\Delta p=J^\top\Lambda.
$$

This transpose follows from virtual work:

$$
\Lambda^\top(J\,\delta q)
=(J^\top\Lambda)^\top\delta q.
$$

It is the virtual-work dual, or pullback, of $J$.

### 2.3 First predict the contact-free motion

The smooth mechanical model and current action give

$$
\ddot q_{\mathrm{free}}
=M^{-1}
\left(G_a a-\nabla V-C(q,\dot q)\dot q-D\dot q\right).
$$

| term | mechanical role |
|---|---|
| $G_a a$ | maps the current action into generalized actuator force |
| $-\nabla V$ | conservative force, including gravity |
| $-C(q,\dot q)\dot q$ | Coriolis and centrifugal force |
| $-D\dot q$ | passive joint damping |
| $M^{-1}$ | converts the total generalized force to acceleration |

In the implementation this free acceleration also includes any configured joint-limit spring and damping forces. “Free” labels the stage immediately before the learned ground-contact impulse.

Predict the velocity at the end of the smooth-mechanics stage:

$$
\dot q_{\mathrm{free}}
=\dot q+h\ddot q_{\mathrm{free}},
\qquad
v_{\mathrm{free}}=J\dot q_{\mathrm{free}}.
$$

The action is already inside $v_{\mathrm{free}}$. When the current tangential velocity is zero and the action predicts impending slip, $v_{\mathrm{free},t}$ becomes nonzero. An enabled contact with sufficient Coulomb budget then selects a static-friction impulse; cone saturation gives sliding, and the clear-flight gate gives zero impulse.

### 2.4 From impulse to outgoing contact velocity

Holding $q$, $M$, and $J$ fixed over the short impulse response,

$$
M(\dot q^+-\dot q_{\mathrm{free}})=J^\top\Lambda.
$$

Therefore

$$
\dot q^+
=\dot q_{\mathrm{free}}+M^{-1}J^\top\Lambda,
$$

and the outgoing contact velocity is

$$
\boxed{
v^+=v_{\mathrm{free}}+W\Lambda,
\qquad
W=JM^{-1}J^\top.
}
$$

The diagonal entries of $W$ behave like inverse effective masses. Its off-diagonal entries describe coupling: an impulse at one contact can change another contact's velocity.

The matrix $W$ is positive-semidefinite because

$$
z^\top Wz
=(J^\top z)^\top M^{-1}(J^\top z)
\ge0.
$$

This is the main reason the contact objective built next is convex.

### 2.5 The two adjoints, and $W$ as their composition

Separating the spaces that $J$ and $J^\top$ act on explains what $W$ is.

Write $J:\mathcal V\to\mathcal U$, from the generalized-velocity space $\mathcal V$ (dimension $n$; $J$ acts on velocities $\dot q$) to the contact-velocity space $\mathcal U$ (dimension $2K$). The relation $v=J\dot q$ carries a generalized velocity to a contact velocity.

The transpose $J^\top$ of §2.2 is an **adjoint** of $J$, and two notions of adjoint are worth keeping apart.

| adjoint | maps | needs a metric? | acts on | returns |
|---|---|---|---|---|
| Hilbert $J^\ast$ | $\mathcal U\to\mathcal V$ | yes, on both spaces | contact velocities | generalized velocities |
| transpose $J^\top$ | $\mathcal U^\ast\to\mathcal V^\ast$ | no | contact forces | generalized forces |

The **Hilbert adjoint** $J^\ast:\mathcal U\to\mathcal V$ exists once both spaces carry inner products, fixed by $\langle Jx,y\rangle_{\mathcal U}=\langle x,J^\ast y\rangle_{\mathcal V}$. It sends velocities to velocities.

The **transpose adjoint** $J^\top:\mathcal U^\ast\to\mathcal V^\ast$ needs no inner product. It is fixed by the pairing alone, $(J^\top\phi)(x)=\phi(Jx)$, and sends a linear functional on contact velocities to one on generalized velocities. The dual spaces $\mathcal U^\ast$ and $\mathcal V^\ast$ are the **force** spaces: a contact force $\Lambda\in\mathcal U^\ast$ pairs with a contact velocity to give power, and a generalized force or momentum $\Delta p\in\mathcal V^\ast$ pairs with a generalized velocity to give power.

The $J^\top$ in $\Delta p=J^\top\Lambda$ (§2.2) is the transpose adjoint: $\Lambda$ is a contact force and $\Delta p$ a generalized momentum, so it operates on the dual of the contact space and lands in the dual of the generalized space. It is this adjoint rather than the Hilbert one because the pairing mechanics supplies is power — force times velocity — the duality pairing between a space and its dual, not an inner product on velocities. Forces are the duals of velocities, so the transpose of a velocity map is a force map.

Given inner products $G_{\mathcal V}$ on $\mathcal V$ and $G_{\mathcal U}$ on $\mathcal U$, the two adjoints coincide up to those metrics, $J^\ast=G_{\mathcal V}^{-1}J^\top G_{\mathcal U}$; in orthonormal coordinates their matrices are equal, which is why one symbol serves both.

**$W$ is the composition of these maps.** Read $W=JM^{-1}J^\top$ right to left as a chain across the four spaces:

$$
\underbrace{\Lambda}_{\mathcal U^\ast}
\ \xrightarrow{\ J^\top\ }\
\underbrace{J^\top\Lambda}_{\mathcal V^\ast}
\ \xrightarrow{\ M^{-1}\ }\
\underbrace{M^{-1}J^\top\Lambda}_{\mathcal V}
\ \xrightarrow{\ J\ }\
\underbrace{W\Lambda}_{\mathcal U}.
$$

A contact force $\Lambda$ becomes a generalized force $J^\top\Lambda$ (transpose adjoint), then the generalized velocity $M^{-1}J^\top\Lambda$ that force produces, then the contact velocity $W\Lambda$. So $W:\mathcal U^\ast\to\mathcal U$ maps a contact **force** to a contact **velocity** — the contact-space inverse mass, or Delassus matrix.

The one place a metric enters the chain is $M^{-1}$. The mass matrix $M:\mathcal V\to\mathcal V^\ast$ is the inner product on generalized velocities, since the kinetic energy is $\tfrac12\dot q^\top M\dot q$; it turns a generalized velocity into its momentum, and $M^{-1}$ turns a generalized force back into the velocity it produces. The maps $J$ and $J^\top$ on either side are the metric-free primal and dual maps. This is why §3.1 substitutes $\Delta\dot q=M^{-1}J^\top\Lambda$ and the kinetic deviation collapses to $\tfrac12\Lambda^\top W\Lambda$: that quadratic form is the kinetic energy of the impulse response.

$W$ then carries a metric role of its own. As the quadratic form $\Lambda^\top W\Lambda$ it is an inner product on the contact-force space $\mathcal U^\ast$, and its inverse $W^{-1}$ is the kinetic metric on the contact-velocity space $\mathcal U$ — the metric in which §5.4 measures the projection $\tfrac12\lVert v^+-v^*\rVert_{W^{-1}}^2$.

---

## 3. The projection as a quadratic program

### 3.1 From least constraint to the quadratic program

Section 1 places the outgoing motion at the $M$-metric-nearest admissible point to the free motion. Two facts turn that projection into a program in the impulse.

**The reaction is $M$-orthogonal to admissible motions.** The contact acts through the impulse $\Lambda$ as the generalized reaction $J^\top\Lambda$ (§2.2), so the realized velocity change is $\Delta\dot q=\dot q^+-\dot q_{\mathrm{free}}=M^{-1}J^\top\Lambda$. For any admissible velocity variation $\delta\dot q$, the reaction's work is $\delta\dot q^\top J^\top\Lambda=(J\delta\dot q)^\top\Lambda$, which a workless constraint holds at zero. In the kinetic metric,

$$
\langle\Delta\dot q,\ \delta\dot q\rangle_M
=\delta\dot q^\top M\,\Delta\dot q
=\delta\dot q^\top J^\top\Lambda
=(J\delta\dot q)^\top\Lambda
=0 .
$$

So the reaction moves the motion only across the admissible set, and $\dot q^+$ is the projection §1 describes.

**The projection, in the impulse, is a quadratic.** Substituting $\Delta\dot q=M^{-1}J^\top\Lambda$ into the projection objective collapses it to a quadratic in $\Lambda$:

$$
\tfrac12(\dot q^+-\dot q_{\mathrm{free}})^\top M(\dot q^+-\dot q_{\mathrm{free}})
=\tfrac12\Lambda^\top JM^{-1}J^\top\Lambda
=\tfrac12\Lambda^\top W\Lambda,
\qquad W=JM^{-1}J^\top .
$$

The kinetic deviation of the outgoing motion from the free motion is exactly $\tfrac12\Lambda^\top W\Lambda$. The projection is then

$$
\min_{\Lambda}\ \tfrac12\Lambda^\top W\Lambda
\qquad\text{subject to}\qquad
v_{\mathrm{free}}+W\Lambda\ \ge\ v^* ,
$$

the admissibility being the non-penetration bound $v^+\ge v^*$ on the outgoing normal velocity (§1).

**The impulse is the multiplier of non-penetration.** Attach a multiplier $s\ge0$ to the constraint $v^+-v^*\ge0$ and set the derivative in $\Lambda$ to zero: this gives $W\Lambda=Ws$, hence $\Lambda=s\ge0$. The impulse *is* the constraint's multiplier, and its one-sidedness $\Lambda\ge0$ is produced by the projection rather than imposed. Complementarity $s^\top(v^+-v^*)=0$ then reads $\Lambda^\top(v^+-v^*)=0$. These are the contact laws, recovered formally in §6.

**Writing the admissible impulses as the feasible set.** The solver keeps the impulse as the variable and the admissible-impulse set as the explicit feasible region, moving the velocity condition into the objective. Complete the square around the impulse that would attain $v^*$ exactly, $\Lambda_{\mathrm{req}}=W^{-1}(v^*-v_{\mathrm{free}})$:

$$
\tfrac12(\Lambda-\Lambda_{\mathrm{req}})^\top W(\Lambda-\Lambda_{\mathrm{req}})
=\tfrac12\Lambda^\top W\Lambda+b^\top\Lambda+\text{const},
\qquad
b=v_{\mathrm{free}}-v^* ,
$$

since $-\Lambda_{\mathrm{req}}^\top W=b^\top$. The objective becomes the $W$-metric deviation of the impulse from the target-achieving impulse, equivalently $\tfrac12\lVert v^+-v^*\rVert_{W^{-1}}^2$: the same projection, now measuring distance from the requested outgoing velocity. Over the admissible impulses this is

$$
\boxed{
\min_{\Lambda\ \text{admissible}}\ \tfrac12\Lambda^\top W\Lambda+b^\top\Lambda,
\qquad b=v_{\mathrm{free}}-v^* .
}
$$

This is the contact quadratic program. Its two forms — least kinetic deviation from the free motion subject to the velocity bound, and least deviation from the target over the admissible impulses — are the one projection viewed from the two reference velocities, with the impulse and the velocity condition exchanging the roles of explicit constraint and emergent multiplier.

This QP is an **inner solve**, run afresh for every state and action; fitting the model is the **outer optimization**, and the differentiable solve carries the outer gradient through the selected impulse. Section 4 makes “admissible” the friction cone, §5 states the complete problem, and §6 reads off the contact laws as its optimality conditions.

### 3.2 The requested outgoing velocity

The **requested outgoing velocity** $v^*$ is the outgoing contact velocity that an active contact response is asked to produce; exact attainment gives $v^+=v^*$. Its normal components combine inelastic contact, restitution, and penetration recovery. Its tangential components equal zero, requesting sticking. The request enters the QP through the linear term $b=v_{\mathrm{free}}-v^*$.

The normal request is built from the penetration depth. For each contact define

$$
d_i
=\min\!\left(\frac{[-g_i]_+}{h},v_{\max}\right),
$$

where $[-g_i]_+=\max(-g_i,0)$ measures penetration. Before the cap, $[-g_i]_+/h$ is the outward speed that would traverse the entire penetration depth in one response horizon. The factor $\beta_i$ scales that correction. The requested outgoing normal velocity is

$$
\boxed{
v_{n,i}^*
=\beta_i d_i+e_i[-v_{n,i}]_+,
\qquad
v_{n,i}=J_{n,i}\dot q.
}
$$

Each term in this construction is nonnegative, so

$$
0\le v_{n,i}^*
\le \beta_i v_{\max}+e_i[-v_{n,i}]_+.
$$

With upward velocity positive, $v_{n,i}^*=0$ requests rest in the normal direction and $v_{n,i}^*>0$ requests outward motion. The current normal velocity $v_{n,i}$ sets the restitution request. The free velocity $v_{\mathrm{free},i}$ includes smooth evolution and the current action across the response horizon, making friction and normal response aware of impending action-induced motion.

| term | requested behavior |
|---|---|
| $e_i[-v_{n,i}]_+$ | rebound speed equal to $e_i$ times the current inward speed |
| $\beta_i d_i$ | fractional outward motion for penetration recovery |
| $v_{t,i}^*=0$ | sticking, if the friction budget permits it |

The cap $v_{\max}$ keeps the requested penetration-recovery speed in the bounded range $0\le d_i\le v_{\max}$ across gap values, and $e_i,\beta_i\in(0,0.5)$. Section 4 introduces the learned friction coefficient $\mu_i$, and §7.2 the learned compliance $c_{0,i}$.

Stack these targets in the same interleaved ordering as the contact velocity:

$$
v^*=(v_{n,1}^*,0,\ldots,v_{n,K}^*,0)^\top.
$$

Componentwise, the linear term reads

$$
b_{n,i}
=v_{\mathrm{free},n,i}
-\beta_i d_i
+e_i\min(v_{n,i},0),
\qquad
b_{t,i}=v_{\mathrm{free},t,i}.
$$

The selected impulse produces $v^+=v_{\mathrm{free}}+W\Lambda$ and the residual $r=v^+-v^*$. Exact attainment gives $r=0$. Separation and sliding produce the mode-specific residuals developed in §5 and §6. Whether an admissible impulse can attain the request depends on the friction budget; §4 defines the feasible set and §4.1 states the exact-attainment test.

### 3.3 The objective as a shifted energy ledger

The projection objective has a direct energy reading. The kinetic-energy change produced by the impulse (§10 derives this discrete ledger) is

$$
\Delta T_c
=\Lambda^\top v_{\mathrm{free}}
+\frac12\Lambda^\top W\Lambda.
$$

Since $b=v_{\mathrm{free}}-v^*$, the objective decomposes as

$$
\frac12\Lambda^\top W\Lambda+b^\top\Lambda
=\Delta T_c-(v^*)^\top\Lambda.
$$

With $v^*=0$ the QP minimizes the contact-induced kinetic-energy change: among the admissible impulses, it removes the most kinetic energy. This is the **maximum-dissipation** reading of the least-constraint projection — the closest admissible outgoing velocity to rest is the one of least outgoing kinetic energy. Restitution and recovery shift that baseline through $-(v^*)^\top\Lambda$, which credits impulse spent producing the requested outward velocity. The restitution coefficient is a speed ratio: in the simple scalar impact, rebound kinetic energy is an $e_i^2$ fraction of incoming kinetic energy. The discrete ledger in §10 evaluates the energy sign for the actual pair of velocity baselines, and penetration recovery requests outward motion and permits positive contact work.

For a scalar frictionless contact these pieces are transparent: with effective mass $m_{\mathrm{eff}}$ and $w=1/m_{\mathrm{eff}}$, the objective is $\tfrac12 w\lambda^2+(v_{\mathrm{free}}-v^*)\lambda$, worked in [Appendix A.1](#a1-one-frictionless-contact-one-number) and [A.2](#a2-scalar-target-three-examples).

---

## 4. The Coulomb feasible set

For contact $i$, the ground impulse obeys

$$
\Lambda_{n,i}\ge0,
\qquad
|\Lambda_{t,i}|\le\mu_i\Lambda_{n,i},
$$

where $\mu_i$ is the dimensionless Coulomb friction coefficient: it sets the tangential impulse budget associated with a normal impulse. At $\Lambda_{n,i}=0$, the second inequality gives $\Lambda_{t,i}=0$.

The implementation learns one $\mu_i$ per contact through $\mu_i=2\,\operatorname{sigmoid}(\hat\mu_i)$, where $\hat\mu_i$ is an unconstrained raw parameter, giving $\mu_i\in(0,2)$ and an initialization near $0.8$. Finite $\hat\mu_i$ approach the ideal frictionless value $\mu_i=0$ as their lower limit.

This feasible set is called the planar Coulomb cone:

$$
\mathcal C_{\mu_i}
=\left\{(\Lambda_n,\Lambda_t):
\Lambda_n\ge0,
|\Lambda_t|\le\mu_i\Lambda_n
\right\}.
$$

It is a wedge in two dimensions:

![coulomb_cone](https://hackmd.io/_uploads/SyH55RjEzl.svg)

It is called a cone because multiplying a feasible impulse by any nonnegative number keeps it feasible. Increasing $\mu$ widens the cone and enlarges the available tangential impulse interval.

The cone sets the available friction magnitude. The objective selects the actual tangential impulse and its sign from the interval $[-\mu\Lambda_n,+\mu\Lambda_n]$. An interior optimum corresponds to sticking. A sloped-boundary optimum represents friction saturation, with its velocity residual distinguishing incipient slip from sliding.

In this planar model the absolute-value constraint is equivalent to two linear inequalities:

$$
\Lambda_t-\mu\Lambda_n\le0,
\qquad
-\Lambda_t-\mu\Lambda_n\le0.
$$

Thus the planar friction problem is a quadratic program with linear inequalities.

For $K$ contacts, $\mathcal C_\mu$ collects the individual cones: every impulse pair $(\Lambda_{n,i},\Lambda_{t,i})$ obeys its own coefficient $\mu_i$.

### 4.1 Reachable velocities and exact attainment

The request $v^*$ lives in velocity space, and the cone $\mathcal C_\mu$ contains the admissible impulses. Three checks organize the request and response:

| property | mathematical statement | meaning |
|---|---|---|
| physically meaningful request | $v_n^*\ge0$, $v_t^*=0$ | the normal request is stationary or outward, and the tangential request is sticking |
| feasible impulse set | $0\in\mathcal C_\mu$ | zero impulse supplies a feasible candidate for every state |
| exact attainment | $\exists\Lambda\in\mathcal C_\mu:\ v_{\mathrm{free}}+W\Lambda=v^*$ | an admissible impulse realizes every requested velocity component |

The first row checks the physical meaning of the constructed request. The second confirms that the cone contains an allowed impulse. The third checks exact dynamic reachability.

The set of reachable outgoing velocities is

$$
\boxed{
\mathcal A(v_{\mathrm{free}})
=v_{\mathrm{free}}+W\mathcal C_\mu
=\left\{v_{\mathrm{free}}+W\Lambda:
\Lambda\in\mathcal C_\mu\right\}.
}
$$

Read this expression as a recipe: take every admissible impulse $\Lambda$ from the cone, compute its velocity change $W\Lambda$, and add that change to $v_{\mathrm{free}}$. The resulting collection is $\mathcal A(v_{\mathrm{free}})$.

Exact attainment means

$$
\boxed{
v^*\in\mathcal A(v_{\mathrm{free}})
\quad\Longleftrightarrow\quad
\exists\Lambda\in\mathcal C_\mu:
W\Lambda=v^*-v_{\mathrm{free}}.
}
$$

The symbol $\in$ means “belongs to,” and $\exists$ means “there is at least one.” Thus the boxed statement says that $v^*$ is reachable exactly when at least one admissible impulse produces the required velocity change.

This condition depends on the request, the free prediction, the contact-space inverse mass $W$, the friction coefficients $\mu_i$, and contact coupling. For invertible $W$, form the required impulse

$$
\Lambda_{\mathrm{req}}
=W^{-1}(v^*-v_{\mathrm{free}}).
$$

Membership $\Lambda_{\mathrm{req}}\in\mathcal C_\mu$ gives exact attainment. The existence equation above supplies the corresponding test for a rank-deficient $W$.

The equations above describe the ideal QP. The target problem in §7 has a unique minimizer; at a contact whose impulse lies in the strict cone interior, its component relation is $v_i^+-v_i^*=-\tilde R_{ii}\Lambda_i$, the compliance of §7.2 acting on that contact's impulse. A clear-flight gate value $s_i=0$ sets $\Lambda_i=0$ and makes that contact's $v_i^*$ inactive bookkeeping. Section 8 explains the residual from the finite numerical solve.

### 4.2 How $\mu$ controls exact sticking

The tangential request $v_t^*=0$ asks for sticking, and whether the cone can supply it is a ratio test. For a candidate exact-attainment impulse with $\Lambda_{n,\mathrm{req}}>0$, define the required friction ratio

$$
\boxed{
\mu_{\mathrm{required}}
=\frac{|\Lambda_{t,\mathrm{req}}|}
{\Lambda_{n,\mathrm{req}}}.
}
$$

Comparing it to the available $\mu$ names the outgoing regime:

| relation | cone location | outgoing regime |
|---|---|---|
| $\mu>\mu_{\mathrm{required}}$ | strict interior | exact sticking |
| $\mu=\mu_{\mathrm{required}}$ | sloped boundary with $r=0$ | exact incipient slip |
| $\mu<\mu_{\mathrm{required}}$ | saturated optimum with tangential residual | sliding |

For a single contact with $W=I$ the sticking request has $\Lambda_{\mathrm{req}}=v^*-v_{\mathrm{free}}$, so the ratio reads directly from velocity,

$$
\mu_{\mathrm{required}}
=\frac{|v_{\mathrm{free},t}|}
{v_n^*-v_{\mathrm{free},n}},
\qquad
v_n^*-v_{\mathrm{free},n}>0,
$$

worked with numbers in [Appendix A.3](#a3-exact-sticking-and-the-friction-ratio). For a general invertible $W$, first compute $\Lambda_{\mathrm{req}}=W^{-1}(v^*-v_{\mathrm{free}})$ and check $\Lambda_{n,i,\mathrm{req}}\ge0$ together with $|\Lambda_{t,i,\mathrm{req}}|\le\mu_i\Lambda_{n,i,\mathrm{req}}$ for every contact. For coupled or rank-deficient $W$, the existence condition in §4.1 gives the complete exact-attainment test.

---

## 5. The ideal contact QP

Collecting the objective of §3 and the cone of §4, the complete ideal problem is

$$
\boxed{
\begin{aligned}
\underset{\Lambda\in\mathbb R^{2K}}{\operatorname{minimize}}
&\quad
\frac12\Lambda^\top W\Lambda+b^\top\Lambda\\
\operatorname{subject\ to}
&\quad
\Lambda_{n,i}\ge0,
\qquad
|\Lambda_{t,i}|\le\mu_i\Lambda_{n,i},
\quad i=1,\ldots,K.
\end{aligned}
}
$$

Every part carries a concrete meaning:

| part | role |
|---|---|
| decision variable $\Lambda$ | physical normal and tangential impulses |
| $W=JM^{-1}J^\top$ | how those impulses change contact velocity |
| $b=v_{\mathrm{free}}-v^*$ | free motion measured relative to the requested outgoing velocity |
| cone constraint | unilateral normal reaction and finite Coulomb friction |

The objective is the least-constraint projection of §3.1, and the cone is the admissible-impulse set of §4. The remainder of this section reads its optimality residual, its selected modes, its multi-contact coupling, and its geometry.

### 5.1 Optimality and the velocity residual

The gradient of the objective is

$$
r
=W\Lambda+b
=v^+-v^*.
$$

So $r$ is exactly the outgoing contact-velocity residual. Unconstrained stationarity gives $v^+=v^*$ whenever $W\Lambda=-b$ has a solution. The cone selects from admissible impulses. A rank-deficient $W$ makes the mechanism's contact-space range the set of attainable velocity changes.

### 5.2 Contact modes selected by the QP

| solution location | interpretation | velocity result in the ideal problem |
|---|---|---|
| cone apex, $\Lambda_i=0$ | contact $i$ is inactive or separating | contact $i$ contributes zero impulse; coupled impulses determine $v_i^+$ through $\sum_jW_{ij}\Lambda_j$ |
| strict cone interior | sticking | $v_{n,i}^+=v_{n,i}^*$ and $v_{t,i}^+=0$ |
| sloped cone boundary with nonzero residual | sliding | friction is saturated; residual slip remains |

A boundary impulse with $r=0$ is the limiting or incipient-slip case: friction has reached its bound and the outgoing tangential velocity remains zero.

The location of the optimum directly supplies zero impulse, sticking, or friction saturation. At saturation, the tangential residual distinguishes incipient slip from sliding. A concrete sliding computation is worked in [Appendix A.4](#a4-a-sliding-contact); it exhibits the **associated joint-cone QP**, which obtains normal and tangential responses from one shared cone objective and couples the outward normal residual to the tangential residual.

### 5.3 Multiple contacts are solved together

In a robot, off-diagonal terms of $W$ couple the contacts: an impulse at one foot affects the velocity of another foot. The joint QP finds one mutually consistent set of impulses for all contacts. A diagonal $W$ is the separable special case.

Duplicate or mechanically redundant contact points can make $W$ singular, meaning several impulse distributions produce the same generalized motion. The positive regularizer in §7 adds curvature along redundant directions, giving the implemented target problem a well-posed, unique optimum.

### 5.4 The geometric picture: projection onto the cone

The whole QP reduces to one point and one region. The objective $\tfrac12\Lambda^\top W\Lambda+b^\top\Lambda$ fills impulse space with nested ellipsoidal level sets around a single lowest point, its unconstrained minimizer $-W^{-1}b$. Since $b=v_{\mathrm{free}}-v^*$, that lowest point is

$$
-W^{-1}b
=W^{-1}(v^*-v_{\mathrm{free}})
=\Lambda_{\mathrm{req}},
$$

exactly the required impulse of §4.1. The feasible region is the Coulomb cone $\mathcal C_\mu$ of §4, a wedge with apex at the origin. Solving the QP asks where the smallest objective ellipsoid centered at $\Lambda_{\mathrm{req}}$ first meets that wedge. This is the exact-attainment test of §4.1 read geometrically:

- $\Lambda_{\mathrm{req}}\in\mathcal C_\mu$: the lowest point is already feasible, so the solver returns it and $r=0$; the request is attained, with a strict interior giving sticking and a facet with $r=0$ giving incipient slip.
- $\Lambda_{\mathrm{req}}\notin\mathcal C_\mu$: the optimum moves to the boundary of the cone, and $r\ne0$.

What "boundary point" means is fixed by the objective's shape. Completing the square around $\Lambda_{\mathrm{req}}$,

$$
\frac12\Lambda^\top W\Lambda+b^\top\Lambda
=\frac12(\Lambda-\Lambda_{\mathrm{req}})^\top
 W(\Lambda-\Lambda_{\mathrm{req}})+\text{const},
$$

so minimizing over the cone minimizes the $W$-weighted distance $(\Lambda-\Lambda_{\mathrm{req}})^\top W(\Lambda-\Lambda_{\mathrm{req}})$. The constrained optimum is the point of the cone closest to $\Lambda_{\mathrm{req}}$ in the metric set by $W$: the tangency point where an objective ellipsoid kisses the wedge. Because $W=JM^{-1}J^\top$ is generally not the identity, its level sets are stretched, and this tangency point differs from the ordinary nearest point. The two coincide only when $W=I$, which is why the $W=I$ examples of Appendix A read as plain nearest-point projections.

That metric is physical. The $W$-distance is a kinetic-energy distance (§3.3), so the constrained optimum is the admissible impulse whose outgoing velocity is closest in kinetic energy to the request. This is the least-constraint projection of §1 restricted to the friction cone.

Where the tangency lands names the mode of §5.2:

| projection target | cone location | selected mode |
|---|---|---|
| ellipsoid center already feasible | strict interior | sticking, $r=0$ on active components |
| sloped facet $\Lambda_t=\pm\mu\Lambda_n$ | boundary | sliding, friction saturated |
| apex $\Lambda=0$ | vertex | separation or inactive contact |

The regularized target of §7 keeps this picture with a shifted metric: $W$ becomes $H=\operatorname{sym}(SW_{\mathrm{full}}S)+R$ and the ellipsoid center moves accordingly, so the added curvature of $R$ rounds the level sets along redundant directions and fixes a unique tangency point.

One caution about the solver. The ADMM cone projection $\Pi_{\mathcal C_\mu}$ of §8 is an ordinary Euclidean projection, but it is an inner step that only enforces feasibility at each iteration. The converged optimum is the $W$- or $H$-metric projection described here; the linear solve at each iteration reintroduces that stretched geometry.

---

## 6. Complementarity and KKT

This section gives formal names to the optimality conditions that §3.1 already derived from the projection.

### 6.1 Scalar complementarity

For one frictionless contact, define

$$
r_n=v_n^+-v_n^*.
$$

The optimum satisfies

$$
\lambda_n\ge0,
\qquad
r_n\ge0,
\qquad
\lambda_n r_n=0.
$$

The last condition means at least one of the two quantities must be zero:

- The combination $\lambda_n=0$ and $r_n>0$ describes separation.
- If $\lambda_n>0$, then $r_n=0$ and the requested normal velocity is attained.

This is written compactly as

$$
0\le\lambda_n\ \perp\ r_n\ge0.
$$

These are the Karush–Kuhn–Tucker, or **KKT**, conditions of the scalar QP. They are the projection's conditions from §3.1 with the multiplier $s$ written as $r_n$: the KKT multiplier and the residual velocity are the same quantity, $r_n=v_n^+-v_n^*$, which is why minimizing manufactures the Signorini law.

### 6.2 Cone complementarity

For the ideal frictional QP, the scalar inequalities become

$$
\Lambda\in\mathcal C_\mu,
\qquad
r\in\mathcal C_\mu^*,
\qquad
\Lambda^\top r=0,
$$

For any cone $\mathcal C$, its **dual cone** is the set of residual directions having a nonnegative dot product with every feasible impulse:

$$
\mathcal C^*
=\{r:z^\top r\ge0\ \text{for every }z\in\mathcal C\}.
$$

For one planar Coulomb cone this becomes

$$
\mathcal C_\mu^*
=\{(r_n,r_t):r_n\ge\mu|r_t|\}.
$$

The dual-cone condition gives a nonnegative directional derivative along every feasible impulse direction. Orthogonality, $\Lambda^\top r=0$, supplies the generalized complementarity condition.

For a sliding contact with $r_t\ne0$, the friction impulse lies on the cone boundary whose tangential sign opposes $r_t$. Orthogonality then gives

$$
r_n=\mu|r_t|.
$$

Since $r_n=v_n^+-v_n^*$ and $r_t=v_t^+$,

$$
v_n^+-v_n^*=\mu|v_t^+|.
$$

Thus the ideal joint-cone QP satisfies the sliding relation $v_n^+-v_n^*=\mu|v_t^+|$: the associated model couples a small outward normal velocity to tangential slip.

### 6.3 The minimizer is the unique physical state

Every admissible impulse is feasible, so the feasible set holds many impulses, and the complementarity conditions select one of them. In exact arithmetic $W$ is positive-semidefinite and the cone is convex, so the objective is convex; the regularizer of §7 makes it strictly convex, giving a single minimizer, and a convex program's KKT conditions are sufficient as well as necessary. The impulse that satisfies the contact laws and the minimizer are therefore one point, determined uniquely — the determinacy of a passive rigid contact carried by convexity.

The tangential reaction dissipates energy, and the maximum-dissipation reading of §3.3 selects it: among admissible tangential impulses the chosen one removes the most kinetic energy. Folding the normal projection and this tangential dissipation into the single cone objective gives the associated joint-cone problem, exact for the normal, inelastic, and restitution response, and extended by the cone objective to coupled Coulomb friction.

---

## 7. The gated, compliant and regularized QP

The formulation extends the ideal QP with three ingredients:

1. a location-dependent gate, so a point in clear flight is exactly inactive;
2. a gate-shaped contact compliance carrying a learned stiffness;
3. a small positive conditioning floor, so duplicate and weakly gated contacts remain numerically stable.

The following states the numerical target problem explicitly.

Throughout this section, write

$$
W_{\mathrm{full}}=JM^{-1}J^\top
$$

for the ungated matrix called $W$ in §§2–6, reserving the shorter name for its gated counterpart.

### 7.1 Compact flight gate

For thresholds $g_{\mathrm{on}}<g_{\mathrm{off}}$, define

$$
u_i
=\operatorname{clip}\!\left(
\frac{g_{\mathrm{off}}-g_i}
{g_{\mathrm{off}}-g_{\mathrm{on}}},0,1
\right),
$$

$$
s_i=6u_i^5-15u_i^4+10u_i^3.
$$

Then:

- $s_i=1$ at and below $g_{\mathrm{on}}$: fully enabled;
- $0<s_i<1$ in the transition band: smoothly relaxed candidate;
- $s_i=0$ at and above $g_{\mathrm{off}}$: exactly disabled.

The gap is a metric quantity, so the band is set in metres. The thresholds are $g_{\mathrm{on}}=0$ and a configurable $g_{\mathrm{off}}$, with a default of $5\times10^{-3}\,\mathrm{m}$ for the exact-geometry contact set.

![contact_gate](https://hackmd.io/_uploads/Hyds50oEMl.svg)


The polynomial is the quintic smoothstep, the unique degree-5 polynomial with

$$
s(0)=s'(0)=s''(0)=0,
\qquad
s(1)=1,
\quad
s'(1)=s''(1)=0.
$$

The clip supplies the two flat regions, giving the exact clear-flight cutoff; the vanishing endpoint derivatives remove the kinks where the ramp meets them, making the assembled gate $C^2$ in the gap. The cubic smoothstep $3u^2-2u^3$ is $C^1$ but leaves $s''=\pm6$ at the joins, and $s_i$ scales an impulse that enters acceleration, so that jump would reach the drift. The derivative factors as $s'(u)=30u^2(u-1)^2\ge0$, so the gate is monotone and never overshoots $[0,1]$.

Repeat the same gate for the normal and tangential component of each contact:

$$
S=\operatorname{diag}(s_1,s_1,\ldots,s_K,s_K).
$$

Introduce a latent cone impulse $y$ and define the physical impulse by

$$
\Lambda=Sy.
$$

Because normal and tangential components share the same nonnegative scale,

$$
y\in\mathcal C_\mu
\quad\Longrightarrow\quad
Sy\in\mathcal C_\mu.
$$

At $s_i=0$, the corresponding physical impulse is exactly zero.

The gate supplies smooth contact candidacy with an exact clear-flight cutoff. A point in the transition band $0<g_i<g_{\mathrm{off}}$ can transmit an attenuated impulse before its gap reaches zero.

Two properties of $s_i$ are used later. It is monotone in $u_i$ and confined to $[0,1]$, with $s_i=1$ attained exactly at $u_i=1$; and the derivative $\partial s_i/\partial g_i=-s'(u_i)/(g_{\mathrm{off}}-g_{\mathrm{on}})$ vanishes at both ends of the band, so the gate meets its flat regions smoothly. Evaluating the quintic just below $u_i=1$ rounds to within a few units in the last place of $1$, so quantities that depend on $1-s_i^2$ are formed with a nonnegative clamp and remain exactly in $[0,1]$ at any working precision.

### 7.2 Gate-shaped compliance and the conditioning floor

#### The contact-space scale

Every diagonal term added to the QP acts in contact space, where the natural unit is an inverse mass. That unit is supplied by

$$
\sigma
=\operatorname{stopgrad}\!\left[
\max\!\left(
\operatorname{mean}\operatorname{diag}(W_{\mathrm{full}}),
10^{-6}
\right)
\right],
$$

with $W_{\mathrm{full}}=JM^{-1}J^\top$ evaluated before gating. `stopgrad` uses $\sigma$ in the forward calculation and treats it as constant during differentiation.

Denominating diagonal terms in $\sigma$ makes them covariant with the mechanism: rescaling $M\mapsto\alpha M$ rescales $W_{\mathrm{full}}$ and $\sigma$ together, so the selected impulse scales as $\alpha$, exactly as a physical impulse does.

#### The diagonal

Let $c_{0,i}>0$ be a compliance coefficient for contact point $i$, shared by that point's normal and tangential slot, and let $\eta>0$ be a conditioning coefficient with a default of $10^{-2}$. The diagonal added to the gated Delassus matrix is

$$
R
=\sigma\,
\operatorname{diag}\!\left[
c_0\odot(\mathbf 1-S^2)+\eta
\right],
$$

with $c_0$ and the gate both repeated across each contact's two slots. Because $s_i\in[0,1]$, the shape factor $1-s_i^2$ lies in $[0,1]$ and $R$ is bounded on both sides:

$$
\eta\,\sigma I
\ \preceq\ R\ \preceq\
\sigma\operatorname{diag}(c_0+\eta).
$$

The lower bound makes $R$ positive-definite for every gate value, so $H$ is positive-definite and the target QP stays strictly convex. The upper bound keeps the matrix handed to the linear solve bounded, so its conditioning is uniform across the band.

#### The physical compliance

Undoing the latent scaling of §7.1 with $\Lambda=Sy$ expresses the same diagonal in physical contact coordinates:

$$
\tilde R
=S^{-1}RS^{-1}
=\sigma\left[
c_0\odot\left(S^{-2}-\mathbf 1\right)+\eta S^{-2}
\right],
$$

that is, $\tilde R_{ii}=\sigma\left[c_{0,i}(1/s_i^2-1)+\eta/s_i^2\right]$ for each slot.

$\tilde R$ is the contact compliance. At an interior optimum the exact relation is

$$
v^+-v^*=-\tilde R\Lambda,
$$

so $\tilde R$ maps impulse to the residual by which the requested outgoing velocity is missed: a small $\tilde R$ enforces the request closely, and a large $\tilde R$ yields under load. Its inverse is the contact impedance.

The two ends of the band read directly off the formula:

- at $s_i=1$ the shape factor vanishes and $\tilde R_{ii}=\eta\sigma$, the conditioning floor alone, so a fully engaged contact enforces its requested velocity to within that floor;
- as $s_i\to0$ the compliance grows without bound and $\Lambda_i\to0$, so the physical response fades smoothly to zero through the transition band.

$c_{0,i}$ therefore sets how soft a partially engaged contact is, and $\eta$ sets the conditioning of the linear algebra. Writing the diagonal so that the compliance term vanishes at full engagement keeps those two roles separate: the force a contact transmits through the taper is selected by $c_{0,i}$, and the same force is insensitive to $\eta$ over several decades.

#### The learned stiffness

$c_{0,i}$ is a constitutive property of the contact, learned alongside restitution, penetration recovery and friction:

$$
c_{0,i}=c_{\mathrm{floor}}+\operatorname{softplus}(\hat c_i),
\qquad
c_{\mathrm{floor}}=\kappa\,c_{0}^{\mathrm{init}},
$$

with one unconstrained parameter $\hat c_i$ per contact point and $\kappa=0.1$ by default. The offset keeps the compliance term present at every parameter value, so the separation of roles above holds throughout training.

Sharing one coefficient between a contact's normal and tangential slot is what preserves the cone implication of §7.1: the scaling applied inside a contact block stays uniform, so cone membership and the friction ratio are untouched.

An equivalent position-level stiffness follows from static equilibrium. A resting contact has zero outgoing normal velocity and zero current normal velocity, so its request reduces to the recovery term and the compliance relation $v^+-v^*=-\tilde R\Lambda$ balances that request against the load carried. With a resting penetration $-g_i$ recovered at fraction $\beta_i$ over the response interval $h$, stationarity gives

$$
\beta_i\frac{-g_i}{h}=\tilde R_{ii}\Lambda_{n,i},
\qquad
k_i=\frac{\beta_i}{\tilde R_{ii}h^2},
$$

the second relation following from the sustained force $\Lambda_{n,i}/h$ acting through a deflection equal to the penetration. The penetration cancels, so a loaded contact carries force in proportion to how far it has sunk, with $k_i$ as its spring constant.

The gate value fixes which compliance enters that relation. A contact carrying load at rest has $g_i\le0=g_{\mathrm{on}}$, so $s_i=1$, the shape factor $1-s_i^2$ vanishes and $\tilde R_{ii}=\eta\sigma$:

$$
k_i=\frac{\beta_i}{\eta\,\sigma\,h^2}.
$$

A resting contact therefore carries load at a penetration set by the conditioning floor $\eta$, the recovery fraction $\beta_i$, the contact-space scale $\sigma$ and the response interval, and $c_{0,i}$ sets the softness of the taper, where its shape factor is active. Raising $g_{\mathrm{on}}$ above zero would place a load-bearing contact inside the compliant region and bring $c_{0,i}$ into this relation as well.

### 7.3 The target optimization problem

During one contact solve, $H$, $c$, and $\mu$ are known, and $y$ is the decision variable. Define

$$
\operatorname{sym}(A)=\frac12(A+A^\top).
$$

The explicit `sym` operation enforces numerical symmetry in floating point. The mathematical problem targeted by the finite solver is

$$
\boxed{
\begin{aligned}
\underset{y\in\mathbb R^{2K}}{\operatorname{minimize}}
&\quad
\frac12y^\top H y+c^\top y\\
\operatorname{subject\ to}
&\quad y\in\mathcal C_\mu,
\end{aligned}
}
$$

with

$$
H=\operatorname{sym}(SW_{\mathrm{full}}S)+R,
\qquad
c=Sb,
\qquad
\Lambda=Sy.
$$

In exact arithmetic $SW_{\mathrm{full}}S$ is positive-semidefinite, and $R$ is positive-definite by the lower bound of §7.2, so $H$ is positive-definite. The target objective is therefore strictly convex and has a unique exact minimizer. The fixed-iteration ADMM output is an approximation to that minimizer.

The ideal residual is $r=W_{\mathrm{full}}\Lambda+b=v^+-v^*$. After symmetrization, the target objective has the mathematical gradient

$$
Hy+c
=S(v^+-v^*)+Ry.
$$

Consequently, the exact KKT conditions of the target problem apply to the combined residual $S(v^+-v^*)+Ry$. The $Ry$ term permits a physical velocity residual and controls latent impulse magnitude. At an interior optimum the exact relation is $v^+-v^*=-\tilde R\Lambda$ with $\tilde R=S^{-1}RS^{-1}$, so the residual is the compliance of §7.2 acting on the selected impulse.

#### The same problem in physical coordinates

Each planar Coulomb cone is invariant under a positive scaling applied uniformly inside its own block, and $S$ scales contact $i$'s two slots by the common factor $s_i$. Hence

$$
y\in\mathcal C_\mu
\quad\Longleftrightarrow\quad
Sy\in\mathcal C_\mu,
$$

and the substitution $\Lambda=Sy$ carries the target problem to an equivalent statement in physical impulse:

$$
\boxed{
\begin{aligned}
\underset{\Lambda\in\mathbb R^{2K}}{\operatorname{minimize}}
&\quad
\frac12\Lambda^\top\!\left(W_{\mathrm{full}}+\tilde R\right)\Lambda+b^\top\Lambda\\
\operatorname{subject\ to}
&\quad \Lambda\in\mathcal C_\mu.
\end{aligned}
}
$$

This is the ideal QP of §5 with the compliance $\tilde R$ added to the Delassus matrix, which is the precise sense in which the formulation is a compliant contact: the outgoing motion is the projection of the free motion onto the friction cone in a metric softened by $\tilde R$.

The two statements are exactly equivalent, and each is convenient for a different purpose. The latent form is bounded and uniformly conditioned across the band, which suits the finite iteration of §8. The physical form exhibits the contact law directly, and is the form in which $c_{0,i}$ and $\eta$ read as a stiffness and a conditioning floor.

---

## 8. How the QP is solved: ADMM at a high level

The alternating direction method of multipliers, or **ADMM**, separates two easy operations:

1. minimize an unconstrained quadratic;
2. project a trial impulse onto the feasible friction cone.

### 8.1 From Lagrange multipliers to ADMM, geometrically

**The problem, restated.** The solver wants a single impulse: the point that minimizes the objective bowl and still lies in the friction cone,

$$
\min_{y}\ \tfrac12 y^\top H y + c^\top y
\qquad\text{subject to}\qquad
y\in\mathcal C_\mu .
$$

By §5.4 this is nested objective ellipsoids set against a wedge: find the lowest ellipsoid that still touches the cone. Each half is easy on its own — minimizing the bowl is the single linear solve $Hy=-c$, and projecting a point onto the cone has the closed form given below. The only difficulty is that the bowl's bottom usually lies outside the cone, so the two easy answers disagree. ADMM is a disciplined way to reconcile them.

**From an equation to a set.** Ordinary Lagrange multipliers price an *equation* $g(y)=0$ by adding a term $\xi^\top g(y)$ and seeking a saddle, where the objective gradient is balanced by a constraint force $\xi\,\nabla g$. Here the constraint is a *set*, $y\in\mathcal C_\mu$, with no equation to price. The fix is to introduce a second copy $z$ of the impulse and require the two copies to agree:

$$
\min_{y,z}\ \tfrac12 y^\top H y + c^\top y + \iota_{\mathcal C_\mu}(z)
\qquad\text{subject to}\qquad
y=z,
$$

with $\iota_{\mathcal C_\mu}(z)=0$ on the cone and $+\infty$ off it. Now $y$ carries only the smooth bowl, $z$ carries only the cone, and they are joined by the single equation $y=z$ — exactly the kind of constraint a multiplier $\xi$ can price. (A frequent snag: in the textbook statement of duality the multiplier is often called $y$; in this note $y$ is the impulse and $\xi$ is the multiplier.)

**Why we maximize over the multiplier.** Fix $\xi$ and minimize the priced objective over $(y,z)$; call the result $g(\xi)$. Dropping the hard agreement $y=z$ and only pricing it lets the minimization roam over more points, so $g(\xi)$ can never exceed the true constrained minimum: every $\xi$ furnishes a *lower bound* on the answer. The tightest bound is the largest one, so we **maximize** over $\xi$. Geometrically, each $\xi$ is a supporting plane resting beneath the true optimum; raising $g(\xi)$ tilts that plane upward until it just kisses the optimum, and at the touch the duality gap is zero and the copies agree, $y=z$. The optimal $\xi$ is precisely the force needed to hold the impulse against the cone — the same contact reaction the KKT residual of §6 describes.

**Why the residual is the step.** At the inner minimum the priced objective is flat in $(y,z)$, so the dual value only still responds to the leftover disagreement, and the gradient of $g$ is exactly the consensus residual $y-z$. Hence

$$
\xi \leftarrow \xi + (y-z)
$$

is gradient ascent on the dual: accumulate the disagreement, and the growing $\xi$ draws the copies together. When they meet, the residual vanishes, the dual is maximized, and the impulse is at once optimal and feasible.

**The penalty and the alternating sweep.** Pricing alone leaves the $y$-step poorly conditioned, so add a penalty $\tfrac\rho2\lVert y-z\rVert^2$ on the same disagreement: each subproblem becomes strictly convex and $\rho$ serves as a robust fixed ascent step (the classical "method of multipliers"). Minimizing this augmented objective over $y$ and $z$ *jointly* is as hard as the coupled original, so ADMM minimizes them **one block at a time**:

- **update $y$** — minimize the bowl plus the penalty pull toward $z$: one linear solve, the $H$-metric projection of §5.4 that reintroduces the stretched geometry;
- **update $z$** — project the pulled $y$ onto the cone: the Euclidean step;
- **update $\xi$** — add the residual $y-z$, building up the reconciling reaction.

Each sweep, $y$ slides toward the bowl's bottom, $z$ snaps back into the cone, and $\xi$ grows by whatever gap remains. The fixed point is the one place where the two copies coincide and each is optimal for its own half: the tangency point of §5.4, the constrained optimum.

### 8.2 The iteration and its guarantees

The penalty $\rho$ sets the step of the iteration, and it is drawn from the part of the diagonal that describes how the mechanism resists an impulse:

$$
\rho
=\max\!\left(
\operatorname{mean}\!\left[
\operatorname{diag}\!\left(SW_{\mathrm{full}}S\right)+\eta\,\sigma
\right],
10^{-6}
\right),
$$

detached from differentiation. Matching $\rho$ to the gated Delassus diagonal and the conditioning floor keeps the step in proportion to the coordinates that carry load, so the fixed budget below converges on the contacts that matter.

Using a primal variable $y$, a feasible auxiliary variable $z$, a scaled dual bookkeeping variable $\xi$, and that penalty, one iteration has the form

$$
y^{k+1}
=(H+\rho I)^{-1}
\left[-c+\rho(z^k-\xi^k)\right],
$$

followed by over-relaxation,

$$
\hat y^{k+1}=1.5y^{k+1}-0.5z^k,
$$

cone projection,

$$
z^{k+1}
=\Pi_{\mathcal C_\mu}(\hat y^{k+1}+\xi^k),
$$

and a dual update,

$$
\xi^{k+1}
=\xi^k+\hat y^{k+1}-z^{k+1}.
$$

The exact target QP has one variable $y$. ADMM represents it with two copies: the unconstrained primal iterate $y^k$ and cone-feasible auxiliary iterate $z^k$. Exact convergence brings the two copies together. The default finite budget performs 12 iterations and returns the current feasible approximation $z^{12}$ as `latent_impulse`. The iteration budget is configurable and fixed during a model execution.

For each planar contact, projection has three cases:

- a trial pair inside the cone maps to itself;
- a trial pair whose nearest feasible point is zero maps to the cone apex;
- each remaining pair maps to the nearest sloped cone boundary.

These operations provide two guarantees:

- **Feasibility is structural:** the returned latent impulse is in the cone because it is the result of an exact cone projection.
- **Optimality is measured:** the finite iteration budget returns an approximate minimizer. With $Q(z)=\tfrac12z^\top Hz+c^\top z$ and $L$ equal to the row-sum bound used in code, the solver reports the normalized projected-gradient fixed-point residual

  $$
  \frac{\left\lVert z-\Pi_{\mathcal C_\mu}\!\left(z-\nabla Q(z)/L\right)\right\rVert}
  {1+\lVert z\rVert}.
  $$

  A value near zero means that one projected-gradient update leaves the returned impulse nearly unchanged, which is the QP optimality condition.

The fixed iteration count also gives rollout backpropagation a fixed computation graph. The dense linear solve and the piecewise-differentiable cone projection allow gradients to pass through the contact response to the learned mass, geometry, material parameters, and actuator map.

---

## 9. Applying the solution

Let $y_{\mathrm{ret}}=z^N$ denote the returned feasible auxiliary iterate after the configured $N$ ADMM iterations. Form the physical impulse

$$
\Lambda=Sy_{\mathrm{ret}}.
$$

The generalized momentum change and outgoing velocity are

$$
\Delta p=J^\top\Lambda,
$$

$$
\dot q^+
=\dot q_{\mathrm{free}}+M^{-1}J^\top\Lambda.
$$

For use in the model drift, the equivalent average generalized force is

$$
F_c=\frac{J^\top\Lambda}{h},
$$

and the corresponding acceleration contribution is

$$
\ddot q_c=M^{-1}F_c.
$$

The internal response horizon defaults to $h=0.002\ \mathrm{s}$. The displayed $\dot q^+$ is the full-$h$ hypothetical response used by the contact solve and diagnostics. A surrounding Euler step of duration $\delta<h$ applies the equivalent force for $\delta$, producing the fraction $\delta/h$ of that impulse response, and then recomputes contact. Longer transition durations use repeated small steps with a fresh contact calculation at each step.

---

## 10. Discrete energy accounting

Virtual work verifies that the kinematic and force maps agree:

$$
\dot q^\top J^\top\frac{\Lambda}{h}
=(J\dot q)^\top\frac{\Lambda}{h}.
$$

This identity establishes work consistency between generalized and contact coordinates. For the solver's frozen-geometry, full-$h$ hypothetical response, the discrete kinetic-energy change is exactly

$$
\boxed{
\Delta T_c
=T^+-T_{\mathrm{free}}
=\Lambda^\top v_{\mathrm{free}}
+\frac12\Lambda^\top W_{\mathrm{full}}\Lambda.
}
$$

The second term accounts for the velocity change produced by the impulse. For a surrounding integration step $\delta<h$, the boxed quantity remains the full-$h$ solver-response diagnostic; the shorter Euler update uses its scaled impulse response for its own energy change.

For the ideal unbiased QP, this expression is the objective itself. With restitution and recovery, the optimized objective is shifted by $-(v^*)^\top\Lambda$, the decomposition displayed in §3.3. The boxed expression remains the exact full-$h$ response ledger used by the contact diagnostics.

In the ideal unbiased problem, maximum-dissipation friction gives $\Delta T_c\le0$. In a scalar or decoupled impact, restitution with a consistent current/free velocity baseline and $0\le e_i\le1$ also gives $\Delta T_c\le0$. The displayed ledger supplies the energy sign for a coupled multi-contact response and its actual velocity baselines. Penetration recovery requests outward motion and permits positive contact work. The audit reports the nonnegative recovery attribution

$$
\sum_i\beta_i d_i\Lambda_{n,i}
$$

as `stabilization_work`. The full-$h$ response quantity `discrete_work` is the total contact-work ledger, combining impact, friction, restitution, and recovery.

---

## 11. End-to-end recipe

At each internal step, the constraint contact path performs the following calculation:

1. Evaluate the contact-point positions $p_i(q)$ from the kinematic tree, giving the gaps $g_i(q)$ and horizontal coordinates.
2. Form $J_n$ and $J_t$ from the revolute-joint lever arms, then stack $J$.
3. Evaluate $M(q)$ and the contact-free acceleration, including the current action.
4. Predict $\dot q_{\mathrm{free}}=\dot q+h\ddot q_{\mathrm{free}}$.
5. Compute $v_{\mathrm{free}}=J\dot q_{\mathrm{free}}$.
6. Build the restitution/recovery target $v^*$ and bias $b=v_{\mathrm{free}}-v^*$.
7. Compute $W_{\mathrm{full}}=JM^{-1}J^\top$ and the contact-space scale $\sigma$.
8. Evaluate the compact gate $S$, the compliance coefficients $c_0$, and the diagonal $R$.
9. Approximate the latent cone-QP minimizer with the fixed ADMM budget (12 iterations by default), returning the feasible auxiliary iterate $y_{\mathrm{ret}}$.
10. Form the physical impulse $\Lambda=Sy_{\mathrm{ret}}$.
11. Apply $J^\top\Lambda$ to generalized momentum, or $J^\top\Lambda/h$ as its average-force equivalent.
12. Report cone feasibility, optimality residual, outgoing velocities, and the discrete energy ledger.

---

## 12. Key properties

| property | role in the contact solve |
|---|---|
| least-constraint projection | the outgoing motion is the admissible motion closest to the free motion in the kinetic metric; the impulse is the multiplier of non-penetration |
| objective | combines contact-induced kinetic-energy change, requested outgoing velocity, and a small conditioning penalty |
| requested velocity | $v^*$ encodes inelastic normal response, rebound, penetration recovery, and the tangential sticking request; exact attainment means $v^*\in v_{\mathrm{free}}+W\mathcal C_\mu$ |
| selected regime | the cone apex represents zero impulse; a strict-interior optimum represents sticking; a sloped-boundary optimum represents friction saturation; its tangential residual distinguishes incipient slip from sliding |
| impulse variable | $\Lambda$ transfers momentum; $J^\top\Lambda/h$ is its equivalent average generalized force |
| coupled contacts | off-diagonal entries of $W_{\mathrm{full}}$ coordinate the impulses across contact points |
| contact geometry | gaps and contact Jacobians are the plane's forward kinematics in closed form, with $J_n$ and $J_t$ the two rows of the point Jacobian |
| compact gate | supplies smooth contact candidacy and an exact clear-flight cutoff |
| compliance $\tilde R$ | maps impulse to the velocity residual $v^+-v^*=-\tilde R\Lambda$; gate-shaped, so the response fades to zero across the band and a fully engaged contact tracks its requested velocity |
| learned stiffness $c_{0,i}$ | a constitutive parameter, trained with restitution, recovery and friction, that selects the force a partially engaged contact transmits |
| conditioning floor $\eta$ | supplies scale-aware curvature at every gate value, bounds the latent impulse, and selects a unique target minimizer |
| bounded diagonal | $\eta\sigma I\preceq R\preceq\sigma\operatorname{diag}(c_0+\eta)$, so positive-definiteness and conditioning hold uniformly across the band |
| cone projection | gives a feasible returned impulse at every ADMM iteration |
| solver residual | measures finite-budget optimality |
| Jacobian duality | $J$ maps generalized motion to contact motion; $J^\top$ maps contact impulse to generalized momentum and preserves virtual work |

---

## Appendix A. Worked examples

These instances put numbers on the ideal problem of §§3–5. The $W=I$ cases make the arithmetic transparent: a unit normal or tangential impulse changes the matching velocity component by one unit, and the off-diagonal couplings vanish, so the $W$-metric projection of §5.4 reads as the ordinary nearest point.

### A.1 One frictionless contact, one number

Take upward velocity as positive, a predicted normal velocity $v_{\mathrm{free}}=-1\ \mathrm{m/s}$ (closing), and effective mass $m_{\mathrm{eff}}=2\ \mathrm{kg}$. A positive normal impulse $\lambda$ changes the velocity by $\lambda/m_{\mathrm{eff}}$:

$$
v^+=v_{\mathrm{free}}+\frac{\lambda}{m_{\mathrm{eff}}}
=-1+\frac{\lambda}{2},
\qquad
\lambda\ge0.
$$

With the inelastic request $v^*=0$, the projection objective of §3 is

$$
\min_{\lambda\ge0}
Q(\lambda)
=\frac{1}{2m_{\mathrm{eff}}}\lambda^2
+v_{\mathrm{free}}\lambda
=\frac14\lambda^2-\lambda.
$$

Differentiating,

$$
Q'(\lambda)=\frac12\lambda-1=0
\quad\Longrightarrow\quad
\lambda^*=2,
$$

so $v^+=0$: inelastic closure. For a separating prediction $v_{\mathrm{free}}=+0.4\ \mathrm{m/s}$,

$$
Q'(\lambda)=\frac{\lambda}{m_{\mathrm{eff}}}+0.4>0
\qquad\text{for every }\lambda\ge0,
$$

so the score increases along the feasible half-line and its minimum is the boundary point $\lambda^*=0$. In general,

$$
\lambda^*
=\max\!\left(0,-m_{\mathrm{eff}}v_{\mathrm{free}}\right),
$$

$$
\begin{array}{lll}
v_{\mathrm{free}}\ge0
&\Longrightarrow&
\lambda^*=0,\quad v^+=v_{\mathrm{free}},\\[3pt]
v_{\mathrm{free}}<0
&\Longrightarrow&
\lambda^*=-m_{\mathrm{eff}}v_{\mathrm{free}},\quad v^+=0.
\end{array}
$$

The linear term $v_{\mathrm{free}}\lambda$ lowers the score as a positive impulse grows; the quadratic term bends it upward; the balance is the impulse that cancels the closing velocity. The kinetic-energy identity of §3.3 gives $T^+-T_{\mathrm{free}}=Q(\lambda)$ here exactly.

### A.2 Scalar target: three examples

With the shifted objective $\min_{\lambda\ge0}\tfrac12w\lambda^2+(v_{\mathrm{free}}-v^*)\lambda$ and $w=1/m_{\mathrm{eff}}$, the solution is

$$
\lambda^*
=\max\!\left(0,\frac{v^*-v_{\mathrm{free}}}{w}\right),
$$

and the impulse required for exact attainment is $\lambda_{\mathrm{req}}=(v^*-v_{\mathrm{free}})/w$. Since $w>0$ and $\lambda\ge0$, the scalar target is exactly attainable precisely when

$$
\boxed{v^*\ge v_{\mathrm{free}}.}
$$

Using $m_{\mathrm{eff}}=2\ \mathrm{kg}$, $w=0.5\ \mathrm{kg}^{-1}$:

| free velocity $v_{\mathrm{free}}$ | request $v^*$ | required impulse $\lambda_{\mathrm{req}}$ | selected outcome |
|---:|---:|---:|---|
| $-1.0$ | $0$ | $2.0\ \mathrm{N\,s}$ | $\lambda^*=2.0$, $v^+=0$: inelastic closure |
| $-1.0$ | $+0.3$ | $2.6\ \mathrm{N\,s}$ | $\lambda^*=2.6$, $v^+=+0.3$: rebound |
| $+0.4$ | $0$ | $-0.8\ \mathrm{N\,s}$ | $\lambda^*=0$, $v^+=+0.4$: separation |

The scalar solution satisfies the three optimality relations of §6.1,

$$
v^+\ge v^*,
\qquad
\lambda\ge0,
\qquad
\lambda(v^+-v^*)=0.
$$

A positive impulse attains $v^+=v^*$; a positive residual $v^+-v^*>0$ selects $\lambda=0$ and carries the free separating velocity into the outgoing state.

### A.3 Exact sticking and the friction ratio

Take one contact with $W=I$ and

$$
v_{\mathrm{free}}=(-1,0.4),
\qquad
v^*=(0,0).
$$

The impulse required to realize the request is

$$
\Lambda_{\mathrm{req}}
=v^*-v_{\mathrm{free}}
=(1,-0.4).
$$

Its normal component is $1\ \mathrm{N\,s}$ and its tangential magnitude is $0.4\ \mathrm{N\,s}$. With $\mu=0.5$, the tangential budget is $\mu\Lambda_{n,\mathrm{req}}=0.5$. The required impulse lies strictly inside the cone because $0.4<0.5$, so the ideal QP selects $\Lambda^*=(1,-0.4)$ and reaches $v^*=(0,0)$ exactly: the point sticks. The required friction ratio of §4.2 is $\mu_{\mathrm{required}}=|{-0.4}|/1=0.4<\mu$.

### A.4 A sliding contact

Take one contact with normalized $W=I$, restitution and recovery parameters $e=\beta=0$, and

$$
v_{\mathrm{free}}=(-1,2),
\qquad
\mu=0.5.
$$

The impulse required to make both velocity components zero is $\Lambda_{\mathrm{req}}=-v_{\mathrm{free}}=(1,-2)$, with required friction ratio $\mu_{\mathrm{required}}=|{-2}|/1=2$. The available coefficient is $\mu=0.5$, so $|{-2}|>0.5(1)$ and the optimum saturates friction.

The cone-constrained optimum lies on the sliding boundary $\Lambda_t=-0.5\Lambda_n$. Write $n=\Lambda_n$, so $\Lambda_t=-0.5n$. Along that boundary the objective becomes

$$
\begin{aligned}
Q(n)
&=\frac12\left(n^2+(-0.5n)^2\right)-n+2(-0.5n)\\
&=0.625n^2-2n,
\end{aligned}
$$

with derivative $Q'(n)=1.25n-2$. Setting it to zero gives $n=1.6$, hence $\Lambda_t=-0.8$:

$$
\Lambda^*=(1.6,-0.8),
\qquad
v^+
=v_{\mathrm{free}}+\Lambda^*
=(0.6,1.2).
$$

Friction reduces the tangential speed from $2$ to $1.2$, leaving residual sliding. The joint cone formulation also produces positive outward normal velocity: it spends extra normal impulse to unlock more tangential friction capacity and balances both effects in the same objective. The resulting sliding relation $v_n^+-v_n^*=\mu|v_t^+|$ of §6.2 couples the outward normal residual to the tangential residual.

---

## Appendix B. Minimal optimization vocabulary

An optimization problem has the general form

$$
\min_{x\in\mathcal F} f(x).
$$

| term | meaning in plain language | contact example |
|---|---|---|
| decision variable | the number or vector the solver is allowed to choose | the impulse $\Lambda$ |
| objective | the score the solver tries to make as small as possible | contact-induced energy change, with target shifts |
| constraint | a rule every allowed answer must obey | $\Lambda_n\ge0$ |
| feasible set $\mathcal F$ | all choices satisfying every constraint | the Coulomb friction cone |
| optimum or minimizer | a feasible choice with the lowest objective | $\Lambda^*$ |

Here **program** is optimization terminology for “optimization problem.”

**What makes it a quadratic program.** A quadratic program, abbreviated **QP**, has an objective of the form $\tfrac12x^\top Hx+c^\top x$ and linear constraints on $x$. The contact objective has exactly this form, with the impulse as $x$. The product $x^\top Hx$ evaluates to a scalar, the multi-dimensional analogue of a term such as $w x^2$. When $H$ is symmetric, the gradient — the vector version of a derivative — is

$$
\nabla_x\left(\frac12x^\top Hx+c^\top x\right)=Hx+c.
$$

At an unconstrained bowl minimum this gradient is zero; constraints can place the constrained optimum on the feasible-set boundary.

**What convex means.** Informally, a convex objective is bowl-shaped, and a convex feasible set contains every line segment joining two of its points. More precisely, every straight line segment between two feasible points stays feasible, and the objective along any such segment stays at or below the straight interpolation of its endpoint values. The useful consequence is that every local minimum of a convex problem is a global minimum; a continuous objective that tends to $+\infty$ as $\lVert x\rVert\to\infty$ attains a minimum on a closed feasible set; and strict curvature gives uniqueness. For a quadratic objective, convexity follows when $H$ is **positive-semidefinite**, $z^\top Hz\ge0$ for every $z$.

---

## Appendix C. Symbol glossary

| symbol | meaning |
|---|---|
| $q,\dot q$ | generalized configuration and velocity |
| $M(q)$ | symmetric positive-definite mass matrix |
| $p_i(q)$ | planar position of contact point $i$ from the kinematic tree |
| $\varrho_i$ | capsule radius of contact point $i$ |
| $g_i(q)=p_{i,z}-\varrho_i$ | signed gap of contact point $i$ to the ground plane |
| $\Theta_{i,k}$ | cumulative hinge angle placing the $k$-th link on the chain to point $i$ |
| $J_{v,i}$ | point Jacobian of contact $i$; its rows are $J_{n,i}$ and $J_{t,i}$ |
| $J$ | stacked normal/tangential contact Jacobian |
| $v_{\mathrm{free}}$ | contact-free predicted velocity |
| $v^*$ | requested outgoing contact velocity |
| $b=v_{\mathrm{free}}-v^*$ | linear term of the ideal QP |
| $W_{\mathrm{full}}=JM^{-1}J^\top$ | Delassus matrix; impulse-to-contact-velocity map |
| $\mu_i$ | dimensionless Coulomb friction coefficient; $\mu_i\Lambda_{n,i}$ is the tangential impulse budget |
| $\hat\mu_i$ | unconstrained raw parameter learning $\mu_i=2\,\operatorname{sigmoid}(\hat\mu_i)$ |
| $e_i$ | restitution speed ratio |
| $\beta_i$ | penetration-recovery fraction |
| $v_{\max}$ | cap on requested penetration-correction speed |
| $\mathcal C_\mu$ | product of the planar Coulomb cones |
| $\mathcal C_\mu^*$ | dual cone containing ideal velocity residuals at an optimum |
| $s_i,S$ | scalar contact gates and their diagonal matrix |
| $y$ | exact latent decision variable of the target QP |
| $y_{\mathrm{ret}}=z^N$ | feasible finite-ADMM approximation returned by the implementation |
| $\Lambda$ | physical impulse; $Sy$ in the exact target problem and $Sy_{\mathrm{ret}}$ in the finite implementation |
| $\sigma$ | contact-space inverse-mass scale, the mean gated-free Delassus diagonal |
| $c_{0,i}$ | learned compliance coefficient of contact point $i$, shared by its two slots |
| $\hat c_i$ | unconstrained raw parameter learning $c_{0,i}=c_{\mathrm{floor}}+\operatorname{softplus}(\hat c_i)$ |
| $c_{\mathrm{floor}}=\kappa c_0^{\mathrm{init}}$ | lower bound keeping the compliance term present at every parameter value |
| $\eta$ | conditioning coefficient of the positive floor |
| $R=\sigma\operatorname{diag}[c_0\odot(\mathbf 1-S^2)+\eta]$ | latent diagonal: gate-shaped compliance plus conditioning floor |
| $\tilde R=S^{-1}RS^{-1}$ | physical contact compliance; $v^+-v^*=-\tilde R\Lambda$ at an interior optimum |
| $k_i=\beta_i/(\tilde R_{ii}h^2)$ | equivalent position-level normal stiffness at rest |
| $H=\operatorname{sym}(SW_{\mathrm{full}}S)+R$ | target QP Hessian |
| $\rho$ | ADMM penalty, from the gated Delassus diagonal and the conditioning floor |
| $I$ | identity matrix |
| $h$ | fixed contact response interval |

---

## Related notes and code

- [Structured dynamics model](structured_dynamics_model.md), especially §2.7 and Appendix B.3.
- [Cheetah instantiation](structured_dynamics_cheetah.md), which explains why locomotion needs the explicit ground-reaction route.
- [Hamiltonian recovery report](hamiltonian_recovery_report.md), which defines the contact solver, cone, work, and recovery diagnostics.
- [`PortHamiltonianModel._constraint_contact_solve`](../models/port_hamiltonian.py), the implementation of the gated regularized QP and fixed ADMM solve.
- [`TestConstraintContactForcePort`](../tests/test_model_based_generator.py), focused tests for action-aware static friction, exact clear flight, cone feasibility, differentiability, and duplicate contacts.
