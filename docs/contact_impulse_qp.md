---
title: Contact Impulse QP, from First Principles
tags: [ct-rl, contact, convex-optimization, quadratic-programming, model-based]
robots: noindex
---

# The Contact-Impulse Quadratic Program, from First Principles

:::info
**Purpose.** This note derives the contact optimization used by the structured dynamics model from one mechanical principle. An ideal frictionless normal reaction has zero virtual-work pairing with every variation tangent to the active constraint, so the outgoing motion is the kinetic-metric projection of the freely predicted motion onto the admissible set. The note builds that projection into the contact quadratic program, then adds the Coulomb cone, a hard predicted-crossing active set, a physical stiffness parameterization, and the fixed-budget solver.

The short version: **the outgoing motion is the admissible motion closest to the free motion in the kinetic-energy metric.** Exact forward kinematics supplies signed gaps. Eligibility includes engaged points with $g_i\le0$ and points whose action-aware free endpoint reaches or crosses the plane during the response step; every remaining normal–tangential impulse block equals zero. Active contacts share one physical-impulse solve. Version 4 learns one shallow stiffness per point, one shared high-to-low stiffness ratio, and one tangential compliance per point. A fixed one-millimetre transition maps penetration to effective normal secant stiffness. The desired penetration-recovery fraction $\beta$ is fixed configuration. Hard predicted-crossing activation selects the eligible impulse blocks.

This is a companion to [§2.7 and Appendix B.3 of the structured dynamics note](structured_dynamics_model.md#27-explicit-contact-port-k-learned-point-contacts). The implementation is _constraint_contact_solve in [models/port_hamiltonian.py](../models/port_hamiltonian.py), with focused tests in [tests/test_predicted_crossing_contact.py](../tests/test_predicted_crossing_contact.py).
:::

[TOC]

Sections 1–5 build the ideal contact problem from the least-constraint principle: the principle (§1), the impulse and the contact map (§2), the projection written as a quadratic program (§3), the friction cone (§4), and the complete ideal problem (§5). Section 6 reads off its optimality conditions, §7 gives the predicted-crossing compliant form, §8 the solver, and §§9–11 its use, with §12 a summary. Worked numerical examples are collected in Appendix A and a minimal optimization vocabulary in Appendix B.

---

## 1. Contact as least constraint

At each internal step the solver has a predicted motion — the velocity the smooth mechanics and current action produce before the ground response — and chooses the ground impulse that corrects it:

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

**The postulate.** An ideal frictionless normal reaction has zero virtual-work pairing with every variation tangent to the active constraint. The ground supplies a reaction covector in the blocked normal direction. This defines an ideal, passive constraint.

**The consequence is a projection.** The reaction belongs to the normal cone of the admissible velocity set at the outgoing motion. The realized motion therefore stays as close as possible to the freely predicted motion subject to the constraint, measured in the kinetic-energy metric — the metric in which "distance" is the kinetic energy of the difference. Writing $\dot q_{\mathrm{free}}$ for the freely predicted generalized velocity, $\dot q^+$ for the realized one, and $M$ for the mass matrix,

$$\boxed{ \dot q^+=\underset{\dot q\in\mathcal A}{\arg\min}\ \tfrac12(\dot q-\dot q_{\mathrm{free}})^\top M(\dot q-\dot q_{\mathrm{free}}), }$$

where $\mathcal A$ is the set of admissible outgoing motions. The outgoing motion is the $M$-metric projection of the free motion onto $\mathcal A$. This is the **principle of least constraint**, and the contact solver builds its ideal target from this projection.

**What admissibility means.** The admissible set collects the conditions a real contact imposes:

- **an outgoing normal bound** — a closing point satisfies $v_n^+\ge v_n^*$, with zero requesting an inelastic stop and a positive value requesting rebound or penetration recovery;
- **a one-sided reaction** — the normal impulse satisfies $\Lambda_n\ge0$ and represents a ground push;
- **a friction bound** — the tangential impulse a contact can carry is limited by the normal impulse through the Coulomb coefficient.

The first condition defines $\mathcal A$ in velocity space; the second and third describe the impulses allowed to enforce it. The rest of the note turns the boxed projection into a quadratic program: §2 introduces the impulse and the map from impulse to contact velocity, §3 writes the projection as that program, §4 gives the friction cone, and §5 states the complete ideal problem. Sections 6–8 add the optimality conditions, the predicted-crossing compliant form, and the solver.

---

## 2. The impulse and the contact map

Computing the projection of §1 needs two things: the variable the contact chooses, and the linear map from that variable to the outgoing contact velocity. Let the robot have $n$ generalized velocities and $K$ declared planar contact points with exact kinematics, each contributing a normal and a tangential coordinate.

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

$$\Lambda= (\Lambda_{n,1},\Lambda_{t,1},\ldots, \Lambda_{n,K},\Lambda_{t,K})^\top.$$

### 2.1 Impulse as the contact variable

An impact acts over a very short time and can contain a very large force. Its finite mechanical effect is the **impulse**,

$$\Lambda=\int_{t}^{t+h} f_c(s)\,ds.$$

Impulse has units of force times time, such as $\mathrm{N\,s}$. It directly changes momentum. Once the impulse is known, the model can expose an equivalent average force over the step by dividing by $h$.

### 2.2 The contact Jacobian

For contact point $i$, write its contact coordinates in normal–tangential order as

$$r_i^c(q)= \begin{bmatrix} g_i(q)\\ p_{i,x}(q) \end{bmatrix},$$

where $p_i(q)=(p_{i,x},p_{i,z})$ is the point's position in the plane of motion and

$$g_i(q)=p_{i,z}(q)-\varrho_i$$

is its signed distance to a ground plane at $z=0$, with $\varrho_i$ the radius of the capsule the point belongs to. The signed-gap convention is

$$g_i>0\ \text{above the ground}, \qquad g_i=0\ \text{at the ground}, \qquad g_i<0\ \text{in penetration}.$$

Both coordinates come from the kinematic tree in closed form. Each contact point is reached from the root through an ordered chain of hinge angles $a_0,a_1,\ldots,a_m$ with a link offset $d_k=(d_{k,x},d_{k,z})$ per joint, all hinges sharing one axis, so rotations compose by addition:

$$\Theta_{i,k}=\sum_{j\le k}q_{a_j},$$

$$p_{i,z}=z_0+q_z+\sum_k\left(-d_{k,x}\sin\Theta_{i,k}+d_{k,z}\cos\Theta_{i,k}\right),$$

$$p_{i,x}=q_x+\sum_k\left(d_{k,x}\cos\Theta_{i,k}+d_{k,z}\sin\Theta_{i,k}\right),$$

where $z_0$ is the root frame's height offset and $q_x,q_z$ are the root translations.

Differentiating a rotation about a shared axis gives the classical revolute-joint column, a cross product of the axis with the lever arm from that joint's pivot to the point:

$$\frac{\partial p_i}{\partial q_{a_j}} =\hat\omega\times(p_i-p_j) =\begin{bmatrix} +(p_{i,z}-p_{j,z})\\ -(p_{i,x}-p_{j,x}) \end{bmatrix},$$

together with the two root columns

$$\frac{\partial p_{i,x}}{\partial q_x}=1, \qquad \frac{\partial p_{i,z}}{\partial q_z}=1.$$

The lever arm $p_i-p_j$ is the sum of every link offset from joint $a_j$ outward, so a single pair of cumulative sums along the chain — forward over the angles, backward over the placed offsets — supplies the whole point Jacobian at once.

Collecting the two components gives the point Jacobian $J_{v,i}$, whose rows are the contact-frame directions for a ground plane of constant normal:

$$J_{n,i}=\hat z^\top J_{v,i}=\frac{\partial g_i}{\partial q}, \qquad J_{t,i}=\hat x^\top J_{v,i}.$$

Two entries follow directly and fix the coupling to the root degrees of freedom: $J_{n,i}$ has a unit entry in the root-height column and vanishes in the root-horizontal column, since a point's height is independent of horizontal translation; $J_{t,i}$ has a unit entry in the root-horizontal column, since translating the root translates every point with it.

Stacking those rows over all contacts gives $J$, which maps generalized velocity to contact velocity:

$$v=J\dot q.$$

The transpose maps a contact-space impulse back to generalized momentum:

$$\Delta p=J^\top\Lambda.$$

This transpose follows from virtual work:

$$\Lambda^\top(J\,\delta q) =(J^\top\Lambda)^\top\delta q.$$

It is the virtual-work dual, or pullback, of $J$.

### 2.3 First predict the contact-free motion

The smooth mechanical model and current action give

$$\ddot q_{\mathrm{free}} =M^{-1} \left(G_a a-\nabla V-C(q,\dot q)\dot q-D\dot q\right).$$

| term | mechanical role |
|---|---|
| $G_a a$ | maps the current action into generalized actuator force |
| $-\nabla V$ | conservative force, including gravity |
| $-C(q,\dot q)\dot q$ | Coriolis and centrifugal force |
| $-D\dot q$ | passive joint damping |
| $M^{-1}$ | converts the total generalized force to acceleration |

In the implementation this free acceleration also includes any configured joint-limit spring and damping forces. “Free” labels the stage immediately before the learned ground-contact impulse.

Predict the velocity at the end of the smooth-mechanics stage:

$$\dot q_{\mathrm{free}} =\dot q+h\ddot q_{\mathrm{free}}, \qquad v_{\mathrm{free}}=J\dot q_{\mathrm{free}}.$$

The action is already inside $v_{\mathrm{free}}$. When the current tangential velocity is zero and the action predicts impending slip, $v_{\mathrm{free},t}$ becomes nonzero. An active contact with sufficient Coulomb budget then selects a static-friction impulse. Cone saturation gives sliding. The hard equality on every inactive block gives zero impulse.

### 2.4 From impulse to outgoing contact velocity

Holding $q$, $M$, and $J$ fixed over the short impulse response,

$$M(\dot q^+-\dot q_{\mathrm{free}})=J^\top\Lambda.$$

Therefore

$$\dot q^+ =\dot q_{\mathrm{free}}+M^{-1}J^\top\Lambda,$$

and the outgoing contact velocity is

$$\boxed{ v^+=v_{\mathrm{free}}+W\Lambda, \qquad W=JM^{-1}J^\top. }$$

The diagonal entries of $W$ behave like inverse effective masses. Its off-diagonal entries describe coupling: an impulse at one contact can change another contact's velocity.

The matrix $W$ is positive-semidefinite because

$$z^\top Wz =(J^\top z)^\top M^{-1}(J^\top z) \ge0.$$

This is the main reason the contact objective built next is convex.

### 2.5 The two adjoints, and $W$ as their composition

Separating the spaces that $J$ and $J^\top$ act on explains what $W$ is.

Write $J:\mathcal V\to\mathcal U$, from the generalized-velocity space $\mathcal V$ (dimension $n$; $J$ acts on velocities $\dot q$) to the contact-velocity space $\mathcal U$ (dimension $2K$). The relation $v=J\dot q$ carries a generalized velocity to a contact velocity.

The transpose $J^\top$ of §2.2 is an **adjoint** of $J$. The following table defines the transpose and Hilbert adjoints by their maps, metrics, inputs, and outputs.

| adjoint | maps | needs a metric? | acts on | returns |
|---|---|---|---|---|
| Hilbert $J^\ast$ | $\mathcal U\to\mathcal V$ | yes, on both spaces | contact velocities | generalized velocities |
| transpose $J^\top$ | $\mathcal U^\ast\to\mathcal V^\ast$ | metric-independent | contact forces | generalized forces |

The **Hilbert adjoint** $J^\ast:\mathcal U\to\mathcal V$ exists once both spaces carry inner products, fixed by $\langle Jx,y\rangle_{\mathcal U}=\langle x,J^\ast y\rangle_{\mathcal V}$. It sends velocities to velocities.

The **transpose adjoint** $J^\top:\mathcal U^\ast\to\mathcal V^\ast$ is metric-independent. The pairing fixes it through $(J^\top\phi)(x)=\phi(Jx)$, and it sends a linear functional on contact velocities to one on generalized velocities. The dual spaces $\mathcal U^\ast$ and $\mathcal V^\ast$ are the **force** spaces: a contact force $\Lambda\in\mathcal U^\ast$ pairs with a contact velocity to give power, and a generalized force or momentum $\Delta p\in\mathcal V^\ast$ pairs with a generalized velocity to give power.

The $J^\top$ in $\Delta p=J^\top\Lambda$ (§2.2) is the transpose adjoint: $\Lambda$ is a contact force and $\Delta p$ a generalized momentum, so it operates on the dual of the contact space and lands in the dual of the generalized space. Mechanics selects this adjoint through the power pairing — force times velocity — between a space and its dual. The Hilbert adjoint serves the velocity-to-velocity map after inner products are chosen. Forces are the duals of velocities, so the transpose of a velocity map is a force map.

Given inner products $G_{\mathcal V}$ on $\mathcal V$ and $G_{\mathcal U}$ on $\mathcal U$, the two adjoints coincide up to those metrics, $J^\ast=G_{\mathcal V}^{-1}J^\top G_{\mathcal U}$; in orthonormal coordinates their matrices are equal, which is why one symbol serves both.

**$W$ is the composition of these maps.** Read $W=JM^{-1}J^\top$ right to left as a chain across the four spaces:

$$\underbrace{\Lambda}_{\mathcal U^\ast} \ \xrightarrow{\ J^\top\ }\ \underbrace{J^\top\Lambda}_{\mathcal V^\ast} \ \xrightarrow{\ M^{-1}\ }\ \underbrace{M^{-1}J^\top\Lambda}_{\mathcal V} \ \xrightarrow{\ J\ }\ \underbrace{W\Lambda}_{\mathcal U}.$$

A contact force $\Lambda$ becomes a generalized force $J^\top\Lambda$ (transpose adjoint), then the generalized velocity $M^{-1}J^\top\Lambda$ that force produces, then the contact velocity $W\Lambda$. So $W:\mathcal U^\ast\to\mathcal U$ maps a contact **force** to a contact **velocity** — the contact-space inverse mass, or Delassus matrix.

The one place a metric enters the chain is $M^{-1}$. The mass matrix $M:\mathcal V\to\mathcal V^\ast$ is the inner product on generalized velocities, since the kinetic energy is $\tfrac12\dot q^\top M\dot q$; it turns a generalized velocity into its momentum, and $M^{-1}$ turns a generalized force back into the velocity it produces. The maps $J$ and $J^\top$ on either side are the pairing-defined primal and dual maps. This is why §3.1 substitutes $\Delta\dot q=M^{-1}J^\top\Lambda$ and the kinetic deviation collapses to $\tfrac12\Lambda^\top W\Lambda$: that quadratic form is the kinetic energy of the impulse response.

$W$ then carries a metric role of its own. The positive-semidefinite quadratic form $\Lambda^\top W\Lambda$ defines a seminorm on contact-force space. Independent contact rows make $W$ positive-definite and turn this seminorm into an inner product; $W^{-1}$ then defines the kinetic metric on contact-velocity space. For rank-deficient contact rows, $W^\dagger$ supplies the corresponding metric on the reachable contact-velocity subspace. Section 5.4 uses these metrics to describe the projection.

---

## 3. The projection as a quadratic program

### 3.1 From least constraint to the quadratic program

Section 1 places the outgoing motion at the $M$-metric-nearest admissible point to the free motion. Two facts turn that projection into a program in the impulse.

**The reaction is $M$-normal to the admissible set.** The contact acts through the impulse $\Lambda$ as the generalized reaction $J^\top\Lambda$ (§2.2), so the realized velocity change is $\Delta\dot q=\dot q^+-\dot q_{\mathrm{free}}=M^{-1}J^\top\Lambda$. For every velocity variation $\delta\dot q$ tangent to the active admissible face, the virtual-work pairing is $\delta\dot q^\top J^\top\Lambda=(J\delta\dot q)^\top\Lambda=0$. In the kinetic metric,

$$\langle\Delta\dot q,\ \delta\dot q\rangle_M =\delta\dot q^\top M\,\Delta\dot q =\delta\dot q^\top J^\top\Lambda =(J\delta\dot q)^\top\Lambda =0 .$$

Thus the reaction lies in the $M$-normal cone of $\mathcal A$ at $\dot q^+$, and $\dot q^+$ is the projection §1 describes.

**The projection, in the impulse, is a quadratic.** Substituting $\Delta\dot q=M^{-1}J^\top\Lambda$ into the projection objective collapses it to a quadratic in $\Lambda$:

$$\tfrac12(\dot q^+-\dot q_{\mathrm{free}})^\top M(\dot q^+-\dot q_{\mathrm{free}}) =\tfrac12\Lambda^\top JM^{-1}J^\top\Lambda =\tfrac12\Lambda^\top W\Lambda, \qquad W=JM^{-1}J^\top .$$

The kinetic deviation of the outgoing motion from the free motion is exactly $\tfrac12\Lambda^\top W\Lambda$. For the frictionless normal projection, retain the normal rows and write

$$\min_{\Lambda_n}\ \tfrac12\Lambda_n^\top W_n\Lambda_n \qquad\text{subject to}\qquad v_{\mathrm{free},n}+W_n\Lambda_n\ \ge\ v_n^* ,$$

the admissibility being the outgoing normal bound $v_n^+\ge v_n^*$ from §1.

**The normal impulse is the multiplier of the outgoing normal bound.** Attach a multiplier $s\ge0$ to the normal constraint $v_n^+-v_n^*\ge0$ and set the derivative in the normal impulse to zero. In the frictionless normal problem with positive-definite normal-coordinate Delassus submatrix $W_n$, stationarity gives $W_n\Lambda_n=W_ns$, hence $\Lambda_n=s\ge0$. The normal impulse *is* the constraint's multiplier, and its one-sidedness emerges from the projection. Complementarity $s^\top(v_n^+-v_n^*)=0$ then reads $\Lambda_n^\top(v_n^+-v_n^*)=0$. Section 6 recovers these contact laws formally, and §4 adds the associated Coulomb tangential response.

**Writing the admissible impulses as the feasible set.** The associated frictional model stacks normal and tangential coordinates in $\Lambda$, sets the tangential request to zero, and uses the Coulomb cone from §4 as its feasible set. The solver keeps the impulse as the variable and moves the velocity request into the objective. For invertible $W$, complete the square around the impulse that would attain $v^*$ exactly, $\Lambda_{\mathrm{req}}=W^{-1}(v^*-v_{\mathrm{free}})$:

$$\tfrac12(\Lambda-\Lambda_{\mathrm{req}})^\top W(\Lambda-\Lambda_{\mathrm{req}}) =\tfrac12\Lambda^\top W\Lambda+b^\top\Lambda+\text{const}, \qquad b=v_{\mathrm{free}}-v^* ,$$

since $-\Lambda_{\mathrm{req}}^\top W=b^\top$. The objective becomes the $W$-metric deviation of the impulse from the target-achieving impulse, equivalently $\tfrac12\lVert v^+-v^*\rVert_{W^{-1}}^2$, and measures distance from the requested outgoing velocity. For singular $W$, the statement holds on its reachable range with $W^\dagger$, and the QP expression remains well-defined over the full impulse space. Over the admissible impulses this is

$$\boxed{ \min_{\Lambda\ \text{admissible}}\ \tfrac12\Lambda^\top W\Lambda+b^\top\Lambda, \qquad b=v_{\mathrm{free}}-v^* . }$$

This is the associated contact quadratic program. Its normal coordinate inherits the least-constraint projection and outgoing bound. Its stacked normal–tangential form uses the same kinetic quadratic, the requested velocity shift, and the Coulomb impulse cone.

This QP is an **inner solve**, run afresh for every state and action; fitting the model is the **outer optimization**, and the differentiable solve carries the outer gradient through the selected impulse. Section 4 makes “admissible” the friction cone, §5 states the complete problem, and §6 reads off the contact laws as its optimality conditions.

### 3.2 The requested outgoing velocity

The **requested outgoing velocity** $v^*$ is the contact velocity that an active response is asked to produce. Its tangential component is zero, requesting sticking when the Coulomb cone permits it. Its normal component depends on why the point is active.

For an already engaged point, define the penetration-correction speed

$$d_i=\frac{[-g_i]_+}{h}.$$

The engaged request combines fractional penetration recovery and restitution:

$$v_{n,i}^*=\beta d_i+e_i[-v_{n,i}]_+, \qquad g_i\le0,$$

where $v_{n,i}=J_{n,i}\dot q$ is the current normal velocity. For a point above the plane whose contact-free prediction crosses it during this response interval, the request is

$$v_{n,i}^*=-\frac{g_i}{h}, \qquad g_i>0\ \text{and predicted crossing}.$$

Under exact target attainment, this **landing request** satisfies $g_i+h v_{n,i}^*=0$. Restitution begins after engagement, preserving the approach through the crossing. Physical compliance permits a velocity residual relative to this target, so an active soft contact may continue deforming after crossing.

The restitution $e_i\in(0,0.5)$ and friction $\mu_i\in(0,2)$ are learned once per contact point. The recovery fraction $\beta\in(0,1]$ is one fixed, state-independent configuration value shared by the contact points; the implemented default is $0.2$, meaning “set the desired recovery component to 20% of the existing penetration over one response horizon.” Version-3 and version-4 checkpoints persist $\beta$ as configuration external to the optimizer. Both versions apply the full $\beta\delta/h$ request at every penetration depth, preserving the loaded resting equilibrium $F=K(\delta)\delta$ derived in §7.2. Solver versions 1–2 retain their historical learned, capped recovery behavior.

Stacking normal and zero tangential targets gives $v^*$, and

$$b=v_{\mathrm{free}}-v^*.$$

Componentwise, the implementation uses

$$b_{n,i} = v_{\mathrm{free},n,i} +\frac{[g_i]_+}{h} -\beta d_i +e_i\min(v_{n,i},0)\,\mathbf 1_{\{g_i\le0\}}, \qquad b_{t,i}=v_{\mathrm{free},t,i}.$$

This bias applies to the active blocks defined in §7. The selected impulse produces $v^+=v_{\mathrm{free}}+W_{\mathrm{full}}\Lambda$. In the ideal problem, exact attainment gives $v^+=v^*$; the physical diagonal of §7.2 permits a constitutive residual.

### 3.3 The objective as a shifted energy ledger

The projection objective has a direct energy reading. The kinetic-energy change produced by the impulse (§10 derives this discrete ledger) is

$$\Delta T_c =\Lambda^\top v_{\mathrm{free}} +\frac12\Lambda^\top W\Lambda.$$

Since $b=v_{\mathrm{free}}-v^*$, the objective decomposes as

$$\frac12\Lambda^\top W\Lambda+b^\top\Lambda =\Delta T_c-(v^*)^\top\Lambda.$$

With $v^*=0$ the QP minimizes the contact-induced kinetic-energy change: among the admissible impulses, it removes the most kinetic energy. This is the **maximum-dissipation** reading of the least-constraint projection — the closest admissible outgoing velocity to rest is the one of least outgoing kinetic energy. Restitution and recovery shift that baseline through $-(v^*)^\top\Lambda$, which credits impulse spent producing the requested outward velocity. The restitution coefficient is a speed ratio: in the simple scalar impact, rebound kinetic energy is an $e_i^2$ fraction of incoming kinetic energy. The discrete ledger in §10 evaluates the energy sign for the actual pair of velocity baselines, and penetration recovery requests outward motion and permits positive contact work.

For a scalar frictionless contact these pieces are transparent: with effective mass $m_{\mathrm{eff}}$ and $w=1/m_{\mathrm{eff}}$, the objective is $\tfrac12 w\lambda^2+(v_{\mathrm{free}}-v^*)\lambda$, worked in [Appendix A.1](#a1-one-frictionless-contact-one-number) and [A.2](#a2-scalar-target-three-examples).

---

## 4. The Coulomb feasible set

For contact $i$, the ground impulse obeys

$$\Lambda_{n,i}\ge0, \qquad |\Lambda_{t,i}|\le\mu_i\Lambda_{n,i},$$

where $\mu_i$ is the dimensionless Coulomb friction coefficient: it sets the tangential impulse budget associated with a normal impulse. At $\Lambda_{n,i}=0$, the second inequality gives $\Lambda_{t,i}=0$.

The implementation learns one $\mu_i$ per contact through $\mu_i=2\,\operatorname{sigmoid}(\hat\mu_i)$, where $\hat\mu_i$ is an unconstrained raw parameter, giving $\mu_i\in(0,2)$ and an initialization near $0.8$. Finite $\hat\mu_i$ approach the ideal frictionless value $\mu_i=0$ as their lower limit.

This feasible set is called the planar Coulomb cone:

$$\mathcal C_{\mu_i} =\left\{(\Lambda_n,\Lambda_t): \Lambda_n\ge0, |\Lambda_t|\le\mu_i\Lambda_n \right\}.$$

It is a wedge in two dimensions:

![coulomb_cone](https://hackmd.io/_uploads/SyH55RjEzl.svg)

It is called a cone because multiplying a feasible impulse by any nonnegative number keeps it feasible. Increasing $\mu$ widens the cone and enlarges the available tangential impulse interval.

The cone sets the available friction magnitude. The objective selects the actual tangential impulse and its sign from the interval $[-\mu\Lambda_n,+\mu\Lambda_n]$. An interior optimum corresponds to sticking. A sloped-boundary optimum represents friction saturation, with its velocity residual distinguishing incipient slip from sliding.

In this planar model the absolute-value constraint is equivalent to two linear inequalities:

$$\Lambda_t-\mu\Lambda_n\le0, \qquad -\Lambda_t-\mu\Lambda_n\le0.$$

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

$$\boxed{ \mathcal A(v_{\mathrm{free}}) =v_{\mathrm{free}}+W\mathcal C_\mu =\left\{v_{\mathrm{free}}+W\Lambda: \Lambda\in\mathcal C_\mu\right\}. }$$

Read this expression as a recipe: take every admissible impulse $\Lambda$ from the cone, compute its velocity change $W\Lambda$, and add that change to $v_{\mathrm{free}}$. The resulting collection is $\mathcal A(v_{\mathrm{free}})$.

Exact attainment means

$$\boxed{ v^*\in\mathcal A(v_{\mathrm{free}}) \quad\Longleftrightarrow\quad \exists\Lambda\in\mathcal C_\mu: W\Lambda=v^*-v_{\mathrm{free}}. }$$

The symbol $\in$ means “belongs to,” and $\exists$ means “there is at least one.” Thus the boxed statement says that $v^*$ is reachable exactly when at least one admissible impulse produces the required velocity change.

This condition depends on the request, the free prediction, the contact-space inverse mass $W$, the friction coefficients $\mu_i$, and contact coupling. For invertible $W$, form the required impulse

$$\Lambda_{\mathrm{req}} =W^{-1}(v^*-v_{\mathrm{free}}).$$

Membership $\Lambda_{\mathrm{req}}\in\mathcal C_\mu$ gives exact attainment. The existence equation above supplies the corresponding test for a rank-deficient $W$.

The equations above describe the ideal QP. The target problem in §7 has a unique minimizer; at an active contact whose impulse lies in the strict cone interior, its component relation is $v_i^+-v_i^*=-C_i\Lambda_i$, the physical compliance of §7.2 acting on that contact's impulse. Each point outside the predicted-crossing active set has both impulse components fixed to zero. Section 8 explains the residual from the finite numerical solve.

### 4.2 How $\mu$ controls exact sticking

The tangential request $v_t^*=0$ asks for sticking, and whether the cone can supply it is a ratio test. For a candidate exact-attainment impulse with $\Lambda_{n,\mathrm{req}}>0$, define the required friction ratio

$$\boxed{ \mu_{\mathrm{required}} =\frac{|\Lambda_{t,\mathrm{req}}|} {\Lambda_{n,\mathrm{req}}}. }$$

Comparing it to the available $\mu$ names the outgoing regime:

| relation | cone location | outgoing regime |
|---|---|---|
| $\mu>\mu_{\mathrm{required}}$ | strict interior | exact sticking |
| $\mu=\mu_{\mathrm{required}}$ | sloped boundary with $r=0$ | exact incipient slip |
| $\mu<\mu_{\mathrm{required}}$ | saturated optimum with tangential residual | sliding |

For a single contact with $W=I$ the sticking request has $\Lambda_{\mathrm{req}}=v^*-v_{\mathrm{free}}$, so the ratio reads directly from velocity,

$$\mu_{\mathrm{required}} =\frac{|v_{\mathrm{free},t}|} {v_n^*-v_{\mathrm{free},n}}, \qquad v_n^*-v_{\mathrm{free},n}>0,$$

worked with numbers in [Appendix A.3](#a3-exact-sticking-and-the-friction-ratio). For a general invertible $W$, first compute $\Lambda_{\mathrm{req}}=W^{-1}(v^*-v_{\mathrm{free}})$ and check $\Lambda_{n,i,\mathrm{req}}\ge0$ together with $|\Lambda_{t,i,\mathrm{req}}|\le\mu_i\Lambda_{n,i,\mathrm{req}}$ for every contact. For coupled or rank-deficient $W$, the existence condition in §4.1 gives the complete exact-attainment test.

---

## 5. The ideal contact QP

Collecting the objective of §3 and the cone of §4, the complete ideal problem is

$$\boxed{ \begin{aligned} \underset{\Lambda\in\mathbb R^{2K}}{\operatorname{minimize}} &\quad \frac12\Lambda^\top W\Lambda+b^\top\Lambda\\ \operatorname{subject\ to} &\quad \Lambda_{n,i}\ge0, \qquad |\Lambda_{t,i}|\le\mu_i\Lambda_{n,i}, \quad i=1,\ldots,K. \end{aligned} }$$

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

$$r =W\Lambda+b =v^+-v^*.$$

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

Duplicate or mechanically redundant active contact points can make $W$ singular, meaning several impulse distributions produce the same generalized motion. The positive physical compliance in §7 adds curvature along redundant directions, giving the implemented target problem a well-posed, unique optimum.

### 5.4 The geometric picture: projection onto the cone

For positive-definite $W$, the whole QP reduces to one point and one region. The objective $\tfrac12\Lambda^\top W\Lambda+b^\top\Lambda$ fills impulse space with nested ellipsoidal level sets around a single lowest point, its unconstrained minimizer $-W^{-1}b$. Since $b=v_{\mathrm{free}}-v^*$, that lowest point is

$$-W^{-1}b =W^{-1}(v^*-v_{\mathrm{free}}) =\Lambda_{\mathrm{req}},$$

exactly the required impulse of §4.1. The feasible region is the Coulomb cone $\mathcal C_\mu$ of §4, a wedge with apex at the origin. Solving the QP asks where the smallest objective ellipsoid centered at $\Lambda_{\mathrm{req}}$ first meets that wedge. This is the exact-attainment test of §4.1 read geometrically:

- Required impulse inside the cone, $\Lambda_{\mathrm{req}}\in\mathcal C_\mu$: the solver returns the ellipsoid center and $r=0$; the request is attained, with a strict interior giving sticking and a facet with $r=0$ giving incipient slip.
- Required impulse outside the cone: the optimum lies on the cone boundary with $\lVert r\rVert>0$.

What "boundary point" means is fixed by the objective's shape. Completing the square around $\Lambda_{\mathrm{req}}$,

$$\frac12\Lambda^\top W\Lambda+b^\top\Lambda =\frac12(\Lambda-\Lambda_{\mathrm{req}})^\top W(\Lambda-\Lambda_{\mathrm{req}})+\text{const},$$

so minimizing over the cone minimizes the $W$-weighted distance $(\Lambda-\Lambda_{\mathrm{req}})^\top W(\Lambda-\Lambda_{\mathrm{req}})$. The constrained optimum is the point of the cone closest to $\Lambda_{\mathrm{req}}$ in the metric set by $W$: the tangency point where an objective ellipsoid kisses the wedge. The matrix $W=JM^{-1}J^\top$ shapes these level sets as kinetic-metric ellipsoids. The specialization $W=I$ makes them spherical and yields the ordinary nearest-point projections used in Appendix A. Singular $W$ produces cylindrical level sets along redundant impulse directions and a seminorm projection on its reachable range. The positive compliance $C$ in versions 3–4 makes $H=W_A+C$ positive-definite on active blocks and selects a unique physical-impulse optimum.

That metric is physical. The $W$-distance is a kinetic-energy distance (§3.3), so the constrained optimum is the admissible impulse whose outgoing velocity is closest in kinetic energy to the request. This is the least-constraint projection of §1 with the friction cone as its feasible set.

Where the tangency lands names the mode of §5.2:

| projection target | cone location | selected mode |
|---|---|---|
| ellipsoid center already feasible | strict interior | sticking, $r=0$ on active components |
| sloped facet $\Lambda_t=\pm\mu\Lambda_n$ | boundary | sliding, friction saturated |
| apex $\Lambda=0$ | vertex | separation or inactive contact |

The physical target of §7 keeps this picture on the selected contact blocks with a shifted metric: the active Delassus matrix gains the positive compliance diagonal $C$. The added curvature rounds the level sets along redundant directions and fixes a unique tangency point. Inactive blocks are equality-constrained to zero.

The ADMM cone projection $\Pi_{\mathcal C_\mu}$ of §8 is an ordinary Euclidean projection that supplies feasibility at each inner iteration. The linear solve carries the $W$- or $H$-metric geometry, so convergence yields the metric projection described here.

---

## 6. Complementarity and KKT

This section gives formal names to the optimality conditions that §3.1 already derived from the projection.

### 6.1 Scalar complementarity

For one frictionless contact, define

$$r_n=v_n^+-v_n^*.$$

The optimum satisfies

$$\lambda_n\ge0, \qquad r_n\ge0, \qquad \lambda_n r_n=0.$$

The last condition means at least one of the two quantities must be zero:

- The combination $\lambda_n=0$ and $r_n>0$ describes separation.
- If $\lambda_n>0$, then $r_n=0$ and the requested normal velocity is attained.

This is written compactly as

$$0\le\lambda_n\ \perp\ r_n\ge0.$$

These are the Karush–Kuhn–Tucker, or **KKT**, conditions of the scalar QP. They are the projection's conditions from §3.1 with the multiplier $s$ written as $r_n$: the KKT multiplier and the residual velocity are the same quantity, $r_n=v_n^+-v_n^*$, which is why minimizing manufactures the Signorini law.

### 6.2 Cone complementarity

For the ideal frictional QP, the scalar inequalities become

$$\Lambda\in\mathcal C_\mu, \qquad r\in\mathcal C_\mu^*, \qquad \Lambda^\top r=0,$$

For any cone $\mathcal C$, its **dual cone** is the set of residual directions having a nonnegative dot product with every feasible impulse:

$$\mathcal C^* =\{r:z^\top r\ge0\ \text{for every }z\in\mathcal C\}.$$

For one planar Coulomb cone this becomes

$$\mathcal C_\mu^* =\{(r_n,r_t):r_n\ge\mu|r_t|\}.$$

The dual-cone condition gives a nonnegative directional derivative along every feasible impulse direction. Orthogonality, $\Lambda^\top r=0$, supplies the generalized complementarity condition.

For a sliding contact with $|r_t|>0$, the friction impulse lies on the cone boundary whose tangential sign opposes $r_t$. Orthogonality then gives

$$r_n=\mu|r_t|.$$

Since $r_n=v_n^+-v_n^*$ and $r_t=v_t^+$,

$$v_n^+-v_n^*=\mu|v_t^+|.$$

Thus the ideal joint-cone QP satisfies the sliding relation $v_n^+-v_n^*=\mu|v_t^+|$: the associated model couples a small outward normal velocity to tangential slip.

### 6.3 Compliance selects a unique physical impulse

The ideal matrix $W$ is positive-semidefinite and the cone is convex, so the KKT conditions characterize its convex minimizer set. Singular $W$ can represent one outgoing generalized motion through several redundant rigid impulse distributions. The positive physical compliance of §7 makes the active version-3 and version-4 targets strictly convex for positive finite compliance coefficients, selecting one unique minimizing physical impulse.

The tangential reaction dissipates energy, and the maximum-dissipation reading of §3.3 selects it: among admissible tangential impulses the chosen one removes the most kinetic energy. Folding the normal projection and this tangential dissipation into the single cone objective gives the associated joint-cone problem, exact for the normal, inelastic, and restitution response, and extended by the cone objective to coupled Coulomb friction.

---

## 7. Predicted-crossing activation and physical compliance

The prototype extends the ideal QP with two ingredients:

1. a hard, action-aware active set that admits points already in contact or predicted to cross the plane during the response interval;
2. a learned normal secant-stiffness profile and positive tangential compliance whose QP coefficients are fixed algebraically at each incoming state.

Exact gaps and the endpoint indicator select eligible impulse blocks. Versions 3–4 optimize the physical impulse $\Lambda$ directly.

Write

$$W_{\mathrm{full}}=JM^{-1}J^\top$$

for the Delassus matrix of all candidate points.

### 7.1 Hard predicted-crossing active set

The contact-free generalized velocity at the end of the response interval is

$$\dot q_{\mathrm{free}}=\dot q+h\ddot q_{\mathrm{free}},$$

where $\ddot q_{\mathrm{free}}$ already includes the current action. The solver evaluates the contact velocity with the Jacobian at the current configuration,

$$v_{\mathrm{free}}=J(q)\dot q_{\mathrm{free}},$$

and uses the first-order semi-implicit endpoint proxy

$$g_{i,\mathrm{free}}^+ =g_i(q)+h\,v_{\mathrm{free},n,i}.$$

The implemented active indicator is

$$a_i = \mathbf 1\!\left\{ g_i\le0 \ \text{or}\ g_{i,\mathrm{free}}^+\le0 \right\}.$$

A positive-gap point activates when its contact-free endpoint reaches the plane during this step. The indicator arguments are the gap and free normal velocity. The acceleration and current action can change $v_{\mathrm{free},n}$, so they can change the decision.

The endpoint proxy covers contacts engaged at the step start and free endpoints at or below the plane. Candidate extensions use continuous collision detection for curved within-step crossings and closure passes for secondary contacts induced by coupled impulses. Rollout audits can determine the value of these extensions.

Repeat each indicator over that point's normal and tangential slots:

$$A=\operatorname{diag}(a_1,a_1,\ldots,a_K,a_K).$$

The admissible set is

$$\mathcal C_{\mu,A} = \left\{ \Lambda: \begin{array}{ll} (\Lambda_{n,i},\Lambda_{t,i})\in\mathcal C_{\mu_i}, & a_i=1,\\ (\Lambda_{n,i},\Lambda_{t,i})=(0,0), & a_i=0. \end{array} \right\}.$$

The equality $a_i=0$ fixes both components of every unselected point at exactly zero, including its tangential-friction budget. All active contacts remain in one simultaneous solve, so their off-diagonal Delassus coupling and static-friction competition are preserved.

The indicator is a hard hybrid decision. Gradients propagate almost everywhere within a fixed active set and fixed cone-projection branch with respect to mechanics, action, $k_{\mathrm{low}}$, $R$, $c_t$, $e$, and $\mu$. The fixed $\beta$ and transition width participate as configuration. Activation boundaries and cone-mode boundaries supply discrete hybrid switches.

### 7.2 MuJoCo-inspired secant stiffness and tangential compliance

Version 4 reads the incoming geometric penetration and maps it through a fixed reflected-quadratic transition:

$$\delta_i=[-g_i]_+, \qquad u_i=\operatorname{clip}\!\left(\frac{\delta_i}{w},0,1\right), \qquad S(u_i)=\begin{cases}2u_i^2,&0\le u_i\le\tfrac12,\\1-2(1-u_i)^2,&\tfrac12<u_i\le1,\end{cases} \qquad w=10^{-3}\ \mathrm m.$$

The one-millimetre depth scale and symmetric power-two shape follow MuJoCo's default `solimp` transition. This prototype applies that shape directly to effective secant stiffness. Each point learns a positive shallow stiffness, and every point shares one learned high-to-low ratio:

$$k_{\mathrm{low},i}=\exp(\hat k_i), \qquad R=1+\operatorname{softplus}(\hat R), \qquad K_i(\delta_i)=k_{\mathrm{low},i}\left[1+(R-1)S(u_i)\right], \qquad k_{\mathrm{high},i}=Rk_{\mathrm{low},i}.$$

Here $K_i(\delta_i)=F_{n,i}(\delta_i)/\delta_i$ is the secant stiffness of the designed resting-load normal force curve in N/m, with its finite limit defining the value at $\delta_i=0$. The shallow plateau supplies $K_i(0)=k_{\mathrm{low},i}$, the transition rises smoothly and monotonically, and depths $\delta_i\ge w$ use $K_i(\delta_i)=k_{\mathrm{high},i}$.

The position-level recovery request is $\beta\delta_i/h$. An impulse $\Lambda_{n,i}=F_{n,i}h$ acts through normal velocity-level compliance $c_{n,i}$. The resting loaded equilibrium $F_{n,i}=K_i(\delta_i)\delta_i$ fixes the coefficient:

$$\frac{\beta\delta_i}{h}=c_{n,i}(\delta_i)\Lambda_{n,i}=c_{n,i}(\delta_i)F_{n,i}h \quad\Longrightarrow\quad \boxed{c_{n,i}(\delta_i)=\frac{\beta}{K_i(\delta_i)h^2}.}$$

The two factors of $h$ convert position error to target velocity and force to impulse. The $h^2$ denominator therefore preserves the same position-level force curve across response steps.

Version 4 learns an independent positive tangential compliance directly in $1/\mathrm{kg}$:

$$c_{t,i}=\exp(\hat c_{t,i}), \qquad C_i(\delta_i)=\operatorname{diag}\!\left(c_{n,i}(\delta_i),c_{t,i}\right), \qquad C(\delta)=\operatorname{blockdiag}\!\left(C_1(\delta_1),\ldots,C_K(\delta_K)\right).$$

Its default initialization equals the shallow normal compliance, $c_{t,i}^{\mathrm{init}}=\beta/(k_{\mathrm{low},i}^{\mathrm{init}}h^2)$. At a strict cone-interior tangent row, the zero tangential target gives

$$v_{t,i}^+=-c_{t,i}\Lambda_{t,i}=-c_{t,i}hF_{t,i}.$$

Thus $c_{t,i}$ directly controls impulse-to-velocity softness. The equivalent frozen-step velocity-force scale is $D_{t,i}=1/(c_{t,i}h)$ in kg/s. A model with tangential displacement memory would give this scale a stored-shear spring interpretation.

The incoming penetration fixes $C(\delta)$ before each solve, so every response remains a convex QP. For each active block at a strict cone-interior optimum,

$$v_i^+-v_i^*=-C_i(\delta_i)\Lambda_i, \qquad a_i=1.$$

For a resting loaded contact, the normal row gives the designed nonlinear force curve:

$$F_{n,i}=\frac{\Lambda_{n,i}}h=K_i(\delta_i)\delta_i.$$

For an unloaded penetrated free mass, the active interior stationarity equation $(W_A+C)_{\mathcal I\mathcal I}\Lambda_{\mathcal I}=-b_{A,\mathcal I}$ jointly sets the impulse through inertia and compliance, where $\mathcal I$ collects active normal–tangential slots. Repeated responses carry the mass toward the loaded equilibrium.

Version 3 is the historical constant profile $K_i(\delta_i)=k_i$ with a tied diagonal $C_i=\beta/(k_i h^2)I_2$. Supplying `contact_stiffness_ratio` selects version 4. The historical `contact_solver="compliant"` branch directly evaluates the state-based Hunt–Crossley penalty force $F=k\delta(1+\alpha\dot\delta)$.

Two parameter semantics matter:

- The log and softplus parameterizations give $k_{\mathrm{low},i}>0$, $R>1$, and $c_{t,i}>0$. Large stiffness and small compliance weaken uniform curvature, so parameter bounds or priors can maintain production conditioning.
- The structured mechanics retain a shared scale gauge: $M$, $V$, $D$, $G_a$, $K$, and generalized contact magnitudes may scale together by $\alpha$, with compliance scaling by $1/\alpha$. Independently calibrated force, mass, or stiffness data anchors this dimensional scale. The parameters and equations retain their physical units; identifiability is a separate property.

### 7.3 The target optimization problem

Mask the virtual-work map and bias,

$$J_A=AJ, \qquad W_A=\operatorname{sym}(A W_{\mathrm{full}} A), \qquad b_A=Ab,$$

and define

$$H=W_A+C.$$

The implemented target is directly in physical impulse:

$$\boxed{ \begin{aligned} \underset{\Lambda\in\mathbb R^{2K}}{\operatorname{minimize}} &\quad \frac12\Lambda^\top H\Lambda+b_A^\top\Lambda\\ \operatorname{subject\ to} &\quad \Lambda\in\mathcal C_{\mu,A}. \end{aligned} }$$

For $a_i=0$, the equality $\Lambda_i=0$ gives $C_i$ zero effect on the physical solution. For active blocks, finite $K_i(\delta_i)>0$, finite $c_{t,i}>0$, and fixed $\beta>0$ make $C_i$ positive definite, so the active objective is strictly convex. The dense masked statement is equivalent at convergence to forming a reduced QP from active blocks and scattering exact zeros back into the full contact vector.

The simultaneous solve makes the exact active solution invariant to the number of zero blocks. With a finite ADMM budget, applying the zero-block equality at every projection keeps the auxiliary zero-block coordinates fixed throughout the iteration and guarantees the returned blockwise feasibility.

## 8. How the QP is solved: ADMM at a high level

The alternating direction method of multipliers, or **ADMM**, separates two easy operations:

1. minimize an unconstrained quadratic;
2. project a trial physical impulse onto the active Coulomb set.

### 8.1 Two copies of the physical impulse

Restate the problem as

$$\min_{\Lambda,z} \frac12\Lambda^\top H\Lambda+b_A^\top\Lambda +\iota_{\mathcal C_{\mu,A}}(z) \qquad\text{subject to}\qquad \Lambda=z.$$

The primal copy $\Lambda$ carries the quadratic bowl. The auxiliary copy $z$ carries the friction-cone and inactive-block constraints. A scaled dual variable $\xi$ accumulates their disagreement.

This split is purely algorithmic. Both copies carry physical-impulse units throughout versions 3–4 and represent the same physical impulse at convergence.

### 8.2 The fixed iteration

For a sample with at least one active slot, the implementation chooses an ADMM penalty from active target curvature,

$$\rho = \max\!\left( \frac{\sum_j A_{jj}H_{jj}} {\sum_j A_{jj}}, 10^{-6} \right),$$

detached from differentiation. If every point is inactive, it uses the detached mean diagonal scale of $W_{\mathrm{full}}$ as a harmless fallback. The diagonal $C$ defines physical compliance, $H=W_A+C$ defines target curvature, and $\rho$ supplies an algorithmic curvature scale for the linear system. Every positive $\rho$ shares the same converged target minimizer; its value shapes the approximation returned by the 12-iteration budget.

Starting from zero primal, auxiliary, and dual vectors, one iteration is

$$\Lambda^{k+1} =(H+\rho I)^{-1} \left[-b_A+\rho(z^k-\xi^k)\right],$$

followed by over-relaxation,

$$\widehat\Lambda^{k+1} =1.5\Lambda^{k+1}-0.5z^k,$$

active-set cone projection,

$$z^{k+1} = \Pi_{\mathcal C_{\mu,A}} (\widehat\Lambda^{k+1}+\xi^k) = A\,\Pi_{\mathcal C_\mu} (\widehat\Lambda^{k+1}+\xi^k),$$

and the dual update

$$\xi^{k+1} = \xi^k+\widehat\Lambda^{k+1}-z^{k+1}.$$

The same binary indicator multiplies both slots of a point. Consequently, projection is the usual closed-form planar Coulomb projection for active blocks and exact zero for inactive blocks. The default finite budget performs 12 iterations and returns $z^{12}$ as the physical impulse.

For each active planar contact, ordinary cone projection has three cases:

- a trial pair inside the cone maps to itself;
- a trial pair in the polar region maps to the cone apex;
- every remaining pair maps to the nearest sloped cone boundary.

These operations provide two structural guarantees for every auxiliary iterate $z^k$ and the returned $z^{12}$:

- $z_{n,i}^k\ge0$ and $|z_{t,i}^k|\le\mu_i z_{n,i}^k$ for active points;
- $z_{n,i}^k=z_{t,i}^k=0$ bit-for-bit for inactive points.

Finite-budget optimality is measured with the similarly masked projected-gradient residual. With $Q(z)=\tfrac12z^\top Hz+b_A^\top z$ and $L$ the row-sum bound used in code,

$$\frac{ \left\lVert z-\Pi_{\mathcal C_{\mu,A}} \left(z-\nabla Q(z)/L\right) \right\rVert }{ 1+\lVert z\rVert }$$

is zero at a projected-gradient fixed point. The fixed iteration count gives rollout backpropagation a fixed computation graph. Gradients propagate almost everywhere through the dense solve and cone projection within each fixed active set and cone-projection branch. Activation and cone-mode boundaries supply the discrete hybrid selections.

## 9. Applying the solution

Let $\Lambda_{\mathrm{ret}}=z^N$ denote the returned feasible physical impulse after the configured $N$ ADMM iterations. It feeds directly into the generalized momentum map.

The generalized momentum change and outgoing velocity are

$$\Delta p=J^\top\Lambda_{\mathrm{ret}},$$

$$\dot q^+ = \dot q_{\mathrm{free}} +M^{-1}J^\top\Lambda_{\mathrm{ret}}.$$

For use in the model drift, the equivalent average generalized force and acceleration contribution are

$$F_c=\frac{J^\top\Lambda_{\mathrm{ret}}}{h}, \qquad \ddot q_c=M^{-1}F_c.$$

The internal response horizon defaults to $h=0.002\ \mathrm{s}$ and is persisted in version-3 and version-4 sidecars. The displayed $\dot q^+$ is the full-$h$ hypothetical response used by the contact solve and diagnostics. A surrounding Euler step of duration $\Delta\tau<h$ applies the equivalent force for $\Delta\tau$, producing the fraction $\Delta\tau/h$ of that response, and then recomputes contact. Longer transitions use repeated small steps with a fresh active-set decision and contact calculation at each step.

## 10. Discrete energy accounting

Virtual work verifies that the kinematic and force maps agree:

$$\dot q^\top J^\top\frac{\Lambda}{h} =(J\dot q)^\top\frac{\Lambda}{h}.$$

This identity establishes work consistency between generalized and contact coordinates. For the solver's frozen-geometry, full-$h$ hypothetical response, the discrete kinetic-energy change is exactly

$$\boxed{ \Delta T_c =T^+-T_{\mathrm{free}} =\Lambda^\top v_{\mathrm{free}} +\frac12\Lambda^\top W_{\mathrm{full}}\Lambda. }$$

The second term accounts for the velocity change produced by the impulse. For a surrounding integration step $\Delta\tau<h$, the boxed quantity remains the full-$h$ solver-response diagnostic; the shorter Euler update uses its scaled impulse response for its own energy change.

For the ideal unbiased QP, this expression is the objective itself. With restitution and recovery, the optimized objective is shifted by $-(v^*)^\top\Lambda$, the decomposition displayed in §3.3. The boxed expression remains the exact full-$h$ response ledger used by the contact diagnostics.

In the ideal unbiased problem, maximum-dissipation friction gives $\Delta T_c\le0$. In a scalar or decoupled impact, restitution with a consistent current/free velocity baseline and $0\le e_i\le1$ also gives $\Delta T_c\le0$. The displayed ledger supplies the energy sign for a coupled multi-contact response and its actual velocity baselines. Penetration recovery requests outward motion and permits positive contact work. The audit reports the nonnegative recovery attribution

$$\sum_i\beta d_i\Lambda_{n,i}$$

as `stabilization_work`. The full-$h$ response quantity `discrete_work` is the total contact-work ledger, combining impact, friction, restitution, and recovery.

---

## 11. End-to-end recipe

At each internal step, solver version 4 performs the following calculation:

1. Evaluate the exact contact-point kinematics, giving metric gaps $g_i(q)$, $J_n$, and $J_t$.
2. Evaluate $M(q)$ and the contact-free acceleration, including the current action.
3. Predict $\dot q_{\mathrm{free}}=\dot q+h\ddot q_{\mathrm{free}}$ and $v_{\mathrm{free}}=J\dot q_{\mathrm{free}}$.
4. Form $g_{i,\mathrm{free}}^+=g_i+h v_{\mathrm{free},n,i}$.
5. Activate points with $g_i\le0$ or $g_{i,\mathrm{free}}^+\le0$; repeat the mask across each normal–tangential block.
6. Use $v_{n,i}^*=-g_i/h$ for positive-gap crossings, or the restitution/recovery request for already engaged points; set $v_{t,i}^*=0$.
7. Read the fixed persisted $\beta$ and transition width, evaluate $k_{\mathrm{low},i}$, $R$, $S(u_i)$, $K_i(\delta_i)$, $c_{n,i}(\delta_i)$, and $c_{t,i}$, then assemble $C_i(\delta_i)=\operatorname{diag}(c_{n,i},c_{t,i})$.
8. Build $H=\operatorname{sym}(A W_{\mathrm{full}}A)+C$ and $b_A=Ab$.
9. Approximate the physical-impulse QP with the fixed ADMM budget, applying the inactive equality after every cone projection.
10. Apply $J^\top\Lambda$ to generalized momentum, or $J^\top\Lambda/h$ as its average-force equivalent.
11. Report the active mask, predicted free gap, shallow and high stiffness plateaus, shared ratio, effective stiffness, both compliance slots, cone feasibility, optimality residual, outgoing velocities, and discrete energy ledger.

### 11.1 Selecting the prototype

The model constructor selects version 4 explicitly:

```python
PortHamiltonianModel(
    17,
    6,
    mode="structured",
    contact_force=6,
    contact_solver="constraint",
    contact_geometry="kinematic",
    contact_stiffness=100_000.0,
    contact_attenuation=0.2,
    contact_stiffness_ratio=2.228395061728395,
    contact_tangent_compliance=0.5,
    contact_dt=0.002,
)
```

The benchmark table exposes the same configuration as `mbq_structured_quad_stiffness_curve_roll`. The recovery evaluator accepts `--contact_geometry kinematic --contact_stiffness 100000 --contact_attenuation 0.2 --contact_stiffness_ratio 2.228395061728395 --contact_tangent_compliance 0.5`.

The initial ratio $2.228395061728395$ evaluates MuJoCo's direct-stiffness resting-load scale $d^2/(1-d)$ at its default shallow and deep impedance plateaus, $d_0=0.9$ and $d_{\mathrm{width}}=0.95$. It initializes the shared learned ratio and remains free to move during fitting.

Choose one material formulation. Version 4 pairs `contact_stiffness`, `contact_stiffness_ratio`, kinematic geometry, and the constraint solver. `contact_stiffness` initializes each shallow plateau. `contact_stiffness_ratio` initializes the shared high-to-low ratio and activates the depth curve. `contact_tangent_compliance` initializes each learned tangential coefficient in $1/\mathrm{kg}$; its default equals the initial shallow normal compliance. Version 3 uses `contact_stiffness` with the ratio omitted. The historical gated formulation uses `contact_compliance` and `contact_gate_off`. Constructor validation enforces these matching combinations.

`contact_attenuation` is the fixed desired penetration-recovery fraction for the
version-3 and version-4 `contact_stiffness` laws. Its valid interval is `(0, 1]`, and its default
is `0.2`. This value sets the recovery component of $v_n^*$ to $0.2\delta/h$,
corresponding to 80% retained penetration under exact uncoupled target attainment.

## 12. Key properties

| property | role in the contact solve |
|---|---|
| least-constraint projection | the outgoing motion is the admissible motion closest to the free motion in the kinetic metric |
| exact geometry | signed gaps and both Jacobian rows come from the declared kinematic tree |
| predicted crossing | eligibility includes engagement now and an action-aware free endpoint at or below the plane |
| hard inactive equality | both impulse slots of every unselected point equal zero, placing eligible ground-response impulse on selected points |
| requested velocity | positive-gap crossings target the plane; engaged contacts request restitution and the fixed desired penetration-recovery fraction $\beta$; tangential velocity targets zero |
| physical impulse | $\Lambda$ transfers momentum directly from the QP into generalized coordinates |
| physical normal stiffness | $k_{\mathrm{low},i}$ and shared $R$ define the monotone profile $K_i(\delta_i)$; the normal static equilibrium is $F_n=K_i(\delta_i)\delta_i$ |
| fixed transition geometry | a persisted one-millimetre reflected quadratic maps incoming penetration to the shallow-to-high stiffness interpolation |
| tangential compliance | each learned $c_{t,i}>0$ controls tangential impulse-to-velocity softness independently of the normal material curve |
| fixed recovery request | $\beta=0.2$ by default is persisted configuration, shared across points, and external to the optimizer |
| timestep semantics | $h^2$ preserves position-level stiffness because target speed scales as $1/h$ and impulse as $h$ |
| simultaneous contacts | $A W_{\mathrm{full}}A$ retains all coupling among active points |
| static friction | the joint normal–tangential cone solve selects an impulse that corrects action-induced slip; independent tangential compliance supplies the constitutive residual |
| cone projection | every returned active block is feasible, and every inactive block is bitwise zero |
| algorithmic penalty | ADMM $\rho$ conditions the iteration; $H$ remains the target-QP Hessian |
| hybrid gradient | gradients propagate almost everywhere within fixed activation and cone-projection branches; their boundaries supply discrete mode switches |
| checkpoint semantics | each checkpoint retains its solver-version semantics; version 4 persists shallow log-stiffness, shared ratio, tangential log-compliance, fixed recovery fraction, response horizon, transition width, and exact kinematic buffers |
| identifiability | the shared mechanics/contact scale gauge remains until dimensional data anchors it |
| candidate extensions | future continuous collision detection would extend coverage to within-step crossings, and future activation closure would extend coverage to contact-coupled secondary impacts |

---

## Appendix A. Worked examples

These instances put numbers on the ideal problem of §§3–5. The $W=I$ cases make the arithmetic transparent: a unit normal or tangential impulse changes the matching velocity component by one unit, and the off-diagonal couplings vanish, so the $W$-metric projection of §5.4 reads as the ordinary nearest point.

### A.1 One frictionless contact, one number

Take upward velocity as positive, a predicted normal velocity $v_{\mathrm{free}}=-1\ \mathrm{m/s}$ (closing), and effective mass $m_{\mathrm{eff}}=2\ \mathrm{kg}$. A positive normal impulse $\lambda$ changes the velocity by $\lambda/m_{\mathrm{eff}}$:

$$v^+=v_{\mathrm{free}}+\frac{\lambda}{m_{\mathrm{eff}}} =-1+\frac{\lambda}{2}, \qquad \lambda\ge0.$$

With the inelastic request $v^*=0$, the projection objective of §3 is

$$\min_{\lambda\ge0} Q(\lambda) =\frac{1}{2m_{\mathrm{eff}}}\lambda^2 +v_{\mathrm{free}}\lambda =\frac14\lambda^2-\lambda.$$

Differentiating,

$$Q'(\lambda)=\frac12\lambda-1=0 \quad\Longrightarrow\quad \lambda^*=2,$$

so $v^+=0$: inelastic closure. For a separating prediction $v_{\mathrm{free}}=+0.4\ \mathrm{m/s}$,

$$Q'(\lambda)=\frac{\lambda}{m_{\mathrm{eff}}}+0.4>0 \qquad\text{for every }\lambda\ge0,$$

so the score increases along the feasible half-line and its minimum is the boundary point $\lambda^*=0$. In general,

$$\lambda^* =\max\!\left(0,-m_{\mathrm{eff}}v_{\mathrm{free}}\right),$$

$$\begin{array}{lll} v_{\mathrm{free}}\ge0 &\Longrightarrow& \lambda^*=0,\quad v^+=v_{\mathrm{free}},\\[3pt] v_{\mathrm{free}}<0 &\Longrightarrow& \lambda^*=-m_{\mathrm{eff}}v_{\mathrm{free}},\quad v^+=0. \end{array}$$

The linear term $v_{\mathrm{free}}\lambda$ lowers the score as a positive impulse grows; the quadratic term bends it upward; the balance is the impulse that cancels the closing velocity. The kinetic-energy identity of §3.3 gives $T^+-T_{\mathrm{free}}=Q(\lambda)$ here exactly.

### A.2 Scalar target: three examples

With the shifted objective $\min_{\lambda\ge0}\tfrac12w\lambda^2+(v_{\mathrm{free}}-v^*)\lambda$ and $w=1/m_{\mathrm{eff}}$, the solution is

$$\lambda^* =\max\!\left(0,\frac{v^*-v_{\mathrm{free}}}{w}\right),$$

and the impulse required for exact attainment is $\lambda_{\mathrm{req}}=(v^*-v_{\mathrm{free}})/w$. Since $w>0$ and $\lambda\ge0$, the scalar target is exactly attainable precisely when

$$\boxed{v^*\ge v_{\mathrm{free}}.}$$

Using $m_{\mathrm{eff}}=2\ \mathrm{kg}$, $w=0.5\ \mathrm{kg}^{-1}$:

| free velocity $v_{\mathrm{free}}$ | request $v^*$ | required impulse $\lambda_{\mathrm{req}}$ | selected outcome |
|---:|---:|---:|---|
| $-1.0$ | $0$ | $2.0\ \mathrm{N\,s}$ | $\lambda^*=2.0$, $v^+=0$: inelastic closure |
| $-1.0$ | $+0.3$ | $2.6\ \mathrm{N\,s}$ | $\lambda^*=2.6$, $v^+=+0.3$: rebound |
| $+0.4$ | $0$ | $-0.8\ \mathrm{N\,s}$ | $\lambda^*=0$, $v^+=+0.4$: separation |

The scalar solution satisfies the three optimality relations of §6.1,

$$v^+\ge v^*, \qquad \lambda\ge0, \qquad \lambda(v^+-v^*)=0.$$

A positive impulse attains $v^+=v^*$; a positive residual $v^+-v^*>0$ selects $\lambda=0$ and carries the free separating velocity into the outgoing state.

### A.3 Exact sticking and the friction ratio

Take one contact with $W=I$ and

$$v_{\mathrm{free}}=(-1,0.4), \qquad v^*=(0,0).$$

The impulse required to realize the request is

$$\Lambda_{\mathrm{req}} =v^*-v_{\mathrm{free}} =(1,-0.4).$$

Its normal component is $1\ \mathrm{N\,s}$ and its tangential magnitude is $0.4\ \mathrm{N\,s}$. With $\mu=0.5$, the tangential budget is $\mu\Lambda_{n,\mathrm{req}}=0.5$. The required impulse lies strictly inside the cone because $0.4<0.5$, so the ideal QP selects $\Lambda^*=(1,-0.4)$ and reaches $v^*=(0,0)$ exactly: the point sticks. The required friction ratio of §4.2 is $\mu_{\mathrm{required}}=|{-0.4}|/1=0.4<\mu$.

### A.4 A sliding contact

Take one contact with normalized $W=I$, restitution and recovery parameters $e=\beta=0$, and

$$v_{\mathrm{free}}=(-1,2), \qquad \mu=0.5.$$

The impulse required to make both velocity components zero is $\Lambda_{\mathrm{req}}=-v_{\mathrm{free}}=(1,-2)$, with required friction ratio $\mu_{\mathrm{required}}=|{-2}|/1=2$. The available coefficient is $\mu=0.5$, so $|{-2}|>0.5(1)$ and the optimum saturates friction.

The cone-constrained optimum lies on the sliding boundary $\Lambda_t=-0.5\Lambda_n$. Write $n=\Lambda_n$, so $\Lambda_t=-0.5n$. Along that boundary the objective becomes

$$\begin{aligned} Q(n) &=\frac12\left(n^2+(-0.5n)^2\right)-n+2(-0.5n)\\ &=0.625n^2-2n, \end{aligned}$$

with derivative $Q'(n)=1.25n-2$. Setting it to zero gives $n=1.6$, hence $\Lambda_t=-0.8$:

$$\Lambda^*=(1.6,-0.8), \qquad v^+ =v_{\mathrm{free}}+\Lambda^* =(0.6,1.2).$$

Friction reduces the tangential speed from $2$ to $1.2$, leaving residual sliding. The joint cone formulation also produces positive outward normal velocity: it spends extra normal impulse to unlock more tangential friction capacity and balances both effects in the same objective. The resulting sliding relation $v_n^+-v_n^*=\mu|v_t^+|$ of §6.2 couples the outward normal residual to the tangential residual.

---

## Appendix B. Minimal optimization vocabulary

An optimization problem has the general form

$$\min_{x\in\mathcal F} f(x).$$

| term | meaning in plain language | contact example |
|---|---|---|
| decision variable | the number or vector the solver is allowed to choose | the impulse $\Lambda$ |
| objective | the score the solver tries to make as small as possible | contact-induced energy change, with target shifts |
| constraint | a rule every allowed answer must obey | $\Lambda_n\ge0$ |
| feasible set $\mathcal F$ | all choices satisfying every constraint | the Coulomb friction cone |
| optimum or minimizer | a feasible choice with the lowest objective | $\Lambda^*$ |

Here **program** is optimization terminology for “optimization problem.”

**What makes it a quadratic program.** A quadratic program, abbreviated **QP**, has an objective of the form $\tfrac12x^\top Hx+c^\top x$ and linear constraints on $x$. The contact objective has exactly this form, with the impulse as $x$. The product $x^\top Hx$ evaluates to a scalar, the multi-dimensional analogue of a term such as $w x^2$. When $H$ is symmetric, the gradient — the vector version of a derivative — is

$$\nabla_x\left(\frac12x^\top Hx+c^\top x\right)=Hx+c.$$

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
| $g_i(q)=p_{i,z}-\varrho_i$ | signed gap to the ground plane |
| $J$ | stacked normal–tangential contact Jacobian |
| $v_{\mathrm{free}}$ | action-aware contact-free predicted contact velocity |
| $g_{i,\mathrm{free}}^+=g_i+h v_{\mathrm{free},n,i}$ | implemented semi-implicit free endpoint gap proxy |
| $a_i$ | binary engaged-or-predicted-crossing indicator |
| $A$ | diagonal active mask, with $a_i$ repeated over the point's two slots |
| $v^*$ | requested outgoing contact velocity |
| $b=v_{\mathrm{free}}-v^*$ | ideal QP linear term |
| $b_A=Ab$ | active-set linear term |
| $W_{\mathrm{full}}=JM^{-1}J^\top$ | full Delassus matrix |
| $W_A=\operatorname{sym}(A W_{\mathrm{full}}A)$ | active masked Delassus matrix |
| $\Lambda$ | physical normal–tangential impulse |
| $\mu_i$ | Coulomb friction coefficient |
| $e_i$ | state-independent learned restitution ratio |
| $\beta$ | fixed desired penetration-recovery fraction, shared across points and persisted by versions 3–4 |
| $\delta_i=[-g_i]_+$ | incoming geometric penetration depth |
| $u_i=\operatorname{clip}(\delta_i/w,0,1)$ | normalized depth in the fixed transition |
| $S(u_i)$ | reflected-quadratic transition value |
| $w=10^{-3}\ \mathrm m$ | fixed persisted transition width |
| $k_{\mathrm{low},i}=\exp(\hat k_i)$ | learned shallow normal secant stiffness in N/m |
| $R=1+\operatorname{softplus}(\hat R)$ | learned shared high-to-low stiffness ratio |
| $K_i(\delta_i)=k_{\mathrm{low},i}[1+(R-1)S(u_i)]$ | effective normal secant stiffness in N/m |
| $k_{\mathrm{high},i}=Rk_{\mathrm{low},i}$ | high normal stiffness plateau in N/m |
| $c_{n,i}=\beta/(K_i(\delta_i)h^2)$ | normal impulse-to-velocity compliance in $1/\mathrm{kg}$ |
| $c_{t,i}=\exp(\hat c_{t,i})$ | learned tangential impulse-to-velocity compliance in $1/\mathrm{kg}$ |
| $C=\operatorname{blockdiag}(\operatorname{diag}(c_{n,i},c_{t,i}))$ | normal–tangential physical compliance |
| $H=W_A+C$ | version-3 and version-4 target QP Hessian |
| $\mathcal C_{\mu,A}$ | Coulomb cones on active blocks and exact zeros on inactive blocks |
| $z^N$ | feasible physical impulse returned after $N$ ADMM iterations |
| $\rho$ | algorithmic ADMM penalty used in the iterative linear subproblem |
| $h$ | fixed contact response interval, persisted by versions 3–4 |

---

## Related notes and code

- [Structured dynamics model](structured_dynamics_model.md), especially §2.7 and Appendix B.3.
- [Cheetah instantiation](structured_dynamics_cheetah.md), which explains why locomotion needs the explicit ground-reaction route.
- [Hamiltonian recovery report](hamiltonian_recovery_report.md), which defines the contact solver, cone, work, and recovery diagnostics.
- [PortHamiltonianModel._constraint_contact_solve](../models/port_hamiltonian.py), the predicted-crossing secant-stiffness implementation.
- [tests/test_predicted_crossing_contact.py](../tests/test_predicted_crossing_contact.py), focused activation, exact-zero unselected impulse, two-plateau stiffness, independent tangential compliance, timestep, gradient, and checkpoint tests.
