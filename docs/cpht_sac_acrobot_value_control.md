# PH-Guided Continuous-Time Actor-Critic for Acrobot Swing-Up and Stabilization

## Scope and terminology

This document formulates a continuous-time, value-based controller for the Acrobot. A port-Hamiltonian (PH) model predicts how elbow torque changes the state. A learned return value supplies the long-term objective. The controller combines their derivatives to select a bounded scalar torque.

The intended task is to swing up, capture the upright configuration, and remain balanced using one learned feedback policy. The method does not require a prescribed stabilizer, a certified switching region, Casimir invariants, or IDA-PBC matching equations. Those are possible additions with separate purposes and assumptions.

The formulation distinguishes three objects throughout:

| Object | Symbol | Purpose |
|---|---|---|
| Physical stored energy | $E$ or its model $E_\phi$ | Describes plant dynamics and energy exchange. |
| Reward rate and return value | $r$, $V_\psi$ | Specify the task and estimate future reward. |
| A stability certificate, if separately constructed | $W$ | Establishes decrease along a particular closed loop on a particular region. |

Neither the physical energy nor the learned value is automatically a Lyapunov function for upright stabilization. A reward constructed from a Lyapunov candidate is still a training objective, not a controller or a transferred stability proof.

The deterministic core is PH-guided Hamilton–Jacobi–Bellman (HJB) value iteration. Section 3.5 adds the entropy-regularized policy needed for a soft actor-critic interpretation. This distinction avoids calling a deterministic value-gradient controller SAC. Continuous-time advantage-rate learning and Hamiltonian value updates motivate the formulation [3, 4]; maximum-entropy actor-critic methods motivate its soft extension [5, 6]. The combination below is a proposed Acrobot method, not a claim that an existing theorem proves its neural implementation converges.

---

## 1. Acrobot environment

### 1.1 Configuration, state, and input

The shoulder is passive and the elbow is actuated. Let $q_1$ be measured from the hanging-down direction and $q_2$ be the elbow angle relative to link 1. Write

$$q=(q_1,q_2),\qquad v=\dot q,\qquad x=(q,v).$$

The hanging and upright configurations are

$$q_{\mathrm{down}}=(0,0),\qquad q^\star=(\pi,0),\qquad v^\star=0.$$

Angles are interpreted modulo $2\pi$ for an unrestricted-joint Acrobot. An instrument using a shoulder angle measured from the horizontal must first apply $q_1=q_{1,\mathrm{horizontal}}+\pi/2$. Joint travel limits, if present, require an explicit bounded configuration domain instead.

The sole physical input is elbow torque:

$$u\in\mathcal U=[-u_{\max},u_{\max}],\qquad B=\begin{bmatrix}0\\1\end{bmatrix}.$$

For a normalized policy action, use $a\in[-1,1]$ and $u=u_{\max}a$. Report $u_{\max}$ in physical units rather than treating the normalized action range as the mechanical torque limit.

### 1.2 Mechanical dynamics

The equations are

$$M(q)\dot v+c(q,v)+\nabla_q U(q)+D(q)v=Bu,$$

where $D(q)=D(q)^\top\succeq0$. A standard parameterization is

$$M(q)=\begin{bmatrix}
a_1+a_2+2a_3\cos q_2 & a_2+a_3\cos q_2\\
a_2+a_3\cos q_2 & a_2
\end{bmatrix},$$

$$c(q,v)=a_3\sin q_2\begin{bmatrix}-2v_1v_2-v_2^2\\v_1^2\end{bmatrix},$$

$$U(q)=-b_1\cos q_1-b_2\cos(q_1+q_2),$$

with

$$a_1=I_1+m_1l_{c1}^2+m_2l_1^2,\quad
a_2=I_2+m_2l_{c2}^2,\quad a_3=m_2l_1l_{c2},$$

$$b_1=(m_1l_{c1}+m_2l_1)g,\qquad b_2=m_2l_{c2}g.$$

Here $I_i$ are inertias about the link centers of mass. Physical parameters must give $M(q)\succ0$ over the operating domain. The Acrobot's coupled swing-up dynamics and energy-based control have extensive precedents [1, 2].

A concrete nominal instance is

$$m_1=m_2=1,\quad l_1=1,\quad l_2=2,\quad
l_{c1}=0.5,\quad l_{c2}=1,\quad I_1=0.083,\quad I_2=0.33,\quad g=9.8,$$

which gives $(a_1,a_2,a_3)=(1.333,1.330,1.000)$ and $(b_1,b_2)=(14.7,9.8)$ in consistent SI units. Use $D=0$ as a nominal case and separately declared damping values as perturbations. The formulation leaves the torque bound, sample interval, and reward weights as experimental parameters; no feasibility or performance result is assumed for unspecified settings.

### 1.3 Canonical PH representation

Introduce mechanical momentum and canonical state:

$$p=M(q)v,\qquad z=(q,p),\qquad
E(q,p)=\tfrac12p^\top M(q)^{-1}p+U(q).$$

Then

$$\dot z=(J-R(q))\nabla_zE(z)+G u,$$

$$J=\begin{bmatrix}0&I_2\\-I_2&0\end{bmatrix},\qquad
R(q)=\begin{bmatrix}0&0\\0&D(q)\end{bmatrix},\qquad
G=\begin{bmatrix}0\\0\\0\\1\end{bmatrix}.$$

In particular,

$$\dot q=M^{-1}p,\qquad
\dot p=-\nabla_q E-DM^{-1}p+Bu,$$

where $\nabla_q E$ is evaluated at fixed $p$. This derivative includes the configuration dependence of kinetic energy; it is not just $\nabla_q U$.

The physical energy balance is

$$\dot E=-v^\top Dv+u v_2.$$

This identity describes power exchange. Physical energy must increase during parts of swing-up, and upright is a maximum of the gravitational potential. Decreasing physical energy is therefore not the task objective.

### 1.4 Task completion and physical time

Define a small evaluation set

$$\mathcal X_G=\{(q,v):
|\operatorname{wrap}(q_1-\pi)|\le\epsilon_1,
|\operatorname{wrap}(q_2)|\le\epsilon_2,
|v_i|\le\epsilon_{v_i}\}.$$

Success requires entry followed by uninterrupted residence for a declared duration $T_{\mathrm{hold}}$. Entering this geometric set is an evaluation event, not proof of invariance. Continue the episode after first entry so that loss of balance affects training and evaluation.

Actions are held for $h_k=t_{k+1}-t_k>0$. Begin with a fixed interval, then test a declared range of intervals. Physical time $t$ and the value-iteration time $\tau$ introduced later are different quantities.

---

## 2. Objective: swing-up, capture, and balance

### 2.1 A fixed upright objective

Define a nonnegative state cost using physical velocity:

$$\ell(z)=w_1(1+\cos q_1)+w_2(1-\cos q_2)
+w_{v1}\left(\frac{v_1}{\omega_1}\right)^2
+w_{v2}\left(\frac{v_2}{\omega_2}\right)^2,$$

where $v=M^{-1}p$, all weights are positive, and $\omega_i>0$ are fixed velocity scales. The reward rate is

$$r(z,u)=-\ell(z)-\tfrac12 w_u u^2,\qquad w_u>0.$$

Its maximum is attained only at upright rest with zero torque, modulo angular periodicity. Thus a fast pass through upright is less desirable than remaining balanced. The reward is specified from measured physical state and fixed task parameters, not recomputed using a changing learned energy model.

This choice distinguishes the desired equilibrium but does not guarantee that optimization discovers swing-up. Excessive velocity penalties, excessive effort penalties, a short effective horizon $1/\beta$, or inadequate exploration can make pumping difficult to learn. These are empirical design questions. Give every compared algorithm the same final reward and report its weights.

### 2.2 Why an orbit-targeting reward is insufficient by itself

An energy-based swing-up function of the form

$$W_{\mathrm{orb}}(q,v)=\tfrac12(E-E_{\mathrm{top}})^2
+\tfrac12 k_Pq_2^2+\tfrac12 k_Dv_2^2$$

vanishes when the links are aligned, the elbow rate is zero, and the energy equals the upright energy. Away from upright, shoulder kinetic energy can satisfy these conditions. Such functions underpin analytical Acrobot swing-up results [2], but their zero set does not distinguish upright rest from all other points of the target orbit. The displayed quadratic angle term uses an unwrapped elbow coordinate; imposing periodic wrapping changes its regularity and must not silently inherit the original analysis.

Successful orbit-reaching behavior is useful experience for training capture. It is not a substitute for an objective that rewards sustained balance. A theorem for the analytical torque law also does not automatically apply to a neural policy trained using its energy function.

### 2.3 Optional potential-based guidance

An existing swing-up progress function can be retained as an auxiliary potential while keeping the upright reward fixed. For a bounded potential $\Phi(z)$ and discount rate $\beta>0$, the exact transition shaping term is

$$F_k=e^{-\beta h_k}\Phi(z_{k+1})-\Phi(z_k).$$

It is added to the accumulated interval reward, not to its rate without scaling. Under the usual terminal or vanishing-tail conditions, the terms telescope and preserve the optimal policy of the underlying task [8]. A bounded transform of an orbit error is one possible potential; its usefulness remains experimental.

In a deterministic continuous-time description,

$$r_\Phi=r+\nabla\Phi^\top f-\beta\Phi,\qquad V_\Phi=V-\Phi.$$

Consequently,

$$r_\Phi+\nabla V_\Phi^\top f-\beta V_\Phi
=r+\nabla V^\top f-\beta V.$$

If this shaped reward is used inside the analytic action optimization, its action dependence must be included: the relevant gradient becomes $\nabla(V_\Phi+\Phi)$. Applying the unshaped torque formula to $V_\Phi$ alone would omit a term.

Pure policy-invariant shaping cannot turn an orbit-only objective into an upright-balance objective. The underlying reward must already specify balance. Likewise, substituting an analytical controller's expression for $\dot\Phi$ while executing another action is not the actual potential derivative.

---

## 3. PH model, value operator, and bounded torque

### 3.1 Structured dynamics model

The first study assumes full state observation and known $M(q)$, so $p=M(q)v$ has a fixed physical meaning. Learn the uncertain potential and damping using

$$E_\phi(q,p;c)=\tfrac12p^\top M(q)^{-1}p+U_\phi(q;c),$$

$$f_\phi(z,u;c)=(J-R_\phi(q;c))\nabla E_\phi(z;c)+G_\phi(c)u,$$

$$R_\phi=\begin{bmatrix}0&0\\0&D_\phi\end{bmatrix},\qquad
D_\phi=L_\phi L_\phi^\top,\qquad G_\phi(c)=b_\phi(c)G.$$

The nominal actuator gain is $b_\phi=1$. If gain is learned, constrain it to the declared physically plausible positive range. Preserve the known elbow input direction; an arbitrary four-dimensional learned input vector could invent actuation of the shoulder or configuration coordinates.

A simple potential model is $U_\phi=-b_{1,\phi}\cos q_1-b_{2,\phi}\cos(q_1+q_2)$ with positive coefficients. A richer model may use smooth periodic features. Use the same smooth periodic representation for the value network, for example

$$V_\psi(z;c)=\widetilde V_\psi(\sin q_1,\cos q_1,\sin q_2,\cos q_2,p_1,p_2,c).$$

Differentiate through this feature map with respect to physical $z$. If inputs are normalized, include their scale factors in the derivatives.

The context $c$ is initially an observed, episode-constant scenario descriptor, such as damping or actuator effectiveness. An inferred context from observation history is an extension: freezing an estimate during a local HJB calculation is an approximation, not a complete belief-state control formulation. An evolving context requires accounting for its transition law in the value problem.

Learning structured energy and dissipation from temporal observations is motivated by PHAST [7]. Here the model is used inside control optimization. Its prediction accuracy, physical identifiability, and usefulness for control are separate measurements.

If inertia is also unknown, do not repeatedly relabel stored momenta with a changing $M_\phi$ while using the same canonical HJB equations. Either use consistently observed canonical coordinates, construct a latent-state model with its inference dynamics, or formulate learning directly in $(q,v)$. The first experiment avoids this additional ambiguity by fixing the inertia model.

### 3.2 Deterministic continuous-time objective

For a fixed context, define

$$V^\star(z;c)=\sup_{u(\cdot)}\int_0^\infty e^{-\beta t}r(z(t),u(t))\,dt,$$

subject to the true plant and $u(t)\in\mathcal U$. Assume the return is finite on the evaluation domain. A learned model approximates the dynamics used in the optimization; solving its HJB equation would yield model-optimal behavior, not necessarily plant-optimal behavior.

For a differentiable candidate value, define

$$\mathcal T_\phi[V](z;c)
=\max_{u\in\mathcal U}\{r(z,u)+\nabla V(z;c)^\top f_\phi(z,u;c)-\beta V(z;c)\}.$$

The exact model-optimal value formally satisfies $\mathcal T_\phi[V]=0$. Nonsmooth optimal values require a viscosity interpretation; a smooth neural approximation and a sampled residual loss do not by themselves solve that numerical issue.

The instantaneous action score is

$$q_{\phi,V}(z,u;c)=r(z,u)+\nabla V^\top f_\phi(z,u;c)-\beta V.$$

This lowercase $q$ has units of reward per time. It is distinct from configuration $q$ by its arguments, and from a finite-interval return-valued critic $Q_h$. Continuous-time $q$-learning explicitly studies this distinction [3].

### 3.3 Acrobot action formula

Start with the full instantaneous score from Section 3.2:

$$q_{\phi,V}(z,u;c)
=r(z,u)+\nabla_z V(z;c)^\top f_\phi(z,u;c)-\beta V(z;c).$$

The reward and dynamics are

$$r(z,u)=-\ell(z)-\tfrac12w_u u^2,\qquad
f_\phi(z,u;c)=(J-R_\phi)\nabla_z E_\phi+G_\phi(c)u.$$

Define the **zero-input drift**

$$f_{\phi,0}(z;c):=f_\phi(z,0;c)=(J-R_\phi)\nabla_z E_\phi.$$

The subscript $\phi$ denotes learned model parameters; the subscript $0$ means zero applied torque. Gravity, momentum, and damping still contribute to this drift. Thus $f_\phi=f_{\phi,0}+G_\phi u$.

Substitute both expressions into the score and collect terms. Suppressing repeated state and context arguments for readability,

$$\begin{aligned}
q_{\phi,V}(z,u;c)
&=-\ell(z)-\tfrac12w_u u^2
+\nabla_z V^\top\bigl(f_{\phi,0}+G_\phi u\bigr)-\beta V\\
&=\underbrace{-\ell(z)+\nabla_z V^\top f_{\phi,0}-\beta V}_{\text{independent of }u\text{ at fixed }z,c}
+\underbrace{\left(-\tfrac12w_u u^2+(G_\phi^\top\nabla_z V)u\right)}_{\text{depends on }u}.
\end{aligned}$$

Here $\nabla_z V^\top G_\phi=G_\phi^\top\nabla_z V$ because the Acrobot has a scalar input. Define its coefficient as

$$\eta_V(z;c)=G_\phi(c)^\top\nabla_zV(z;c)
=b_\phi(c)\frac{\partial V}{\partial p_2}.$$

For the Acrobot, $G_\phi=b_\phi[0,0,0,1]^\top$ and $z=(q_1,q_2,p_1,p_2)$, so multiplication by $G_\phi^\top$ selects the $p_2$ derivative. Equivalently, the full score in mechanical components is

$$\begin{aligned}
q_{\phi,V}(z,u;c)
={}&-\ell(z)-\tfrac12w_u u^2
+(\nabla_q V)^\top M(q)^{-1}p\\
&+(\nabla_p V)^\top\left[-\nabla_q E_\phi-D_\phi M(q)^{-1}p\right]
+b_\phi\frac{\partial V}{\partial p_2}u-\beta V.
\end{aligned}$$

The state $z$ and context $c$ are held fixed during this instantaneous action maximization. Although the chosen torque changes future states, that effect is represented by the value-gradient term; it does not make $\ell(z)$ depend on the current optimization variable $u$. Therefore maximizing the full score over $u$ is equivalent to maximizing just

$$-\tfrac12w_u u^2+\eta_Vu.$$

Its derivative with respect to torque is $-w_u u+\eta_V$, and its second derivative is $-w_u<0$. The unconstrained maximizer is consequently $u=\eta_V/w_u$. Enforcing the scalar torque interval gives the exact bounded maximizer

$$\boxed{u_V(z;c)=\operatorname{clip}\!\left(
\frac{\eta_V(z;c)}{w_u},-u_{\max},u_{\max}\right).}$$

No matching equations are missing. Underactuation appears through the rank-one port $G_\phi$. The difficult unknown is the long-term value gradient: it must encode how elbow torque affects later shoulder motion, braking, and balance through the coupled plant.

The derivative $\partial V/\partial p_2$ is not $\partial V/\partial v_2$. In velocity coordinates, with $\widehat V(q,v)=V(q,M(q)v)$, the equivalent signal is

$$\eta_V=b_\phi B^\top M(q)^{-\top}\nabla_v\widehat V.$$

The full drift must also be represented in those coordinates. Merely replacing the momentum derivative with a velocity derivative changes the controller.

### 3.4 The saturation-aware HJB operator

Define

$$\chi(\eta)=\max_{|u|\le u_{\max}}
\left\{-\tfrac12w_u u^2+\eta u\right\}
=\begin{cases}
\dfrac{\eta^2}{2w_u},& |\eta|\le w_u u_{\max},\\[4pt]
u_{\max}|\eta|-\dfrac12w_u u_{\max}^2,& |\eta|>w_u u_{\max}.
\end{cases}$$

Then the complete deterministic operator is

$$\boxed{\mathcal T_\phi[V]
=-\ell+\nabla V^\top f_{\phi,0}-\beta V+\chi(\eta_V).}$$

The unconstrained term $\eta_V^2/(2w_u)$ is valid only when the unconstrained optimizer lies within the torque interval. Smoothly squashing that optimizer with $\tanh$ respects bounds but generally does not maximize this same quadratic expression. A squashed learned actor is instead optimized using its actual output.

### 3.5 Soft policy extension

Use a density $\pi(a\mid z,c)$ over the dimensionless action $a\in[-1,1]$, with respect to $da$. Fix this entropy convention for all experiments. For temperature $\alpha>0$, define

$$\mathcal T_{\phi,\alpha}[V]
=\sup_\pi\mathbb E_{a\sim\pi}
\left[q_{\phi,V}(z,u_{\max}a;c)-\alpha\log\pi(a\mid z,c)\right].$$

This is a continuous-time entropy-rate formulation: reward and entropy are both accumulated per unit physical time. In the idealized continuously randomized description, the drift is averaged over the action distribution. Finite action holds approximate that description and must be evaluated separately. Randomized actions are not an additional Brownian diffusion term in the plant model.

For this one-dimensional quadratic action score, the maximizing density can be evaluated directly:

$$\pi_V^\star(a\mid z,c)=\frac1{Z_V(z,c)}
\exp\!\left[\frac{u_{\max}\eta_V a-\tfrac12w_u u_{\max}^2a^2}{\alpha}\right],\qquad -1\le a\le1,$$

$$Z_V=\int_{-1}^1
\exp\!\left[\frac{u_{\max}\eta_V a-\tfrac12w_u u_{\max}^2a^2}{\alpha}\right]da.$$

Thus

$$\boxed{\mathcal T_{\phi,\alpha}[V]
=-\ell+\nabla V^\top f_{\phi,0}-\beta V+\alpha\log Z_V.}$$

This density is a truncated Gaussian, and its mode is the normalized clipped deterministic optimizer. Its mean generally differs from the mode near the bounds. Use stable log-domain quadrature or a stable truncated-normal calculation. As $\alpha\to0$, the soft operator tends to the bounded deterministic operator.

A learned actor can instead use $a=\tanh(\mu_\theta+\sigma_\theta\epsilon)$, $\epsilon\sim\mathcal N(0,1)$, with the transformed log density including the $\tanh$ Jacobian. Its loss is

$$L_\pi(\theta)=\mathbb E_{z,c,a\sim\pi_\theta}
\left[\alpha\log\pi_\theta(a\mid z,c)-q_{\phi,V_\psi}(z,u_{\max}a;c)\right].$$

During the actor update, freeze model and value parameters while retaining differentiation with respect to the sampled action. The actor may receive $(z,c,\eta_V)$ as features. An independent $Q$ network is not necessary for this exact local score; the actor approximates or amortizes its policy improvement step. Comparing it with the analytic density measures actor approximation error.

The soft policy must be paired with the soft value operator, not silently with a hard maximum. Persistent exploration can prevent exact rest, so report both stochastic execution and a declared deterministic evaluation rule. A deterministic mean or mode is a different deployment policy and requires its own balance evaluation [5, 6].

---

## 4. Learning and staged experiments

### 4.1 Transition data and model fitting

Store physical transitions

$$\mathcal D_k=(q_k,v_k,u_k,R_k,q_{k+1},v_{k+1},h_k,c_k,d_k),$$

where

$$R_k=\int_0^{h_k}e^{-\beta s}r(z(t_k+s),u_k)\,ds$$

and $d_k$ denotes a genuine terminal event. A data-collection time limit is not automatically terminal for an infinite-horizon task. If physical limit violations terminate an episode, specify the terminal continuation cost so early termination does not become a way to avoid future negative reward.

Convert to canonical states with the fixed inertia model. For sufficiently small noise-free intervals, estimate

$$\dot z_k^{\mathrm{data}}\approx\frac{z_{k+1}-z_k}{h_k},\qquad
L_{\mathrm{dyn}}=\mathbb E\left[\|f_\phi(z_k,u_k;c_k)-\dot z_k^{\mathrm{data}}\|_S^2\right].$$

Use a continuous local angle lift for the difference and a declared scaling matrix $S$. Endpoint differencing approximates an instantaneous derivative only to finite-step accuracy; observation noise is amplified by division by $h_k$. If unwrapping is ambiguous or noise is material, use observation sequences and integrated prediction losses instead.

An alternative fits $F_{\phi,h_k}(z_k,u_k)$ directly to $z_{k+1}$, where $F_{\phi,h}$ integrates the model with the held action. This costs a short rollout but avoids treating a long-interval average derivative as an instantaneous one. Automatic differentiation through $\nabla E_\phi$ must retain the derivative graph during model fitting.

### 4.2 Fitted value updates

Freeze a target value $V_{\bar\psi}$ and the current model. Choose either the hard or soft operator consistently and form

$$y_V(z,c)=\operatorname{stopgrad}\left[
V_{\bar\psi}(z,c)+\Delta\tau\,\mathcal T_{\phi,\alpha}[V_{\bar\psi}](z,c)\right],$$

$$L_V=\tfrac12\mathbb E\left[(V_\psi(z,c)-y_V(z,c))^2\right].$$

For the deterministic experiment, replace $\mathcal T_{\phi,\alpha}$ with $\mathcal T_\phi$. The iteration step $\Delta\tau$ has time units but is an optimization parameter, independent of the action-hold duration $h_k$. This is a fitted approximation to $\partial_\tau V=\mathcal T[V]$ [4].

An alternative is HJB residual minimization, $\tfrac12\mathbb E[\mathcal T[V_\psi]^2]$. Treat these as alternative numerical choices before combining losses. Neural approximation, changing models, insufficient state coverage, and inappropriate iteration steps can all cause failure; the derivative operator is not automatically a contraction under a naive neural update.

For the deterministic reward, $V^\star(z^\star)=0$ because upright rest can be maintained with zero torque in the nominal plant. This supplies a useful anchor. It does not apply unchanged to the entropy-regularized value, whose objective includes exploration. Use declared terminal or boundary conditions and check predictions with actual returns rather than treating a small sampled HJB residual as sufficient validation.

### 4.3 Finite action holds and a Bellman reference

For a fixed interval $h$, an exact held-action backup on a known model is

$$\mathcal B_h[V](z)=\max_{u\in\mathcal U}
\left\{\int_0^h e^{-\beta s}r(F_s(z,u),u)\,ds
+e^{-\beta h}V(F_h(z,u))\right\}.$$

Under smoothness and uniform local regularity,

$$\mathcal B_h[V]=V+h\mathcal T[V]+O(h^2).$$

This provides a finite-interval cross-check on the differential operator. At a noninfinitesimal hold duration, future state dependence generally destroys the simple quadratic action formula; the instantaneous maximizer need not maximize this whole backup.

For a soft held-action backup, replace the maximum with a supremum over the action density and include

$$-\alpha c_\beta(h)\log\pi(a\mid z),\qquad
c_\beta(h)=\frac{1-e^{-\beta h}}{\beta}.$$

The score being averaged is the actual discounted interval reward plus discounted continuation value. This convention gives the same entropy rate in the small-$h$ limit. Using a fixed entropy bonus per decision while changing $h$ changes the physical-time objective.

The simple sampled reward estimate is $R_k\approx c_\beta(h_k)r(z_k,u_k)$ when the rate is nearly constant during the hold, or $h_kr(z_k,u_k)$ to first order. Discount with $e^{-\beta h_k}$ consistently. Variable exogenous intervals can be supplied as policy inputs when known at decision time; event-driven termination of an action requires its own stopping-time formulation.

### 4.4 One training iteration

1. Sample state transitions and scenario contexts from replay, including near-upright and incoming swing-up states.
2. Fit the PH dynamics with measured transitions. Check model error on held-out trajectories.
3. Freeze the model and target value, compute the saturation-aware hard operator or the soft integral, and update the value.
4. Construct $\eta_V$. Execute the bounded analytic policy or update the actor using the matching action score and entropy convention.
5. Collect further transitions with the physical actuator bounds and declared hold interval. Update replay and target networks.
6. Periodically freeze the policy and evaluate swing-up, capture, and retention on held-out starts.

All policy gradients here optimize predicted task return. None enforces a Lyapunov inequality on every executed action.

### 4.5 Stage A — learn local capture and balance

Begin with the oracle PH model, a near-upright reset distribution, and the fixed upright reward. This separates value and policy optimization failures from model-identification failures. Then repeat with the learned PH model.

Include states recorded shortly before and during successful swing-up passes. Their velocities are essential: a small angle error can coexist with a speed that makes capture difficult under the torque limit. Keep held-out incoming trajectories for evaluation.

The milestone is sustained balance under the learned torque policy from these states. A separate LQR or IDA-PBC controller can be a diagnostic baseline, but it is not required to define the learning problem. If used to generate demonstrations, account for that additional information.

### 4.6 Stage B — preserve swing-up while learning capture

Initialize from a successful swing-up policy where the architecture permits, or use its trajectories as data for a new policy. Sample a mixture of hanging starts, broader starts, and incoming capture states. Keep the final upright reward fixed and recompute that reward for retained transitions if they originally used an orbit-only objective. Old critic targets do not remain valid after an objective change.

The policy still commands torque throughout the episode. No controller handoff is necessary. Measure both retained swing-up success and new balance success, because fine-tuning only near upright can erase the learned pumping behavior.

Replacing an existing actor by the analytic value-gradient controller is a new policy, not an automatic preservation of its behavior. Report that replacement separately from actor fine-tuning or distillation.

### 4.7 Stage C — learn the full maneuver over a broader domain

Extend resets to earlier portions of successful trajectories and then to a declared broader initial-state distribution. Increase the difficulty only when held-out capture and retention remain satisfactory. This curriculum expands training coverage; it does not expand a certified region of attraction by definition.

The value function is intended to cover the whole task, so it has no IDA-PBC matching domain that must first be extended. Its limitations are instead model accuracy, approximation quality, data coverage, objective design, and physical reachability. Sufficient exploration is still needed to discover pumping from low-energy states; replay near the top alone does not establish that capability.

---

## 5. What the PH model contributes and how to evaluate it

### 5.1 Contribution to learning

The PH model enters the value update through $\nabla V^\top f_{\phi,0}$ and enters action ranking through $G_\phi^\top\nabla V$. It can improve learning if its structural restrictions make these quantities more accurate with limited data. It also exposes energy-balance residuals and separates damping from conservative motion.

With the known canonical elbow port, however, $\eta_V=\partial V/\partial p_2$: the energy model does not appear explicitly in the final action formula. It affects that formula indirectly by shaping the learned value during HJB updates. If actuator gain is learned, it also affects the input signal directly.

An equally accurate unstructured control-affine model can use the same HJB action optimization. Therefore the testable PH claim is improved model learning and resulting control under comparable resources, not exclusive access to an action formula.

### 5.2 Required comparisons

| Comparison | Question answered |
|---|---|
| Conventional SAC and a specified continuous-time actor-critic baseline [5, 6] | Does the proposed method improve full-task learning under the same reward and time budget? |
| Oracle PH model versus learned PH model | How much failure comes from dynamics learning? |
| Learned PH model versus an unstructured control-affine model | Does PH structure help beyond having a model? |
| Analytic bounded policy versus learned actor | Does actor approximation help or hurt this scalar control problem? |
| Upright reward with and without orbit-potential shaping | Does retained swing-up guidance improve capture learning? |
| Local, incoming-trajectory, and broad reset distributions | Which stages benefit from the curriculum? |
| Differential value updates versus finite-interval backups | How much error arises from finite action holds? |

Hold final rewards, actuator limits, observations, initial-state distributions, and physical training durations fixed within each comparison. Account for model-fitting computation, simulator queries, oracle parameters, and any demonstration data. An unstructured model baseline should receive the same known input port and other supplied information when isolating the value of the PH drift structure.

Use matched seeds and report uncertainty across seeds and evaluation starts. Separate frozen-policy transfer from retraining or online adaptation. For an initial transfer study, vary damping and actuator gain while keeping inertia and state coordinates fixed.

### 5.3 Metrics

Report full-task success probability, time to first capture, uninterrupted hold duration, upright occupancy, torque-squared integral, saturation fraction, and violation counts for any declared physical limits. Stratify results by initial energy and by the velocity distribution at approach to upright.

Report model vector-field and rollout errors on held-out data, physical energy-balance error, and action-score error against the oracle model where available. Include the deterministic executed policy in model-based validation. A high return or accurate open-loop forecast alone does not demonstrate reliable capture.

An empirical capture region is the set of tested states from which a policy met the hold criterion. A certified region requires a separate argument over all states in a specified set. Keep these labels distinct.

### 5.4 Stability and the optional role of IDA-PBC

IDA-PBC chooses a desired energy and interconnection, then solves matching equations so the available torque realizes the prescribed closed-loop structure [9]. This formulation instead learns a value and optimizes over the available torque. Both are ways to design feedback; underactuation does not require choosing the former.

If a stability claim is desired after training, freeze the policy and seek a certificate $W$ satisfying, for example,

$$W(z)>0\quad(z\ne z^\star),\qquad
\nabla W(z)^\top f_{\mathrm{true}}(z,\mu_\theta(z))\le-\alpha_W(\|z-z^\star\|)$$

on an appropriate invariant sublevel set, with positive-definite $\alpha_W$. Sampled execution, saturation, and model uncertainty must be included in the actual argument. Negative-semidefinite decrease needs an additional invariant-set analysis. Stochastic execution requires a suitable stochastic or practical-stability statement.

Even the exact discounted deterministic value is not automatically a Lyapunov certificate. Writing $C^\star=-V^\star$ and $L=\ell+\tfrac12w_u u^2$, the exact HJB identity along the optimal feedback gives

$$\dot C^\star=\beta C^\star-L.$$

Its sign is not guaranteed by $L\ge0$ alone. Approximate neural values and models provide still less of a stability guarantee.

IDA-PBC may supply a teacher, a candidate certificate, or a restricted policy family in a later experiment. Adding any of these changes the information or constraints given to learning and must be evaluated accordingly. It is not needed to fill a missing torque equation in the present method.

### 5.5 Status and possible contribution

This document defines a method and evaluation program. It establishes the Acrobot action formula and the bounded hard and soft operators algebraically; it does not establish successful training, a certified basin, or superiority over existing methods.

A contribution could be a reproducible improvement in sample efficiency, capture reliability, or transfer across varied environments under fair comparisons. The evidence should identify whether the gain comes from PH dynamics, the value update, the actor, or the reward and curriculum. Acrobot success alone would not establish that the same benefit generalizes to other underactuated systems.

---

## References

[1] M. W. Spong, “[The Swing Up Control Problem for the Acrobot](https://www.diag.uniroma1.it/~oriolo/ur_LL/material/Spong_SwingUp.pdf),” *IEEE Control Systems Magazine*, vol. 15, no. 1, pp. 49–55, 1995.

[2] X. Xin and M. Kaneda, “[Analysis of the Energy-Based Swing-Up Control of the Acrobot](https://doi.org/10.1002/rnc.1184),” *International Journal of Robust and Nonlinear Control*, vol. 17, no. 16, pp. 1503–1524, 2007.

[3] Y. Jia and X. Y. Zhou, “[q-Learning in Continuous Time](https://arxiv.org/abs/2207.00713),” arXiv:2207.00713, first submitted 2022; revised 2025. Continuous-time advantage rates and entropy-regularized policy improvement.

[4] M. Nguyen, “[Decoupled Continuous-Time Reinforcement Learning via Hamiltonian Flow](https://arxiv.org/abs/2602.14587),” arXiv:2602.14587, 2026. Motivation for alternating generator-based action evaluation and value-flow updates; no convergence theorem for the specific learned-model construction above is assumed.

[5] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “[Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://proceedings.mlr.press/v80/haarnoja18b.html),” *Proceedings of the International Conference on Machine Learning*, PMLR 80, pp. 1861–1870, 2018.

[6] H. Han and S. Ji, “[Continuous Soft Actor-Critic: An Off-Policy Learning Method Robust to Time Discretization](https://papers.nips.cc/paper_files/paper/2025/hash/6ac12f42db406e6be14d669884e73212-Abstract-Conference.html),” *Advances in Neural Information Processing Systems*, vol. 38, 2025. A distinct continuous-time actor-critic baseline; its martingale-based algorithm is not identical to the fitted PH value flow here.

[7] S. Bhardwaj and C. Bajaj, “[PHAST: Port-Hamiltonian Architecture for Structured Temporal Dynamics Forecasting](https://arxiv.org/abs/2602.17998),” arXiv:2602.17998, 2026. Structured dynamics forecasting and identification, distinct from the proposed control evaluation.

[8] A. Y. Ng, D. Harada, and S. Russell, “[Policy Invariance under Reward Transformations: Theory and Application to Reward Shaping](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf),” *Proceedings of the International Conference on Machine Learning*, pp. 278–287, 1999. The variable-interval and continuous-time expressions above follow from the same discounted telescoping identity.

[9] R. Ortega, A. van der Schaft, B. Maschke, and G. Escobar, “[Interconnection and Damping Assignment Passivity-Based Control of Port-Controlled Hamiltonian Systems](https://doi.org/10.1016/S0005-1098(01)00278-3),” *Automatica*, vol. 38, no. 4, pp. 585–596, 2002.
