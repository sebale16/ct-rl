---
title: Region of Attraction for a Torque-Limited Acrobot via Sums of Squares
tags: [ct-rl, acrobot, sums-of-squares, region-of-attraction, torque-limit]
robots: noindex
---

# Region of Attraction for a Torque-Limited Acrobot via Sums of Squares

This note describes the time-invariant upright-balancing calculation behind
Figure 2 of Majumdar, Ahmadi and Tedrake,
[*Control Design along Trajectories with Sums of Squares Programming*](https://groups.csail.mit.edu/robotics-center/public_papers/Majumdar13.pdf).

The paper's printed Algorithm 1 is stated for a time-varying funnel. Here it is
specialized to one time-invariant equilibrium and modified for a saturated
single input exactly as described in Section IV-A of the paper. This is the
controller-design calculation relevant to Figure 2; the time-varying swing-up
calculation is not implemented.

## I. Problem

Let

$$
\dot x = f(x)+g(x)u,
\qquad \bar x=x-x^\star,
$$

where $x^\star$ is upright rest. We search for both a polynomial feedback law
$u(\bar x)$ and a Lyapunov function $V(\bar x)$. The requested torque is passed
through

$$
s(u)=
\begin{cases}
u_{\max},&u\ge u_{\max},\\
u_{\min},&u\le u_{\min},\\
u,&\text{otherwise}.
\end{cases}
$$

For

$$
B_\rho=\{\bar x\mid V(\bar x)\le\rho\},
$$

the goal is to maximize $\rho$ while proving that $V$ decreases inside
$B_\rho$ under the saturated controller. Following the Figure 2 experiment,
the calculation uses a cubic $u$, a quadratic $V$, a symmetric 5 N m torque
limit, and a degree-three Taylor model of the dynamics.

## II. Unsaturated SOS condition

Without saturation, a sufficient certificate is

$$
\begin{aligned}
V(\bar x)&\ \text{SOS},\\
-\dot V(\bar x)+L(\bar x)(V(\bar x)-\rho)&\ \text{SOS},\\
L(\bar x)&\ \text{SOS},\\
V(\mathbf 1)&=1.
\end{aligned}
$$

On $B_\rho$, the multiplier term is non-positive. Therefore the second SOS
condition implies $-\dot V\ge0$. The last equality removes the otherwise free
joint scaling of $V$ and $\rho$.

The program is not jointly convex. It contains the products $LV$ and $L\rho$,
and $\dot V$ contains products between the coefficients of $V$ and $u$. It is
convex when the appropriate blocks of variables are held fixed, which is why
the paper alternates between subproblems.

## III. Saturation branches

For a symmetric bound $\tau_{\max}$, define three rates:

$$
\begin{aligned}
\dot V_0 &= \nabla V^\top(f+gu),\\
\dot V_+ &= \nabla V^\top(f+g\tau_{\max}),\\
\dot V_- &= \nabla V^\top(f-g\tau_{\max}).
\end{aligned}
$$

The following three polynomials are required to be SOS:

$$
\begin{aligned}
p_0={}&-\dot V_0+L_0(V-\rho)
       -M_{01}(\tau_{\max}-u)-M_{02}(\tau_{\max}+u),\\
p_+={}&-\dot V_++L_+(V-\rho)-M_+(u-\tau_{\max}),\\
p_-={}&-\dot V_-+L_-(V-\rho)-M_-(-\tau_{\max}-u),
\end{aligned}
$$

with every $L_i$ and $M_k$ SOS. Each expression following an $M_k$ is
non-negative precisely on its active saturation branch. On that branch and
inside $B_\rho$, all multiplier terms are non-positive, so $p_i\ge0$ implies
$\dot V_i\le0$.

## IV. Implemented three-step alternation

The saturation multipliers introduce products $M_k u$, so the saturated
controller and the multipliers cannot be searched simultaneously. The
time-invariant specialization of the paper's saturation-modified Algorithm 1
is:

**Algorithm 1: saturated time-invariant controller design**

1. Initialize $V$ and $u$ from LQR and choose a feasible small constant
   $\rho>0$.
2. With $V$, $u$, and $\rho$ fixed, solve a feasibility problem for all
   $L_i$ and $M_k$.
3. With $V$, $L_i$, and $M_k$ fixed, maximize $\rho$ over a cubic controller
   $u$ with $u(0)=0$.
4. With $u$ and $L_i$ fixed, maximize $\rho$ over a quadratic $V$ and the
   saturation multipliers $M_k$, subject to $V$ being SOS and $V(\mathbf1)=1$.
5. Repeat steps 2--4 until the relative improvement is below the tolerance or
   the iteration limit is reached.

The objective in steps 3 and 4 is constrained to be at least the level
already certified by the preceding step. This does not change either
subproblem: in exact arithmetic the preceding solution is feasible, which is
the paper's argument that the sequence of objectives is nondecreasing. It does
prevent a numerical SDP solution from being accepted as an apparent decrease.

The multipliers returned by the preceding value step remain an existing
feasible witness for the same $(V,u,\rho)$ if a fresh multiplier solve fails
numerically; no smaller level is substituted. There is no rho backoff or
coordinate-rescaling workaround in the algorithm.

## V. Initialization and fixed-LQR comparison

Linearization at upright and the continuous-time Riccati equation give

$$
V_{\mathrm{LQR}}(\bar x)=\bar x^\top P\bar x,
\qquad u_{\mathrm{LQR}}(\bar x)=-K\bar x.
$$

$V_{\mathrm{LQR}}$ is normalized before the alternation. The paper says that a
sufficiently small constant $\rho$ works well for initialization.

Separately, bisection finds the largest certified level for the fixed normalized
LQR pair. That calculation provides the blue fixed-controller baseline
corresponding to the comparison in Figure 2. It is not one of the three
alternating controller-design steps.

## VI. What is and is not reproduced

The formulation reproduces the structure of the Figure 2 calculation:

- polynomial dynamics about upright;
- the three exact saturation branches from Approach 1;
- LQR initialization;
- an optimized cubic time-invariant controller;
- an optimized quadratic Lyapunov function; and
- the three-step SOS alternation.

It does **not** yet reproduce the numerical red and blue curves printed in the
paper. The experiment used a hardware-identified Acrobot model, but the paper
does not publish its identified coefficients, its LQR $Q/R$ choices, the SOS
multiplier degrees, or the final polynomial coefficients. The present
calculation instead uses the Xin--Kaneda simulation parameters and the Lai et
al. LQR costs.

Consequently, the resulting set is a certificate for a different degree-three
polynomial surrogate, not a claim that the published Figure 2 has been
numerically recreated. Exact reproduction requires the original identified
model and design settings.

The paper notes that Taylor-expanded trajectories were close to the original
plant in its experiment. That observation is not a proof for this different
model. A certificate produced here applies rigorously to the polynomial field;
validation or a robust certificate is still required before treating it as a
guarantee for the exact trigonometric plant.
