---
title: Region of Attraction for a Torque-Limited Acrobot via Sums of Squares
tags: [ct-rl, acrobot, sums-of-squares, region-of-attraction, torque-limit]
robots: noindex
---

# Region of Attraction for a Torque-Limited Acrobot via Sums of Squares

Following Majumdar, Ahmadi and Tedrake, *Control Design along Trajectories with
Sums of Squares Programming*, specialised to the time-invariant case about a
fixed point and to a fixed feedback law.

## I. Problem statement

Let $\dot x = f(x) + g(x)u$ be the control system, $x^\star$ the upright
equilibrium, and $\bar x = x - x^\star$. The feedback $u(\bar x) = -K\bar x$ is
given and is not a decision variable. The actuator is bounded, so the applied
input is passed through the saturation function

$$
s(u(\bar x)) =
\begin{cases}
u_{\max} & u(\bar x) \ge u_{\max}\\[2pt]
u_{\min} & u(\bar x) \le u_{\min}\\[2pt]
u(\bar x) & \text{otherwise.}
\end{cases}
$$

For a Lyapunov candidate $V(\bar x)$ define the sublevel set

$$
B_\rho = \{\bar x \mid V(\bar x) \le \rho\}.
$$

We seek the largest $\rho$ for which $B_\rho$ is an inner estimate of the region
of attraction of $x^\star$ under the saturated law.

The dynamics are not polynomial: the gravitational and inertial terms are
trigonometric in the joint angles, and $\ddot q = M^{-1}(\tau - H - G)$ is
rational. Both are removed by Taylor expanding $f$ and $g$ about $\bar x = 0$ to
degree three, so that the certificate applies to the polynomial vector field.

## II. The sums-of-squares program

With $L(\bar x)$ a multiplier term, the region of attraction is certified by

$$
\begin{aligned}
\underset{\rho,\; L(\bar x),\; V(\bar x)}{\text{maximize}} \quad & \rho &&(1)\\
\text{subject to} \quad
& V(\bar x) \ \text{SOS} &&(2)\\
& -\dot V(\bar x) + L(\bar x)\big(V(\bar x) - \rho\big) \ \text{SOS} &&(3)\\
& L(\bar x) \ \text{SOS} &&(4)\\
& V\Big(\textstyle\sum_j e_j\Big) = 1 &&(5)
\end{aligned}
$$

where $e_j$ is the $j$-th standard basis vector. Condition (5) is a
normalization constraint, linear in the coefficients of $V$, and introduces no
conservativeness: any valid Lyapunov function can be scaled to satisfy it.

For $\bar x \in B_\rho$ we have $V(\bar x) - \rho \le 0$, and $L(\bar x) \ge 0$
by (4), so (3) gives $\dot V(\bar x) \le 0$.

## III. Incorporating actuator limits

A piecewise analysis of $\dot V$ enforces the Lyapunov conditions on each branch
of the saturation. Define

$$
\begin{aligned}
\dot V_{\min}(\bar x) &= \frac{\partial V(\bar x)}{\partial \bar x}^{\!\top}
\big(f(\bar x) + g(\bar x)u_{\min}\big) &&(6)\\
\dot V_{\max}(\bar x) &= \frac{\partial V(\bar x)}{\partial \bar x}^{\!\top}
\big(f(\bar x) + g(\bar x)u_{\max}\big) &&(7)
\end{aligned}
$$

and require

$$
\begin{aligned}
u(\bar x) \le u_{\min} &\implies \dot V_{\min}(\bar x) < 0 &&(8)\\
u(\bar x) \ge u_{\max} &\implies \dot V_{\max}(\bar x) < 0 &&(9)\\
u_{\min} \le u(\bar x) \le u_{\max} &\implies \dot V(\bar x) < 0 &&(10)
\end{aligned}
$$

Each implication is enforced with additional multipliers $M_k(\bar x)$,
replacing (3) by

$$
\begin{aligned}
& -\dot V_{\min} + L_1(V - \rho) + M_1\,(u - u_{\min}) \ \text{SOS} &&(11)\\
& -\dot V_{\max} + L_2(V - \rho) + M_2\,(u_{\max} - u) \ \text{SOS} &&(12)\\
& -\dot V + L_3(V - \rho) + M_3\,(u - u_{\max}) + M_4\,(u_{\min} - u) \ \text{SOS} &&(13)\\
& L_i(\bar x),\ M_k(\bar x) \ \text{SOS} &&(14)
\end{aligned}
$$

Each region is written as a set of inequalities that are non-positive where the
branch is active, so on that branch every multiplier term in (11)–(13) is
non-positive and the corresponding rate is certified negative. For $m$ inputs
this requires $3^m$ conditions; the Acrobot has $m = 1$.

## IV. Bilinear alternation

The program is not convex: $\dot V$ is linear in $V$, so the products
$L_i(V - \rho)$ and $L_i \dot V_i$ are bilinear in the decision variables. The
conditions are, however, linear in $L_i$ and $M_k$ for fixed $V$ and $\rho$, and
linear in $V$ and $\rho$ for fixed $L_i$ and $M_k$. We therefore alternate
between the two sets.

In Step 2, $\rho$ appears linearly in the constraints and is optimized directly.
In Step 1, $L_i$ is a decision variable multiplying $\rho$, so $\rho$ is
maximized by bisection.

**Algorithm 1** Region of Attraction under Actuator Limits

$$
\begin{array}{ll}
\hline
1\!: & \text{Initialize } V(\bar x) \text{ from the LQR cost-to-go, scaled to satisfy (5)}\\
2\!: & \rho_{\text{prev}} \leftarrow 0\\
3\!: & \textit{converged} \leftarrow \textbf{false}\\
4\!: & \textbf{while } \neg\,\textit{converged} \ \textbf{do}\\
5\!: & \quad \textbf{Step 1}: \text{Maximize } \rho \text{ by bisection, searching for } L_i(\bar x)\\
   & \quad\qquad \text{and } M_k(\bar x) \text{ subject to (11)–(14), with } V(\bar x) \text{ fixed.}\\
6\!: & \quad \textbf{Step 2}: \text{Maximize } \rho \text{ by searching for } V(\bar x) \text{ and } \rho\\
   & \quad\qquad \text{subject to (2), (5) and (11)–(13), with } L_i(\bar x),\, M_k(\bar x) \text{ fixed.}\\
7\!: & \quad \textbf{if } \dfrac{\rho - \rho_{\text{prev}}}{\rho_{\text{prev}}} < \varepsilon \ \textbf{then}\\
8\!: & \quad\quad \textit{converged} \leftarrow \textbf{true}\\
9\!: & \quad \textbf{end if}\\
10\!: & \quad \rho_{\text{prev}} \leftarrow \rho\\
11\!: & \textbf{end while}\\
\hline
\end{array}
$$

Each iteration attains an objective at least as good as the previous one, since
a solution to the previous iteration remains feasible for the current one. With
$\rho$ bounded above for any system with a bounded region of attraction, the
sequence of optimal values converges.

## V. Initialization

The alternation requires an initial Lyapunov candidate. Linearizing about
$x^\star$ and solving the algebraic Riccati equation

$$
A^\top S + SA - SBR^{-1}B^\top S + Q = 0
$$

gives $V_{\text{guess}}(\bar x) = \bar x^\top S \bar x$, scaled to satisfy the
normalization (5).
