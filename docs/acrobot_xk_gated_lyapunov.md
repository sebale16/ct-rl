# Gated Xin–Kaneda/LQR Lyapunov candidate

`controllers/acrobot_gated_lyapunov.py` constructs the local hand-off discussed
for the `acrobot-xk` reward experiments. It does not change an existing reward.

Two constructions live there. `GatedLyapunov`, described first, blends the two
pieces on the Xin–Kaneda switching test with each piece normalized on its own
scale. `NonsmoothLyapunov`, described from *Lai's nonsmooth Lyapunov function*
onward, is the published repair of the ridge that blend leaves behind, and is
the one intended as a reward base.

## Coordinates and local design

The state stays in the Xin–Kaneda paper frame,

\[
x=[q_1,q_2,\dot q_1,\dot q_2],\qquad
x_u=[\pi/2,0,0,0],
\]

and both angular errors are wrapped before evaluating the local quantities. The
exact nonlinear paper plant is linearized analytically at upright. With
\(Q=I_4\) and \(R=0.5\), the continuous algebraic Riccati equation gives

\[
V_L(e)=e^T P e,\qquad \tau=-K e.
\]

Those design weights match the local controller used for the same Acrobot
parameters by Lai et al. (2009). They are a stated design choice: the
Xin–Kaneda switching condition does not uniquely determine an LQR cost.

## Conservative smooth gate

Xin and Kaneda's equation (74) admits the local controller when

\[
\rho_1(e)=|e_1|+|e_2|+0.1|e_3|+0.1|e_4|<0.04.
\]

Directly using \(\rho_1\) makes the gate nondifferentiable on every coordinate
hyperplane. The implementation instead uses

\[
\tilde\rho(e)=\sum_{i=1}^4
  \sqrt{(w_i e_i)^2+\epsilon^2},
\quad w=[1,1,0.1,0.1],\quad \epsilon=10^{-6}.
\]

Because \(\tilde\rho\ge\rho_1\), every state assigned nonzero LQR membership is
strictly inside the printed switching region. Let

\[
s=\operatorname{clip}\!\left(
\frac{0.04-\tilde\rho}{0.04-0.02},0,1\right),\qquad
\mu=6s^5-15s^4+10s^3.
\]

The quintic has zero first and second derivatives at both endpoints. Thus
\(\mu=0\) outside the 0.04 boundary and \(\mu=1\) inside the 0.02 boundary.

## Normalization and candidate

The Xin–Kaneda component is normalized by its value at hanging rest,

\[
\bar V_X=V_X/(E_s^2/2).
\]

The LQR component is normalized by the maximum of \(e^T P e\) over the
weighted-\(L^1\) switching polytope. Since a convex quadratic reaches a maximum
over that polytope at a vertex, the scale is exactly

\[
c_L=0.04^2\max_i P_{ii}/w_i^2,
\qquad \bar V_L=V_L/c_L.
\]

The constructed candidate is

\[
V_g=(1-\mu)\bar V_X+\mu\bar V_L.
\]

Its implementation exposes the value, exact state gradient, and directional
rate. In the transition band the gradient includes the necessary term

\[
\nabla\mu\,(\bar V_L-\bar V_X).
\]

## Important limitation

This construction is not yet a global Lyapunov function for upright. Outside
the gate it is exactly the Xin–Kaneda function, whose zero set is the complete
homoclinic orbit. Consequently, `V_g` is also zero at points on that orbit far
from upright. Turning on the positive LQR value inside the gate also creates a
transition ridge along the ideal orbit.

The module therefore calls it a **gated Lyapunov candidate** and does not wire
it into `r0`–`r3`. Before using it as a reward base, the choice is explicit:

On the exact homoclinic orbit, the default construction rises from zero to a
normalized value of approximately `0.0491` in the transition band and then
falls to zero at upright. This is the concrete reward ridge, not merely a
formal possibility.

1. accept it as local potential shaping and measure whether the transition
   ridge impedes capture; or
2. change the global objective so upright—not the full orbit—is its unique zero
   set, which no longer preserves the original Xin–Kaneda reward objective.

The tests pin down the Riccati equation, the nonlinear linearization, the exact
switch-region containment, gate endpoints, analytic gradient, local decrease
under LQR, and the retained remote zero set.

---

## Lai's nonsmooth Lyapunov function

The ridge has a published fix. Lai, Wu, She and Yang, *Comprehensive Unified
Control Strategy for Underactuated Two-Link Manipulators*, IEEE Trans. SMC-B
39(2), 2009 (and the three-stage version in Lai et al., ICRA 2006, *Stability
Analysis and Control Law Design for Acrobots*) assemble one Lyapunov function
for the whole motion space out of the same two pieces:

- a swing-up piece, their eq. (20),
  \[V_1 = \tfrac12\left[\alpha_1\tilde E^2 + \alpha_2 q_2^2 + \beta(x)\dot q_2^2 + \Delta\right],\]
  which is the Xin–Kaneda function with \(\beta\) state-dependent in place of
  \(k_D\), plus a constant \(\Delta\);
- a local piece, their eq. (44), \(V_2 = e^T P e\) with \(P\) from the CARE.

Their Definition 3 calls the switched function a **nonsmooth Lyapunov function**
(NSLF) and demands \(J[x(t_2)] < J[x(t_1)]\) across the switching surface;
Theorem 2 then gives stability of the switched system. Equation (71) secures
the demand by setting

\[\Delta = \max_{x\in\Sigma_2} V_2(x),\]

so \(V_1 \ge \Delta \ge V_2\) wherever a switch can occur. Their words for what
this buys: it "avoids a shock in the switching surface."

Switching the *reward's* Lyapunov function on the same gate needs no further
theory on the RL side. A piecewise \(V\) is still a function of state alone, so
using \(\Phi = -V\) leaves the Ng–Harada–Russell policy invariance behind `r3`
untouched. \(\Delta\) is what the shaping quality turns on.

### The local piece is theirs exactly

Lai et al. use this Acrobot's parameters with \(Q=I_4\), \(R=0.5\) and print
\(F = [-260.559,\ -104.448,\ -112.604,\ -52.944]\) in their eq. (75).
`lqr_solution` returns \([-260.26,\ -104.30,\ -112.48,\ -52.88]\), agreeing to
0.1%. The `LQRDesign` defaults were already chosen to match; the test now pins
it against the published feedback.

### Region

Their attractive area \(\Sigma_2\) (eq. 17) replaces the 2007 switching test
with two angle conditions, a weighted velocity cap, and an energy band,

\[|x_1|\le\epsilon_1,\quad |x_1+x_2|\le\epsilon_2,\quad
\|(w_3x_3,w_4x_4)\|\le\epsilon_5,\quad |E-E_0|\le\epsilon_E,\]

both angles wrapped. The velocity condition is vacuous at their own weights
(\(w=10^{-3}\), \(\epsilon_5=10^3\), so it caps \(\|\dot q\|\) at \(10^6\)), so
speed enters only through the energy band — which is what makes the region
reachable by a swing-up that regulates energy. It is carried anyway, so that
this object reproduces the printed test exactly.

**One home, two frames.** `controllers/lai_she.py` already implements this same
paper as a controller and needs the same region test. The 2009 paper measures
the shoulder from upright while this repository's Xin–Kaneda plant measures it
from the horizontal, and the two are related by \(x = -e\) with \(e\) the
wrapped upright error. Every condition above is even in its argument, so the
region is the *same function* in both frames. `AttractiveRegion.residual_of`
therefore takes the four scalars directly and serves both;
`AttractiveRegion.exact_residual` is the Xin–Kaneda-frame wrapper and
`LaiSheController.in_attractive_area` is the paper-frame one, built through
`Design.attractive_region()`. The two modules also reach the Riccati step
through one `riccati_feedback`, while keeping their own plants and
linearizations — the two papers print different link inertia (\(8.33\times
10^{-2}\) against \(0.083\)), and the controller pins its rebuilt gain to
within 0.21 N·m of eq. (75), which merging the parameter sets would break.

The smoothing follows the 2007 gate: each absolute value is replaced by
\(\sqrt{\cdot^2+\epsilon^2}\), the velocity norm likewise, and the maximum by
an 8-norm. Every replacement only raises the residual, so any state given
nonzero local membership sits strictly inside the printed region.

### Choosing the tolerances

Lai et al. print \(\epsilon_a = \pi/6\), \(\epsilon_E = 1\ \mathrm{J}\). On this
plant \(\pi/6\) is too loose to certify: sampling \(\Sigma_2\), the Riccati
value fails to be decreasable under any torque on 1.7% of it, and \(-Ke\)
demands up to 277 N·m against the evaluation protocol's 64 N·m actuator.

The two tolerances trade off against different things. Reachability, measured
by running the Xin–Kaneda analytic controller over the evaluation protocol's
32 starts for 20 s, is governed by the energy band:

| gate | entered |
|---|---|
| 2007 test, \(\zeta=0.04\) | 2 / 32 |
| \(\pi/6\), 1.0 J | 16 / 32 |
| \(\pi/12\), 1.0 J | 9 / 32 |
| \(\pi/30\), 1.0 J | 9 / 32 |
| \(\pi/12\), 0.5 J | 3 / 32 |
| \(\pi/24\), 0.5 J | 3 / 32 |

These come from RK4 on the paper plant under continuous feedback, so they sit
above the protocol's own metric-6 reading of 0 / 32 at \(\zeta = 0.04\), which
carries the 0.5 ms zero-order hold; the note on metric 6 in the reward document
puts that hold's contribution at about 0.02 of the residual, half the \(\zeta\)
box. The comparison between rows is the point, not the absolute counts.

Tightening the angle box eightfold costs almost nothing; halving the energy
band costs two thirds of the entries. Certifiability runs the other way, and is
also governed by the energy band, because as \(\epsilon_a\to0\) the admissible
kinetic energy still tends to \(\epsilon_E\). Worst CLF margin
\(\min_{|\tau|\le\tau_{\max}} \dot V_2\) over \(\Sigma_2\), in raw \(e^TPe\)
units, with the fraction of the region where it is negative:

| \(\epsilon_E\) | \(\tau_{\max}=64\) | \(48\) | \(32\) |
|---|---|---|---|
| 2.00 J | +521 (100%) | +2694 (73%) | +4867 (41%) |
| 1.50 J | −9.0 (100%) | +1578 (89%) | +3528 (49%) |
| 1.00 J | −5.7 (100%) | +502 (99%) | +2191 (66%) |
| 0.50 J | −5.2 (100%) | −5.0 (100%) | +955 (92%) |

At \(\epsilon_E\le1.5\ \mathrm{J}\) the margin is negative on the whole region
at 64 N·m. Its magnitude stays small because the region contains upright, where
both the drift and the torque authority vanish; the infimum of the margin over
any region containing the equilibrium is zero by continuity, so the test asserts
that the margin never turns positive rather than that it is bounded away from
zero.

The defaults are therefore \(\epsilon_a = \pi/30\), \(\epsilon_E = 1\
\mathrm{J}\): Lai's energy band unchanged, their angle box tightened to where
the local piece is certifiable at the actuator this protocol uses.

Note that \(-Ke\) still demands about 104 N·m on that boundary. The CLF margin
is the weaker property that some *admissible* torque decreases \(V_2\), which is
the one that matters when no balancing controller is ever switched in — the
standing constraint in the region-based note.

### Entering the region is not holding it

That gap between the CLF margin and the linear gain is not academic. Pairing
Xin–Kaneda swing-up with the published feedback, latched on first entry to
\(\Sigma_2\), over 98 releases from rest at displacements \(0.02\)–\(0.5\) rad
(2 ms, RK4):

| \(\tau_{\max}\) | enter | hold upright | largest \(V\) rise |
|---|---|---|---|
| 64 N·m | 30 / 98 | **10 / 30** | \(+7.8\times10^{1}\) |
| 104 N·m | 28 / 92 | 16 / 28 | \(+2.0\) |
| 150 N·m | 28 / 92 | 28 / 28 | \(+2.4\) |

Localizing where the value rises separates the construction from the
controller cleanly:

| phase | largest rise |
|---|---|
| swing-up under the Xin–Kaneda law | \(\sim10^{-7}\) |
| crossing the transition band | **exactly \(0\)** |
| after the switch, under \(-Ke\) | \(10^{-5}\) to \(1.1\times10^{-1}\), or \(+78\) when the balance fails |

Every rise comes from the linear balance law overshooting inside the region.
The swing-up contributes discretization noise, and the band — where the blend
carries the \(\nabla\mu\) term and neither law is guaranteed to decrease the
mixture — contributed no rise at all on any of these trajectories. That is
encouraging for the open question flagged under *Important limitation* above,
though a sweep is not a proof.

The reading for a learned policy is that \(\Sigma_2\) is the right *reward*
boundary while a linear gain is the wrong thing to hand control to at 64 N·m.
`benchmarks/render_acrobot_nslf.py` renders this pairing, and pins its release
rather than sampling one, for exactly this reason.

### Normalization and the offset

Both pieces are divided by one scale, the Xin–Kaneda value at hanging rest
\(E_s^2/2 = 1200.5\), so their levels are comparable at all:

\[\bar V = (1-\mu)\,\frac{V_X+\Delta}{E_s^2/2} + \mu\,\frac{e^TPe}{E_s^2/2}.\]

At the defaults \(\Delta = 1215.4\), or \(\bar\Delta = 1.0124\) — just over one
unit of the hanging value. Lai's own choice for \(\Delta\) is the crude bound
\(\sum|P_{ij}|x_{i,\max}x_{j,\max}\) of their eq. (72), which at these
tolerances gives 6910 — 5.7× the true maximum, enough to put the offset almost
six times above the entire range of the swing-up piece;
`max_local_value_on_region` computes the maximum instead. For each
admissible pose the energy band caps the kinetic energy, so a convex quadratic
attains its maximum on the resulting velocity ellipsoid's boundary; that
boundary is searched exactly and the poses are gridded. The value is converged:
grids of 61², 91² and 361² poses agree to five figures.

Because \(\Delta\) is constant it shifts the value and never the gradient, so a
reward built on this function has exactly the same shaping during swing-up as
one built on the bare Xin–Kaneda value, and shifts `r3` by a constant.

### What it fixes

Along the homoclinic orbit approaching upright:

| | `GatedLyapunov` | `NonsmoothLyapunov` |
|---|---|---|
| value on the orbit, outside the gate | 0.0000 | 1.0124 |
| peak value inside the transition band | **0.0491** | no rise |
| largest increase along the orbit | \(+7.4\times10^{-4}\) | \(0\) |
| value at upright | 0 | 0 |

### The remaining cost: the transition band in `r3`

The blend puts \(\nabla\mu\,(\bar V_L-\bar V_X)\) into the gradient, and
\(|\nabla\mu|\) scales as one over the band width. Since `r3` carries
\(-\eta\dot V\), that term is felt exactly where the hand-off happens.
Measuring \(|\dot V|\) at the 99th percentile under random \(|\tau|\le64\):

| construction | in band | outside | ratio |
|---|---|---|---|
| `GatedLyapunov`, band 0.02 wide in a 0.04 gate | 446 | 9.9 | 45× |
| `NonsmoothLyapunov`, `transition_fraction = 0.5` | 76 | 10.0 | 7.7× |

The wider region is what makes the band affordable. `transition_fraction` moves
it further: at \(\pi/45\), the ratio falls 9.2× → 5.4× → 4.1× → 3.9× as the
fraction goes 0.75 → 0.5 → 0.25 → 0.1, against a shrinking fully-local core.
The default 0.5 is the compromise. A hard switch is also admissible for a
potential — with \(\Delta\) in place it never steps up — but then the shaping
term has to be the discrete increment \(\gamma\Phi(x')-\Phi(x)\) rather than
the continuous-time \(\dot V\).

### Tests

Beyond the gated candidate's own tests: the published Lai feedback, exact
membership against eq. (17), conservatism of the smooth residual and its
analytic gradient, \(\Delta\) dominating \(e^TPe\) over the region and its
insensitivity to the search grid, gate endpoints and containment, endpoint
selection of the two pieces, the offset leaving the swing-up gradient
unchanged, the analytic gradient in all three gate zones, monotonicity along
the homoclinic orbit against the ridge it replaces, and the CLF margin staying
negative over the region at 64 N·m.
