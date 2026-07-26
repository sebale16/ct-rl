# Acrobot swing-up reward versions

One mechanism, one set of geometric primitives. Across versions only the
per-step reward changes, and from v4.1 the episode reset as well. Each section
states the reward, the outcome it produced, and the move that outcome forced.

## Shared setup

**Mechanism.** Two length-1 capsule links. Shoulder pivot at $z = 2$; only the
elbow is actuated (gear 2, torque range $[-1, 1]$), the shoulder is passive.
Joint damping 0.05, timestep 0.01 s. Fully extended upright the tip is at
$z = 4$; hanging, the tip is at $z = 0$; the Gym height criterion
$-\cos\theta_1 - \cos(\theta_1{+}\theta_2) > 1$ is tip $z > 3$.

**Reset.** Episodes start near the fully hanging pose (shoulder $= \pi$, elbow
$= 0$) with small angle and velocity noise. From v4.1 the training reset moves
off hanging — uniform random angles, then in v4.2 a curriculum that widens over
training — while evaluation always fixes the start at either hanging or uniform.

**Geometric primitives**, recomputed each step:

- $d = \lVert \text{tip} - \text{target} \rVert$, target at $(0, 0, 4)$, radius $0.2$.
- $\text{precise} = \operatorname{tol}(d,\ (0, 0.2),\ \text{margin}=1)$: the stock target reward, $1$ at $d \le 0.2$, decaying to $0.1$ at $d = 1.2$.
- $\bar{u} = \tfrac{1}{2}(u_1 + u_2)$, mean link uprightness, $u_i \in [0, 1]$ with $u_i = 1$ when link $i$ points straight up.
- $\text{extension} = \operatorname{clip}((1 + \cos\theta_\text{elbow})/2,\ 0,\ 1)$, $= 1$ when the elbow is straight.
- $\operatorname{tol}(x,\ (a,b),\ \text{margin}=m)$ is $1$ inside $[a, b]$, decaying on a Gaussian sigmoid to $0.1$ at distance $m$ outside.

Every version returns a per-step reward in $[0, 1]$; the return is the
discounted sum. Episodes are 10 s through v4, 20 s for v4.1/v4.2, 30 s for v5.

---

## v1 — stock swing-up

$$\text{reward} = \text{precise}$$

A single narrow Gaussian on tip-to-target distance. Near hanging $d \approx 4$,
so the reward is flat at zero and carries no gradient toward the goal; with the
stock uniform reset, learning is dominated by reset luck (best return 43).

**Move to v2:** supply a dense signal that reaches the hanging pose.

## v2 — tip-distance progress + precise tail

$$\text{reward} = 0.8\,\operatorname{clip}(1 - d/4,\ 0,\ 1) + 0.2\,\text{precise}$$

A linear tip-distance ramp gives signal everywhere. All modes and seeds pinned
at 664–683: the ramp creates a bent hover just below the target that collects
$\approx 0.7$ per step indefinitely, and any capture attempt first crosses a
low-reward gap that costs that income, so capture is never attempted.

**Move to v3:** reward the configuration of being up rather than tip proximity,
removing the bent-hover ridge.

## v3 — anti-fold pose progress + precise tail

$$\text{reward} = 0.8\,(\text{extension}\cdot\bar{u}) + 0.2\,\text{precise}$$

Reward extended-and-upright pose instead of distance. Plateaus at 230–260 and
never lifts the tip above $z = 1.87$, below the shoulder mount. Energy pumping
is rhythmic elbow bending, and the extension factor zeroes the dense term
exactly during that motion, so pumping earns no more than aimless swinging.

**Move to v4:** pay for the pumping motion itself.

## v4 — energy regulation + velocity-gated hold

$$\tilde{E} = \frac{E - E_\text{hang}}{E_\text{up} - E_\text{hang}}, \qquad E = \tfrac{1}{2}\dot{q}^{\top} M(q)\,\dot{q} - \textstyle\sum_i m_i\,\vec{g}\cdot\vec{x}_i$$
$$\text{ramp} = \operatorname{tol}(\tilde{E},\ (1, 1),\ \text{margin}=1) \cdot \tfrac{1 + \bar{u}}{2}$$
$$\text{hold} = \text{precise} \cdot \operatorname{tol}(\lVert \dot{q} \rVert,\ (0, 0.5),\ \text{margin}=2)$$
$$\text{reward} = 0.2\,\text{ramp} + 0.8\,\text{hold}$$

$E_\text{hang}$ and $E_\text{up}$ are measured at rest at episode start, so the
normalized energy $\tilde{E} = 0$ hanging and $1$ upright. The **ramp** pays for
holding total energy near the upright-rest level: any elbow motion that pumps
energy toward $\tilde{E} = 1$ raises it regardless of pose, and overshoot is
discounted symmetrically. The $(1 + \bar{u})/2$ tilt halves parking on the
$\tilde{E} = 1$ manifold away from the top. The **hold** pays the precise target
reward gated on low speed, so sustained near-1 income exists only while
balancing slowly at the goal. At $\tilde{E} \approx 1$ the passive dynamics pass
through upright arbitrarily slowly, so a policy that learned the ramp reaches
the hold region at low speed by construction.

Per-step audit; worst sustainable off-goal income $\approx 0.2$ against 1.0 at the goal:

| state | v4 | ramp | hold | $\tilde{E}$ | v3 | v2 |
|---|---|---|---|---|---|---|
| hanging rest | 0.010 | 0.050 | 0.000 | 0.00 | 0.000 | 0.000 |
| upright rest (goal) | 1.000 | 1.000 | 1.000 | 1.00 | 1.000 | 1.000 |
| fold-up static | 0.130 | 0.649 | 0.001 | 0.75 | 0.000 | 0.400 |
| bent hover, wobbling | 0.205 | 0.966 | 0.014 | 1.01 | 0.758 | 0.690 |
| slow pass near goal | 0.990 | 0.998 | 0.989 | 1.01 | 0.997 | 0.958 |
| fast spin at top | 0.138 | 0.666 | 0.006 | 1.42 | 1.000 | 1.000 |
| fast swing at bottom | 0.063 | 0.313 | 0.000 | 0.55 | 0.000 | 0.000 |

**Outcome:** the first genuine swing-up of the series — the best policy reaches
tip $z = 4.0$ and clears the height on 48 % of episodes, passing within
$d = 0.013$ of the target. Every learned policy passes the top with surplus
energy ($\tilde{E} > 1$, fast), so the hold gate never opens: swing-through at
rate $\approx 0.08$–$0.19$ rather than capture at $\approx 1$.

**Move to v4.1:** remove the surplus-energy income so the top pass slows to a
speed the hold gate can reward.

## v4.1 — asymmetric overshoot margin + stricter slow gate

The pumping ramp keeps v4's tolerance at or below $\tilde{E} = 1$ and tightens
it above, and the hold gate narrows:

$$\text{ramp} = \operatorname{tol}\!\Big(\tilde{E},\ (1, 1),\ \text{margin} = \begin{cases}1.0 & \tilde{E} \le 1\\ 0.25 & \tilde{E} > 1\end{cases}\Big) \cdot \tfrac{1 + \bar{u}}{2}, \qquad \text{hold gate} = \operatorname{tol}(\lVert \dot{q} \rVert,\ (0, 0.1),\ \text{margin}=0.5)$$

Above $\tilde{E} = 1$ the ramp income now collapses in exactly the regime the v4
policies converged to:

| $\tilde{E}$ | v4 energy factor | v4.1 |
|---|---|---|
| 1.00 | 1.000 | 1.000 |
| 1.10 | 0.977 | 0.692 |
| 1.25 | 0.866 | 0.100 |
| 1.50 | 0.562 | $\approx 0$ |

**From hanging this removed its own ladder.** The reward now peaks on the slow
$\tilde{E} = 1$ manifold, but from hanging that manifold is reachable only
through the overshoot the margin penalizes, so the from-hanging runs never
captured and never even reached the height. The fix is the **uniform random
start**: drawing initial joint angles uniformly puts near-top, near-$\tilde{E} =
1$ states directly in the start distribution (about 18 % of resets begin above
the height), so the slow capture is trained where it is rewarded and its value
propagates outward to lower-energy starts. Episodes were extended to 20 s to
leave stabilization time after a late arrival, and checkpoints are selected on a
strict capture event — tip within $0.2$ and speed below $0.2$ held for one
continuous second — evaluated separately from both the uniform and the hanging
start.

**Outcome** (6 matched seeds, capture-pressured 20 s reward, evaluated from both
starts):

| method | start | swings up | holds |
|---|---|---|---|
| CT-SAC | uniform | 6/6 seeds | up to 0.12 occupancy — only method that brakes |
| CT-SAC | hanging | 1/6 seeds | 0.00 |
| PPO | uniform | 6/6 seeds | $\approx 0$ — passes through |
| PPO | hanging | 3/6 seeds | $\approx 0$ — passes through |

From the uniform start CT-SAC now genuinely brakes at the top, the first
sustained hold of the series. From hanging the hold stays at zero: a full
from-hanging swing-up arrives too fast for the gate, and the braking is learned
only from the near-top starts where the arrival is already slow. Swing-up from
hanging and the catch live in different parts of the start distribution.

**Move to v4.2:** connect them by scheduling the start energy from near-top down
to hanging, so the catch value learned first extends onto progressively longer
swing-ups.

## v4.2 — reverse-curriculum reset

The per-step reward is identical to v4.1. Only the training reset changes:
episodes begin in a band around the upright whose half-width grows with training
progress, from a narrow near-upright cap up to the full circle. Equivalently the
least-energy reachable start falls from $\tilde{E} \approx 1$ toward $0$ over
training. Early episodes start already near the top, where the slow capture is
directly learnable; as the band widens the start reaches down toward hanging,
carrying the capture value onto longer swing-ups. At full width the reset
coincides with v4.1's uniform draw, so the second half of training runs on the
full task distribution with the catch already in place. Evaluation fixes the
start as before. Outcome: queued.

## v5 — height occupancy (unshaped control arm)

$$\text{reward} = \mathbb{1}[\text{tip } z > 3]$$

The return is the physical time the tip spends above the height, with no
termination and no signal below it, so there is nothing to park on and the
implicit optimum is staying up. Episodes are 30 s from uniform random starts.
This isolates whether v4's shaping is necessary. Outcome: learnable from uniform
starts, height occupancy $\le 0.12$ held-out — partial balance, no sustained
capture, so the shaped velocity-gated hold remains the stronger balance signal.

## v4.3 and v6.1 — mastery-gated tip-state branches

These IDs preserve the v4.1 and v6 rewards respectively and replace only the
reset curriculum. The first reset starts near the target with Cartesian tip
velocity directed toward it, explicitly teaching braking. Subsequent starts
have zero velocity and lower the tip through a discrete height ladder to exact
hanging at rest. A level changes only after deterministic evaluation shows
that the tip was stabilized continuously for one physical second from the
current height; no global-step schedule is involved. See
[`tip_height_curriculum.md`](tip_height_curriculum.md) for the geometry,
threshold, and configuration.

---

## Summary

| ver | reward | outcome |
|---|---|---|
| v1 | $\text{precise}$ | no signal from hang (best 43) |
| v2 | $0.8\,(1 - d/4) + 0.2\,\text{precise}$ | bent-hover attractor (664–683) |
| v3 | $0.8\,\text{extension}\cdot\bar{u} + 0.2\,\text{precise}$ | zeros pumping (230–260, tip $\le 1.87$) |
| v4 | $0.2\,\text{ramp} + 0.8\,\text{hold}$ | swing-up found (tip 4.0, 48 % over height), fast swing-through |
| v4.1 | v4 with overshoot margin $1.0\to0.25$, slow gate $(0,0.1)$; uniform starts, 20 s | from uniform CT-SAC brakes (hold 0.12); from hanging no hold |
| v4.2 | v4.1 reward, reset energy scheduled $\tilde{E}: 1 \to 0$ | queued |
| v4.3 | v4.1 reward, mastery-gated tip height/velocity reset | new branch |
| v5 | $\mathbb{1}[\text{tip } z > 3]$ occupancy | learnable, occupancy $\le 0.12$, partial balance |
| v6.1 | v6 reward, mastery-gated tip height/velocity reset | new branch |
