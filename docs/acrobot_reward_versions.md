# Acrobot swing-up reward versions: per-step reward calculations

Every local Acrobot task shares one mechanism, one reset, and one set of geometric primitives; only the per-step reward differs. This document states each version's reward exactly.

## Shared setup

**Mechanism.** Two length-1 capsule links. Shoulder pivot at $z = 2$; only the elbow is actuated (gear 2, control range $[-1, 1]$); the shoulder is passive. Joint damping 0.05, timestep 0.01 s, RK4. Fully extended upright the tip is at $z = 4$; hanging, the tip is at $z = 0$.

**Reset.** Start near the fully hanging pose — shoulder $= \pi$, elbow $= 0$ — with uniform noise $\pm 0.05$ rad on each angle and $\pm 0.01$ on each velocity. Reseeding makes fixed evaluation starts repeatable. This replaces the stock uniform $[-\pi, \pi)$ reset. (v5 deviates: uniform random start by default — see its section.)

**Shared geometric terms** (recomputed each step from the physics state):

- $d = \lVert \text{tip} - \text{target} \rVert$, the tip-to-target distance; target at $(0, 0, 4)$, radius $r = 0.2$.
- $\text{precise} = \operatorname{tol}(d,\ (0, r),\ \text{margin}=1)$ — the stock dense target reward. $\text{precise} = 1$ at $d \le 0.2$ and decays to $0.1$ at $d = 1.2$.
- $\bar{u} = \tfrac{1}{2}(u_1 + u_2)$, the mean link uprightness, with $u_i = \operatorname{clip}\!\big((v_i + 1)/2,\ 0,\ 1\big)$ and $v_i$ the vertical ($z$) component of link $i$; $u_i = 1$ when link $i$ points straight up, $0$ straight down.
- $\text{extension} = \operatorname{clip}\!\big((1 + \cos\theta_\text{elbow})/2,\ 0,\ 1\big)$, $= 1$ when the elbow is straight, $0$ fully folded.
- Reward rate convention: every version returns a per-step reward; episode return is the (discounted) sum over the 10 s episode.

$\operatorname{tol}(x,\ (a,b),\ \text{margin}=m)$ is $1$ inside $[a, b]$, decaying on a Gaussian sigmoid to $0.1$ at distance $m$ outside, $\to 0$ beyond.

---

## v1 — stock swing-up (baseline)

$$\text{reward} = \text{precise} = \operatorname{tol}(d,\ (0, 0.2),\ \text{margin}=1)$$

The stock task: a single narrow Gaussian on tip-to-target distance. Near the hanging pose $d \approx 4$, so $\text{precise} \approx 0$ and there is no gradient toward the goal; combined with the stock uniform $[-\pi, \pi)$ reset, learning is reset-luck dominated.

## v2 — tip-distance progress + precise tail

$$\text{progress} = \operatorname{clip}(1 - d/4,\ 0,\ 1)$$
$$\text{reward} = 0.8\,\text{progress} + 0.2\,\text{precise} \qquad (\text{clipped to } [0, 1])$$

A linear tip-distance ramp gives dense signal from the hanging pose (fixing v1), with a small precise term near the exact goal. Kept verbatim for checkpoint provenance.

## v3 — anti-fold progress + precise tail

$$\text{progress} = \text{extension} \cdot \bar{u}$$
$$\text{reward} = 0.8\,\text{progress} + 0.2\,\text{precise} \qquad (\text{clipped to } [0, 1])$$

Replaces v2's distance ramp with a pose-purity term: reward the configuration of being extended ($\text{extension}$) and upright ($\bar{u}$) rather than mere tip proximity, removing the bent-hover ridge.

## v4 — energy regulation + velocity-gated hold

$$\tilde{E} = \frac{E - E_\text{hang}}{E_\text{up} - E_\text{hang}}$$
$$\text{ramp} = \operatorname{tol}(\tilde{E},\ (1, 1),\ \text{margin}=1) \cdot \frac{1 + \bar{u}}{2}$$
$$\text{slow} = \operatorname{tol}(\lVert \dot{q} \rVert,\ (0, 0.5),\ \text{margin}=2)$$
$$\text{hold} = \text{precise} \cdot \text{slow}$$
$$\text{reward} = 0.2\,\text{ramp} + 0.8\,\text{hold} \qquad (\text{clipped to } [0, 1])$$

Mechanical energy $E = \tfrac{1}{2}\dot{q}^{\top} M(q)\,\dot{q} - \sum_i m_i\,\vec{g}\cdot\vec{x}_i$. $E_\text{hang}$ and $E_\text{up}$ are measured at the hanging-rest and upright-rest poses at episode start, so the normalized energy $\tilde{E} = 0$ at hanging rest and $\tilde{E} = 1$ at upright rest.

- **ramp** pays for holding total energy near the upright-rest level. Any action moving $E$ toward $E_\text{up}$ raises $\operatorname{tol}(\tilde{E},\ (1,1),\ 1)$ regardless of pose, so the pumping motion v3 zeroed is now rewarded directly; energy overshoot (spinning) is discounted symmetrically. The $(1 + \bar{u})/2$ tilt halves the value of parking on the $\tilde{E} = 1$ manifold away from the top (e.g. holding $\tilde{E} = 1$ as kinetic energy at the bottom).
- **hold** is the precise reward gated by a Gaussian tolerance on speed $\lVert \dot{q} \rVert$. Sustained near-1 income exists only while balancing slowly at the exact target.

At $E \approx E_\text{up}$ the passive dynamics pass through upright arbitrarily slowly (the homoclinic orbit), so a policy that has learned the ramp reaches the hold region at low speed by construction — capture is discoverable without fighting the dense term.

Per-step audit, worst sustainable off-goal income $\approx 0.2$ against 1.0 at the goal:

| state | v4 | ramp | hold | $\tilde{E}$ | v3 | v2 |
|---|---|---|---|---|---|---|
| hanging rest | 0.010 | 0.050 | 0.000 | 0.00 | 0.000 | 0.000 |
| upright rest (goal) | 1.000 | 1.000 | 1.000 | 1.00 | 1.000 | 1.000 |
| fold-up static | 0.130 | 0.649 | 0.001 | 0.75 | 0.000 | 0.400 |
| bent hover, wobbling | 0.205 | 0.966 | 0.014 | 1.01 | 0.758 | 0.690 |
| slow pass near goal | 0.990 | 0.998 | 0.989 | 1.01 | 0.997 | 0.958 |
| fast spin at top | 0.138 | 0.666 | 0.006 | 1.42 | 1.000 | 1.000 |
| fast swing at bottom | 0.063 | 0.313 | 0.000 | 0.55 | 0.000 | 0.000 |

**Model-free pilot outcome** (γ = 0.995, 500 k): first genuine swing-up of the whole series — best checkpoint reaches tip $z = 4.0$ and clears tip $z > 3$ on 47.5 % of eval episodes, passing within $d = 0.013$ of the target. But every learned policy passes the top with surplus energy ($\tilde{E} > 1$, fast), so the hold term never triggers (occupancy $\approx 0.001$): swing-*through* at rate $\approx 0.08$–$0.19$, not capture at $\approx 1$.

## v4.1 — asymmetric energy margin (capture pressure)

Identical to v4 except the energy tolerance margin is piecewise: 1.0 below the upright-rest energy, $0.25$ above it.

$$\text{ramp} = \operatorname{tol}\!\big(\tilde{E},\ (1, 1),\ \text{margin} = \begin{cases}1.0 & \tilde{E} \le 1\\ 0.25 & \tilde{E} > 1\end{cases}\big) \cdot \frac{1 + \bar{u}}{2}$$

Everything at or below $\tilde{E} = 1$ — the entire pumping ramp, every audit row above except the two overshoot states — is unchanged, so v4 checkpoints remain meaningfully comparable. Above $\tilde{E} = 1$ the ramp income collapses exactly in the regime the v4 pilots converged to:

| $\tilde{E}$ | v4 energy factor | v4.1 |
|---|---|---|
| 1.00 | 1.000 | 1.000 |
| 1.10 | 0.977 | 0.692 |
| 1.25 | 0.866 | 0.100 |
| 1.50 | 0.562 | $\approx 0$ |

Passing the top with surplus energy now loses the dense income, so the policy is pushed to regulate $\tilde{E} \to 1$ — where top passes are slow by the homoclinic argument and the hold term is enterable. On scripted trajectories the preference for an energy-regulated pump over an overshooting one rises from $2.31\times$ (v4) to $3.20\times$ (v4.1). Capture also fixes the evaluation instability seen in the v4 pilots: balance at the top is a fixed point, so a capturing policy has a stable deterministic readout, where the swing-through limit cycle is phase-critical and its greedy readout is bistable.

**Uniform random starts** (`uniform_start=True`, the v4.1 default). The hanging-start v4.1 pilot removed its own discovery path: the capture-pressured reward has its maximum at the slow hold on the $\tilde{E} = 1$ manifold, but from hanging that region is reachable only through the overshoot the margin now penalizes. The result was strictly worse than v4 — CT-SAC never even reached the height on held-out starts (max tip $2.0$, height/hold occupancy $0$), and the best_model gate stayed empty. Starting from uniform random joint angles instead puts near-top, near-$\tilde{E} = 1$ states directly in the start distribution: 18 % of resets begin above the height, and averaged over the whole start stream the hold reward is $\approx 0.07$ — already above the 0.05 best_model gate before any learning — so the hold is trained directly and its value propagates outward to lower-energy starts. Energy calibration is pose-independent and composes with the reset unchanged. `uniform_start=False` restores the near-hanging reset (and the overshoot margin defaults to 1.0, so `BalanceV4(uniform_start=False)` reproduces v4 exactly).

## v5 — height occupancy (unshaped control arm)

$$\text{reward} = \begin{cases} 1 & \text{tip } z > 3\\ 0 & \text{otherwise} \end{cases}$$

Unshaped occupancy of the Gym height criterion ($-\cos\theta_1 - \cos(\theta_1{+}\theta_2) > 1 \iff$ tip $z > 3$): the return is the physical time the tip spends above one link length over the pivot, accumulated over the fixed-length episode with no termination. Zero signal below the height, so there is no parking surface; maximal income is *staying* above the height, which makes balancing near the top the implicit optimum without any velocity gate or target shaping.

**Uniform random starts** (`uniform_start=True`, the default): episodes begin at uniform random joint angles with near-zero velocity — the stock-style reset — instead of the shared near-hanging pose. 18.5 % of such resets already start above the height, so the sparse income appears in the replay data from the first episodes and its value can propagate outward to progressively lower starts; from the hanging start alone the reward would never be observed (nothing unshaped has ever exceeded tip $z = 1.87$ here). Above-height resets are unstable inverted poses, so collecting their income immediately trains the balance skill. `uniform_start=False` restores the shared near-hanging reset for from-hanging probes.

Runs use 30 s episodes (a competent scripted pump first crosses at $t \approx 10$–11.5 s from hanging, leaving up to $\sim 20$ s of collectable occupancy) and $\gamma \in \{0.999, 0.9995\}$. It isolates whether v4's shaping is necessary: v4 logs the same height criterion continuously, so the pair answers whether the unshaped objective is learnable at 100 Hz continuous torque when the signal is reachable from the start distribution.

---

## Summary

| ver | dense term | goal/tail term | intent | outcome |
|---|---|---|---|---|
| v1 | — | $\text{precise}$ | baseline | no signal from hang (best 43) |
| v2 | $0.8\cdot\operatorname{clip}(1 - d/4)$ | $0.2\cdot\text{precise}$ | dense from hang | bent-hover attractor (664–683) |
| v3 | $0.8\cdot\text{extension}\cdot\bar{u}$ | $0.2\cdot\text{precise}$ | anti-fold pose | zeros pumping ($\approx 230$–260, tip $\le 1.87$) |
| v4 | $0.2\cdot\text{ramp}(\tilde{E}, \bar{u})$ | $0.8\cdot\text{precise}\cdot\text{slow}$ | reward pumping | swing-up found (tip 4.0, 48 % over height) but fast swing-through; no capture |
| v4.1 | v4 with overshoot margin $1.0 \to 0.25$ | unchanged | regulate $\tilde{E}\to 1$, make capture the attractor | hanging start failed (no capture); uniform-start rerun queued |
| v5 | — | $\mathbb{1}[\text{tip } z > 3]$ occupancy | unshaped control arm, uniform random starts | learnable, height occupancy $\le 0.12$ held-out; partial balance |

All reward outputs are in $[0, 1]$.

## Held-out evaluation (20 seeds/checkpoint)

Each checkpoint is evaluated from both start distributions side by side (`evaluations/eval_acrobot_v41_v5.py`, `start` column): `uniform` — the training reset — and `hanging`, the canonical swing-up-from-down task. The hanging column is the true-task capability; the numbers below are the `uniform` pass. Height occupancy is the dt-weighted time fraction with tip $z > 3$; hold occupancy is the v4 velocity-gated exact-target term.

| version | framework | max tip | frac tip $>3$ | height occ | hold occ |
|---|---|---|---|---|---|
| v4.1 (hanging) | CT-SAC | 2.02 | 0.00 | 0.000 | 0.000 |
| v4.1 (hanging) | SB3 SAC | 3.53 | 0.35 | 0.016 | 0.002 |
| v4.1 (hanging) | SB3 PPO | 3.91 | 0.70 | 0.042 | 0.001 |
| v5 (uniform) | CT-SAC | 4.00 | 0.60 | 0.080 | — |
| v5 (uniform) | SB3 PPO | 4.00 | 0.80 | 0.118 | — |

Two readings drive the v4.1 uniform-start rerun: hanging-start v4.1 either never reaches the height (CT-SAC) or reaches without holding (SB3, hold occ $\approx 0$); and v5 shows uniform starts convert an unlearnable-from-hanging objective into a partially learnable one. v4.1's hold term is a stronger balance signal than v5's raw occupancy, so uniform-start v4.1 is the combined bet: reachability from the start distribution, plus a reward that specifically shapes the slow exact-target capture. Occupancy at $\le 0.12$ even for v5 means no arm yet sustains balance broadly; capture remains the open problem.
