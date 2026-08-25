# Why 1 ms control broke CT-SAC on Acrobot-XK

## 1. What the actor is actually trained to do

CT-SAC's actor loss, at every gradient step, is

$$
\mathcal{L}_{\text{actor}} = \text{price}(\alpha)\cdot \log \pi_\theta(a \mid s) \;-\; Q(s, a), \qquad a \sim \pi_\theta(\cdot \mid s)
$$

with $a$ drawn from the policy via the reparameterization trick, and

$$
\text{price}(\alpha) =
\begin{cases}
\alpha \cdot \Delta t_{\text{ref}} & \text{if the reward is a rate} \\
\alpha & \text{otherwise}
\end{cases}
$$

Gradient ascent on $-\mathcal{L}_{\text{actor}}$ pushes $\theta$ in the direction that increases $Q(s, \pi_\theta(s))$. That is the *entire* mechanism by which the actor learns anything: **it only ever sees the critic's opinion of an action, never the environment's reward directly.**

So whether the policy learns to hold the acrobot in the tube depends entirely on whether $Q(s,\cdot)$ actually prefers the actions that keep it there.

## 2. What $Q(s,a)$ means at a fixed control interval

For a control step of length $dt$, the one-step Bellman relation is

$$
Q^\pi(s,a) \;=\; r(s,a)\cdot dt \;+\; \gamma \cdot V^\pi(s'), \qquad \gamma = e^{-\lambda\, dt}
$$

where $s'$ is the state reached after applying $a$ for one interval of length $dt$, and $\gamma$ is the continuous-time discount over that interval — the same structure CT-SAC's own critic target uses (an immediate term scaled by $\Delta t_{\text{ref}}$, plus a $\gamma_{dt}$-weighted continuation for the rest).

Read literally: $Q(s,a)$ is *"the value of taking action $a$ for one control interval, then reverting to the policy $\pi$ for everything after."*

## 3. The comparison the actor's gradient actually makes

The actor doesn't need $Q(s,a)$ in isolation — it needs to know which of two nearby actions is better, since that is what a gradient step compares. Take the controller's action $a_1$ and a deviated action $a_2 = a_1 + \Delta$, both at the same state $s$. Their difference splits into exactly two pieces:

$$
Q(s,a_1) - Q(s,a_2) \;=\;
\underbrace{\big[r(s,a_1) - r(s,a_2)\big]\cdot dt}_{\text{immediate}}
\;+\;
\gamma\underbrace{\big[V^\pi(s_1') - V^\pi(s_2')\big]}_{\text{future}}
$$

The **future** term is the only place a value function can express "this action leads somewhere better in the long run." Everything about the acrobot's homoclinic dynamics, the shape of the tube, the horizon of the task — all of it has to enter through that one term.

## 4. Why the future term vanishes as $dt \to 0$

$s_1'$ and $s_2'$ are the states reached after **one** control step under $a_1$ and $a_2$ respectively. As $dt \to 0$, an action only has $dt$ seconds to act on the physics before the *next* action — sampled from the same policy $\pi$ in both branches — takes over. So

$$
\lim_{dt \to 0} \big(V^\pi(s_1') - V^\pi(s_2')\big) = 0,
$$

because $s_1', s_2' \to s$ together. Measured directly on this task (analytical controller, in-tube states, a $\Delta = 5.8\ \text{N·m}$ torque deviation, $dt = 1\ \text{ms}$):

| after one 1 ms step | value |
|---|---|
| change in joint angle | $\sim 2\times10^{-5}\ \text{rad}$ |
| change in joint velocity | $\sim 0.07\ \text{rad/s}$ |
| resulting change in $V^\pi$ | $95\%\ \text{CI} = [-0.016,\ +0.013]$ — indistinguishable from 0 |

So at $dt = 1\ \text{ms}$, $Q(s,a_1) - Q(s,a_2)$ is dominated by noise. Both terms — immediate and future — are close to zero.

## 5. What "indistinguishable from zero" means here

That claim is doing real work in the argument, so it is worth stating precisely.

The measurement takes $n = 20$ in-tube states $s_1,\dots,s_n$ and, for each, computes

$$
A_i \;=\; G\big(\text{deviate at } s_i \text{ for one step, then follow } \pi\big) \;-\; G\big(\text{follow } \pi \text{ throughout}\big)
$$

where $G$ is the discounted return over a 2 s horizon. Both the dynamics and the controller are deterministic, so **each $A_i$ is an exact number, not a noisy estimate.** There is no measurement error anywhere in this.

The quantity being estimated is therefore the mean advantage over the distribution of in-tube states,

$$
\mu \;=\; \mathbb{E}_{s \sim \rho_{\text{tube}}}\big[A(s)\big],
$$

and the usual estimator and its spread are

$$
\bar{A} = \frac{1}{n}\sum_{i=1}^{n} A_i,
\qquad
s^2 = \frac{1}{n-1}\sum_{i=1}^{n}\big(A_i - \bar{A}\big)^2,
\qquad
\mathrm{SE} = \frac{s}{\sqrt{n}}.
$$

For the 1 ms measurement:

$$
\bar{A} = -0.00532, \qquad s = 0.0241, \qquad \mathrm{SE} = 0.00539, \qquad n = 20.
$$

Testing $H_0: \mu = 0$ gives

$$
t \;=\; \frac{\bar{A}}{\mathrm{SE}} \;=\; \frac{-0.00532}{0.00539} \;=\; -0.99,
\qquad
t_{0.975,\,19} = 2.09,
$$

so $|t| < t_{0.975,19}$ and $H_0$ is not rejected. Equivalently, the 95% confidence interval

$$
\bar{A} \pm t_{0.975,\,19}\cdot \mathrm{SE} \;=\; [-0.0159,\ +0.0053]
$$

contains zero.

**What this does not say.** It does not say $\mu = 0$. Failing to reject a null is not evidence for it. What the interval provides is an *upper bound*: with 95% confidence, $|\mu| < 0.016$.

**Where the spread comes from.** Since each $A_i$ is exact, $s = 0.0241$ is not noise — it is genuine heterogeneity across the tube. At some in-tube states the deviation happens to push in the direction the controller was already going and helps; at others it hurts. The mean displacement is about a fifth of the spread, so both signs occur across the tube — in a separate 12-state run where the individual values were recorded, six of twelve came out positive. In effect-size terms,

$$
d \;=\; \frac{\bar{A}}{s} \;=\; \frac{-0.00532}{0.0241} \;=\; -0.22,
$$

a small mean displacement relative to the state-to-state variation it sits inside.

**What it would take to resolve.** Since $\mathrm{SE}$ falls as $1/\sqrt{n}$, reaching $|t| = 3$ at the observed effect size needs

$$
n \;\geq\; \left(\frac{3s}{\bar{A}}\right)^{2} \;=\; \left(\frac{3 \times 0.0241}{0.00532}\right)^{2} \;\approx\; 185
$$

states, against the 20 used. So the effect is resolvable in principle; it is simply far too small to matter.

**Why the bound is enough for the argument.** Every conclusion below rests on comparisons where even the *most generous* end of the interval is negligible. Taking $|\mu| < 0.016$ at face value:

$$
\underbrace{0.016}_{\text{upper bound on one-step}}
\;\ll\;
\underbrace{0.047}_{\text{critic's spurious action-range noise}}
\;\lll\;
\underbrace{7.10}_{\text{cost of the persistent bias}}
$$

The one-step advantage is bounded below the critic's own fitting error, which is in turn two orders of magnitude below the effect that actually decides tube residence. Whether $\mu$ is exactly zero or merely small changes none of that.

## 6. But the physical consequence of the deviation is not zero

That comparison — *one* deviated step, then back to $\pi$ — is not the thing that determines whether the acrobot stays in the tube. What determines that is a different quantity: what if the policy is biased by $\Delta$ on **every** step, forever? Call the persistently-biased policy $\pi_\Delta$:

$$
\Delta_{\text{persistent}} \;=\; V^{\pi_\Delta}(s) \;-\; V^{\pi}(s)
$$

Measured under the same conditions, with the same $\Delta = 5.8\ \text{N·m}$ error:

$$
\underbrace{-0.005}_{\text{one-step, }Q(s,a_1)-Q(s,a_2)} \qquad \text{vs.} \qquad \underbrace{-7.10}_{\Delta_{\text{persistent}}}
$$

The one-step figure is not distinguishable from zero; the persistent figure is measured cleanly, with all 20 test states agreeing in sign, and the tube residence collapses from 100% to 1% of the episode. This is a real, large, unambiguous effect — it is simply not the quantity $Q(s,a)$ measures:

$$
\frac{|\Delta_{\text{one-step}}|}{|\Delta_{\text{persistent}}|} \;\approx\; \frac{1}{1400}
$$

$Q(s,a)$ prices a **one-step** deviation; a systematically biased policy is a **persistent** deviation. These are different objects.

## 7. Why the actor cannot bridge the gap by iterating

In principle, a small-but-real one-step advantage should still compound correctly over many actor updates — that is the premise behind policy iteration. This would still work here **if the one-step signal were merely small but accurate.** The problem is that it is not accurate — it is smaller than the critic's own approximation error.

Measured by sweeping the *entire* action range (not just one deviation) at fixed in-tube states:

$$
\underbrace{\max_a Q_\phi(s,a) - \min_a Q_\phi(s,a)}_{\approx\ 0.050,\ \text{learned critic}}
\qquad\text{vs.}\qquad
\underbrace{\max_a Q^\pi(s,a) - \min_a Q^\pi(s,a)}_{\approx\ 0.003,\ \text{true value}}
$$

Of the $0.050$ the critic reports, only $\sim 0.003$ corresponds to anything real; the remaining $\sim 0.047$ is the critic's own fitting noise. A direct rank-correlation test between the critic's action ordering and the true one-step ordering confirmed this — Spearman $\rho$ consistent with zero across critic heads, with a confidence interval spanning zero in every case. So the actor's gradient at $dt = 1\ \text{ms}$ is overwhelmingly composed of noise rather than signal, independent of whether that noise happens to point toward or away from the tube on any given update.

## 8. Why this looked like "the imitation term is a crutch"

Every training run in this study included a term pulling the policy toward the analytical controller's action $a^*(s)$:

$$
\mathcal{L}_{\text{actor}} = \text{price}(\alpha)\cdot\log\pi_\theta(a\mid s) \;-\; Q(s,a) \;+\; c\cdot \mathcal{L}_{\text{imit}}\big(\pi_\theta, a^*\big)
$$

$\mathcal{L}_{\text{imit}}$ — whether a KL to a smeared controller distribution or an MSE to its mean action — penalizes the **persistent** gap between $\pi$ and $a^*$ directly, at every state, without routing through $Q$ at all. It supplies exactly the $O(1)$ signal that $Q(s,a)$ cannot.

That is why every arm behaved identically once $c$ was annealed to zero: capture dropped from $\sim 0.3\text{–}0.5$ to **exactly $0$ on the precise step $c$ reached zero**, in all six seeds, for two structurally different imitation losses (KL and MSE), at two horizons. The moment the only $O(1)$ term in the actor's objective vanished, all that remained was $Q(s,a)$ — and at $dt = 1\ \text{ms}$, $Q(s,a)$ carries almost no usable information about which action to take.

## 9. Summary

At a 1 ms control interval, the actor's objective $Q(s,a)$ is built to answer the question *"what if I deviate for one millisecond?"* — whose true answer is smaller than the critic's own approximation error. The question that actually decides the outcome, *"what if the policy is biased on every step?"*, is asked by nothing except the imitation term. Once that term is removed, the actor has no functioning gradient at all.
