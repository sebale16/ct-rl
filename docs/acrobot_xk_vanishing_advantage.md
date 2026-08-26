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
| resulting change in $V^\pi$ | $-0.018$, $95\%\ \text{CI} = [-0.025,\ -0.011]$ (see §5) |

So at $dt = 1\ \text{ms}$ both terms of $Q(s,a_1) - Q(s,a_2)$ are tiny: the immediate term because it is multiplied by $dt$, the future term because it evaluates the same $V^\pi$ at two states $0.07\ \text{rad/s}$ apart. Section 5 measures how tiny, and against what.

## 5. How small is "small"? Measuring the one-step advantage properly

That bound is doing real work in the argument, so it is worth measuring rather than asserting.

The measurement takes $n$ in-tube states $s_1,\dots,s_n$ and, for each, computes

$$
A_i \;=\; G\big(\text{deviate at } s_i \text{ for one step, then follow } \pi\big) \;-\; G\big(\text{follow } \pi \text{ throughout}\big)
$$

where $G$ is the discounted return over a 2 s horizon. Both the dynamics and the controller are deterministic, so **each $A_i$ is an exact number, not a noisy estimate.** There is no measurement error anywhere in this. The quantity being estimated is the mean advantage over the distribution of in-tube states,

$$
\mu \;=\; \mathbb{E}_{s \sim \rho_{\text{tube}}}\big[A(s)\big],
\qquad
\bar{A} = \frac{1}{n}\sum_{i=1}^{n} A_i,
\qquad
\mathrm{SE} = \frac{s}{\sqrt{n}},
\qquad
t = \frac{\bar{A}}{\mathrm{SE}}.
$$

A first pass at $n = 20$, drawn from a single evaluation episode, could not separate $\mu$ from zero:

$$
\bar{A} = -0.00532,\quad s = 0.0241,\quad \mathrm{SE} = 0.00539,\quad |t| = 0.99 \;<\; t_{0.975,19} = 2.09,
$$

giving a 95% interval $[-0.0159,\,+0.0053]$ that contains zero. Since $\mathrm{SE}$ falls as $1/\sqrt{n}$, resolving an effect of that size at $|t| = 3$ needs roughly $n \geq (3s/\bar{A})^2 \approx 185$ states. Running exactly that, pooled across four protocol episodes so the sample spans the tube rather than one trajectory through it:

$$
\bar{A} = -0.01766,\quad s = 0.04783,\quad \mathrm{SE} = 0.00352,\quad |t| = 5.02 \;>\; t_{0.975,184} = 1.97,
$$

$$
\text{95\% CI} = [-0.0246,\ -0.0107].
$$

**The one-step advantage is real and negative.** It is not zero. The small sample understated it by a factor of three and, with less than half the spread, was unrepresentative as well as underpowered — the wider pool reveals $s = 0.0478$ against $0.0241$.

**The spread is not noise.** Since each $A_i$ is exact, $s = 0.0478$ is genuine heterogeneity across the tube: at some in-tube states the deviation pushes in the direction the controller was already going and helps, at others it hurts. Even at $n = 185$, **75 of 185 come out positive**. The effect size

$$
d \;=\; \frac{\bar{A}}{s} \;=\; \frac{-0.01766}{0.04783} \;=\; -0.37
$$

is a modest mean displacement inside a much larger state-to-state variation. This is why the sign of a single one-step comparison is close to a coin flip even though the mean is now firmly established.

**Why this does not change the argument.** Nothing below required $\mu = 0$ — only that $|\mu|$ is negligible against the two quantities it must compete with. With the resolved value:

$$
\underbrace{0.018}_{\text{one-step advantage}}
\;<\;
\underbrace{0.047}_{\text{critic's spurious action-range variation}}
\;\lll\;
\underbrace{7.10}_{\text{cost of the persistent bias}}
$$

The one-step advantage remains **below the critic's own fitting error**, and roughly 400 times smaller than the effect that actually decides tube residence. A signal that exists but sits beneath the noise of the function approximator meant to represent it is, for the actor, not usable — which is the claim the rest of this document rests on.

## 6. But the physical consequence of the deviation is not zero

That comparison — *one* deviated step, then back to $\pi$ — is not the thing that determines whether the acrobot stays in the tube. What determines that is a different quantity: what if the policy is biased by $\Delta$ on **every** step, forever? Call the persistently-biased policy $\pi_\Delta$:

$$
\Delta_{\text{persistent}} \;=\; V^{\pi_\Delta}(s) \;-\; V^{\pi}(s)
$$

Measured under the same conditions, with the same $\Delta = 5.8\ \text{N·m}$ error:

$$
\underbrace{-0.018}_{\text{one-step, }Q(s,a_1)-Q(s,a_2)} \qquad \text{vs.} \qquad \underbrace{-7.10}_{\Delta_{\text{persistent}}}
$$

The one-step figure is real but sits below the critic's own fitting error (§7); the persistent figure is measured cleanly, with all 20 test states agreeing in sign, and the tube residence collapses from 100% to 1% of the episode. Both effects are genuine — they are simply different quantities, and only the first is what $Q(s,a)$ measures:

$$
\frac{|\Delta_{\text{one-step}}|}{|\Delta_{\text{persistent}}|} \;\approx\; \frac{1}{400}
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
