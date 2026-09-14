# Is CT-SAC actually continuous-time?

A note on where CT-SAC's continuous-time content lives, and where it is lost.

## 1. The observation

CT-SAC is named for continuous time, and the implicit promise in that name is that it copes with fine control timescales better than discrete-time SAC does. On acrobot-XK at a 1 ms control interval it does not: the trained critic loses the ability to *order* actions — its rank correlation with the true return is $+0.05 \pm 0.20$ across six seeds, indistinguishable from zero, against $+0.43$ for the same setup at 10 ms (§7) — and every arm depends on an imitation term to supply a usable gradient.

That invites a blunt reading — that CT-SAC is really just SAC generalized to irregular transition intervals, with nothing to say about fine timescales. The blunt reading is close, but the actual situation is narrower and more fixable.

## 2. What CT-SAC's theory says it estimates

From the `CTSAC` class docstring:

> The critic target estimates the instantaneous **advantage-rate**
> $q_V(x,a) = r + (\mathcal{L}^a V)(x) - \beta V(x)$
> where $(\mathcal{L}^a V)$ is the controlled generator.

This is the correct continuous-time object. $q_V$ is a **rate**: it has units of reward per second, it is $O(1)$, and it contains no $dt$. It is the quantity whose maximizer over $a$ is the HJB-optimal action, and it does not degenerate as the control interval shrinks.

So the theory is not the problem. CT-SAC targets the right thing.

## 3. The rate is well-conditioned — measured

The action-dependence of $Q$ and the action-dependence of $q_V$ differ by a factor of the target reference interval $T$:

$$
Q(s,a) - V(s) \;\approx\; T\, q_V(x,a).
$$

This is the claim to test, and it is testable without any network: sweep the first control step over the torque range from a fixed in-tube state, let the analytical controller drive the remaining $2\ \text{s}$, and take the spread of the discounted return. That is $\max_a Q^\pi(s,a) - \min_a Q^\pi(s,a)$ exactly, with no function approximation anywhere. Holding the state set and the horizon fixed and varying only the control interval (`evaluations/one_step_advantage_vs_dt.py --action-grid 9`, $n = 60$ in-tube states):

| $dt$ | action-range of $Q^\pi$ | $\div\, dt$ $\Rightarrow$ range of $q_V$ |
|---|---|---|
| $1\ \text{ms}$ | $0.245$ | $\mathbf{245}$ |
| $2\ \text{ms}$ | $0.570$ | $285$ |
| $5\ \text{ms}$ | $1.503$ | $301$ |
| $10\ \text{ms}$ | $2.685$ | $268$ |
| $20\ \text{ms}$ | $4.124$ | $206$ |
| $50\ \text{ms}$ | $6.185$ | $124$ |

**Across a tenfold change in control interval the action signal stored inside $Q$ grows $11.0\times$ while the advantage-rate stays within $\pm 11\%$ of $275$.** On a log-log fit over $1$–$10\ \text{ms}$ the exponents are $+1.04$ for $\text{range}\,Q$ and $+0.04$ for the rate, against the $+1$ and $0$ this section predicts. The rate is the well-conditioned object and the multiplication by $T$ is what shrinks it; that is confirmed.

Beyond $10\ \text{ms}$ the rate falls away — $206$ then $124$ — because a $20$–$50\ \text{ms}$ open-loop deviation is no longer a small perturbation and the torque limit caps the reachable spread. The invariance is a fine-timescale statement, and $1$–$10\ \text{ms}$ is where it holds.

> **Correction.** Earlier revisions of this section tabulated an action-range of $0.0030$ at $1\ \text{ms}$ and hence a rate of $3.00$. Both are too small by roughly $80\times$. The re-measurement above is $0.245$ and $245$. The trained critics' own targets, evaluated at the same states, report action-ranges of $0.05$ to $2.0$ depending on the arm — the same order of magnitude as $0.245$, and nowhere near $0.003$. Two ways to obtain a spuriously small number here are catalogued in `ct_sac_advantage_measurement_traps.md`; both were hit while building the audit and neither raises an error. The $0.0030$ figure should not be relied on. Note also that a rate of $3.00$ was never consistent with the $-0.018$ measured in §9 for a deviation covering only $15\%$ of the same action range.

## 4. Where it is destroyed

The model-free target (`_finite_difference_target`) is

$$
Q \;=\; r\,T \;+\; V(s) \;+\; T\cdot\frac{e^{-\beta\, dt}V(s') - V(s)}{dt},
$$

which is to say: CT-SAC computes the $O(1)$ rate, then **multiplies it by $T$ and adds it back onto $V$**, storing the sum in one network.

The sharpest test of that sentence holds *one trained critic fixed* and re-forms its target at a range of control intervals, stepping the physics for each. Nothing about the network changes; only the interval the target is built over does. Averaged over three seeds of each arm, log-log exponents over $1$–$10\ \text{ms}$ (`--dt-sweep`):

| arm | slope of $\text{range}_a\,Q$ | slope of the rate | rate variation over $10\times\ dt$ |
|---|---|---|---|
| 1 ms, imitation held | $+1.21$ | $+0.21$ | $1.59\times$ |
| 1 ms, imitation annealed to 0 | $+1.04$ | $+0.04$ | $1.14\times$ |
| 10 ms, no imitation | $+0.95$ | $-0.05$ | $1.15\times$ |
| 10 ms, no imitation + demo | $+1.15$ | $+0.15$ | $1.55\times$ |

Predicted: $+1.00$ and $0.00$. Measured: $+1.09$ and $+0.09$ on average. **The signal the actor reads is proportional to $T$; the rate underneath it is not.**

What this costs in relative terms, on the trained critics at their own intervals (six seeds, in-tube states):

| arm | action signal as a fraction of the critic's own $V$ spread |
|---|---|
| 1 ms, imitation annealed to 0 | $1.0\%$ |
| 10 ms, no imitation | $4.9\%$ |

> **Correction.** This section previously concluded that at $1\ \text{ms}$ the signal is $0.125\%$ of the output range and that the critic's fitting error is **seventeen times** the signal it must carry. Neither survives measurement. The signal is $\sim 1\%$ of the value spread, not $0.125\%$; and the critic's action-shape error is about *half* the action signal at $1\ \text{ms}$ (signal-to-error $1.97$, six-seed mean) and about a quarter of it at $10\ \text{ms}$ ($3.7$). The signal is not buried beneath the approximation error at either interval. The failure at $1\ \text{ms}$ is real but it is not this; §7 measures what it actually is.

## 5. The precise statement

> **CT-SAC's critic target is continuous-time. Its critic parameterization is discrete-time SAC's.**

One network for $Q(s,a)$, exactly as in the 2018 paper. The continuous-time content is computed correctly and then discarded by the representation that stores it.

This is not a quirk of this codebase. It is the known failure mode of value-based methods under fine discretization:

- Baird (1994), *Advantage updating* — introduced specifically because Q-learning degenerates as $\Delta t \to 0$.
- Doya (2000), *Reinforcement learning in continuous time and space* — HJB formulation that avoids forming $Q$ at all.
- Tallec, Blier & Ollivier (ICML 2019), *Making Deep Q-learning Methods Robust to Time Discretization* — the modern statement of exactly this result.

CT-SAC has the continuous-time **target** from that literature without the continuous-time **parameterization** that makes it usable.

## 6. What is genuinely continuous-time and does work

To be fair to the algorithm, these are real and they hold at any $dt$:

| feature | effect |
|---|---|
| $\lambda$ as a rate ($\text{s}^{-1}$), $\gamma_{dt} = e^{-\lambda dt}$ | the discount horizon is physical seconds, invariant to $dt$ |
| reward as a rate, $r\cdot T$ | reward accounting is $dt$-invariant |
| entropy price $\alpha \cdot T$ | matches the entropy term to the reward's interval |
| finite-difference generator for $dt \neq T$ | correct handling of irregular transition intervals |
| model-based generator, $\mathcal{L}^a V = b\cdot\nabla V$ | the action enters analytically through the drift, with no sampled next state |

The entropy price is worth singling out: it was **missing** until recently, so the entropy term was over-weighted by $1/T$ — a factor of a thousand at a 1 ms step. That was a genuine bug, and fixing it is what brought $\alpha$ back into a physically sensible range.

The irregular-interval capability in row four is real, and it is the part the blunt reading correctly identifies. It is just not the only continuous-time content — the generator formulation in rows one and five is too.

## 7. What is not

Anything the **actor** touches. The actor loss reads the learned critic,

$$
\mathcal{L}_{\text{actor}} = \text{price}(\alpha)\log\pi_\theta(a\mid s) - Q_\phi(s,a),
$$

and $Q_\phi$ is where the rate has already been multiplied back down by $T$. So the question that decides everything is narrow and answerable: **does the trained $Q_\phi$ rank actions the way the true return ranks them?**

Answering it needs a reference the critic does not share. Comparing $Q_\phi$ against its own target does not work — they are built from the same $V_\phi$ and agree on whatever it gets wrong, which inflates the correlation from $0.05$ to $0.5$ on the very arm that fails. The reference has to be the rollout: at each in-tube state, take each of nine grid actions for one control step, then let the analytical controller drive $2\ \text{s}$, and rank by discounted return. That ordering depends on neither the seed nor the arm, so it is computed once per interval and every critic is scored against it. Per-state Spearman $\rho$, averaged over $60$ in-tube states, six seeds per arm (`--truth-grid`):

| arm | $T$ | $\rho(Q_\phi,\ Q^\pi)$ | $|t|$ over seeds |
|---|---|---|---|
| 1 ms, imitation held at 0.5 | $1\ \text{ms}$ | $+0.178 \pm 0.084$ | $5.2$ |
| **1 ms, imitation annealed to 0** | $1\ \text{ms}$ | $\mathbf{+0.047 \pm 0.200}$ | $\mathbf{0.57}$ |
| 10 ms, no imitation | $10\ \text{ms}$ | $+0.432 \pm 0.149$ | $7.1$ |
| 10 ms, no imitation + demo | $10\ \text{ms}$ | $+0.546 \pm 0.055$ | $24.3$ |

**The arm that collapses is the arm whose critic cannot order actions at all.** Its six seeds read $+0.049, +0.175, +0.350, -0.074, -0.227, +0.004$ — two of them significantly *anti*-correlated with the truth — and the mean is $0.57$ standard errors from zero. The two 10 ms arms, which never had an imitation term to lose, sit at $\rho \approx 0.43$–$0.55$ with every seed positive.

That is the cleanest statement of the failure available. Both the annealed 1 ms arm and the 10 ms arms finish training with no imitation term; the control interval is the only difference between them, and only the 10 ms critics come out able to tell a good torque from a bad one.

### The interval is the cause, not the weights — measured

Everything above compares a critic trained at 1 ms with a *different* critic trained at 10 ms, so "the interval did it" is inference: the two differ in more than their interval. This probe removes that. One loaded critic is held fixed while the interval is swept, and at each interval both sides are rebuilt for that interval — the critic's target as a run with that reference interval would form it, and fresh rollout truth (deviate one step of that length, then follow the law for 4 s). Seven arms, six seeds each, 60 in-tube states (`--rho-dt-sweep`):

| arm (trained at) | 1 ms | 2 ms | 5 ms | 10 ms | 20 ms | 50 ms |
|---|---|---|---|---|---|---|
| 1 ms, imitation held | $+0.019$ | $+0.265$ | $+0.640$ | $+0.770$ | $+0.840$ | $+0.656$ |
| 1 ms, annealed to 0 | $+0.051$ | $+0.078$ | $+0.201$ | $+0.373$ | $+0.599$ | $+0.760$ |
| 10 ms, no imitation | $+0.073$ | $+0.075$ | $+0.169$ | $+0.364$ | $+0.594$ | $+0.788$ |
| 10 ms, no imitation + demo | $+0.047$ | $+0.093$ | $+0.228$ | $+0.495$ | $+0.801$ | $+0.899$ |
| 1 ms, h2s MSE | $-0.003$ | $+0.029$ | $+0.183$ | $+0.395$ | $+0.654$ | $+0.765$ |
| 10 ms, mbq V-head + demo | $+0.051$ | $+0.050$ | $+0.130$ | $+0.288$ | $+0.593$ | $+0.859$ |
| 10 ms, mbq V-head, anneal 0 | $+0.143$ | $+0.156$ | $+0.175$ | $+0.306$ | $+0.692$ | $+0.889$ |
| **pooled** ($n = 42$ each) | $\mathbf{+0.054}$ | $+0.106$ | $+0.247$ | $+0.427$ | $+0.682$ | $\mathbf{+0.802}$ |

Paired $1 \to 10\ \text{ms}$ on the **same** critic: $+0.373 \pm 0.034$, $|t| = 11.1$, $n = 42$. Rank correlation of $\rho$ with $\log dt$ across all 252 rows: $+0.847$ ($p = 1.7\times10^{-70}$).

**Every critic does this, including the ones trained at 10 ms.** Hand a 10 ms critic a 1 ms interval and its ordering collapses to $+0.073$; hand a 1 ms critic a 50 ms interval and it climbs to $+0.76$. The interval is the driver and the weights are not.

The two `mbqvhead` rows are a built-in control on one measurement choice. Value reads happen before the reference interval is overridden, so $V$ stays the network's own value at its own entropy price rather than being re-defined mid-sweep — but that only matters where $V$ is the sampled soft value, since the entropy price enters it. Those two arms have a V-head, where $V$ is a pure network read with no entropy price at all, and their curves have the same shape as the rest. The choice provably did not drive the result.

### And it is not that the signal is too small

The same sweep records the target's action-range, which grows essentially in proportion to the interval:

| $dt$ | 1 ms | 2 ms | 5 ms | 10 ms | 20 ms | 50 ms |
|---|---|---|---|---|---|---|
| $\text{range}_a\,y$ | $0.163$ | $0.351$ | $1.105$ | $2.518$ | $5.010$ | $17.40$ |

At 1 ms that range is $0.163$ against a true action-range of $0.245$ (§3) — the same order of magnitude — and it still ranks at $\rho = 0.054$. So the target's action-dependence is **not too small to resolve**. It is the right size and points in close to random directions: the network's local variation across the reachable next states has an amplitude comparable to the true variation and no correlation with it.

That is why this note no longer describes the failure as a signal sitting beneath a noise floor. The one-step advantage at 1 ms is small — correctly so, and §9 measures it exactly — but smallness is not what breaks the actor. What breaks it is that the *ordering* at that separation is uninformative, and the sweep above shows the separation is what controls it.

Two caveats worth keeping attached to the table. $Q_\phi$ is the value of the critic's own policy while $Q^\pi$ is the analytical controller's, so this measures whether the critic prefers the actions that are actually good at these states, not whether it has approximated its own $Q$ well. And $\rho$ is a rank statistic: it says the ordering carries little information, not that the magnitudes are small — §3 shows they are not.

## 8. The fix — one candidate ruled out, one demonstrated

### The advantage head does not reach the fault

The parameterization this note is named for is $Q_\phi(s,a) = V_\psi(s) + T\,A_\chi(s,a)$, with $A_\chi$ its own head trained on the rate target. §7 relocates the problem away from it: the *target* is already misordered before the critic sees it.

| | $\rho(Q_\phi,\ Q^\pi)$ | $\rho(\text{target},\ Q^\pi)$ |
|---|---|---|
| 1 ms, imitation annealed to 0 | $+0.047$ | $\mathbf{+0.043}$ |
| 10 ms, no imitation | $+0.432$ | $+0.304$ |

The critic is faithfully reproducing an uninformative regression target. No change to how $Q$ is *stored* recovers an ordering the target does not contain, and the rate target $(y-V)/T$ is the same finite difference divided by $T$ — rescaling signal and error together leaves the ordering untouched. The second claimed benefit, an $O(1)$ actor gradient, is weaker than it appears too: $\partial Q/\partial a$ and $\partial(\alpha T\log\pi)/\partial a$ are both $O(T)$, so the actor gradient is *uniformly* scaled and Adam is per-parameter scale-invariant.

### The model-based generator was the obvious candidate. It was tested and it is worse.

`_model_based_target` replaces the finite difference with the analytic first-order generator $T\,(b(x,a)\cdot\nabla V)$, evaluating $V_\phi$ and $\nabla V_\phi$ once per state and letting the action enter through the exact drift. Acrobot-XK's oracle drift is control-affine to float32 roundoff (measured $\max|b(a)-b(0)-Ga| = 1.4\times10^{-6}$, relative $1.2\times10^{-8}$), so this looked like it should turn "read a learned value at many nearby points" into "project one exact drift column onto one gradient."

Forming **both** targets from the **same** loaded weights, at the same states over the same action grid, and ranking each against rollout truth (`--compare-targets`, six arms × six seeds × 60 states):

| arm | $T$ | $\rho(\text{finite diff})$ | $\rho(\text{generator})$ | agreement | $V(s'\!\mid\!a)$ affine residual |
|---|---|---|---|---|---|
| 1 ms, imitation held | $0.001$ | $+0.101$ | $-0.116$ | $+0.65$ | $0.30$ |
| 1 ms, annealed to 0 | $0.001$ | $+0.043$ | $+0.025$ | $+0.87$ | $0.13$ |
| 10 ms, no imitation | $0.010$ | $+0.304$ | $+0.101$ | $+0.39$ | $0.33$ |
| 10 ms, no imitation + demo | $0.010$ | $+0.553$ | $+0.079$ | $+0.23$ | $0.44$ |
| 10 ms, mbq + V-head + demo | $0.010$ | $+0.349$ | $+0.090$ | $+0.38$ | $0.36$ |
| 10 ms, mbq + V-head, anneal 0 | $0.010$ | $+0.360$ | $+0.145$ | $+0.38$ | $0.37$ |

**The finite difference ranks above the generator in 33 of 36 arm-seed cells**; the paired difference is $+0.231 \pm 0.028$, $|t| = 8.28$. The two right-hand columns say why, and the two intervals fail differently:

- **At 1 ms the generator changes nothing.** The step is small enough that the first-order term and the finite difference substantially agree ($\rho = 0.65$–$0.87$) — and *both* are uninformative about the truth ($+0.043$ and $+0.025$). Two estimators that agree with each other and disagree with reality are not reporting an estimator problem. They are faithfully reporting the local structure of a $V_\phi$ that is simply wrong in the direction the action pushes.
- **At 10 ms the generator actively loses information.** The estimators diverge ($\rho = 0.23$–$0.39$) and the finite difference is far better. Over the larger displacement $V_\phi(s'_a)$ bends — affine residual $0.37$ against $0.21$ at 1 ms — and the first-order term discards exactly that curvature.

Training with the generator does not rescue it either. The two `mbqvhead` arms, whose critic target *was* the model-based one throughout training, rank **below** their finite-difference counterparts at the same interval: $+0.303$ against $+0.546$ for the demo-matched pair (difference $0.243 \pm 0.058$).

> **Correction.** An earlier revision of this section recommended the model-based generator, on the argument that evaluating $\nabla V$ once per state removes per-action approximation error. Both halves are wrong. $V_\phi(s'_a)$ is not affine in $a$ even at 1 ms (residual $0.21$), so there is no clean gradient to substitute for; and the substitution measurably degrades the ordering wherever the two estimators differ. The argument was a prediction, it was tested, and it failed.

### What the fault actually is, and what is left

At 1 ms both available estimators agree with each other and neither agrees with the truth. That localizes the fault past the parameterization (§7) and past the estimator (here), onto the learned value function's **local accuracy in the velocity directions the torque acts on** — the only thing both estimators read. Nothing in this codebase targets that quantity directly, and the one mechanism that plausibly could — a V-head whose gradient is made load-bearing by a model-based target — measurably does not.

### The discount horizon is a lever, and its size is now known

The target's action-dependence splits into two terms with different horizon dependence:

$$
y(s,a) \;=\; \underbrace{T\,r(s,a)}_{\text{no }\lambda} \;+\; \underbrace{\gamma_T V(s'_a)}_{\text{slope} \,\sim\, 1/(\lambda+\mu)} .
$$

Raising $\lambda$ does not strengthen the value term — it *shrinks* it ($V \sim \bar r/\lambda$), handing the ordering to the reward term, which is computed exactly and never suffers the local-accuracy problem above. Since the value term's action-dependence vanishes as $\lambda \to \infty$, $\rho(\text{reward term},\ Q^\pi)$ **is** the ordering the myopic limit would inherit, and it is measurable without retraining:

| | reward/value action-range | $\rho(\text{reward},\ Q^\pi)$ | $\rho(\text{value},\ Q^\pi)$ |
|---|---|---|---|
| 1 ms | $0.0001$–$0.003$ | $\mathbf{+0.275}$ | $+0.04$ |
| 10 ms | $0.087$–$0.140$ | $+0.956$ | $+0.24$–$0.50$ |

Two things follow. **The lever is real**: at 1 ms the myopic ceiling is $+0.275$ against the $+0.043$ the current target achieves, roughly what the 10 ms target manages today. **And it is expensive**: at 1 ms the reward term carries only $0.1$–$0.3\%$ of the target's action-range, so rebalancing needs the value term to shrink by $\sim\!670\times$ — $\lambda \approx 330\ \text{s}^{-1}$, a discount horizon of $\sim3\ \text{ms}$, the same order as the control interval. Merely matching 10 ms's balance still needs a $30\ \text{ms}$ horizon, $70\times$ shorter than the current $2\ \text{s}$. At that point the agent is maximizing instantaneous reward, which is what the Xin-Kaneda law already does analytically.

The one horizon contrast on disk (2 s against 10 s at 1 ms, matched on imitation loss, anneal and $\tau$) is **inconclusive**: $\rho_{\text{critic}} = +0.072$ against $+0.020$, the predicted direction but well inside a six-seed spread of $0.16$. Consistent with the arithmetic — that pair moves $\lambda$ by $5\times$ where $\sim\!300\times$ is needed. Two caveats attach to it: $\eta$ differs across the pair ($0.23$ against $0.26$), and the rollout truth was computed over a fixed $2\ \text{s}$ horizon regardless of $\lambda$, which captures only $18\%$ of the discounted mass at $\lambda = 0.1$. A proper horizon sweep needs a truth horizon that scales with $1/\lambda$.

**A lever nobody has pulled**: $\rho(\text{reward},\ Q^\pi)$ is seed-independent but varies with the reward-shaping parameter — $+0.275$ at $\eta = 0.23$ against $+0.464$ at $\eta = 0.26$, each measured against its own reward's return. How well the immediate reward proxies the true return is a property of the shaping, it is cheaper to change than either the horizon or the control interval, and it has not been tuned for this.

What *is* demonstrated to work is giving the action a longer lever before bootstrapping, so the target stops depending on $V_\phi$ being right over a $10^{-5}$-scale displacement. Raising the control interval to 10 ms is the brute-force form of that, and §9 shows it holds the tube with no imitation term at all. The principled form is an $n$-step target — control at 1 ms, bootstrap from $n$ steps later — which decouples control resolution from the lever length. **It is not implemented here**: the replay buffer and both target paths are strictly one-transition.

So the honest state of the fix is: one candidate ruled out by measurement, one candidate demonstrated but costing control resolution, and one candidate worth building that does not yet exist.

## 9. Empirical corroboration

The 10 ms control sweep is the same claim approached from the other side. Raising the control interval tenfold does not change the theory at all — it simply makes $T\,q_V$ larger inside $Q$. At 10 ms the policy holds the homoclinic tube with **no imitation term at any point in training** — three of six seeds reaching capture $\geq 0.97$.

The one-step cost of a $5.8\ \text{N·m}$ error, re-measured by exact rollout on one fixed set of $185$ in-tube states at every interval (`evaluations/one_step_advantage_vs_dt.py`):

| $dt$ | mean $A$ | $|t|$ | $A/dt$ | states with the **wrong sign** |
|---|---|---|---|---|
| $1\ \text{ms}$ | $-0.0177$ | $5.0$ | $-17.7$ | $75/185$ $(41\%)$ |
| $2\ \text{ms}$ | $-0.0601$ | $7.3$ | $-30.0$ | $59/185$ |
| $5\ \text{ms}$ | $-0.2709$ | $10.3$ | $-54.2$ | $32/185$ |
| $10\ \text{ms}$ | $-0.6901$ | $13.1$ | $-69.0$ | $15/185$ $(8\%)$ |
| $20\ \text{ms}$ | $-1.4442$ | $17.6$ | $-72.2$ | $4/185$ |
| $50\ \text{ms}$ | $-2.9176$ | $27.8$ | $-58.4$ | $0/185$ |

The $1\ \text{ms}$ row reproduces the original $n = 185$ measurement of this quantity to four significant figures — mean $-0.01766$, $s = 0.04783$, $|t| = 5.02$, $75/185$ positive — which is the strongest available check that the re-measurement protocol and the original agree.

**The last column is the result that matters.** At $1\ \text{ms}$ a one-step action comparison points the wrong way at $41\%$ of in-tube states; at $10\ \text{ms}$, at $8\%$; by $50\ \text{ms}$, never. An actor averaging such comparisons over a minibatch is averaging something close to a coin flip at $1\ \text{ms}$ and something nearly unanimous at $50\ \text{ms}$. This is the mechanism §7 detects downstream in the trained critics.

> **Correction.** This section previously gave $-0.204$ ($|t| = 4.8$) for the $10\ \text{ms}$ cost. The re-measurement on a state set held fixed across intervals gives $-0.690$ ($|t| = 13.1$), so the $1 \to 10\ \text{ms}$ growth is $39\times$, not the $\sim 11\times$ implied. Note this fixed-deviation probe is *super*-linear in $dt$ — its rate $A/dt$ moves by $4\times$ over the range — unlike the full action-range sweep of §3, whose rate is flat to $\pm 11\%$. The difference is real: a single deviation sitting next to the controller's own action lands in a locally flat part of $Q(s,\cdot)$ at fine $dt$, while the range over *all* actions does not. §3's sweep is the one that tests the parameterization claim; this one tests what a specific control error costs.

## 10. How these numbers were produced

| probe | script | what it answers |
|---|---|---|
| per-arm critic ledger | `evaluations/action_error_target_audit.py` | action-range of the target and of $Q_\phi$, the critic's action-shape error, $V$ spread, at each arm's own interval |
| interval sweep, one fixed critic | same, `--dt-sweep` | does the signal inside $Q$ scale with $T$ while the rate stays flat, with the network held constant |
| ordering vs ground truth | same, `--truth-grid` | does $Q_\phi$ rank actions the way the true return does |
| ordering vs interval, critic fixed | same, `--rho-dt-sweep` | is the interval the cause, with the weights held constant |
| estimator comparison | same, `--compare-targets` | finite difference against the analytic generator, same weights |
| fixed-error advantage | `evaluations/one_step_advantage_vs_dt.py` | the §5/§9 rollout advantage, at six intervals |
| true action-range | same, `--action-grid` | §3's table, with no network involved |
| consolidation | `evaluations/summarize_advantage_validation.py` | the tables above, with log-log slopes |

Run by `benchmarks/action_error_target_audit.slurm` and `benchmarks/action_error_truthrank.slurm`; outputs under `results/action_error_*` and `results/one_step_advantage_*`.

Three points of method are what make the numbers above trustworthy where the earlier ones were not. Every arm is evaluated on the **same** physical states, snapshotted as $(q, \dot q)$, so a 1 ms and a 10 ms critic are compared at identical points. $V$ is read exactly as `CTSAC._state_value` reads it, using the policy mean so that a one-sample Monte-Carlo wobble cannot be mistaken for the action signal. And every step asserts the physical duration it actually advanced (`_check_interval`) — which is how the two silent failure modes in `ct_sac_advantage_measurement_traps.md` were caught rather than published.

## 11. The other environments, and what they rule out

Acrobot-XK is the only task with two control intervals on disk. Cheetah and cartpole have one each, so they cannot reproduce the 1 ms vs 10 ms contrast — but they can test whether §4's framing is sufficient on its own, because both train to competence with **no imitation term of any kind**. Both use irregular time sampling, so the audit pins them to their nominal interval, which is also their target reference interval (`fix_interval`; see `ct_sac_advantage_measurement_traps.md` §3). Twelve seeds each:

| task | $T$ | action signal / $V$ spread | signal-to-error | $\rho(Q_\phi,\ \text{target})$ |
|---|---|---|---|---|
| cheetah-run, mbq+V-head | $10\ \text{ms}$ | $0.82\%$ | $1.74$ | $+0.52$ |
| cartpole-swingup, mf+V-head | $10\ \text{ms}$ | $0.84\%$ | $3.86$ | $+0.56$ |
| cartpole-swingup, mf | $10\ \text{ms}$ | $0.80\%$ | $4.56$ | $+0.68$ |
| *acrobot-XK, 1 ms, annealed to 0* | $1\ \text{ms}$ | $1.03\%$ | $1.97$ | $+0.19$ |

**This rules out the "small fraction of the output range" explanation.** Cheetah and cartpole carry an action signal worth $0.8\%$ of their value spread — *smaller*, in relative terms, than the $1.03\%$ the failing 1 ms acrobot arm carries — and they learn from it without any imitation term. Cheetah's signal-to-error ratio of $1.74$ is likewise *below* the failing arm's $1.97$. Neither quantity separates the cases.

What does separate them is the ordering: $\rho \approx 0.5$–$0.7$ against $+0.19$. That is the same conclusion §7 reaches on acrobot with rollout ground truth, arrived at from a different direction, and it is why §4's original "the signal is a rounding error on a bigger number" framing had to be replaced. The signal being small relative to $V$ is normal and survivable; the *ranking* going uninformative is not.

(These $\rho$ values compare each critic with its own target, since neither task has an analytical controller to build a rollout reference from. That comparison is optimistic — on acrobot it reads $+0.19$ where the truth-based one reads $+0.05$ — so the acrobot row is quoted on the same optimistic metric to keep the column comparable.)

Humanoid-walk is excluded: its oracle chain's `best_model/` directories are empty and its model-free chain saved no `train_state`, so the tuned temperature its value read depends on is not recoverable from disk.
