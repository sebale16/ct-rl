# Is CT-SAC actually continuous-time?

A note on where CT-SAC's continuous-time content lives, and where it is lost.

## 1. The observation

CT-SAC is named for continuous time, and the implicit promise in that name is that it copes with fine control timescales better than discrete-time SAC does. On acrobot-XK at a 1 ms control interval it does not: the actor's ability to distinguish between actions collapses to below the critic's own approximation error, and every arm depends on an imitation term to supply a usable gradient (see `acrobot_xk_vanishing_advantage.md`).

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

Measuring the left side directly on a trained acrobot-XK critic and dividing through:

| quantity | value |
|---|---|
| action-range of $Q$ at an in-tube state (measured) | $0.0030$ |
| $\div\, T = 0.001$ $\Rightarrow$ action-range of $q_V$ | $\mathbf{3.00}$ |
| spread of $V$ across states (measured) | $2.40$ |

**The advantage-rate varies by $3.0$ across the torque range — the same order as the entire state-value spread.** Expressed as a rate, the action signal is not small. It is large, well-scaled, and would be straightforward for a network to represent.

## 4. Where it is destroyed

The model-free target (`_finite_difference_target`) is

$$
Q \;=\; r\,T \;+\; V(s) \;+\; T\cdot\frac{e^{-\beta\, dt}V(s') - V(s)}{dt},
$$

which is to say: CT-SAC computes the $O(1)$ rate, then **multiplies it by $T$ and adds it back onto $V$**, storing the sum in one network. The consequences at $T = 1\ \text{ms}$:

$$
\underbrace{3.00}_{\text{signal as a rate}}
\;\xrightarrow{\ \times\, T\ }\;
\underbrace{0.0030}_{\text{signal inside } Q}
\quad\text{sitting on}\quad
\underbrace{2.40}_{\text{scale of } V}
\;=\; 0.125\%\ \text{of the output range.}
$$

The critic's own measured fitting error across that same action range is $0.050$ — **seventeen times the signal it is meant to carry.**

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

and $Q_\phi$ is where the rate has already been multiplied back down by $T$. No amount of care in the target reaches the actor through a representation that has buried the signal.

## 8. The fix, and why it is small

The algorithm already names the object it needs. Parameterizing the critic as

$$
Q_\phi(s,a) \;=\; V_\psi(s) \;+\; T\cdot A_\chi(s,a)
$$

with $A_\chi$ its own head, trained on the **rate** target — the existing target minus $V$, divided by $T$ — changes two things:

1. $A$'s approximation error scales with $|A| \approx 3$ rather than with $|V| \approx 2.4$, so the action signal is no longer a rounding error on a much larger number.
2. The actor maximizes $A$, which is $O(1)$ in the action and $dt$-invariant by construction.

This is Baird's advantage updating, in the form CT-SAC's own target already implies. The explicit $V$ head (`model_v_net_arch`) supplies half of it and is already implemented; what is missing is the advantage head and repointing the actor at it.

## 9. Empirical corroboration

The 10 ms control sweep is the same claim approached from the other side. Raising the control interval tenfold does not change the theory at all — it simply makes $T\,q_V$ ten times larger inside $Q$, lifting the signal above the critic's noise floor. The measured one-step cost of a $5.8\ \text{N·m}$ error rises from $-0.018$ at 1 ms to $-0.204$ at 10 ms ($|t| = 4.8$), and at 10 ms the policy holds the homoclinic tube with **no imitation term at any point in training** — three of six seeds reaching capture $\geq 0.97$.

That is the parameterization problem being worked around by brute force. Fixing the parameterization would be the way to get the same effect without giving up control resolution.
