# Stage A Experiment: Oracle PH Value Learning for Acrobot Capture and Balance

## Scope and experimental question

Stage A tests whether a learned return value, combined with exact Acrobot dynamics and an analytical bounded-action maximizer, can produce reliable upright balance and capture. Only the value function is learned. The physical model is fixed and known.

The experiment has two parts:

1. **Local retention:** initialize near upright and learn to remain balanced.
2. **Incoming-state capture:** initialize from states encountered during successful swing-up approaches and learn to capture and retain balance.

The second part requires an explicit dataset of incoming states. A run using only local resets does not establish compatibility with a successful swing-up policy.

The primary question is:

> With dynamics-learning error removed, can fitted HJB value updates learn a value gradient that commands useful elbow torque for capture and sustained balance?

This experiment does not train a PH dynamics model, initialize from an LQR or IDA-PBC controller, switch to an analytical stabilizer, or attempt swing-up from hanging. Those would answer different questions. It is a test of the value-learning and action-selection construction, not yet a test of the benefit of learning PH dynamics.

---

## 1. Hypotheses and experimental arms

### 1.1 Hypotheses

**H1 — Local retention.** A value function trained from local states can generate bounded torque that retains the Acrobot near upright for the required duration and through the end of the episode.

**H2 — Incoming-state capture.** Adding representative incoming swing-up states to training allows the same policy to brake and capture from held-out incoming trajectories without losing its local retention ability.

**H3 — Training-policy choice.** Entropy-regularized value learning with a stochastic analytical training policy may improve capture learning relative to deterministic value learning with collection noise. This is a comparison between two training configurations; it does not isolate entropy independently from exploration or value anchoring.

### 1.2 Two initial configurations

| Quantity | Hard maximization | Soft maximization |
|---|---|---|
| Action maximization in the value update | Exact bounded quadratic maximum | Continuous entropy-regularized integral |
| Temperature $\alpha$ | $0$ | $0.1\kappa\approx0.00388246$ |
| Training action | Bounded deterministic torque plus collection noise | Sample from the truncated-Gaussian action density |
| Extra collection noise | Standard deviation $0.02$ in normalized action units | None |
| Upright value anchor | $V(z^\star)=0$ | No zero-value anchor |
| Deployment and evaluation | Deterministic bounded action | Deterministic mode of the analytical action density |

Both configurations use the same mechanics, physical reward, reset distributions, value-network width, and nominal training budget. Their values represent different objectives because the soft objective includes entropy. Compare physical task metrics rather than the magnitudes of their learned values.

These are untuned starting configurations. Neither is presumed to stabilize the plant before the experiment is run.

---

## 2. Plant, state, and task

### 2.1 Acrobot mechanics

The Acrobot has two links and one motor at the elbow. Let $q_1$ be the shoulder angle measured from hanging down, $q_2$ the relative elbow angle, and $v=\dot q$. Upright is

$$q^\star=(\pi,0),\qquad v^\star=(0,0).$$

The nominal physical parameters are

$$m_1=m_2=1,\quad l_1=1,\quad l_2=2,\quad
l_{c1}=0.5,\quad l_{c2}=1,\quad I_1=0.083,\quad I_2=0.33,\quad g=9.8,$$

in consistent SI units, with inertias about the centers of mass. This geometry follows the Acrobot studied by Xin and Kaneda [1]. Define

$$M(q)=\begin{bmatrix}
a_1+a_2+2a_3\cos q_2 & a_2+a_3\cos q_2\\
a_2+a_3\cos q_2 & a_2
\end{bmatrix},$$

where $(a_1,a_2,a_3)=(1.333,1.330,1.000)$, and

$$U(q)=-14.7\cos q_1-9.8\cos(q_1+q_2).$$

For canonical state $z=(q,p)$, with $p=M(q)v$, the fixed oracle is

$$E(z)=\tfrac12p^\top M(q)^{-1}p+U(q),\qquad
f(z,u)=(J-R)\nabla E(z)+Gu,$$

$$J=\begin{bmatrix}0&I_2\\-I_2&0\end{bmatrix},\qquad
R=\begin{bmatrix}0&0\\0&D\end{bmatrix},\qquad
G=\begin{bmatrix}0\\0\\0\\1\end{bmatrix}.$$

The nominal damping is $D=0$. The learner receives full state observations and the exact mechanics; no context inference is needed for this fixed plant. The simulator is MuJoCo with the same mechanical parameters.

### 2.2 Input, timing, and limits

| Parameter | Initial setting |
|---|---:|
| Elbow torque limit $u_{\max}$ | $20\ \mathrm{N\,m}$ |
| Normalized action $a=u/u_{\max}$ | $[-1,1]$ |
| Action-hold interval $h$ | $0.01\ \mathrm{s}$ |
| Physics integration interval | $0.001\ \mathrm{s}$ |
| Maximum episode duration | $5\ \mathrm{s}$ |
| Physical discount rate $\beta$ | $0.1\ \mathrm{s}^{-1}$ |
| Joint-speed failure threshold | $|v_i|\ge2\pi\ \mathrm{rad/s}$ |
| Unwrapped elbow-angle failure threshold | $|q_2|\ge\pi$ |
| Unwrapped shoulder failure threshold | $q_1\le\pi/2$ or $q_1\ge3\pi/2$ |

Actions are held constant between decisions. Episodes continue after capture. The five-second endpoint is a data-collection and evaluation truncation; it does not assert that the infinite-horizon value becomes zero there. Physical-limit crossings terminate the episode with an absorbing failure cost.

The shoulder interval is centered on upright $q_1=\pi$ and is checked without
angle wrapping. Reaching either endpoint counts as failure, consistently with
the speed and elbow thresholds. Resets and incoming states must start strictly
inside these limits. Earlier pilot and diagnostic runs had no shoulder failure
boundary, a speed threshold of $12$ rad/s, and an elbow threshold of $4\pi$;
runs with the new restrictions are a revised local stabilization task.

### 2.3 Fixed physical reward

Let $\omega_s=4.5844\ \mathrm{rad/s}$. Define the unscaled state cost

$$\ell_0(z)=10(1+\cos q_1)+5(1-\cos q_2)
+\left(\frac{v_1}{\omega_s}\right)^2
+\left(\frac{v_2}{\omega_s}\right)^2,$$

and the uniform reward multiplier

$$C_{\max}=10+10+2\left(\frac{2\pi}{4.5844}\right)^2
+\tfrac12(0.01)(20)^2\approx25.75685752,\qquad
\kappa=C_{\max}^{-1}\approx0.0388246120.$$

The baseline state cost, effort coefficient, and reward rate are

$$\ell=\kappa\ell_0,\qquad w_u=\kappa(0.01),\qquad
\boxed{r(z,u)=-\ell(z)-\tfrac12w_u u^2.}$$

The multiplier is computed once for each new experiment from its declared
weights and limits, then held fixed. Ordinary reward rates lie within $[-1,0]$
inside the state limits. The terminal integration interval can slightly
overshoot a limit before the failure is detected. There is no clipping. The
baseline uses the identity state-cost map; the optional log variant below
changes that map.

The reward is maximal at upright rest with zero torque. It penalizes residual velocity, so passing through the upright pose rapidly does not have the same instantaneous reward as balancing there. It uses the physical state and fixed coefficients throughout training.

No orbit-targeting Lyapunov term or potential-shaping term is included in these initial arms. Such terms can be evaluated later as separate interventions.

For a failure state, define a pessimistic continuing reward rate using an upper bound on the ordinary cost within the declared limits:

$$r_F=-\kappa C_{\max}=-1,\qquad V_F=\frac{r_F}{\beta}=-10.$$

Before scaling, the revised-domain failure rate would be approximately
$-25.7569$ and its value $-257.5686$. Uniform scaling includes the absorbing
continuation and preserves policy rankings for the deterministic discounted
objective on this domain. For the soft objective, temperature and its bounds
are also multiplied by $\kappa$ to preserve the relative entropy weighting.
Initial online and target value outputs are scaled by $\kappa$, so the initial
analytic policy is unchanged. The entropy target and optimizer learning rates
are unchanged. These statements do not imply identical neural optimization
trajectories or resolve the moving-target instability.

The simulated return includes the discounted absorbing continuation after a failure. Value regression at sampled failure states uses the boundary target $V_F$. This is an explicit task boundary condition, not a stability certificate. For both arms, the absorbing state has no further action choice or entropy reward.

#### Finite log state-cost variant

An additional reward comparison applies a finite log transform to the entire
normalized state cost (both angles and both velocities), while preserving the
quadratic effort penalty. Write the baseline state cost as $\ell_b=\kappa\ell_0$
and define

$$C=\kappa\left[10+10+2\left(\frac{2\pi}{4.5844}\right)^2\right]
\approx0.922350776,$$

$$\epsilon=10\kappa\left[1-\cos(5\pi/180)\right]
\approx0.001477394.$$

Thus $\epsilon$ is the baseline cost of a shoulder deviation of $5^\circ$,
with zero elbow angle and zero joint velocities. This reference sets the
transform's cost scale; it does not change the capture tolerance. Both
constants are computed once from the experiment's declared parameters.

The variant uses

$$\boxed{\ell_{\log}(z)=C\frac{\log(1+\ell_b(z)/\epsilon)}{\log(1+C/\epsilon)},
\qquad r_{\log}(z,u)=-\ell_{\log}(z)-\tfrac12w_u u^2.}$$

The map is increasing, sends $0$ to $0$ and $C$ to $C$, and has finite derivative

$$\frac{d\ell_{\log}}{d\ell_b}
=\frac{C}{(\epsilon+\ell_b)\log(1+C/\epsilon)}.$$

At a shoulder-only deviation of $0.1^\circ$, it multiplies the state-cost
gradient by approximately $96.93$. At exact upright rest, the state gradient
is still zero because the baseline state gradient is zero. The ordinary reward
rate remains within $[-1,0]$ inside the declared limits, and the absorbing
failure target remains $-10$. The effort coefficient and temperature settings
are the same as in the corresponding baseline configuration.

In the subsequent value-learning equations, $\ell$ denotes the selected state
cost: $\ell_b$ for the baseline or $\ell_{\log}$ for this variant. The simulated
reward and the HJB target must use the same selection. Because only the state
cost is transformed, the analytical quadratic action maximizer and soft
action density remain applicable.

Compare this variant with the baseline separately under hard maximization,
fixed-temperature soft maximization, and adaptive-temperature soft
maximization. It changes the relative cost of state error and effort and need
not preserve optimal policies. A larger immediate reward gradient does not
establish better learned control; evaluate held-out upright errors, retention,
failures, and torque use across matched training seeds. Compare these physical
metrics rather than returns computed under different reward definitions.

### 2.4 Reset distributions

The default local reset samples independent components:

$$q_1-\pi,\ q_2\sim\operatorname{Uniform}[-0.05,0.05],\qquad
v_1,v_2\sim\operatorname{Uniform}[-0.1,0.1].$$

For incoming-state experiments, preserve the complete $(q_1,q_2,v_1,v_2)$ state of each swing-up approach. Record the angle convention and convert it consistently before constructing momentum. A shoulder angle measured from the horizontal is converted by adding $\pi/2$; velocities are unchanged by that offset.

Split incoming data by source trajectory into training, checkpoint-validation, and final-test sets before selecting states. Nearby samples from one trajectory must not be divided between those sets. Record the source policy, mechanical parameters, torque bound, and rule used to select approach states. Retain difficult approaches rather than selecting only states already balanced.

With an incoming training set, the initial reset mixture is 50% local and 50% incoming. Without it, every reset is local. Report incoming-set coverage and velocities explicitly; angle proximity alone does not characterize capture difficulty.

---

## 3. Value learning and torque selection

### 3.1 What is learned

The trainable object is a scalar return value $V_\psi(z)$. Its network has two hidden layers of width 64 with $\tanh$ activations and a scalar output. The input features are

$$\varphi(z)=(\sin q_1,\sin q_2,\cos q_1,\cos q_2,p_1/10,p_2/10).$$

The momentum scale is fixed, and derivatives include its chain-rule factor. The network output is averaged with its reflected counterpart under

$$S(q_1,q_2,p_1,p_2)=(2\pi-q_1,-q_2,-p_1,-p_2).$$

This uses the reflection symmetry of the nominal dynamics and task. For the hard arm, subtract the value at upright to impose $V_\psi(z^\star)=0$. The soft arm retains its learned value offset because entropy changes the objective. Neither construction imposes a Lyapunov decrease inequality.

There is no separately learned actor or action-value critic, and no learned physical energy, mass, damping, or actuator map.

### 3.2 Analytical action maximization

At a fixed state, the instantaneous HJB score is

$$A_V(z,u)=r(z,u)+\nabla V(z)^\top f(z,u)-\beta V(z).$$

Define the zero-torque drift $f_0(z)=f(z,0)$ and the actuator-space value signal

$$\eta_V(z)=G^\top\nabla V(z)=\frac{\partial V}{\partial p_2}.$$

The score becomes

$$A_V(z,u)=\underbrace{-\ell+\nabla V^\top f_0-\beta V}_{\text{independent of }u}
+\underbrace{\left(-\tfrac12w_u u^2+\eta_Vu\right)}_{\text{depends on }u},\qquad w_u=0.01\kappa.$$

The hard maximizer is

$$u_V(z)=\operatorname{clip}\left(\frac{\eta_V(z)}{w_u},-20,20\right).$$

The maximized action-dependent score is

$$\chi(\eta)=\begin{cases}
\eta^2/(2w_u),&|\eta|\le20w_u,\\
20|\eta|-\tfrac12w_u(20)^2,&|\eta|>20w_u.
\end{cases}$$

Consequently, the hard operator is $\mathcal T[V]=-\ell+\nabla V^\top f_0-\beta V+\chi(\eta_V)$.

For the soft arm, the analytical density over normalized actions is

$$\pi_V(a\mid z)\propto
\exp\left[\frac{20\eta_Va-\tfrac12w_u(20)^2a^2}{\alpha}\right],\qquad a\in[-1,1].$$

Replace $\chi$ by

$$\chi_\alpha(\eta)=\alpha\log\int_{-1}^1
\exp\left[\frac{20\eta a-\tfrac12w_u(20)^2a^2}{\alpha}\right]da.$$

The integral uses 128-point Gauss–Legendre quadrature with log-domain accumulation. Entropy is relative to the continuous normalized-action measure $da$, not a categorical distribution over quadrature nodes. The density is sampled during training; its bounded deterministic mode is used during evaluation. Continuous-time advantage-rate learning motivates this operator construction [2].

### 3.3 Critic target and training budget

Using a frozen target value $\bar V$, form

$$y_V(z)=\operatorname{stopgrad}\left[\bar V(z)+\Delta\tau\,\mathcal T[\bar V](z)\right],\qquad
L_V=\frac1B\sum_{i=1}^B\bigl(V_\psi(z_i)-y_V(z_i)\bigr)^2.$$

Use the soft operator in the soft arm and the absorbing target $V_F$ at failure states. This is a fitted approximation to a Hamiltonian value flow [3]. It is not a theorem that the chosen neural optimizer converges.

| Training parameter | Initial setting |
|---|---:|
| Value updates per run | 10,000 |
| Environment decisions per value update | 1 |
| Batch size | 256 states |
| Fresh reset-distribution states per batch | 128 |
| Replayed states per batch | 128 |
| Replay capacity | 100,000 state entries |
| Value-flow step $\Delta\tau$ | $0.02\ \mathrm{s}$ in iteration time |
| Optimizer | Adam |
| Learning rate | $3\times10^{-4}$ |
| Gradient-norm clipping threshold | 10 |
| Target update | $\bar\psi\leftarrow0.99\bar\psi+0.01\psi$ after each update |
| Validation cadence | Before training and every 1,000 updates |
| Validation episodes per group | 16 |

Replay contains the pre-action and post-action states, with failure labels. It supplies state coverage; ordinary value targets use the known reward and oracle drift rather than a recorded successor-value difference. Fresh states are direct oracle queries drawn from the local or mixed reset distribution. Account for this additional model access in comparisons with model-free learning.

The nominal interaction budget is approximately 100 seconds of simulated training time. Early physical failures can shorten an action interval, so use the measured duration sum. Evaluation and reset-state oracle queries are additional work. The accumulated value-iteration time, $10{,}000\Delta\tau=200$, is not simulated robot time.

### 3.4 Relation to CT-SAC

Both constructions use Bellman/HJB reasoning. For a physical reward rate, the first-order oracle CT-SAC target at a replayed action has the form

$$y_Q(z,u)=\bar V(z)+T A_{\bar V}(z,u).$$

Stage A directly maximizes the score and regresses $V$. CT-SAC regresses an action-value critic and improves a separate actor; an optional value head estimates the actor's soft expectation of target action values. When $T=\Delta\tau$ and the same deterministic value, model, and reward are used, $y_V(z)=\max_u y_Q(z,u)$.

The experiment therefore evaluates a different approximation and optimization arrangement, not a new Bellman principle. Comparison with CT-SAC requires a matched upright task and information budget; results from an orbit-only reward are not a controlled baseline for this balance objective.

### 3.5 Optional adaptive temperature

The two baseline configurations keep temperature fixed. A separate soft variant
can adapt $\alpha>0$ to a declared differential-entropy target. For the analytic
density on normalized actions $a\in[-1,1]$, write

$$g_V(z,a)=u_{\max}\eta_V(z)a-\tfrac12w_u u_{\max}^2a^2,$$

$$H(\pi_{V,\alpha}(\cdot\mid z))
=\log Z_V(z)-\mathbb E_{\pi_{V,\alpha}}[g_V(z,a)/\alpha].$$

After each value update, measure this entropy using the updated online value
gradient and current temperature on the nonterminal minibatch states. With
$\lambda=\log\alpha$, take an Adam step on the SAC-style surrogate

$$L_\alpha(\lambda)=\lambda\,\operatorname{stopgrad}
\left(\overline{H(\pi_{V,\alpha})}-H_{\mathrm{target}}\right).$$

Thus low entropy increases temperature and high entropy decreases it. The
entropy measurement is detached: this optimization step changes only temperature.
An all-terminal minibatch skips the temperature update. Defaults are initial
$\alpha=0.1\kappa$, target entropy $-1$, learning rate $3\times10^{-4}$, and bounds
$10^{-4}\kappa\le\alpha\le10\kappa$. The next HJB target and sampled policy both use the
updated temperature. Deterministic mode evaluation, at a fixed value gradient,
does not depend directly on temperature.

The entropy convention is relative to $da$, not physical torque measure $du$ or
categorical node probabilities. Negative differential entropy is valid; the
maximum on this action interval is $\log 2$. Targets must be strictly below that
maximum. Entropy is evaluated by the same continuous-action quadrature used for
the soft HJB integral; concentrated policies require checking quadrature
resolution. The parameter bounds do not guarantee that the target is attained.

This option introduces another changing component of the learning objective.
Report its temperature and entropy trajectories separately from fixed-temperature
results. It is excluded from the fixed-label versus moving-label diagnostic below,
which isolates feedback from updating the value target network. No convergence
or stabilization guarantee follows from temperature adaptation alone.

---

## 4. Evaluation and progression criteria

### 4.1 Capture predicate and retention

Define

$$\mathcal X_G=\{(q,v):
|\operatorname{wrap}(q_1-\pi)|\le0.1,
|\operatorname{wrap}(q_2)|\le0.1,
|v_1|,|v_2|\le0.25\}.$$

The required uninterrupted hold is one second. Residence time is accumulated over consecutive qualifying physics-step endpoints; the interval before the first qualifying endpoint is not credited. This is a sampled measurement, not a continuous-time invariance proof.

**All default local resets already lie inside $\mathcal X_G$.** Their first-entry time is therefore zero and cannot measure learned capture. For that group, terminal retention, departures, and hold durations are the meaningful outcomes. Incoming-state capture should additionally be reported on the subset that starts outside $\mathcal X_G$, with its sample count stated.

Distinguish:

- **Any-hold success:** an uninterrupted one-second hold occurs anywhere during the episode.
- **Terminal-retention success:** the episode reaches its full five-second horizon and the final uninterrupted hold lasts at least one second.

Use terminal retention as the primary measure. Any-hold success can count a policy that initially stays near upright but later loses balance. Report state-limit failures as failures, including when an earlier hold succeeded.

### 4.2 Metrics and checkpoint selection

For each reset group, report terminal-retention rate, any-hold rate, longest and terminal hold duration, upright occupancy fraction, torque-squared integral, torque saturation fraction, and physical-limit failure rate. For incoming starts outside the capture set, also report time to first entry and time to the first completed hold; unsuccessful episodes are censored rather than assigned zero time.

The basic evaluation records first-entry time and hold durations. Time to the first completed hold requires retaining the corresponding event time in trajectory analysis. Do not infer it from the longest hold alone.

Occupancy and saturation fractions use the actual simulated episode duration, which may be shorter after a failure. Always present them alongside failure rates and episode duration. The reported physical return includes an absorbing continuation on failure; otherwise it is accumulated over the observed five-second window and does not include a learned terminal-value estimate.

Select checkpoints lexicographically by mean terminal-retention rate, then mean terminal hold duration, then mean physical return. If both local and incoming groups are present, give the groups equal weight for selection, while retaining their separate results. The initialized policy is also a candidate; report whether the selected checkpoint actually improved over initialization.

### 4.3 Proposed full study

The following expands the initial single-run setup into an evaluation protocol. It is a proposed study, not a completed result.

1. Run each training configuration with five independent training seeds, 0 through 4, using 10,000 updates per seed.
2. Use validation reset seeds 20000 through 20015 for checkpoint selection. These starts are outside training but are not an untouched final test because selection repeatedly reads them.
3. Freeze each selected checkpoint and evaluate on 128 new reset seeds, 30000 through 30127, per group. For incoming evaluation, use source trajectories never used for training, tuning, or checkpoint selection.
4. Report every training seed, the mean and spread across seeds, and uncertainty in success-rate estimates. Do not count all episodes as independent training replications.
5. Evaluate a zero-torque policy on the same local starts to quantify how much short residence occurs without learned stabilization. A matched CT-SAC comparison is a subsequent benchmark requiring its own runs; it is not supplied by the two value-learning arms alone.

For incoming data, the study needs three trajectory-disjoint partitions even if an individual training run consumes only its training and validation partitions. The final test remains separate until the checkpoint is frozen. If a partition contains fewer than 128 distinct usable starts, report unique-state and unique-trajectory counts; repeated draws do not create additional independent approaches.

### 4.4 Working progression rule

A proposed engineering criterion for moving beyond Stage A is at least 90% terminal-retention success on the untouched local test group in at least four of the five training seeds, with physical-limit failure rates reported explicitly. This is a practical threshold, not a stability theorem or an already achieved result.

To claim compatibility with successful swing-ups, also require at least 90% terminal-retention success on the held-out incoming group in at least four seeds, and report the initially-outside subset separately. Passing the local criterion alone justifies further capture experiments; it does not justify claiming the swing-up-to-balance gap is solved.

If these criteria are missed, inspect learned action gains near upright, value-gradient accuracy, state coverage, failure-boundary effects, and value-update stability before adding dynamics learning. Keep the test set untouched when changing hyperparameters; use validation data for that development.

---

## 5. Fixed-label versus moving-label diagnostic

### 5.1 Question and controlled comparison

This diagnostic asks whether a stationary regression assignment can be fitted
reliably, and whether repeatedly recomputing HJB targets introduces deterioration.
It does not compare an Euler prediction with the value at a physical successor
state. The step $\Delta\tau$ advances the value-learning iteration at a fixed state.

Let $\bar V_0$ be a frozen initial target network and define

$$y_0(z)=\bar V_0(z)+\Delta\tau\,\mathcal H[\bar V_0](z),$$

where

$$\mathcal H[V](z)=-\ell(z)+\nabla V(z)^\top f_0(z)
-\beta V(z)+\chi_\alpha\!\left(\frac{\partial V}{\partial p_2}(z)\right).$$

Here $\chi_0$ is the bounded action maximum and $\chi_\alpha$ for $\alpha>0$
is the entropy-regularized action integral. At absorbing failure states, replace
the label by $V_F$ in both arms.

Create two copies of the same online network, target network, and optimizer state.
They may start from initialization or from a chosen training checkpoint. Apply
the same minibatches and number of optimizer updates to both copies:

- **Fixed labels:** minimize the squared error to $y_0(z)$ throughout the run;
  leave the initial target network frozen.
- **Moving labels:** recompute $y_k(z)$ from the current target network and use
  the usual target-parameter averaging after every optimizer update.

The first regression update is identical in both arms. Later assignments can
differ because only the moving-label arm updates the network generating labels.

### 5.2 Fixed training and held-out states

Before either arm learns, collect separate trajectories with the initial policy
for the training and held-out partitions. For each partition, combine an equal
number of reset-distribution collocation states and visited states sampled without
replacement from its distinct rollout states. Preserve failure-state indicators;
time-limit truncations are not absorbing failures. Do not split adjacent states
from a single trajectory between training and held-out partitions.

The default diagnostic uses 4,096 training states, 2,048 held-out states, and
4,096 collection decisions per partition. During fitting, use the same half-reset,
half-visited minibatch indices for both arms. The held-out states are used only
for measurement. Control-evaluation trajectories are also excluded from fitting.
No new states enter either fixed partition as the policies change.

An optional reset-only comparison omits trajectory collection. It tests local
regression but cannot explain difficulties caused by visited-state coverage or
absorbing boundary labels. Report the actual terminal-state count in each
partition; a partition with no terminals does not test boundary fitting.

### 5.3 Measurements and interpretation

At initialization and every 1,000 updates, save both networks and measure:

- Training and held-out root-mean-square error against each arm's current labels.
- Error against the original labels and the magnitude of label drift.
- Separate interior, absorbing-boundary, reset-state, and visited-state errors.
- Online value, state-gradient, and $\eta_V$ magnitudes at the same states.
- Online HJB residuals on nonterminal states, online/target value disagreement,
  and analytic-action saturation.
- Deterministic retention, hold durations, return, and failures on fixed control
  evaluation starts.

Retain per-state measurements in addition to averages. Boundary states obey an
assigned terminal value, so their HJB residuals should not be pooled with the
interior residual. A nonfinite loss, derivative, or prediction is recorded as a
numerical failure for that arm; the other arm can continue.

If fixed-label training error decreases while held-out error grows, the network
is overfitting the fixed assignment. If both decrease, this demonstrates that
the assignment can be fitted and generalized over the sampled distribution.
Neither result establishes that the assignment is the correct optimal value.

If fixed-label fitting behaves well but moving-label learning deteriorates,
target feedback is implicated under the controlled state distribution. This
is evidence for further investigation, not a proof of divergence or isolation
of a unique cause. Conversely, stability in both arms does not rule out an
instability involving online collection and changing state coverage. Repeat
the comparison across network and dataset seeds before drawing conclusions.

---

## 6. Evidence available and interpretation

### 6.1 Completed preliminary checks

As of 14 September 2026, numerical and integration checks have exercised agreement between the canonical oracle and MuJoCo, physical energy balance, torque saturation, the soft action integral, checkpoint reloads, reward timing, and hold-duration tracking.

Short training smoke runs produced the following final-checkpoint observations:

| Configuration | Updates | Training interaction time | Evaluation episodes | Terminal retention | Mean longest hold | Physical-limit failures |
|---|---:|---:|---:|---:|---:|---:|
| Hard | 300 | $2.999\ \mathrm{s}$ | 4 | $0/4$ | $0.1955\ \mathrm{s}$ | $4/4$ |
| Soft | 100 | $1.000\ \mathrm{s}$ | 2 | $0/2$ | $0.1550\ \mathrm{s}$ | $2/2$ |

Both used local starts and deterministic evaluation, with the earlier state limits and unscaled reward. Their different budgets and small evaluation sets do not support a comparison between training methods or validate the revised limits and scaling above. They show that updates and evaluations execute, while providing no evidence yet of sustained balance. No incoming-state capture result or full five-seed study is established by these checks.

### 6.2 What a successful result would establish

A successful local study would establish empirical retention for the tested initial-state distribution, torque bound, timing, and horizon. A successful incoming study would additionally show that the learned policy can complete the capture stage from the tested swing-up approaches.

Neither result would establish global swing-up, a certified region of attraction, superiority of a learned PH model, or the absence of rare failures. Longer-duration retention and perturbation recovery are useful subsequent tests and must be reported with their own horizons and disturbances.

The useful outcome of Stage A is evidence about the value-learning mechanism itself: whether exact mechanics plus a learned value gradient can solve the local control problem before uncertainty, model fitting, or full swing-up exploration are introduced.

---

## References

[1] X. Xin and M. Kaneda, “[Analysis of the Energy-Based Swing-Up Control of the Acrobot](https://doi.org/10.1002/rnc.1184),” *International Journal of Robust and Nonlinear Control*, vol. 17, no. 16, pp. 1503–1524, 2007. Mechanical setting and energy-based swing-up context; its analytical controller is not used as the Stage A policy.

[2] Y. Jia and X. Y. Zhou, “[q-Learning in Continuous Time](https://arxiv.org/abs/2207.00713),” arXiv:2207.00713, first submitted 2022; revised 2025. Continuous-time advantage rates and entropy-regularized policy improvement.

[3] M. Nguyen, “[Decoupled Continuous-Time Reinforcement Learning via Hamiltonian Flow](https://arxiv.org/abs/2602.14587),” arXiv:2602.14587, 2026. Motivation for generator-based value updates; no convergence guarantee for the particular neural experiment above is assumed.
