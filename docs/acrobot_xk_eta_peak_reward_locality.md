# Where the shaped Acrobot reward attains its peak

Checks the $\eta$ values carried by the committed `_xkdemo*` training arms against the requirement that peak reward — $r \approx 0$, the top of the `log_reciprocal` transform — is reached only close to the target, never during the swing-up transient.

Companion to [reward shaping for CT-SAC on acrobot swing-up](reward_shaping_for_acrobot_swingup.md), whose notation this uses throughout.

## The question

The arms train on $\ln(1/-r)$ (`reward_transform="log_reciprocal"`, `environment/acrobot_xk.py`), which maps $r \to 0^-$ to $+\infty$ and is clamped at $-r = \varepsilon_{\rm floor} = 10^{-6}$, so the transformed reward saturates at

$$\ln(1/\varepsilon_{\rm floor}) = 13.8155 .$$

Under $r_2$ and $r_3$ the shaping term $-\eta\dot{\bar V}$ is positive whenever $\dot V < 0$, so it can cancel a large negative $r_1$ and drive $r$ to zero at a state that is nowhere near the target. Every such state pays the maximum the transform can pay. The check asks whether any committed $\eta$ does this.

## Method

One rollout of the analytical Xin–Kaneda controller under the fixed protocol — 32 release starts (seeds 20000–20031), $20\ \mathrm{s}$, $1\ \mathrm{ms}$ control and physics step, $k_D = 35.8$, $k_P = 61.2$, $k_V = 66.3$, $\tau_{\max} = 20\ \mathrm{N\cdot m}$ — giving $640{,}032$ samples of $(r_0, \bar V, \dot{\bar V}_{\rm actual}, \dot{\bar V}_{\rm XK}, x)$. This is the same rollout `benchmarks/plot_acrobot_xk_r3_eta_sweep.py` collects, and $\eta$ enters only through the closed forms

$$r_2 = r_1 - \eta\dot{\bar V}, \qquad r_3 = r_1 + \eta\,(\lambda\bar V - \dot{\bar V}),$$

so every $\eta$ is evaluated offline on the one trajectory set.

Each sample is labelled by its eq.-74 switching residual $\rho(x) = |x_1| + |x_2| + 0.1|x_3| + 0.1|x_4|$ ($\zeta = 0.04$) and by membership of the homoclinic tube $\mathcal H$ (`TubeSpec` defaults).

The offline expression was checked against the environment's own reward. For `reward_kind=r3, eta=0.24, discount_rate=0.5, lyapunov_rate_source=xk_closed_loop`, three consecutive steps agree with `xk_reward_terms` to `0.00e+00` absolute difference, and the transform reproduces $-\ln(-r)$ exactly.

## The literal criterion cannot hold, for a reason predating $\eta$

The rollout never enters the switching set: $\min_x \rho(x) = 0.0445 > \zeta$, across all 32 seeds and the full $20\ \mathrm{s}$. This is the same fact commit `8bc819a` built `never_enters_lqr_level` on.

More consequentially, $r_1$ does not peak near the switching set even at $\eta = 0$. Its target is $W_r$, the whole homoclinic orbit: on that orbit $\tilde E = 0$, $q_2 = 0$, $\dot q_2 = 0$, hence $V = 0$ and $r_1 = 0$ at *every* phase of the swing — including the bottom, where $|q_1 - \pi/2|$ alone puts $\rho$ above $3$. Measured on the rollout, $r_1$ attains its maximum at $\rho = 0.4167$ rather than at the $\rho = 0.0445$ closest approach, and 41.6% of samples lie inside $\mathcal H$ with in-tube residual median $1.10$ and maximum $3.66$.

So "peak reward only near the LQR region" is not a property any of these rewards has, and the $\eta$ sweep cannot confer it. The operative question is the one $\eta$ actually controls: whether shaping lets $r$ reach the top of the transform at states the unshaped $r_1$ scores far from zero. Two criteria capture it:

- **no-saturate** — $r \le -\varepsilon_{\rm floor}$ at every sample, so $\ln(1/-r)$ never clamps.
- **tube-peak** — $\max_{x \notin \mathcal H} r < \max_{x \in \mathcal H} r$, so the best-scoring state on the rollout lies in the target tube.

Both are exact linear conditions in $\eta$ (a grid scan for the second), evaluated per reward base, rate source, and discount horizon. They are enough to locate the failure below, and both turn out to be too weak to set $\eta$ by; the tolerance that does is developed after the diagnosis.

## Result: three of the four committed $\eta$ saturate the transform

| config | $\eta$ | sweep bound | no-saturate | tube-peak | verdict |
|---|---|---|---|---|---|
| $r_3$, $-\bar V$ base, XK $\dot V$, $\lambda = 0.1$ (10 s) | 0.24 | 0.2762 | 0.2629 | 0.2615 | **clean** |
| $r_3$, $-\bar V$ base, XK $\dot V$, $\lambda = 0.5$ (2 s) | 0.24 | 0.2498 | 0.2379 | 0.2370 | **582 samples saturated** |
| $r_3$, $r_0$ base, actual $\dot V$, $\lambda = 0.1$ (10 s) | 0.76 | 0.8417 | 0.7073 | 0.8280 | **339 samples saturated** |
| $r_3$, $r_0$ base, actual $\dot V$, $\lambda = 0.5$ (2 s) | 0.76 | 0.7606 | 0.6200 | 0.7485 | **2031 samples saturated** |

"sweep bound" is `never_enters_lqr_level` as `plot_acrobot_xk_r3_eta_sweep.py` computes it; the committed $\eta$ clear it in all four cases, which is why they were selected.

Where the maximum sits at the committed $\eta$:

| config | $\max r$ | episode / time | $\rho$ | in $\mathcal H$ | $r_1$ there | $\eta\,(\lambda\bar V - \dot{\bar V})$ |
|---|---|---|---|---|---|---|
| $-\bar V$, XK, 10 s | $-5.60\times10^{-5}$ | 2 / $20.000\ \mathrm{s}$ | 0.417 | yes | $-5.77\times10^{-5}$ | $+1.79\times10^{-6}$ |
| $-\bar V$, XK, 2 s | $+1.51\times10^{-4}$ | 11 / $4.388\ \mathrm{s}$ | 0.744 | no | $-1.72\times10^{-2}$ | $+1.74\times10^{-2}$ |
| $r_0$, actual, 10 s | $+2.44\times10^{-5}$ | 2 / $7.891\ \mathrm{s}$ | 3.349 | yes | $-3.50\times10^{-4}$ | $+3.74\times10^{-4}$ |
| $r_0$, actual, 2 s | $+8.96\times10^{-4}$ | 13 / $4.480\ \mathrm{s}$ | 1.639 | no | $-5.60\times10^{-2}$ | $+5.69\times10^{-2}$ |

The last column is the mechanism: mid-pump, the controller drives $V$ down hard, and $\eta\,(\lambda\bar V - \dot{\bar V})$ overtakes $|r_1|$ — by $0.87\%$, $6.97\%$ and $1.60\%$ of $|r_1|$ respectively, enough to cross zero. The state there is a fast-swinging chain, $\rho$ between $0.74$ and $3.35$.

In transformed units, against the run's own closest approach to the switching set ($\rho = 0.0445$, episode 2, $t = 19.396\ \mathrm{s}$):

| config | peak $\ln(1/-r)$ | at closest approach | settled tail (last 1 s) | excess |
|---|---|---|---|---|
| $-\bar V$, XK, 10 s | 9.7910 | 9.7814 | 6.6031 | $+0.0096$ |
| $-\bar V$, XK, 2 s | 13.8155 | 9.8850 | 6.7089 | $+3.9305$ |
| $r_0$, actual, 10 s | 13.8155 | 9.8067 | 6.6203 | $+4.0088$ |
| $r_0$, actual, 2 s | 13.8155 | 10.1917 | 7.0071 | $+3.6238$ |

The clean config's peak sits $0.0096$ nats above its closest approach. The other three place a $13.8155$ plateau roughly $3.6$–$4.0$ nats above the closest approach, and about $7$ nats above the settled orbit — the pumping transient outscores both the target set and the trajectory's best approach to upright, by the widest margin anywhere on the rollout.

## Root cause: the tolerance is large relative to the ceiling

`never_enters_lqr_level` tests raw reward against `reward_ceiling` $+$ `VIOLATION_TOLERANCE`, with

$$\texttt{reward\_ceiling} = \max_x r_1 = -5.77\times10^{-5}, \qquad \texttt{VIOLATION\_TOLERANCE} = 10^{-3}.$$

The tolerance is about $17\times$ the ceiling's own magnitude. It was sized in `8bc819a` to absorb a floating-point-scale ($10^{-8}$–$10^{-6}$) knife edge at the single ceiling-defining sample, and at that scale it does its job. But it also admits every $r$ in $(\,\texttt{ceiling},\ 9.4\times10^{-4}\,]$, a window that contains $r = 0$ and everything up to $9.4\times10^{-4}$ — the whole saturating band. All three failures land inside it: $1.51\times10^{-4}$, $2.44\times10^{-5}$, and $8.96\times10^{-4}$.

The `no-saturate` criterion avoids the knife edge without the loose window, because it compares against $-\varepsilon_{\rm floor}$ rather than against a ceiling realized at a particular sample. Its numerator $-\varepsilon_{\rm floor} - r_1 \ge 5.67\times10^{-5} > 0$ everywhere, so no $\eta > 0$ trips it degenerately.

## `no-saturate` alone relocates the peak without fixing it

Taking $\min(\text{no-saturate},\ \text{tube-peak})$ down to the $0.01$ grid gives $0.24$, $0.23$, $0.70$, $0.62$. Every one clears the pump — over $t \in [4, 5]\ \mathrm{s}$ the closest any comes to the floor is $504\times$ it, with no clamped sample. Over the whole rollout, though, the two $r_0$-base values only move the peak:

| config | $\eta$ | clamped | $\min(-r)/\varepsilon_{\rm floor}$ | peak at | excess over closest |
|---|---|---|---|---|---|
| $-\bar V$, XK, 10 s | 0.24 | 0 | 55.9 | $\rho = 0.417$, $t = 20.0\ \mathrm{s}$ | $+0.0096$ |
| $-\bar V$, XK, 2 s | 0.23 | 0 | 50.7 | $\rho = 0.417$, $t = 20.0\ \mathrm{s}$ | $+0.0100$ |
| $r_0$, actual, 10 s | 0.70 | 0 | 4.4 | $\rho = 3.424$, $t = 7.907\ \mathrm{s}$ | $+2.5285$ |
| $r_0$, actual, 2 s | 0.62 | 0 | 1.02 | $\rho = 3.467$, $t = 7.916\ \mathrm{s}$ | $+3.7052$ |

The peak migrates from $t \approx 4.4\ \mathrm{s}$ to $t \approx 7.9\ \mathrm{s}$ and stays at $\rho \approx 3.4$, still $2.5$–$3.7$ nats above the closest approach, with $\eta = 0.62$ sitting within $2\%$ of the floor.

`tube-peak` passes there because $\rho \approx 3.4$ is *inside* $\mathcal H$ — the bottom of the swing, on the orbit. A criterion phrased on tube membership cannot separate the bottom of the swing from the top, which is the separation this check needs.

## Exact $\eta$: closed form for the closed-loop rate, none for the actual rate

The rollout-and-scan above is a numerical stand-in for an algebraic fact. Writing $r_3^{\rm cl}$ out with $V$ substituted in full and $\dot{\bar V}$ replaced by the closed-loop surrogate,

$$r_3^{\rm cl}(x) = -(1-\lambda\eta)\bar V(x) + \eta k_V \dot q_2^2 / V_{\rm down} = -(1-\lambda\eta)\tfrac{\tilde E^2}{2V_{\rm down}} - (1-\lambda\eta)\tfrac{k_P q_2^2}{2V_{\rm down}} + \Big[\eta k_V - \tfrac12(1-\lambda\eta)k_D\Big]\tfrac{\dot q_2^2}{V_{\rm down}},$$

a diagonal quadratic in $(\tilde E, q_2, \dot q_2)$ with no cross terms and, because the closed-loop surrogate drops $u$, no action dependence. $r_3^{\rm cl}(x) \le 0$ for *every* $x$ — not just the rollout — iff both bracketed coefficients are non-negative: $\lambda\eta \le 1$ and $\eta k_V \le \tfrac12(1-\lambda\eta)k_D$. The second is always the binding one, giving the exact, unconditional bound

$$\eta^\ast(\lambda) = \frac{k_D}{\lambda k_D + 2k_V}.$$

At this plant's gains ($k_D = 35.8$, $k_V = 66.3$): $\eta^\ast(0.1) = 0.2629$, $\eta^\ast(0.5) = 0.2379$ — matching the rollout-scanned "no-saturate, XK closed loop" row above to four figures. The grid search was rediscovering this closed form one $0.01$ step at a time; it's now exact and requires no rollout. The committed values are these floored to $0.01$ — $0.26$ and $0.23$ — staying strictly inside the bound rather than landing on it.

The actual rate does not have this property, for a structural reason rather than a numerical one. Because only the elbow is actuated and the plant is conservative, $\dot E = \dot q_2\tau_2$ exactly, and carrying that through $\ddot q_2 = (M^{-1})_{22}\tau_2 - (M^{-1}(H+G))_2$ shows every term of $\dot V_{\rm actual}$ carries an explicit $\dot q_2$ factor:

$$\dot V_{\rm actual}(x, u) = \dot q_2 \cdot G(x, u), \qquad G(x,\tau_2) = \big[\tilde E + k_D(M^{-1})_{22}\big]\tau_2 + k_P q_2 - k_D\big(M^{-1}(H+G_{\rm grav})\big)_2.$$

At $q_2 = 0$, $H \equiv 0$ identically (its only nonzero entries carry a $\sin q_2$ factor), which collapses $G$ to a function of $(\tilde E, q_1, \tau_2)$ alone: $G = [\tilde E + 216.0]\tau_2 + 527.5\cos q_1$ at this plant's gains and $\tau_{\max} = 20\ \mathrm{N\cdot m}$. Fixing $\tilde E = 0$, $q_2 = 0$ and any $\tau_2$ with $G \ne 0$ (true for all but one $\tau_2$ in the admissible interval), the reward along $\dot q_2 \to 0$ is

$$r_3(x, u) = -(1-\lambda\eta)\tfrac{k_D}{2}\dot q_2^2 / V_{\rm down} - \eta \dot q_2 G / V_{\rm down} = \tfrac{|\dot q_2|}{V_{\rm down}}\Big(\eta|G| - (1-\lambda\eta)\tfrac{k_D}{2}|\dot q_2|\Big),$$

which is strictly positive for every $0 < |\dot q_2| < 2\eta|G| / [(1-\lambda\eta)k_D]$ — an interval that exists for *every* $\eta > 0$. No finite $\eta$ makes $r_3$ non-positive everywhere on the actual rate; the quadratic Lyapunov cushion is always beaten by the linear-in-$\dot q_2$ term as $\dot q_2 \to 0$, because real $\dot V$ is only negative-definite under Xin–Kaneda's own control law, not under an arbitrary action.

Weakening the ask to what the rollout-scanned bound was actually approximating — no state outside the target tube $\mathcal H$ can outscore the tube itself, rather than $r_3 \le 0$ everywhere — is solvable, because outside $\mathcal H$ at least one of $\tilde E, q_2, \dot q_2$ has a guaranteed nonzero floor that competes with the linear gain instead of vanishing with it. Bounding $|G|$ by its worst case over $\tau_2 \in [-\tau_{\max}, \tau_{\max}]$, $q_1$ free ($\bar G \approx 4850$ near $\tilde E = 0$), the tightest branch — violate only $\dot q_2$, pinned at the tube edge $\epsilon_\omega\omega_s$ — gives

$$\eta \lesssim \frac{\epsilon_\omega \omega_s V_{\rm down}}{2\bar G}\cdot k_D \approx 2.7\times10^{-3} \text{ (r0 base)}, \qquad \approx 8.5\times10^{-4} \text{ ($-\bar V$ base)},$$

roughly three orders of magnitude below the committed $0.58$/$0.2$. At that scale the shaping term is indistinguishable from $\eta = 0$ ($r_1$) at any practical training horizon, so the actual-rate arms are dropped rather than retagged.

## Applied

`benchmarks/hyperparams/ct_sac.csv` retags the $-\bar V$/XK-closed-loop demonstration family to $\eta^\ast(\lambda)$ floored to $0.01$ instead of the grid-scanned value — `eta0p24` $\to$ `eta0p26` on $\lambda=0.1$ (10 s), `eta0p18` $\to$ `eta0p23` on $\lambda=0.5$ (2 s) — across all four rows each (`_xkdemo`, `_xkdemo20k`, `_xkdemo100k`, and the matched non-demo `_logrecip` control), eight rows in total.

The $r_0$-base/actual-rate family (`eta0p58` on $\lambda=0.1$, `eta0p2` on $\lambda=0.5$) is removed outright — all four rows each, eight rows — per the negligible-$\eta$ result above rather than retagged to $\approx 0.0027$.

Only the `q2dot4pi` cap's `_xkdemo*` rows are touched, since only it has them; the `r2` ladders and the broader `q2dot4pi`/`q2dot4sqrt2pi` exploratory sweeps (`eta=0, 0.01, 0.03, 0.1, 0.3, 1` and the earlier `0.24`/`0.28`/`0.76`/`0.84`/`0.86`/`0.77`/`0.85` ladder points) are untouched — they're exploratory grid points, not the committed recommendation, and are unaffected by this closed form.

## Reproducing

The closed-loop bound is now algebraic, not empirical — $\eta^\ast(\lambda) = k_D/(\lambda k_D + 2k_V)$ reproduces directly from the plant's $k_D$, $k_V$ and the chosen $\lambda$, no rollout needed. The `plot_acrobot_xk_r3_eta_sweep` script (job scratch, not the repo) still reproduces the superseded rollout-scanned numbers for comparison:

```
MUJOCO_GL=disable python -m benchmarks.plot_acrobot_xk_r3_eta_sweep \
    --reward-base lyapunov --lyapunov-rate-source xk_closed_loop
```

The actual-rate bounds ($8.5\times10^{-4}$, $2.7\times10^{-3}$) have no script; they come from the $\dot V = \dot q_2 G(x,u)$ factorization and the tube-boundary branch above, worked symbolically rather than by rollout.
