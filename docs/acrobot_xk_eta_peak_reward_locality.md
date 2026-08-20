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

## Recommended $\eta$: a ceiling-relative tolerance

Replacing the absolute $10^{-3}$ with a tolerance scaled to $|\max_x r_1|$ restores the property directly — no sample may beat the unshaped ceiling by more than a set fraction of it:

| config | tol $=100\%$ | $50\%$ | $10\%$ | $1\%$ |
|---|---|---|---|---|
| $-\bar V$, XK, $\lambda = 0.1$ | 0.2629 | 0.2624 | 0.2621 | 0.0775 |
| $-\bar V$, XK, $\lambda = 0.5$ | 0.2379 | 0.2375 | 0.1890 | 0.0189 |
| $r_0$, actual, $\lambda = 0.1$ | 0.7094 | 0.6443 | 0.5879 | 0.1073 |
| $r_0$, actual, $\lambda = 0.5$ | 0.6220 | 0.5624 | 0.2063 | 0.0207 |

At $10\%$, floored to the $0.01$ grid:

| config | committed | recommended | peak at | excess | $\min(-r)/\varepsilon_{\rm floor}$ |
|---|---|---|---|---|---|
| $r_3$, $-\bar V$, XK, $\lambda = 0.1$ | 0.24 | **0.24** (up to 0.26) | $\rho = 0.417$, $t = 20.000\ \mathrm{s}$ | $+0.0102$ | 55.8 |
| $r_3$, $-\bar V$, XK, $\lambda = 0.5$ | 0.24 | **0.18** | $\rho = 0.417$, $t = 20.000\ \mathrm{s}$ | $+0.0082$ | 52.2 |
| $r_3$, $r_0$ base, actual, $\lambda = 0.1$ | 0.76 | **0.58** | $\rho = 0.825$, $t = 18.492\ \mathrm{s}$ | $+0.0472$ | 53.5 |
| $r_3$, $r_0$ base, actual, $\lambda = 0.5$ | 0.76 | **0.20** | $\rho = 0.336$, $t = 19.914\ \mathrm{s}$ | $+0.0030$ | 53.5 |

Every peak now falls at $t \approx 18.5$–$20\ \mathrm{s}$ — settled on the orbit, late in the episode — with excess at most $0.05$ nats and a uniform $\approx 50\times$ floor margin, matching what the unshaped $r_1$ does on its own. The $1\%$ column is the knife edge `8bc819a` documented reappearing: it collapses toward the ceiling-defining sample and is not usable.

Isolating the three factors in the `no-saturate` bound:

| base | rate source | $\lambda = 0.1$ | $\lambda = 0.5$ |
|---|---|---|---|
| $-\bar V$ | actual | 0.2604 | 0.2358 |
| $-\bar V$ | XK closed loop | 0.2629 | 0.2379 |
| $r_0$ | actual | 0.7073 | 0.6200 |
| $r_0$ | XK closed loop | 0.8389 | 0.7591 |

The base dominates: $r_0$ admits roughly $2.7$–$3.2\times$ the $\eta$ that $-\bar V$ does. During the pump ($t \in [4, 5]\ \mathrm{s}$) $r_0$ runs at median $-3.61\times10^{-2}$ against $-\bar V$'s $-1.47\times10^{-2}$, so it carries about $2.5\times$ the magnitude for $\eta\dot{\bar V}$ to cancel before $r$ reaches zero — which tracks the ratio of the bounds. The rate source is a minor term, worth $1\%$ on the $-\bar V$ base and $19\%$ on $r_0$; the two $\dot{\bar V}$ definitions have nearly equal magnitude through the pump (median $-3.74\times10^{-2}$ actual, $-3.60\times10^{-2}$ closed loop). Raising $\lambda$ from $0.1$ to $0.5$ tightens every bound by about $10\%$, since $r_3$'s $+\lambda\eta\bar V$ term grows with $\lambda$ and $\bar V \ge 0$.

These bounds are properties of the analytical rollout. A learned policy visiting states off that trajectory can reach a $\dot V$ the rollout never produces, so they are necessary rather than sufficient; the training-time guard against that is the reward's own lower envelope, which is unchanged.

## Reproducing

Scripts are in the job scratch directory, not the repo. Regenerate the rollout with the sweep's own collector, then apply the closed forms above:

```
MUJOCO_GL=disable python -m benchmarks.plot_acrobot_xk_r3_eta_sweep \
    --reward-base lyapunov --lyapunov-rate-source xk_closed_loop
```

The sweep's `--violation-tolerance` reproduces the "sweep bound" column, and passing it $0.1 \cdot |\max_x r_1|$ reproduces the recommended $\eta$ — $5.77\times10^{-6}$ on the $-\bar V$ base, $5.92\times10^{-6}$ on the $r_0$ base, since each base has its own ceiling. The `no-saturate` and `tube-peak` columns are not exposed by it.
