# Two ways to measure the CT-SAC action signal wrongly

Both were hit while building `evaluations/action_error_target_audit.py`, both
produce plausible-looking numbers rather than errors, and both change the
measured action signal by more than an order of magnitude. They are recorded
here because any measurement of "how much does the action move `Q`" on these
environments has to rule them out first.

## 1. `env.dt_default` is not the run's control interval

`ContinuousEnv` sets `dt_default = dt`, but `DMCContinuousEnv` overwrites it:

```python
# environment/dmc.py
self.dt_default = self._env.control_timestep()
```

That is dm_control's **native** control timestep for the domain — 10 ms for
acrobot, 10 ms for cartpole and cheetah, 25 ms for humanoid — regardless of the
`dt` the row asked for. On a 1 ms acrobot-XK run:

```
env.dt          = 0.001    # what the run actually steps
env.dt_default  = 0.01     # dm_control's native control timestep
```

`CTSAC` reads it deliberately, as the fallback for `target_reference_dt` when a
row does not set one, and that use is correct. A *measurement* that reaches for
it as "the control interval" is not: passing `dt_default` as the target's `dt`
sends `_finite_difference_target_from_values` down its `dt != T` branch, which
divides the value difference by `dt_ratio = 10`, while the env meanwhile steps
ten times further than intended. The two errors partially cancel, so the result
looks like a small action signal rather than a crash.

Use `env.dt`, and assert the step actually advanced that much — `step_dt`
returns the physical duration it used, and the audit checks it on every step
(`_check_interval`).

## 2. A dm_control env swallows the first step after construction

Setting the physics state and stepping is the obvious way to evaluate many
actions from one state:

```python
with physics.reset_context():
    physics.data.qpos[:] = qpos
    physics.data.qvel[:] = qvel
env.step_dt(action)          # WRONG on a never-reset env
```

`dm_control.rl.control.Environment.step` begins with

```python
if self._reset_next_step:
    return self.reset()
```

so on a freshly built env — or after any `LAST` timestep — the first `step`
discards the action, **re-randomizes the state**, and returns a reset timestep.
The next state then has nothing to do with either the state that was set or the
action that was passed. Sweeping an action grid this way measures the spread of
the initial-state distribution, not the action's effect: on acrobot-XK at 1 ms
it inflated the measured action-range of the target from `0.080` to `0.684`, a
factor of 8.6, with no error raised anywhere.

Reset first, then overwrite:

```python
env.reset()
with physics.reset_context():
    physics.data.qpos[:] = qpos
    physics.data.qvel[:] = qvel
env._step_index = 0          # rewind the uniform time grid too
```

This is what `.claude_scratch/advantage_n185.py` did and why its numbers stood
up to re-measurement (`evaluations/one_step_advantage_vs_dt.py` reproduces them
to four significant figures).

## 3. Irregular time sampling has no single interval to attribute a signal to

Cheetah, cartpole and humanoid rows all set `env_time_sampling=irregular` with
`min_dt=0.002, max_dt=0.03` and a tail-heavy distribution. Every transition
draws its own duration. That is a real feature of those runs, but a 2 ms step
and a 30 ms step from the same state carry different amounts of action signal —
by exactly the factor this whole question is about — so pooling them yields an
"action signal" that belongs to neither.

The audit pins those environments to their nominal `dt` (`fix_interval`), which
for all three is also the target reference interval their critic target uses.
The loaded weights are untouched; only the interval the measurement asks about
is fixed.
