# Lai--She three-stage Acrobot controller

This implementation follows Lai, She, Yang, and Wu, “Stability Analysis and
Control Law Design for Acrobots,” ICRA 2006. The implementation is in
`controllers/lai_she.py`; the comparable video renderer is
`benchmarks/render_acrobot_lai_she.py`.

## Plant and coordinates

Section IV specifies

- `m1 = m2 = 1 kg`, `L1 = 1 m`, `L2 = 2 m`;
- `Lg1 = 0.5 m`, `Lg2 = 1 m`;
- `I1 = 0.083 kg m^2`, `I2 = 0.33 kg m^2`; and
- the simulation equations use `g = 9.8 m/s^2`, giving `E0 = 24.5 J`.

These are exactly the parameters already used by the custom
`acrobot-swingup-xk` MuJoCo plant. The geometry is reused rather than copied,
but the controller does not reuse Xin--Kaneda's coordinates. Lai et al. measure
the shoulder from the upward vertical, whereas that plant measures it from the
horizontal with the opposite joint sense. At the controller boundary,

```text
x1 = pi/2 - q1
x2 = -q2
x3 = -qdot1
x4 = -qdot2
tau_plant = -tau_paper
```

Consequently, the paper's upright `x1=0` is plant `q1=pi/2`, and the paper's
hanging `x1=pi` is plant `q1=-pi/2`. Tests compare the transformed analytical
accelerations with MuJoCo over random states and torques.

## Control stages

The controller is a one-way state machine.

1. `C1`, equation (20), combines posture and energy regulation until the
   denominator approaches its singular surface.
2. `C2`, equations (26)--(30), straightens the second link while a two-input
   fuzzy regulator adjusts `lambda2` to control the energy change.
3. `C3`, equations (33)--(38), balances the acrobot with an LQR after the state
   first enters the attractive set in equation (16).

The published scalar values are used directly: `beta1=beta2=pi/6`, energy
tolerance `1.2 J`, `kp1=kd1=kp2=kd2=1`, `ke1=0.2`, `lambda1=38`, `Phi1=10`,
`zeta=-2`, `Phi2=5`, and `lambda_alpha=0.5`.

The paper prints the first denominator condition as `denominator <= zeta`.
At hanging, however, the denominator is about `-3.767`, so that condition is
already true and immediately selects `C2`; `C2` then leaves hanging invariant.
As energy rises under `C1`, the denominator approaches zero from below. The
stated singularity-avoidance purpose and Fig. 2's delayed first switch therefore
require the crossing `denominator >= zeta`, which is the operational condition
implemented here.

## Fuzzy and LQR details not reported in the papers

The 2006 paper gives the complete 5-by-5 fuzzy rule table but does not give
numerical membership breakpoints. The cited 1999 fuzzy-controller paper
confirms triangular membership functions and center-of-gravity defuzzification,
but likewise does not specify breakpoints for the 2006 controller's two inputs
`e` and `w`. The implementation uses five symmetric triangular sets on
normalized inputs. The energy scale defaults to the physical hanging-to-upright
span, `49 J`; the power scale is an explicit `Design` parameter.

The 2006 paper also omits the LQR `Q` and `R`. They are explicit `Design`
parameters, defaulting to `Q=I`, `R=1`. The resulting gain is checked for local
closed-loop stability, but it is not labeled as a published gain.

The older fuzzy paper's `3 N m` torque limit applies to its different two-stage
controller. The requested three-stage paper derives its laws without an
actuator limit, so the renderer uses a `200 N m` MuJoCo gear to avoid clipping
the demonstrated trajectory; the selected rollout peaks well below that.

## Reproduce the video

From the repository root:

```bash
MUJOCO_GL=egl .venv/bin/python -m benchmarks.render_acrobot_lai_she
```

This writes
`videos/acrobot_lai_she/acrobot_lai_she_lqr_switch.mp4`. The default seeded
rollout switches from `C1` to `C2` at `1.292 s`, enters `C3` at `21.878 s`, and
converges to the upright equilibrium by the end of the video.
