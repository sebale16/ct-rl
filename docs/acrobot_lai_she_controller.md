# Lai--She 2009 unified WCLF Acrobot controller

The implementation follows the Acrobot specialization in Lai, She, Yang, and
Wu, “Comprehensive Unified Control Strategy for Underactuated Two-Link
Manipulators,” IEEE Transactions on Systems, Man, and Cybernetics—Part B,
39(2), 2009. It replaces the earlier, under-specified 2006 fuzzy-transition
implementation on this branch.

## Paper-coordinate plant

`environment/acrobot_wclf.py` constructs a MuJoCo plant directly in the paper's
coordinates: `x1=0` is upright, `x1=pi` is hanging, and positive elbow torque
has the paper's sign. No coordinate conversion is hidden in this environment.
An explicitly tested adapter remains available for applying the controller to
the repository's horizontal-frame Acrobot:

```text
x1 = pi/2 - q1
x2 = -q2
x3 = -qdot1
x4 = -qdot2
tau_horizontal = -tau_paper
```

The Table-II plant is reproduced exactly:

```text
m1 = m2 = 1 kg              L1 = 1 m, L2 = 2 m
Lg1 = 0.5 m, Lg2 = 1 m      I1 = 8.33e-2 kg m^2, I2 = 0.33 kg m^2
g = 9.8 m/s^2               E0 = 24.5 J
```

Random-state tests compare the paper equations with MuJoCo accelerations to
within `1e-12`.

## WCLF swing-up

The swing-up controller is equation (25):

```text
tau2 = (-alpha2*x2 - beta*f2 - 0.5*beta_dot*x4 - gamma*x4)
       / (alpha1*(E-E0) + beta*b2)
```

with the state-dependent parameters from equations (36), (41), and (42):

```text
beta(x)  = eta / b2(x)
gamma(x) = gamma0 * (E(x) + E0 + epsilon)
```

The published Acrobot values are `alpha1=0.5`, `alpha2=30`, `eta=25`,
`gamma0=1.6`, and `epsilon=0.5`. Since `beta*b2=eta`, the denominator is
bounded below by `eta-2*alpha1*E0=0.5`, eliminating the old singularity.
`beta_dot` is evaluated analytically from `b2(x2)` and `x4`; it is not a finite
difference or an omitted tuning value.

The implementation verifies the paper's identity

```text
V1_dot = -gamma(x) * x4^2
```

against directional finite differences over random states.

## LQR balance and switching

The attractive-set values in equation (73) are used verbatim:

```text
epsilon1 = epsilon2 = pi/6
epsilon3 = epsilon4 = 1e-3
epsilon5 = 1e3
epsilon_E = 1 J
```

On first entry, the controller switches one way to equation (46). The paper
specifies `Q=I4`, `R=0.5`, and explicitly publishes

```text
F = [-260.559, -104.448, -112.604, -52.944]
tau2 = -F*x
```

That printed gain is used directly. Recomputing the CARE from rounded Table-II
values differs by about `0.2`, so it is retained only as a verification value.

## Initial-state caveat

The paper reports `x(0)=[pi,0,0,0]`, which is an exact equilibrium of both the
plant and feedback law. In exact arithmetic it cannot begin swinging. The
renderer therefore exposes `--initial-perturbation` and defaults it to `0.2`
rad. This is the sole calibrated quantity and gives a WCLF-to-LQR switch near
the paper's reported `7.664 s`. It is intentionally displayed as a reproduction
choice rather than misrepresented as a published parameter.

The paper does not impose a torque limit. The MuJoCo actuator gear is only a
normalized command interface and defaults to `50 N m`, above the demonstrated
trajectory's peak; saturation is checked and reported.

## Reproduce the video

```bash
MUJOCO_GL=egl .venv/bin/python -m benchmarks.render_acrobot_lai_she
```

The replacement video is written to
`videos/acrobot_lai_she/acrobot_lai_she_lqr_switch.mp4`.
