"""Polynomial Acrobot dynamics for sums-of-squares programs.

Sums-of-squares certifies nonnegativity of *polynomials*, by the Gram reduction
``p(e) = z(e)' Q z(e)`` with ``Q`` positive semidefinite and ``z`` the vector of
monomials.  Two features of the Acrobot fall outside that: the gravity and
Coriolis terms are trigonometric in the joint angles, and ``qddot =
M^-1 (tau - H - G)`` is rational in ``cos q2``.

Majumdar, Ahmadi and Tedrake resolve both by Taylor expanding the vector field
(their footnote 3), having found that the expanded dynamics give trajectories
nearly identical to the original ones and preferring that to the overhead of
handling trigonometric terms directly.  This module follows them: ``f`` and
``g`` are expanded about upright, which removes the transcendental terms and
clears the inverse in one step.  The alternative -- lifting to ``s = sin``,
``c = cos`` with ``s^2 + c^2 = 1`` re-imposed by a free multiplier -- keeps the
dynamics exact at the cost of a much larger monomial basis.

The consequence is worth stating plainly: a certificate built on these
polynomials is rigorous for the polynomial vector field, and only approximate
for the true plant.  :func:`taylor_fidelity` measures that gap, and any claim
about the real Acrobot should be checked against the true dynamics separately
before it is treated as a plant-level guarantee.

Requires Drake, which is not a dependency of this repository.  Run under an
environment that has it, with this repository on the path::

    MOSEKLM_LICENSE_FILE=/home/seb/mosek/mosek.lic \\
    PYTHONPATH=/path/to/ct-rl /path/to/drake-venv/bin/python \\
        -m evaluations.acrobot_sos_dynamics
"""

from __future__ import annotations

import numpy as np
from pydrake.symbolic import TaylorExpand, Variable, Variables, cos, sin

from controllers.xin_kaneda import PAPER_PARAMS as XIN_KANEDA_PARAMS

UPRIGHT_STATE = np.array([0.5 * np.pi, 0.0, 0.0, 0.0], dtype=np.float64)


def error_variables(prefix: str = "e") -> np.ndarray:
    """Indeterminates for the upright error ``e = x - x_upright``."""
    return np.array([Variable(f"{prefix}{i}") for i in range(4)])


def exact_drift_and_gain(error, params=XIN_KANEDA_PARAMS):
    """Symbolic ``f(e), g(e)`` with ``edot = f(e) + g(e) tau``.

    Trigonometric and rational, so not usable in an SOS program directly.
    """
    a1, a2, a3 = params.a1, params.a2, params.a3
    b1, b2 = params.b1, params.b2
    q1 = 0.5 * np.pi + error[0]
    q2, d1, d2 = error[1], error[2], error[3]

    cos_q2 = cos(q2)
    m11 = a1 + a2 + 2.0 * a3 * cos_q2
    m12 = a2 + a3 * cos_q2
    determinant = m11 * a2 - m12 * m12

    coriolis_1 = a3 * sin(q2) * (-2.0 * d1 * d2 - d2 * d2)
    coriolis_2 = a3 * sin(q2) * d1 * d1
    gravity_1 = b1 * cos(q1) + b2 * cos(q1 + q2)
    gravity_2 = b2 * cos(q1 + q2)
    forcing_1 = -coriolis_1 - gravity_1
    forcing_2 = -coriolis_2 - gravity_2

    drift = np.array([
        d1,
        d2,
        (a2 * forcing_1 - m12 * forcing_2) / determinant,
        (m11 * forcing_2 - m12 * forcing_1) / determinant,
    ])
    gain = np.array([
        0.0 * d1,
        0.0 * d1,
        -m12 / determinant,
        m11 / determinant,
    ])
    return drift, gain


def taylor_drift_and_gain(
    error, params=XIN_KANEDA_PARAMS, drift_order=3, gain_order=2
):
    """``f, g`` Taylor expanded about upright, so both are polynomial.

    ``drift_order = 3`` is the degree Majumdar et al. use. For this model,
    ``g`` depends on the relative angle through even functions, so its cubic
    expansion has no degree-three term and ``gain_order = 2`` is equivalent.
    """
    drift, gain = exact_drift_and_gain(error, params)
    origin = {v: 0.0 for v in error}
    return (
        np.array([TaylorExpand(x, origin, drift_order) for x in drift]),
        np.array([TaylorExpand(x, origin, gain_order) for x in gain]),
    )


def taylor_fidelity(params=XIN_KANEDA_PARAMS, drift_order=3, gain_order=2,
                    radius=0.35, samples=400, seed=0):
    """Largest deviation of the expanded field from the true plant.

    Returns ``(max |df|, max |dg|)`` over a box of the given radius in the
    error coordinates.
    """
    from controllers.acrobot_gated_lyapunov import plant_drift_and_gain

    error = error_variables()
    drift, gain = taylor_drift_and_gain(error, params, drift_order, gain_order)
    rng = np.random.default_rng(seed)
    worst_drift = worst_gain = 0.0
    for _ in range(samples):
        value = rng.uniform(-radius, radius, 4)
        env = {error[i]: value[i] for i in range(4)}
        expanded_drift = np.array([x.Evaluate(env) for x in drift])
        expanded_gain = np.array([x.Evaluate(env) for x in gain])
        true_drift, true_gain = plant_drift_and_gain(params, UPRIGHT_STATE + value)
        worst_drift = max(worst_drift, np.abs(expanded_drift - true_drift).max())
        worst_gain = max(worst_gain, np.abs(expanded_gain - true_gain).max())
    return worst_drift, worst_gain


if __name__ == "__main__":
    print(f"{'f order':>8} {'g order':>8} {'max |df|':>12} {'max |dg|':>12}"
          "   over |e| <= 0.35")
    # Degrees beyond five expand very slowly and buy little; three is the
    # order the paper uses.
    for orders in ((3, 2), (5, 4)):
        df, dg = taylor_fidelity(drift_order=orders[0], gain_order=orders[1])
        print(f"{orders[0]:>8} {orders[1]:>8} {df:>12.3e} {dg:>12.3e}")
