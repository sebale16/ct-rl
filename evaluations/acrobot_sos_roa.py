"""Certified region of attraction for the saturated LQR, by sums of squares.

Implements the time-invariant specialisation of Majumdar, Ahmadi and Tedrake,
*Control Design along Trajectories with Sums of Squares Programming*, for a
fixed feedback law.  ``docs/acrobot_sos_roa.md`` states the program and the
alternation; this module implements the multiplier step of that alternation,
which is the step that certifies a level set of a *given* Lyapunov candidate.

The switch used in training, ``AttractiveRegion``, is a heuristic box: entering
it does not imply capture.  A certified sublevel set makes entry imply
convergence, and folds the torque limit in rather than ignoring it.

Saturation is handled by the paper's piecewise analysis rather than a sector
bound: with ``s(u)`` the saturation of ``u(e) = -Ke``, the Lyapunov condition is
imposed separately on each branch,

    u <= u_min          =>  d/dt V under tau = u_min  < 0
    u >= u_max          =>  d/dt V under tau = u_max  < 0
    u_min <= u <= u_max =>  d/dt V under tau = u(e)   < 0

each by an S-procedure with its own multipliers.  With ``V`` fixed, ``rho``
enters the constraints linearly, so the largest certified level is a single
semidefinite program -- no bisection.

What this module does *not* do is search over ``V``.  Step A alone finds the
largest sublevel set of the candidate it is handed, and on this plant the LQR
cost-to-go is badly shaped for the purpose: ``cond(P) ~ 1.6e5``, so its level
sets are long thin needles and the certified region stays small.  Enlarging it
requires the V step of the alternation, which is bilinear and is not
implemented here.

Because the program is built on Taylor-expanded dynamics, its certificate is
rigorous for the polynomial vector field only.  :func:`verify_on_true_plant`
checks a returned level against the true, unexpanded plant by sampling; a
result that fails that check should not be believed.

Requires Drake and an SDP solver.  MOSEK is strongly preferred -- the
open-source solvers disagree by orders of magnitude on this program.  Drake
finds a MOSEK licence through ``MOSEKLM_LICENSE_FILE`` and does not search
``~/mosek``.  Run as::

    MOSEKLM_LICENSE_FILE=$HOME/mosek/mosek.lic \\
    PYTHONPATH=/path/to/ct-rl /path/to/drake-venv/bin/python \\
        -m evaluations.acrobot_sos_roa
"""

from __future__ import annotations

import numpy as np
from pydrake.solvers import MathematicalProgram, MosekSolver, Solve
from pydrake.symbolic import Polynomial, Variables

from controllers.acrobot_gated_lyapunov import (
    LQRDesign,
    lqr_solution,
    plant_drift_and_gain,
)
from controllers.xin_kaneda import PAPER_PARAMS
from evaluations.acrobot_sos_dynamics import (
    UPRIGHT_STATE,
    error_variables,
    taylor_drift_and_gain,
)


def _default_solver():
    mosek = MosekSolver()
    return mosek if mosek.available() and mosek.enabled() else None


def certify(
    tau_max,
    params=PAPER_PARAMS,
    design=LQRDesign(),
    lyapunov=None,
    multiplier_degree=2,
    region_degree=2,
    drift_order=3,
    gain_order=2,
    solver=None,
):
    """Largest ``rho`` with ``{e' S e <= rho}`` certified under saturation.

    ``lyapunov`` is the matrix ``S``; it defaults to the LQR cost-to-go ``P``.
    Returns ``None`` if the program is infeasible.
    """
    _, _, gain_matrix, riccati = lqr_solution(params, design)
    feedback = gain_matrix[0]
    matrix = riccati if lyapunov is None else np.asarray(lyapunov, dtype=np.float64)

    error = error_variables()
    variables = Variables(error)
    drift, control_gain = taylor_drift_and_gain(
        error, params, drift_order, gain_order
    )

    program = MathematicalProgram()
    program.AddIndeterminates(error)

    value = error @ matrix @ error
    jacobian = value.Jacobian(error)
    command = -feedback @ error
    squared_norm = error @ error

    def rate(torque):
        return sum(
            jacobian[i] * (drift[i] + control_gain[i] * torque) for i in range(4)
        )

    level = program.NewContinuousVariables(1, "rho")[0]
    # Each branch of the saturation, with the inequalities that mark where that
    # branch is active written so they are non-positive on it.
    branches = (
        (rate(command), (tau_max - command, tau_max + command)),
        (rate(tau_max), (command - tau_max,)),
        (rate(-tau_max), (-tau_max - command,)),
    )
    for branch_rate, active in branches:
        multiplier = program.NewSosPolynomial(
            variables, multiplier_degree
        )[0].ToExpression()
        expression = squared_norm * (value - level) - multiplier * branch_rate
        for condition in active:
            region = program.NewSosPolynomial(
                variables, region_degree
            )[0].ToExpression()
            expression = expression - region * condition
        program.AddSosConstraint(expression)

    program.AddLinearCost(-level)
    chosen = solver or _default_solver()
    result = Solve(program) if chosen is None else chosen.Solve(program)
    if not result.is_success():
        return None
    certified = float(result.GetSolution(level))
    return certified if certified > 0.0 else None


def verify_on_true_plant(
    level,
    tau_max,
    params=PAPER_PARAMS,
    design=LQRDesign(),
    lyapunov=None,
    samples=100000,
    seed=0,
):
    """Worst ``d/dt (e' S e)`` on ``{e' S e = level}`` for the *true* plant.

    Independent of the semidefinite program: it samples the level set of the
    unexpanded dynamics under the saturated feedback.  A negative value is
    consistent with the certificate; a positive one refutes it.
    """
    _, _, gain_matrix, riccati = lqr_solution(params, design)
    feedback = gain_matrix[0]
    matrix = riccati if lyapunov is None else np.asarray(lyapunov, dtype=np.float64)

    factor = np.linalg.cholesky(matrix)
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(4, samples))
    directions /= np.linalg.norm(directions, axis=0)
    errors = np.sqrt(level) * np.linalg.solve(factor.T, directions)

    worst = -np.inf
    for column in range(samples):
        error = errors[:, column]
        drift, control_gain = plant_drift_and_gain(params, UPRIGHT_STATE + error)
        torque = float(np.clip(-(feedback @ error), -tau_max, tau_max))
        worst = max(worst, float(2.0 * error @ matrix @ (drift + control_gain * torque)))
    return worst


if __name__ == "__main__":
    if _default_solver() is None:
        print("MOSEK is unavailable or unlicensed; the open-source SDP solvers "
              "disagree by orders of magnitude on this program, so the numbers "
              "below should not be trusted.\n")
    print(f"{'tau_max':>9} {'certified rho':>14} {'worst true-plant rate':>23}")
    for limit in (5.0, 10.0, 20.0, 64.0):
        certified = certify(limit)
        if certified is None:
            print(f"{limit:>9.0f} {'infeasible':>14}")
            continue
        worst = verify_on_true_plant(certified, limit)
        verdict = "consistent" if worst < 0 else "REFUTED"
        print(f"{limit:>9.0f} {certified:>14.5f} {worst:>+16.4f}  {verdict}")
