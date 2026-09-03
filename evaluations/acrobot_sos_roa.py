"""Region of attraction for the torque-limited Acrobot, by sums of squares.

Implements the structure of the time-invariant Acrobot controller-design
experiment behind Figure 2 of Majumdar, Ahmadi and Tedrake. The LQR controller
initializes a three-step alternation over saturation multipliers, a cubic
polynomial controller, and a quadratic Lyapunov function. The repository does
not contain the paper's hardware-identified model, so this is not a numerical
reproduction of its plotted curves; see ``docs/acrobot_sos_roa.md``.

The vector field is the degree-three Taylor expansion of Section I, so a level
this module returns is certified for the polynomial field and not for the
exact plant.

Requires Drake and an SDP solver.  MOSEK is strongly preferred -- the
open-source solvers disagree by orders of magnitude on this program.  Drake
finds a MOSEK licence through ``MOSEKLM_LICENSE_FILE`` and does not search
``~/mosek``.  Run as::

    MOSEKLM_LICENSE_FILE=/home/seb/mosek/mosek.lic \\
    PYTHONPATH=/path/to/ct-rl /path/to/drake-venv/bin/python \\
        -m evaluations.acrobot_sos_roa
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from pydrake.solvers import MathematicalProgram, MosekSolver, Solve
from pydrake.symbolic import Expression, MonomialBasis, Variables

from controllers.acrobot_gated_lyapunov import (
    LQRDesign,
    lqr_solution,
)
from controllers.xin_kaneda import PAPER_PARAMS as XIN_KANEDA_PARAMS
from evaluations.acrobot_sos_dynamics import (
    error_variables,
    taylor_drift_and_gain,
)


def _default_solver():
    """MOSEK if it is available and licensed, otherwise ``None``."""
    mosek = MosekSolver()
    return mosek if mosek.available() and mosek.enabled() else None


@dataclass(frozen=True)
class SaturatedSystem:
    """The expanded plant, initial LQR law, and actuator bound.

    ``command`` is only the initialization; the controller-design step replaces
    it with a cubic polynomial.
    """

    error: np.ndarray
    drift: np.ndarray
    control_gain: np.ndarray
    command: Expression
    feedback: np.ndarray
    tau_max: float

    @property
    def variables(self) -> Variables:
        return Variables(self.error)

    @property
    def squared_norm(self) -> Expression:
        return self.error @ self.error


def saturated_system(
    tau_max,
    params=XIN_KANEDA_PARAMS,
    design=LQRDesign(),
    drift_order=3,
    gain_order=2,
    error=None,
) -> SaturatedSystem:
    """Build the expanded plant and its initial LQR controller.

    ``f`` and ``g`` are Taylor expanded about upright to the given orders, so
    both are polynomial.  ``error`` supplies the indeterminates; Drake gives
    every ``Variable`` its own identity, so anything meant to be compared has
    to be handed the same four.
    """
    _, _, gain_matrix, _ = lqr_solution(params, design)
    feedback = gain_matrix[0]
    error = error_variables() if error is None else error
    drift, control_gain = taylor_drift_and_gain(
        error, params, drift_order, gain_order
    )
    return SaturatedSystem(
        error=error,
        drift=drift,
        control_gain=control_gain,
        command=-feedback @ error,
        feedback=feedback,
        tau_max=float(tau_max),
    )


def branch_conditions(
    system: SaturatedSystem,
    value: Expression,
    command: Optional[Expression] = None,
):
    """The three saturation branches, as ``(rate, active inequalities)``.

    The rates are the ``Vdot_i`` of Section III in the accompanying note: the
    unsaturated rate under
    ``u(e)``, and the two saturated rates under ``+u_max`` and ``-u_max``.
    Each accompanying inequality ``h_ik`` is written non-negative exactly where
    its branch is active, so the SOS expressions subtract ``M_k h_ik``.
    """
    jacobian = value.Jacobian(system.error)

    def rate(torque):
        return sum(
            jacobian[i] * (system.drift[i] + system.control_gain[i] * torque)
            for i in range(4)
        )

    command = system.command if command is None else command
    tau_max = system.tau_max
    return (
        (rate(command), (tau_max - command, tau_max + command)),
        (rate(tau_max), (command - tau_max,)),
        (rate(-tau_max), (-tau_max - command,)),
    )


def normalizing_scale(value: Expression, error: np.ndarray) -> float:
    """``V(sum_j e_j)``, the divisor condition (5) asks for."""
    ones = {variable: Expression(1.0) for variable in error}
    scale = value.Substitute(ones).Evaluate()
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("V must be positive at the all-ones point to normalize")
    return float(scale)


def normalize(value: Expression, error: np.ndarray) -> Expression:
    """Scale ``V`` to satisfy condition (5), ``V(sum_j e_j) = 1``.

    A scale pin on the pair ``(V, rho)``, as long as ``rho`` is divided by the
    same factor.  The value step requires it: with the multipliers fixed the
    constraints are not invariant under scaling ``V``, so an unpinned candidate
    could be inflated to report any level at all.
    """
    return value / normalizing_scale(value, error)


def lyapunov_conditions(
    system: SaturatedSystem,
    value: Expression,
    level,
    multipliers: Sequence[Expression],
    regions: Sequence[Sequence[Expression]],
    command: Optional[Expression] = None,
):
    """The three SOS expressions, one per branch of the saturation.

    Each is ``-Vdot_i + L_i (V - rho) - sum_k M_k h_ik``.  Inside ``B_rho`` the
    term in ``L_i`` is non-positive, and on the active branch each term in
    ``M_k`` is too, so an expression that is a sum of squares puts ``-Vdot_i``
    above their sum and hence above zero.

    Linear in ``(V, rho)`` for fixed multipliers, and linear in the multipliers
    for fixed ``V`` and ``rho``; the product of the two is what makes the
    program of Section II non-convex.
    """
    conditions = []
    for (branch_rate, active), multiplier, group in zip(
        branch_conditions(system, value, command), multipliers, regions
    ):
        expression = -branch_rate + multiplier * (value - level)
        for condition, region in zip(active, group):
            expression = expression - region * condition
        conditions.append(expression)
    return conditions


def multipliers_at_level(
    system: SaturatedSystem,
    value: Expression,
    level,
    command: Optional[Expression] = None,
    multiplier_degree=2,
    region_degree=2,
    solver=None,
):
    """Multipliers certifying one fixed level, or ``None`` if there are none.

    The inner program of Step 1.  With ``V`` and ``rho`` both fixed, (11)-(13)
    are linear in ``L_i`` and ``M_k``, so this is a feasibility program. A
    solver result other than success is treated as a failed feasibility solve.

    The multipliers are left unrestricted at the origin.  Branch 1 forces
    ``L_1``, ``M_3`` and ``M_4`` to vanish there, since ``h`` enters it
    positive; the saturated branches do not, and their region multipliers must
    be positive there to switch those conditions off near the equilibrium.
    """
    program = MathematicalProgram()
    program.AddIndeterminates(system.error)

    multipliers, regions = [], []
    for _, active in branch_conditions(system, value, command):
        multipliers.append(
            program.NewSosPolynomial(system.variables, multiplier_degree)[0]
            .ToExpression()
        )
        regions.append(tuple(
            program.NewSosPolynomial(system.variables, region_degree)[0]
            .ToExpression()
            for _ in active
        ))
    for expression in lyapunov_conditions(
        system, value, level, multipliers, regions, command
    ):
        program.AddSosConstraint(expression)

    chosen = solver or _default_solver()
    result = Solve(program) if chosen is None else chosen.Solve(program)
    if not result.is_success():
        return None
    return (
        tuple(result.GetSolution(m) for m in multipliers),
        tuple(tuple(result.GetSolution(r) for r in group) for group in regions),
    )


def maximize_level_by_bisection(
    system: SaturatedSystem,
    value: Expression,
    command: Optional[Expression] = None,
    guess=1.0,
    multiplier_degree=2,
    region_degree=2,
    expansions=40,
    refinements=24,
    solver=None,
):
    """Largest level certifiable for a fixed ``V`` and controller.

    ``rho`` multiplies the decision variable ``L_i`` in the SOS conditions, so
    the two cannot be searched together and the level is bracketed instead. Returns
    ``(rho, multipliers, regions)``, or ``None`` if no level in the bracket is
    feasible.

    Bisection is sound because feasibility only loosens as the level falls:
    multipliers certifying ``rho`` also certify any smaller level, since the
    expression then gains ``L_i`` times a positive number.  The feasible levels
    are therefore an interval reaching down to zero, and the bracket is opened
    from ``guess`` in whichever direction it turns out to lie.
    """
    def attempt(candidate):
        return multipliers_at_level(
            system, value, candidate, command, multiplier_degree, region_degree,
            solver,
        )

    lower, upper, found = 0.0, None, None
    solved = attempt(guess)
    if solved is None:
        upper, probe = guess, guess
        for _ in range(expansions):
            probe *= 0.5
            solved = attempt(probe)
            if solved is not None:
                lower, found = probe, solved
                break
        if found is None:
            return None
    else:
        lower, found, probe = guess, solved, guess
        for _ in range(expansions):
            probe *= 2.0
            solved = attempt(probe)
            if solved is None:
                upper = probe
                break
            lower, found = probe, solved
        if upper is None:
            return lower, found[0], found[1]

    for _ in range(refinements):
        middle = 0.5 * (lower + upper)
        solved = attempt(middle)
        if solved is None:
            upper = middle
        else:
            lower, found = middle, solved
    return lower, found[0], found[1]


def _free_polynomial(program, variables, minimum_degree, maximum_degree, name):
    if maximum_degree < minimum_degree:
        raise ValueError(
            f"{name} degree must be at least {minimum_degree}, got {maximum_degree}"
        )
    basis = [
        monomial
        for monomial in MonomialBasis(variables, maximum_degree)
        if monomial.total_degree() >= minimum_degree
    ]
    coefficients = program.NewContinuousVariables(len(basis), name)
    return sum(
        coefficient * monomial.ToExpression()
        for coefficient, monomial in zip(coefficients, basis)
    )


def improve_controller(
    system: SaturatedSystem,
    value: Expression,
    multipliers: Sequence[Expression],
    regions: Sequence[Sequence[Expression]],
    controller_degree=3,
    minimum_level=0.0,
    solver=None,
):
    """Controller step: maximize ``rho`` over a cubic ``u(e)``.

    ``V``, ``L_i``, and ``M_k`` are fixed. The controller has no
    constant term, so the upright equilibrium remains at the origin.
    ``minimum_level`` encodes the previous feasible objective and therefore
    makes the paper's monotonicity invariant explicit to the SDP solver.
    """
    program = MathematicalProgram()
    program.AddIndeterminates(system.error)

    command = _free_polynomial(
        program, system.variables, 1, controller_degree, "k"
    )
    level = program.NewContinuousVariables(1, "rho")[0]
    program.AddBoundingBoxConstraint(minimum_level, np.inf, level)
    for expression in lyapunov_conditions(
        system, value, level, multipliers, regions, command
    ):
        program.AddSosConstraint(expression)

    program.AddLinearCost(-level)
    chosen = solver or _default_solver()
    result = Solve(program) if chosen is None else chosen.Solve(program)
    if not result.is_success():
        return None
    certified = float(result.GetSolution(level))
    if certified <= 0.0:
        return None
    return certified, result.GetSolution(command)


def improve_candidate(
    system: SaturatedSystem,
    command: Expression,
    multipliers: Sequence[Expression],
    value_degree=2,
    region_degree=2,
    positivity=0.0,
    minimum_level=0.0,
    solver=None,
):
    """Value step: maximize ``rho`` over ``V`` and saturation multipliers.

    The controller and the ``L_i`` multipliers are fixed. Following Section
    IV-A of the paper, the ``M_k`` branch multipliers remain decision variables
    in this step. The paper's Figure 2 uses a degree-two ``V``.
    ``minimum_level`` is the objective already certified by the controller
    step, so its lower bound is redundant in exact arithmetic.
    """
    program = MathematicalProgram()
    program.AddIndeterminates(system.error)

    value = _free_polynomial(program, system.variables, 2, value_degree, "v")
    program.AddSosConstraint(value - positivity * system.squared_norm)

    ones = {variable: Expression(1.0) for variable in system.error}
    program.AddLinearEqualityConstraint(value.Substitute(ones) == 1.0)

    regions = []
    for _, active in branch_conditions(system, value, command):
        regions.append(tuple(
            program.NewSosPolynomial(system.variables, region_degree)[0]
            .ToExpression()
            for _ in active
        ))
    level = program.NewContinuousVariables(1, "rho")[0]
    program.AddBoundingBoxConstraint(minimum_level, np.inf, level)
    for expression in lyapunov_conditions(
        system, value, level, multipliers, regions, command
    ):
        program.AddSosConstraint(expression)

    program.AddLinearCost(-level)
    chosen = solver or _default_solver()
    result = Solve(program) if chosen is None else chosen.Solve(program)
    if not result.is_success():
        return None
    certified = float(result.GetSolution(level))
    if certified <= 0.0:
        return None
    solved_regions = tuple(
        tuple(result.GetSolution(region) for region in group)
        for group in regions
    )
    return certified, result.GetSolution(value), solved_regions


@dataclass(frozen=True)
class Iteration:
    """Certified level after each step of one controller-design pass."""

    after_multiplier_step: float
    after_controller_step: Optional[float]
    after_value_step: Optional[float]


@dataclass(frozen=True)
class Alternation:
    """Result of the time-invariant Figure 2 alternation.

    ``{V <= level}`` is certified for ``controller`` on the polynomial
    model. ``failure_step`` identifies a solver failure; it is ``None`` after
    convergence or an ordinary iteration limit.
    """

    level: float
    value: Expression
    controller: Expression
    lqr_level: float
    lqr_value: Expression
    error: np.ndarray
    system: SaturatedSystem
    history: Tuple[Iteration, ...]
    converged: bool
    failure_step: Optional[str] = None


def alternate(
    tau_max,
    params=XIN_KANEDA_PARAMS,
    design=LQRDesign(),
    lyapunov=None,
    value_degree=2,
    controller_degree=3,
    multiplier_degree=2,
    region_degree=0,
    drift_order=3,
    gain_order=2,
    iterations=8,
    tolerance=1e-3,
    positivity=0.0,
    initial_level=1e-7,
    solver=None,
) -> Optional[Alternation]:
    """Time-invariant specialization used for the paper's Figure 2.

    The LQR controller and its Riccati value initialize a three-step alternation:
    (1) fix ``V``, ``u``, and ``rho`` and find all saturation multipliers;
    (2) fix ``V`` and all multipliers and optimize a cubic ``u`` and ``rho``;
    (3) fix ``u`` and ``L_i`` and
    optimize a quadratic ``V``, ``rho``, and the saturation multipliers
    ``M_k``. This is Section II combined with the single-input saturation
    modification in Section IV-A.
    """
    if not np.isfinite(initial_level) or initial_level <= 0.0:
        raise ValueError("initial_level must be finite and positive")

    _, _, _, riccati = lqr_solution(params, design)
    matrix = riccati if lyapunov is None else np.asarray(lyapunov, dtype=np.float64)

    error = error_variables()
    system = saturated_system(
        tau_max, params, design, drift_order, gain_order, error=error
    )

    value = normalize(error @ matrix @ error, error)
    controller = system.command
    lqr_value = value
    lqr_certificate = maximize_level_by_bisection(
        system, value, controller, guess=initial_level,
        multiplier_degree=multiplier_degree, region_degree=region_degree,
        solver=solver,
    )
    if lqr_certificate is None:
        return None
    lqr_level = lqr_certificate[0]

    level = min(float(initial_level), lqr_level)
    previous, converged, history = level, False, []
    carried_multipliers = (lqr_certificate[1], lqr_certificate[2])

    for _ in range(iterations):
        multiplier_step = multipliers_at_level(
            system, value, level, controller, multiplier_degree, region_degree,
            solver
        )
        used_carried_multipliers = False
        if multiplier_step is None:
            multiplier_step = carried_multipliers
            used_carried_multipliers = multiplier_step is not None
        if multiplier_step is None:
            return None if not history else Alternation(
                level=level, value=value, controller=controller,
                lqr_level=lqr_level, lqr_value=lqr_value, error=error,
                system=system, history=tuple(history), converged=False,
                failure_step="multipliers",
            )
        multipliers, regions = multiplier_step
        multiplier_level = level

        controller_step = improve_controller(
            system,
            value,
            multipliers,
            regions,
            controller_degree=controller_degree,
            minimum_level=multiplier_level,
            solver=solver,
        )
        if (
            controller_step is None
            and not used_carried_multipliers
            and carried_multipliers is not None
        ):
            multipliers, regions = carried_multipliers
            controller_step = improve_controller(
                system,
                value,
                multipliers,
                regions,
                controller_degree=controller_degree,
                minimum_level=multiplier_level,
                solver=solver,
            )
        if controller_step is None:
            history.append(Iteration(multiplier_level, None, None))
            return Alternation(
                level=level, value=value, controller=controller,
                lqr_level=lqr_level, lqr_value=lqr_value, error=error,
                system=system, history=tuple(history), converged=False,
                failure_step="controller",
            )
        controller_level, controller = controller_step

        value_step = improve_candidate(
            system,
            controller,
            multipliers,
            value_degree=value_degree,
            region_degree=region_degree,
            positivity=positivity,
            minimum_level=controller_level,
            solver=solver,
        )
        if value_step is None:
            history.append(Iteration(multiplier_level, controller_level, None))
            return Alternation(
                level=controller_level, value=value, controller=controller,
                lqr_level=lqr_level, lqr_value=lqr_value, error=error,
                system=system, history=tuple(history), converged=False,
                failure_step="value",
            )
        level, value, regions = value_step
        carried_multipliers = (multipliers, regions)
        history.append(Iteration(
            multiplier_level, controller_level, level
        ))

        if previous is not None and (level - previous) / previous < tolerance:
            converged = True
            previous = level
            break
        previous = level

    return Alternation(
        level=previous,
        value=value,
        controller=controller,
        lqr_level=lqr_level,
        lqr_value=lqr_value,
        error=error,
        system=system,
        history=tuple(history),
        converged=converged,
    )


if __name__ == "__main__":
    if _default_solver() is None:
        print("MOSEK is unavailable or unlicensed; the open-source SDP solvers "
              "disagree by orders of magnitude on this program, so the numbers "
              "below should not be trusted.\n")

    print("Figure 2 time-invariant controller design, tau_max = 5 N*m\n")
    run = alternate(5.0)
    if run is None:
        print("The initial LQR controller certified no positive level.")
    else:
        verdict = (
            "converged" if run.converged else
            f"stopped at {run.failure_step}" if run.failure_step else
            "iteration limit"
        )
        print(f"{verdict}; {len(run.history)} passes")
        print(f"  fixed-LQR certified level: {run.lqr_level:.6e}")
        print(f"{'pass':>6} {'start rho':>18} {'controller':>18} {'value':>18}")
        for index, step in enumerate(run.history, start=1):
            controller_level = (
                "failed" if step.after_controller_step is None
                else f"{step.after_controller_step:.6e}"
            )
            value_level = (
                "failed" if step.after_value_step is None
                else f"{step.after_value_step:.6e}"
            )
            print(f"{index:>6} {step.after_multiplier_step:>18.6e}"
                  f" {controller_level:>18} {value_level:>18}")
        print(f"  cubic-controller certified level: {run.level:.6e}\n")
