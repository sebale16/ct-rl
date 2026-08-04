"""Contact stiffness must belong to a learned parameter, not to the regularizer.

``_constraint_contact_solve`` builds, in latent coordinates y with S = diag(gate)
repeated over each contact's normal and tangent slot,

    W_full = J M^-1 J^T,   W = sym(S W_full S),   H = W + R,   c = S b

and returns the physical impulse Lambda = S y. Every Coulomb cone is invariant
under the positive uniform scaling S applies inside a contact block, so that
change of variables is exact even with the cone active and the problem is
equivalently, in physical coordinates,

    min 0.5 Lambda' (W_full + Rtilde) Lambda + b' Lambda,  Rtilde = S^-1 R S^-1.

The gate therefore cancels out of W_full and out of b and survives ONLY in
Rtilde -- which is literally the contact compliance, since at an interior optimum
v+ - v* = -Rtilde Lambda. With the historical R = reg * scale * I that made
``contact_regularization``, documented as pure conditioning, the entire contact
stiffness: sweeping it moved a held static force by 12x while sweeping the
learned restitution and Baumgarte parameters moved it by exactly zero.

The fix gives the contact its own gate-shaped compliance,

    R = [c0 (1 - s^2) + reg] * scale * I_per_contact,
    c0 = floor + softplus(raw),  floor = fraction * c0_init > 0,

so Rtilde = scale [c0 (1/s^2 - 1) + reg/s^2]. At s = 1 this is exactly the old
R = reg * scale * I, so the fully engaged limit is untouched; as s -> 0, R stays
bounded and >= reg * scale * I (H stays PD) while Rtilde diverges and the impulse
tapers to zero. The floor exists because c0 is learned and receives exactly zero
gradient from the fully engaged contacts (dR/dc0 = (1 - s^2) scale vanishes at
s = 1), so without it a fit can drive c0 to 0 and restore the pathology.

Two things about ``s = 1`` are worth stating plainly, because they bound what
this change can be claimed to do: the reg-sensitivity of a *fully engaged*
contact is unchanged by design, and so the model's contact stiffness at a given
penetration is unchanged. What changes is that reg no longer sets the force where
the gate is tapering, hence no longer sets the resting height.

These tests pin:
  1. AC1, the headline and the permanent guard: with the law enabled the held
     static force is insensitive to ``contact_regularization`` in the tapering
     region, in a pure-torch probe that needs no MuJoCo. The same probe
     reproduces the pre-fix pathology when the law is disabled, so the test
     cannot pass by measuring nothing; and it reproduces it again if c0 is
     driven to zero, which is what the floor prevents.
  2. AC2: sweeping c0 moves the force by orders of magnitude, monotonically.
  3. AC3: a dropped cheetah rests at the floor whatever the gate band is, and
     carries its own weight there (needs dm_control). It does NOT reproduce
     MuJoCo's resting penetration; see that class's docstring.
  4. AC4: the disabled path is untouched -- same state_dict keys, and the
     regularizer AND the ADMM step size it builds are bit-for-bit the legacy
     literal expressions.
  5. AC5: gradient reaches c0 through cholesky_ex/cholesky_solve and the cone
     projections, and is not lost to the detached ``rho``/``scale``.
  6. H is positive definite at gate 0, 1 and in between.
  7. The ADMM step size does not depend on c0, so the shipped iteration count
     still converges once the compliance is enabled.
  8. Sidecar compatibility: solver version 2, round-trip, and a loud failure on
     an enabled/disabled mismatch in either direction.
  9. The entry points can actually construct an enabled model.
"""

import importlib.util
import unittest

import numpy as np
import torch as th

from models.port_hamiltonian import DOFLayout, PortHamiltonianModel

# dm_control/MuJoCo are optional; only AC3 needs them. The absence of the
# packages is the ONLY thing allowed to skip: an earlier `except Exception`
# around this import turned any breakage in the probe helper into three skips
# labelled "MuJoCo not installed", silently deleting the only MuJoCo
# cross-checks of the whole change. Anything else must fail loudly.
_HAVE_MUJOCO = all(
    importlib.util.find_spec(name) is not None
    for name in ("mujoco", "dm_control")
)
if _HAVE_MUJOCO:
    from evaluations.contact_compliance_probe import (
        MuJoCoCheetahMechanics, drop_test, mujoco_reference_rest,
    )

REGS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)


def _model(**kwargs):
    """A float64 kinematic-geometry cheetah. Only the contact port is exercised."""
    th.manual_seed(0)
    kwargs.setdefault("contact_gate_off", 0.06)
    model = PortHamiltonianModel(
        obs_dim=17,
        action_dim=6,
        mode="structured",
        structured_hidden=(32, 32),
        contact_force=6,
        contact_geometry="kinematic",
        contact_solver="constraint",
        dof_layout=DOFLayout.cheetah(),
        **kwargs,
    )
    return model.eval().double()


class SyntheticStance:
    """A static probe state with a hand-made mass matrix, so no MuJoCo is needed.

    ``_constraint_contact_solve`` takes M and the contact-free acceleration as
    arguments, so a fixed SPD M and a gravity-only qdd_free give a completely
    deterministic held-force probe. Every contact gap is affine in the root height
    with unit slope, so one shift puts the lowest contact point at a chosen
    clearance.
    """

    def __init__(self, model, gap: float):
        lo, hi = model.layout.pos_slice
        self.npos = hi - lo
        self.nv = model.layout.nv
        self.pos = th.zeros(1, self.npos, dtype=th.float64)
        self.qd = th.zeros(1, self.nv, dtype=th.float64)
        height_pos = int(model._kin_height_pos.reshape(-1)[0])
        with th.no_grad():
            current = float(model._contact_geometry(self.pos, self.qd)[0].min())
        self.pos[0, height_pos] = gap - current
        # A deterministic SPD mass matrix: unit masses with weak neighbour
        # coupling. Its gauge only rescales ``scale``, which R is proportional to.
        idx = th.arange(self.nv, dtype=th.float64)
        band = th.exp(-(idx[:, None] - idx[None, :]).abs())
        self.M = (th.eye(self.nv, dtype=th.float64) * 2.0 + 0.3 * band)[None]
        self.qdd_free = th.zeros(1, self.nv, dtype=th.float64)
        self.qdd_free[0, int(model._kin_height_cfg.reshape(-1)[0])] = -9.81

    def solve(self, model):
        with th.no_grad():
            return model._constraint_contact_solve(
                self.pos, self.qd, self.M, self.qdd_free
            )

    def force(self, model) -> float:
        out = self.solve(model)
        return float(out["normal_force"][0].sum())

    def gate(self, model) -> th.Tensor:
        with th.no_grad():
            return model._contact_gate(
                model._contact_geometry(self.pos, self.qd)[0]
            )[0]


def _force_vs_reg(gap: float, raw_fill=None, **kwargs) -> list:
    forces = []
    for reg in REGS:
        model = _model(contact_regularization=reg, **kwargs)
        if raw_fill is not None:
            with th.no_grad():
                model._contact_compliance_raw.fill_(raw_fill)
        forces.append(SyntheticStance(model, gap).force(model))
    return forces


def _effective_c0(model) -> th.Tensor:
    """The c0 the solver uses: the floor plus softplus of the raw parameter."""
    return model._contact_compliance_floor + th.nn.functional.softplus(
        model._contact_compliance_raw
    )


class TestRegularizerInsensitivity(unittest.TestCase):
    """AC1. This is the permanent guard against the whole class of bug."""

    GAP = 0.050  # gate ~0.035 in a 0.06 band: deep in the tapering region

    def test_pre_fix_law_makes_the_regularizer_the_contact_stiffness(self):
        """The pathology, so the post-fix assertion cannot pass vacuously."""
        forces = _force_vs_reg(self.GAP)
        self.assertGreater(min(forces), 0.0, forces)
        ratio = max(forces) / min(forces)
        self.assertGreater(
            ratio, 3.0,
            "the legacy R = reg * scale * I is expected to make the held force "
            "swing by more than 3x over five decades of contact_regularization; "
            f"got {ratio:.4g} from {forces}",
        )
        # monotone: less regularization is a stiffer contact
        self.assertEqual(forces, sorted(forces), forces)

    def test_gate_shaped_compliance_makes_the_force_insensitive_to_reg(self):
        forces = _force_vs_reg(self.GAP, contact_compliance=40.0)
        self.assertGreater(min(forces), 0.0, forces)
        ratio = max(forces) / min(forces)
        self.assertLess(
            ratio, 1.05,
            "with the gate-shaped compliance the held force must not depend on "
            "contact_regularization: five decades of reg may move it by at most "
            f"5%, got {ratio:.6g} from {forces}",
        )

    def test_insensitivity_holds_across_the_tapering_region(self):
        for gap in (0.010, 0.020, 0.030, 0.040):
            forces = _force_vs_reg(gap, contact_compliance=40.0)
            ratio = max(forces) / max(min(forces), 1e-300)
            self.assertLess(ratio, 1.05, f"gap {gap}: {forces}")

    def test_insensitivity_survives_a_collapsed_raw_parameter(self):
        """c0 is learned and unbounded above, so it must be bounded below.

        Nothing in the loss defends the taper: dR/dc0 = (1 - s^2) scale is
        exactly zero at s = 1, so the contacts that carry the robot send no
        gradient to c0 at all and a fit is free to drive the raw parameter to
        -inf. Without the floor that restores the pre-fix pathology exactly
        (measured at gate 0.5, reg over five decades: ratio 1.0033 at c0 = 40,
        1.5587 at c0 = 1e-4, 1.5590 with the law disabled).
        """
        forces = _force_vs_reg(self.GAP, raw_fill=-1e3, contact_compliance=40.0)
        model = _model(contact_regularization=1e-2, contact_compliance=40.0)
        with th.no_grad():
            model._contact_compliance_raw.fill_(-1e3)
        self.assertGreater(float(_effective_c0(model).min()), 0.0)
        th.testing.assert_close(
            _effective_c0(model),
            th.full((6,), 0.1 * 40.0, dtype=th.float64), rtol=1e-6, atol=1e-12,
        )
        ratio = max(forces) / min(forces)
        self.assertLess(
            ratio, 1.05,
            "softplus(raw) -> 0 must not hand the contact stiffness back to "
            f"contact_regularization: got a {ratio:.6g}x swing from {forces}",
        )

    def test_the_compliance_floor_is_a_fixed_fraction_of_the_initial_c0(self):
        for value in (0.05, 1.0, 40.0):
            model = _model(contact_compliance=value)
            self.assertAlmostEqual(
                model._contact_compliance_floor,
                PortHamiltonianModel._CONTACT_COMPLIANCE_FLOOR_FRACTION * value,
                places=12,
            )
        self.assertEqual(_model()._contact_compliance_floor, 0.0)

    def test_conditioning_is_not_worse_than_the_legacy_law(self):
        """reg is demoted to a floor, so H must not become harder to factor."""
        for reg in REGS:
            legacy = _model(contact_regularization=reg)
            fixed = _model(contact_regularization=reg, contact_compliance=40.0)
            stance = SyntheticStance(legacy, self.GAP)
            conds = []
            for model in (legacy, fixed):
                with th.no_grad():
                    H = _recover_H(model, stance)
                conds.append(float(np.linalg.cond(H[0].numpy())))
            self.assertLessEqual(conds[1], conds[0] * 1.001,
                                 f"reg={reg}: cond(H) legacy {conds[0]} -> "
                                 f"fixed {conds[1]}")


def _capture_cholesky_argument(model, stance):
    """The exact matrix the solver hands to ``cholesky_ex``, i.e. ``H + rho I``."""
    captured = []
    real = th.linalg.cholesky_ex

    def spy(A, *a, **kw):
        captured.append(A.detach().clone())
        return real(A, *a, **kw)

    th.linalg.cholesky_ex = spy
    try:
        stance.solve(model)
    finally:
        th.linalg.cholesky_ex = real
    return captured[-1]


def _admm_rho(model, stance):
    """The solver's own ADMM step size, rebuilt branch for branch.

    It is the mean of the gated Delassus diagonal plus the conditioning floor,
    which is independent of R -- deliberately, since with the compliance active
    ``diag(H)`` is dominated by the gated-off contacts' ``c0 * scale``.
    """
    W, _, scale, _ = _W_and_scale(model, stance)
    reg = float(model.contact_regularization)
    if model._contact_compliance_raw is None:
        eye = th.eye(W.shape[-1], dtype=W.dtype)[None]
        legacy = W + reg * scale[:, None, None] * eye
        return legacy.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-6)
    return (
        (W.diagonal(dim1=-2, dim2=-1) + reg * scale[:, None])
        .mean(-1)
        .clamp_min(1e-6)
    )


def _recover_H(model, stance):
    """The solver's own H, read off the matrix it hands to ``cholesky_ex``.

    ``cholesky_ex`` is called on ``H + rho I``; ``rho`` is rebuilt from the gated
    Delassus diagonal exactly as the solver builds it, so this recovers H without
    re-deriving R.
    """
    A = _capture_cholesky_argument(model, stance)
    rho = _admm_rho(model, stance)
    eye = th.eye(A.shape[-1], dtype=A.dtype)[None]
    return A - rho[:, None, None] * eye


def _W_and_scale(model, stance):
    """``sym(S W_full S)`` and the detached scale, rebuilt from the geometry."""
    with th.no_grad():
        g, _, _, J_n, J_t = model._contact_geometry(stance.pos, stance.qd)
        gate = model._contact_gate(g)
        B, K = gate.shape
        J = th.stack((J_n, J_t), dim=2).reshape(B, 2 * K, -1)
        M_inv_Jt = th.linalg.solve(stance.M, J.transpose(1, 2))
        W_full = th.bmm(J, M_inv_Jt)
        scale = W_full.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-6)
        S = gate.unsqueeze(-1).expand(-1, -1, 2).reshape(B, 2 * K)
        W = th.bmm(J * S.unsqueeze(-1), M_inv_Jt * S.unsqueeze(1))
        W = 0.5 * (W + W.transpose(1, 2))
    return W, W_full, scale, S


class TestComplianceSweep(unittest.TestCase):
    """AC2. The learned parameter must actually be the knob."""

    GAP = 0.030

    def test_force_falls_by_orders_of_magnitude_as_c0_rises(self):
        """The sweep is over the *physical* c0, set through the constructor.

        c0 = floor + softplus(raw) with floor = fraction * c0_init, so raw alone
        cannot reach below the floor; requesting an initial c0 sets both and the
        effective value is exactly the requested one.
        """
        values = (1e-4, 1e-2, 0.1, 1.0, 4.0, 10.0, 40.0)
        forces = []
        for value in values:
            model = _model(contact_regularization=1e-2, contact_compliance=value)
            th.testing.assert_close(
                _effective_c0(model),
                th.full((6,), value, dtype=th.float64), rtol=1e-6, atol=1e-12,
            )
            forces.append(SyntheticStance(model, self.GAP).force(model))
        self.assertGreater(
            forces[0] / forces[-1], 10.0,
            f"c0 must move the force by at least an order of magnitude: {forces}",
        )
        # strictly decreasing: more compliance is less force
        for lo, hi in zip(forces, forces[1:]):
            self.assertLess(hi, lo, forces)

    def test_tiny_c0_recovers_the_legacy_force(self):
        """c0 -> 0 is the old law; the fix adds a knob, it does not move the base.

        ``_inverse_softplus`` clamps its argument at 1e-8, so the smallest
        reachable c0 is ~1e-8 and the residual disagreement is the physical
        effect of that c0: at gate 0.0355 it adds c0 (1/s^2 - 1) = 8e-6 to an
        Rtilde of 7.9, i.e. ~1e-6 relative, which is what is measured.
        """
        legacy = _model(contact_regularization=1e-2)
        fixed = _model(contact_regularization=1e-2, contact_compliance=1e-12)
        self.assertLess(float(_effective_c0(fixed).max()), 1e-7)
        stance = SyntheticStance(legacy, self.GAP)
        reference = stance.force(legacy)
        self.assertAlmostEqual(stance.force(fixed), reference,
                               delta=1e-5 * reference)

    def test_full_gate_is_exactly_the_legacy_regularizer(self):
        """Every slot at s = 1 must carry R = reg * scale, whatever c0 is."""
        reg = 3e-3
        model = _model(contact_regularization=reg, contact_compliance=40.0)
        stance = SyntheticStance(model, -0.010)  # deepest point well through
        gate = stance.gate(model)
        self.assertTrue(bool((gate == 1.0).any()), gate)
        H = _recover_H(model, stance)
        W, _, scale, S = _W_and_scale(model, stance)
        R = (H - W)[0]
        expect = reg * float(scale[0])
        engaged = S[0] == 1.0
        self.assertLess(
            float((R.diagonal()[engaged] - expect).abs().max()), 1e-9 * expect,
            f"R at s = 1 is {R.diagonal()[engaged]}, expected {expect}",
        )
        self.assertLess(
            float((R - th.diag(R.diagonal())).abs().max()), 1e-9 * expect)


class TestADMMStepSizeIsNotPoisonedByTheCompliance(unittest.TestCase):
    """The taper must not cost convergence at the shipped iteration count.

    ``rho`` is a single scalar per sample. Taking it from ``mean(diag H)`` folds
    in the gated-off contacts' ``c0 * scale``, which is deliberately enormous
    (measured 12.69 versus 0.093, a factor of 136 at the cheetah's resting pose),
    so the ADMM step becomes orders of magnitude too small for the coordinates
    that carry load and the fixed 12 iterations stop 68% short of the answer.
    The step size is therefore taken from the gated Delassus diagonal plus the
    conditioning floor, which is what ``diag H`` already was before the
    compliance existed -- so it is bit-exact when disabled and c0-independent
    when enabled.
    """

    DEFAULT_ITERATIONS = 12
    # (band, gap) grid. The kinematic default band puts every contact at s = 0 or
    # s = 1; the 0.06 band is the harder case, with two contacts engaged at
    # different partial gates at once.
    GRID = tuple(
        (band, gap)
        for band in (0.005, 0.06)
        for gap in (-0.010, -0.002, 0.0, 0.001, 0.015, 0.030)
    )

    def _force(self, gap, iterations, band=0.005, **kwargs):
        model = _model(contact_regularization=1e-2, contact_gate_off=band,
                       contact_iterations=iterations, **kwargs)
        return SyntheticStance(model, gap).force(model)

    def test_default_iteration_count_converges_at_a_load_bearing_state(self):
        """The band the kinematic geometry actually defaults to."""
        for gap in (-0.010, -0.002, 0.0):
            for kwargs in ({}, {"contact_compliance": 40.0}):
                truncated = self._force(gap, self.DEFAULT_ITERATIONS, **kwargs)
                converged = self._force(gap, 4000, **kwargs)
                self.assertGreater(converged, 1.0, (gap, kwargs))
                self.assertLess(
                    abs(truncated - converged) / converged, 1e-6,
                    f"gap={gap} {kwargs}: {self.DEFAULT_ITERATIONS} ADMM "
                    f"iterations give {truncated} N against a converged "
                    f"{converged} N",
                )

    def test_truncation_error_is_at_parity_with_the_legacy_law(self):
        """No claim that 12 iterations are exact -- only that the compliance
        does not make them worse than they always were.

        Measured worst case over this grid: 9.2e-3 relative with the compliance
        (band 0.06, 10 mm of penetration, two contacts at different partial
        gates) against 1.1e-2 relative for the legacy law (band 0.06, gap
        +0.015). With ``rho`` taken from ``mean(diag H)`` the compliance's worst
        case over the same grid is 6.0e-1.
        """
        for band, gap in self.GRID:
            converged = self._force(gap, 4000, band=band,
                                    contact_compliance=40.0)
            if converged <= 0.0:  # fully gated off: nothing to converge
                continue
            truncated = self._force(gap, self.DEFAULT_ITERATIONS, band=band,
                                    contact_compliance=40.0)
            self.assertLess(
                abs(truncated - converged) / converged, 2e-2,
                f"band={band} gap={gap}: {truncated} N against a converged "
                f"{converged} N",
            )

    def test_solver_residual_is_at_parity_with_the_legacy_law(self):
        residuals = {}
        for band, gap in self.GRID:
            for kwargs in ({}, {"contact_compliance": 40.0}):
                model = _model(contact_regularization=1e-2,
                               contact_gate_off=band, contact_iterations=12,
                               **kwargs)
                out = SyntheticStance(model, gap).solve(model)
                residuals[(band, gap, bool(kwargs))] = float(
                    out["solver_residual"][0]
                )
        worst_legacy = max(v for k, v in residuals.items() if not k[2])
        worst_fixed = max(v for k, v in residuals.items() if k[2])
        # measured: 1.26e-4 for both, at different states
        self.assertLess(worst_fixed, 2e-4, residuals)
        self.assertLess(worst_fixed, 10.0 * worst_legacy, residuals)

    def test_a_fully_engaged_solve_does_not_depend_on_c0_at_all(self):
        """At s = 1 the law is R = reg * scale * I whatever c0 is.

        That must hold for the *truncated* iterate too, not just in the limit:
        if the step size depends on c0 then so does the 12-iteration answer,
        which is how the regression above manifests. The narrow band is what
        makes every gate exactly 0 or 1 here, so the law is genuinely inert and
        the comparison is exact rather than approximate.
        """
        gap = -0.010  # deepest point through the floor, gate exactly 1
        legacy = self._force(gap, self.DEFAULT_ITERATIONS)
        for c0 in (0.5, 4.0, 40.0, 400.0):
            forced = self._force(gap, self.DEFAULT_ITERATIONS,
                                 contact_compliance=c0)
            self.assertAlmostEqual(
                forced, legacy, delta=1e-9 * legacy,
                msg=f"c0={c0}: {forced} N against the legacy {legacy} N at s = 1",
            )

    def test_step_size_the_solver_actually_used_is_independent_of_c0(self):
        """Read it off the matrix the solver factors, without assuming a formula.

        The solver factors ``A = H + rho I``, and enabling the compliance changes
        H only by ``diag(c0 (1 - s^2) scale)``. So if ``rho`` is c0-independent,
        ``A_enabled - A_disabled`` is exactly that diagonal; if ``rho`` folds in
        c0 (the regression), every diagonal entry is additionally shifted by
        ``rho_enabled - rho_disabled``, which at c0 = 40 is ~12.6 -- three orders
        of magnitude above the tolerance below at the s = 1 slots.
        """
        reg = 1e-2
        legacy = _model(contact_regularization=reg)
        stance = SyntheticStance(legacy, 0.001)
        A_legacy = _capture_cholesky_argument(legacy, stance)
        _, _, scale, S = _W_and_scale(legacy, stance)
        for c0 in (1e-3, 1.0, 40.0, 4000.0):
            model = _model(contact_regularization=reg, contact_compliance=c0)
            A = _capture_cholesky_argument(model, stance)
            expect = th.diag_embed(
                _effective_c0(model).repeat_interleave(2)
                * (1.0 - S.square())
                * scale[:, None]
            )
            th.testing.assert_close(A - A_legacy, expect,
                                    rtol=1e-9, atol=1e-9 * c0 * float(scale[0]))


class TestPositiveDefiniteness(unittest.TestCase):
    """R >= reg * scale * I and R <= (c0 + reg) * scale * I, so H stays PD."""

    def test_H_is_pd_at_every_gate(self):
        # gaps chosen to give gate exactly 1, exactly 0, and values in between
        for gap in (-0.01, 0.0, 0.001, 0.015, 0.03, 0.045, 0.0599, 0.06, 0.2):
            for reg in (1e-2, 1e-6):
                model = _model(contact_regularization=reg, contact_compliance=40.0)
                stance = SyntheticStance(model, gap)
                H = _recover_H(model, stance)
                eig = th.linalg.eigvalsh(H[0])
                W, _, scale, S = _W_and_scale(model, stance)
                floor = reg * float(scale[0])
                self.assertGreater(float(eig.min()), 0.0,
                                   f"gap={gap} reg={reg}: eig min {eig.min()}")
                R = (H - W)[0].diagonal()
                # 1e-6 slack: H is reconstructed from the solver's own
                # ``H + rho I`` argument, and rho ~ c0 * scale here, so the
                # subtraction loses digits relative to the reg floor.
                self.assertGreaterEqual(float(R.min()), floor * (1 - 1e-6),
                                        f"gap={gap} reg={reg}: R min {R.min()} "
                                        f"below the reg floor {floor}")
                ceiling = (40.0 + reg) * float(scale[0])
                self.assertLessEqual(float(R.max()), ceiling * (1 + 1e-9))

    def test_physical_compliance_matches_the_closed_form(self):
        """Rtilde = scale [c0 (1/s^2 - 1) + reg/s^2] in the running solver."""
        reg, raw = 1e-3, 7.5
        model = _model(contact_regularization=reg, contact_compliance=40.0)
        with th.no_grad():
            model._contact_compliance_raw.fill_(raw)
        c0 = float(_effective_c0(model)[0])
        stance = SyntheticStance(model, 0.030)
        H = _recover_H(model, stance)
        W, _, scale, S = _W_and_scale(model, stance)
        R = (H - W)[0].diagonal()
        s = S[0]
        # Rtilde = R / s^2 amplifies the H-reconstruction error by 1/s^2, so the
        # identity is checked where the gate is appreciable.
        active = s > 1e-3
        self.assertTrue(bool(active.any()))
        Rtilde = (R / s.square())[active]
        expect = float(scale[0]) * (
            c0 * (1.0 / s[active].square() - 1.0) + reg / s[active].square()
        )
        th.testing.assert_close(Rtilde, expect, rtol=1e-9, atol=0.0)


class TestDisabledPathUntouched(unittest.TestCase):
    """AC4. Default-disabled must be the pre-fix code, not merely close to it."""

    # state_dict lists a module's own parameters before its buffers, so
    # _contact_raw precedes both version markers.
    EXPECTED_KINEMATIC_KEYS = (
        "_contact_raw",
        "_contact_geometry_version",
        "_contact_solver_version",
        "_tangent_onehot",
    )

    def test_no_new_state_dict_key_when_disabled(self):
        keys = list(_model().state_dict().keys())
        self.assertNotIn("_contact_compliance_raw", keys)
        contact = [k for k in keys if k.startswith(("_contact", "_tangent", "_kin"))]
        self.assertEqual(tuple(contact), self.EXPECTED_KINEMATIC_KEYS, contact)
        self.assertIsNone(_model()._contact_compliance_raw)

    def test_enabled_adds_exactly_one_key_after_contact_raw(self):
        keys = list(_model(contact_compliance=40.0).state_dict().keys())
        base = list(_model().state_dict().keys())
        self.assertEqual(len(keys), len(base) + 1)
        self.assertEqual([k for k in keys if k not in base],
                         ["_contact_compliance_raw"])
        self.assertEqual(keys.index("_contact_compliance_raw"),
                         keys.index("_contact_raw") + 1)

    def test_disabled_parameters_are_bit_identical_to_an_unaware_model(self):
        """Registering the parameter must not disturb the RNG stream."""
        a, b = _model().state_dict(), _model(contact_compliance=40.0).state_dict()
        for key, value in a.items():
            if key == "_contact_solver_version":
                self.assertEqual((int(value), int(b[key])), (1, 2))
                continue
            self.assertTrue(th.equal(value, b[key]), key)

    def test_disabled_regularizer_is_the_legacy_literal_expression(self):
        """H must be bit-for-bit ``W + reg * scale * eye``, not a diag_embed."""
        for reg in REGS:
            model = _model(contact_regularization=reg)
            stance = SyntheticStance(model, 0.030)
            calls = []
            real = th.diag_embed

            def spy(*a, **kw):
                calls.append(a)
                return real(*a, **kw)

            th.diag_embed = spy
            try:
                A = _capture_cholesky_argument(model, stance)
            finally:
                th.diag_embed = real
            self.assertEqual(calls, [], "the disabled path must not build a "
                                        "per-coordinate diagonal regularizer")
            W, _, scale, _ = _W_and_scale(model, stance)
            eye = th.eye(W.shape[-1], dtype=W.dtype)[None]
            legacy = W + reg * scale[:, None, None] * eye
            rho = legacy.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-6)
            self.assertTrue(
                th.equal(A, legacy + rho[:, None, None] * eye),
                f"reg={reg}: the disabled regularizer is no longer the literal "
                "W + contact_regularization * scale * eye",
            )

    def test_drift_is_finite_and_unchanged_in_shape(self):
        model = _model()
        th.manual_seed(3)
        x = th.randn(4, 17, dtype=th.float64)
        x[:, 0] = th.linspace(-0.02, 0.05, 4, dtype=th.float64)
        a = th.randn(4, 6, dtype=th.float64)
        with th.no_grad():
            out = model._structured_drift(x, a)
        self.assertEqual(tuple(out.shape), (4, 17))
        self.assertTrue(bool(th.isfinite(out).all()))


class TestGradientReachesC0(unittest.TestCase):
    """AC5. ``rho`` and ``scale`` are detached; c0 must not be lost with them."""

    def _loss_and_grad(self, gap: float, reg: float = 1e-2):
        model = _model(contact_regularization=reg, contact_compliance=40.0)
        stance = SyntheticStance(model, gap)
        out = model._constraint_contact_solve(
            stance.pos, stance.qd, stance.M, stance.qdd_free
        )
        loss = out["contact_acceleration"].square().sum()
        model.zero_grad(set_to_none=True)
        loss.backward()
        return float(loss), model._contact_compliance_raw.grad

    def test_gradient_is_finite_nonzero_and_of_sane_magnitude(self):
        loss, grad = self._loss_and_grad(0.030)
        self.assertIsNotNone(grad)
        self.assertTrue(bool(th.isfinite(grad).all()), grad)
        self.assertGreater(float(grad.abs().max()), 0.0, grad)
        # loss ~ (impulse/dt)^2 and dL/dc0 ~ -2 L / c0-ish: nothing exploding
        self.assertLess(float(grad.abs().max()), 1e12, grad)
        self.assertGreater(loss, 0.0)

    def test_only_partially_gated_contacts_receive_gradient(self):
        """dR/dc0 = (1 - s^2) scale, which is exactly zero at s = 1 and the
        impulse is exactly zero at s = 0, so a nonzero gradient is the signature
        of a contact in the taper -- not of a leak."""
        model = _model(contact_regularization=1e-2, contact_compliance=40.0)
        stance = SyntheticStance(model, 0.030)
        gate = stance.gate(model)
        out = model._constraint_contact_solve(
            stance.pos, stance.qd, stance.M, stance.qdd_free
        )
        out["contact_acceleration"].square().sum().backward()
        grad = model._contact_compliance_raw.grad
        partial = (gate > 0.0) & (gate < 1.0)
        self.assertTrue(bool(partial.any()), gate)
        self.assertTrue(bool((grad[partial] != 0).all()), (grad, gate))
        self.assertTrue(bool((grad[~partial] == 0).all()), (grad, gate))

    def test_finite_difference_agrees_with_autograd(self):
        """Checked in the tapering region, where 12 ADMM iterations converge.

        ``rho`` is the detached mean diagonal of H and now depends on c0, so at
        near-full engagement the truncated 12-iteration iterate has a real
        dependence on c0 through rho that autograd deliberately does not carry.
        The two agree to 1e-7 there once the solve is converged; see
        results/contact_compliance/admm_rho_diagnostic.json.
        """
        reg, gap = 1e-2, 0.030
        base_raw = 2.0

        def loss_at(raw):
            model = _model(contact_regularization=reg, contact_compliance=40.0)
            with th.no_grad():
                model._contact_compliance_raw.fill_(raw)
            stance = SyntheticStance(model, gap)
            out = model._constraint_contact_solve(
                stance.pos, stance.qd, stance.M, stance.qdd_free
            )
            loss = out["contact_acceleration"].square().sum()
            return model, loss

        model, loss = loss_at(base_raw)
        model.zero_grad(set_to_none=True)
        loss.backward()
        analytic = float(model._contact_compliance_raw.grad.sum())
        h = 1e-6
        _, up = loss_at(base_raw + h)
        _, down = loss_at(base_raw - h)
        numeric = (float(up) - float(down)) / (2 * h)
        self.assertAlmostEqual(analytic, numeric,
                               delta=max(1e-5, 1e-4 * abs(numeric)))

    def test_gradient_also_reaches_the_existing_constitutive_parameters(self):
        model = _model(contact_regularization=1e-2, contact_compliance=40.0)
        stance = SyntheticStance(model, 0.030)
        out = model._constraint_contact_solve(
            stance.pos, stance.qd, stance.M, stance.qdd_free
        )
        out["contact_acceleration"].square().sum().backward()
        self.assertTrue(bool(th.isfinite(model._contact_raw.grad).all()))


class TestConstructorAndSidecars(unittest.TestCase):
    """Compatibility: version 2, round-trip, and loud failure on a mismatch."""

    def test_solver_version_marker(self):
        self.assertEqual(int(_model()._contact_solver_version), 1)
        self.assertEqual(
            int(_model(contact_compliance=40.0)._contact_solver_version), 2)

    def test_round_trip_enabled(self):
        src = _model(contact_compliance=40.0)
        with th.no_grad():
            src._contact_compliance_raw.copy_(
                th.arange(6, dtype=th.float64) - 2.0)
        dst = _model(contact_compliance=1.0)
        dst.load_state_dict(src.state_dict())
        self.assertTrue(th.equal(dst._contact_compliance_raw,
                                 src._contact_compliance_raw))
        self.assertEqual(int(dst._contact_solver_version), 2)

    def test_round_trip_disabled(self):
        src, dst = _model(), _model()
        dst.load_state_dict(src.state_dict())
        self.assertEqual(int(dst._contact_solver_version), 1)
        self.assertIsNone(dst._contact_compliance_raw)

    def test_loading_a_version2_sidecar_into_a_disabled_model_raises(self):
        sd = _model(contact_compliance=40.0).state_dict()
        with self.assertRaises(RuntimeError) as ctx:
            _model().load_state_dict(sd)
        self.assertIn("contact compliance mismatch", str(ctx.exception))

    def test_loading_a_version1_sidecar_into_an_enabled_model_raises(self):
        sd = _model().state_dict()
        with self.assertRaises(RuntimeError) as ctx:
            _model(contact_compliance=40.0).load_state_dict(sd)
        self.assertIn("contact compliance mismatch", str(ctx.exception))

    def test_unsupported_version_still_raises(self):
        sd = dict(_model().state_dict())
        sd["_contact_solver_version"] = th.tensor(3, dtype=th.int64)
        with self.assertRaises(RuntimeError) as ctx:
            _model().load_state_dict(sd)
        self.assertIn("unsupported contact solver version 3", str(ctx.exception))

    def test_markerless_sidecar_still_loads_as_the_legacy_compliant_law(self):
        model = PortHamiltonianModel(
            obs_dim=17, action_dim=6, mode="structured", structured_hidden=(32, 32),
            contact_force=4, contact_geometry="learned",
            contact_solver="compliant", dof_layout=DOFLayout.cheetah(),
        )
        sd = dict(model.state_dict())
        sd.pop("_contact_solver_version")
        sd.pop("_contact_geometry_version")
        fresh = PortHamiltonianModel(
            obs_dim=17, action_dim=6, mode="structured", structured_hidden=(32, 32),
            contact_force=4, contact_geometry="learned",
            contact_solver="constraint", dof_layout=DOFLayout.cheetah(),
        )
        fresh.load_state_dict(sd)
        self.assertEqual(fresh.contact_solver, "compliant")

    def test_compliance_requires_the_constraint_solver(self):
        with self.assertRaises(ValueError) as ctx:
            PortHamiltonianModel(
                obs_dim=17, action_dim=6, mode="structured",
                structured_hidden=(32, 32), contact_force=4,
                contact_geometry="learned", contact_solver="compliant",
                contact_compliance=40.0, dof_layout=DOFLayout.cheetah(),
            )
        self.assertIn("contact_solver='constraint'", str(ctx.exception))

    def test_non_positive_compliance_is_refused(self):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                _model(contact_compliance=bad)

    def test_compliance_without_contact_points_is_refused(self):
        """Otherwise the caller believes a law is active on a contactless model."""
        with self.assertRaises(ValueError) as ctx:
            PortHamiltonianModel(
                obs_dim=17, action_dim=6, mode="structured",
                structured_hidden=(32, 32), contact_force=0,
                contact_compliance=40.0, dof_layout=DOFLayout.cheetah(),
            )
        self.assertIn("contact_force=0", str(ctx.exception))

    def test_false_and_none_both_mean_disabled(self):
        for value in (None, False):
            model = _model(contact_compliance=value)
            self.assertIsNone(model._contact_compliance_raw)
            self.assertEqual(int(model._contact_solver_version), 1)

    def test_true_uses_the_class_default(self):
        model = _model(contact_compliance=True)
        self.assertEqual(model.contact_compliance,
                         PortHamiltonianModel._DEFAULT_CONTACT_COMPLIANCE)
        th.testing.assert_close(
            _effective_c0(model),
            th.full((6,), PortHamiltonianModel._DEFAULT_CONTACT_COMPLIANCE,
                    dtype=th.float64),
            rtol=1e-6, atol=1e-9,
        )

    def test_c0_initializes_to_the_requested_physical_value(self):
        for value in (0.05, 1.0, 7.5, 40.0):
            model = _model(contact_compliance=value)
            # The raw initializer is written with th.full at the default dtype,
            # exactly like _contact_raw, so a .double() model carries a value
            # that was rounded to float32 first.
            th.testing.assert_close(
                _effective_c0(model),
                th.full((6,), value, dtype=th.float64),
                rtol=1e-6, atol=1e-9,
            )


class TestEntryPointsCanBuildAnEnabledModel(unittest.TestCase):
    """A law nothing can construct is a law nothing can train.

    ``benchmarks/run_ct_rl.py`` pops every other ``dynamics_contact_*`` column
    into the model kwargs; a column it does not pop is forwarded to the CTSAC
    constructor instead, which raises TypeError on the unexpected keyword. And
    ``evaluations/hamiltonian_recovery.py`` cannot load a sidecar written with the
    compliance enabled unless it can build an enabled model, because the solver
    version marker (correctly) refuses the mismatch.
    """

    def test_csv_column_reaches_the_model_kwargs(self):
        from benchmarks.run_ct_rl import _pop_structured_model_kwargs

        cases = {
            "40": 40.0, "0.5": 0.5, "1": 1.0,  # a number is c0 itself
            "true": True, "yes": True,
            "false": None, "none": None,
        }
        for cell, expected in cases.items():
            algo_kwargs = {
                "dynamics_contact_solver": "constraint",
                "dynamics_contact_compliance": cell,
            }
            model_kwargs = _pop_structured_model_kwargs(algo_kwargs)
            self.assertEqual(model_kwargs.get("contact_compliance", "<absent>"),
                             expected, cell)
            self.assertNotIn("dynamics_contact_compliance", algo_kwargs, cell)

    def test_absent_or_blank_column_leaves_the_law_disabled(self):
        from benchmarks.run_ct_rl import _pop_structured_model_kwargs

        for algo_kwargs in ({"dynamics_contact_solver": "constraint"},
                            {"dynamics_contact_solver": "constraint",
                             "dynamics_contact_compliance": ""}):
            model_kwargs = _pop_structured_model_kwargs(algo_kwargs)
            self.assertNotIn("contact_compliance", model_kwargs)
            self.assertNotIn("dynamics_contact_compliance", algo_kwargs)

    def test_the_popped_kwargs_actually_construct_an_enabled_model(self):
        from benchmarks.run_ct_rl import _pop_structured_model_kwargs

        model_kwargs = _pop_structured_model_kwargs({
            "dynamics_contact_solver": "constraint",
            "dynamics_contact_geometry": "kinematic",
            "dynamics_contact_compliance": "40",
        })
        model = PortHamiltonianModel(
            obs_dim=17, action_dim=6, mode="structured",
            structured_hidden=(32, 32), contact_force=6,
            dof_layout=DOFLayout.cheetah(), **model_kwargs,
        )
        self.assertEqual(model.contact_compliance, 40.0)
        self.assertEqual(int(model._contact_solver_version), 2)

    def test_recovery_flag_parses_and_reaches_the_constructor(self):
        from evaluations.hamiltonian_recovery import _parse_contact_compliance

        self.assertIsNone(_parse_contact_compliance(None))
        self.assertIsNone(_parse_contact_compliance(""))
        self.assertIsNone(_parse_contact_compliance("false"))
        self.assertIs(_parse_contact_compliance("true"), True)
        self.assertEqual(_parse_contact_compliance("40"), 40.0)
        # the flag exists and is wired into every model this module builds
        import inspect

        from evaluations import hamiltonian_recovery as hr

        self.assertIn("contact_compliance",
                      inspect.signature(hr.fit_model).parameters)
        source = inspect.getsource(hr.main)
        self.assertIn("--contact_compliance", source)
        self.assertEqual(source.count("contact_compliance=contact_compliance"), 2)

    def test_a_version2_sidecar_round_trips_through_the_recovery_flag(self):
        from evaluations.hamiltonian_recovery import _parse_contact_compliance

        src = _model(contact_compliance=40.0)
        with th.no_grad():
            src._contact_compliance_raw.copy_(th.arange(6, dtype=th.float64))
        sidecar = src.state_dict()
        # what --contact_compliance 40 builds
        dst = _model(contact_compliance=_parse_contact_compliance("40"))
        dst.load_state_dict(sidecar)
        self.assertTrue(th.equal(dst._contact_compliance_raw,
                                 src._contact_compliance_raw))
        # and what the flag's absence builds, which must still refuse it loudly
        with self.assertRaises(RuntimeError):
            _model(contact_compliance=_parse_contact_compliance(None)
                   ).load_state_dict(sidecar)


@unittest.skipUnless(_HAVE_MUJOCO, "dm_control/MuJoCo not installed")
class TestDropTestAgainstMuJoCo(unittest.TestCase):
    """AC3. A dropped cheetah must rest at the floor, whatever the gate band is.

    What the fix buys, measured: the resting height stops depending on the width
    of the gate band. It does NOT reproduce MuJoCo's resting penetration -- the
    converged model rests 5.5e-4 m above where MuJoCo settles (-5.58e-4 m read
    through the model's own gap function), which is the same offset the
    compliance-disabled model has at a narrow band. Only the *band dependence*
    is fixed: the legacy law levitates 2.58e-2 m in a 0.06 m band.
    """

    STEPS = 3000  # 1500 is not settled: |qvel| is still ~1e-2 m/s there

    @classmethod
    def setUpClass(cls):
        cls.mech = MuJoCoCheetahMechanics()

    def _probe(self, **kwargs):
        """The probe harness model: same seed/hidden sizes as the battery."""
        from evaluations.contact_compliance_probe import make_probe_model
        return make_probe_model(contact_regularization=1e-2, **kwargs)

    def test_reference_rest_gap(self):
        ref = mujoco_reference_rest(self._probe(contact_gate_off=0.005),
                                    self.mech, z0=0.6, steps=self.STEPS)
        self.assertLess(abs(ref["model_min_gap_at_mujoco_rest_m"]), 5e-3)

    def test_compliance_rests_at_the_floor_and_carries_body_weight(self):
        ref = mujoco_reference_rest(self._probe(contact_gate_off=0.005),
                                    self.mech, z0=0.6, steps=self.STEPS)
        mujoco_gap = ref["model_min_gap_at_mujoco_rest_m"]
        gaps = {}
        for band in (0.005, 0.06):
            model = self._probe(contact_gate_off=band, contact_compliance=40.0)
            drop = drop_test(model, self.mech, steps=self.STEPS, z0=0.6)
            gaps[band] = drop["rest_min_gap_m"]
            # at the floor: no invisible shelf anywhere in the band
            self.assertLess(
                abs(drop["rest_min_gap_m"]), 1e-4,
                msg=f"band {band}: rest gap {drop['rest_min_gap_m']} m",
            )
            # and still within a millimetre of where MuJoCo settles
            self.assertLess(abs(drop["rest_min_gap_m"] - mujoco_gap), 1e-3)
            self.assertAlmostEqual(
                drop["rest_total_normal_force_over_weight"], 1.0, delta=0.02,
                msg=f"band {band}: {drop['rest_total_normal_force_N']} N",
            )
            self.assertEqual(drop["cone_violation"], 0.0)
            self.assertFalse(drop["diverged"])
        self.assertLess(
            abs(gaps[0.005] - gaps[0.06]), 1e-5,
            f"the resting height must not depend on the gate band: {gaps}",
        )

    def test_legacy_law_levitates_in_a_wide_band(self):
        """The pathology AC3 fixes, kept measured so the fix cannot silently rot."""
        model = self._probe(contact_gate_off=0.06)
        drop = drop_test(model, self.mech, steps=self.STEPS, z0=0.6)
        self.assertGreater(drop["rest_min_gap_m"], 5e-3)
        self.assertAlmostEqual(drop["rest_total_normal_force_over_weight"], 1.0,
                               delta=0.02)


class _Float32Stance:
    """A float32 twin of ``SyntheticStance``, for the gate-overshoot guard.

    ``SyntheticStance`` is float64 by construction, and float64 is exactly the
    dtype in which the defect below cannot occur, so this class exists to reach
    it. The interface is whatever ``_capture_cholesky_argument``, ``_admm_rho``
    and ``_W_and_scale`` consume: ``pos``, ``qd``, ``M``, ``qdd_free``, ``solve``.
    """

    def __init__(self, model, gap: float):
        lo, hi = model.layout.pos_slice
        nv = model.layout.nv
        self.pos = th.zeros(1, hi - lo)
        self.qd = th.zeros(1, nv)
        height_pos = int(model._kin_height_pos.reshape(-1)[0])
        with th.no_grad():
            current = float(model._contact_geometry(self.pos, self.qd)[0].min())
        self.pos[0, height_pos] = gap - current
        idx = th.arange(nv, dtype=th.float32)
        band = th.exp(-(idx[:, None] - idx[None, :]).abs())
        self.M = (th.eye(nv) * 2.0 + 0.3 * band)[None]
        self.qdd_free = th.zeros(1, nv)
        self.qdd_free[0, int(model._kin_height_cfg.reshape(-1)[0])] = -9.81

    def solve(self, model):
        with th.no_grad():
            return model._constraint_contact_solve(
                self.pos, self.qd, self.M, self.qdd_free
            )

    def gate(self, model) -> th.Tensor:
        with th.no_grad():
            return model._contact_gate(
                model._contact_geometry(self.pos, self.qd)[0]
            )[0]


class TestFloat32GateOvershootCannotBreakTheFloor(unittest.TestCase):
    """R >= reg * scale * I must survive the float32 gate exceeding 1.

    Every other test in this file runs in float64, where the quintic smoothstep
    u^3 (10 - 15u + 6u^2) is exactly bounded by 1 -- so ``1 - s^2 >= 0`` holds for
    free and none of them can see this. In float32, the production dtype, the
    quintic overshoots to 1.0000009536743164 just below u = 1. Then ``1 - s^2``
    reaches -1.9e-6 and ``c0 * (1 - s^2)`` becomes a *negative* diagonal
    contribution, so R falls below reg * scale once c0 * 1.9e-6 exceeds reg.

    That is reachable rather than hypothetical: c0 is learned, unbounded above,
    and a 100k-step fit was measured taking it from 40 to 58. The solver clamps
    the shape factor at zero for this reason. Without the clamp R goes negative
    at the state below; H still happens to stay positive definite there because
    W's own diagonal dominates, which is exactly what makes this worth pinning --
    the symptom would first appear at some state where it does not, and
    ``cholesky_ex`` would report it as a diverged mass matrix, blaming the wrong
    term entirely.
    """

    GAP = 1.0e-7          # measured to sit inside the float32 overshoot band
    BAND = 0.06
    BIG_C0 = 1.0e5        # past the c0 * 1.9e-6 > reg threshold at reg = 1e-2
    REG = 1e-2

    def _model32(self, **kwargs):
        th.manual_seed(0)
        kwargs.setdefault("contact_gate_off", self.BAND)
        kwargs.setdefault("contact_regularization", self.REG)
        model = PortHamiltonianModel(
            obs_dim=17,
            action_dim=6,
            mode="structured",
            structured_hidden=(32, 32),
            contact_force=6,
            contact_geometry="kinematic",
            contact_solver="constraint",
            dof_layout=DOFLayout.cheetah(),
            **kwargs,
        )
        return model.eval().float()

    def test_the_quintic_overshoots_one_by_order_eps_in_every_dtype(self):
        """The hazard is real, reachable through the model's own gate, and it is
        a rounding artifact rather than a float32 quirk.

        S(u) = u^3 (10 - 15u + 6u^2) is mathematically <= 1 on [0, 1] with
        S(1) = 1 exactly, but evaluating it just below u = 1 rounds above 1 by a
        few eps. Measured: +7.2e-7 in float32 (6 eps) and +4.4e-16 in float64
        (2 eps). Which grid points trip it is dtype- and grid-dependent, so this
        asserts the bound rather than a specific count -- what matters is that
        the excess is O(eps) and therefore that the c0 threshold for breaking the
        reg floor scales as reg / eps: about 6e3 in float32 and 1e13 in float64.
        """
        model = self._model32(contact_compliance=self.BIG_C0)
        for dtype in (th.float32, th.float64):
            probe = model.double() if dtype is th.float64 else model.float()
            eps = th.finfo(dtype).eps
            g = th.linspace(0.0, 2.5e-4, 4001, dtype=dtype)
            with th.no_grad():
                gate = probe._contact_gate(g)
            excess = float((gate - 1.0).clamp_min(0.0).max())
            self.assertLessEqual(excess, 16 * eps,
                                 f"dtype={dtype}: gate exceeds 1 by {excess:.3e},"
                                 f" more than the O(eps) rounding this guards")
            # and the shape factor the solver feeds R would go negative by the
            # same order, which is precisely what clamp_min(0.0) absorbs
            unclamped = float((1.0 - gate.square()).min())
            self.assertGreaterEqual(unclamped, -32 * eps)
            self.assertGreaterEqual(
                float((1.0 - gate.square()).clamp_min(0.0).min()), 0.0)
        # float32 must actually reach it, or the guard below is untested
        probe = model.float()
        with th.no_grad():
            gate = probe._contact_gate(
                th.linspace(0.0, 2.5e-4, 4001, dtype=th.float32))
        self.assertGreater(float(gate.max()), 1.0)

    def test_the_clamped_shape_factor_is_never_negative(self):
        model = self._model32(contact_compliance=self.BIG_C0)
        for dtype in (th.float32, th.float64):
            probe = model.double() if dtype is th.float64 else model.float()
            g = th.linspace(0.0, 2.5e-4, 4001, dtype=dtype)
            with th.no_grad():
                gate = probe._contact_gate(g)
            shape = (1.0 - gate.square()).clamp_min(0.0)
            self.assertGreaterEqual(float(shape.min()), 0.0, f"dtype={dtype}")

    def test_R_respects_the_reg_floor_in_float32_at_a_large_c0(self):
        """The regression guard: fails without ``clamp_min(0.0)`` in the solver.

        R is read back off the matrix the solver hands to ``cholesky_ex``, so this
        pins the shipped expression rather than a re-derivation of it.
        """
        model = self._model32(contact_compliance=self.BIG_C0)
        stance = _Float32Stance(model, self.GAP)
        gate = stance.gate(model)
        # If the probe state stops overshooting the gate this test becomes
        # vacuous and would keep passing through a regression, so assert it.
        self.assertGreater(
            float(gate.max()), 1.0,
            "probe state no longer overshoots the float32 gate; re-pick GAP "
            "inside the overshoot band or this guard is vacuous")
        W, _, scale, _ = _W_and_scale(model, stance)
        R = (_recover_H(model, stance) - W)[0].diagonal()
        floor = self.REG * float(scale[0])
        self.assertGreaterEqual(
            float(R.min()), floor * (1 - 1e-4),
            f"R min {float(R.min()):.6e} is below the reg * scale floor "
            f"{floor:.6e}: the float32 gate overshoot is not being clamped")

    def test_the_floor_is_breached_at_the_default_c0_too_not_just_a_large_one(self):
        """The invariant fails well before positive-definiteness does.

        Measured R.min / floor without the clamp at this state:
            c0 = 40      0.999046      (the shipped default)
            c0 = 1e3     0.976158
            c0 = 6e3     0.856949
            c0 = 1e5    -1.384186      R itself goes negative
        So R >= reg * scale * I -- which the solver comment asserts and which the
        taper argument relies on -- is already false at c0 = 40, by 0.1 %. Only
        the outright sign change, and hence any chance of a cholesky failure,
        needs c0 of order reg / eps. This is pinned separately from the large-c0
        case so that a regression is caught at the value actually in use rather
        than only at a stress value nobody runs.
        """
        model = self._model32(contact_compliance=40.0)
        stance = _Float32Stance(model, self.GAP)
        W, _, scale, _ = _W_and_scale(model, stance)
        R = (_recover_H(model, stance) - W)[0].diagonal()
        self.assertGreaterEqual(float(R.min()),
                                self.REG * float(scale[0]) * (1 - 1e-4))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
