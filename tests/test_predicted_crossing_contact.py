"""Focused physical invariants for solver-version-3 ground contact.

The fixture is a two-DOF point mass: horizontal translation is cyclic and the
single observed position is the vertical gap itself.  Calling the contact solve
with a prescribed mass matrix and free acceleration keeps these tests about the
contact law, rather than about randomly initialized mechanics networks.

``F = k * penetration`` is an equilibrium constitutive statement.  A penetrated
mass with no external load should accelerate out of the floor, so the static
tests apply a downward load and choose ``penetration = load / k``.  At that
loaded equilibrium the outgoing normal velocity is zero and the QP must return
the balancing force for every response timestep.
"""

import math
import unittest

import torch as th

from models.port_hamiltonian import (
    DOFLayout,
    PlanarContactPoint,
    PortHamiltonianModel,
)


_DTYPE = th.float64
_POINT = PlanarContactPoint(
    name="point_foot",
    radius=1.0,
    height_pos=0,
    base_height=1.0,
    angle_pos=(),
    offsets=(),
)
_LAYOUT = DOFLayout(
    obs_dim=3,
    pos_slice=(0, 1),
    vel_slice=(1, 3),
    cyclic_cfg=(0,),
    obs_pos_to_cfg=(1,),
    act_to_cfg=(0, 1),
    contact_tangent_cfg=0,
    contact_points=(_POINT,),
)
_SECOND_POINT = PlanarContactPoint(
    name="point_foot_2",
    radius=1.0,
    height_pos=0,
    base_height=1.0,
    angle_pos=(),
    offsets=(),
)
_TWO_POINT_LAYOUT = DOFLayout(
    obs_dim=3,
    pos_slice=(0, 1),
    vel_slice=(1, 3),
    cyclic_cfg=(0,),
    obs_pos_to_cfg=(1,),
    act_to_cfg=(0, 1),
    contact_tangent_cfg=0,
    contact_points=(_POINT, _SECOND_POINT),
)


def _model(
    *,
    stiffness=2_000.0,
    compliance=None,
    dt=0.002,
    iterations=200,
    attenuation=None,
):
    """Build a deterministic one-point version-3 model (or a legacy control)."""
    th.manual_seed(19)
    return PortHamiltonianModel(
        obs_dim=3,
        action_dim=2,
        mode="structured",
        structured_hidden=(8,),
        contact_force=1,
        contact_geometry="kinematic",
        contact_solver="constraint",
        contact_dt=dt,
        contact_iterations=iterations,
        contact_compliance=compliance,
        contact_stiffness=stiffness,
        contact_attenuation=attenuation,
        dof_layout=_LAYOUT,
    ).double().eval()


def _two_point_model(*, stiffness=2_000.0, dt=0.002, iterations=200):
    th.manual_seed(19)
    return PortHamiltonianModel(
        obs_dim=3,
        action_dim=2,
        mode="structured",
        structured_hidden=(8,),
        contact_force=2,
        contact_geometry="kinematic",
        contact_solver="constraint",
        contact_dt=dt,
        contact_iterations=iterations,
        contact_stiffness=stiffness,
        dof_layout=_TWO_POINT_LAYOUT,
    ).double().eval()


def _loaded_equilibrium(model, load=3.0, mass=2.0):
    """Inputs for a static point carrying ``load`` newtons.

    With ``delta = load/k`` and free vertical acceleration ``-load/mass``, a
    physical contact force of ``k*delta`` exactly cancels the free velocity in
    the QP's response interval.
    """
    stiffness = float(th.exp(model._contact_stiffness_log.detach())[0])
    penetration = float(load) / stiffness
    pos = th.tensor([[-penetration]], dtype=_DTYPE)
    qd = th.zeros(1, 2, dtype=_DTYPE)
    M = th.diag(th.tensor([1.0, float(mass)], dtype=_DTYPE)).unsqueeze(0)
    qdd_free = th.tensor([[0.0, -float(load) / float(mass)]], dtype=_DTYPE)
    return pos, qd, M, qdd_free, stiffness, penetration


class TestPredictedCrossingActivation(unittest.TestCase):
    def test_crossing_activates_but_nearby_non_crossing_is_exactly_silent(self):
        dt = 0.002
        model = _model(stiffness=100_000.0, dt=dt, iterations=100)

        # Both points are currently 1 mm above the plane and have substantial
        # tangential slip.  Only the first reaches the plane under the free
        # semi-implicit prediction: 1 mm + 2 ms*(-1 m/s) = -1 mm.
        pos = th.tensor([[0.001], [0.001]], dtype=_DTYPE)
        qd = th.tensor([[2.0, -1.0], [10.0, -0.4]], dtype=_DTYPE)
        M = th.eye(2, dtype=_DTYPE).expand(2, -1, -1)
        qdd_free = th.zeros_like(qd)

        with th.no_grad():
            out = model._constraint_contact_solve(pos, qd, M, qdd_free)

        th.testing.assert_close(
            out["predicted_gap_free"][:, 0],
            th.tensor([-0.001, 0.0002], dtype=_DTYPE),
            rtol=0.0,
            atol=1e-15,
        )
        self.assertEqual(out["active_contact"][:, 0].tolist(), [True, False])
        self.assertGreater(float(out["normal_impulse"][0, 0]), 0.0)

        # A positive-gap crossing uses a landing target, not restitution before
        # impact: the allowed outgoing normal velocity is exactly -g/h.
        th.testing.assert_close(
            out["normal_velocity_target"][0, 0],
            -pos[0, 0] / dt,
            rtol=0.0,
            atol=1e-12,
        )

        # The inactive-set equality applies to both slots.  This must be bitwise
        # zero even though the inactive point has a large tangential velocity.
        for key in (
            "normal_impulse",
            "tangent_impulse",
            "normal_force",
            "tangent_force",
        ):
            actual = out[key][1]
            self.assertTrue(th.equal(actual, th.zeros_like(actual)), (key, actual))
        self.assertTrue(
            th.equal(
                out["generalized_impulse"][1],
                th.zeros_like(out["generalized_impulse"][1]),
            )
        )
        self.assertEqual(float(out["solver_residual"][1]), 0.0)

    def test_free_acceleration_can_activate_a_crossing_from_rest(self):
        model = _model(stiffness=100_000.0, dt=0.002, iterations=100)
        pos = th.tensor([[0.001]], dtype=_DTYPE)
        qd = th.zeros(1, 2, dtype=_DTYPE)
        M = th.eye(2, dtype=_DTYPE).unsqueeze(0)
        # The vertical free endpoint speed is -1 m/s, so the predicted endpoint
        # gap is -1 mm even though the current point is instantaneously at rest.
        qdd_free = th.tensor([[0.0, -500.0]], dtype=_DTYPE)

        with th.no_grad():
            out = model._constraint_contact_solve(pos, qd, M, qdd_free)

        self.assertTrue(bool(out["active_contact"][0, 0]))
        self.assertAlmostEqual(
            float(out["predicted_gap_free"][0, 0]), -0.001, places=14
        )
        self.assertGreater(float(out["normal_impulse"][0, 0]), 0.0)


class TestPhysicalStiffness(unittest.TestCase):
    def test_recovery_fraction_is_fixed_persisted_configuration(self):
        for attenuation in (0.05, 0.2, 0.4):
            with self.subTest(attenuation=attenuation):
                model = _model(attenuation=attenuation)
                # Version 3 learns only restitution and friction here; there is
                # no dormant trainable beta row.
                self.assertEqual(tuple(model._contact_raw.shape), (2, 1))
                self.assertNotIn(
                    "_contact_attenuation", dict(model.named_parameters())
                )
                self.assertIn("_contact_attenuation", model.state_dict())

                pos, qd, M, qdd_free, stiffness, _ = _loaded_equilibrium(model)
                out = model._constraint_contact_solve(pos, qd, M, qdd_free)
                th.testing.assert_close(
                    out["beta"],
                    th.tensor([attenuation], dtype=_DTYPE),
                    rtol=0.0,
                    atol=0.0,
                )
                self.assertAlmostEqual(
                    float(out["physical_compliance"][0].detach()),
                    attenuation / (stiffness * model.contact_dt**2),
                    places=12,
                )
                self.assertAlmostEqual(
                    float(out["normal_force"][0, 0].detach()), 3.0
                )

    def test_loaded_static_force_is_k_delta_and_timestep_invariant(self):
        timesteps = (0.0005, 0.001, 0.002, 0.004)
        # 200 N also exercises a penetration deep enough that the legacy
        # recovery-velocity cap would have broken F=k*delta at the shorter h.
        loads = (1.0, 2.0, 4.0, 200.0)
        forces_by_load = {load: [] for load in loads}

        for dt in timesteps:
            model = _model(stiffness=2_000.0, dt=dt, iterations=200)
            for load in loads:
                with self.subTest(dt=dt, load=load), th.no_grad():
                    pos, qd, M, qdd_free, stiffness, penetration = (
                        _loaded_equilibrium(model, load=load, mass=2.0)
                    )
                    out = model._constraint_contact_solve(pos, qd, M, qdd_free)
                    expected_force = stiffness * penetration
                    force = float(out["normal_force"][0, 0])
                    impulse = float(out["normal_impulse"][0, 0])

                    self.assertAlmostEqual(force, expected_force, places=11)
                    self.assertAlmostEqual(impulse, expected_force * dt, places=13)
                    self.assertLess(
                        abs(float(out["contact_velocity_post"][0, 0, 0])),
                        1e-12,
                    )
                    self.assertLess(float(out["solver_residual"][0]), 1e-12)
                    forces_by_load[load].append(force)

        for load, forces in forces_by_load.items():
            with self.subTest(load=load, invariant_across_dt=True):
                self.assertLess(max(forces) - min(forces), 1e-10)

    def test_simultaneous_contacts_share_load_and_ignore_an_inactive_block(self):
        model = _two_point_model(stiffness=2_000.0)
        M = th.diag(th.tensor([1.0, 2.0], dtype=_DTYPE)).unsqueeze(0)
        qd = th.zeros(1, 2, dtype=_DTYPE)
        qdd_free = th.tensor([[0.0, -2.0]], dtype=_DTYPE)  # 4 N load

        # Two identical 2 kN/m contacts at 1 mm penetration carry 2 N each.
        with th.no_grad():
            shared = model._constraint_contact_solve(
                th.tensor([[-0.001]], dtype=_DTYPE), qd, M, qdd_free
            )
        th.testing.assert_close(
            shared["normal_force"][0],
            th.tensor([2.0, 2.0], dtype=_DTYPE),
            rtol=0.0,
            atol=5e-7,
        )

        # Move only the second collision surface 10 cm away.  The first point
        # now carries the same load alone at 2 mm penetration, while the
        # inactive point's normal and tangent blocks remain exact zero.
        with th.no_grad():
            model._kin_radius[1] -= 0.1
            alone = model._constraint_contact_solve(
                th.tensor([[-0.002]], dtype=_DTYPE), qd, M, qdd_free
            )
        self.assertEqual(alone["active_contact"][0].tolist(), [True, False])
        self.assertAlmostEqual(float(alone["normal_force"][0, 0]), 4.0, places=6)
        self.assertEqual(float(alone["normal_impulse"][0, 1]), 0.0)
        self.assertEqual(float(alone["tangent_impulse"][0, 1]), 0.0)

    def test_loaded_contact_stiffness_gradient_matches_finite_difference(self):
        model = _model(stiffness=2_000.0, dt=0.002, iterations=200)
        pos, qd, M, qdd_free, _, _ = _loaded_equilibrium(
            model, load=3.0, mass=2.0
        )

        force = model._constraint_contact_solve(pos, qd, M, qdd_free)[
            "normal_force"
        ].sum()
        model.zero_grad(set_to_none=True)
        force.backward()
        analytic = float(model._contact_stiffness_log.grad[0])

        h = 1e-5
        parameter = model._contact_stiffness_log
        with th.no_grad():
            parameter.add_(h)
            up = float(
                model._constraint_contact_solve(pos, qd, M, qdd_free)[
                    "normal_force"
                ].sum()
            )
            parameter.add_(-2.0 * h)
            down = float(
                model._constraint_contact_solve(pos, qd, M, qdd_free)[
                    "normal_force"
                ].sum()
            )
            parameter.add_(h)
        numeric = (up - down) / (2.0 * h)

        self.assertTrue(math.isfinite(analytic))
        self.assertGreater(analytic, 0.0)
        self.assertAlmostEqual(
            analytic,
            numeric,
            delta=max(1e-7, 1e-5 * abs(numeric)),
        )


class TestVersionThreeSidecars(unittest.TestCase):
    def test_round_trip_restores_stiffness_response_dt_and_outputs(self):
        source = _model(
            stiffness=2_300.0, dt=0.0015, iterations=120, attenuation=0.35
        )
        with th.no_grad():
            source._contact_stiffness_log.fill_(math.log(3_456.0))
            source._kin_radius[0] -= 0.01

        target = _model(
            stiffness=17.0, dt=0.004, iterations=120, attenuation=0.1
        )
        target.load_state_dict(source.state_dict(), strict=True)

        self.assertEqual(int(source._contact_solver_version), 3)
        self.assertEqual(int(target._contact_solver_version), 3)
        self.assertTrue(
            th.equal(target._contact_stiffness_log, source._contact_stiffness_log)
        )
        self.assertEqual(target.contact_dt, source.contact_dt)
        self.assertTrue(
            th.equal(target._contact_response_dt, source._contact_response_dt)
        )
        self.assertEqual(target.contact_attenuation, source.contact_attenuation)
        self.assertTrue(
            th.equal(target._contact_attenuation, source._contact_attenuation)
        )
        self.assertTrue(th.equal(target._kin_radius, source._kin_radius))

        pos = th.tensor([[0.0005], [-0.001]], dtype=_DTYPE)
        qd = th.tensor([[0.2, -0.8], [0.0, 0.0]], dtype=_DTYPE)
        M = th.eye(2, dtype=_DTYPE).expand(2, -1, -1)
        qdd_free = th.tensor([[0.0, 0.0], [0.0, -2.0]], dtype=_DTYPE)
        with th.no_grad():
            before = source._constraint_contact_solve(pos, qd, M, qdd_free)
            after = target._constraint_contact_solve(pos, qd, M, qdd_free)
        for key in (
            "active_contact",
            "predicted_gap_free",
            "contact_stiffness",
            "normal_impulse",
            "tangent_impulse",
            "normal_force",
            "tangent_force",
            "generalized_force",
        ):
            self.assertTrue(th.equal(before[key], after[key]), key)

    def test_provisional_or_corrupt_fixed_recovery_sidecar_is_refused(self):
        model = _model()
        missing = model.state_dict().copy()
        del missing["_contact_attenuation"]
        with self.assertRaisesRegex(RuntimeError, "missing its fixed"):
            model.load_state_dict(missing, strict=True)

        corrupt = model.state_dict().copy()
        corrupt["_contact_attenuation"] = th.tensor(float("nan"))
        with self.assertRaisesRegex(RuntimeError, "must lie in"):
            model.load_state_dict(corrupt, strict=True)

    def test_version_two_and_three_sidecars_refuse_each_other(self):
        version_three = _model(stiffness=2_000.0)
        version_two = _model(stiffness=None, compliance=40.0)

        with self.assertRaises(RuntimeError) as caught:
            version_two.load_state_dict(version_three.state_dict(), strict=True)
        self.assertIn("contact law mismatch", str(caught.exception))
        self.assertIn("predicted-crossing physical-stiffness", str(caught.exception))

        with self.assertRaises(RuntimeError) as caught:
            version_three.load_state_dict(version_two.state_dict(), strict=True)
        self.assertIn("contact law mismatch", str(caught.exception))
        self.assertIn("gate-shaped-compliance", str(caught.exception))


class TestVersionThreeConfiguration(unittest.TestCase):
    def test_physical_stiffness_requires_metric_kinematics_and_no_gate(self):
        common = dict(
            obs_dim=3,
            action_dim=2,
            mode="structured",
            structured_hidden=(8,),
            contact_force=1,
            contact_solver="constraint",
            contact_stiffness=2_000.0,
            dof_layout=_LAYOUT,
        )
        with self.assertRaisesRegex(ValueError, "metric signed gap"):
            PortHamiltonianModel(contact_geometry="learned", **common)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            PortHamiltonianModel(
                contact_geometry="kinematic", contact_compliance=40.0, **common
            )
        with self.assertRaisesRegex(ValueError, "no force envelope"):
            PortHamiltonianModel(
                contact_geometry="kinematic", contact_gate_off=0.005, **common
            )
        for bad in (0.0, -0.1, 1.1, float("nan")):
            with self.subTest(attenuation=bad):
                with self.assertRaisesRegex(ValueError, "contact_attenuation"):
                    PortHamiltonianModel(
                        contact_geometry="kinematic",
                        contact_attenuation=bad,
                        **common,
                    )

        without_stiffness = dict(common)
        without_stiffness["contact_stiffness"] = None
        with self.assertRaisesRegex(ValueError, "only applies"):
            PortHamiltonianModel(
                contact_geometry="kinematic",
                contact_attenuation=0.2,
                **without_stiffness,
            )

if __name__ == "__main__":  # pragma: no cover
    unittest.main()
