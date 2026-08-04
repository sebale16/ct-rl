"""Exact contact kinematics must be exact, and must not disturb the learned path.

``PortHamiltonianModel(contact_geometry="kinematic")`` replaces the learned
``gap_net``/``tangent_net`` pair with closed-form planar forward kinematics read
out of ``DOFLayout.contact_points``.  Contact geometry over a known flat floor is
not a dynamics unknown, so there is nothing left to fit there and nothing left to
get wrong: the gap is exactly MuJoCo's capsule-endpoint signed distance and both
contact Jacobians are exact derivatives of it.

These tests pin four things.
  1. Truth: gaps and endpoint positions agree with MuJoCo forward kinematics, and
     the gap of a penetrating declared endpoint equals MuJoCo's ``contact.dist``.
  2. Consistency: the hand-written cumsum/reverse-cumsum Jacobians equal autograd,
     satisfy the planar cross-product identity, and are the exact time derivative
     of the gap.
  3. Containment: a ``contact_geometry="learned"`` model is bit-for-bit what it
     was before exact kinematics existed -- same parameters, same drift.
  4. Compatibility: a sidecar written before the geometry marker existed loads and
     comes back as learned geometry.
"""

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch as th

from models.port_hamiltonian import (
    CHEETAH_CONTACT_POINTS,
    DOFLayout,
    PlanarContactPoint,
    PortHamiltonianModel,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

try:  # dm_control/MuJoCo are optional for the pure-torch tests below.
    import mujoco
    from dm_control import suite

    _HAVE_MUJOCO = True
except Exception:  # pragma: no cover - exercised only on installs without MuJoCo
    _HAVE_MUJOCO = False


def _double_model(**kwargs):
    """A model whose every tensor -- including the geometry spec -- is float64.

    The kinematic spec is registered with ``th.tensor``/``th.zeros``, i.e. at the
    default dtype, so switching the default before construction is what gives the
    geometry buffers their full double precision.  Promoting a float32 model with
    ``.double()`` afterwards would only widen already-rounded constants.
    """
    previous = th.get_default_dtype()
    th.set_default_dtype(th.float64)
    try:
        return PortHamiltonianModel(**kwargs)
    finally:
        th.set_default_dtype(previous)


def _cheetah_kinematic_model(**kwargs):
    options = dict(
        obs_dim=17,
        action_dim=6,
        mode="structured",
        contact_force=len(CHEETAH_CONTACT_POINTS),
        contact_geometry="kinematic",
        dof_layout=DOFLayout.cheetah(),
        structured_hidden=(16, 16),
    )
    options.update(kwargs)
    return options


def _sample_cheetah_qpos(model, n, seed):
    """Broad qpos sampling: upright, upside down, airborne and penetrating."""
    rng = np.random.default_rng(seed)
    qpos = np.zeros((n, model.nq))
    qpos[:, 0] = rng.uniform(-2.0, 20.0, size=n)        # root x
    qpos[:, 1] = rng.uniform(-1.0, 2.0, size=n)         # root z
    qpos[:, 2] = rng.uniform(-np.pi, np.pi, size=n)     # root pitch
    for joint in range(3, model.njnt):
        adr = int(model.jnt_qposadr[joint])
        low, high = model.jnt_range[joint]
        if not model.jnt_limited[joint]:
            low, high = -np.pi, np.pi
        pad = 0.25 * (high - low)
        qpos[:, adr] = rng.uniform(low - pad, high + pad, size=n)
    return qpos


def _endpoint_specs(mj_model, points):
    """(geom id, end sign, capsule half length) for each declared contact point."""
    geom_id = {
        mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, i): i
        for i in range(mj_model.ngeom)
    }
    specs = []
    for point in points:
        geom, end = point.name.rsplit("_", 1)
        assert end in ("plus", "minus"), point.name
        gid = geom_id[geom]
        specs.append((gid, 1.0 if end == "plus" else -1.0, float(mj_model.geom_size[gid][1])))
    return specs


@unittest.skipUnless(_HAVE_MUJOCO, "dm_control/mujoco not available")
class KinematicGapMatchesMuJoCo(unittest.TestCase):
    """The closed form is MuJoCo's own kinematics, not an approximation of it."""

    TOL = 1e-9

    @classmethod
    def setUpClass(cls):
        env = suite.load("cheetah", "run")
        cls.mj_model = env.physics.model.ptr
        cls.mj_data = env.physics.data.ptr
        cls.layout = DOFLayout.cheetah()
        cls.model = _double_model(**_cheetah_kinematic_model())
        cls.specs = _endpoint_specs(cls.mj_model, cls.layout.contact_points)

    def _mujoco_endpoints(self, qpos):
        out = np.zeros((qpos.shape[0], len(self.specs), 2))
        for i in range(qpos.shape[0]):
            self.mj_data.qpos[:] = qpos[i]
            mujoco.mj_kinematics(self.mj_model, self.mj_data)
            for k, (gid, sign, half) in enumerate(self.specs):
                axis = self.mj_data.geom_xmat[gid].reshape(3, 3)[:, 2]
                point = self.mj_data.geom_xpos[gid] + sign * half * axis
                out[i, k, 0] = point[0]
                out[i, k, 1] = point[2]
        return out

    def test_gap_and_endpoint_positions_match_mujoco(self):
        qpos = _sample_cheetah_qpos(self.mj_model, 2000, seed=0)
        reference = self._mujoco_endpoints(qpos)
        pos = th.tensor(qpos[:, 1:9])
        gap, _, _, _, _ = self.model._contact_geometry(pos, th.zeros(pos.shape[0], 9, dtype=pos.dtype))
        z, x_relative, _, _, _ = self.model._kinematic_point_kinematics(pos)
        radius = np.array([p.radius for p in self.layout.contact_points])

        height_error = np.abs(z.detach().numpy() - reference[:, :, 1]).max()
        gap_error = np.abs(
            gap.detach().numpy() + radius - reference[:, :, 1]
        ).max()
        # Root x is cyclic and absent from the observation, so the model gives the
        # endpoint offset from the root; adding qpos[0] must recover world x.
        x_error = np.abs(
            x_relative.detach().numpy() + qpos[:, 0:1] - reference[:, :, 0]
        ).max()
        self.assertLess(height_error, self.TOL)
        self.assertLess(gap_error, self.TOL)
        self.assertLess(x_error, self.TOL)

    def test_penetrating_gap_equals_mujoco_contact_distance(self):
        """A negative declared gap is a real MuJoCo contact of the same depth."""
        qpos = _sample_cheetah_qpos(self.mj_model, 400, seed=3)
        # Keep the pose plausible so MuJoCo's narrow phase produces endpoint
        # contacts rather than the pathological deep interpenetrations of the
        # broad sampler.
        qpos[:, 1] = np.clip(qpos[:, 1], -0.35, 0.6)
        qpos[:, 2] = np.clip(qpos[:, 2], -1.2, 1.2)
        pos = th.tensor(qpos[:, 1:9])
        gap, _, _, _, _ = self.model._contact_geometry(pos, th.zeros(pos.shape[0], 9, dtype=pos.dtype))
        gap = gap.detach().numpy()
        geoms = [spec[0] for spec in self.specs]

        checked, worst = 0, 0.0
        for i in range(qpos.shape[0]):
            self.mj_data.qpos[:] = qpos[i]
            self.mj_data.qvel[:] = 0.0
            mujoco.mj_forward(self.mj_model, self.mj_data)
            floor_contacts = []
            for c in range(self.mj_data.ncon):
                contact = self.mj_data.contact[c]
                if contact.geom1 != 0 and contact.geom2 != 0:
                    continue
                other = contact.geom2 if contact.geom1 == 0 else contact.geom1
                floor_contacts.append((int(other), float(contact.dist)))
            for k in range(gap.shape[1]):
                if gap[i, k] >= 0.0:
                    continue
                distances = [
                    abs(dist - gap[i, k])
                    for geom, dist in floor_contacts
                    if geom == geoms[k]
                ]
                self.assertTrue(
                    distances,
                    f"declared endpoint {self.layout.contact_points[k].name} has "
                    f"gap {gap[i, k]:.6f} but MuJoCo reports no floor contact",
                )
                worst = max(worst, min(distances))
                checked += 1
        self.assertGreater(checked, 100, "sampler produced too few penetrations")
        self.assertLess(worst, self.TOL)


class KinematicJacobians(unittest.TestCase):
    """The cumsum Jacobians are exact derivatives, in config-DOF space."""

    def setUp(self):
        self.layout = DOFLayout.cheetah()
        self.model = _double_model(**_cheetah_kinematic_model())
        rng = np.random.default_rng(11)
        self.pos = th.tensor(rng.uniform(-2.0, 2.0, size=(24, 8)))
        self.qd = th.tensor(rng.uniform(-3.0, 3.0, size=(24, 9)))

    def test_normal_and_tangent_jacobians_match_autograd(self):
        model, pos = self.model, self.pos
        pos_to_cfg = model._pos_to_cfg

        def gap_of(sample):
            z, _, _, _, _ = model._kinematic_point_kinematics(sample.unsqueeze(0))
            return (z - model._kin_radius).squeeze(0)

        def offset_of(sample):
            _, x, _, _, _ = model._kinematic_point_kinematics(sample.unsqueeze(0))
            return x.squeeze(0)

        _, _, _, J_n, J_t = model._contact_geometry(pos, self.qd)
        for i in range(pos.shape[0]):
            dg = th.autograd.functional.jacobian(gap_of, pos[i])
            dh = th.autograd.functional.jacobian(offset_of, pos[i])
            self.assertLess(float((J_n[i][:, pos_to_cfg] - dg).abs().max()), 1e-12)
            self.assertLess(float((J_t[i][:, pos_to_cfg] - dh).abs().max()), 1e-12)
            # Config DOFs with no observed position (the cyclic root x) carry the
            # friction direction only: no normal component, unit tangent.
            self.assertEqual(float(J_n[i][:, 0].abs().max()), 0.0)
            self.assertTrue(th.all(J_t[i][:, 0] == 1.0))

    def test_tangent_onehot_is_not_double_counted(self):
        """dx/d(root x) is exactly one, not two."""
        _, _, _, _, J_t = self.model._contact_geometry(self.pos, self.qd)
        tangent_cfg = self.layout.contact_tangent_cfg
        self.assertTrue(th.all(J_t[:, :, tangent_cfg] == 1.0))

    def test_jacobian_cross_product_identity(self):
        """dz/dtheta_j == -(x_point - x_j) and dx/dtheta_j == +(z_point - z_j)."""
        model, pos = self.model, self.pos
        z, x_relative, cx, cz, _ = model._kinematic_point_kinematics(pos)
        _, _, _, J_n, J_t = model._contact_geometry(pos, self.qd)
        for k, point in enumerate(self.layout.contact_points):
            # Position of the joint that angle j rotates about: the running sum of
            # segment displacements strictly before segment j.
            x_joint = th.cumsum(cx[:, k, :], dim=1) - cx[:, k, :]
            z_joint = th.cumsum(cz[:, k, :], dim=1) - cz[:, k, :]
            z_joint = z_joint + model._kin_base_height[k] + pos[:, point.height_pos:point.height_pos + 1]
            for j, angle in enumerate(point.angle_pos):
                cfg = self.layout.obs_pos_to_cfg[angle]
                expected_normal = -(x_relative[:, k] - x_joint[:, j])
                expected_tangent = z[:, k] - z_joint[:, j]
                self.assertLess(
                    float((J_n[:, k, cfg] - expected_normal).abs().max()), 1e-12
                )
                self.assertLess(
                    float((J_t[:, k, cfg] - expected_tangent).abs().max()), 1e-12
                )

    def test_gap_rate_is_the_time_derivative_of_the_gap(self):
        """gdot from J_n qd matches a central difference of g along qd."""
        model = self.model
        pos, qd = self.pos, self.qd
        # Only the observed positions move; the cyclic root x does not enter g.
        pos_rate = qd[:, model._pos_to_cfg]
        gap, gdot, v_t, _, _ = model._contact_geometry(pos, qd)
        eps = 1e-6
        forward, _, _, _, _ = model._contact_geometry(pos + eps * pos_rate, qd)
        backward, _, _, _, _ = model._contact_geometry(pos - eps * pos_rate, qd)
        numeric = (forward - backward) / (2.0 * eps)
        self.assertLess(float((gdot - numeric).abs().max()), 1e-6)
        # The tangential velocity is the root-x rate plus the offset rate.
        _, x_forward, _, _, _ = model._kinematic_point_kinematics(pos + eps * pos_rate)
        _, x_backward, _, _, _ = model._kinematic_point_kinematics(pos - eps * pos_rate)
        numeric_tangent = (x_forward - x_backward) / (2.0 * eps) + qd[:, 0:1]
        self.assertLess(float((v_t - numeric_tangent).abs().max()), 1e-6)


class KinematicShapesDtypeAndDevice(unittest.TestCase):
    def test_shapes_and_float32_propagation(self):
        model = PortHamiltonianModel(**_cheetah_kinematic_model())
        K = len(CHEETAH_CONTACT_POINTS)
        pos = th.randn(5, 8)
        qd = th.randn(5, 9)
        g, gdot, v_t, J_n, J_t = model._contact_geometry(pos, qd)
        self.assertEqual(tuple(g.shape), (5, K))
        self.assertEqual(tuple(gdot.shape), (5, K))
        self.assertEqual(tuple(v_t.shape), (5, K))
        self.assertEqual(tuple(J_n.shape), (5, K, 9))
        self.assertEqual(tuple(J_t.shape), (5, K, 9))
        for tensor in (g, gdot, v_t, J_n, J_t):
            self.assertEqual(tensor.dtype, th.float32)
            self.assertEqual(tensor.device, pos.device)

    def test_float64_propagation(self):
        model = _double_model(**_cheetah_kinematic_model())
        pos = th.randn(4, 8, dtype=th.float64)
        qd = th.randn(4, 9, dtype=th.float64)
        for tensor in model._contact_geometry(pos, qd):
            self.assertEqual(tensor.dtype, th.float64)
        self.assertEqual(model._kin_dx.dtype, th.float64)
        # Index buffers stay integral under a dtype change.
        self.assertEqual(model._kin_angle_pos.dtype, th.int64)
        self.assertEqual(model._kin_angle_cfg.dtype, th.int64)

    def test_spec_buffers_are_not_parameters_and_not_persistent(self):
        model = PortHamiltonianModel(**_cheetah_kinematic_model())
        self.assertFalse(hasattr(model, "gap_net"))
        self.assertFalse(hasattr(model, "tangent_net"))
        names = {name for name, _ in model.named_parameters()}
        self.assertFalse(any(name.startswith("_kin_") for name in names))
        state = model.state_dict()
        self.assertFalse(any(key.startswith("_kin_") for key in state))
        # The constitutive parameters stay learned; only the geometry is pinned.
        self.assertIn("_contact_raw", state)
        self.assertTrue(model._contact_raw.requires_grad)

    def test_device_follows_the_module(self):
        model = PortHamiltonianModel(**_cheetah_kinematic_model())
        model = model.to("cpu")
        self.assertEqual(model._kin_dx.device, th.device("cpu"))
        self.assertEqual(model._kin_angle_cfg.device, th.device("cpu"))

    def test_kinematic_drift_and_diagnostics_run(self):
        model = PortHamiltonianModel(**_cheetah_kinematic_model())
        obs = th.randn(6, 17) * 0.2
        action = th.randn(6, 6) * 0.2
        drift = model.drift(obs, action)
        self.assertEqual(tuple(drift.shape), (6, 17))
        self.assertTrue(bool(th.all(th.isfinite(drift))))
        diagnostics = model.contact_diagnostics(obs, action)
        self.assertEqual(
            tuple(diagnostics["gap"].shape), (6, len(CHEETAH_CONTACT_POINTS))
        )

    def test_constitutive_parameters_stay_trainable(self):
        """Geometry is pinned; restitution/Baumgarte/friction still get gradient."""
        model = PortHamiltonianModel(**_cheetah_kinematic_model())
        obs = th.zeros(4, 17)
        obs[:, 0] = -0.6          # drop the root so the feet are through the floor
        action = th.randn(4, 6) * 0.1
        model.drift(obs, action).square().sum().backward()
        self.assertIsNotNone(model._contact_raw.grad)
        self.assertGreater(float(model._contact_raw.grad.abs().max()), 0.0)


class KinematicConfigurationErrors(unittest.TestCase):
    def test_unknown_geometry_rejected(self):
        with self.assertRaises(ValueError) as caught:
            PortHamiltonianModel(**_cheetah_kinematic_model(contact_geometry="exact"))
        self.assertIn("contact_geometry", str(caught.exception))

    def test_contact_force_must_match_declared_points(self):
        for wrong in (4, 8):
            with self.assertRaises(ValueError) as caught:
                PortHamiltonianModel(**_cheetah_kinematic_model(contact_force=wrong))
            message = str(caught.exception)
            self.assertIn("contact_points", message)
            self.assertIn(str(len(CHEETAH_CONTACT_POINTS)), message)

    def test_layout_without_contact_points_rejected(self):
        """A layout that declares no geometry cannot support exact kinematics."""
        layout = DOFLayout(
            obs_dim=6, pos_slice=(0, 3), vel_slice=(3, 6), cyclic_cfg=(),
            obs_pos_to_cfg=(0, 1, 2), contact_tangent_cfg=0,
        )
        self.assertEqual(layout.contact_points, ())
        with self.assertRaises(ValueError) as caught:
            PortHamiltonianModel(
                obs_dim=6,
                action_dim=1,
                mode="structured",
                contact_force=2,
                contact_geometry="kinematic",
                dof_layout=layout,
            )
        self.assertIn("contact_points", str(caught.exception))

    def test_tangent_dof_must_be_cyclic(self):
        """An observed tangent DOF would be written by the chain AND the onehot."""
        point = PlanarContactPoint(
            name="toe", radius=0.046, height_pos=0, base_height=0.7,
            angle_pos=(1,), offsets=((0.1, -0.1),),
        )
        layout = DOFLayout(
            obs_dim=6, pos_slice=(0, 3), vel_slice=(3, 6), cyclic_cfg=(),
            obs_pos_to_cfg=(0, 1, 2), contact_tangent_cfg=2,
            contact_points=(point,),
        )
        with self.assertRaises(ValueError) as caught:
            PortHamiltonianModel(
                obs_dim=6, action_dim=1, mode="structured", contact_force=1,
                contact_geometry="kinematic", dof_layout=layout,
                structured_hidden=(8,),
            )
        self.assertIn("cyclic", str(caught.exception))

    def test_offsets_must_match_chain_length(self):
        with self.assertRaises(AssertionError):
            PlanarContactPoint(
                name="bad",
                radius=0.046,
                height_pos=0,
                base_height=0.7,
                angle_pos=(1, 2),
                offsets=((0.0, 0.0),),
            )

    def test_radius_must_be_positive(self):
        with self.assertRaises(AssertionError):
            PlanarContactPoint(
                name="bad",
                radius=0.0,
                height_pos=0,
                base_height=0.7,
                angle_pos=(1,),
                offsets=((0.0, 0.0),),
            )

    def test_angle_index_must_be_observable(self):
        point = PlanarContactPoint(
            name="offgrid",
            radius=0.046,
            height_pos=0,
            base_height=0.7,
            angle_pos=(99,),
            offsets=((0.0, 0.0),),
        )
        with self.assertRaises(AssertionError):
            DOFLayout.cheetah().__class__(
                obs_dim=17,
                pos_slice=(0, 8),
                vel_slice=(8, 17),
                cyclic_cfg=(0,),
                obs_pos_to_cfg=tuple(range(1, 9)),
                contact_tangent_cfg=0,
                contact_points=(point,),
            )

    def test_chains_must_be_prefix_consistent(self):
        first = PlanarContactPoint(
            name="a", radius=0.046, height_pos=0, base_height=0.7,
            angle_pos=(1, 2, 3), offsets=((0.0, 0.0),) * 3,
        )
        # Same distal angle 3, but reached through a different parent.
        second = PlanarContactPoint(
            name="b", radius=0.046, height_pos=0, base_height=0.7,
            angle_pos=(1, 5, 3), offsets=((0.0, 0.0),) * 3,
        )
        with self.assertRaises(AssertionError) as caught:
            DOFLayout(
                obs_dim=17,
                pos_slice=(0, 8),
                vel_slice=(8, 17),
                cyclic_cfg=(0,),
                obs_pos_to_cfg=tuple(range(1, 9)),
                contact_tangent_cfg=0,
                contact_points=(first, second),
            )
        self.assertIn("prefix-consistent", str(caught.exception))

    def test_shipped_cheetah_layout_validates(self):
        layout = DOFLayout.cheetah()
        self.assertEqual(len(layout.contact_points), 6)
        self.assertEqual(
            [p.name for p in layout.contact_points],
            [
                "bfoot_minus", "bfoot_plus", "ffoot_minus",
                "ffoot_plus", "bshin_plus", "fshin_minus",
            ],
        )
        for point in layout.contact_points:
            self.assertEqual(point.height_pos, 0)
            self.assertEqual(point.base_height, 0.7)
            self.assertEqual(point.radius, 0.046)
            self.assertEqual(point.angle_pos[0], 1)   # every chain starts at rooty

    def test_other_layouts_declare_no_contact_points(self):
        self.assertEqual(DOFLayout.raw_state(3).contact_points, ())
        self.assertEqual(DOFLayout.cartpole().contact_points, ())
        self.assertEqual(DOFLayout.acrobot().contact_points, ())


# The last commit before exact contact kinematics existed. Pinned explicitly
# rather than tracking HEAD: once this branch is committed, HEAD stops being a
# pre-kinematic reference and the comparison below would quietly turn into a
# skip -- disarming the bit-exactness guard exactly when it starts mattering.
PRE_KINEMATIC_COMMIT = "761705e"


def _head_module():
    """Import ``models/port_hamiltonian.py`` as of :data:`PRE_KINEMATIC_COMMIT`.

    Returns ``None`` only when the file genuinely cannot be recovered (no git,
    shallow clone), and asserts that what came back really is pre-kinematic so a
    mis-pinned commit fails loudly instead of comparing new code with itself.
    """
    try:
        source = subprocess.run(
            ["git", "show", f"{PRE_KINEMATIC_COMMIT}:models/port_hamiltonian.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
        ).stdout.decode()
    except Exception:
        return None
    assert "contact_geometry: str" not in source, (
        f"{PRE_KINEMATIC_COMMIT} already contains the kinematic geometry change; "
        "re-pin PRE_KINEMATIC_COMMIT to a commit that predates it"
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "head_port_hamiltonian.py"
        path.write_text(source)
        spec = importlib.util.spec_from_file_location("head_port_hamiltonian", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["head_port_hamiltonian"] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop("head_port_hamiltonian", None)
    return module


class LearnedGeometryIsBitExact(unittest.TestCase):
    """The default path must be byte-identical to the pre-kinematic model."""

    CASES = [
        dict(contact_solver="constraint", analytic_derivatives=True),
        dict(contact_solver="constraint", analytic_derivatives=False),
        dict(contact_solver="compliant", analytic_derivatives=True),
        dict(contact_solver="compliant", analytic_derivatives=False),
    ]

    @staticmethod
    def _build(module, **kwargs):
        th.manual_seed(1234)
        return module.PortHamiltonianModel(
            17, 6, mode="structured", contact_force=4,
            structured_hidden=(16, 16), **kwargs
        )

    @staticmethod
    def _inputs(batch=7):
        obs = th.arange(batch * 17, dtype=th.float32).reshape(batch, 17) * 0.017 - 0.5
        action = th.arange(batch * 6, dtype=th.float32).reshape(batch, 6) * 0.03 - 0.2
        return obs, action

    def test_default_is_learned(self):
        model = PortHamiltonianModel(
            17, 6, mode="structured", contact_force=4, structured_hidden=(8,)
        )
        self.assertEqual(model.contact_geometry, "learned")
        self.assertTrue(hasattr(model, "gap_net"))
        self.assertTrue(hasattr(model, "tangent_net"))
        self.assertEqual(int(model._contact_geometry_version), 0)

    def test_drift_and_weights_match_git_head(self):
        head = _head_module()
        if head is None:
            self.skipTest(
                "cannot recover models/port_hamiltonian.py at "
                f"{PRE_KINEMATIC_COMMIT} to compare with"
            )
        obs, action = self._inputs()
        for case in self.CASES:
            with self.subTest(**case):
                reference = self._build(head, **case)
                current = self._build(sys.modules["models.port_hamiltonian"], **case)
                reference_state = reference.state_dict()
                current_state = current.state_dict()
                self.assertEqual(
                    set(current_state) - set(reference_state),
                    {"_contact_geometry_version"},
                    "the learned path may add only the geometry marker",
                )
                self.assertEqual(set(reference_state) - set(current_state), set())
                for key, value in reference_state.items():
                    self.assertTrue(
                        th.equal(value, current_state[key]),
                        f"parameter {key} differs from HEAD",
                    )
                self.assertTrue(
                    th.equal(reference.drift(obs, action), current.drift(obs, action)),
                    "learned-geometry drift differs from HEAD",
                )

    def test_learned_geometry_ignores_declared_contact_points(self):
        """Adding contact_points to the cheetah layout must not change learned models."""
        th.manual_seed(7)
        with_points = PortHamiltonianModel(
            17, 6, mode="structured", contact_force=4,
            dof_layout=DOFLayout.cheetah(), structured_hidden=(16, 16),
        )
        bare = DOFLayout(
            obs_dim=17, pos_slice=(0, 8), vel_slice=(8, 17), cyclic_cfg=(0,),
            obs_pos_to_cfg=tuple(range(1, 9)), m_invariant_pos=(0,),
            contact_tangent_cfg=0,
        )
        th.manual_seed(7)
        without_points = PortHamiltonianModel(
            17, 6, mode="structured", contact_force=4,
            dof_layout=bare, structured_hidden=(16, 16),
        )
        obs, action = self._inputs()
        self.assertTrue(
            th.equal(with_points.drift(obs, action), without_points.drift(obs, action))
        )


def _learned_cheetah_state(seed):
    """A learned-geometry cheetah sidecar and the same thing without the marker."""
    th.manual_seed(seed)
    source = PortHamiltonianModel(
        **_cheetah_kinematic_model(contact_geometry="learned", contact_force=6)
    )
    state = source.state_dict()
    legacy = {k: v for k, v in state.items() if k != "_contact_geometry_version"}
    return source, state, legacy


class GeometryVersionMarkerRoundtrip(unittest.TestCase):
    """The marker restores a sidecar's geometry or refuses; it never switches.

    Unlike the force-law marker, the geometry decides which PARAMETERS exist:
    ``gap_net``/``tangent_net`` are present for the learned geometry and absent
    for the exact one.  Rebuilding them under an optimizer that was constructed
    earlier (CT-SAC builds ``dynamics_optimizer`` in ``__init__`` and loads the
    sidecar afterwards) would leave the optimizer tracking the wrong tensors, so
    a mismatch is a hard error rather than a silent flip.
    """

    def test_matching_geometry_loads(self):
        for geometry in ("learned", "kinematic"):
            with self.subTest(geometry=geometry):
                th.manual_seed(6)
                source = PortHamiltonianModel(
                    **_cheetah_kinematic_model(
                        contact_geometry=geometry, contact_force=6
                    )
                )
                state = source.state_dict()
                self.assertEqual(
                    int(state["_contact_geometry_version"]),
                    1 if geometry == "kinematic" else 0,
                )
                th.manual_seed(99)
                target = PortHamiltonianModel(
                    **_cheetah_kinematic_model(
                        contact_geometry=geometry, contact_force=6
                    )
                )
                result = target.load_state_dict(state, strict=True)
                self.assertEqual(list(result.missing_keys), [])
                self.assertEqual(list(result.unexpected_keys), [])
                self.assertEqual(target.contact_geometry, geometry)
                obs, action = th.randn(3, 17) * 0.2, th.randn(3, 6) * 0.2
                self.assertTrue(
                    th.equal(source.drift(obs, action), target.drift(obs, action))
                )

    def test_legacy_sidecar_without_marker_loads_into_a_learned_model(self):
        """A sidecar written before the marker existed is unambiguously learned."""
        source, _, legacy = _learned_cheetah_state(5)
        th.manual_seed(77)
        target = PortHamiltonianModel(
            **_cheetah_kinematic_model(contact_geometry="learned", contact_force=6)
        )
        result = target.load_state_dict(legacy, strict=True)
        self.assertEqual(list(result.missing_keys), [])
        self.assertEqual(list(result.unexpected_keys), [])
        self.assertEqual(target.contact_geometry, "learned")
        self.assertEqual(int(target._contact_geometry_version), 0)
        obs, action = th.randn(3, 17) * 0.2, th.randn(3, 6) * 0.2
        self.assertTrue(
            th.equal(source.drift(obs, action), target.drift(obs, action))
        )

    def test_legacy_sidecar_into_a_kinematic_model_raises(self):
        """Regression: this used to flip the model to learned geometry silently.

        The silent flip undid the requested configuration AND desynchronized any
        optimizer already built over ``model.parameters()`` -- the rebuilt
        ``gap_net``/``tangent_net`` were 12 fresh tensors the optimizer had never
        seen, so the geometry the flip installed could never train.
        """
        _, _, legacy = _learned_cheetah_state(5)
        target = PortHamiltonianModel(**_cheetah_kinematic_model())
        optimizer = th.optim.Adam(target.parameters())
        tracked = {
            id(p) for group in optimizer.param_groups for p in group["params"]
        }
        with self.assertRaises(RuntimeError) as caught:
            target.load_state_dict(legacy, strict=True)
        message = str(caught.exception)
        self.assertIn("contact geometry mismatch", message)
        self.assertIn("kinematic", message)
        self.assertIn("learned", message)
        # The refusal must leave the model exactly as constructed, and the
        # optimizer must still cover every parameter it has.
        self.assertEqual(target.contact_geometry, "kinematic")
        self.assertFalse(hasattr(target, "gap_net"))
        self.assertTrue(hasattr(target, "_kin_dx"))
        self.assertEqual(
            [n for n, p in target.named_parameters() if id(p) not in tracked], []
        )

    def test_kinematic_sidecar_into_a_learned_model_raises(self):
        th.manual_seed(6)
        source = PortHamiltonianModel(**_cheetah_kinematic_model())
        target = PortHamiltonianModel(
            **_cheetah_kinematic_model(contact_geometry="learned", contact_force=6)
        )
        optimizer = th.optim.Adam(target.parameters())
        tracked = {
            id(p) for group in optimizer.param_groups for p in group["params"]
        }
        with self.assertRaises(RuntimeError) as caught:
            target.load_state_dict(source.state_dict(), strict=True)
        self.assertIn("contact geometry mismatch", str(caught.exception))
        self.assertEqual(target.contact_geometry, "learned")
        self.assertTrue(hasattr(target, "gap_net"))
        # No orphans either: every optimizer entry is still a live parameter.
        live = {id(p) for _, p in target.named_parameters()}
        self.assertTrue(tracked <= live)

    def test_no_switching_helper_exists(self):
        """There must be no in-place geometry switch for a caller to trip over."""
        self.assertFalse(hasattr(PortHamiltonianModel, "_set_contact_geometry"))

    def test_unsupported_marker_raises(self):
        model = PortHamiltonianModel(**_cheetah_kinematic_model())
        state = model.state_dict()
        state["_contact_geometry_version"] = th.tensor(7, dtype=th.int64)
        with self.assertRaises(RuntimeError) as caught:
            model.load_state_dict(state, strict=True)
        self.assertIn("contact geometry version", str(caught.exception))

    def test_geometry_and_solver_markers_are_orthogonal(self):
        """The solver marker still flips freely; only the geometry must match."""
        th.manual_seed(9)
        source = PortHamiltonianModel(
            **_cheetah_kinematic_model(contact_geometry="learned", contact_force=6,
                                       contact_solver="compliant")
        )
        state = source.state_dict()
        self.assertEqual(int(state["_contact_solver_version"]), 0)
        self.assertEqual(int(state["_contact_geometry_version"]), 0)
        target = PortHamiltonianModel(
            **_cheetah_kinematic_model(
                contact_solver="constraint", contact_geometry="learned",
                contact_force=6,
            )
        )
        target.load_state_dict(state, strict=True)
        self.assertEqual(target.contact_solver, "compliant")
        self.assertEqual(target.contact_geometry, "learned")


class ContactGateBandIsMetric(unittest.TestCase):
    """The gate band is a physical clearance once the gap is exact metres.

    The constraint solve is near-rigid wherever the gate is appreciably nonzero:
    ``J_solver = gate*J`` and ``M_inv_Jt = gate*M^-1 J^T`` give ``W = gate^2
    W_full`` and ``bias = gate*bias_phys``, so the physical impulse is
    ``-(W_full + reg*scale/gate^2 I)^-1 bias_phys`` -- the gate cancels except
    through the ungated regularization.  A gate band of 6 cm therefore supports
    the full body weight at 2.6 cm of clearance and no learnable parameter can
    remove it, because ``e``/``beta``/``mu`` do not scale the normal impulse.
    """

    def _kinematic(self, **kwargs):
        return PortHamiltonianModel(**_cheetah_kinematic_model(**kwargs))

    def _pos_at_min_gap(self, model, min_gap):
        """Zero joint angles, root height chosen to put the lowest point there."""
        pos = th.zeros(1, 8)
        gap, *_ = model._contact_geometry(pos, th.zeros(1, 9))
        pos[0, 0] = min_gap - float(gap.min())
        return pos

    def _static_normal_force(self, model, min_gap):
        """Normal force holding a unit-inertia body under gravity at ``min_gap``."""
        pos = self._pos_at_min_gap(model, min_gap)
        qd = th.zeros(1, 9)
        mass = th.eye(9).unsqueeze(0)
        qdd_free = th.zeros(1, 9)
        qdd_free[0, model.layout.obs_pos_to_cfg[0]] = -9.81
        with th.no_grad():
            out = model._constraint_contact_solve(pos, qd, mass, qdd_free)
        return float(out["normal_force"][0].sum()), out

    def test_default_band_is_metric_for_kinematic_geometry(self):
        self.assertEqual(self._kinematic()._contact_gate_off, 0.005)

    def test_default_band_is_unchanged_for_learned_geometry(self):
        learned = PortHamiltonianModel(
            17, 6, mode="structured", contact_force=4, structured_hidden=(8,)
        )
        self.assertEqual(learned._contact_gate_off, 0.06)
        self.assertEqual(
            PortHamiltonianModel._contact_gate_off, 0.06, "class default moved"
        )

    def test_a_centimetre_of_clearance_carries_no_load(self):
        """Regression: at 2.6 cm of clearance the old band carried a body weight."""
        model = self._kinematic()
        for clearance in (0.01, 0.026, 0.05):
            with self.subTest(clearance=clearance):
                force, out = self._static_normal_force(model, clearance)
                self.assertEqual(float(out["gate"].max()), 0.0)
                self.assertEqual(force, 0.0)

    def test_the_band_still_engages_at_contact(self):
        model = self._kinematic()
        for clearance in (0.001, 0.0, -0.001):
            with self.subTest(clearance=clearance):
                force, out = self._static_normal_force(model, clearance)
                self.assertGreater(float(out["gate"].max()), 0.0)
                self.assertGreater(force, 0.0)

    def test_band_is_a_constructor_argument(self):
        wide = self._kinematic(contact_gate_off=0.06)
        self.assertEqual(wide._contact_gate_off, 0.06)
        self.assertGreater(self._static_normal_force(wide, 0.026)[0], 0.0)
        for bad in (0.0, -1.0, float("nan")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self._kinematic(contact_gate_off=bad)

    def test_constitutive_parameters_cannot_undo_the_band(self):
        """Sweeping e/beta/mu leaves the load-at-clearance essentially untouched."""
        model = self._kinematic(contact_gate_off=0.06)
        forces = []
        for raw in (-6.0, 0.0, 6.0):
            with th.no_grad():
                model._contact_raw.fill_(raw)
            forces.append(self._static_normal_force(model, 0.026)[0])
        self.assertTrue(all(f > 0.5 * forces[0] for f in forces), forces)


@unittest.skipUnless(_HAVE_MUJOCO, "dm_control/mujoco not available")
class DroppedCheetahRestsOnTheFloor(unittest.TestCase):
    """End-to-end: only the geometry and the solve are under test.

    MuJoCo supplies the mass matrix and the contact-free acceleration, so a
    resting height that disagrees with the floor can only come from the contact
    port.  The mechanics head is random and irrelevant here -- it is never
    called.
    """

    STEPS = 2000
    DT = 0.002

    @classmethod
    def setUpClass(cls):
        env = suite.load("cheetah", "run")
        cls.physics = env.physics
        cls.mj_model = env.physics.model
        cls.mj_data = env.physics.data
        cls.nv = int(cls.mj_model.nv)

    def _mujoco_terms(self, qpos, qvel):
        """MuJoCo's own M(q) and the contact-free acceleration at zero action."""
        self.mj_data.qpos[:] = qpos
        self.mj_data.qvel[:] = qvel
        self.mj_data.ctrl[:] = 0.0
        self.physics.forward()
        mass = np.zeros((self.nv, self.nv))
        mujoco.mj_fullM(self.mj_model.ptr, mass, self.mj_data.qM)
        force = (
            self.mj_data.qfrc_actuator[: self.nv]
            + self.mj_data.qfrc_passive[: self.nv]
            - self.mj_data.qfrc_bias[: self.nv]
        )
        return mass, np.linalg.solve(mass, force)

    def _drop(self, model, z0=0.6):
        qpos = np.zeros(int(self.mj_model.nq))
        qpos[1] = z0
        qvel = np.zeros(self.nv)
        out = None
        for _ in range(self.STEPS):
            mass, free = self._mujoco_terms(qpos, qvel)
            with th.no_grad():
                out = model._constraint_contact_solve(
                    th.tensor(qpos[1:9])[None],
                    th.tensor(qvel[:9])[None],
                    th.tensor(mass)[None],
                    th.tensor(free)[None],
                )
            qvel = qvel + self.DT * (free + out["contact_acceleration"][0].numpy())
            qpos[:9] = qpos[:9] + self.DT * qvel
            self.assertTrue(np.all(np.isfinite(qpos)), "drop diverged")
        return qpos, qvel, out

    @staticmethod
    def _model(**kwargs):
        previous = th.get_default_dtype()
        th.set_default_dtype(th.float64)
        try:
            th.manual_seed(0)
            return PortHamiltonianModel(
                **_cheetah_kinematic_model(structured_hidden=(32, 32), **kwargs)
            )
        finally:
            th.set_default_dtype(previous)

    def test_rest_gap_is_millimetric(self):
        """Regression: the 6 cm band rested the cheetah 2.6 cm above the floor."""
        model = self._model()
        qpos, qvel, out = self._drop(model)
        gap = out["gap"][0].numpy()
        self.assertLess(np.abs(qvel).max(), 1e-2, "did not reach a static rest")
        # Weight-bearing, but resting essentially on the plane rather than above it.
        self.assertGreater(float(out["normal_force"][0].sum()), 100.0)
        self.assertLess(gap.min(), 2e-3, f"levitating at gaps {gap}")
        self.assertGreater(gap.min(), -2e-3, f"sinking at gaps {gap}")

    def test_the_historical_band_levitates(self):
        """The control: the same harness with the old band still floats.

        Keeps the regression honest -- if the drop harness ever stopped being
        able to detect levitation, this would stop failing too.
        """
        model = self._model(contact_gate_off=0.06)
        _, _, out = self._drop(model)
        gap = out["gap"][0].numpy()
        self.assertGreater(gap.min(), 0.02, f"expected levitation, got {gap}")
        self.assertGreater(float(out["normal_force"][0].sum()), 100.0)


class CompliantLawRejectsMetricGaps(unittest.TestCase):
    """The legacy penalty law is initialized for a gap of arbitrary scale.

    ``softplus(0.5413) ~ 1`` N/m against a gap in metres is four orders of
    magnitude too soft to carry a body weight, and a dropped cheetah integrated
    with MuJoCo's own mass matrix falls 13.5 m through the floor in 3 s.  The
    combination is refused rather than started from a state that cannot
    integrate.
    """

    def test_compliant_plus_kinematic_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            PortHamiltonianModel(
                **_cheetah_kinematic_model(contact_solver="compliant")
            )
        message = str(caught.exception)
        self.assertIn("kinematic", message)
        self.assertIn("compliant", message)

    def test_the_two_supported_combinations_still_build(self):
        PortHamiltonianModel(**_cheetah_kinematic_model(contact_solver="constraint"))
        PortHamiltonianModel(
            **_cheetah_kinematic_model(
                contact_geometry="learned", contact_force=6,
                contact_solver="compliant",
            )
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
