"""The analytic mechanics derivatives must equal the functorch reference.

``PortHamiltonianModel(analytic_derivatives=True)`` replaces
``vmap(jacfwd(...))``/``vmap(grad(...))`` over the mass, potential and contact
nets with explicit batched JVP/VJP passes. That is a pure implementation change:
every quantity the drift consumes, and every parameter gradient taken through
it, must match the reference path to float32 roundoff. These tests pin that for
each shipped layout, with and without the contact port, on values *and*
gradients -- the dynamics fit trains through these derivatives, so a correct
forward with a wrong backward would silently corrupt the learned model.
"""

import unittest

import torch as th

from models.port_hamiltonian import DOFLayout, PortHamiltonianModel

# float32 accumulation over a 128-wide MLP: relative agreement lands at ~1e-6,
# so compare with a scale-aware tolerance rather than a bare atol.
RTOL = 2e-5
ATOL = 2e-6


def _paired_models(**kwargs):
    """Two models with identical weights, one analytic and one functorch."""
    th.manual_seed(0)
    analytic = PortHamiltonianModel(analytic_derivatives=True, **kwargs)
    th.manual_seed(0)
    reference = PortHamiltonianModel(analytic_derivatives=False, **kwargs)
    reference.load_state_dict(analytic.state_dict())
    return analytic, reference


CASES = {
    "cheetah_contact": dict(
        obs_dim=17, action_dim=6, mode="structured", contact_force=4,
        dof_layout=DOFLayout.cheetah(),
    ),
    "cheetah_no_contact": dict(
        obs_dim=17, action_dim=6, mode="structured",
        dof_layout=DOFLayout.cheetah(),
    ),
    "cheetah_compliant": dict(
        obs_dim=17, action_dim=6, mode="structured", contact_force=4,
        contact_solver="compliant", dof_layout=DOFLayout.cheetah(),
    ),
    # Periodic sin/cos features and joint limits: exercises the feature-map
    # chain rule and the rail-spring gradient that cheetah does not reach.
    "cartpole": dict(
        obs_dim=4, action_dim=1, mode="structured",
        dof_layout=DOFLayout.cartpole(),
    ),
    "acrobot": dict(
        obs_dim=4, action_dim=1, mode="structured",
        dof_layout=DOFLayout.acrobot(),
    ),
}


class TestAnalyticMatchesFunctorch(unittest.TestCase):
    def _sample(self, model, batch=6, seed=1):
        gen = th.Generator().manual_seed(seed)
        x = th.randn(batch, model.obs_dim, generator=gen)
        a = th.randn(batch, model.action_dim, generator=gen)
        return x, a

    def test_feature_map_matches_per_sample_reference(self):
        """The batched feature map must reproduce ``_position_features`` exactly."""
        for name, kwargs in CASES.items():
            model, _ = _paired_models(**kwargs)
            x, _ = self._sample(model)
            pos = x[:, model.layout.pos_slice[0]:model.layout.pos_slice[1]]
            for tag, excluded in (
                ("mass", model._mass_excluded_pos),
                ("potential", model._potential_excluded_pos),
            ):
                with self.subTest(case=name, tag=tag):
                    batched, _, _ = model._batched_features(pos, tag)
                    expected = th.stack(
                        [model._position_features(row, excluded) for row in pos]
                    )
                    self.assertEqual(batched.shape, expected.shape)
                    th.testing.assert_close(batched, expected, rtol=RTOL, atol=ATOL)

    def test_mass_batch_matches_vmap(self):
        for name, kwargs in CASES.items():
            model, _ = _paired_models(**kwargs)
            x, _ = self._sample(model)
            pos = x[:, model.layout.pos_slice[0]:model.layout.pos_slice[1]]
            with self.subTest(case=name), th.no_grad():
                expected = th.stack([model._mass(row) for row in pos])
                th.testing.assert_close(
                    model._mass_batch(pos), expected, rtol=RTOL, atol=ATOL
                )

    def test_mechanics_terms_match_reference(self):
        """M, dH/dq and Mdot@qd from one trace vs the jacfwd/grad reference."""
        for name, kwargs in CASES.items():
            analytic, reference = _paired_models(**kwargs)
            x, _ = self._sample(analytic)
            pos = x[:, analytic.layout.pos_slice[0]:analytic.layout.pos_slice[1]]
            qd = x[:, analytic.layout.vel_slice[0]:analytic.layout.vel_slice[1]]
            with self.subTest(case=name), th.no_grad():
                got = analytic._mechanics_terms(pos, qd)
                want = reference._mechanics_terms(pos, qd)
                for tensor, expected, label in zip(got, want, ("M", "dHdq", "Mdot_qd")):
                    with self.subTest(term=label):
                        th.testing.assert_close(
                            tensor, expected, rtol=RTOL, atol=ATOL
                        )

    def test_contact_geometry_matches_reference(self):
        for name, kwargs in CASES.items():
            if not kwargs.get("contact_force"):
                continue
            analytic, reference = _paired_models(**kwargs)
            x, _ = self._sample(analytic)
            pos = x[:, analytic.layout.pos_slice[0]:analytic.layout.pos_slice[1]]
            qd = x[:, analytic.layout.vel_slice[0]:analytic.layout.vel_slice[1]]
            with self.subTest(case=name), th.no_grad():
                got = analytic._contact_geometry(pos, qd)
                want = reference._contact_geometry(pos, qd)
                labels = ("g", "gdot", "v_t", "J_n", "J_t")
                for tensor, expected, label in zip(got, want, labels):
                    with self.subTest(term=label):
                        th.testing.assert_close(
                            tensor, expected, rtol=RTOL, atol=ATOL
                        )

    def test_drift_matches_reference(self):
        """The quantity the critic target and the fit actually consume."""
        for name, kwargs in CASES.items():
            analytic, reference = _paired_models(**kwargs)
            x, a = self._sample(analytic)
            with self.subTest(case=name), th.no_grad():
                th.testing.assert_close(
                    analytic.drift(x, a), reference.drift(x, a), rtol=RTOL, atol=ATOL
                )

    def test_free_acceleration_matches_reference(self):
        for name, kwargs in CASES.items():
            if not kwargs.get("contact_force"):
                continue
            analytic, reference = _paired_models(**kwargs)
            x, a = self._sample(analytic)
            with self.subTest(case=name), th.no_grad():
                got = analytic._structured_free_acceleration(x, a)
                want = reference._structured_free_acceleration(x, a)
                for tensor, expected in zip(got, want):
                    th.testing.assert_close(tensor, expected, rtol=RTOL, atol=ATOL)

    def test_parameter_gradients_match_reference(self):
        """A wrong backward would corrupt the fit while the forward looked fine."""
        for name, kwargs in CASES.items():
            analytic, reference = _paired_models(**kwargs)
            x, a = self._sample(analytic)
            grads = {}
            for tag, model in (("analytic", analytic), ("reference", reference)):
                model.zero_grad(set_to_none=True)
                # A weighted, asymmetric loss so no term can cancel out.
                weights = th.linspace(0.5, 1.5, x.shape[1])
                (model.drift(x, a) * weights).square().sum().backward()
                grads[tag] = {
                    key: value.grad.detach().clone()
                    for key, value in model.named_parameters()
                    if value.grad is not None
                }
            with self.subTest(case=name):
                self.assertEqual(
                    set(grads["analytic"]), set(grads["reference"]),
                    "both paths must produce gradients for the same parameters",
                )
                self.assertTrue(grads["analytic"], "expected non-empty gradients")
                for key, value in grads["analytic"].items():
                    with self.subTest(parameter=key):
                        expected = grads["reference"][key]
                        scale = max(float(expected.abs().max()), 1.0)
                        th.testing.assert_close(
                            value, expected, rtol=1e-4, atol=1e-5 * scale
                        )

    def test_analytic_path_is_the_default(self):
        model = PortHamiltonianModel(
            obs_dim=17, action_dim=6, mode="structured",
            contact_force=4, dof_layout=DOFLayout.cheetah(),
        )
        self.assertTrue(model.analytic_derivatives)

    def test_feature_index_plans_stay_out_of_the_state_dict(self):
        """Checkpoint compatibility: the new buffers must not be persisted."""
        model = PortHamiltonianModel(
            obs_dim=4, action_dim=1, mode="structured",
            dof_layout=DOFLayout.cartpole(),
        )
        keys = set(model.state_dict())
        self.assertFalse(
            [key for key in keys if "_feat_" in key],
            f"feature-plan buffers leaked into the state_dict: {sorted(keys)}",
        )
        reference = PortHamiltonianModel(
            obs_dim=4, action_dim=1, mode="structured",
            dof_layout=DOFLayout.cartpole(), analytic_derivatives=False,
        )
        self.assertEqual(keys, set(reference.state_dict()))
        reference.load_state_dict(model.state_dict())

    def test_bare_linear_geometry_net_is_supported(self):
        """Tests and recovery diagnostics swap in a single Linear for the geometry."""
        analytic, reference = _paired_models(
            obs_dim=17, action_dim=6, mode="structured", contact_force=4,
            dof_layout=DOFLayout.cheetah(),
        )
        for model in (analytic, reference):
            th.manual_seed(3)
            model.gap_net = th.nn.Linear(model.layout.npos, model.contact_force)
            th.manual_seed(4)
            model.tangent_net = th.nn.Linear(model.layout.npos, model.contact_force)
        x, a = self._sample(analytic)
        with th.no_grad():
            th.testing.assert_close(
                analytic.drift(x, a), reference.drift(x, a), rtol=RTOL, atol=ATOL
            )

    def test_unsupported_activation_is_rejected(self):
        """Silent disagreement with autograd is worse than a loud failure."""
        from torch import nn

        from models.port_hamiltonian import _mlp_trace

        net = nn.Sequential(nn.Linear(3, 4), nn.Softplus(), nn.Linear(4, 2))
        with self.assertRaises(TypeError):
            _mlp_trace(net, th.zeros(2, 3))


if __name__ == "__main__":
    unittest.main()
