import json
import os
import tempfile
import unittest

import numpy as np

import controllers.xin_kaneda as xk
from controllers.acrobot_gated_lyapunov import UPRIGHT_STATE, upright_error
from controllers.acrobot_sos_switched import (
    SOSCertificate,
    XKSOSSwitchedController,
    load_certificate,
)

GAINS = xk.Gains(k_v=66.3, k_d=35.8, k_p=61.2)

# A small, hand-designed certificate for tests -- not the real extracted
# result, just something with known, checkable numbers. Diagonal P makes
# the ellipsoid's single-axis bounds exact: |e_i| <= sqrt(rho / P[i,i]).
DIAGONAL_P = np.diag([4.0, 1.0, 0.25, 0.25])
RHO = 1.0
# u(e) = 2*e0 - 3*e0*e2 + e2**3  (degree 1, degree 2, degree 3 terms)
CONTROLLER_TERMS = (
    ((1, 0, 0, 0), 2.0),
    ((1, 0, 1, 0), -3.0),
    ((0, 0, 3, 0), 1.0),
)


def make_certificate(rho=RHO, p=DIAGONAL_P, terms=CONTROLLER_TERMS, tau_max=20.0):
    return SOSCertificate(
        tau_max=tau_max, rho=rho, P=p, controller_terms=terms,
    )


class TestSOSCertificate(unittest.TestCase):
    def test_rejects_bad_shapes_and_definiteness(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            SOSCertificate(tau_max=20.0, rho=1.0, P=np.eye(3), controller_terms=())
        with self.assertRaisesRegex(ValueError, "symmetric"):
            bad = np.eye(4)
            bad[0, 1] = 5.0
            SOSCertificate(tau_max=20.0, rho=1.0, P=bad, controller_terms=())
        with self.assertRaisesRegex(ValueError, "positive definite"):
            SOSCertificate(
                tau_max=20.0, rho=1.0, P=np.diag([1.0, 1.0, 1.0, -1.0]),
                controller_terms=(),
            )
        with self.assertRaisesRegex(ValueError, "rho"):
            make_certificate(rho=0.0)
        with self.assertRaisesRegex(ValueError, "rho"):
            make_certificate(rho=-1.0)

    def test_residual_and_containment_on_the_diagonal_case(self):
        cert = make_certificate()
        # Exactly on the boundary along e0 alone: 4*e0^2 = rho=1 -> e0=0.5.
        boundary = UPRIGHT_STATE + np.array([0.5, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(cert.residual(boundary), 1.0, places=10)
        self.assertTrue(cert.contains(boundary))
        just_outside = UPRIGHT_STATE + np.array([0.50001, 0.0, 0.0, 0.0])
        self.assertFalse(cert.contains(just_outside))
        self.assertTrue(cert.contains(UPRIGHT_STATE))

    def test_residual_batch_matches_row_by_row(self):
        cert = make_certificate()
        rng = np.random.RandomState(3)
        states = UPRIGHT_STATE + rng.uniform(-1.0, 1.0, (40, 4))
        batched = cert.residual_batch(states)
        expected = np.array([cert.residual(s) for s in states])
        np.testing.assert_allclose(batched, expected, atol=1e-10)
        np.testing.assert_array_equal(cert.contains_batch(states), expected <= cert.rho)

    def test_command_matches_the_hand_derivation(self):
        cert = make_certificate()
        state = UPRIGHT_STATE + np.array([0.1, 0.0, 0.2, 0.0])
        # u = 2*e0 - 3*e0*e2 + e2**3 = 2*0.1 - 3*0.1*0.2 + 0.2**3
        expected = 2 * 0.1 - 3 * 0.1 * 0.2 + 0.2**3
        self.assertAlmostEqual(cert.command(state), expected, places=12)

    def test_command_batch_matches_row_by_row(self):
        cert = make_certificate()
        rng = np.random.RandomState(4)
        states = UPRIGHT_STATE + rng.uniform(-0.5, 0.5, (30, 4))
        batched = cert.command_batch(states)
        expected = np.array([cert.command(s) for s in states])
        np.testing.assert_allclose(batched, expected, atol=1e-10)

    def test_sample_uniform_always_lands_inside_and_uses_all_coordinates(self):
        cert = make_certificate()
        rng = np.random.RandomState(7)
        samples = np.array([cert.sample_uniform(rng) for _ in range(500)])
        residuals = cert.residual_batch(samples)
        self.assertTrue(np.all(residuals <= cert.rho + 1e-9))
        errors = samples - UPRIGHT_STATE
        # Both angle and rate coordinates should actually vary, and take
        # both signs -- this is not the release law's zero-velocity start.
        for i in range(4):
            self.assertGreater(errors[:, i].std(), 0.0)
        self.assertTrue((errors[:, 2] != 0.0).any())
        self.assertTrue((errors[:, 3] != 0.0).any())
        self.assertTrue((errors > 0).any() and (errors < 0).any())

    def test_load_certificate_round_trips_through_json(self):
        cert = make_certificate()
        payload = {
            "tau_max": cert.tau_max,
            "rho": cert.rho,
            "lqr_level": 1.23e-6,
            "converged": False,
            "passes": 4,
            "P": cert.P.tolist(),
            "controller_terms": [[list(exps), c] for exps, c in cert.controller_terms],
        }
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cert.json")
            with open(path, "w") as f:
                json.dump(payload, f)
            loaded = load_certificate(path)
        self.assertEqual(loaded.tau_max, cert.tau_max)
        self.assertEqual(loaded.rho, cert.rho)
        np.testing.assert_allclose(loaded.P, cert.P)
        self.assertEqual(loaded.controller_terms, cert.controller_terms)
        self.assertEqual(loaded.passes, 4)


class TestXKSOSSwitchedController(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = xk.PAPER_PARAMS

    def _controller(self, certificate=None):
        return XKSOSSwitchedController(
            self.params, GAINS, certificate or make_certificate()
        )

    def test_starts_and_stays_in_swing_up_away_from_the_region(self):
        c = self._controller()
        hanging = np.array([-0.5 * np.pi, 0.0, 0.0, 0.0])
        self.assertEqual(c.stage, c.SWING_UP)
        action = c(hanging)
        self.assertEqual(c.stage, c.SWING_UP)
        self.assertEqual(action.shape, (1,))
        self.assertTrue(np.all(np.isfinite(action)))

    def test_switches_to_balance_on_entering_the_region_and_latches(self):
        c = self._controller()
        self.assertTrue(c.certificate.contains(UPRIGHT_STATE))
        action = c(UPRIGHT_STATE)
        self.assertEqual(c.stage, c.BALANCE)
        self.assertEqual(c.switch_step, 0)
        expected = float(
            np.clip(
                c.certificate.command(UPRIGHT_STATE), -c.torque_limit, c.torque_limit
            )
            / self.params.gear
        )
        self.assertAlmostEqual(float(action[0]), expected, places=10)

        hanging = np.array([-0.5 * np.pi, 0.0, 0.0, 0.0])
        c(hanging)
        self.assertEqual(c.stage, c.BALANCE)

    def test_reset_returns_to_swing_up(self):
        c = self._controller()
        c(UPRIGHT_STATE)
        self.assertEqual(c.stage, c.BALANCE)
        c.reset()
        self.assertEqual(c.stage, c.SWING_UP)
        self.assertIsNone(c.switch_step)

    def test_rejects_a_nonpositive_torque_limit(self):
        with self.assertRaisesRegex(ValueError, "torque_limit"):
            XKSOSSwitchedController(
                self.params, GAINS, make_certificate(), torque_limit=0.0
            )

    def test_actions_matches_call_row_by_row(self):
        rng = np.random.RandomState(11)
        hanging = np.array([-0.5 * np.pi, 0.0, 0.0, 0.0])
        states = np.vstack([
            hanging + rng.uniform(-2.0, 2.0, (30, 4)),
            UPRIGHT_STATE + rng.normal(scale=0.05, size=(10, 4)),
        ])
        batched = self._controller().actions(states)
        self.assertEqual(batched.shape, (states.shape[0], 1))
        self.assertTrue(np.all(np.isfinite(batched)))
        for i, state in enumerate(states):
            expected = self._controller()(state)
            self.assertAlmostEqual(float(batched[i, 0]), float(expected[0]), places=9)
        membership = self._controller().certificate.contains_batch(states)
        self.assertGreater(int(membership.sum()), 0)
        self.assertLess(int(membership.sum()), states.shape[0])

    def test_actions_accepts_a_single_row(self):
        c = self._controller()
        one = c.actions(UPRIGHT_STATE)
        self.assertEqual(one.shape, (1, 1))
        np.testing.assert_allclose(one, c.actions(UPRIGHT_STATE.reshape(1, -1)))


if __name__ == "__main__":
    unittest.main()
