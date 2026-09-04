"""``roa_start_fraction``: mixing SOS-certified-region starts into
``release_start`` (see environment.acrobot_xk.BalanceXK.initialize_episode
and controllers.acrobot_sos_switched.SOSCertificate.sample_uniform)."""

import os
import unittest

os.environ.setdefault("MUJOCO_GL", "disable")

import numpy as np

from controllers.acrobot_sos_switched import SOSCertificate
from environment.acrobot_xk import HANGING_SHOULDER, swingup_xk

# Small, hand-designed certificate, independent of the real extracted result
# (see tests/test_acrobot_sos_switched.py) -- just needs to be a valid
# SOSCertificate to exercise the mixing logic.
FAKE_CERTIFICATE = SOSCertificate(
    tau_max=20.0,
    rho=1.0,
    P=np.diag([4.0, 1.0, 0.25, 0.25]),
    controller_terms=(((1, 0, 0, 0), -1.0),),
)


def _env(roa_start_fraction=0.0, **kwargs):
    env = swingup_xk(
        release_start=True,
        angle_noise=0.0,
        velocity_noise=0.0,
        roa_start_fraction=roa_start_fraction,
        **kwargs,
    )
    # Bypass the file loader so this test doesn't depend on
    # results/acrobot_sos_roa_tau20_certificate.json existing.
    env.task._sos_certificate = FAKE_CERTIFICATE
    return env


class TestROAStartFraction(unittest.TestCase):
    def test_rejects_out_of_range_fraction(self):
        with self.assertRaisesRegex(ValueError, "roa_start_fraction"):
            swingup_xk(release_start=True, roa_start_fraction=1.5)
        with self.assertRaisesRegex(ValueError, "roa_start_fraction"):
            swingup_xk(release_start=True, roa_start_fraction=-0.1)

    def test_rejects_a_positive_fraction_without_release_start(self):
        with self.assertRaisesRegex(ValueError, "release_start=True"):
            swingup_xk(release_start=False, roa_start_fraction=0.2)

    def test_zero_fraction_matches_plain_release_start_exactly(self):
        env = _env(roa_start_fraction=0.0)
        for seed in range(20):
            env.task.reseed(seed)
            env.reset()
            qpos = np.asarray(env.physics.data.qpos)
            qvel = np.asarray(env.physics.data.qvel)
            self.assertEqual(qpos[1], 0.0)  # elbow locked at 0
            self.assertTrue(np.all(qvel == 0.0))  # released from rest
            self.assertNotAlmostEqual(qpos[0], HANGING_SHOULDER, places=6)

    def test_full_fraction_always_draws_from_the_certified_region(self):
        env = _env(roa_start_fraction=1.0)
        residuals = []
        any_nonzero_velocity = False
        for seed in range(50):
            env.task.reseed(seed)
            env.reset()
            state = np.concatenate(
                [env.physics.data.qpos, env.physics.data.qvel]
            )
            residuals.append(FAKE_CERTIFICATE.residual(state))
            if np.any(env.physics.data.qvel != 0.0):
                any_nonzero_velocity = True
        self.assertTrue(np.all(np.asarray(residuals) <= FAKE_CERTIFICATE.rho + 1e-9))
        self.assertTrue(any_nonzero_velocity)

    def test_partial_fraction_mixes_both_kinds_of_start(self):
        env = _env(roa_start_fraction=0.5)
        release_like, roa_like = 0, 0
        for seed in range(200):
            env.task.reseed(seed)
            env.reset()
            qvel = np.asarray(env.physics.data.qvel)
            if np.all(qvel == 0.0):
                release_like += 1
            else:
                roa_like += 1
        # Both kinds actually occur across 200 draws at a 50/50 split;
        # loose bounds since ROA draws can also land at exactly zero
        # velocity (probability ~0, not asserted against).
        self.assertGreater(release_like, 50)
        self.assertGreater(roa_like, 50)


if __name__ == "__main__":
    unittest.main()
