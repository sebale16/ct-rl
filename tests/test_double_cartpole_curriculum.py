import unittest

import numpy as np

try:
    from environment.double_cartpole_v2 import (
        CartpoleTwoPolesCurriculum,
        two_poles_curriculum,
    )

    HAVE_DMC = True
except Exception:  # pragma: no cover - exercised only without dm_control
    HAVE_DMC = False


@unittest.skipUnless(HAVE_DMC, "dm_control / double cartpole not available")
class TestDoubleCartPoleCurriculum(unittest.TestCase):
    """Reverse-curriculum reset for the double cartpole; stock reward intact."""

    _MIN_SPREAD = 0.5

    def _env(self, **kw):
        env = two_poles_curriculum(random=0, velocity_noise=0.0, **kw)
        self.addCleanup(env.close)
        return env

    def _mean_upright(self, env, frac, n=300):
        env.task.set_curriculum_fraction(frac)
        ups = np.empty(n)
        for i in range(n):
            env.reset()
            cos = np.asarray(env.physics.pole_angle_cosine())
            ups[i] = ((cos + 1.0) / 2.0).mean()
        return ups

    def test_fraction_zero_starts_near_upright(self):
        env = self._env(curriculum_min_spread=self._MIN_SPREAD)
        # Both poles near vertical: mean uprightness close to 1.
        self.assertGreater(self._mean_upright(env, 0.0).mean(), 0.9)

    def test_band_widens_monotonically_with_fraction(self):
        env = self._env(curriculum_min_spread=self._MIN_SPREAD)
        means = [
            self._mean_upright(env, f, n=250).mean() for f in (0.0, 0.25, 0.5, 1.0)
        ]
        for hi, lo in zip(means, means[1:]):
            self.assertGreater(hi, lo)  # uprightness falls as the band widens

    def test_fraction_one_matches_the_uniform_reset(self):
        env = self._env(curriculum_min_spread=self._MIN_SPREAD)
        curric = self._mean_upright(env, 1.0, n=400).mean()
        env2 = two_poles_curriculum(
            random=0, velocity_noise=0.0, curriculum=False, uniform_start=True
        )
        self.addCleanup(env2.close)
        ups = []
        for _ in range(400):
            env2.reset()
            ups.append(
                ((np.asarray(env2.physics.pole_angle_cosine()) + 1.0) / 2.0).mean()
            )
        self.assertAlmostEqual(curric, float(np.mean(ups)), delta=0.05)

    def test_set_curriculum_fraction_clamps(self):
        task = self._env().task
        task.set_curriculum_fraction(-1.0)
        self.assertEqual(task.curriculum_fraction, 0.0)
        task.set_curriculum_fraction(5.0)
        self.assertEqual(task.curriculum_fraction, 1.0)
        with self.assertRaises(ValueError):
            task.set_curriculum_fraction(float("nan"))

    def test_invalid_min_spread_rejected(self):
        for bad in (0.0, -0.1, np.pi + 0.1, float("nan")):
            with self.assertRaises(ValueError):
                CartpoleTwoPolesCurriculum(random=0, curriculum_min_spread=bad)

    def test_reward_is_unchanged_stock_two_poles(self):
        env = self._env()
        p = env.physics
        p.data.qpos[:] = 0.0
        p.data.qvel[:] = 0.0
        p.data.ctrl[:] = 0.0
        p.forward()
        # Stock two_poles reward is 1.0 at the upright-centered-still pose.
        self.assertAlmostEqual(float(env.task.get_reward(p)), 1.0, places=6)

    def test_curriculum_disabled_falls_back_to_uniform(self):
        env = two_poles_curriculum(
            random=0, velocity_noise=0.0, curriculum=False, uniform_start=True
        )
        self.addCleanup(env.close)
        # A pushed fraction is ignored when the curriculum is off.
        env.task.set_curriculum_fraction(0.0)
        ups = []
        for _ in range(200):
            env.reset()
            ups.append(
                ((np.asarray(env.physics.pole_angle_cosine()) + 1.0) / 2.0).mean()
            )
        self.assertLess(float(np.mean(ups)), 0.65)  # full range, not the band


if __name__ == "__main__":
    unittest.main()
