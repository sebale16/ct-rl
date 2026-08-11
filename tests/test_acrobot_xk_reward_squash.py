"""Contract tests for the bounded ``r1`` reward.

``r1 = -V`` is unbounded: ``V`` carries ``1/2 Etil^2`` and ``Etil`` grows with
kinetic energy, so ``V`` scales as ``qdot^4``.  Training reached ``V ~ 7.8e9``
and the auto-tuned temperature followed it to ``1e6`` on every Lyapunov arm.
``reward_squash = V0`` replaces ``V`` with ``V / (1 + V/V0)``, which is bounded
by ``V0`` and strictly increasing, so the state ordering -- including the
penalty on unwrapped elbow winding -- is unchanged.
"""

import unittest

import numpy as np

from environment.acrobot_xk import BalanceXK


V0 = 1200.0


def _task(**kwargs):
    return BalanceXK(reward_kind="r1", **kwargs)


class TestRewardSquashContract(unittest.TestCase):
    def test_squash_is_identity_when_disabled(self):
        task = _task()
        self.assertIsNone(task.reward_squash)
        for v in (0.0, 4.13, 300.1, 1155.0, 1e6, 7.8e9):
            self.assertEqual(task._squash(v), v)

    def test_squash_fixes_the_target_and_is_tangent_there(self):
        task = _task(reward_squash=V0)
        self.assertEqual(task._squash(0.0), 0.0)
        # d/dV [V/(1+V/V0)] = 1 at V = 0, so the near-goal gradient that
        # stabilization depends on is exactly the unsquashed one.
        h = 1e-6
        self.assertAlmostEqual((task._squash(h) - task._squash(0.0)) / h, 1.0, places=6)

    def test_squash_is_bounded_by_v0(self):
        task = _task(reward_squash=V0)
        # Bounded everywhere, and strictly below V0 across the whole range the
        # physics can reach.  In IEEE double the expression rounds up to exactly
        # V0 only beyond V ~ 1e19, which is ten orders past the worst V ever
        # observed in training (7.8e9), so the bound is what matters, not the
        # strictness of the inequality at the float ceiling.
        for v in (1e3, 1e6, 1e9, 7.8e9, 1e13):
            self.assertLess(task._squash(v), V0)
        for v in (1e19, 1e30, 1e300):
            self.assertLessEqual(task._squash(v), V0)
        self.assertGreater(task._squash(1e30), 0.999 * V0)

    def test_squash_is_strictly_increasing_over_the_reachable_range(self):
        # Order preservation is the property that keeps the unwrapped-elbow
        # winding penalty meaningful, so it must hold everywhere the physics
        # can go.  1e13 is four orders past the worst V seen in training.
        task = _task(reward_squash=V0)
        grid = np.unique(
            np.concatenate(
                [np.linspace(0.0, 1e4, 20001), np.logspace(4.0, 13.0, 20000)]
            )
        )
        squashed = np.array([task._squash(float(v)) for v in grid])
        self.assertTrue(np.all(np.diff(squashed) > 0.0))

    def test_saturation_tail_is_flat_to_within_rounding(self):
        # Past V ~ 5e17 the squash is numerically pinned to V0 and consecutive
        # values can differ by one ULP in either direction.  That is the
        # intended saturation, not a monotonicity defect: the reward
        # differences involved are ~2e-13, and the states are eight orders
        # beyond anything reachable.  Assert flatness, not strict ordering.
        task = _task(reward_squash=V0)
        grid = np.logspace(18.0, 300.0, 20000)
        squashed = np.array([task._squash(float(v)) for v in grid])
        self.assertTrue(np.all(np.abs(squashed - V0) < 1e-9))

    def test_near_goal_distortion_is_negligible(self):
        # V at the homoclinic tube boundary is ~4.13; V0 is 290x larger, so the
        # region the capture criterion lives in must be essentially untouched.
        task = _task(reward_squash=V0)
        self.assertLess(abs(task._squash(4.13) - 4.13) / 4.13, 0.005)

    def test_release_start_range_stays_in_the_linear_region(self):
        # The release-start distribution spans V ~ 1062-1197, so compression
        # over the whole legitimate operating range must stay within 2x.
        task = _task(reward_squash=V0)
        for v in (1062.0, 1197.0):
            self.assertGreater(task._squash(v), v / 2.0)

    def test_squash_rejects_non_positive_and_non_finite(self):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    _task(reward_squash=bad)

    def test_squash_is_rejected_for_r0_and_r2(self):
        # r2 = -V - eta Vdot keeps an unbounded, signed Vdot term that this
        # squash does not touch, so the combination must not silently pass.
        with self.assertRaises(ValueError):
            BalanceXK(reward_kind="r0", reward_squash=V0)
        with self.assertRaises(ValueError):
            BalanceXK(reward_kind="r2", eta=0.1, reward_squash=V0)

    def test_default_construction_is_unchanged(self):
        for kind in ("r0", "r1"):
            self.assertIsNone(BalanceXK(reward_kind=kind).reward_squash)
        self.assertIsNone(BalanceXK(reward_kind="r2", eta=0.0).reward_squash)




class TestSpinTerminationContract(unittest.TestCase):
    """``spin_limit`` ends the episode; ``reward_offset`` decides its sign.

    With ``r1 = -s(V) <= 0`` a termination is a payout: it avoids the rest of
    the episode's negative reward.  Measured on this plant, a spinner that
    terminates at 1286 steps returns -9.8e5 while a passive episode returns
    -2.35e7, so spinning wins by 24x with no penalty and still wins at
    ``spin_penalty = 1e5``.  Deterring it by penalty alone needs > 2.3e7, the
    same magnitude that drove the temperature to 1e6.  Offsetting the reward to
    >= 0 deters it at ``spin_penalty = 0``.
    """

    def test_spin_limit_validation(self):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    _task(spin_limit=bad)

    def test_spin_penalty_requires_a_limit(self):
        with self.assertRaises(ValueError):
            _task(spin_penalty=1.0)
        for bad in (-1.0, float("nan")):
            with self.assertRaises(ValueError):
                _task(spin_limit=1.0, spin_penalty=bad)

    def test_offset_and_squash_are_r1_only(self):
        with self.assertRaises(ValueError):
            BalanceXK(reward_kind="r0", reward_offset=1.0)
        with self.assertRaises(ValueError):
            BalanceXK(reward_kind="r2", eta=0.1, reward_offset=1.0)

    def test_offset_lifts_r1_to_non_negative(self):
        # r1 = offset - s(V), and s(V) < V0, so offset = V0 gives r1 > 0.
        task = _task(reward_squash=V0, reward_offset=V0)
        for v in (0.0, 4.13, 1200.5, 1e6, 7.8e9):
            self.assertGreater(V0 - task._squash(v), 0.0)

    def test_defaults_leave_the_task_unchanged(self):
        task = _task()
        self.assertIsNone(task.spin_limit)
        self.assertEqual(task.spin_penalty, 0.0)
        self.assertEqual(task.reward_offset, 0.0)


if __name__ == "__main__":
    unittest.main()
