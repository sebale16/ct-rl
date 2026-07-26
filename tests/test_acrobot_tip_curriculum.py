import csv
import unittest

import numpy as np

try:
    from dm_control.suite import acrobot as dmc_acrobot

    from environment.acrobot_v2 import (
        ACROBOT_BRAKE_TIP_HEIGHT,
        ACROBOT_BRAKE_TIP_SPEED,
        ACROBOT_DESCENT_TIP_HEIGHTS,
        BalanceV4,
        BalanceV6,
        BalanceV43,
        BalanceV61,
        V41_ENERGY_OVERSHOOT_MARGIN,
        V41_SPEED_BOUNDS,
        V41_SPEED_MARGIN,
        swingup_v43,
        swingup_v61,
    )

    HAVE_DMC = True
except Exception:  # pragma: no cover - exercised only without dm_control
    HAVE_DMC = False


@unittest.skipUnless(HAVE_DMC, "dm_control / Acrobot-v2 not available")
class TestAcrobotTipHeightVelocityCurriculum(unittest.TestCase):
    def setUp(self):
        self.physics = self._new_physics()

    @staticmethod
    def _new_physics():
        return dmc_acrobot.Physics.from_xml_string(
            *dmc_acrobot.get_model_and_assets()
        )

    @staticmethod
    def _set_state(physics, qpos, qvel, control):
        physics.named.data.qpos[["shoulder", "elbow"]] = qpos
        physics.data.qvel[:] = np.asarray(qvel, dtype=np.float64)
        physics.data.ctrl[:] = float(control)
        physics.forward()

    def _reset_stage(self, task, stage):
        task.set_curriculum_stage(stage)
        task.initialize_episode(self.physics)
        self.physics.forward()

    def test_default_ladder_is_brake_then_resting_descent(self):
        task = BalanceV43(random=0)
        levels = task.curriculum_levels

        self.assertEqual(len(levels), 1 + len(ACROBOT_DESCENT_TIP_HEIGHTS))
        self.assertEqual(levels[0].tip_height, ACROBOT_BRAKE_TIP_HEIGHT)
        self.assertEqual(
            levels[0].incoming_tip_speed, ACROBOT_BRAKE_TIP_SPEED
        )
        self.assertEqual(
            tuple(level.tip_height for level in levels[1:]),
            ACROBOT_DESCENT_TIP_HEIGHTS,
        )
        self.assertTrue(
            all(level.incoming_tip_speed == 0.0 for level in levels[1:])
        )

    def test_brake_reset_has_exact_height_speed_and_target_direction(self):
        task = BalanceV43(random=0)
        self._reset_stage(task, 0)

        tip = np.asarray(
            self.physics.named.data.site_xpos["tip"], dtype=np.float64
        )
        target = np.asarray(
            self.physics.named.data.site_xpos["target"], dtype=np.float64
        )
        tip_velocity = task._tip_cartesian_velocity(self.physics)

        self.assertAlmostEqual(tip[2], ACROBOT_BRAKE_TIP_HEIGHT, places=12)
        self.assertAlmostEqual(
            np.linalg.norm(tip_velocity), ACROBOT_BRAKE_TIP_SPEED, places=12
        )
        self.assertGreater(float(tip_velocity @ (target - tip)), 0.0)
        self.assertAlmostEqual(float(self.physics.data.qpos[1]), 0.0, places=12)
        self.assertAlmostEqual(float(self.physics.data.qvel[1]), 0.0, places=12)

        side = float(np.sign(self.physics.data.qpos[0]))
        expected_angle = np.arccos(
            (ACROBOT_BRAKE_TIP_HEIGHT - 2.0) / 2.0
        )
        self.assertAlmostEqual(
            float(self.physics.data.qpos[0]), side * expected_angle, places=12
        )
        self.assertAlmostEqual(
            float(self.physics.data.qvel[0]),
            -side * ACROBOT_BRAKE_TIP_SPEED / 2.0,
            places=12,
        )

    def test_descent_resets_have_exact_height_and_zero_velocity(self):
        task = BalanceV43(random=7)

        for stage, height in enumerate(
            ACROBOT_DESCENT_TIP_HEIGHTS, start=1
        ):
            with self.subTest(stage=stage, height=height):
                self._reset_stage(task, stage)
                actual_height = float(
                    self.physics.named.data.site_xpos["tip", "z"]
                )
                self.assertAlmostEqual(actual_height, height, places=12)
                np.testing.assert_array_equal(
                    np.asarray(self.physics.data.qvel), np.zeros(2)
                )
                self.assertAlmostEqual(
                    task._tip_cartesian_speed(self.physics), 0.0, places=12
                )

        np.testing.assert_allclose(
            np.asarray(self.physics.data.qpos),
            [np.pi, 0.0],
            rtol=0.0,
            atol=1e-12,
        )
        self.assertTrue(task.curriculum_complete)

    def test_stage_setter_clips_and_rejects_non_integer_values(self):
        task = BalanceV43(random=0)

        task.set_curriculum_stage(-10)
        self.assertEqual(task.curriculum_stage, 0)
        task.set_curriculum_stage(10_000)
        self.assertEqual(
            task.curriculum_stage, task.num_curriculum_stages - 1
        )
        with self.assertRaises(ValueError):
            task.set_curriculum_stage(1.5)
        with self.assertRaises(ValueError):
            task.set_curriculum_stage(float("nan"))

    def test_reseed_reproduces_mirrored_reset_sequence(self):
        task = BalanceV43(random=999)

        def draw_sequence():
            sequence = []
            for _ in range(12):
                task.initialize_episode(self.physics)
                sequence.append(
                    np.concatenate(
                        [
                            np.asarray(self.physics.data.qpos).copy(),
                            np.asarray(self.physics.data.qvel).copy(),
                        ]
                    )
                )
            return np.asarray(sequence)

        task.reseed(12345)
        first = draw_sequence()
        task.reseed(12345)
        second = draw_sequence()
        np.testing.assert_array_equal(first, second)
        self.assertEqual(set(np.sign(first[:, 0])), {-1.0, 1.0})

    def test_curriculum_false_is_exact_hanging_for_both_rewards(self):
        for task_type in (BalanceV43, BalanceV61):
            with self.subTest(task_type=task_type.__name__):
                task = task_type(
                    random=0,
                    angle_noise=0.7,
                    velocity_noise=3.0,
                    curriculum=False,
                )
                task.initialize_episode(self.physics)
                np.testing.assert_allclose(
                    np.asarray(self.physics.data.qpos),
                    [np.pi, 0.0],
                    rtol=0.0,
                    atol=1e-12,
                )
                np.testing.assert_array_equal(
                    np.asarray(self.physics.data.qvel), np.zeros(2)
                )

    def test_v43_reward_is_exactly_v41_reward(self):
        base_physics = self._new_physics()
        new_physics = self._new_physics()
        base = BalanceV4(
            random=0,
            hold_weight=0.8,
            energy_overshoot_margin=V41_ENERGY_OVERSHOOT_MARGIN,
            speed_bounds=V41_SPEED_BOUNDS,
            speed_margin=V41_SPEED_MARGIN,
            uniform_start=False,
            curriculum=False,
        )
        new = BalanceV43(random=0, hold_weight=0.8)
        base.initialize_episode(base_physics)
        new.initialize_episode(new_physics)
        state = ((0.4, -0.3), (1.5, -2.5), 0.6)
        self._set_state(base_physics, *state)
        self._set_state(new_physics, *state)

        base_terms = base.reward_terms(base_physics)
        new_terms = new.reward_terms(new_physics)
        self.assertEqual(new_terms["reward"], base_terms["reward"])
        for key in (
            "progress",
            "precision",
            "energy_norm",
            "slow_gate",
            "hold",
        ):
            self.assertEqual(new_terms[key], base_terms[key])

    def test_v61_reward_is_exactly_v6_reward(self):
        base_physics = self._new_physics()
        new_physics = self._new_physics()
        base = BalanceV6(
            random=0, curriculum=False, uniform_start=False
        )
        new = BalanceV61(random=0)
        base.initialize_episode(base_physics)
        new.initialize_episode(new_physics)
        state = ((0.4, -0.3), (1.5, -2.5), 0.6)
        self._set_state(base_physics, *state)
        self._set_state(new_physics, *state)

        base_terms = base.reward_terms(base_physics)
        new_terms = new.reward_terms(new_physics)
        for key in ("reward", "angle_cost", "velocity_cost", "action_cost"):
            self.assertEqual(new_terms[key], base_terms[key])

    def test_new_strict_capture_uses_cartesian_tip_speed(self):
        for task_type in (BalanceV43, BalanceV61):
            with self.subTest(task_type=task_type.__name__):
                task = task_type(random=0)
                task.initialize_episode(self.physics)
                # At the extended upright pose, J_tip qdot = 2*qdot1 + qdot2.
                # The links can counter-rotate while the tip is instantaneously
                # stationary, which is deliberately accepted by the tip-only
                # stabilization predicate.
                self._set_state(
                    self.physics,
                    qpos=(0.0, 0.0),
                    qvel=(1.0, -2.0),
                    control=0.0,
                )
                terms = task.reward_terms(self.physics)

                self.assertGreater(terms["speed"], 0.2)
                self.assertAlmostEqual(terms["tip_speed"], 0.0, places=12)
                self.assertEqual(terms["strict_capture"], 1.0)
                for key in (
                    "curriculum_stage",
                    "curriculum_num_stages",
                    "curriculum_start_tip_height",
                    "curriculum_start_tip_speed",
                    "curriculum_complete",
                ):
                    self.assertIn(key, terms)

    def test_factories_select_new_reward_preserving_tasks(self):
        for factory, task_type in (
            (swingup_v43, BalanceV43),
            (swingup_v61, BalanceV61),
        ):
            with self.subTest(factory=factory.__name__):
                env = factory(time_limit=0.1, random=0)
                try:
                    env.reset()
                    self.assertIsInstance(env.task, task_type)
                finally:
                    env.close()


class TestV61EntropyAblationConfig(unittest.TestCase):
    def test_fixed_entropy_modes_exactly_mirror_v6(self):
        expected_alpha = {
            "fixed_a2p0": "2.0",
            "fixed_a0p5": "0.5",
            "fixed_a0p1": "0.1",
        }
        with open(
            "benchmarks/hyperparams/ct_sac.csv", newline=""
        ) as csv_file:
            rows = list(csv.DictReader(csv_file))

        for mode, alpha in expected_alpha.items():
            with self.subTest(mode=mode):
                parent = [
                    row
                    for row in rows
                    if row["env_id"] == "acrobot-swingup-v6"
                    and row["mode"] == mode
                ]
                branch = [
                    row
                    for row in rows
                    if row["env_id"] == "acrobot-swingup-v6.1"
                    and row["mode"] == mode
                ]
                self.assertEqual(len(parent), 1)
                self.assertEqual(len(branch), 1)
                self.assertEqual(branch[0]["algo_alpha"], alpha)

                parent_fields = dict(parent[0])
                branch_fields = dict(branch[0])
                parent_fields.pop("env_id")
                branch_fields.pop("env_id")
                self.assertEqual(branch_fields, parent_fields)


if __name__ == "__main__":
    unittest.main()
