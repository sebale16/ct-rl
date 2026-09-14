"""Physics, split isolation and policy/replay contracts for Experiment 3."""

import os
os.environ.setdefault("MUJOCO_GL", "disable")

from dataclasses import asdict
import unittest

import numpy as np

from benchmarks.acrobot_experiment3 import make_manifest, DEFAULT_MANIFEST, read_manifest
from controllers.xin_kaneda import AcrobotParams, homoclinic_speed
from environment.acrobot_generalization import (
    ConfigurationDemonstrator, RandomizedAcrobotEnv, parameter_context,
)
from environment.acrobot_xk import BalanceXK, PlantScales, swingup_xk


class TestParameterizedPlant(unittest.TestCase):
    def test_scaled_simulator_matches_mechanics_and_metric_scales(self):
        for scales in (PlantScales(), PlantScales(0.6, 1.4, 1.3, 0.7)):
            with self.subTest(scales=scales):
                env = swingup_xk(plant_scales=asdict(scales))
                self.addCleanup(env.close)
                env.reset()
                params = AcrobotParams.from_physics(env.physics)
                np.testing.assert_allclose(
                    [params.a1, params.a2, params.a3, params.b1, params.b2],
                    scales.grouped, atol=1e-12)
                self.assertAlmostEqual(env.task.energy_top, params.energy_top)
                self.assertAlmostEqual(env.task.energy_span, params.energy_span)
                self.assertAlmostEqual(env.task._rate_scale, homoclinic_speed(params))
                for q2 in (-2.1, 0.0, 1.3):
                    with env.physics.reset_context():
                        env.physics.data.qpos[:] = [0.3, q2]
                        env.physics.data.qvel[:] = [0.4, -0.7]
                    np.testing.assert_allclose(BalanceXK._mass_matrix(env.physics),
                                               params.mass_matrix(q2), atol=1e-12)
                    np.testing.assert_allclose(env.physics.data.qfrc_bias,
                                               params.bias(np.array([0.3, q2]),
                                                           np.array([0.4, -0.7])), atol=1e-12)
                    self.assertAlmostEqual(BalanceXK._mechanical_energy(env.physics),
                                           params.energy(np.array([0.3, q2]),
                                                         np.array([0.4, -0.7])))

    def test_non_nominal_failure_bounds_cover_actual_rewards(self):
        rng = np.random.default_rng(31)
        for kind in ("r0", "r1", "r2", "r3"):
            for source in (("actual",) if kind == "r0" else ("actual", "xk_closed_loop")):
                kwargs = dict(reward_kind=kind, lyapunov_rate_source=source,
                              plant_scales={"mass1": 0.6, "length1": 1.4, "length2": 0.6})
                if kind in ("r2", "r3"):
                    kwargs["eta"] = 0.1
                if kind == "r3":
                    kwargs["discount_rate"] = 0.5
                env = swingup_xk(**kwargs)
                self.addCleanup(env.close)
                env.reset()
                task = env.task
                for _ in range(30):
                    with env.physics.reset_context():
                        env.physics.data.qpos[:] = rng.uniform(
                            [-np.pi, -task.elbow_angle_limit], [np.pi, task.elbow_angle_limit])
                        caps = np.array([task.shoulder_rate_scale_limit * task._rate_scale,
                                         task.elbow_rate_limit])
                        env.physics.data.qvel[:] = rng.uniform(-caps, caps)
                        env.physics.data.ctrl[:] = rng.uniform(-1, 1)
                    self.assertGreaterEqual(task.get_reward(env.physics), task.failure_reward_rate)

    def test_rejects_invalid_physical_scales(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                swingup_xk(plant_scales={"mass1": value})


class TestGeneralizationProtocol(unittest.TestCase):
    def test_manifest_is_reproducible_disjoint_and_has_true_extrapolation(self):
        manifest, _ = read_manifest(DEFAULT_MANIFEST)
        self.assertEqual(manifest, make_manifest())
        rows = [row for split in manifest["splits"].values() for row in split]
        self.assertEqual(len(rows), len({tuple(row["scales"][key] for key in
                                            ("mass1", "mass2", "length1", "length2"))
                                        for row in rows}))
        for split in ("train", "validation", "interpolation"):
            for row in manifest["splits"][split]:
                self.assertTrue(all(0.8 <= x <= 1.2 for x in row["scales"].values()))
        for row in manifest["splits"]["extrapolation"]:
            self.assertEqual(sum(x < 0.8 or x > 1.2 for x in row["scales"].values()), 1)

    def test_context_is_constant_per_episode_and_present_on_both_transition_ends(self):
        configs = make_manifest(train_count=3)["splits"]["train"]
        plain = RandomizedAcrobotEnv(configs, seed=5, dt=0.01, physics_dt=0.001,
                                    episode_duration=0.04, task_kwargs={"release_start": True})
        context = RandomizedAcrobotEnv(configs, seed=5, conditioned=True,
                                      dt=0.01, physics_dt=0.001, episode_duration=0.04,
                                      task_kwargs={"release_start": True})
        self.addCleanup(plain.close)
        self.addCleanup(context.close)
        selected = set()
        for _ in range(8):
            obs, info = plain.reset()
            augmented, other = context.reset()
            selected.add(info["configuration_id"])
            self.assertEqual(info, other)
            np.testing.assert_array_equal(obs, augmented[:4])
            expected = parameter_context(context.current_configuration["scales"])
            np.testing.assert_array_equal(augmented[4:], expected)
            before, _, _, _, after, *_ = context.step_dt(np.array([0.1]))
            np.testing.assert_array_equal(before[4:], expected)
            np.testing.assert_array_equal(after[4:], expected)
            self.assertEqual(context.current_configuration["id"], info["configuration_id"])
        self.assertGreater(len(selected), 1)
        first, first_info = context.reset(seed=92)
        again, again_info = context.reset(seed=92)
        np.testing.assert_array_equal(first, again)
        self.assertEqual(first_info, again_info)

    def test_demonstrator_tracks_live_plant_after_resets(self):
        configs = make_manifest(train_count=3)["splits"]["train"]
        env = RandomizedAcrobotEnv(configs, conditioned=True, seed=2,
                                  dt=0.01, physics_dt=0.001, episode_duration=0.04)
        self.addCleanup(env.close)
        demonstrator = ConfigurationDemonstrator(env)
        for _ in range(6):
            obs, _ = env.reset()
            action = demonstrator(obs)
            self.assertTrue(env.action_space.contains(action.astype(np.float32)))
            self.assertEqual(demonstrator.controller.params,
                             AcrobotParams.from_physics(env._env.physics))


if __name__ == "__main__":
    unittest.main()
