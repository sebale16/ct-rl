import unittest

import numpy as np

from common.utils import (
    load_ct_hyperparams_from_table,
    load_sb3_hyperparams_from_table,
)

try:
    from environment.dmc import DMCContinuousEnv
    from environment.tip_curriculum import (
        FRACTION_CURRICULUM_ENV_IDS,
        PERFORMANCE_CURRICULUM_ENV_IDS,
    )
    from evaluations.sustained_capture import strict_capture_spec_for

    HAVE_DMC = True
except Exception:  # pragma: no cover - dependency-limited installations
    HAVE_DMC = False


@unittest.skipUnless(HAVE_DMC, "dm_control not available")
class TestTipCurriculumRegistration(unittest.TestCase):
    def _env(self, domain: str, task: str, *, curriculum: bool = True):
        env = DMCContinuousEnv(
            domain_name=domain,
            task_name=task,
            seed=0,
            time_sampling="uniform",
            dt=0.01,
            physics_dt=0.002,
            episode_duration=0.05,
            task_kwargs={"curriculum": curriculum},
        )
        self.addCleanup(env.close)
        return env

    def test_all_new_ids_register_as_performance_curricula(self):
        cases = (
            ("acrobot", "swingup-v4.3"),
            ("acrobot", "swingup-v6.1"),
            ("cartpole", "two_poles-v2"),
        )
        for domain, task in cases:
            with self.subTest(domain=domain, task=task):
                env = self._env(domain, task)
                _, info = env.reset(seed=123)
                self.assertTrue(env.has_curriculum)
                self.assertEqual(env.curriculum_kind, "performance")
                self.assertEqual(env.curriculum_stage, 0)
                self.assertEqual(env.num_curriculum_stages, 6)
                self.assertEqual(info[f"{domain}_curriculum_stage"], 0.0)
                self.assertEqual(
                    info[f"{domain}_curriculum_num_stages"], 6.0
                )
                metrics = env.curriculum_log_metrics()
                self.assertEqual(metrics["stage"], 0.0)
                self.assertEqual(metrics["num_stages"], 6.0)
                self.assertEqual(metrics["progress"], 0.0)
                self.assertEqual(metrics["start_tip_height_norm"], 1.0)
                self.assertEqual(
                    metrics["start_potential_energy_norm"], 1.0
                )
                self.assertEqual(metrics["start_tip_speed"], 0.0)
                self.assertIn(f"{domain}_strict_capture", info)
                self.assertEqual(info[f"{domain}_strict_capture"], 1.0)

    def test_wrapper_stage_forwarding_reaches_exact_hanging_final_reset(self):
        for domain, task in (
            ("acrobot", "swingup-v4.3"),
            ("acrobot", "swingup-v6.1"),
            ("cartpole", "two_poles-v2"),
        ):
            with self.subTest(domain=domain, task=task):
                env = self._env(domain, task)
                env.set_curriculum_stage(10_000)
                _, info = env.reset()
                np.testing.assert_array_equal(
                    env._env.physics.data.qvel,
                    np.zeros(env._env.physics.model.nv),
                )
                self.assertEqual(env.curriculum_stage, 5)
                expected_height = -1.0 if domain == "cartpole" else 0.0
                self.assertAlmostEqual(
                    info[f"{domain}_tip_height"], expected_height
                )

    def test_disabled_primary_eval_is_fixed_final_task(self):
        for domain, task in (
            ("acrobot", "swingup-v4.3"),
            ("acrobot", "swingup-v6.1"),
            ("cartpole", "two_poles-v2"),
        ):
            with self.subTest(domain=domain, task=task):
                env = self._env(domain, task, curriculum=False)
                _, info = env.reset()
                self.assertFalse(env.has_curriculum)
                self.assertIsNone(env.curriculum_kind)
                self.assertEqual(env.curriculum_stage, 5)
                self.assertEqual(
                    info[f"{domain}_curriculum_start_tip_speed"], 0.0
                )
                expected_height = -1.0 if domain == "cartpole" else 0.0
                self.assertAlmostEqual(
                    info[f"{domain}_tip_height"], expected_height
                )

    def test_curriculum_id_sets_do_not_overlap(self):
        self.assertFalse(
            FRACTION_CURRICULUM_ENV_IDS & PERFORMANCE_CURRICULUM_ENV_IDS
        )
        self.assertEqual(
            PERFORMANCE_CURRICULUM_ENV_IDS,
            {
                "acrobot-swingup-v4.3",
                "acrobot-swingup-v6.1",
                "cartpole-two_poles-v2",
            },
        )

    def test_fraction_curriculum_logs_schedule_and_reset_band(self):
        for domain, task in (
            ("acrobot", "swingup-v4.2"),
            ("acrobot", "swingup-v6"),
            ("cartpole", "two_poles-curriculum"),
        ):
            with self.subTest(domain=domain, task=task):
                env = self._env(domain, task)
                env.set_curriculum_fraction(0.5)
                metrics = env.curriculum_log_metrics()
                self.assertEqual(metrics["fraction"], 0.5)
                self.assertEqual(metrics["progress"], 0.5)
                self.assertEqual(metrics["complete"], 0.0)
                self.assertAlmostEqual(
                    metrics["angle_spread_rad"],
                    0.5 + 0.5 * (np.pi - 0.5),
                )

    def test_capture_registry_uses_each_tasks_tip_signal(self):
        for env_id in (
            "acrobot-swingup-v4.3",
            "acrobot-swingup-v6.1",
        ):
            spec = strict_capture_spec_for(
                algorithm="ct_sac", env_id=env_id
            )
            self.assertEqual(spec.info_key, "acrobot_strict_capture")
        cartpole_spec = strict_capture_spec_for(
            algorithm="ct_sac", env_id="cartpole-two_poles-v2"
        )
        self.assertEqual(
            cartpole_spec.info_key, "cartpole_strict_capture"
        )


class TestTipCurriculumHyperparameterCoverage(unittest.TestCase):
    def test_new_tasks_match_v42_algorithm_coverage(self):
        loaders = {
            "ct_sac": load_ct_hyperparams_from_table,
            "ct_td3": load_ct_hyperparams_from_table,
            "ppo": load_sb3_hyperparams_from_table,
            "sac": load_sb3_hyperparams_from_table,
        }
        for env_id in (
            "acrobot-swingup-v4.3",
            "acrobot-swingup-v6.1",
            "cartpole-two_poles-v2",
        ):
            for algorithm, loader in loaders.items():
                with self.subTest(env_id=env_id, algorithm=algorithm):
                    total_steps, _, _, _, _ = loader(
                        algorithm, env_id, "final_mf"
                    )
                    self.assertGreater(total_steps, 0)

    def test_cartpole_modes_mirror_existing_two_pole_task(self):
        loaders = {
            "ct_sac": load_ct_hyperparams_from_table,
            "ct_td3": load_ct_hyperparams_from_table,
            "ppo": load_sb3_hyperparams_from_table,
            "sac": load_sb3_hyperparams_from_table,
        }
        for algorithm, loader in loaders.items():
            with self.subTest(algorithm=algorithm):
                existing = loader(
                    algorithm,
                    "cartpole-two_poles-curriculum",
                    "final_mf",
                )
                performance_gated = loader(
                    algorithm, "cartpole-two_poles-v2", "final_mf"
                )
                # The SB3 loader exposes the CSV's ``env_id`` column as
                # metadata key ``id``; that identifier is the one field these
                # otherwise identical rows must differ on.
                existing[1].pop("id", None)
                performance_gated[1].pop("id", None)
                self.assertEqual(performance_gated, existing)


if __name__ == "__main__":
    unittest.main()
