"""Contracts for the selected Acrobot-XK irregular-time benchmark arms."""

import math
import unittest

from common.utils import (
    load_ct_hyperparams_from_table,
    load_sb3_hyperparams_from_table,
)


ENV_ID = "acrobot-swingup-xk"
STEM = (
    "xk_r3_eta0p23_ctrl10ms_h2s_temp0p01_xkdot_q2dot4pi_"
    "logrecip"
)
CT_SAC_MODES = {
    f"{STEM}_xkklrev_xkdemo20k_tau1p25e2_anneal0span60k_irregular1m": {
        "demonstration_steps": 20_000,
        "imitation_coef": 1.0,
        "imitation_coef_final": None,
    },
    f"{STEM}_xkklrev_xkdemo20k_tau1p25e2_anneal0p5span60k_irregular1m": {
        "demonstration_steps": 20_000,
        "imitation_coef": 1.0,
        "imitation_coef_final": 0.5,
    },
    f"{STEM}_xkdemo20k_tau1p25e2_irregular1m": {
        "demonstration_steps": 20_000,
        "imitation_coef": None,
        "imitation_coef_final": None,
    },
    f"{STEM}_tau1p25e2_irregular1m": {
        "demonstration_steps": None,
        "imitation_coef": None,
        "imitation_coef_final": None,
    },
}
BASELINE_MODE = f"{STEM}_tau1p25e2_irregular1m"


def _assert_irregular_contract(case, total, env, log):
    case.assertEqual(total, 1_000_000)
    case.assertEqual(env["time_sampling"], "irregular")
    case.assertEqual(env["dt"], 0.01)
    case.assertEqual(env["physics_dt"], 0.001)
    case.assertEqual(env["min_dt"], 0.001)
    case.assertEqual(env["max_dt"], 0.03)
    case.assertEqual(env["max_steps"], 20_000)
    case.assertEqual(env["episode_duration"], 20)
    case.assertEqual(
        env["time_sampling_kwargs"],
        {"tail_p": 0.99, "tail_split": 0.9},
    )
    case.assertEqual(log["save_freq"], 100_000)
    case.assertEqual(log["eval_freq"], 100_000)


class TestAcrobotXKIrregularBenchmarkModes(unittest.TestCase):
    def test_selected_ct_sac_quartet_is_model_free_and_runs_for_one_million(self):
        for mode, expected in CT_SAC_MODES.items():
            with self.subTest(mode=mode):
                total, env, model, algo, log = load_ct_hyperparams_from_table(
                    "ct_sac", ENV_ID, mode
                )
                _assert_irregular_contract(self, total, env, log)
                self.assertNotIn("use_model_based_q", algo)
                self.assertEqual(model["periodic_obs_indices"], (0,))
                self.assertEqual(algo["discount_rate"], 0.5)
                self.assertEqual(algo["target_reference_dt"], 0.01)
                self.assertEqual(algo["tau"], 0.0125)
                self.assertEqual(
                    algo.get("demonstration_steps"),
                    expected["demonstration_steps"],
                )
                self.assertEqual(
                    algo.get("imitation_coef"), expected["imitation_coef"]
                )
                self.assertEqual(
                    algo.get("imitation_coef_final"),
                    expected["imitation_coef_final"],
                )

    def test_sac_ct_td3_and_td3_share_the_pure_rl_environment(self):
        loaders = {
            "sac": load_sb3_hyperparams_from_table,
            "ct_td3": load_ct_hyperparams_from_table,
            "td3": load_sb3_hyperparams_from_table,
        }
        expected_task = None
        expected_gamma = math.exp(-0.5 * 0.01)
        for algorithm, loader in loaders.items():
            with self.subTest(algorithm=algorithm):
                total, env, _, algo, log = loader(
                    algorithm, ENV_ID, BASELINE_MODE
                )
                _assert_irregular_contract(self, total, env, log)
                task = env["task_kwargs"]
                if expected_task is None:
                    expected_task = task
                else:
                    self.assertEqual(task, expected_task)
                self.assertEqual(task["reward_kind"], "r3")
                self.assertEqual(task["reward_transform"], "log_reciprocal")
                self.assertEqual(task["discount_rate"], 0.5)
                self.assertAlmostEqual(algo["gamma"], expected_gamma, places=15)
                self.assertEqual(algo["tau"], 0.0125)
                self.assertEqual(algo["learning_starts"], 10_000)

    def test_sac_log_std_is_a_policy_setting(self):
        _, _, policy, algo, _ = load_sb3_hyperparams_from_table(
            "sac", ENV_ID, BASELINE_MODE
        )
        self.assertEqual(policy["log_std_init"], -1)
        self.assertNotIn("log_std_init", algo)

    def test_td3_policy_delay_reaches_the_algorithm_kwargs(self):
        _, _, _, algo, _ = load_sb3_hyperparams_from_table(
            "td3", ENV_ID, BASELINE_MODE
        )
        self.assertEqual(algo["policy_delay"], 2)


if __name__ == "__main__":
    unittest.main()
