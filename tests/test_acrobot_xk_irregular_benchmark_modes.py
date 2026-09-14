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
#: Target-network Polyak rate per gradient step, by mode-name suffix.  The
#: design constraint is a fixed target lag in PHYSICAL time,
#: ``T_target = E[dt] / (tau * updates_per_env_step)``, with
#: ``updates_per_env_step = gradient_steps / train_freq = 1`` for every arm
#: here.  ``tau1p25e2`` was chosen for a REGULAR 10 ms step; the irregular
#: sampler's realized mean interval is ~4 ms, so those arms actually run a
#: 0.32 s lag.  ``tau5e3`` restores 0.80 s at the realized mean.
TAU_BY_SUFFIX = {"tau1p25e2": 0.0125, "tau5e3": 0.005}
TARGET_LAG_SECONDS = 0.80

#: Mean interval of the ``two_tail_uniform`` sampler this env configures:
#: ``tail_p`` of the mass on the endpoints, ``tail_split`` of that on
#: ``min_dt``, the remainder uniform on the interior physics_dt grid.
SAMPLER_MEAN_DT = 0.99 * (0.9 * 0.001 + 0.1 * 0.030) + 0.01 * 0.0155


def _ct_sac_arms(tau_suffix):
    """The four CT-SAC arms at one tau setting, keyed by mode name."""
    return {
        f"{STEM}_xkklrev_xkdemo20k_{tau_suffix}_anneal0span60k_irregular1m": {
            "demonstration_steps": 20_000,
            "imitation_coef": 1.0,
            "imitation_coef_final": None,
        },
        f"{STEM}_xkklrev_xkdemo20k_{tau_suffix}_anneal0p5span60k_irregular1m": {
            "demonstration_steps": 20_000,
            "imitation_coef": 1.0,
            "imitation_coef_final": 0.5,
        },
        f"{STEM}_xkdemo20k_{tau_suffix}_irregular1m": {
            "demonstration_steps": 20_000,
            "imitation_coef": None,
            "imitation_coef_final": None,
        },
        f"{STEM}_{tau_suffix}_irregular1m": {
            "demonstration_steps": None,
            "imitation_coef": None,
            "imitation_coef_final": None,
        },
    }


CT_SAC_MODES = {t: _ct_sac_arms(t) for t in TAU_BY_SUFFIX}
BASELINE_MODES = {t: f"{STEM}_{t}_irregular1m" for t in TAU_BY_SUFFIX}
DEMO_MODES = {t: f"{STEM}_xkdemo20k_{t}_irregular1m" for t in TAU_BY_SUFFIX}
BASELINE_MODE = BASELINE_MODES["tau1p25e2"]
DEMO_MODE = DEMO_MODES["tau1p25e2"]


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
        for tau_suffix, arms in CT_SAC_MODES.items():
          for mode, expected in arms.items():
            with self.subTest(mode=mode):
                total, env, model, algo, log = load_ct_hyperparams_from_table(
                    "ct_sac", ENV_ID, mode
                )
                _assert_irregular_contract(self, total, env, log)
                self.assertNotIn("use_model_based_q", algo)
                self.assertEqual(model["periodic_obs_indices"], (0,))
                self.assertEqual(algo["discount_rate"], 0.5)
                self.assertEqual(algo["target_reference_dt"], 0.01)
                self.assertEqual(algo["tau"], TAU_BY_SUFFIX[tau_suffix])
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
        for tau_suffix, baseline in BASELINE_MODES.items():
          for algorithm, loader in loaders.items():
            # Only the CT algorithms have a physical-time target lag to
            # correct, so tau5e3 rows exist for ct_td3 alone.
            if tau_suffix != "tau1p25e2" and algorithm != "ct_td3":
                continue
            with self.subTest(algorithm=algorithm, tau=tau_suffix):
                total, env, _, algo, log = loader(
                    algorithm, ENV_ID, baseline
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
                if algorithm == "ct_td3":
                    self.assertEqual(algo["tau"], TAU_BY_SUFFIX[tau_suffix])
                self.assertEqual(algo["learning_starts"], 10_000)

    def test_sac_ct_td3_and_td3_baseline_uses_the_raw_state_observation(self):
        # xin_kaneda (and every other Acrobot-XK demonstration controller)
        # reads the raw [q1, q2, qdot1, qdot2] state; the demo-seeded arm
        # below and its baseline must therefore share raw_state_obs=True,
        # matching the ct_sac quartet's own env_raw_state_obs=True.
        loaders = {
            "sac": load_sb3_hyperparams_from_table,
            "ct_td3": load_ct_hyperparams_from_table,
            "td3": load_sb3_hyperparams_from_table,
        }
        for algorithm, loader in loaders.items():
            with self.subTest(algorithm=algorithm):
                _, env, _, _, _ = loader(algorithm, ENV_ID, BASELINE_MODE)
                self.assertEqual(str(env["raw_state_obs"]).lower(), "true")

    def test_sac_ct_td3_and_td3_demo_arm_seeds_from_xin_kaneda(self):
        loaders = {
            "sac": load_sb3_hyperparams_from_table,
            "ct_td3": load_ct_hyperparams_from_table,
            "td3": load_sb3_hyperparams_from_table,
        }
        for algorithm, loader in loaders.items():
            with self.subTest(algorithm=algorithm):
                total, env, _, algo, log = loader(algorithm, ENV_ID, DEMO_MODE)
                _assert_irregular_contract(self, total, env, log)
                self.assertEqual(str(env["raw_state_obs"]).lower(), "true")
                self.assertEqual(algo["demonstration_controller"], "xin_kaneda")
                self.assertEqual(algo["demonstration_steps"], 20_000)

    def test_sac_log_std_is_a_policy_setting(self):
        _, _, policy, algo, _ = load_sb3_hyperparams_from_table(
            "sac", ENV_ID, BASELINE_MODE
        )
        self.assertEqual(policy["log_std_init"], -1)
        self.assertNotIn("log_std_init", algo)

    def test_tau5e3_arms_restore_the_designed_target_lag(self):
        # tau was picked for a regular 10 ms step.  Under this sampler the
        # realized mean interval is ~4 ms, so the tau1p25e2 arms run a target
        # network ~2.5x faster in physical time than intended; tau5e3 puts it
        # back at 0.80 s.  sac/td3 are excluded: their targets are discrete,
        # with no physical-time design point.
        self.assertAlmostEqual(SAMPLER_MEAN_DT, 0.004016, places=6)
        for algorithm in ("ct_sac", "ct_td3"):
            for tau_suffix, baseline in BASELINE_MODES.items():
                with self.subTest(algorithm=algorithm, tau=tau_suffix):
                    _, _, _, algo, _ = load_ct_hyperparams_from_table(
                        algorithm, ENV_ID, baseline
                    )
                    updates_per_env_step = (
                        algo["gradient_steps"] / algo["train_freq"]
                    )
                    self.assertEqual(updates_per_env_step, 1)
                    lag = SAMPLER_MEAN_DT / (algo["tau"] * updates_per_env_step)
                    if tau_suffix == "tau5e3":
                        self.assertAlmostEqual(lag, TARGET_LAG_SECONDS, places=2)
                    else:
                        self.assertLess(lag, 0.5 * TARGET_LAG_SECONDS)

    def test_td3_policy_delay_reaches_the_algorithm_kwargs(self):
        _, _, _, algo, _ = load_sb3_hyperparams_from_table(
            "td3", ENV_ID, BASELINE_MODE
        )
        self.assertEqual(algo["policy_delay"], 2)


if __name__ == "__main__":
    unittest.main()
