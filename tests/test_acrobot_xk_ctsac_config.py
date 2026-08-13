"""Contract tests for the Acrobot-XK CT-SAC reward-arm matrix."""

import unittest

from common.utils import load_ct_hyperparams_from_table


ENV_ID = "acrobot-swingup-xk"
ETA_MODES = {
    "xk_r2_eta0": 0.0,
    "xk_r2_eta0p01": 0.01,
    "xk_r2_eta0p03": 0.03,
    "xk_r2_eta0p1": 0.1,
    "xk_r2_eta0p3": 0.3,
    "xk_r2_eta1": 1.0,
}
REWARD_MODES = {"xk_r0": ("r0", None), "xk_r1": ("r1", None)}
REWARD_MODES.update({mode: ("r2", eta) for mode, eta in ETA_MODES.items()})
FIXED_HALF_MS_MODES = {
    f"{mode}_fixed0p5ms": expected for mode, expected in REWARD_MODES.items()
}
FIXED_ONE_MS_H10_MODES = {
    f"{mode}_fixed1ms_h10s": expected
    for mode, expected in REWARD_MODES.items()
}
FIXED_ONE_MS_H10_MODES.update(
    {
        f"{mode.replace('xk_r2', 'xk_r3')}_fixed1ms_h10s": ("r3", eta)
        for mode, eta in ETA_MODES.items()
    }
)
FIXED_ONE_MS_H2_MODES = {
    mode.replace("_h10s", "_h2s"): expected
    for mode, expected in FIXED_ONE_MS_H10_MODES.items()
}
FIXED_ONE_MS_TEMP1_MODES = {
    f"{mode}_temp1": expected
    for mode, expected in (
        FIXED_ONE_MS_H10_MODES | FIXED_ONE_MS_H2_MODES
    ).items()
}


def _load(mode):
    return load_ct_hyperparams_from_table("ct_sac", ENV_ID, mode)


class TestAcrobotXKCTSACConfig(unittest.TestCase):
    def test_training_arms_differ_only_in_reward_task_parameters(self):
        reference = None
        for mode, (reward_kind, eta) in REWARD_MODES.items():
            with self.subTest(mode=mode):
                total, env, model, algo, log = _load(mode)
                self.assertEqual(total, 1_000_000)
                task = env.pop("task_kwargs")
                self.assertEqual(task["reward_kind"], reward_kind)
                self.assertEqual(task.get("eta"), eta)
                self.assertTrue(task["release_start"])
                self.assertEqual(task["damping"], 0)
                self.assertEqual(task["torque_limit"], 64)
                self.assertNotIn("failure_reward_rate", task)
                self.assertEqual(model["periodic_obs_indices"], (0,))

                common = (env, model, algo, log)
                if reference is None:
                    reference = common
                else:
                    self.assertEqual(common, reference)

    def test_fixed_half_millisecond_training_arms(self):
        reference = None
        for mode, (reward_kind, eta) in FIXED_HALF_MS_MODES.items():
            with self.subTest(mode=mode):
                total, env, model, algo, log = _load(mode)
                self.assertEqual(total, 1_000_000)
                task = env.pop("task_kwargs")
                self.assertEqual(task["reward_kind"], reward_kind)
                self.assertEqual(task.get("eta"), eta)
                self.assertTrue(task["release_start"])
                self.assertEqual(task["damping"], 0)
                self.assertEqual(task["torque_limit"], 64)
                self.assertNotIn("failure_reward_rate", task)

                self.assertEqual(env["time_sampling"], "uniform")
                self.assertEqual(env["dt"], 0.0005)
                self.assertEqual(env["physics_dt"], 0.0005)
                self.assertEqual(env["episode_duration"], 20)
                self.assertEqual(env["max_steps"], 40_000)
                self.assertNotIn("min_dt", env)
                self.assertNotIn("max_dt", env)
                self.assertNotIn("time_sampling_kwargs", env)
                self.assertEqual(model["periodic_obs_indices"], (0,))

                common = (env, model, algo, log)
                if reference is None:
                    reference = common
                else:
                    self.assertEqual(common, reference)

    def test_fixed_arms_retain_the_irregular_arm_contract(self):
        timing_keys = {
            "time_sampling",
            "dt",
            "physics_dt",
            "min_dt",
            "max_dt",
            "time_sampling_kwargs",
        }
        for fixed_mode in FIXED_HALF_MS_MODES:
            irregular_mode = fixed_mode.removesuffix("_fixed0p5ms")
            with self.subTest(fixed_mode=fixed_mode):
                fixed_total, fixed_env, fixed_model, fixed_algo, fixed_log = _load(
                    fixed_mode
                )
                (
                    irregular_total,
                    irregular_env,
                    irregular_model,
                    irregular_algo,
                    irregular_log,
                ) = _load(irregular_mode)

                self.assertEqual(fixed_total, irregular_total)
                self.assertEqual(
                    fixed_env.pop("task_kwargs"), irregular_env.pop("task_kwargs")
                )
                for key in timing_keys:
                    fixed_env.pop(key, None)
                    irregular_env.pop(key, None)
                self.assertEqual(fixed_env, irregular_env)
                self.assertEqual(fixed_model, irregular_model)
                self.assertEqual(fixed_algo, irregular_algo)
                self.assertEqual(fixed_log, irregular_log)

    def test_corrected_one_millisecond_arms_use_physical_ten_second_discount(self):
        reference = None
        for mode, (reward_kind, eta) in FIXED_ONE_MS_H10_MODES.items():
            with self.subTest(mode=mode):
                total, env, model, algo, log = _load(mode)
                self.assertEqual(total, 1_000_000)
                task = env.pop("task_kwargs")
                self.assertEqual(task["reward_kind"], reward_kind)
                self.assertEqual(task.get("eta"), eta)
                self.assertTrue(task["release_start"])
                self.assertEqual(task["damping"], 0)
                self.assertEqual(task["torque_limit"], 20)
                self.assertNotIn("failure_reward_rate", task)
                if reward_kind == "r3":
                    self.assertEqual(task["discount_rate"], 0.1)
                    self.assertEqual(
                        task["discount_rate"], algo["discount_rate"]
                    )
                else:
                    self.assertNotIn("discount_rate", task)

                self.assertEqual(env["time_sampling"], "uniform")
                self.assertEqual(env["dt"], 0.001)
                self.assertEqual(env["physics_dt"], 0.001)
                self.assertEqual(env["max_steps"], 20_000)
                self.assertEqual(env["episode_duration"], 20)
                self.assertNotIn("gamma", algo)
                self.assertEqual(algo["discount_rate"], 0.1)
                self.assertEqual(algo["target_reference_dt"], 0.001)
                self.assertEqual(str(algo["reward_is_rate"]).lower(), "true")
                self.assertEqual(algo["alpha"], "auto_0.001")

                common = (env, model, algo, log)
                if reference is None:
                    reference = common
                else:
                    self.assertEqual(common, reference)

    def test_one_millisecond_arms_include_physical_two_second_discount(self):
        for mode, (reward_kind, eta) in FIXED_ONE_MS_H2_MODES.items():
            with self.subTest(mode=mode):
                total, env, model, algo, log = _load(mode)
                task = env["task_kwargs"]

                self.assertEqual(total, 1_000_000)
                self.assertEqual(task["reward_kind"], reward_kind)
                self.assertEqual(task.get("eta"), eta)
                self.assertTrue(task["release_start"])
                self.assertEqual(task["damping"], 0)
                self.assertEqual(task["torque_limit"], 20)
                self.assertNotIn("failure_reward_rate", task)
                if reward_kind == "r3":
                    self.assertEqual(task["discount_rate"], 0.5)
                    self.assertEqual(
                        task["discount_rate"], algo["discount_rate"]
                    )
                else:
                    self.assertNotIn("discount_rate", task)

                self.assertEqual(env["time_sampling"], "uniform")
                self.assertEqual(env["dt"], 0.001)
                self.assertEqual(env["physics_dt"], 0.001)
                self.assertEqual(env["max_steps"], 20_000)
                self.assertEqual(env["episode_duration"], 20)
                self.assertNotIn("gamma", algo)
                self.assertEqual(algo["discount_rate"], 0.5)
                self.assertEqual(algo["target_reference_dt"], 0.001)
                self.assertEqual(str(algo["reward_is_rate"]).lower(), "true")
                self.assertEqual(algo["alpha"], "auto_0.001")

                h10_mode = mode.replace("_h2s", "_h10s")
                h10_total, h10_env, h10_model, h10_algo, h10_log = _load(
                    h10_mode
                )
                h2_env = dict(env)
                h10_env = dict(h10_env)
                h2_task = dict(h2_env.pop("task_kwargs"))
                h10_task = dict(h10_env.pop("task_kwargs"))
                h2_task.pop("discount_rate", None)
                h10_task.pop("discount_rate", None)
                h2_algo = dict(algo)
                h10_algo = dict(h10_algo)
                h2_algo.pop("discount_rate")
                h10_algo.pop("discount_rate")

                self.assertEqual(total, h10_total)
                self.assertEqual(h2_env, h10_env)
                self.assertEqual(h2_task, h10_task)
                self.assertEqual(model, h10_model)
                self.assertEqual(h2_algo, h10_algo)
                self.assertEqual(log, h10_log)

    def test_corrected_arms_include_temperature_one_variants(self):
        self.assertEqual(len(FIXED_ONE_MS_TEMP1_MODES), 28)
        for mode, expected in FIXED_ONE_MS_TEMP1_MODES.items():
            with self.subTest(mode=mode):
                total, env, model, algo, log = _load(mode)
                base_mode = mode.removesuffix("_temp1")
                (
                    base_total,
                    base_env,
                    base_model,
                    base_algo,
                    base_log,
                ) = _load(base_mode)

                task = env["task_kwargs"]
                self.assertEqual(
                    (task["reward_kind"], task.get("eta")), expected
                )
                self.assertEqual(algo["alpha"], "auto_1.0")
                self.assertEqual(base_algo["alpha"], "auto_0.001")

                algo = dict(algo)
                base_algo = dict(base_algo)
                algo.pop("alpha")
                base_algo.pop("alpha")
                self.assertEqual(total, base_total)
                self.assertEqual(env, base_env)
                self.assertEqual(model, base_model)
                self.assertEqual(algo, base_algo)
                self.assertEqual(log, base_log)

    def test_training_and_evaluation_timing_contracts(self):
        _, train, train_model, train_algo, train_log = _load("xk_r0")
        self.assertEqual(train["time_sampling"], "irregular")
        self.assertEqual(train["dt"], 0.01)
        self.assertEqual(train["physics_dt"], 0.0005)
        self.assertEqual(train["min_dt"], 0.0005)
        self.assertEqual(train["max_dt"], 0.03)
        self.assertEqual(train["episode_duration"], 20)
        self.assertEqual(train["max_steps"], 40_000)

        total, evaluation, model, algo, log = _load("xk_eval")
        self.assertEqual(total, 1_000_000)
        self.assertEqual(evaluation["time_sampling"], "uniform")
        self.assertEqual(evaluation["dt"], 0.001)
        self.assertEqual(evaluation["physics_dt"], 0.001)
        self.assertEqual(evaluation["episode_duration"], 20)
        self.assertEqual(evaluation["max_steps"], 20_000)
        self.assertNotIn("min_dt", evaluation)
        self.assertNotIn("max_dt", evaluation)
        self.assertEqual(
            evaluation["task_kwargs"],
            {
                "reward_kind": "r0",
                "release_start": True,
                "damping": 0,
                "torque_limit": 20,
            },
        )
        self.assertEqual(model, train_model)
        self.assertEqual(algo, train_algo)
        self.assertEqual(log["interval"], train_log["interval"])
        self.assertEqual(log["save_freq"], train_log["save_freq"])
        self.assertEqual(log["eval_freq"], 100_000)


if __name__ == "__main__":
    unittest.main()
