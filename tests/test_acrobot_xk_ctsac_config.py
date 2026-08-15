"""Contract tests for the Acrobot-XK CT-SAC reward-arm matrix."""

import csv
import math
import unittest
from collections import Counter
from pathlib import Path

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

PREVIOUS_ELBOW_RATE_LIMIT = 4.0 * math.pi
NEW_ELBOW_RATE_LIMIT = PREVIOUS_ELBOW_RATE_LIMIT * math.sqrt(2.0)
LATEST_HIGH_RATE_MODES = {
    f"{mode}_q2dot4sqrt2pi": expected
    for mode, expected in FIXED_ONE_MS_TEMP1_MODES.items()
}
XK_CLOSED_LOOP_MODES = {}
for cap_label, cap in (
    ("q2dot4pi", PREVIOUS_ELBOW_RATE_LIMIT),
    ("q2dot4sqrt2pi", NEW_ELBOW_RATE_LIMIT),
):
    for mode, (reward_kind, eta) in FIXED_ONE_MS_TEMP1_MODES.items():
        if reward_kind == "r0":
            continue
        discount_rate = 0.1 if "_h10s_" in mode else 0.5
        XK_CLOSED_LOOP_MODES[f"{mode}_xkdot_{cap_label}"] = (
            reward_kind,
            eta,
            discount_rate,
            cap,
        )
    for horizon, eta_label, eta, discount_rate in (
        ("h10s", "0p35", 0.35, 0.1),
        ("h2s", "0p31", 0.31, 0.5),
    ):
        mode = (
            f"xk_r3_eta{eta_label}_fixed1ms_{horizon}_temp1_"
            f"xkdot_{cap_label}"
        )
        XK_CLOSED_LOOP_MODES[mode] = ("r3", eta, discount_rate, cap)

R0_BASE_ETA_VALUES = (
    ("0", 0.0),
    ("0p01", 0.01),
    ("0p03", 0.03),
    ("0p1", 0.1),
    ("0p3", 0.3),
    ("1", 1.0),
)
R0_BASE_MODES = {}
for horizon, discount_rate in (("h10s", 0.1), ("h2s", 0.5)):
    for cap_label, elbow_rate_limit in (
        ("q2dot4pi", PREVIOUS_ELBOW_RATE_LIMIT),
        ("q2dot4sqrt2pi", NEW_ELBOW_RATE_LIMIT),
    ):
        mode = f"xk_r0base_r1_fixed1ms_{horizon}_temp1_{cap_label}"
        R0_BASE_MODES[mode] = {
            "reward_kind": "r1",
            "eta": None,
            "rate_source": None,
            "discount_rate": discount_rate,
            "horizon": horizon,
            "cap_label": cap_label,
            "elbow_rate_limit": elbow_rate_limit,
        }
        for source_label, rate_source in (
            ("actualdot", "actual"),
            ("xkdot", "xk_closed_loop"),
        ):
            for reward_kind in ("r2", "r3"):
                for eta_label, eta in R0_BASE_ETA_VALUES:
                    mode = (
                        f"xk_r0base_{reward_kind}_eta{eta_label}_fixed1ms_"
                        f"{horizon}_temp1_{source_label}_{cap_label}"
                    )
                    R0_BASE_MODES[mode] = {
                        "reward_kind": reward_kind,
                        "eta": eta,
                        "rate_source": rate_source,
                        "discount_rate": discount_rate,
                        "horizon": horizon,
                        "cap_label": cap_label,
                        "elbow_rate_limit": elbow_rate_limit,
                    }


def _load(mode):
    return load_ct_hyperparams_from_table("ct_sac", ENV_ID, mode)


class TestAcrobotXKCTSACConfig(unittest.TestCase):
    def test_every_acrobot_xk_mode_evaluates_every_100k_steps(self):
        table = (
            Path(__file__).parents[1]
            / "benchmarks"
            / "hyperparams"
            / "ct_sac.csv"
        )
        with table.open(newline="") as handle:
            modes = [
                row
                for row in csv.DictReader(handle)
                if row["env_id"] == ENV_ID
            ]

        self.assertTrue(modes)
        self.assertEqual(
            len(modes),
            len({row["mode"] for row in modes}),
            "Acrobot-XK mode names must be unique",
        )
        self.assertEqual(
            {
                row["mode"]: row["log_eval_freq"]
                for row in modes
                if int(row["log_eval_freq"]) != 100_000
            },
            {},
        )

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

    def test_latest_temperature_one_arms_have_higher_rate_cap_variants(self):
        self.assertEqual(len(LATEST_HIGH_RATE_MODES), 28)
        for mode, expected in LATEST_HIGH_RATE_MODES.items():
            with self.subTest(mode=mode):
                total, env, model, algo, log = _load(mode)
                base_mode = mode.removesuffix("_q2dot4sqrt2pi")
                (
                    base_total,
                    base_env,
                    base_model,
                    base_algo,
                    base_log,
                ) = _load(base_mode)

                task = dict(env["task_kwargs"])
                self.assertEqual(
                    (task["reward_kind"], task.get("eta")), expected
                )
                self.assertAlmostEqual(
                    task.pop("elbow_angle_limit"),
                    PREVIOUS_ELBOW_RATE_LIMIT,
                )
                self.assertAlmostEqual(
                    task.pop("elbow_rate_limit"), NEW_ELBOW_RATE_LIMIT
                )
                self.assertEqual(task.pop("shoulder_rate_scale_limit"), 2)

                self.assertEqual(total, base_total)
                self.assertEqual(task, base_env["task_kwargs"])
                env = dict(env)
                base_env = dict(base_env)
                env.pop("task_kwargs")
                base_env.pop("task_kwargs")
                self.assertEqual(env, base_env)
                self.assertEqual(model, base_model)
                self.assertEqual(algo, base_algo)
                self.assertEqual(log, base_log)

    def test_xk_closed_loop_reward_matrix_covers_both_rate_caps(self):
        self.assertEqual(len(XK_CLOSED_LOOP_MODES), 56)
        counts = {"r1": 0, "r2": 0, "r3": 0}
        for mode, expected in XK_CLOSED_LOOP_MODES.items():
            reward_kind, eta, discount_rate, elbow_rate_limit = expected
            with self.subTest(mode=mode):
                total, env, model, algo, log = _load(mode)
                task = env["task_kwargs"]
                counts[reward_kind] += 1

                self.assertEqual(total, 1_000_000)
                self.assertEqual(task["reward_kind"], reward_kind)
                self.assertEqual(task.get("eta"), eta)
                self.assertEqual(
                    task["lyapunov_rate_source"], "xk_closed_loop"
                )
                self.assertEqual(task["k_v"], 66.3)
                self.assertAlmostEqual(
                    task["elbow_angle_limit"], PREVIOUS_ELBOW_RATE_LIMIT
                )
                self.assertAlmostEqual(
                    task["elbow_rate_limit"], elbow_rate_limit
                )
                self.assertEqual(task["shoulder_rate_scale_limit"], 2)
                self.assertTrue(task["release_start"])
                self.assertEqual(task["damping"], 0)
                self.assertEqual(task["torque_limit"], 20)
                self.assertNotIn("failure_reward_rate", task)
                if reward_kind == "r3":
                    self.assertEqual(task["discount_rate"], discount_rate)
                else:
                    self.assertNotIn("discount_rate", task)

                self.assertEqual(env["time_sampling"], "uniform")
                self.assertEqual(env["dt"], 0.001)
                self.assertEqual(env["physics_dt"], 0.001)
                self.assertEqual(env["max_steps"], 20_000)
                self.assertEqual(env["episode_duration"], 20)
                self.assertEqual(algo["discount_rate"], discount_rate)
                self.assertEqual(algo["target_reference_dt"], 0.001)
                self.assertEqual(algo["alpha"], "auto_1.0")
                self.assertEqual(str(algo["reward_is_rate"]).lower(), "true")
                self.assertEqual(model["periodic_obs_indices"], (0,))
                self.assertEqual(log["eval_freq"], 100_000)

        self.assertEqual(counts, {"r1": 4, "r2": 24, "r3": 28})

    def test_r0_q2dot4pi_baselines_match_named_discount_horizons(self):
        for horizon, discount_rate in (("h2s", 0.5), ("h10s", 0.1)):
            mode = f"xk_r0_fixed1ms_{horizon}_temp1_q2dot4pi"
            with self.subTest(mode=mode):
                _, env, _, algo, _ = _load(mode)
                task = env["task_kwargs"]
                self.assertEqual(task["reward_kind"], "r0")
                self.assertNotIn("discount_rate", task)
                self.assertEqual(algo["discount_rate"], discount_rate)
                self.assertAlmostEqual(
                    task["elbow_rate_limit"], PREVIOUS_ELBOW_RATE_LIMIT
                )

    def test_xk_closed_loop_r3_includes_eta_sweep_optima(self):
        for cap_label in ("q2dot4pi", "q2dot4sqrt2pi"):
            with self.subTest(cap=cap_label, horizon="h10s"):
                mode = (
                    "xk_r3_eta0p35_fixed1ms_h10s_temp1_xkdot_"
                    + cap_label
                )
                _, env, _, algo, _ = _load(mode)
                self.assertEqual(env["task_kwargs"]["eta"], 0.35)
                self.assertEqual(algo["discount_rate"], 0.1)
            with self.subTest(cap=cap_label, horizon="h2s"):
                mode = (
                    "xk_r3_eta0p31_fixed1ms_h2s_temp1_xkdot_"
                    + cap_label
                )
                _, env, _, algo, _ = _load(mode)
                self.assertEqual(env["task_kwargs"]["eta"], 0.31)
                self.assertEqual(algo["discount_rate"], 0.5)

    def test_r0_base_reward_matrix_contract(self):
        table = (
            Path(__file__).parents[1]
            / "benchmarks"
            / "hyperparams"
            / "ct_sac.csv"
        )
        with table.open(newline="") as handle:
            rows = [
                row
                for row in csv.DictReader(handle)
                if row["env_id"] == ENV_ID
                and row["mode"].startswith("xk_r0base_")
            ]

        self.assertEqual(len(R0_BASE_MODES), 100)
        self.assertEqual(len(rows), 100)
        self.assertEqual(
            {row["mode"] for row in rows}, set(R0_BASE_MODES)
        )
        self.assertEqual(
            Counter(
                expected["reward_kind"]
                for expected in R0_BASE_MODES.values()
            ),
            {"r1": 4, "r2": 48, "r3": 48},
        )
        self.assertEqual(
            Counter(
                expected["rate_source"]
                for expected in R0_BASE_MODES.values()
            ),
            {None: 4, "actual": 48, "xk_closed_loop": 48},
        )
        self.assertEqual(
            Counter(
                expected["horizon"]
                for expected in R0_BASE_MODES.values()
            ),
            {"h10s": 50, "h2s": 50},
        )
        self.assertEqual(
            Counter(
                expected["cap_label"]
                for expected in R0_BASE_MODES.values()
            ),
            {"q2dot4pi": 50, "q2dot4sqrt2pi": 50},
        )
        self.assertFalse(
            any(
                forbidden in mode
                for mode in R0_BASE_MODES
                for forbidden in ("eta0p35", "eta0p31")
            )
        )
        self.assertEqual(
            {
                expected["eta"]
                for expected in R0_BASE_MODES.values()
                if expected["eta"] is not None
            },
            {0.0, 0.01, 0.03, 0.1, 0.3, 1.0},
        )

        references = {}
        for mode, expected in R0_BASE_MODES.items():
            with self.subTest(mode=mode):
                total, env, model, algo, log = _load(mode)
                expected_task = {
                    "reward_kind": expected["reward_kind"],
                    "reward_base": "r0",
                    "release_start": True,
                    "damping": 0,
                    "torque_limit": 20,
                    "elbow_angle_limit": PREVIOUS_ELBOW_RATE_LIMIT,
                    "elbow_rate_limit": expected["elbow_rate_limit"],
                    "shoulder_rate_scale_limit": 2,
                }
                if expected["eta"] is not None:
                    expected_task["eta"] = expected["eta"]
                if expected["rate_source"] is not None:
                    expected_task["lyapunov_rate_source"] = expected[
                        "rate_source"
                    ]
                if expected["rate_source"] == "xk_closed_loop":
                    expected_task["k_v"] = 66.3
                if expected["reward_kind"] == "r3":
                    expected_task["discount_rate"] = expected[
                        "discount_rate"
                    ]

                self.assertEqual(total, 1_000_000)
                self.assertEqual(env["task_kwargs"], expected_task)
                self.assertEqual(env["time_sampling"], "uniform")
                self.assertEqual(env["dt"], 0.001)
                self.assertEqual(env["physics_dt"], 0.001)
                self.assertEqual(env["max_steps"], 20_000)
                self.assertEqual(env["episode_duration"], 20)
                self.assertNotIn("gamma", algo)
                self.assertEqual(
                    algo["discount_rate"], expected["discount_rate"]
                )
                self.assertEqual(algo["target_reference_dt"], 0.001)
                self.assertEqual(str(algo["reward_is_rate"]).lower(), "true")
                self.assertEqual(algo["alpha"], "auto_1.0")
                self.assertEqual(model["periodic_obs_indices"], (0,))
                self.assertEqual(log["eval_freq"], 100_000)

                common_env = dict(env)
                common_env.pop("task_kwargs")
                common = (common_env, model, algo, log)
                reference = references.setdefault(
                    expected["horizon"], common
                )
                self.assertEqual(common, reference)

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
