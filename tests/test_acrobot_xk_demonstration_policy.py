"""``_build_demonstration_policy``: CT-SAC's analytical-controller warm start
for acrobot-swingup-xk (see algorithms.ct_sac.CTSAC's demonstration_policy)."""

import unittest

import numpy as np

from benchmarks.run_ct_rl import ACROBOT_XK_ENV_ID, _build_demonstration_policy
from controllers.xin_kaneda import XinKanedaController
from environment.dmc import DMCContinuousEnv
from environment.monitor import Monitor


def _acrobot_env(*, raw_state_obs=True, **task_kwargs):
    return Monitor(
        DMCContinuousEnv(
            "acrobot",
            "swingup-xk",
            seed=20000,
            raw_state_obs=raw_state_obs,
            time_sampling="uniform",
            dt=0.001,
            physics_dt=0.001,
            max_steps=200,
            episode_duration=0.2,
            task_kwargs={
                "release_start": True,
                "damping": 0.0,
                "torque_limit": 20.0,
                **task_kwargs,
            },
        )
    )


class TestBuildDemonstrationPolicy(unittest.TestCase):
    def test_builds_a_working_xin_kaneda_controller_from_the_training_env(self):
        env = _acrobot_env(reward_kind="r0")
        policy = _build_demonstration_policy(
            algo="ct_sac",
            env_id=ACROBOT_XK_ENV_ID,
            env_kwargs={"raw_state_obs": True},
            train_env=env,
            controller_name="xin_kaneda",
        )
        self.assertIsInstance(policy, XinKanedaController)
        self.assertEqual(policy.torque_limit, 20.0)
        # Paper's own Section-7 gains, since this task_kwargs sets neither.
        self.assertAlmostEqual(policy.gains.k_v, 66.3)
        self.assertAlmostEqual(policy.gains.k_d, 35.8)
        self.assertAlmostEqual(policy.gains.k_p, 61.2)

        obs, _ = env.reset(seed=20000)
        action = policy(obs)
        self.assertEqual(action.shape, env.action_space.shape)
        self.assertTrue(np.all(action >= env.action_space.low))
        self.assertTrue(np.all(action <= env.action_space.high))

    def test_reads_gains_and_torque_limit_from_task_kwargs_when_set(self):
        env = _acrobot_env(
            reward_kind="r2",
            eta=0.1,
            lyapunov_rate_source="xk_closed_loop",
            k_v=70.0,
            k_d=40.0,
            k_p=65.0,
            torque_limit=15.0,
        )
        policy = _build_demonstration_policy(
            algo="ct_sac",
            env_id=ACROBOT_XK_ENV_ID,
            env_kwargs={
                "raw_state_obs": True,
                "task_kwargs": {
                    "k_v": 70.0,
                    "k_d": 40.0,
                    "k_p": 65.0,
                    "torque_limit": 15.0,
                },
            },
            train_env=env,
            controller_name="xin_kaneda",
        )
        self.assertEqual(policy.torque_limit, 15.0)
        self.assertAlmostEqual(policy.gains.k_v, 70.0)
        self.assertAlmostEqual(policy.gains.k_d, 40.0)
        self.assertAlmostEqual(policy.gains.k_p, 65.0)

    def test_rejects_non_ct_sac_algorithms(self):
        env = _acrobot_env(reward_kind="r0")
        with self.assertRaisesRegex(ValueError, "only wired for algo='ct_sac'"):
            _build_demonstration_policy(
                algo="ct_td3",
                env_id=ACROBOT_XK_ENV_ID,
                env_kwargs={"raw_state_obs": True},
                train_env=env,
                controller_name="xin_kaneda",
            )

    def test_rejects_unknown_controller_names(self):
        env = _acrobot_env(reward_kind="r0")
        with self.assertRaisesRegex(ValueError, "must be 'xin_kaneda'"):
            _build_demonstration_policy(
                algo="ct_sac",
                env_id=ACROBOT_XK_ENV_ID,
                env_kwargs={"raw_state_obs": True},
                train_env=env,
                controller_name="some_other_controller",
            )

    def test_rejects_non_acrobot_xk_envs(self):
        env = _acrobot_env(reward_kind="r0")
        with self.assertRaisesRegex(ValueError, "requires env_id="):
            _build_demonstration_policy(
                algo="ct_sac",
                env_id="cartpole-swingup",
                env_kwargs={"raw_state_obs": True},
                train_env=env,
                controller_name="xin_kaneda",
            )

    def test_rejects_envs_without_raw_state_obs(self):
        env = _acrobot_env(raw_state_obs=False, reward_kind="r0")
        with self.assertRaisesRegex(ValueError, "raw_state_obs=True"):
            _build_demonstration_policy(
                algo="ct_sac",
                env_id=ACROBOT_XK_ENV_ID,
                env_kwargs={"raw_state_obs": False},
                train_env=env,
                controller_name="xin_kaneda",
            )


if __name__ == "__main__":
    unittest.main()
