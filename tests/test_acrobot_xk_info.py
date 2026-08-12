import unittest
from types import SimpleNamespace

import numpy as np

try:
    from environment.dmc import DMCContinuousEnv

    HAVE_DMC = True
except ImportError:
    DMCContinuousEnv = None
    HAVE_DMC = False


class _XKTaskStub:
    last_reward_terms = {
        "reward": -3.0,
        "r0": -1.25,
        "v_dot": 2.0,
        "reward_kind": "r2",
        "vector_debug_value": [1.0, 2.0],
    }

    def xk_diagnostic_terms(self, physics):
        del physics
        return {
            "energy_error_norm": -0.04,
            "elbow_norm": 0.01,
            "elbow_rate_norm": -0.02,
            "in_homoclinic_tube": 1.0,
        }


@unittest.skipUnless(HAVE_DMC, "dm_control / DMCContinuousEnv not available")
class TestAcrobotXKInfo(unittest.TestCase):
    @staticmethod
    def _wrapper_stub():
        wrapper = DMCContinuousEnv.__new__(DMCContinuousEnv)
        wrapper._env = SimpleNamespace(task=_XKTaskStub(), physics=object())
        return wrapper

    def test_reset_info_exposes_reward_independent_tube_diagnostics(self):
        info = self._wrapper_stub()._acrobot_reward_info(update=False)

        self.assertEqual(info["acrobot_xk_energy_error_norm"], -0.04)
        self.assertEqual(info["acrobot_xk_elbow_norm"], 0.01)
        self.assertEqual(info["acrobot_xk_elbow_rate_norm"], -0.02)
        self.assertEqual(info["acrobot_xk_in_homoclinic_tube"], 1.0)
        self.assertEqual(info["acrobot_xk_homoclinic_capture"], 1.0)
        self.assertNotIn("acrobot_xk_reward", info)

    def test_step_info_adds_only_scalar_reward_decomposition_fields(self):
        info = self._wrapper_stub()._acrobot_reward_info(update=True)

        self.assertEqual(info["acrobot_xk_reward"], -3.0)
        self.assertEqual(info["acrobot_xk_r0"], -1.25)
        self.assertEqual(info["acrobot_xk_v_dot"], 2.0)
        self.assertNotIn("acrobot_xk_reward_kind", info)
        self.assertNotIn("acrobot_xk_vector_debug_value", info)

    def test_real_xk_modes_publish_the_same_capture_contract(self):
        modes = (
            ("r0", None, None),
            ("r1", None, None),
            ("r2", 0.25, None),
            ("r3", 0.25, 0.1),
        )
        for reward_kind, eta, discount_rate in modes:
            with self.subTest(reward_kind=reward_kind):
                task_kwargs = {
                    "reward_kind": reward_kind,
                    "release_start": True,
                }
                if eta is not None:
                    task_kwargs["eta"] = eta
                if discount_rate is not None:
                    task_kwargs["discount_rate"] = discount_rate
                env = DMCContinuousEnv(
                    "acrobot",
                    "swingup-xk",
                    seed=7,
                    raw_state_obs=True,
                    time_sampling="uniform",
                    dt=0.002,
                    physics_dt=0.002,
                    episode_duration=0.01,
                    task_kwargs=task_kwargs,
                )
                self.addCleanup(env.close)

                _, reset_info = env.reset(seed=7)
                _, reward, _, _, step_info = env.step(
                    np.zeros(1, dtype=np.float32)
                )

                for info in (reset_info, step_info):
                    self.assertIn("acrobot_xk_energy_error_norm", info)
                    self.assertIn("acrobot_xk_elbow_norm", info)
                    self.assertIn("acrobot_xk_elbow_rate_norm", info)
                    self.assertIn("acrobot_xk_homoclinic_capture", info)
                self.assertEqual(step_info["acrobot_xk_reward"], reward)


if __name__ == "__main__":
    unittest.main()
