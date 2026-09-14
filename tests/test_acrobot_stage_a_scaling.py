"""Reward units must agree in the simulator, HJB, analytic policy, and checkpoints."""
from dataclasses import asdict
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from algorithms.acrobot_ph_value import AcrobotPHValue
from benchmarks.run_acrobot_stage_a import build_agent, parser
from environment.acrobot_stage_a import AcrobotStageAEnv


class TestStageAScaling(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def build(self, *options):
        torch.manual_seed(17)
        return build_agent(parser().parse_args(["--output", "/tmp/not-created-stage-a-scaling", *options]))

    def env(self, agent, config):
        env = AcrobotStageAEnv(config, agent.oracle, agent.reward)
        self.addCleanup(env.close)
        return env

    def test_normalization_includes_effort_and_failure_continuation(self):
        raw, config = self.build("--reward-scale", "1")
        scaled, _ = self.build()
        raw_env, scaled_env = self.env(raw, config), self.env(scaled, config)
        factor = scaled.metadata["reward_scale"]
        self.assertAlmostEqual(factor, .03882461201677423)
        self.assertAlmostEqual(scaled.reward.effort_weight, raw.reward.effort_weight * factor)
        self.assertEqual(scaled.reward.velocity_scale, raw.reward.velocity_scale)
        self.assertAlmostEqual(scaled_env.failure_rate, -1.)
        self.assertAlmostEqual(scaled_env.failure_value, -10.)
        goal = [math.pi, 0., 0., 0.]
        self.assertAlmostEqual(scaled_env._rate(goal, 0.), 0.)
        # Include a nonterminal interval and a speed-failure interval.
        for qv, expect_failure in ((goal, False), ([math.pi, 0., 0., 2 * math.pi - .01], True)):
            raw_env.reset(options={"qv": qv})
            scaled_env.reset(options={"qv": qv})
            a, b = raw_env.step(np.ones(1)), scaled_env.step(np.ones(1))
            self.assertEqual(a[2], expect_failure)
            self.assertEqual(a[2:4], b[2:4])
            np.testing.assert_array_equal(a[0], b[0])
            self.assertAlmostEqual(b[1], factor * a[1], places=10)
        qv = np.array([3 * math.pi / 2 - 1e-7, math.pi - 1e-7,
                       2 * math.pi - 1e-7, 2 * math.pi - 1e-7])
        self.assertGreater(scaled_env._rate(qv, 20.), -1.)
        self.assertAlmostEqual(scaled_env._rate(qv, 20.), -1., places=6)

    def test_hjb_and_initial_policy_consistent_in_both_reward_units(self):
        for mode in ("stage_a_hard", "stage_a_soft"):
            with self.subTest(mode=mode):
                raw, _ = self.build("--mode", mode, "--reward-scale", "1")
                scaled, _ = self.build("--mode", mode)
                factor = scaled.metadata["reward_scale"]
                z = raw.oracle.canonical(torch.tensor([[math.pi + .3, .2, .1, -.3], [math.pi - .1, -.3, -.2, .1]]))
                for a, b in zip(raw.fitted_targets(z), scaled.fitted_targets(z)):
                    torch.testing.assert_close(b, a * factor, atol=1e-7, rtol=1e-4)
                np.testing.assert_allclose(raw.act(z), scaled.act(z), atol=1e-7, rtol=1e-4)
                self.assertAlmostEqual(scaled.temperature, raw.temperature * factor)
                if mode == "stage_a_soft":
                    np.testing.assert_allclose(
                        raw.act(z, deterministic=False, rng=np.random.default_rng(42)),
                        scaled.act(z, deterministic=False, rng=np.random.default_rng(42)), atol=1e-7)

    def test_new_limits_and_checkpoint_restore_without_double_scaling(self):
        agent, config = self.build()
        self.assertEqual(config.velocity_limit, 2 * math.pi)
        self.assertEqual(config.elbow_limit, math.pi)
        env = self.env(agent, config)
        for joint in (0, 1):
            for sign in (-1, 1):
                qv = [math.pi, 0., 0., 0.]
                qv[2 + joint] = sign * 2 * math.pi
                with self.assertRaisesRegex(ValueError, "within state limits"):
                    env.reset(options={"qv": qv})
        for direction in (-1, 1):
            with self.assertRaisesRegex(ValueError, "within state limits"):
                env.reset(options={"qv": [math.pi, direction * math.pi, 0., 0.]})
            env.reset(options={"qv": [math.pi, direction * (math.pi - 1e-4), 0., direction * .2]})
            _, _, terminated, truncated, info = env.step(np.zeros(1))
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["state_limit_failure"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agent.pt"
            agent.save(path, metadata={**agent.metadata, "environment": asdict(config)})
            loaded, restored = self.build("--checkpoint", str(path), "--reward-scale", "1")
            self.assertEqual(asdict(loaded.reward), asdict(agent.reward))
            self.assertEqual(loaded.metadata["reward_scale"], agent.metadata["reward_scale"])
            self.assertEqual(restored, config)
            self.assertAlmostEqual(self.env(loaded, restored).failure_value, -10.)

    def test_explicit_scale_and_changed_limits_recompute_consistently(self):
        agent, config = self.build("--reward-scale", ".5")
        self.assertAlmostEqual(agent.reward.angle1_weight, 5.)
        self.assertAlmostEqual(agent.reward.effort_weight, .005)
        self.assertAlmostEqual(self.env(agent, config).failure_rate, -25.75685752037776 / 2)
        agent, config = self.build("--velocity-limit", "2", "--elbow-limit", ".8", "--shoulder-limit", ".4")
        self.assertAlmostEqual(self.env(agent, config).failure_rate, -1.)


if __name__ == "__main__":
    unittest.main()
