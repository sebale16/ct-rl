"""Finite log state shaping must agree across physics, HJB, and saved runs."""
from dataclasses import asdict
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from algorithms.acrobot_ph_value import AcrobotPHValue
from benchmarks.run_acrobot_stage_a import build_agent, parser, run
from benchmarks.check_acrobot_stage_a_targets import parser as diagnostic_parser, run as diagnostic_run
from environment.acrobot_stage_a import AcrobotStageAEnv
from models.acrobot_oracle import UprightReward


class TestLogStateReward(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def build(self, *options):
        torch.manual_seed(7)
        return build_agent(parser().parse_args(["--output", "/tmp/not-created-stage-a-log", *options]))

    def env(self, agent, config):
        env = AcrobotStageAEnv(config, agent.oracle, agent.reward)
        self.addCleanup(env.close)
        return env

    def test_default_log_parameters_bound_and_zero_goal(self):
        base, _ = self.build()
        agent, config = self.build("--mode", "stage_a_hard_log")
        r = agent.reward
        self.assertAlmostEqual(r.log_epsilon, .0014773937019467033)
        self.assertAlmostEqual(r.log_cost_bound, .9223507759664514)
        self.assertEqual(r.effort_weight, base.reward.effort_weight)
        env = self.env(agent, config)
        self.assertAlmostEqual(env.failure_rate, -1.)
        self.assertAlmostEqual(env.failure_value, -10.)
        goal = torch.tensor([math.pi, 0., 0., 0.], dtype=torch.float64, requires_grad=True)
        cost = r.state_cost(goal, agent.oracle)
        self.assertEqual(float(cost.detach()), 0.)
        grad = torch.autograd.grad(cost, goal)[0]
        torch.testing.assert_close(grad, torch.zeros(4, dtype=torch.float64), rtol=0, atol=1e-15)
        hessian = torch.autograd.functional.hessian(lambda z: r.state_cost(z, agent.oracle), goal)
        self.assertTrue(torch.isfinite(hessian).all())
        self.assertTrue(torch.linalg.eigvalsh(hessian).min() > 0)
        costs = torch.linspace(0, r.log_cost_bound, 101, dtype=torch.float64)
        transformed = r.transform_state_cost(costs)
        self.assertTrue(torch.all(transformed[1:] > transformed[:-1]))
        self.assertAlmostEqual(float(transformed[-1]), r.log_cost_bound)
        self.assertTrue(torch.all(transformed >= costs - 1e-15))

    def test_chain_rule_matches_predicted_local_gain(self):
        base, _ = self.build()
        agent, _ = self.build("--mode", "stage_a_hard_log")
        z = torch.tensor([math.pi + math.radians(.1), 0., 0., 0.], dtype=torch.float64, requires_grad=True)
        c = base.reward.state_cost(z, base.oracle)
        base_grad = torch.autograd.grad(c, z)[0]
        transformed = agent.reward.state_cost(z, agent.oracle)
        grad = torch.autograd.grad(transformed, z)[0]
        r = agent.reward
        gain = r.log_cost_bound / ((r.log_epsilon + float(c.detach())) * math.log1p(r.log_cost_bound / r.log_epsilon))
        torch.testing.assert_close(grad, base_grad * gain, rtol=1e-10, atol=1e-12)
        self.assertAlmostEqual(gain, 96.9300587382481, places=8)
        torque = torch.tensor(3., dtype=torch.float64, requires_grad=True)
        rate = r.rate(z.detach(), torque, agent.oracle)
        self.assertAlmostEqual(float(torch.autograd.grad(rate, torque)[0]), -r.effort_weight * 3.)

    def test_simulator_reward_and_hjb_use_identical_transform(self):
        agent, config = self.build("--mode", "stage_a_hard_log")
        env = self.env(agent, config)
        rng = np.random.default_rng(8)
        qv = np.c_[rng.uniform(math.pi / 2 + .01, 3 * math.pi / 2 - .01, 20),
                   rng.uniform(-3., 3., 20), rng.uniform(-6., 6., (20, 2))]
        torque = rng.uniform(-20, 20, 20)
        z = agent.oracle.canonical(torch.tensor(qv, dtype=torch.float64))
        np.testing.assert_allclose([env._rate(x, u) for x, u in zip(qv, torque)],
                                   agent.reward.rate(z, torch.tensor(torque), agent.oracle).numpy(), atol=1e-12)
        base, _ = self.build()
        z32 = z.float()
        _, op_base, _ = base.operator(z32)
        _, op_log, _ = agent.operator(z32)
        extra_cost = agent.reward.state_cost(z32, agent.oracle) - base.reward.state_cost(z32, base.oracle)
        torch.testing.assert_close(op_log - op_base, -extra_cost)
        np.testing.assert_array_equal(base.act(z32), agent.act(z32))
        # A real terminal transition retains the same absorbing label.
        env.reset(options={"qv": [math.pi, 0., 0., 2 * math.pi - .01]})
        next_z, reward, terminated, _, info = env.step(np.ones(1))
        self.assertTrue(terminated)
        self.assertLessEqual(reward, math.exp(-config.discount_rate * info["dt_used"]) * env.failure_value)
        labels, _, _ = agent.fitted_targets(next_z[None], terminal_mask=[True], terminal_value=env.failure_value)
        self.assertAlmostEqual(float(labels[0]), -10.)

    def test_scaling_and_task_limits_recompute_log_reference(self):
        normalized, _ = self.build("--mode", "stage_a_hard_log")
        raw, _ = self.build("--mode", "stage_a_hard_log", "--reward-scale", "1")
        factor = normalized.metadata["reward_scale"]
        self.assertAlmostEqual(normalized.reward.log_epsilon, raw.reward.log_epsilon * factor)
        self.assertAlmostEqual(normalized.reward.log_cost_bound, raw.reward.log_cost_bound * factor)
        z = torch.tensor([[math.pi + .1, .2, .3, -.2]])
        torch.testing.assert_close(normalized.reward.state_cost(z, normalized.oracle),
                                   raw.reward.state_cost(z, raw.oracle) * factor)
        rescaled = normalized.reward.scaled(.5)
        torch.testing.assert_close(rescaled.rate(z, 2., normalized.oracle),
                                   normalized.reward.rate(z, 2., normalized.oracle) * .5)
        changed, config = self.build("--state-cost-transform", "log", "--log-reference-angle-deg", "1",
                                     "--velocity-limit", "2", "--shoulder-limit", ".4", "--elbow-limit", ".8")
        env = self.env(changed, config)
        self.assertAlmostEqual(env.failure_rate, -1.)
        self.assertAlmostEqual(changed.reward.log_epsilon,
                               2 * changed.reward.angle1_weight * math.sin(math.radians(1) / 2)**2)

    def test_runner_checkpoint_and_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "train"
            args = parser().parse_args(["--mode", "stage_a_soft_auto_log", "--output", str(output), "--updates", "3",
                                       "--batch-size", "8", "--eval-episodes", "1", "--episode-seconds", ".02", "--hold-seconds", ".01"])
            with contextlib.redirect_stdout(io.StringIO()):
                run(args)
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["reward"]["state_cost_transform"], "log")
            self.assertAlmostEqual(config["reward"]["log_epsilon"], .0014773937019467033)
            restored, _ = build_agent(parser().parse_args(["--checkpoint", str(output / "last.pt"),
                                                          "--output", str(Path(tmp) / "eval")]))
            self.assertEqual(asdict(restored.reward), config["reward"])
            self.assertTrue(restored.config.auto_temperature)
            fixed = Path(tmp) / "diagnostic"
            with contextlib.redirect_stdout(io.StringIO()):
                result = diagnostic_run(diagnostic_parser().parse_args([
                    "--mode", "stage_a_hard_log", "--output", str(fixed), "--updates", "2",
                    "--train-states", "8", "--heldout-states", "8", "--collection-steps", "0", "--skip-rollouts"]))
            self.assertEqual(result["status"], "completed")
            # Legacy checkpoint records omit all transform fields.
            path = Path(tmp) / "legacy.pt"
            AcrobotPHValue().save(path)
            state = torch.load(path, weights_only=True)
            for key in ("state_cost_transform", "log_epsilon", "log_cost_bound"):
                del state["reward"][key]
            torch.save(state, path)
            self.assertEqual(AcrobotPHValue.load(path).reward.state_cost_transform, "identity")

    def test_presets_and_invalid_parameters(self):
        for mode, temperature, auto in (("stage_a_hard_log", 0., False), ("stage_a_soft_log", .1, False),
                                       ("stage_a_soft_auto_log", .1, True)):
            args = parser().parse_args(["--mode", mode, "--output", "/tmp/not-created-stage-a-log"])
            self.assertEqual((args.state_cost_transform, args.temperature, args.auto_temperature), ("log", temperature, auto))
        for angle in (0., -1., 181., float("inf"), float("nan")):
            with self.subTest(angle=angle), self.assertRaises(ValueError):
                self.build("--state-cost-transform", "log", "--log-reference-angle-deg", str(angle))
        with self.assertRaisesRegex(ValueError, "log_epsilon"):
            UprightReward(state_cost_transform="log")
        args = parser().parse_args(["--output", "/tmp/not-created-stage-a-log"])
        args.state_cost_transform = "invalid"
        with self.assertRaisesRegex(ValueError, "state_cost_transform"):
            build_agent(args)


if __name__ == "__main__":
    unittest.main()
