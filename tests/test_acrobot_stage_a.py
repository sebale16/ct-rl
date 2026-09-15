import math
import csv
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("MUJOCO_GL", "disable")

import numpy as np
from scipy.integrate import quad
import torch
from torch import nn

from algorithms.acrobot_ph_value import AcrobotPHValue, ValueFlowConfig, bounded_action_score
from benchmarks.run_acrobot_stage_a import evaluate, parser, run
from benchmarks.acrobot_stage_a_config import DEFAULT_HYPERPARAMS_DIR
from environment.acrobot_stage_a import AcrobotStageAEnv, StageAConfig, load_incoming_states
from models.acrobot_oracle import AcrobotOracle, UprightReward


class TestAcrobotStageA(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(13)

    def env(self, **kwargs):
        env = AcrobotStageAEnv(**kwargs)
        self.addCleanup(env.close)
        return env

    def test_oracle_matches_mujoco_with_coordinate_conversion(self):
        oracle = AcrobotOracle(damping=0.13)
        env = self.env(oracle=oracle, config=StageAConfig(shoulder_limit=None))
        rng = np.random.default_rng(5)
        for _ in range(12):
            qv = np.r_[rng.uniform(-np.pi, np.pi, 2), rng.uniform(-1., 1., 2)]
            env.reset(options={"qv": qv})
            torque = rng.uniform(-20., 20.)
            env.physics.data.ctrl[:] = torque / 20.
            env.physics.forward()
            tangent = np.r_[qv[2:], env.physics.data.qacc.copy()]
            x = torch.tensor(qv, dtype=torch.float64)
            z, dz = torch.autograd.functional.jvp(oracle.canonical, x, torch.tensor(tangent))
            torch.testing.assert_close(oracle.drift(z, torque), dz, atol=2e-9, rtol=2e-9)

    def test_energy_balance_and_momentum_roundtrip(self):
        oracle = AcrobotOracle(damping=0.2)
        qv = torch.randn(20, 4, dtype=torch.float64)
        z = oracle.canonical(qv).detach().requires_grad_(True)
        torch.testing.assert_close(oracle.velocity(z), qv[:, 2:])
        grad = torch.autograd.grad(oracle.energy(z).sum(), z)[0]
        torque = torch.linspace(-20., 20., 20, dtype=torch.float64)
        rate = (grad * oracle.drift(z, torque)).sum(-1)
        expected = torque * qv[:, 3] - 0.2 * qv[:, 2:].square().sum(-1)
        torch.testing.assert_close(rate, expected)

    def test_saturated_operator_matches_action_grid_and_gradient(self):
        eta = torch.tensor([-1., -.02, 0., .05, 2.], dtype=torch.float64, requires_grad=True)
        chi = bounded_action_score(eta, .01, 20.)
        u = torch.linspace(-20., 20., 20001, dtype=torch.float64)
        brute = (eta[:, None] * u - .005 * u.square()).max(-1).values
        torch.testing.assert_close(chi, brute)
        grad = torch.autograd.grad(chi.sum(), eta)[0]
        torch.testing.assert_close(grad, (eta / .01).clamp(-20, 20))

    def test_operator_is_full_reward_plus_oracle_directional_derivative(self):
        class KnownValue(nn.Module):
            def forward(self, z):
                return z[..., 0].sin() + 2 * z[..., 2] + 0.4 * z[..., 3]
        agent = AcrobotPHValue()
        agent.target = KnownValue()
        z = torch.tensor([[3., .1, .2, .3], [2.8, -.1, .3, -.2]])
        values, operator, eta = agent.operator(z)
        u = (eta / agent.reward.effort_weight).clamp(-20, 20)
        gradient = torch.stack((z[:, 0].cos(), torch.zeros(2), torch.full((2,), 2.), torch.full((2,), .4)), -1)
        expected = (agent.reward.rate(z, u, agent.oracle)
                    + (gradient * agent.oracle.drift(z, u)).sum(-1) - .1 * values)
        torch.testing.assert_close(operator, expected)
        torch.testing.assert_close(eta, torch.full((2,), .4))

    def test_soft_operator_matches_continuous_integral(self):
        agent = AcrobotPHValue(config=ValueFlowConfig(temperature=.1))
        eta = torch.tensor([-2., -.02, 0., .05, 2.], dtype=torch.float64)
        actual = agent.soft_action_score(eta)
        expected = []
        for e in eta.tolist():
            mode = np.clip(e / .01, -20., 20.)
            peak = e * mode - .005 * mode**2
            integral = quad(lambda a: np.exp((20 * e * a - 2 * a*a - peak) / .1), -1, 1)[0]
            expected.append(peak + .1 * np.log(integral))
        np.testing.assert_allclose(actual.numpy(), expected, atol=1e-9, rtol=1e-9)
        actions = agent.act(np.tile([math.pi, 0., 0., 0.], (200, 1)), deterministic=False, rng=np.random.default_rng(0))
        self.assertTrue(np.all(np.abs(actions) <= 1))
        self.assertGreater(actions.std(), .01)

    def test_upright_reward_anchor_and_periodic_value(self):
        agent = AcrobotPHValue()
        goal = torch.tensor([math.pi, 0., 0., 0.])
        self.assertAlmostEqual(float(agent.value(goal).detach()), 0., places=7)
        self.assertLess(abs(float(agent.act(goal)[0])), 1e-5)
        self.assertAlmostEqual(float(agent.reward.rate(goal, 0., agent.oracle)), 0., places=7)
        fast = agent.oracle.canonical(torch.tensor([math.pi, 0., 0.5, 0.]))
        self.assertLess(float(agent.reward.rate(fast, 0., agent.oracle)), 0.)
        shifted = goal + torch.tensor([2*math.pi, -2*math.pi, 0., 0.])
        torch.testing.assert_close(agent.value(goal), agent.value(shifted), atol=1e-7, rtol=0)

    def test_hold_does_not_terminate_and_elapsed_reward_uses_seconds(self):
        config = StageAConfig(dt=.01, episode_seconds=.05, hold_seconds=.02,
                              angle_radius=0., velocity_radius=0.)
        env = self.env(config=config)
        env.reset(seed=4)
        for i in range(5):
            _, reward, term, trunc, info = env.step(np.zeros(1))
            self.assertFalse(term)
            self.assertEqual(trunc, i == 4)
            self.assertAlmostEqual(reward, 0., places=10)
        self.assertTrue(info["retained_success"])
        self.assertAlmostEqual(info["max_hold_seconds"], .05)
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(1))

    def test_torque_clipping_and_discounted_reward_quadrature(self):
        env = self.env(config=StageAConfig(dt=.002, episode_seconds=.004, hold_seconds=.002))
        qv = np.array([math.pi + .02, .01, .02, .01])
        env.reset(options={"qv": qv})
        rate = env._rate(qv, 20.)
        _, reward, _, _, info = env.step(np.array([4.]))
        self.assertEqual(info["torque"], 20.)
        self.assertAlmostEqual(info["dt_used"], .002)
        self.assertLess(abs(reward - .002 * rate), 1e-5)

    def test_incoming_frame_conversion_and_velocity_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "incoming.npz"
            original = np.array([[math.pi/2-.1, .02, .4, -.3]])
            np.savez(path, states=original, frame="xin_kaneda_qv")
            bank = load_incoming_states(path)
            np.testing.assert_allclose(bank[0], [math.pi-.1, .02, .4, -.3])
            env = self.env(incoming_states=bank)
            _, info = env.reset(seed=1, options={"incoming": True})
            np.testing.assert_allclose(info["qv"], bank[0])
            np.savez(path, states=original, frame="guess")
            with self.assertRaises(ValueError):
                load_incoming_states(path)

    def test_state_limit_exit_has_absorbing_cost_and_terminal_value_target(self):
        env = self.env()
        env.reset(options={"qv": [math.pi, 0., 0., env.config.velocity_limit - .01]})
        z, reward, terminated, truncated, info = env.step(np.ones(1))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["state_limit_failure"])
        self.assertLess(info["dt_used"], env.config.dt)
        tail = math.exp(-env.config.discount_rate * info["dt_used"]) * env.failure_value
        self.assertLessEqual(reward, tail)
        self.assertLess(reward, -100.)
        agent = AcrobotPHValue()
        before = float(agent.value(torch.tensor(z)).detach())
        metrics = agent.update(z[None], terminal_mask=[True], terminal_value=env.failure_value)
        self.assertAlmostEqual(metrics["value_loss"] / (before - env.failure_value)**2, 1., places=5)

    def test_shoulder_boundaries_reject_resets_and_terminate_outward_motion(self):
        env = self.env()
        for angle in (math.pi / 2, 3 * math.pi / 2, -math.pi, 3 * math.pi):
            with self.subTest(invalid_angle=angle), self.assertRaisesRegex(ValueError, "within state limits"):
                env.reset(options={"qv": [angle, 0., 0., 0.]})
        for boundary, direction in ((math.pi / 2, -1), (3 * math.pi / 2, 1)):
            with self.subTest(boundary=boundary):
                env.reset(options={"qv": [boundary - direction * 1e-4, 0., direction * .2, 0.]})
                _, reward, terminated, truncated, info = env.step(np.zeros(1))
                self.assertTrue(terminated)
                self.assertFalse(truncated)
                self.assertTrue(info["state_limit_failure"])
                self.assertLess(info["dt_used"], env.config.dt)
                self.assertLessEqual(reward, math.exp(-env.config.discount_rate * info["dt_used"]) * env.failure_value)
                self.assertLess(np.abs(info["qv"][2:]).max(), env.config.velocity_limit)
                self.assertLess(abs(info["qv"][1]), env.config.elbow_limit)
        with self.assertRaisesRegex(ValueError, "within declared state limits"):
            AcrobotStageAEnv(incoming_states=[[0., 0., 0., 0.]])

    def test_shoulder_domain_reward_bound_and_configuration(self):
        env = self.env()
        self.assertAlmostEqual(env.failure_rate, -25.75685752037776)
        self.assertAlmostEqual(env.failure_value, -257.5685752037775)
        # Approach all maximal cost terms from inside the allowed domain.
        qv = np.array([3 * math.pi / 2 - 1e-7, math.pi - 1e-7,
                       2 * math.pi - 1e-7, 2 * math.pi - 1e-7])
        self.assertGreater(env._rate(qv, 20.), env.failure_rate)
        self.assertAlmostEqual(env._rate(qv, 20.), env.failure_rate, places=5)
        for limit in (0., -.1, float("inf"), float("nan"), .04):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                StageAConfig(shoulder_limit=limit)
        with tempfile.TemporaryDirectory() as tmp:
            from benchmarks.run_acrobot_stage_a import build_agent
            from dataclasses import asdict
            path = Path(tmp) / "agent.pt"
            agent = AcrobotPHValue()
            environment = asdict(StageAConfig())
            agent.save(path, metadata={"environment": environment})
            args = parser().parse_args(["--checkpoint", str(path), "--output", str(Path(tmp) / "eval")])
            _, config = build_agent(args)
            self.assertEqual(config.shoulder_limit, math.pi / 2)
            del environment["shoulder_limit"]
            environment.update(velocity_limit=12., elbow_limit=4 * math.pi)
            agent.save(path, metadata={"environment": environment})
            _, legacy_config = build_agent(args)
            self.assertIsNone(legacy_config.shoulder_limit)
            legacy = self.env(config=legacy_config)
            legacy.reset(options={"qv": [0., 0., 0., 0.]})
            self.assertAlmostEqual(legacy.failure_value, -457.0337302665053)

    def test_incoming_training_requires_separate_evaluation_bank(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "states.npz"
            np.savez(path, states=[[math.pi, 0., .1, .1]], frame="downward_vertical_qv")
            args = parser().parse_args(["--output", str(Path(tmp) / "out"), "--incoming-states", str(path)])
            with self.assertRaisesRegex(ValueError, "held-out"):
                run(args)
            args.incoming_eval_states = path
            with self.assertRaisesRegex(ValueError, "overlap"):
                run(args)

    def test_value_update_checkpoint_and_no_learned_ph(self):
        agent = AcrobotPHValue()
        before = [p.detach().clone() for p in agent.value.parameters()]
        z = agent.oracle.canonical(torch.tensor([[3.1, .02, .1, -.1], [3.2, -.03, -.1, .1]]))
        metrics = agent.update(z)
        self.assertTrue(all(np.isfinite(list(metrics.values()))))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, agent.value.parameters())))
        self.assertTrue(all(not p.requires_grad for p in agent.target.parameters()))
        self.assertFalse(hasattr(agent.oracle, "parameters"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            agent.save(path, metadata={"environment": {"dt": .01}})
            loaded = AcrobotPHValue.load(path)
            np.testing.assert_array_equal(agent.act(z), loaded.act(z))
            self.assertEqual(loaded.updates, 1)
            self.assertEqual(loaded.metadata, {"environment": {"dt": .01}})

    def test_runner_and_checkpoint_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "train"
            args = parser().parse_args(["--output", str(output), "--updates", "3", "--batch-size", "8",
                                        "--episode-seconds", ".02", "--hold-seconds", ".01", "--eval-episodes", "1"])
            run(args)
            self.assertTrue((output / "last.pt").is_file())
            self.assertTrue((output / "best.pt").is_file())
            metadata = json.loads((output / "config.json").read_text())
            self.assertEqual(metadata["hyperparams"]["mode"], "stage_a_hard")
            self.assertEqual(metadata["arguments"]["updates"], 3)
            self.assertEqual(metadata["hyperparams"]["row"]["total_updates"], "10000")
            reeval = Path(tmp) / "eval"
            run(parser().parse_args(["--output", str(reeval), "--checkpoint", str(output / "last.pt"), "--eval-episodes", "1"]))
            self.assertTrue((reeval / "evaluations.jsonl").is_file())
            self.assertFalse((reeval / "last.pt").exists())
            with self.assertRaises(FileExistsError):
                run(args)


class TestStageAHyperparameters(unittest.TestCase):
    def test_default_and_soft_rows_and_cli_precedence(self):
        p = parser()
        hard = p.parse_args(["--output", "/tmp/not-created-stage-a"])
        self.assertEqual(hard.mode, "stage_a_hard")
        self.assertEqual(hard.temperature, 0.)
        self.assertEqual(hard.hidden_width, 64)
        self.assertEqual(hard.shoulder_limit, math.pi / 2)
        custom = p.parse_args(["--output", "/tmp/not-created-stage-a", "--shoulder-limit", ".4"])
        self.assertEqual(custom.shoulder_limit, .4)
        soft = p.parse_args(["--output", "/tmp/not-created-stage-a", "--mode", "stage_a_soft"])
        self.assertEqual(soft.temperature, .1)
        self.assertEqual(soft.exploration_std, 0.)
        explicit = p.parse_args(["--output", "/tmp/not-created-stage-a", "--mode", "stage_a_soft",
                                "--temperature", ".02", "--dt", ".02", "--updates", "7"])
        self.assertEqual((explicit.temperature, explicit.dt, explicit.updates), (.02, .02, 7))
        self.assertEqual(explicit.hyperparams_source["row"]["algo_temperature"], "0.1")

    def test_custom_table_blank_cells_and_new_model_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "acrobot_ph_value.csv"
            path.write_text("mode,env_id,algo_learning_rate,model_hidden_width,algo_target_rate,total_updates\n"
                            "small,acrobot-stage-a,0.001,16,0.2,\n")
            args = parser().parse_args(["--output", str(Path(tmp) / "out"), "--hyperparams_dir", tmp, "--mode", "small"])
            self.assertEqual((args.learning_rate, args.hidden_width, args.target_rate, args.updates), (.001, 16, .2, 10000))
            self.assertEqual(args.hyperparams_source["path"], str(path))
            self.assertEqual(len(args.hyperparams_source["sha256"]), 64)

    def test_missing_duplicate_and_unknown_columns_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "acrobot_ph_value.csv"
            for contents in (
                "mode,env_id\nother,acrobot-stage-a\n",
                "mode,env_id\nstage_a_hard,acrobot-stage-a\nstage_a_hard,acrobot-stage-a\n",
                "mode,env_id,algo_typo\nstage_a_hard,acrobot-stage-a,1\n",
            ):
                path.write_text(contents)
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parser().parse_args(["--output", str(Path(tmp) / "out"), "--hyperparams-dir", tmp])

    def test_checkpoint_evaluation_does_not_require_training_table(self):
        args = parser().parse_args(["--output", "/tmp/not-created-stage-a", "--checkpoint", "last.pt",
                                    "--hyperparams-dir", "/no/such/directory"])
        self.assertIsNone(args.hyperparams_source)

    def test_table_defaults_match_dataclasses(self):
        args = parser().parse_args(["--output", "/tmp/not-created-stage-a"])
        for key, value in vars(UprightReward()).items():
            if key in ("log_epsilon", "log_cost_bound"):
                self.assertIsNone(value)  # Resolved from the task limits, not CLI constants.
            else:
                self.assertEqual(getattr(args, key), value)
        for key, value in vars(ValueFlowConfig()).items():
            self.assertEqual(getattr(args, key), value)
        with (DEFAULT_HYPERPARAMS_DIR / "acrobot_ph_value.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(None not in row and all(v is not None for v in row.values()) for row in rows))


if __name__ == "__main__":
    unittest.main()
