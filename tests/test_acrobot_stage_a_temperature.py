"""Automatic temperature uses continuous action entropy and a separate optimizer."""
from copy import deepcopy
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.integrate import quad
import torch

from algorithms.acrobot_ph_value import AcrobotPHValue, ValueFlowConfig
from benchmarks.run_acrobot_stage_a import parser, run
from benchmarks.check_acrobot_stage_a_targets import parser as diagnostic_parser, run as diagnostic_run


class TestStageATemperature(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        self.states = torch.tensor([[math.pi, 0., 0., 0.], [math.pi + .01, .02, .1, -.1]])

    def agent(self, **kwargs):
        return AcrobotPHValue(config=ValueFlowConfig(temperature=.1, auto_temperature=True, **kwargs))

    def test_entropy_matches_continuous_integral(self):
        agent = self.agent()
        eta = torch.tensor([0., .05, .5, 2.], dtype=torch.float64)
        expected = []
        for e in eta.tolist():
            mode = np.clip(e / .01, -20., 20.)
            peak = e * mode - .005 * mode**2
            def scaled(a):
                return (20 * e * a - 2 * a*a - peak) / agent.temperature
            integral = quad(lambda a: np.exp(scaled(a)), -1, 1, epsabs=1e-12)[0]
            mean_score = quad(lambda a: np.exp(scaled(a)) * scaled(a), -1, 1, epsabs=1e-12)[0] / integral
            expected.append(math.log(integral) - mean_score)
        np.testing.assert_allclose(agent.policy_entropy(eta).numpy(), expected, atol=1e-8)
        self.assertTrue(np.all(np.array(expected) < math.log(2)))

    def test_update_direction_and_no_value_changes(self):
        for target, direction in ((0., 1), (-1., -1)):
            agent = self.agent(target_entropy=target)
            value = deepcopy(agent.value.state_dict())
            old = agent.temperature
            metrics = agent.update_temperature(self.states)
            self.assertGreater(direction * (agent.temperature - old), 0.)
            self.assertEqual(metrics["temperature"], agent.temperature)
            for name, param in agent.value.state_dict().items():
                torch.testing.assert_close(param, value[name], rtol=0, atol=0)
            self.assertEqual(agent.updates, 0)

    def test_bounds_and_terminal_exclusion(self):
        agent = self.agent(temperature_min=.09999)
        agent.update_temperature(self.states)
        self.assertAlmostEqual(agent.temperature, .09999, places=7)
        old = agent.temperature
        result = agent.update_temperature(self.states, terminal_mask=[True, True])
        self.assertTrue(result["temperature_update_skipped"])
        self.assertEqual(agent.temperature, old)
        agent = self.agent(target_entropy=0., temperature_max=.10001)
        agent.update_temperature(self.states)
        self.assertAlmostEqual(agent.temperature, .10001, places=7)

    def test_current_temperature_changes_soft_score_and_sampling(self):
        agent = self.agent()
        eta = torch.tensor([0., .1])
        score = agent.soft_action_score(eta)
        states = self.states[0].repeat(500, 1)
        low = agent.act(states, deterministic=False, rng=np.random.default_rng(1))
        mode = agent.act(states)
        with torch.no_grad():
            agent.log_temperature.fill_(math.log(1.))
        high = agent.act(states, deterministic=False, rng=np.random.default_rng(1))
        self.assertGreater(high.std(), low.std() * 2)
        self.assertFalse(torch.equal(score, agent.soft_action_score(eta)))
        np.testing.assert_array_equal(mode, agent.act(states))

    def test_checkpoint_continuation_includes_temperature_optimizer(self):
        agent = self.agent()
        agent.update(self.states)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agent.pt"
            agent.save(path)
            loaded = AcrobotPHValue.load(path)
            self.assertEqual(agent.temperature, loaded.temperature)
            original_metrics = agent.update(self.states)
            loaded_metrics = loaded.update(self.states)
            self.assertEqual(original_metrics, loaded_metrics)
            for a, b in zip(agent.value.parameters(), loaded.value.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            # Old checkpoints have neither optimizer keys nor new config fields.
            legacy = AcrobotPHValue()
            legacy.save(path)
            state = torch.load(path, weights_only=True)
            for key in ("auto_temperature", "temperature_learning_rate", "target_entropy", "temperature_min", "temperature_max"):
                del state["config"][key]
            del state["log_temperature"], state["temperature_optimizer"]
            torch.save(state, path)
            loaded = AcrobotPHValue.load(path)
            self.assertEqual(loaded.temperature, 0.)
            np.testing.assert_array_equal(legacy.act(self.states), loaded.act(self.states))

    def test_runner_logging_and_diagnostic_rejects_auto(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "train"
            args = parser().parse_args(["--mode", "stage_a_soft_auto", "--output", str(output),
                                       "--updates", "3", "--batch-size", "8", "--eval-episodes", "1",
                                       "--episode-seconds", ".02", "--hold-seconds", ".01"])
            with contextlib.redirect_stdout(io.StringIO()):
                run(args)
            rows = [json.loads(line) for line in (output / "training.jsonl").read_text().splitlines()]
            self.assertTrue(all("policy_entropy" in row and "temperature" in row for row in rows))
            self.assertNotEqual(rows[0]["temperature"], rows[-1]["temperature"])
            with self.assertRaisesRegex(ValueError, "requires fixed temperature"):
                diagnostic_run(diagnostic_parser().parse_args(["--mode", "stage_a_soft_auto",
                                                               "--output", str(Path(tmp) / "diag")]))

    def test_csv_boolean_and_invalid_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "acrobot_ph_value.csv").write_text(
                "mode,env_id,algo_temperature,algo_auto_temperature\n"
                "auto,acrobot-stage-a,0.1,true\n")
            args = ["--output", str(Path(tmp) / "out"), "--mode", "auto", "--hyperparams-dir", tmp]
            self.assertTrue(parser().parse_args(args).auto_temperature)
            self.assertFalse(parser().parse_args(args + ["--no-auto-temperature"]).auto_temperature)
        with self.assertRaisesRegex(ValueError, "positive initial"):
            ValueFlowConfig(auto_temperature=True)
        with self.assertRaisesRegex(ValueError, "below log"):
            ValueFlowConfig(target_entropy=math.log(2))


if __name__ == "__main__":
    unittest.main()
