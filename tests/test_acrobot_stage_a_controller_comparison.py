"""Controls for causal pairing, deployment derivatives, and online collection."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.linalg import solve_continuous_are
import torch

from algorithms.acrobot_ph_value import AcrobotPHValue, ValueFlowConfig
from benchmarks.compare_acrobot_stage_a_controllers import (
    ARMS, OnlineReplay, comparison_step, neighborhood, parser, run, upright_probe,
)
from benchmarks.check_acrobot_stage_a_targets import labels_for
from environment.acrobot_stage_a import StageAConfig


class ControllerComparisonTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.agent = AcrobotPHValue(config=ValueFlowConfig(hidden_width=8))
        self.data = {
            "states": np.array([[3.1, .02, .1, .2], [3.2, -.02, -.1, -.2],
                                [3., .05, 1., 2.], [3.3, -.04, -1., -2.]], dtype=np.float32),
            "terminal": np.array([False, False, False, True]),
            "rollout": np.array([False, False, True, True]),
        }

    def test_first_update_is_shared_and_only_fixed_target_stays_frozen(self):
        self.agent.update(self.data["states"])
        self.data["initial_labels"] = labels_for(self.agent, self.data, -10., 4)
        agents = {arm: deepcopy(self.agent) for arm in ARMS}
        target_before = deepcopy(self.agent.target.state_dict())
        comparison_step(agents, self.data, np.arange(4), -10., first_update=True)
        for arm in ARMS[1:]:
            for a, b in zip(agents[ARMS[0]].value.parameters(), agents[arm].value.parameters()):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        for name, tensor in agents["fixed_labels"].target.state_dict().items():
            torch.testing.assert_close(tensor, target_before[name], atol=0, rtol=0)
        self.assertTrue(any(not torch.equal(tensor, target_before[name])
                            for name, tensor in agents["moving_labels"].target.state_dict().items()))
        changed = labels_for(agents["moving_labels"], self.data, -10., 4)
        self.assertEqual(changed[-1], -10.)
        self.assertTrue(np.any(changed[:-1] != self.data["initial_labels"][:-1]))
        # Subsequent changes to C's dataset cannot change A or B's updates.
        controls = deepcopy(agents)
        comparison_step(agents, self.data, np.arange(4), -10.,
                        (self.data["states"] * .9, np.zeros(4, dtype=bool)))
        comparison_step(controls, self.data, np.arange(4), -10.)
        for arm in ARMS[:2]:
            for a, b in zip(agents[arm].value.parameters(), controls[arm].value.parameters()):
                torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_known_lqr_value_has_stable_controller_and_correct_physical_derivatives(self):
        origin = torch.tensor([np.pi, 0., 0., 0.], dtype=torch.float64)
        a = torch.autograd.functional.jacobian(self.agent.oracle.drift, origin).numpy()
        b = np.array([[0.], [0.], [0.], [1.]])
        p = solve_continuous_are(a, b, np.eye(4), [[self.agent.reward.effort_weight]])

        class Quadratic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("p", torch.tensor(p, dtype=torch.float32))
                self.register_buffer("origin", origin.float())

            def forward(self, z):
                # Form the equilibrium in the input dtype, including float64 probes.
                e = z - z.new_tensor([np.pi, 0., 0., 0.])
                return -.5 * torch.einsum("...i,ij,...j->...", e, self.p, e)

        self.agent.value = Quadratic()
        self.agent.target = deepcopy(self.agent.value)
        before = deepcopy(self.agent.value.state_dict())
        _, states = neighborhood(self.agent, StageAConfig())
        metrics, arrays = upright_probe(self.agent, states, .0001)
        self.assertLess(metrics["hessian_max"], 0.)
        self.assertLess(metrics["continuous_max_real_pole"], 0.)
        self.assertLess(metrics["held_spectral_radius"], 1.)
        np.testing.assert_allclose(arrays["hessian_canonical"], -p, rtol=1e-6)
        # Compare the reported torque Jacobian against physical q/v perturbations.
        network = deepcopy(self.agent.value).double()
        actual = []
        for dim in range(4):
            plus, minus = origin.clone(), origin.clone()
            plus[dim] += 1e-5
            minus[dim] -= 1e-5
            gradients = [torch.autograd.functional.jacobian(network, self.agent.oracle.canonical(qv))[3]
                         for qv in (plus, minus)]
            actual.append(float((gradients[0] - gradients[1]) / (2e-5 * self.agent.reward.effort_weight)))
        np.testing.assert_allclose(arrays["torque_jacobian_qv"], actual, rtol=1e-7, atol=1e-7)
        for name, tensor in self.agent.value.state_dict().items():
            torch.testing.assert_close(tensor, before[name], atol=0, rtol=0)
            self.assertEqual(tensor.dtype, torch.float32)
        target, residual, _ = self.agent.fitted_targets(states)
        np.testing.assert_allclose(arrays["target_label"], target.numpy(), rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(arrays["target_hjb"], residual.numpy(), rtol=1e-6, atol=1e-7)

    def test_replay_keeps_failure_masks_through_overwrite(self):
        replay = OnlineReplay(self.data, 2)
        np.testing.assert_array_equal(replay.terminal, [False, True])
        replay.append([3., 0., 0., 0.], True)
        replay.append([3.1, 0., 0., 0.], False)
        np.testing.assert_array_equal(replay.terminal, [True, False])
        self.assertEqual(replay.count, 4)
        with self.assertRaises(ValueError):
            OnlineReplay(self.data, 1)

    def test_runner_reproducibility_rollouts_and_shared_first_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            options = ["--no-finite-horizon", "--updates", "3", "--batch-size", "8",
                       "--train-states", "16", "--heldout-states", "12", "--collection-steps", "16",
                       "--buffer-size", "16", "--probe-every", "1", "--eval-every", "2",
                       "--eval-episodes", "1", "--episode-seconds", ".02", "--hold-seconds", ".01",
                       "--hidden-width", "8"]
            first, second = Path(tmp) / "first", Path(tmp) / "second"
            with contextlib.redirect_stdout(io.StringIO()):
                report = run(parser().parse_args(["--output", str(first), *options]))
                run(parser().parse_args(["--output", str(second), *options]))
            self.assertEqual(report["completed_updates"], dict.fromkeys(ARMS, 3))
            self.assertAlmostEqual(report["online_collection_seconds"], .02)
            self.assertEqual((first / "upright.jsonl").read_bytes(), (second / "upright.jsonl").read_bytes())
            evals = [json.loads(line) for line in (first / "evaluations.jsonl").read_text().splitlines()]
            self.assertEqual(len(evals), 9)  # Three arms at updates 0, 2, 3.
            self.assertTrue(all(r["groups"][0]["episodes"][0]["seed"] == 20000 for r in evals))
            checkpoints = [AcrobotPHValue.load(first / arm / "checkpoint_00000001.pt") for arm in ARMS]
            for agent in checkpoints[1:]:
                for a, b in zip(checkpoints[0].value.parameters(), agent.value.parameters()):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            with np.load(first / "online_replay_final.npz") as data:
                self.assertEqual(int(data["total_insertions"]), 12)
                # Time-limit truncations must not become failure labels.
                self.assertFalse(data["terminal"].any())
            with self.assertRaises(FileExistsError):
                run(parser().parse_args(["--output", str(first), *options]))
            branch = Path(tmp) / "branch"
            checkpoint = first / "moving_labels" / "checkpoint_00000003.pt"
            with contextlib.redirect_stdout(io.StringIO()):
                resumed = run(parser().parse_args(["--output", str(branch), *options,
                    "--checkpoint", str(checkpoint), "--updates", "1"]))
            self.assertEqual(resumed["completed_updates"], dict.fromkeys(ARMS, 1))
            loaded = AcrobotPHValue.load(branch / "online" / "checkpoint_00000001.pt")
            self.assertEqual(loaded.updates, 4)
            self.assertEqual(json.loads((branch / "config.json").read_text())["initial_updates"], 3)

    def test_rejects_changing_objective_before_writing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            for flags in (["--finite-horizon"], ["--no-finite-horizon", "--mode", "stage_a_soft_auto"]):
                output = Path(tmp) / "invalid"
                with self.assertRaises(ValueError):
                    run(parser().parse_args(["--output", str(output), *flags]))
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
