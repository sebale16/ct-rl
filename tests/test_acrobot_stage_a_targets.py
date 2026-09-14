"""Checks for paired regression controls, held-out probes, and reproducibility."""
from copy import deepcopy
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("MUJOCO_GL", "disable")

import numpy as np
import torch

from algorithms.acrobot_ph_value import AcrobotPHValue, ValueFlowConfig
from benchmarks.check_acrobot_stage_a_targets import batch_indices, labels_for, parser, probe, run


class TestStageATargetDiagnostic(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        self.agent = AcrobotPHValue(config=ValueFlowConfig(hidden_width=8))
        self.data = {
            "states": self.agent.oracle.canonical(torch.tensor([
                [3.1, .05, .1, -.1], [2.9, -.2, -.3, .4], [3., .1, 0., 12.01], [3.2, .03, -.1, .1]
            ])).numpy(),
            "terminal": np.array([False, False, True, False]),
            "rollout": np.array([False, False, True, True]),
        }

    def test_paired_first_step_then_only_moving_labels_change(self):
        # Start with nonempty Adam state as when branching from a checkpoint.
        self.agent.update(self.data["states"])
        labels = labels_for(self.agent, self.data, -450., 2)
        self.assertEqual(labels[2], -450.)
        fixed, moving = deepcopy(self.agent), deepcopy(self.agent)
        initial_target = deepcopy(fixed.target.state_dict())
        fixed.fit_labels(self.data["states"], labels, refresh_target=False)
        moving.update(self.data["states"], terminal_mask=self.data["terminal"], terminal_value=-450.)
        for a, b in zip(fixed.value.parameters(), moving.value.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for name, value in fixed.target.state_dict().items():
            torch.testing.assert_close(value, initial_target[name], rtol=0, atol=0)
        np.testing.assert_array_equal(labels_for(fixed, self.data, -450., 2), labels)
        changed = labels_for(moving, self.data, -450., 2)
        self.assertTrue(np.any(changed[~self.data["terminal"]] != labels[~self.data["terminal"]]))
        self.assertEqual(changed[2], -450.)
        self.assertFalse(torch.equal(next(fixed.value.parameters()), next(self.agent.value.parameters())))

    def test_fixed_label_generalization_and_probe_does_not_train(self):
        # An attainable supervised task must reduce error on unseen states.
        teacher = deepcopy(self.agent)
        with torch.no_grad():
            teacher.value.net[-1].weight.mul_(3.)
        rng = np.random.default_rng(14)
        qv = np.r_[np.pi, 0., 0., 0.] + rng.uniform(-.5, .5, (128, 4))
        states = self.agent.oracle.canonical(torch.tensor(qv, dtype=torch.float32)).numpy()
        with torch.no_grad():
            labels = teacher.value(torch.tensor(states)).numpy()
        before = np.mean((self.agent.value(torch.tensor(states[64:])).detach().numpy() - labels[64:])**2)
        for _ in range(200):
            self.agent.fit_labels(states[:64], labels[:64], refresh_target=False)
        after = np.mean((self.agent.value(torch.tensor(states[64:])).detach().numpy() - labels[64:])**2)
        self.assertLess(after, before * .2)
        self.data["initial_labels"] = labels_for(self.agent, self.data, -450., 2)
        before_state = deepcopy(self.agent.value.state_dict())
        updates = self.agent.updates
        metrics, arrays = probe(self.agent, self.data, self.data["initial_labels"], -450., 2)
        self.assertEqual(metrics["label_drift_rms"], 0.)
        self.assertEqual(metrics["terminal_fraction"], .25)
        np.testing.assert_allclose(arrays["action"], self.agent.act(self.data["states"]).ravel(), atol=1e-8, rtol=1e-5)
        interior = ~self.data["terminal"]
        self.assertAlmostEqual(metrics["hjb_rms"], np.sqrt(np.mean(arrays["hjb"][interior].astype(float)**2)))
        self.assertEqual(self.agent.updates, updates)
        for name, value in self.agent.value.state_dict().items():
            torch.testing.assert_close(value, before_state[name], rtol=0, atol=0)

    def test_minibatches_preserve_mix(self):
        indices = batch_indices(self.data, 7, np.random.default_rng(0))
        self.assertEqual(int(self.data["rollout"][indices].sum()), 4)
        local = {**self.data, "rollout": np.zeros(4, dtype=bool)}
        self.assertEqual(len(batch_indices(local, 7, np.random.default_rng(0))), 7)

    def test_runner_artifacts_reproducibility_and_checkpoint_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            common = ["--updates", "3", "--batch-size", "8", "--train-states", "16", "--heldout-states", "12",
                      "--collection-steps", "16", "--eval-every", "2", "--eval-episodes", "1",
                      "--episode-seconds", ".02", "--hold-seconds", ".01", "--hidden-width", "8"]
            output = Path(tmp) / "first"
            args = parser().parse_args(["--output", str(output), *common])
            with contextlib.redirect_stdout(io.StringIO()):
                summary = run(args)
            self.assertEqual(summary["completed_updates"], {"fixed": 3, "moving": 3})
            with np.load(output / "dataset.npz") as archive:
                saved = {key: archive[key].copy() for key in archive.files}
            self.assertFalse({x.tobytes() for x in saved["train_states"]} & {x.tobytes() for x in saved["heldout_states"]})
            probes = [json.loads(line) for line in (output / "probes.jsonl").read_text().splitlines()]
            self.assertEqual(len(probes), 12)  # two arms, two splits, three assessments
            self.assertTrue(all(row["label_drift_rms"] == 0. for row in probes if row["arm"] == "fixed"))
            evaluations = [json.loads(line) for line in (output / "evaluations.jsonl").read_text().splitlines()]
            self.assertEqual(len(evaluations), 6)
            self.assertEqual(evaluations[0]["groups"], evaluations[1]["groups"])
            for arm in ("fixed", "moving"):
                agent = AcrobotPHValue.load(output / arm / "checkpoint_00000003.pt")
                self.assertEqual(agent.updates, 3)
                self.assertEqual(agent.metadata["arm"], arm)
                self.assertTrue((output / arm / "probe_00000003.npz").exists())
            repeat = Path(tmp) / "repeat"
            with contextlib.redirect_stdout(io.StringIO()):
                run(parser().parse_args(["--output", str(repeat), *common, "--skip-rollouts"]))
            self.assertEqual((output / "metrics.csv").read_bytes(), (repeat / "metrics.csv").read_bytes())
            with np.load(repeat / "dataset.npz") as archive:
                for key in saved:
                    np.testing.assert_array_equal(saved[key], archive[key])
            resume = Path(tmp) / "resume"
            with contextlib.redirect_stdout(io.StringIO()):
                run(parser().parse_args(["--output", str(resume), *common, "--skip-rollouts", "--collection-steps", "0",
                                        "--checkpoint", str(output / "moving" / "checkpoint_00000003.pt")]))
            continued = AcrobotPHValue.load(resume / "moving" / "checkpoint_00000003.pt")
            self.assertEqual(continued.updates, 6)
            self.assertEqual(continued.metadata["initial_updates"], 3)
            with self.assertRaises(FileExistsError):
                run(args)

    def test_nonfinite_arm_is_reported_while_other_arm_continues(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            args = parser().parse_args(["--output", tmp, "--updates", "2", "--train-states", "8",
                                        "--heldout-states", "8", "--collection-steps", "0", "--skip-rollouts"])
            with patch.object(AcrobotPHValue, "update", side_effect=FloatingPointError("non-finite test loss")), \
                    contextlib.redirect_stdout(io.StringIO()):
                summary = run(args)
            self.assertEqual(summary["status"], "nonfinite_failure")
            self.assertEqual(summary["completed_updates"], {"fixed": 2, "moving": 0})
            self.assertEqual(summary["failures"]["moving"]["update"], 1)
            self.assertTrue((Path(tmp) / "fixed" / "checkpoint_00000002.pt").exists())


if __name__ == "__main__":
    unittest.main()
