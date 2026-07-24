# tests/test_common_callbacks.py

import unittest
import os
import shutil
from unittest.mock import MagicMock, patch

import numpy as np

from common.callbacks import (
    BaseCallback,
    CallbackList,
    EveryNTimesteps,
    CheckpointCallback,
    ProgressBarCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
    StopTrainingOnNoModelImprovement,
    StopTrainingOnMaxEpisodes,
)
from evaluations.evaluation_helpers import EpisodeEvaluationResults
from evaluations.sustained_capture import SustainedCaptureSpec


class DummyModel:
    def __init__(self) -> None:
        self.num_timesteps = 0


class MockAlgorithm:
    def __init__(self):
        self.num_timesteps = 0
        self.save = MagicMock()
        self.logger = MagicMock()
        self.model = MagicMock()


class CounterCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.trigger_count = 0
        self.stop_on_step = -1

    def _on_step(self) -> bool:
        self.trigger_count += 1
        if self.n_calls == self.stop_on_step:
            return False
        return True


class TestCallbacks(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_checkpoints"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_base_callback_on_step_counts_calls(self):
        model = DummyModel()
        cb = CounterCallback()
        cb.init_callback(model)
        cb.on_training_start({}, {})
        for _ in range(5):
            model.num_timesteps += 1
            cb.on_step()
        self.assertEqual(cb.n_calls, 5)
        self.assertEqual(cb.trigger_count, 5)

    def test_callback_list_chains_callbacks(self):
        model = DummyModel()
        c1 = CounterCallback()
        c2 = CounterCallback()
        clist = CallbackList([c1, c2])
        clist.init_callback(model)
        clist.on_training_start({}, {})

        for _ in range(3):
            model.num_timesteps += 1
            clist.on_step()

        self.assertEqual(c1.trigger_count, 3)
        self.assertEqual(c2.trigger_count, 3)

    def test_callback_list_stops_on_false(self):
        model = DummyModel()
        c1 = CounterCallback()
        c2 = CounterCallback()
        c2.stop_on_step = 3  # c2 will return False on the 3rd call
        clist = CallbackList([c1, c2])
        clist.init_callback(model)

        continue_training = True
        for i in range(5):
            if not clist.on_step():
                continue_training = False
                break
        self.assertFalse(continue_training)
        self.assertEqual(c2.n_calls, 3)

    def test_every_n_timesteps_triggers_inner(self):
        model = DummyModel()
        inner = CounterCallback()
        cb = EveryNTimesteps(n_steps=5, callback=inner)
        cb.init_callback(model)
        cb.on_training_start({}, {})

        for t in range(1, 16):
            model.num_timesteps = t
            cb.on_step()

        # Should trigger at timesteps 5, 10, 15 -> 3 times
        self.assertEqual(inner.trigger_count, 3)

    def test_checkpoint_callback_saves_model(self):
        algo = MockAlgorithm()
        cb = CheckpointCallback(
            save_freq=10, save_path=self.test_dir, name_prefix="test_model"
        )
        cb.init_callback(algo)

        for t in range(1, 25):
            algo.num_timesteps = t
            cb.on_step()

        # Should be called at timesteps 10 and 20
        self.assertEqual(algo.save.call_count, 2)
        algo.save.assert_any_call(
            os.path.join(self.test_dir, "test_model_10_steps.pth")
        )
        algo.save.assert_any_call(
            os.path.join(self.test_dir, "test_model_20_steps.pth")
        )

    @patch("common.callbacks.tqdm")
    def test_progress_bar_callback(self, mock_tqdm):
        algo = MockAlgorithm()
        algo._total_timesteps = 100
        pbar_instance = MagicMock()
        mock_tqdm.return_value = pbar_instance

        cb = ProgressBarCallback()
        cb.init_callback(algo)
        cb.on_training_start({}, {})

        mock_tqdm.assert_called_once_with(total=100)

        for t in range(1, 11):
            algo.num_timesteps = t * 5
            cb.on_step()

        self.assertEqual(pbar_instance.update.call_count, 10)
        pbar_instance.update.assert_called_with(5)  # check last call

        cb.on_training_end()
        pbar_instance.close.assert_called_once()

    @patch("common.callbacks.evaluate_policy_per_episode")
    def test_eval_callback(self, mock_evaluate):
        algo = MockAlgorithm()
        mock_eval_env = MagicMock()
        mock_evaluate.return_value = ([100.0], [10])  # (rewards, lengths)

        cb = EvalCallback(
            eval_env=mock_eval_env,
            eval_freq=10,
            n_eval_episodes=1,
            best_model_save_path=self.test_dir,
        )
        cb.init_callback(algo)

        for t in range(1, 25):
            algo.num_timesteps = t
            cb.on_step()

        # Should be called at timesteps 10 and 20
        self.assertEqual(mock_evaluate.call_count, 2)
        self.assertEqual(algo.save.call_count, 1)  # Only saves on new best
        algo.save.assert_called_with(os.path.join(self.test_dir, "best_model.pth"))

        # Check if logs were recorded
        self.assertTrue(algo.logger.record.called)
        algo.logger.record.assert_any_call("eval/mean_reward", 100.0)

    @patch("common.callbacks.evaluate_policy_per_episode")
    def test_eval_callback_log_prefix_namespaces_metrics(self, mock_evaluate):
        algo = MockAlgorithm()
        mock_evaluate.return_value = ([100.0], [10])

        cb = EvalCallback(
            eval_env=MagicMock(),
            eval_freq=10,
            n_eval_episodes=1,
            best_model_save_path=self.test_dir,
            gate_occupancy_key="acrobot_hold",
            log_prefix="eval_hanging",
        )
        cb.init_callback(algo)

        mock_evaluate.return_value = ([100.0], [10], [0.3])
        algo.num_timesteps = 10
        cb.on_step()

        recorded = {c.args[0] for c in algo.logger.record.call_args_list}
        self.assertIn("eval_hanging/mean_reward", recorded)
        self.assertIn("eval_hanging/hold_occupancy", recorded)
        # The default namespace must not leak when a prefix is set, and the
        # shared time key stays un-prefixed.
        self.assertNotIn("eval/mean_reward", recorded)
        self.assertIn("time/total_timesteps", recorded)

    @patch("common.callbacks.evaluate_policy_per_episode")
    def test_eval_callback_selects_strict_capture_rank_not_reward(
        self, mock_evaluate
    ):
        algo = MockAlgorithm()
        mock_evaluate.side_effect = [
            EpisodeEvaluationResults(
                returns=[10.0, 10.0],
                lengths=[10, 10],
                capture_successes=[False, False],
                capture_durations=[0.3, 0.2],
            ),
            EpisodeEvaluationResults(
                returns=[1000.0, 1000.0],
                lengths=[10, 10],
                capture_successes=[False, False],
                capture_durations=[0.1, 0.1],
            ),
            EpisodeEvaluationResults(
                returns=[-5.0, -5.0],
                lengths=[10, 10],
                capture_successes=[True, False],
                capture_durations=[1.0, 0.0],
            ),
        ]

        cb = EvalCallback(
            eval_env=MagicMock(),
            eval_freq=10,
            n_eval_episodes=2,
            best_model_save_path=self.test_dir,
            log_path=os.path.join(self.test_dir, "eval"),
            capture_spec=SustainedCaptureSpec(),
        )
        cb.init_callback(algo)

        algo.num_timesteps = 10
        cb.on_step()
        self.assertEqual(algo.save.call_count, 1)

        # Reward improves dramatically, but strict capture gets worse.
        algo.num_timesteps = 20
        cb.on_step()
        self.assertEqual(algo.save.call_count, 1)
        self.assertEqual(cb.best_mean_reward, 10.0)

        # A higher strict success rate wins even with a lower reward.
        algo.num_timesteps = 30
        cb.on_step()
        self.assertEqual(algo.save.call_count, 2)
        self.assertEqual(cb.best_capture_success_rate, 0.5)
        self.assertEqual(cb.best_capture_duration, 0.5)
        self.assertEqual(cb.best_mean_reward, -5.0)
        algo.save.assert_called_with(
            os.path.join(self.test_dir, "best_model.pth")
        )

        recorded = {c.args[0] for c in algo.logger.record.call_args_list}
        self.assertIn("eval/strict_capture_success_rate", recorded)
        self.assertIn("eval/strict_capture_mean_max_duration", recorded)

        saved = np.load(
            os.path.join(self.test_dir, "eval", "evaluations.npz"),
            allow_pickle=True,
        )
        np.testing.assert_array_equal(saved["capture_timesteps"], [10, 20, 30])
        np.testing.assert_array_equal(
            np.asarray(saved["capture_successes"].tolist(), dtype=bool),
            [[False, False], [False, False], [True, False]],
        )

    def test_stop_on_reward_threshold(self):
        # This callback needs a parent EvalCallback
        parent = MagicMock()
        parent.best_mean_reward = -np.inf

        cb = StopTrainingOnRewardThreshold(reward_threshold=100.0)
        cb.parent = parent

        # First step, reward is low
        parent.best_mean_reward = 50.0
        self.assertTrue(cb._on_step())

        # Second step, reward meets threshold
        parent.best_mean_reward = 101.0
        self.assertFalse(cb._on_step())

    def test_stop_on_no_improvement(self):
        parent = MagicMock()
        parent.best_mean_reward = -np.inf

        cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=2, min_evals=1)
        cb.parent = parent

        # Eval 1: new best
        parent.best_mean_reward = 100.0
        self.assertTrue(cb._on_step())

        # Eval 2: no improvement
        parent.best_mean_reward = 90.0
        self.assertTrue(cb._on_step())

        # Eval 3: no improvement again, should stop
        parent.best_mean_reward = 95.0
        self.assertFalse(cb._on_step())

    def test_stop_on_max_episodes(self):
        algo = MockAlgorithm()
        cb = StopTrainingOnMaxEpisodes(max_episodes=3)
        cb.init_callback(algo)

        # Episode 1 done
        cb.update_locals({"dones": [True]})
        self.assertTrue(cb.on_step())

        # Episode 2 done
        cb.update_locals({"dones": [True]})
        self.assertTrue(cb.on_step())

        # No episode done
        cb.update_locals({"dones": [False]})
        self.assertTrue(cb.on_step())

        # Episode 3 done, should stop
        cb.update_locals({"dones": [True]})
        self.assertFalse(cb.on_step())


if __name__ == "__main__":
    unittest.main()
