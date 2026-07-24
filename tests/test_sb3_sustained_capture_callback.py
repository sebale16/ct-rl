from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np

try:
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError as exc:  # pragma: no cover - exercised only without SB3
    SB3_IMPORT_ERROR = exc
else:
    from common.sb3_callbacks import (
        SB3CaptureEvaluation,
        SustainedCaptureEvalCallback,
        evaluate_sb3_policy_with_capture,
    )
    from evaluations.sustained_capture import (
        STRICT_CAPTURE_INFO_KEY,
        SustainedCaptureSpec,
    )
    SB3_IMPORT_ERROR = None


class _ScriptedPolicy:
    def __init__(self) -> None:
        self.episode_starts: list[np.ndarray] = []

    def predict(
        self,
        observations,
        *,
        state,
        episode_start,
        deterministic,
    ):
        del state, deterministic
        self.episode_starts.append(np.asarray(episode_start, dtype=bool).copy())
        return np.zeros((len(observations), 1), dtype=np.float32), None


class _ScriptedCaptureVecEnv:
    """Two auto-reset slots with deliberately unequal episode quotas."""

    num_envs = 2

    def __init__(self) -> None:
        # An episode specifies its reset predicate, endpoint transitions, and
        # Monitor-provided terminal return. Slot 0 finishes every global step;
        # slot 1 needs two episodes, so slot 0 must be ignored after its quota.
        self._episodes = [
            [
                {
                    "initial": True,
                    "steps": [(True, 1.0, 1.0)],
                    "monitor_reward": 101.0,
                }
            ],
            [
                {
                    "initial": False,
                    "steps": [
                        (True, 0.6, 2.0),
                        (True, 0.6, 3.0),
                    ],
                    "monitor_reward": 202.0,
                },
                {
                    "initial": True,
                    "steps": [(True, 1.0, 4.0)],
                    "monitor_reward": 303.0,
                },
            ],
        ]
        self.step_calls = 0
        self.reset_infos: list[dict[str, float]] = []
        self._episode_indices = [0, 0]
        self._step_indices = [0, 0]

    def _episode(self, slot: int) -> dict:
        episodes = self._episodes[slot]
        return episodes[self._episode_indices[slot] % len(episodes)]

    def reset(self):
        self._episode_indices = [0, 0]
        self._step_indices = [0, 0]
        self.reset_infos = [
            {STRICT_CAPTURE_INFO_KEY: float(self._episode(i)["initial"])}
            for i in range(self.num_envs)
        ]
        return np.zeros((self.num_envs, 1), dtype=np.float32)

    def step(self, actions):
        del actions
        self.step_calls += 1
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict] = []

        for slot in range(self.num_envs):
            episode = self._episode(slot)
            step_index = self._step_indices[slot]
            inside, dt_used, reward = episode["steps"][step_index]
            rewards[slot] = reward
            done = step_index + 1 == len(episode["steps"])
            dones[slot] = done
            info = {
                STRICT_CAPTURE_INFO_KEY: float(inside),
                "dt_used": float(dt_used),
            }

            if done:
                info["episode"] = {
                    "r": float(episode["monitor_reward"]),
                    "l": len(episode["steps"]),
                }
                self._episode_indices[slot] += 1
                self._step_indices[slot] = 0
                # Match DummyVecEnv: the terminal info describes the episode
                # just completed, while reset_infos already describes the
                # automatically reset next episode.
                self.reset_infos[slot] = {
                    STRICT_CAPTURE_INFO_KEY: float(
                        self._episode(slot)["initial"]
                    )
                }
            else:
                self._step_indices[slot] += 1
            infos.append(info)

        observations = np.zeros((self.num_envs, 1), dtype=np.float32)
        return observations, rewards, dones, infos

    def render(self):
        raise AssertionError("render must not be called")


class _DelegatingVecWrapper:
    """Expose reset infos only through ``venv``, as SB3 wrappers may do."""

    def __init__(self, venv: _ScriptedCaptureVecEnv) -> None:
        self.venv = venv
        self.num_envs = venv.num_envs

    def reset(self):
        return self.venv.reset()

    def step(self, actions):
        return self.venv.step(actions)

    def render(self):
        return self.venv.render()


if SB3_IMPORT_ERROR is None:

    class _MinimalGymEnv(gym.Env):
        observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            del options
            return np.zeros(1, dtype=np.float32), {
                STRICT_CAPTURE_INFO_KEY: 0.0
            }

        def step(self, action):
            del action
            return (
                np.zeros(1, dtype=np.float32),
                0.0,
                False,
                False,
                {
                    STRICT_CAPTURE_INFO_KEY: 0.0,
                    "dt_used": 0.01,
                },
            )


@unittest.skipIf(
    SB3_IMPORT_ERROR is not None,
    f"Stable-Baselines3 unavailable: {SB3_IMPORT_ERROR}",
)
class SB3SustainedCaptureTests(unittest.TestCase):
    def test_evaluator_honors_vector_quotas_and_auto_reset_infos(self):
        inner_env = _ScriptedCaptureVecEnv()
        env = _DelegatingVecWrapper(inner_env)
        policy = _ScriptedPolicy()

        result = evaluate_sb3_policy_with_capture(
            policy,
            env,
            n_eval_episodes=3,
            deterministic=True,
            render=False,
            capture_spec=SustainedCaptureSpec(),
        )

        # SB3's quota formula assigns one episode to slot 0 and two to slot 1.
        # Slot 0 actually completes three episodes while slot 1 reaches its
        # quota, but only its first may appear in the evaluation results.
        self.assertEqual(inner_env.step_calls, 3)
        self.assertEqual(result.rewards, [101.0, 202.0, 303.0])
        self.assertEqual(result.lengths, [1, 2, 1])
        self.assertEqual(result.capture_successes, [True, False, True])
        np.testing.assert_allclose(
            result.capture_durations, [1.0, 0.6, 1.0]
        )

        # Auto-reset dones must be forwarded to recurrent policies slot-wise.
        np.testing.assert_array_equal(policy.episode_starts[0], [True, True])
        np.testing.assert_array_equal(policy.episode_starts[1], [True, False])
        np.testing.assert_array_equal(policy.episode_starts[2], [True, True])

    def test_callback_reseeds_persists_and_selects_only_by_capture_rank(self):
        eval_env = DummyVecEnv([_MinimalGymEnv])
        original_seed = eval_env.seed
        eval_env.seed = MagicMock(wraps=original_seed)

        model = MagicMock()
        model.num_timesteps = 0
        model.get_env.return_value = eval_env
        model.get_vec_normalize_env.return_value = None
        model.logger = MagicMock()

        evaluations = [
            # First result always establishes a checkpoint.
            SB3CaptureEvaluation(
                rewards=[10.0, 10.0],
                lengths=[10, 10],
                capture_successes=[False, False],
                capture_durations=[0.4, 0.5],
            ),
            # Much better reward, but a worse strict rank: do not save.
            SB3CaptureEvaluation(
                rewards=[1000.0, 1000.0],
                lengths=[10, 10],
                capture_successes=[False, False],
                capture_durations=[0.3, 0.4],
            ),
            # Worse reward, same success rate, longer residence: save.
            SB3CaptureEvaluation(
                rewards=[-10.0, -10.0],
                lengths=[10, 10],
                capture_successes=[False, False],
                capture_durations=[0.5, 0.6],
            ),
            # Worse reward again, but a higher success rate: save.
            SB3CaptureEvaluation(
                rewards=[-100.0, -100.0],
                lengths=[10, 10],
                capture_successes=[True, False],
                capture_durations=[1.0, 0.2],
            ),
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            callback = SustainedCaptureEvalCallback(
                eval_env,
                capture_spec=SustainedCaptureSpec(),
                reset_seed=1234,
                n_eval_episodes=2,
                eval_freq=1,
                log_path=str(root / "eval"),
                best_model_save_path=str(root / "best"),
                deterministic=True,
                render=False,
                verbose=0,
            )
            callback.init_callback(model)

            with patch(
                "common.sb3_callbacks.evaluate_sb3_policy_with_capture",
                side_effect=evaluations,
            ) as evaluate:
                for timestep in range(1, len(evaluations) + 1):
                    model.num_timesteps = timestep
                    self.assertTrue(callback.on_step())

            self.assertEqual(evaluate.call_count, len(evaluations))
            self.assertEqual(
                eval_env.seed.call_args_list,
                [call(1234)] * len(evaluations),
            )
            self.assertEqual(model.save.call_count, 3)
            model.save.assert_called_with(str(root / "best" / "best_model"))
            self.assertEqual(callback.best_capture_success_rate, 0.5)
            self.assertAlmostEqual(callback.best_capture_duration, 0.6)
            # The legacy attribute follows the strictly selected checkpoint;
            # the much larger unselected reward cannot break a capture tie.
            self.assertEqual(callback.best_mean_reward, -100.0)

            saved = np.load(root / "eval" / "evaluations.npz")
            np.testing.assert_array_equal(saved["timesteps"], [1, 2, 3, 4])
            np.testing.assert_array_equal(
                saved["capture_successes"],
                [
                    [False, False],
                    [False, False],
                    [False, False],
                    [True, False],
                ],
            )
            np.testing.assert_allclose(
                saved["capture_durations"],
                [
                    [0.4, 0.5],
                    [0.3, 0.4],
                    [0.5, 0.6],
                    [1.0, 0.2],
                ],
            )
            np.testing.assert_allclose(
                saved["results"],
                [
                    [10.0, 10.0],
                    [1000.0, 1000.0],
                    [-10.0, -10.0],
                    [-100.0, -100.0],
                ],
            )

            recorded_keys = {
                invocation.args[0]
                for invocation in model.logger.record.call_args_list
            }
            self.assertIn("eval/strict_capture_success_rate", recorded_keys)
            self.assertIn(
                "eval/best_strict_capture_success_rate", recorded_keys
            )


if __name__ == "__main__":
    unittest.main()
