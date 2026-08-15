# tests/test_algorithms_off_policy.py
import numpy as np

from algorithms.off_policy import OffPolicyAlgorithm
from environment import DMCContinuousEnv
from models import ActorQCriticModel
from .test_algorithms_base import AlgorithmTest


class DummyOffPolicyAlgorithm(OffPolicyAlgorithm):
    """A minimal off-policy algorithm for testing the base class."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _policy_act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Returns a dummy action (e.g., zeros)."""
        if self.is_vec_env:
            return np.zeros((self.n_envs, self.action_dim))
        else:
            return np.zeros(self.action_dim)

    def train(self, gradient_steps: int, batch_size: int) -> None:
        """Does nothing, just for fulfilling the abstract method requirement."""
        pass


class TestOffPolicyAlgorithms(AlgorithmTest):
    def _agent(self):
        env = DMCContinuousEnv(
            "cartpole", "swingup", episode_duration=0.1, dt=0.02
        )
        self.addCleanup(env.close)
        model = ActorQCriticModel(
            observation_space=env.observation_space,
            action_space=env.action_space,
            q_net_arch=[8],
            pi_net_arch=[8],
            device="cpu",
        )
        return DummyOffPolicyAlgorithm(
            env=env,
            model=model,
            device="cpu",
            buffer_size=8,
            batch_size=2,
        )

    def test_learn_runs(self):
        """Tests the base OffPolicyAlgorithm using the dummy implementation."""
        model_kwargs = {"q_net_arch": [16], "pi_net_arch": [16]}
        algo_kwargs = {
            "learning_starts": 10,
            "batch_size": 4,
            "buffer_size": 100,
            "gradient_steps": 1,
            "train_freq": 1,
            "seed": 123,
        }

        # Test with a vectorized environment
        self._test_learn_runs(
            algo_class=DummyOffPolicyAlgorithm,
            algo_kwargs=algo_kwargs,
            model_class=ActorQCriticModel,
            model_kwargs=model_kwargs,
            is_vec_env=True,
        )

    def test_store_distinguishes_truncation_from_cap_failure(self):
        agent = self._agent()
        obs_dim = agent.env.observation_space.shape[0]
        act_dim = agent.env.action_space.shape[0]
        obs = np.zeros((1, obs_dim), dtype=np.float32)
        reset_obs = np.full((1, obs_dim), -9.0, dtype=np.float32)
        terminal_obs = np.full(obs_dim, 3.0, dtype=np.float32)
        action = np.zeros((1, act_dim), dtype=np.float32)

        # A vector-style time-limit reset is an episode boundary, but not a
        # critic terminal. Its true terminal observation/time must be stored.
        agent._store_transition(
            obs=obs,
            action=action,
            reward=np.array([-0.5], dtype=np.float32),
            done=np.array([True]),
            next_obs=reset_obs.copy(),
            t=np.array([0.08], dtype=np.float64),
            next_t=np.array([0.0], dtype=np.float64),
            infos=[
                {
                    "terminal_observation": terminal_obs,
                    "terminal_next_t": 0.1,
                }
            ],
            terminated=np.array([False]),
            truncated=np.array([True]),
        )
        buf = agent.replay_buffer
        self.assertEqual(buf.dones[0, 0], 0.0)
        self.assertEqual(buf.episode_ends[0, 0], 1.0)
        self.assertEqual(buf.cap_failures[0, 0], 0.0)
        np.testing.assert_array_equal(buf.next_observations[0, 0], terminal_obs)
        self.assertAlmostEqual(float(buf.dt[0, 0]), 0.02, places=7)

        # A cap is both a true terminal and an episode boundary, with its
        # analytical continuation primitives carried separately.
        agent._store_transition(
            obs=obs,
            action=action,
            reward=np.array([-1.0], dtype=np.float32),
            done=np.array([True]),
            next_obs=reset_obs.copy(),
            t=np.array([0.02], dtype=np.float64),
            next_t=np.array([0.0], dtype=np.float64),
            infos=[
                {
                    "terminal_observation": terminal_obs,
                    "terminal_next_t": 0.04,
                    "absorbing_failure": 1.0,
                    "absorbing_failure_reward_rate": -1.0,
                    "absorbing_failure_remaining_seconds": 19.96,
                }
            ],
            terminated=np.array([True]),
            truncated=np.array([False]),
        )
        self.assertEqual(buf.dones[1, 0], 1.0)
        self.assertEqual(buf.episode_ends[1, 0], 1.0)
        self.assertEqual(buf.cap_failures[1, 0], 1.0)
        self.assertEqual(buf.failure_reward_rates[1, 0], -1.0)
        self.assertAlmostEqual(
            float(buf.failure_remaining_times[1, 0]), 19.96, places=5
        )

    def test_store_accepts_zero_and_positive_endpoint_continuation_rates(self):
        for rate, vector_info in ((0.0, False), (2.5, True)):
            with self.subTest(rate=rate, vector_info=vector_info):
                agent = self._agent()
                obs_dim = agent.env.observation_space.shape[0]
                act_dim = agent.env.action_space.shape[0]
                obs = np.zeros((1, obs_dim), dtype=np.float32)
                terminal_obs = np.ones(obs_dim, dtype=np.float32)
                info = {
                    "terminal_observation": terminal_obs,
                    "terminal_next_t": 0.02,
                    "absorbing_failure": 1.0,
                    "absorbing_failure_reward_rate": rate,
                    "absorbing_failure_remaining_seconds": 0.08,
                }
                agent._store_transition(
                    obs=obs,
                    action=np.zeros((1, act_dim), dtype=np.float32),
                    reward=np.array([rate], dtype=np.float32),
                    done=np.array([True]),
                    next_obs=np.zeros_like(obs),
                    t=np.array([0.0]),
                    next_t=np.array([0.0]),
                    infos=[info] if vector_info else info,
                    terminated=np.array([True]),
                    truncated=np.array([False]),
                )
                self.assertEqual(agent.replay_buffer.cap_failures[0, 0], 1.0)
                self.assertEqual(
                    agent.replay_buffer.failure_reward_rates[0, 0], rate
                )
