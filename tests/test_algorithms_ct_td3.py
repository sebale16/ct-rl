import unittest
import torch as th
import numpy as np

from environment import DMCContinuousEnv, VecContinuousEnv, Monitor
from environment import DMCContinuousEnv, VecContinuousEnv, Monitor
from algorithms.ct_td3 import CTTD3
from models.actor_q_critic import ActorQCriticModel
from models.noise import GaussianActionNoise


class TestCTTD3(unittest.TestCase):
    def setUp(self):
        """
        Set up a small environment and a CTTD3 agent to test.
        """
        self.env = DMCContinuousEnv(
            domain_name="cartpole",
            task_name="swingup",
            time_sampling="uniform",
            dt=0.02,
            episode_duration=0.1,  # short episodes
        )

        self.model = ActorQCriticModel(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            q_net_arch=[16, 16],
            pi_net_arch=[16, 16],
            deterministic_policy=True,
            use_actor_target=True,
        )

        action_dim = self.env.action_space.shape[-1]
        self.agent = CTTD3(
            env=self.env,
            model=self.model,
            learning_starts=10,
            batch_size=8,
            buffer_size=100,
            action_noise=GaussianActionNoise(
                mean=np.zeros(action_dim), sigma=0.2 * np.ones(action_dim)
            ),
            target_action_noise=GaussianActionNoise(
                mean=np.zeros(action_dim), sigma=0.1 * np.ones(action_dim)
            ),
            seed=123,
        )

    def test_learn_runs(self):
        """
        Test that the learn method runs for a few timesteps without crashing.
        """
        try:
            self.agent.learn(total_timesteps=20)
        except Exception as e:
            self.fail(f"agent.learn() raised an exception: {e}")

    def test_learn_runs_vectorized(self):
        """
        Test that the learn method runs with a vectorized environment.
        """
        n_envs = 3
        env_fns = [
            lambda: Monitor(
                DMCContinuousEnv("cartpole", "swingup", episode_duration=0.1, dt=0.02)
            )
            for _ in range(n_envs)
        ]
        vec_env = VecContinuousEnv(env_fns)

        action_dim = vec_env.action_space.shape[-1]
        agent = CTTD3(
            env=vec_env,
            model="ActorQCriticModel",
            model_kwargs={
                "q_net_arch": [16],
                "pi_net_arch": [16],
                "deterministic_policy": True,
                "use_actor_target": True,
            },
            learning_starts=10,
            batch_size=8,
            buffer_size=100,
            action_noise=GaussianActionNoise(
                mean=np.zeros(action_dim), sigma=0.1 * np.ones(action_dim)
            ),
            seed=123,
        )
        try:
            agent.learn(total_timesteps=20)
        except Exception as e:
            self.fail(f"agent.learn() with vectorized env raised an exception: {e}")

    def test_learn_runs_vectorized(self):
        """
        Test that the learn method runs with a vectorized environment.
        """
        n_envs = 3
        env_fns = [
            lambda: Monitor(
                DMCContinuousEnv("cartpole", "swingup", episode_duration=0.1, dt=0.02)
            )
            for _ in range(n_envs)
        ]
        vec_env = VecContinuousEnv(env_fns)

        action_dim = vec_env.action_space.shape[-1]
        agent = CTTD3(
            env=vec_env,
            model="ActorQCriticModel",
            model_kwargs={
                "q_net_arch": [16],
                "pi_net_arch": [16],
                "deterministic_policy": True,
                "use_actor_target": True,
            },
            learning_starts=10,
            batch_size=8,
            buffer_size=100,
            action_noise=GaussianActionNoise(
                mean=np.zeros(action_dim), sigma=0.1 * np.ones(action_dim)
            ),
            seed=123,
        )
        try:
            agent.learn(total_timesteps=20)
        except Exception as e:
            self.fail(f"agent.learn() with vectorized env raised an exception: {e}")


class TestDemonstrationWarmStart(unittest.TestCase):
    """CTTD3's replay-buffer warm start, mirroring algorithms.ct_sac.CTSAC's
    (see tests/test_algorithms_ct_sac.py::TestDemonstrationWarmStart) minus
    the imitation-loss term, which CTTD3 doesn't implement."""

    def setUp(self):
        self.env = DMCContinuousEnv(
            domain_name="cartpole",
            task_name="swingup",
            time_sampling="uniform",
            dt=0.02,
            episode_duration=0.1,
        )
        self.model = ActorQCriticModel(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            q_net_arch=[16, 16],
            pi_net_arch=[16, 16],
            deterministic_policy=True,
            use_actor_target=True,
        )
        self.demo_action = self.env.action_space.low
        action_dim = self.env.action_space.shape[-1]
        self.action_noise = GaussianActionNoise(
            mean=np.zeros(action_dim), sigma=0.2 * np.ones(action_dim)
        )

    def _agent(self, **kwargs):
        return CTTD3(
            env=self.env,
            model=self.model,
            learning_starts=10,
            batch_size=8,
            buffer_size=100,
            action_noise=self.action_noise,
            seed=123,
            **kwargs,
        )

    def test_unset_demonstration_policy_matches_base_class_default(self):
        agent = self._agent()
        self.assertIsNone(agent.demonstration_policy)
        self.assertEqual(agent.demonstration_steps, agent.learning_starts)

    def test_demonstration_steps_overrides_learning_starts(self):
        calls = []
        agent = self._agent(
            demonstration_policy=lambda obs: calls.append(1) or self.demo_action,
            demonstration_steps=3,
        )
        obs = self.env.observation_space.sample()
        for t in range(10):
            agent.num_timesteps = t
            agent._sample_action(obs)
        self.assertEqual(len(calls), 3)

    def test_negative_demonstration_steps_rejected(self):
        with self.assertRaises(ValueError):
            self._agent(
                demonstration_policy=lambda obs: self.demo_action,
                demonstration_steps=-1,
            )

    def test_learn_runs_with_demonstration_policy_and_calls_it_the_configured_count(
        self,
    ):
        calls = []
        agent = self._agent(
            demonstration_policy=lambda obs: calls.append(1) or self.demo_action,
            demonstration_steps=5,
        )
        try:
            agent.learn(total_timesteps=15)
        except Exception as e:
            self.fail(f"agent.learn() with demonstration_policy raised: {e}")
        # n_envs=1, so num_timesteps advances by 1 per _sample_action call:
        # exactly the first `demonstration_steps` calls defer to the demo.
        self.assertEqual(len(calls), 5)

    def test_stateful_demonstration_policy_is_reset_on_episode_boundaries(self):
        reset_calls = []

        class _StatefulDemo:
            def __call__(self, obs):
                return self.env_action_low

            def reset(self):
                reset_calls.append(1)

        demo = _StatefulDemo()
        demo.env_action_low = self.demo_action
        agent = self._agent(demonstration_policy=demo, demonstration_steps=15)
        agent.learn(total_timesteps=15)
        self.assertGreater(len(reset_calls), 0)
