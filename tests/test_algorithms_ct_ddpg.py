import unittest
import torch as th
import numpy as np

from environment import DMCContinuousEnv, VecContinuousEnv, Monitor
from algorithms.ct_ddpg import CTDDPG
from models.actor_q_critic import ActorQCriticModel
from models.noise import GaussianActionNoise


class TestCTDDPG(unittest.TestCase):
    def setUp(self):
        """
        Set up a small environment and a CTDDPG agent to test.
        """
        self.env = DMCContinuousEnv(
            domain_name="cartpole",
            task_name="swingup",
            time_sampling="uniform",
            dt=0.02,
            episode_duration=0.1,  # short episodes
        )

        model_kwargs = {
            "q_net_arch": [16, 16],
            "pi_net_arch": [16, 16],
            "deterministic_policy": True,
            "use_actor_target": True,
        }

        action_dim = self.env.action_space.shape[-1]
        test_action_noise = GaussianActionNoise(
            mean=np.zeros(action_dim), sigma=0.1 * np.ones(action_dim)
        )
        self.agent = CTDDPG(
            env=self.env,
            model="ActorQCriticModel",
            model_kwargs=model_kwargs,
            learning_starts=10,
            batch_size=8,
            buffer_size=100,
            gradient_steps=2,
            train_freq=2,
            action_noise=test_action_noise,
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

        model_kwargs = {
            "q_net_arch": [16],
            "pi_net_arch": [16],
            "deterministic_policy": True,
            "use_actor_target": True,
        }
        action_dim = vec_env.action_space.shape[-1]
        test_action_noise = GaussianActionNoise(
            mean=np.zeros(action_dim), sigma=0.1 * np.ones(action_dim)
        )

        agent = CTDDPG(
            env=vec_env,
            model="ActorQCriticModel",
            model_kwargs=model_kwargs,
            learning_starts=10,
            batch_size=8,
            buffer_size=100,
            action_noise=test_action_noise,
            seed=123,
        )
        try:
            agent.learn(total_timesteps=20)
        except Exception as e:
            self.fail(f"agent.learn() with vectorized env raised an exception: {e}")
