import unittest
import torch as th

from environment import DMCContinuousEnv, VecContinuousEnv, Monitor
from algorithms.ct_sac import CTSAC
from models.actor_q_critic import ActorQCriticModel


class TestCTSAC(unittest.TestCase):
    def setUp(self):
        """
        Set up a small environment and a CTSAC agent to test.
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
        )

        self.agent = CTSAC(
            env=self.env,
            model=self.model,
            learning_starts=10,
            batch_size=8,
            buffer_size=100,
            gradient_steps=2,
            train_freq=2,
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

        agent = CTSAC(
            env=vec_env,
            model="ActorQCriticModel",
            model_kwargs={"q_net_arch": [16], "pi_net_arch": [16]},
            learning_starts=10,
            batch_size=8,
            buffer_size=100,
            seed=123,
        )

        try:
            agent.learn(total_timesteps=20)
        except Exception as e:
            self.fail(f"agent.learn() with vectorized env raised an exception: {e}")
