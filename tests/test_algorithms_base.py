# tests/test_algorithms_base.py
import unittest
from typing import Type

from algorithms.base import BaseAlgorithm
from environment import DMCContinuousEnv, VecContinuousEnv, Monitor


class AlgorithmTest(unittest.TestCase):
    def _test_learn_runs(
        self,
        algo_class: Type[BaseAlgorithm],
        algo_kwargs: dict,
        model_class,
        model_kwargs: dict,
        is_vec_env: bool,
    ):
        """
        Generic test to ensure the learn method runs without crashing.
        """
        if is_vec_env:
            n_envs = 2
            env_fns = [
                lambda: Monitor(
                    DMCContinuousEnv(
                        "cartpole", "swingup", episode_duration=0.1, dt=0.02
                    )
                )
                for _ in range(n_envs)
            ]
            env = VecContinuousEnv(env_fns)
        else:
            env = DMCContinuousEnv("cartpole", "swingup", episode_duration=0.1, dt=0.02)

        model = model_class(
            observation_space=env.observation_space,
            action_space=env.action_space,
            **model_kwargs,
        )
        total_timesteps = 20
        agent = algo_class(env=env, model=model, **algo_kwargs)
        agent.learn(total_timesteps=total_timesteps)
        self.assertGreaterEqual(agent.num_timesteps, total_timesteps)
