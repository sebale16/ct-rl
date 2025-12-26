# tests/test_algorithms_cppo.py
import unittest

from algorithms.cppo import CPPO
from models import ActorVCriticModel
from .test_algorithms_base import (
    AlgorithmTest,
    DMCContinuousEnv,
    VecContinuousEnv,
    Monitor,
)


class TestCPPO(AlgorithmTest):
    def test_learn_runs_single_env(self):
        model_kwargs = {"v_net_arch": [16], "pi_net_arch": [16]}
        algo_kwargs = {
            "n_steps": 10,
            "batch_size": 4,
            "n_epochs": 2,
            "seed": 123,
        }
        try:
            self._test_learn_runs(
                CPPO,
                algo_kwargs,
                ActorVCriticModel,
                model_kwargs,
                is_vec_env=False,
            )
        except Exception as e:
            self.fail(f"CPPO.learn() with single env raised an exception: {e}")

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

        agent = CPPO(
            env=vec_env,
            model=ActorVCriticModel,
            model_kwargs={"v_net_arch": [16], "pi_net_arch": [16]},
            n_steps=10,
            batch_size=4,
            n_epochs=2,
            seed=123,
        )

        try:
            agent.learn(total_timesteps=20)
        except Exception as e:
            self.fail(f"CPPO.learn() with vectorized env raised an exception: {e}")
