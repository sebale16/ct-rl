# tests/test_algorithms_q_learning.py
import unittest

from algorithms.q_learning import qLearning
from models import CoupledVqModel
from .test_algorithms_base import AlgorithmTest


class TestqLearning(AlgorithmTest):
    def test_learn_runs_single_env(self):
        model_kwargs = {"v_net_arch": [16], "q_net_arch": [16], "pi_net_arch": [16]}
        algo_kwargs = {
            "learning_starts": 10,
            "batch_size": 4,
            "buffer_size": 100,
            "gradient_steps": 1,
            "train_freq": 1,
            "seed": 123,
        }
        try:
            self._test_learn_runs(
                qLearning,
                algo_kwargs,
                CoupledVqModel,
                model_kwargs,
                is_vec_env=False,
            )
        except Exception as e:
            self.fail(f"CoupledSarsa.learn() with single env raised an exception: {e}")
