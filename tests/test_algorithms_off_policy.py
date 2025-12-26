# tests/test_algorithms_off_policy.py
import numpy as np

from algorithms.off_policy import OffPolicyAlgorithm
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
