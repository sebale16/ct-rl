# tests/test_algorithms_on_policy.py
import torch as th
from typing import Tuple

from algorithms.on_policy import OnPolicyAlgorithm
from models import ActorVCriticModel
from .test_algorithms_base import AlgorithmTest
from common.buffers import RolloutBatch


class DummyOnPolicyAlgorithm(OnPolicyAlgorithm):
    """A minimal on-policy algorithm for testing the base class."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _policy_act_value(
        self, obs_tensor: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Returns dummy actions, log_probs, and values."""
        n_envs = obs_tensor.shape[0]
        action_dim = self.action_dim
        actions = th.zeros((n_envs, action_dim), device=self.device)
        log_probs = th.zeros(n_envs, device=self.device)
        values = th.zeros(n_envs, device=self.device)
        return actions, log_probs, values

    def _train_critic_batch(self, batch: RolloutBatch) -> None:
        pass

    def _train_actor_batch(self, batch: RolloutBatch) -> None:
        pass

    def _train_actor_critic_batch(self, batch: RolloutBatch) -> None:
        pass


class TestOnPolicyAlgorithms(AlgorithmTest):
    def test_learn_runs(self):
        """Tests the base OnPolicyAlgorithm using the dummy implementation."""
        model_kwargs = {"v_net_arch": [16], "pi_net_arch": [16]}
        algo_kwargs = {
            "n_steps": 10,
            "batch_size": 4,
            "n_epochs": 2,
            "seed": 123,
        }
        # Test with a vectorized environment
        self._test_learn_runs(
            algo_class=DummyOnPolicyAlgorithm,
            algo_kwargs=algo_kwargs,
            model_class=ActorVCriticModel,
            model_kwargs=model_kwargs,
            is_vec_env=True,
        )
