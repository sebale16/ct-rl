# tests/test_models_individual_model.py
import unittest

import numpy as np
import torch as th
from gymnasium import spaces

from models import (
    CoupledVqModel,
    ActorVCriticModel,
    ActorQCriticModel,
)
from common.torch_layers import FlattenExtractor, get_device


class TestModelsIndividualModel(unittest.TestCase):
    def setUp(self) -> None:
        self.device = get_device("auto")
        self.obs_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.obs_batch = th.zeros(8, 4).to(self.device)
        self.act_batch = th.zeros(8, 2).to(self.device)

    def test_coupled_vq_model_forward(self):
        model = CoupledVqModel(
            observation_space=self.obs_space,
            action_space=self.act_space,
            v_net_arch=[32],
            q_net_arch=[32],
            pi_net_arch=[32],
            device=self.device,
        )
        v = model.value(self.obs_batch)
        q = model.rate(self.obs_batch, self.act_batch)
        a, logp = model.act(self.obs_batch)

        self.assertEqual(v.shape, (8, 1))
        self.assertEqual(q.shape, (8, 1))
        self.assertEqual(a.shape, (8, 2))
        self.assertEqual(logp.shape, (8, 1))

    def test_actor_v_critic_shared_features(self):
        feat_extractor = FlattenExtractor(self.obs_space)
        model = ActorVCriticModel(
            observation_space=self.obs_space,
            action_space=self.act_space,
            v_net_arch=[16],
            pi_net_arch=[16],
            feature_extractor=feat_extractor,
            features_dim=feat_extractor.features_dim,
            device=self.device,
        )
        v = model.value(self.obs_batch)
        a, logp = model.act(self.obs_batch)
        self.assertEqual(v.shape, (8, 1))
        self.assertEqual(a.shape, (8, 2))
        self.assertEqual(logp.shape, (8, 1))

    def test_actor_q_critic_model_min_q_and_targets(self):
        model = ActorQCriticModel(
            observation_space=self.obs_space,
            action_space=self.act_space,
            q_net_arch=[32],
            pi_net_arch=[32],
            n_critics=2,
            deterministic_policy=False,
            device=self.device,
        )
        a, logp = model.act(self.obs_batch)
        self.assertEqual(a.shape, (8, 2))
        self.assertEqual(logp.shape, (8, 1))

        min_q = model.min_q(self.obs_batch, self.act_batch)
        self.assertEqual(min_q.shape, (8, 1))

        # soft update should not crash
        model.soft_update_targets(tau=0.005)


if __name__ == "__main__":
    unittest.main()
