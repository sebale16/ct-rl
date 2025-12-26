# tests/test_common_torch_layers.py

import unittest

import torch as th
from gymnasium import spaces

from common.torch_layers import (
    get_device,
    create_mlp,
    MlpExtractor,
    FlattenExtractor,
)


class TestTorchLayers(unittest.TestCase):
    def test_create_mlp_forward_shape(self):
        mlp = create_mlp(input_dim=4, output_dim=2, hidden_dims=[8, 8])
        x = th.randn(5, 4)
        y = mlp(x)
        self.assertEqual(y.shape, (5, 2))

    def test_mlp_extractor_shapes(self):
        device = get_device("auto")
        extractor = MlpExtractor(feature_dim=4, net_arch=[8, 8], device=device)
        x = th.randn(3, 4).to(device)
        pi_latent, vf_latent = extractor(x)
        self.assertEqual(pi_latent.shape[1], extractor.latent_dim_pi)
        self.assertEqual(vf_latent.shape[1], extractor.latent_dim_vf)

    def test_flatten_extractor(self):
        obs_space = spaces.Box(low=0.0, high=1.0, shape=(3, 2), dtype=float)
        extractor = FlattenExtractor(obs_space)
        x = th.randn(4, 3, 2)
        y = extractor(x)
        self.assertEqual(y.shape, (4, 6))


if __name__ == "__main__":
    unittest.main()
