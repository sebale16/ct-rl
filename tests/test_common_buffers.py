# tests/test_common_buffers.py

import unittest

import numpy as np
import torch as th
from gymnasium import spaces

from common.buffers import ReplayBuffer, RolloutBuffer


class TestBuffers(unittest.TestCase):
    def test_replay_buffer_add_and_sample(self):
        obs_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=float)
        act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
        buf = ReplayBuffer(
            buffer_size=10,
            observation_space=obs_space,
            action_space=act_space,
            n_envs=1,
        )

        for k in range(5):
            obs = np.ones((1, 3), dtype=np.float32) * k
            next_obs = np.ones((1, 3), dtype=np.float32) * (k + 1)
            action = np.zeros((1, 2), dtype=np.float32)
            reward = np.array([1.0], dtype=np.float32)
            done = np.array([0.0], dtype=np.float32)
            t = np.array([0.1 * k], dtype=np.float32)
            next_t = np.array([0.1 * (k + 1)], dtype=np.float32)
            buf.add(obs, action, reward, done, next_obs, t, next_t)

        batch = buf.sample(batch_size=3)
        self.assertIsInstance(batch.observations, th.Tensor)
        self.assertEqual(batch.observations.shape[1], 3)
        self.assertEqual(batch.actions.shape[1], 2)
        # dt should equal next_t - t
        self.assertTrue(th.allclose(batch.dt, batch.next_t - batch.t, atol=1e-6))

    def test_rollout_buffer_add_and_compute(self):
        obs_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
        act_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=float)
        buf = RolloutBuffer(
            buffer_size=4,
            observation_space=obs_space,
            action_space=act_space,
            gamma=1.0,
            n_envs=1,
        )

        for k in range(4):
            obs = np.ones((1, 2), dtype=np.float32) * k
            next_obs = np.ones((1, 2), dtype=np.float32) * (k + 1)
            action = np.zeros((1, 1), dtype=np.float32)
            reward = np.array([1.0], dtype=np.float32)
            done = np.array([0.0], dtype=np.float32)
            episode_start = np.array([1.0 if k == 0 else 0.0], dtype=np.float32)
            value = th.zeros(1)
            log_prob = th.zeros(1)
            t = np.array([0.1 * k], dtype=np.float32)
            next_t = np.array([0.1 * (k + 1)], dtype=np.float32)
            buf.add(
                obs=obs,
                next_obs=next_obs,
                action=action,
                reward=reward,
                done=done,
                episode_start=episode_start,
                value=value,
                log_prob=log_prob,
                t=t,
                next_t=next_t,
            )

    def test_rollout_buffer_get_batches(self):
        obs_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
        act_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=float)
        buf = RolloutBuffer(
            buffer_size=3, observation_space=obs_space, action_space=act_space, n_envs=1
        )

        for k in range(3):
            obs = np.zeros((1, 2), dtype=np.float32)
            next_obs = np.zeros((1, 2), dtype=np.float32)
            action = np.zeros((1, 1), dtype=np.float32)
            reward = np.array([0.0], dtype=np.float32)
            done = np.array([0.0], dtype=np.float32)
            episode_start = np.array([1.0 if k == 0 else 0.0], dtype=np.float32)
            value = th.zeros(1)
            log_prob = th.zeros(1)
            t = np.array([0.1 * k], dtype=np.float32)
            next_t = np.array([0.1 * (k + 1)], dtype=np.float32)
            buf.add(
                obs=obs,
                next_obs=next_obs,
                action=action,
                reward=reward,
                done=done,
                episode_start=episode_start,
                value=value,
                log_prob=log_prob,
                t=t,
                next_t=next_t,
            )

        batches = list(buf.get(batch_size=2))
        self.assertGreaterEqual(len(batches), 1)
        batch0 = batches[0]
        self.assertEqual(batch0.observations.shape[1], 2)
        self.assertEqual(batch0.actions.shape[1], 1)
        self.assertEqual(batch0.t.shape[-1], 1)  # flattened (N, 1) after to_torch


if __name__ == "__main__":
    unittest.main()
