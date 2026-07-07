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


class TestReplaySequenceSampling(unittest.TestCase):
    """sample_sequences returns contiguous action-conditioned windows with a
    cumulative validity mask that breaks at episode ends and the ring seam."""

    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def _fill(self, buf, n, dones=None, offset=0.0):
        dones = dones if dones is not None else [0] * n
        for i in range(n):
            buf.add(
                obs=np.full((buf.n_envs, 3), offset + i, np.float32),
                action=np.full((buf.n_envs, 2), offset + i, np.float32),
                reward=np.zeros((buf.n_envs,), np.float32),
                done=np.full((buf.n_envs,), dones[i], np.float32),
                next_obs=np.full((buf.n_envs, 3), offset + i + 1, np.float32),
                t=np.full((buf.n_envs,), 0.01 * i, np.float32),
                next_t=np.full((buf.n_envs,), 0.01 * (i + 1), np.float32),
            )

    def test_windows_chain_and_shapes(self):
        buf = ReplayBuffer(10, self.obs_space, self.act_space, device="cpu", n_envs=1)
        self._fill(buf, 6)
        seq = buf._get_sequence_samples(np.array([0, 2]), np.array([0, 0]), 3)
        self.assertEqual(tuple(seq.observations.shape), (2, 3))
        self.assertEqual(tuple(seq.actions.shape), (2, 3, 2))
        self.assertEqual(tuple(seq.next_observations.shape), (2, 3, 3))
        self.assertEqual(tuple(seq.dt.shape), (2, 3, 1))
        self.assertEqual(tuple(seq.mask.shape), (2, 3, 1))
        # start=0 -> targets obs 1,2,3; start=2 -> targets 3,4,5; all valid
        self.assertEqual(seq.next_observations[0, :, 0].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(seq.next_observations[1, :, 0].tolist(), [3.0, 4.0, 5.0])
        self.assertEqual(seq.actions[1, :, 0].tolist(), [2.0, 3.0, 4.0])
        self.assertTrue((seq.mask == 1).all())

    def test_mask_breaks_after_done(self):
        # done at index 2: that transition is still a valid target, later steps not
        buf = ReplayBuffer(10, self.obs_space, self.act_space, device="cpu", n_envs=1)
        self._fill(buf, 6, dones=[0, 0, 1, 0, 0, 0])
        seq = buf._get_sequence_samples(np.array([1]), np.array([0]), 4)
        self.assertEqual(seq.mask[0, :, 0].tolist(), [1.0, 1.0, 0.0, 0.0])

    def test_mask_breaks_at_ring_seam(self):
        # size-5 buffer with 7 adds: slots hold obs [5, 6, 2, 3, 4], pos=2 (seam)
        buf = ReplayBuffer(5, self.obs_space, self.act_space, device="cpu", n_envs=1)
        self._fill(buf, 7)
        # window 0,1,2 hits the seam slot (oldest sample) at k=2
        seq = buf._get_sequence_samples(np.array([0]), np.array([0]), 3)
        self.assertEqual(seq.mask[0, :, 0].tolist(), [1.0, 1.0, 0.0])
        # window 3,4,0 wraps the array end but stays chronological: fully valid
        seq = buf._get_sequence_samples(np.array([3]), np.array([0]), 3)
        self.assertEqual(seq.mask[0, :, 0].tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(seq.next_observations[0, :, 0].tolist(), [4.0, 5.0, 6.0])

    def test_windows_stay_within_env(self):
        buf = ReplayBuffer(10, self.obs_space, self.act_space, device="cpu", n_envs=2)
        # both envs written in lockstep with identical values; overwrite env 1
        # with offset values to make cross-env leakage visible
        self._fill(buf, 6)
        buf.observations[:6, 1, :] += 100.0
        buf.next_observations[:6, 1, :] += 100.0
        seq = buf._get_sequence_samples(np.array([1]), np.array([1]), 3)
        self.assertEqual(seq.next_observations[0, :, 0].tolist(), [102.0, 103.0, 104.0])


if __name__ == "__main__":
    unittest.main()
