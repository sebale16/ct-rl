# tests/test_env_vec_env.py
import unittest
import numpy as np
import gymnasium as gym

from environment.base import ContinuousEnv
from environment.vec_env import VecContinuousEnv


class DummyEnv(ContinuousEnv):
    def __init__(self, episode_length=3):
        super().__init__(max_steps=episode_length)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self._state = 0
        self.id = np.random.randint(0, 100)

    def _reset_physics(self, *, seed=None, options=None):
        self._state = 0
        return np.array([self._state], dtype=np.float32), {"id": self.id}

    def _step_physics(self, action, dt):
        self._state += 1
        obs = np.array([self._state], dtype=np.float32)
        reward = 1.0
        terminated = False
        truncated = self._state >= self.max_steps
        return obs, reward, terminated, truncated, {}, dt


class TestVecEnv(unittest.TestCase):
    def test_vec_env_creation_and_reset(self):
        n_envs = 4
        env_fns = [lambda: DummyEnv(episode_length=3) for _ in range(n_envs)]
        vec_env = VecContinuousEnv(env_fns)

        self.assertEqual(vec_env.num_envs, n_envs)
        obs, infos = vec_env.reset()
        self.assertEqual(obs.shape, (n_envs, 1))
        self.assertEqual(len(infos), n_envs)

    def test_vec_env_step_and_autoreset(self):
        n_envs = 2
        episode_len = 3
        env_fns = [lambda: DummyEnv(episode_length=episode_len) for _ in range(n_envs)]
        vec_env = VecContinuousEnv(env_fns)
        vec_env.reset()

        for _ in range(episode_len):
            actions = np.random.rand(n_envs, 1).astype(np.float32)
            (
                obs_t,
                t,
                actions_out,
                rewards,
                next_obs,
                next_t,
                terminated,
                truncated,
                infos,
            ) = vec_env.step_dt(actions)

            # Check shapes of all returned values
            self.assertEqual(obs_t.shape, (n_envs, 1))
            self.assertEqual(t.shape, (n_envs,))
            self.assertEqual(actions_out.shape, (n_envs, 1))
            self.assertEqual(rewards.shape, (n_envs,))
            self.assertEqual(next_obs.shape, (n_envs, 1))
            self.assertEqual(next_t.shape, (n_envs,))
            self.assertEqual(terminated.shape, (n_envs,))
            self.assertEqual(truncated.shape, (n_envs,))

        # After `episode_len` steps, all envs should be done and auto-reset
        dones = np.logical_or(terminated, truncated)
        self.assertTrue(np.all(dones))
        # Check that terminal observation is stashed and obs is from a new episode
        self.assertTrue("terminal_observation" in infos[0])
        self.assertAlmostEqual(next_obs[0, 0], 0.0)
