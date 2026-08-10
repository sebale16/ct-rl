import unittest

import numpy as np

from algorithms.ct_sac import CTSAC
from algorithms.off_policy import _realized_simulated_seconds
from environment.vec_env import VecContinuousEnv
from models.actor_q_critic import ActorQCriticModel
from tests.test_env_base import DummyLinearEnv


class TestSimulatedInteractionTime(unittest.TestCase):
    def test_terminal_auto_reset_uses_stashed_episode_endpoint(self):
        seconds = _realized_simulated_seconds(
            t=np.array([0.02, 1.40], dtype=np.float32),
            next_t=np.array([0.0, 1.43], dtype=np.float32),
            done=np.array([True, False]),
            infos=[{"terminal_next_t": 0.04}, {}],
        )
        self.assertAlmostEqual(seconds, 0.05, places=6)

    def test_ct_sac_sums_physical_duration_across_vector_slots(self):
        # The second slot terminates on this rollout. VecContinuousEnv returns
        # next_t=0 after auto-reset, so its 0.03 seconds are only recoverable
        # from info["terminal_next_t"].
        env = VecContinuousEnv(
            [
                lambda: DummyLinearEnv(
                    time_sampling="uniform",
                    dt=0.01,
                    episode_duration=0.02,
                ),
                lambda: DummyLinearEnv(
                    time_sampling="uniform",
                    dt=0.03,
                    episode_duration=0.03,
                ),
            ]
        )
        model = ActorQCriticModel(
            observation_space=env.observation_space,
            action_space=env.action_space,
            q_net_arch=[8],
            pi_net_arch=[8],
        )
        agent = CTSAC(
            env=env,
            model=model,
            seed=7,
            learning_starts=100,
            buffer_size=8,
            batch_size=2,
        )

        agent.learn(total_timesteps=2)

        self.assertEqual(agent.num_timesteps, 2)
        self.assertAlmostEqual(agent.num_simulated_seconds, 0.04, places=6)
        np.testing.assert_allclose(
            agent.replay_buffer.dt[0, :],
            np.array([0.01, 0.03]),
            rtol=0.0,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
