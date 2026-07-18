import unittest
import tempfile

import numpy as np
import torch as th

from algorithms.ct_sac import CTSAC
from common.checkpoint import load_checkpoint, save_checkpoint
from evaluations.evaluation_helpers import evaluate_policy_per_episode
from models.actor_q_critic import ActorQCriticModel
from tests.test_env_base import DummyLinearEnv


def _agent(seed: int, *, width: int = 16, with_v: bool = True) -> CTSAC:
    env = DummyLinearEnv(
        time_sampling="irregular",
        dt=0.02,
        physics_dt=0.01,
        min_dt=0.01,
        max_dt=0.03,
        episode_duration=0.2,
    )
    model_kwargs = {
        "q_net_arch": [width],
        "pi_net_arch": [width],
    }
    if with_v:
        model_kwargs["v_net_arch"] = [width]
    return CTSAC(
        env=env,
        model=ActorQCriticModel,
        model_kwargs=model_kwargs,
        seed=seed,
        learning_starts=10,
        buffer_size=32,
        batch_size=4,
    )


def _model_tensors(agent: CTSAC) -> list[th.Tensor]:
    modules = [agent.model.actor, *agent.model.q_nets]
    if agent.model.has_v_head:
        modules.append(agent.model.v_net)
    return [
        value.detach().clone()
        for module in modules
        for value in module.state_dict().values()
    ]


class TestFreshRunReproducibility(unittest.TestCase):
    def test_same_seed_reproduces_initial_model_and_warmup_actions(self):
        first = _agent(7)
        first_state = _model_tensors(first)
        first_actions = np.stack(
            [first._sample_action(np.zeros(1, dtype=np.float32)) for _ in range(8)]
        )

        second = _agent(7)
        second_state = _model_tensors(second)
        second_actions = np.stack(
            [second._sample_action(np.zeros(1, dtype=np.float32)) for _ in range(8)]
        )

        self.assertEqual(len(first_state), len(second_state))
        for first_value, second_value in zip(first_state, second_state):
            th.testing.assert_close(first_value, second_value)
        np.testing.assert_array_equal(first_actions, second_actions)

    def test_different_seed_changes_initial_model_and_warmup_actions(self):
        first = _agent(7)
        second = _agent(8)

        self.assertTrue(
            any(
                not th.equal(a, b)
                for a, b in zip(
                    _model_tensors(first),
                    _model_tensors(second),
                )
            )
        )
        first_action = first._sample_action(np.zeros(1, dtype=np.float32))
        second_action = second._sample_action(np.zeros(1, dtype=np.float32))
        self.assertFalse(np.array_equal(first_action, second_action))

    def test_post_initialization_rng_stream_is_architecture_independent(self):
        _agent(19, width=8)
        numpy_small = np.random.random(5)
        torch_small = th.rand(5)

        _agent(19, width=64)
        numpy_large = np.random.random(5)
        torch_large = th.rand(5)

        np.testing.assert_array_equal(numpy_small, numpy_large)
        th.testing.assert_close(torch_small, torch_large)

    def test_optional_value_head_does_not_change_shared_initialization(self):
        plain = _agent(23, with_v=False)
        with_value = _agent(23, with_v=True)

        for plain_value, value_head_value in zip(
            plain.model.actor.state_dict().values(),
            with_value.model.actor.state_dict().values(),
        ):
            th.testing.assert_close(plain_value, value_head_value)
        for plain_q, value_head_q in zip(
            plain.model.q_nets, with_value.model.q_nets
        ):
            for plain_value, value_head_value in zip(
                plain_q.state_dict().values(), value_head_q.state_dict().values()
            ):
                th.testing.assert_close(plain_value, value_head_value)

    def test_checkpoint_restores_dedicated_dynamics_sampling_stream(self):
        agent = _agent(31)
        agent._dynamics_sample_rng = np.random.default_rng(1234)
        agent._dynamics_sample_rng.random(3)
        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint(agent, tmp)
            expected = agent._dynamics_sample_rng.random(5)
            agent._dynamics_sample_rng.random(50)
            load_checkpoint(agent, tmp)
            actual = agent._dynamics_sample_rng.random(5)
        np.testing.assert_array_equal(actual, expected)

    def test_fixed_evaluation_seed_replays_the_same_episode_stream(self):
        agent = _agent(41)
        first = evaluate_policy_per_episode(
            agent.model, agent.env, n_eval_episodes=3, reset_seed=20000
        )
        second = evaluate_policy_per_episode(
            agent.model, agent.env, n_eval_episodes=3, reset_seed=20000
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
