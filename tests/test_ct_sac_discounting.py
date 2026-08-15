"""Physical-time discount and CT-SAC finite-difference target contracts."""

import math
import unittest
from unittest.mock import patch

import numpy as np
import torch as th

from algorithms.ct_sac import CTSAC, ModelBasedTargetNumericalError
from common.buffers import ReplayBatch
from environment import DMCContinuousEnv
from models.actor_q_critic import ActorQCriticModel


class TestCTSACPhysicalDiscounting(unittest.TestCase):
    def _agent(self, **kwargs):
        env = DMCContinuousEnv(
            "cartpole",
            "swingup",
            time_sampling="uniform",
            dt=0.02,
            episode_duration=0.1,
        )
        self.addCleanup(env.close)
        model = ActorQCriticModel(
            observation_space=env.observation_space,
            action_space=env.action_space,
            q_net_arch=[8],
            pi_net_arch=[8],
            device="cpu",
        )
        return CTSAC(
            env=env,
            model=model,
            device="cpu",
            learning_starts=10,
            batch_size=4,
            buffer_size=32,
            **kwargs,
        )

    def test_nominal_rate_target_reduces_exactly_to_soft_sac(self):
        reference_dt = 0.0005
        rate = 0.1
        agent = self._agent(
            discount_rate=rate,
            target_reference_dt=reference_dt,
            reward_is_rate=True,
        )
        self.assertEqual(agent.target_reference_dt, reference_dt)
        self.assertEqual(agent.dt_default, reference_dt)
        self.assertAlmostEqual(agent.discount_horizon_seconds, 10.0)

        reward = th.tensor([[-1200.0]], dtype=th.float64)
        done = th.zeros_like(reward)
        value_next = th.tensor([[5.0]], dtype=th.float64)
        discount = math.exp(-rate * reference_dt)
        expected = reference_dt * reward + discount * value_next

        # At dt=T, V(s) must cancel exactly; its magnitude cannot affect the
        # target or recreate the historical -19 V(s) coefficient.
        for value_current in (
            th.tensor([[7.0]], dtype=th.float64),
            th.tensor([[1.0e12]], dtype=th.float64),
        ):
            actual = agent._finite_difference_target_from_values(
                value_current,
                value_next,
                reward,
                done,
                reference_dt,
            )
            th.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

        self.assertAlmostEqual(agent.gamma, discount, places=15)
        self.assertAlmostEqual(agent.beta, rate * reference_dt, places=15)

    def test_nominal_cap_target_uses_only_analytical_failure_continuation(self):
        reference_dt = 0.001
        rate = 0.1
        agent = self._agent(
            discount_rate=rate,
            target_reference_dt=reference_dt,
            reward_is_rate=True,
        )
        obs = th.zeros((1, agent.env.observation_space.shape[0]))
        reward = th.tensor([[-0.75]], dtype=th.float64)
        remaining = th.tensor([[19.999]], dtype=th.float64)
        failure_rate = th.tensor([[-1.0]], dtype=th.float64)
        dt = th.tensor([[reference_dt]], dtype=th.float64)
        continuation = -(-math.expm1(-rate * float(remaining))) / rate
        expected = reference_dt * reward + math.exp(
            -rate * reference_dt
        ) * continuation

        # At h=T neither V(s) nor V(s') belongs to the cap target.
        with patch.object(
            agent,
            "_state_value",
            side_effect=AssertionError("nominal cap target read a learned value"),
        ):
            actual = agent._absorbing_failure_target(
                obs,
                reward,
                dt,
                failure_rate,
                remaining,
                th.tensor(0.1),
            )
        th.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_cap_target_zero_discount_and_no_remaining_horizon(self):
        reference_dt = 0.001
        agent = self._agent(
            discount_rate=0.0,
            target_reference_dt=reference_dt,
            reward_is_rate=True,
        )
        obs = th.zeros((3, agent.env.observation_space.shape[0]), dtype=th.float64)
        rewards = th.tensor([[-0.25], [0.0], [0.5]], dtype=th.float64)
        endpoint_rates = th.tensor([[-1.0], [0.0], [2.0]], dtype=th.float64)
        remaining = th.tensor([[3.0], [4.0], [0.0]], dtype=th.float64)
        actual = agent._absorbing_failure_target(
            obs,
            rewards,
            th.full((3, 1), reference_dt, dtype=th.float64),
            endpoint_rates,
            remaining,
            th.tensor(0.1),
        )
        expected = reference_dt * rewards + endpoint_rates * remaining
        th.testing.assert_close(actual, expected, rtol=0.0, atol=1e-15)

    def test_irregular_cap_target_reanchors_to_known_endpoint(self):
        reference_dt = 0.001
        rate = 0.1
        agent = self._agent(
            discount_rate=rate,
            target_reference_dt=reference_dt,
            reward_is_rate=True,
        )
        obs = th.zeros((1, agent.env.observation_space.shape[0]), dtype=th.float64)
        reward = th.tensor([[-0.4]], dtype=th.float64)
        dt = th.tensor([[0.002]], dtype=th.float64)
        remaining = th.tensor([[4.0]], dtype=th.float64)
        value_current = th.tensor([[7.0]], dtype=th.float64)
        continuation = -(-math.expm1(-rate * 4.0)) / rate
        expected = -0.4 * reference_dt + value_current + (
            math.exp(-rate * 0.002) * continuation - value_current
        ) / 2.0
        with patch.object(agent, "_state_value", return_value=value_current) as state_value:
            actual = agent._absorbing_failure_target(
                obs,
                reward,
                dt,
                th.tensor([[-1.0]], dtype=th.float64),
                remaining,
                th.tensor(0.1),
            )
        state_value.assert_called_once()
        th.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_mixed_batch_splits_caps_before_regular_target_evaluation(self):
        reference_dt = 0.001
        rate = 0.1
        agent = self._agent(
            discount_rate=rate,
            target_reference_dt=reference_dt,
            reward_is_rate=True,
        )
        obs_dim = agent.env.observation_space.shape[0]
        act_dim = agent.env.action_space.shape[0]
        zeros_obs = th.zeros((2, obs_dim))
        batch = ReplayBatch(
            observations=zeros_obs,
            actions=th.zeros((2, act_dim)),
            next_observations=th.stack(
                [th.full((obs_dim,), float("nan")), th.ones(obs_dim)]
            ),
            rewards=th.tensor([[-0.75], [-0.25]]),
            dones=th.tensor([[1.0], [0.0]]),
            episode_ends=th.tensor([[1.0], [0.0]]),
            cap_failures=th.tensor([[1.0], [0.0]]),
            failure_reward_rates=th.tensor([[-1.0], [0.0]]),
            failure_remaining_times=th.tensor([[2.0], [0.0]]),
            t=th.zeros((2, 1)),
            next_t=th.full((2, 1), reference_dt),
            dt=th.full((2, 1), reference_dt),
        )
        regular_value = th.tensor([[123.0]])

        def regular_target(obs, next_obs, rewards, dones, dt, alpha):
            self.assertEqual(obs.shape[0], 1)
            self.assertTrue(th.isfinite(next_obs).all())
            return regular_value

        with (
            patch.object(agent, "_finite_difference_target", side_effect=regular_target),
            patch.object(
                agent,
                "_state_value",
                side_effect=AssertionError("nominal cap row read a learned value"),
            ),
        ):
            actual = agent._critic_target(batch, th.tensor(0.1))
        continuation = -(-math.expm1(-rate * 2.0)) / rate
        expected_cap = -0.75 * reference_dt + math.exp(
            -rate * reference_dt
        ) * continuation
        th.testing.assert_close(
            actual,
            th.tensor([[expected_cap], [123.0]]),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_cap_split_precedes_model_based_and_guard_dispatch(self):
        reference_dt = 0.001
        rate = 0.1
        for guarded in (False, True):
            with self.subTest(guarded=guarded):
                agent = self._agent(
                    discount_rate=rate,
                    target_reference_dt=reference_dt,
                    reward_is_rate=True,
                )
                agent.use_model_based_q = True
                agent.dynamics_model = object()
                agent.target_guard_kappa = 1.0 if guarded else 0.0
                agent.target_guard_cap = 0.0
                obs_dim = agent.env.observation_space.shape[0]
                act_dim = agent.env.action_space.shape[0]
                batch = ReplayBatch(
                    observations=th.zeros((2, obs_dim)),
                    actions=th.zeros((2, act_dim)),
                    next_observations=th.stack(
                        [th.full((obs_dim,), float("nan")), th.ones(obs_dim)]
                    ),
                    rewards=th.tensor([[-0.5], [-0.25]]),
                    dones=th.tensor([[1.0], [0.0]]),
                    episode_ends=th.tensor([[1.0], [0.0]]),
                    cap_failures=th.tensor([[1.0], [0.0]]),
                    failure_reward_rates=th.tensor([[-1.0], [0.0]]),
                    failure_remaining_times=th.tensor([[1.0], [0.0]]),
                    t=th.zeros((2, 1)),
                    next_t=th.full((2, 1), reference_dt),
                    dt=th.full((2, 1), reference_dt),
                )

                def regular_target(obs, actions, next_obs, *args):
                    self.assertEqual(obs.shape[0], 1)
                    self.assertTrue(th.isfinite(next_obs).all())
                    return th.tensor([[77.0]])

                selected = (
                    "_guarded_model_based_target"
                    if guarded
                    else "_model_based_target"
                )
                with (
                    patch.object(agent, selected, side_effect=regular_target) as call,
                    patch.object(
                        agent,
                        "_state_value",
                        side_effect=AssertionError(
                            "nominal cap row read a learned value"
                        ),
                    ),
                ):
                    actual = agent._critic_target(batch, th.tensor(0.1))
                call.assert_called_once()
                self.assertEqual(actual[1].item(), 77.0)
                self.assertTrue(th.isfinite(actual[0]).all())

    def test_irregular_durations_use_physical_seconds_and_reference_scale(self):
        reference_dt = 0.0005
        rate = 0.1
        agent = self._agent(
            discount_rate=rate,
            target_reference_dt=reference_dt,
            reward_is_rate=True,
        )
        value_current = th.full((3, 1), 7.0, dtype=th.float64)
        value_next = th.full((3, 1), 5.0, dtype=th.float64)
        rewards = th.full((3, 1), -2.0, dtype=th.float64)
        dones = th.tensor([[0.0], [0.0], [1.0]], dtype=th.float64)
        dt = th.tensor(
            [reference_dt / 2.0, reference_dt, 2.0 * reference_dt],
            dtype=th.float64,
        )

        actual = agent._finite_difference_target_from_values(
            value_current, value_next, rewards, dones, dt
        )
        ratio = dt.reshape(-1, 1) / reference_dt
        discount = th.exp(-rate * dt.reshape(-1, 1))
        future = value_current + (
            discount * value_next - value_current
        ) / ratio
        expected = reference_dt * rewards + (1.0 - dones) * future
        th.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        # Terminal transitions retain the interval reward and do not bootstrap.
        th.testing.assert_close(
            actual[-1], reference_dt * rewards[-1], rtol=0.0, atol=0.0
        )

    def test_interval_reward_mode_does_not_scale_reward(self):
        reference_dt = 0.0005
        rate = 0.1
        agent = self._agent(
            discount_rate=rate,
            target_reference_dt=reference_dt,
            reward_is_rate=False,
        )
        value_current = th.tensor([[7.0]], dtype=th.float64)
        value_next = th.tensor([[5.0]], dtype=th.float64)
        reward = th.tensor([[-2.0]], dtype=th.float64)
        actual = agent._finite_difference_target_from_values(
            value_current,
            value_next,
            reward,
            th.zeros_like(reward),
            reference_dt,
        )
        expected = reward + math.exp(-rate * reference_dt) * value_next
        th.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_small_duration_discount_correction_is_stable(self):
        reference_dt = 0.0005
        rate = 0.1
        agent = self._agent(
            discount_rate=rate,
            target_reference_dt=reference_dt,
        )
        value = th.tensor([[1.0e6]], dtype=th.float64)
        reward = th.zeros((1, 1), dtype=th.float64)
        done = th.zeros_like(reward)
        dt = th.tensor([[1.0e-12]], dtype=th.float64)
        actual = agent._finite_difference_target_from_values(
            value, value, reward, done, dt
        )
        expected_fraction = (
            reference_dt / dt * th.expm1(-rate * dt) * value
        )
        expected = value + expected_fraction
        th.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-8)

    def test_small_duration_discount_correction_is_stable_in_float32(self):
        reference_dt = 0.0005
        rate = 0.1
        agent = self._agent(
            discount_rate=rate,
            target_reference_dt=reference_dt,
        )
        value = th.tensor([[1.0e4]], dtype=th.float32)
        reward = th.zeros((1, 1), dtype=th.float32)
        done = th.zeros_like(reward)
        dt = th.tensor([[1.0e-8]], dtype=th.float32)
        actual = agent._finite_difference_target_from_values(
            value, value, reward, done, dt
        )
        expected = value + (reference_dt / dt) * th.expm1(-rate * dt) * value
        th.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-4)

    def test_legacy_gamma_maps_to_a_physical_rate(self):
        gamma = 0.995
        reference_dt = 0.01
        agent = self._agent(gamma=gamma, target_reference_dt=reference_dt)
        expected_rate = -math.log(gamma) / reference_dt
        self.assertAlmostEqual(agent.discount_rate, expected_rate, places=15)
        self.assertAlmostEqual(
            math.exp(-agent.discount_rate * 0.0005),
            gamma ** (0.0005 / reference_dt),
            places=15,
        )
        value_current = th.tensor([[7.0]], dtype=th.float64)
        value_next = th.tensor([[5.0]], dtype=th.float64)
        reward = th.tensor([[-2.0]], dtype=th.float64)
        dt = th.tensor([[0.0005]], dtype=th.float64)
        actual = agent._finite_difference_target_from_values(
            value_current,
            value_next,
            reward,
            th.zeros_like(reward),
            dt,
        )
        ratio = dt / reference_dt
        expected = reward + value_current + (
            gamma ** ratio * value_next - value_current
        ) / ratio
        th.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_replay_duration_uses_nominal_target_branch_at_episode_end(self):
        reference_dt = 0.0005
        rate = 0.1
        agent = self._agent(
            discount_rate=rate,
            target_reference_dt=reference_dt,
            reward_is_rate=True,
        )
        obs_shape = agent.env.observation_space.shape
        action_shape = agent.env.action_space.shape
        agent.replay_buffer.add(
            obs=np.zeros((1, *obs_shape), dtype=np.float32),
            action=np.zeros((1, *action_shape), dtype=np.float32),
            reward=np.zeros(1, dtype=np.float32),
            done=np.zeros(1, dtype=np.float32),
            next_obs=np.zeros((1, *obs_shape), dtype=np.float32),
            t=np.array([19.9995], dtype=np.float64),
            next_t=np.array([20.0], dtype=np.float64),
        )
        stored_dt = th.as_tensor(agent.replay_buffer.dt[0, 0])
        self.assertEqual(float(stored_dt), float(np.float32(reference_dt)))

        value_next = th.tensor([[5.0]])
        reward = th.tensor([[-1200.0]])
        expected = reference_dt * reward + math.exp(
            -rate * reference_dt
        ) * value_next
        for value_current in (th.tensor([[7.0]]), th.tensor([[1.0e12]])):
            actual = agent._finite_difference_target_from_values(
                value_current,
                value_next,
                reward,
                th.zeros_like(reward),
                stored_dt,
            )
            th.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_rejects_ambiguous_or_invalid_time_parameters(self):
        with self.assertRaisesRegex(ValueError, "either gamma.*discount_rate"):
            self._agent(
                gamma=0.995,
                discount_rate=0.1,
                target_reference_dt=0.0005,
            )
        for bad_rate in (-1.0, float("nan"), float("inf")):
            with self.subTest(discount_rate=bad_rate):
                with self.assertRaisesRegex(ValueError, "discount_rate"):
                    self._agent(discount_rate=bad_rate)
        for bad_reference in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(target_reference_dt=bad_reference):
                with self.assertRaisesRegex(ValueError, "target_reference_dt"):
                    self._agent(target_reference_dt=bad_reference)
        with self.assertRaisesRegex(ValueError, "target_reference_dt.*explicit"):
            self._agent(discount_rate=0.1)
        with self.assertRaisesRegex(ValueError, "target_reference_dt.*explicit"):
            self._agent(reward_is_rate=True)
        self.assertTrue(
            self._agent(
                reward_is_rate="True", target_reference_dt=0.0005
            ).reward_is_rate
        )
        self.assertFalse(self._agent(reward_is_rate="False").reward_is_rate)
        with self.assertRaisesRegex(ValueError, "reward_is_rate"):
            self._agent(reward_is_rate="not-a-boolean")

    def test_r3_reward_and_critic_discount_rates_must_match(self):
        env = DMCContinuousEnv(
            "acrobot",
            "swingup-xk",
            time_sampling="uniform",
            dt=0.001,
            physics_dt=0.001,
            episode_duration=0.01,
            raw_state_obs=True,
            task_kwargs={
                "reward_kind": "r3",
                "eta": 0.1,
                "discount_rate": 0.1,
            },
        )
        self.addCleanup(env.close)
        model = ActorQCriticModel(
            observation_space=env.observation_space,
            action_space=env.action_space,
            q_net_arch=[8],
            pi_net_arch=[8],
            device="cpu",
        )
        with self.assertRaisesRegex(ValueError, "task discount_rate.*match"):
            CTSAC(
                env=env,
                model=model,
                device="cpu",
                discount_rate=0.2,
                target_reference_dt=0.001,
                reward_is_rate=True,
                learning_starts=10,
                batch_size=4,
                buffer_size=32,
            )

    def test_rejects_nonpositive_or_nonfinite_transition_durations(self):
        agent = self._agent(
            discount_rate=0.1,
            target_reference_dt=0.0005,
        )
        value = th.ones((1, 1))
        reward = th.zeros((1, 1))
        done = th.zeros((1, 1))
        for bad_dt in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(dt=bad_dt):
                with self.assertRaisesRegex(ValueError, "dt"):
                    agent._finite_difference_target_from_values(
                        value, value, reward, done, bad_dt
                    )

    def test_rejects_nonfinite_model_free_target_components(self):
        agent = self._agent(
            discount_rate=0.1,
            target_reference_dt=0.0005,
            reward_is_rate=True,
        )
        value = th.ones((1, 1))
        done = th.zeros((1, 1))
        with self.assertRaisesRegex(
            ModelBasedTargetNumericalError, "component=reward_term"
        ):
            agent._finite_difference_target_from_values(
                value,
                value,
                th.full((1, 1), float("inf")),
                done,
                0.0005,
            )


if __name__ == "__main__":
    unittest.main()
