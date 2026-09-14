"""Physical-time discount and CT-SAC/CT-TD3 target contracts."""

import math
import unittest
from unittest.mock import patch

import numpy as np
import torch as th

from algorithms.ct_sac import CTSAC, ModelBasedTargetNumericalError
from algorithms.ct_td3 import CTTD3
from common.buffers import ReplayBatch
from environment import DMCContinuousEnv
from models.actor_q_critic import ActorQCriticModel


def _absorbing_continuation(rate: th.Tensor, remaining: th.Tensor, discount_rate: float) -> th.Tensor:
    """Reference G_F = rate * (1-exp(-discount_rate*remaining))/discount_rate.

    Mirrors ``CTSAC._absorbing_failure_target``'s frozen-rate integral so
    tests can state expectations from the documented contract rather than
    from the implementation itself.
    """
    if discount_rate == 0.0:
        return rate * remaining
    return rate * (-th.expm1(-discount_rate * remaining) / discount_rate)


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

        # At dt=T, V(s) must cancel exactly. Its magnitude can affect neither
        # the target nor the historical -19 V(s) coefficient.
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

    def test_cap_target_sums_the_frozen_rate_over_the_remaining_horizon(self):
        reference_dt = 0.001
        discount_rate = 0.1
        agent = self._agent(
            discount_rate=discount_rate,
            target_reference_dt=reference_dt,
            reward_is_rate=True,
        )
        rewards = th.tensor([[-0.75], [0.0], [0.5]], dtype=th.float64)
        remaining = th.tensor([[2.0], [3.0], [0.0]], dtype=th.float64)
        obs = th.zeros((3, agent.env.observation_space.shape[0]), dtype=th.float64)
        continuation = _absorbing_continuation(rewards, remaining, discount_rate)
        expected = reference_dt * rewards + math.exp(
            -discount_rate * reference_dt
        ) * continuation

        # The nominal (dt == T) row re-anchors algebraically, so it must
        # never read a learned value.
        with patch.object(
            agent,
            "_state_value",
            side_effect=AssertionError("cap target read a learned value"),
        ):
            actual = agent._absorbing_failure_target(
                obs,
                rewards,
                th.full_like(rewards, reference_dt),
                rewards,
                remaining,
                th.tensor(0.1),
            )
        th.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_cap_target_interval_reward_mode_sums_undiscounted(self):
        reference_dt = 0.001
        agent = self._agent(
            discount_rate=0.0,
            target_reference_dt=reference_dt,
            reward_is_rate=False,
        )
        rewards = th.tensor([[-0.25], [0.0], [0.5]], dtype=th.float64)
        rates = th.tensor([[-0.25], [0.0], [0.5]], dtype=th.float64)
        remaining = th.tensor([[4.0], [0.0], [2.0]], dtype=th.float64)
        obs = th.zeros((3, agent.env.observation_space.shape[0]), dtype=th.float64)
        # At zero discount, continuation is a literal rate * remaining-time
        # sum, and the one-time interval reward is left unscaled.
        expected = rewards + rates * remaining
        actual = agent._absorbing_failure_target(
            obs,
            rewards,
            th.full_like(rewards, reference_dt),
            rates,
            remaining,
            th.tensor(0.1),
        )
        th.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)

    def test_irregular_cap_transition_reanchors_through_the_absorbing_value(self):
        reference_dt = 0.001
        discount_rate = 0.1
        agent = self._agent(
            discount_rate=discount_rate,
            target_reference_dt=reference_dt,
            reward_is_rate=True,
        )
        obs_dim = agent.env.observation_space.shape[0]
        act_dim = agent.env.action_space.shape[0]
        dt = th.tensor([[0.002]], dtype=th.float64)
        rewards = th.tensor([[-0.4]], dtype=th.float64)
        rate = th.tensor([[-50.0]], dtype=th.float64)
        remaining = th.tensor([[4.0]], dtype=th.float64)
        batch = ReplayBatch(
            observations=th.zeros((1, obs_dim), dtype=th.float64),
            actions=th.zeros((1, act_dim)),
            next_observations=th.full((1, obs_dim), float("nan")),
            rewards=rewards,
            dones=th.ones((1, 1)),
            episode_ends=th.ones((1, 1)),
            cap_failures=th.ones((1, 1)),
            failure_reward_rates=rate,
            failure_remaining_times=remaining,
            t=th.zeros((1, 1)),
            next_t=dt,
            dt=dt,
        )
        # h != T re-anchors through the current-state value, exactly as an
        # ordinary irregular transition does. The endpoint alone differs from
        # the model-free path. It is G_F, and not a learned V(s').
        value_current = th.tensor([[9.0]], dtype=th.float64)
        with patch.object(
            agent, "_state_value", return_value=value_current
        ) as call:
            actual = agent._critic_target(batch, th.tensor(0.1))
        call.assert_called_once()
        gamma_dt = math.exp(-discount_rate * 0.002)
        continuation = _absorbing_continuation(rate, remaining, discount_rate)
        ratio = 0.002 / reference_dt
        future = value_current + (gamma_dt * continuation - value_current) / ratio
        expected = reference_dt * rewards + future
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
        cap_rate = th.tensor([[-1.0]], dtype=actual.dtype)
        cap_remaining = th.tensor([[2.0]], dtype=actual.dtype)
        continuation = _absorbing_continuation(cap_rate, cap_remaining, rate)
        expected_cap = -0.75 * reference_dt + math.exp(
            -rate * reference_dt
        ) * continuation.item()
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


class TestCTTD3CapFailureTarget(unittest.TestCase):
    """CT-TD3's cap-failure target, which mirrors ``CTSAC``'s.

    CT-TD3 works in rescaled time, so its per-interval continuation ``C_F`` is
    CT-SAC's physical ``G_F`` divided by the reference interval ``T``.  These
    tests state the contract in CT-TD3's own units and then cross-check the
    correspondence against CT-SAC directly.
    """

    def _agent(self, dt=0.02, **kwargs):
        env = DMCContinuousEnv(
            "cartpole",
            "swingup",
            time_sampling="uniform",
            dt=dt,
            episode_duration=0.1,
        )
        self.addCleanup(env.close)
        model = ActorQCriticModel(
            observation_space=env.observation_space,
            action_space=env.action_space,
            q_net_arch=[8],
            pi_net_arch=[8],
            deterministic_policy=True,
            use_actor_target=True,
            device="cpu",
        )
        return CTTD3(
            env=env,
            model=model,
            device="cpu",
            learning_starts=10,
            batch_size=4,
            buffer_size=32,
            **kwargs,
        )

    @staticmethod
    def _batch(*, dt, rewards, rate, remaining, next_q_is_nan=True, n_obs=5):
        """A one-row, all-cap batch. ``next_observations`` is deliberately NaN:
        a cap target that reaches for a learned endpoint would propagate it."""
        return ReplayBatch(
            observations=th.zeros((1, n_obs), dtype=th.float64),
            actions=th.zeros((1, 1), dtype=th.float64),
            next_observations=th.full(
                (1, n_obs), float("nan") if next_q_is_nan else 0.0,
                dtype=th.float64,
            ),
            rewards=rewards,
            dones=th.ones((1, 1), dtype=th.float64),
            episode_ends=th.ones((1, 1), dtype=th.float64),
            cap_failures=th.ones((1, 1), dtype=th.float64),
            failure_reward_rates=rate,
            failure_remaining_times=remaining,
            t=th.zeros((1, 1), dtype=th.float64),
            next_t=dt,
            dt=dt,
        )

    def test_nominal_cap_target_cancels_the_current_value_anchor(self):
        agent = self._agent(gamma=0.98)
        beta = agent.beta
        # DMCContinuousEnv overrides dt_default with control_timestep(), so
        # the nominal interval is the env's, not the constructor's dt kwarg.
        self.assertAlmostEqual(agent.dt_default, 0.01, places=12)
        dt = th.tensor([[agent.dt_default]], dtype=th.float64)  # rho == 1
        rewards = th.tensor([[-0.4]], dtype=th.float64)
        rate = th.tensor([[-50.0]], dtype=th.float64)
        remaining = th.tensor([[0.08]], dtype=th.float64)
        batch = self._batch(dt=dt, rewards=rewards, rate=rate, remaining=remaining)

        # Rescaled remaining time; C_F is the per-interval frozen return.
        remaining_scaled = remaining * agent.time_rescale
        continuation = _absorbing_continuation(rate, remaining_scaled, beta)
        expected = rewards + math.exp(-beta) * continuation

        # At rho == 1 the anchor must cancel exactly, whatever its magnitude.
        for q_cur in (
            th.tensor([[9.0]], dtype=th.float64),
            th.tensor([[1.0e12]], dtype=th.float64),
        ):
            actual = agent._critic_target(
                batch, q_cur, th.full((1, 1), float("nan"), dtype=th.float64)
            )
            th.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_irregular_cap_transition_reanchors_through_the_absorbing_value(self):
        agent = self._agent(gamma=0.98)
        beta = agent.beta
        dt = th.tensor([[0.05]], dtype=th.float64)  # rho == 2.5
        rewards = th.tensor([[-0.4]], dtype=th.float64)
        rate = th.tensor([[-50.0]], dtype=th.float64)
        remaining = th.tensor([[0.08]], dtype=th.float64)
        batch = self._batch(dt=dt, rewards=rewards, rate=rate, remaining=remaining)
        q_cur = th.tensor([[9.0]], dtype=th.float64)

        rho = float(dt) * agent.time_rescale
        continuation = _absorbing_continuation(
            rate, remaining * agent.time_rescale, beta
        )
        endpoint = math.exp(-beta * rho) * continuation
        expected = rewards + q_cur + (endpoint - q_cur) / rho

        actual = agent._critic_target(
            batch, q_cur, th.full((1, 1), float("nan"), dtype=th.float64)
        )
        th.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_cap_target_matches_ct_sac_up_to_the_reference_interval(self):
        """CT-TD3's per-interval target is CT-SAC's physical one divided by T."""
        T = 0.01  # DMC cartpole-swingup control_timestep
        discount_rate = 0.5  # physical, s^-1
        td3 = self._agent(gamma=math.exp(-discount_rate * T))
        self.assertAlmostEqual(td3.dt_default, T, places=12)
        sac_env = DMCContinuousEnv(
            "cartpole", "swingup", time_sampling="uniform", dt=T,
            episode_duration=0.1,
        )
        self.addCleanup(sac_env.close)
        sac = CTSAC(
            env=sac_env,
            model=ActorQCriticModel(
                observation_space=sac_env.observation_space,
                action_space=sac_env.action_space,
                q_net_arch=[8], pi_net_arch=[8], device="cpu",
            ),
            device="cpu", learning_starts=10, batch_size=4, buffer_size=32,
            discount_rate=discount_rate, target_reference_dt=T,
            reward_is_rate=True,
        )
        self.assertAlmostEqual(td3.beta, discount_rate * T, places=12)

        rewards = th.tensor([[-0.4]], dtype=th.float64)
        rate = th.tensor([[-50.0]], dtype=th.float64)
        remaining = th.tensor([[4.0]], dtype=th.float64)
        anchor = th.tensor([[9.0]], dtype=th.float64)

        for h in (T, 0.5 * T, 3.0 * T):
            dt = th.tensor([[h]], dtype=th.float64)
            td3_target = td3._critic_target(
                self._batch(dt=dt, rewards=rewards, rate=rate, remaining=remaining),
                anchor,
                th.full((1, 1), float("nan"), dtype=th.float64),
            )
            # CT-SAC's anchor is V(s); feed it the same number so only the
            # scale convention differs.
            with patch.object(sac, "_state_value", return_value=anchor * T):
                sac_target = sac._absorbing_failure_target(
                    th.zeros((1, sac_env.observation_space.shape[0]),
                             dtype=th.float64),
                    rewards, dt, rate, remaining,
                    th.tensor(0.1, dtype=th.float64),
                )
            th.testing.assert_close(
                td3_target, sac_target / T, rtol=1e-10, atol=1e-10
            )

    def test_mixed_batch_splits_caps_before_regular_target_evaluation(self):
        agent = self._agent(gamma=0.98)
        dt = th.tensor([[0.02], [0.02]], dtype=th.float64)
        rewards = th.tensor([[-0.4], [0.25]], dtype=th.float64)
        batch = ReplayBatch(
            observations=th.zeros((2, 5), dtype=th.float64),
            actions=th.zeros((2, 1), dtype=th.float64),
            next_observations=th.zeros((2, 5), dtype=th.float64),
            rewards=rewards,
            dones=th.tensor([[1.0], [0.0]], dtype=th.float64),
            episode_ends=th.tensor([[1.0], [0.0]], dtype=th.float64),
            cap_failures=th.tensor([[1.0], [0.0]], dtype=th.float64),
            failure_reward_rates=th.tensor([[-50.0], [0.0]], dtype=th.float64),
            failure_remaining_times=th.tensor([[0.08], [0.0]], dtype=th.float64),
            t=th.zeros((2, 1), dtype=th.float64),
            next_t=dt,
            dt=dt,
        )
        q_cur = th.tensor([[9.0], [2.0]], dtype=th.float64)
        # A non-finite next-Q on the CAP row only: it must never be read.
        q_next = th.tensor([[float("nan")], [3.0]], dtype=th.float64)

        actual = agent._critic_target(batch, q_cur, q_next)
        self.assertTrue(bool(th.all(th.isfinite(actual))))
        # The ordinary row is untouched by the split.
        expected_regular = agent._finite_difference_target(
            q_cur[1:], q_next[1:], rewards[1:], batch.dones[1:],
            dt[1:] * agent.time_rescale,
        )
        th.testing.assert_close(
            actual[1:], expected_regular, rtol=1e-12, atol=1e-12
        )

    def test_cap_rows_must_be_terminal_episode_ends(self):
        agent = self._agent(gamma=0.98)
        dt = th.tensor([[0.02]], dtype=th.float64)
        batch = self._batch(
            dt=dt,
            rewards=th.tensor([[-0.4]], dtype=th.float64),
            rate=th.tensor([[-50.0]], dtype=th.float64),
            remaining=th.tensor([[0.08]], dtype=th.float64),
        )
        batch.dones = th.zeros((1, 1), dtype=th.float64)
        with self.assertRaises(ValueError):
            agent._critic_target(
                batch, th.zeros((1, 1), dtype=th.float64),
                th.zeros((1, 1), dtype=th.float64),
            )

    def test_reward_is_rate_makes_the_cap_target_equal_ct_sac_exactly(self):
        """With reward_is_rate the T factor moves inside, so CT-TD3's target
        is CT-SAC's own, not CT-SAC's divided by the reference interval."""
        T = 0.01
        discount_rate = 0.5
        td3 = self._agent(gamma=math.exp(-discount_rate * T),
                          target_reference_dt=T, reward_is_rate=True)
        self.assertTrue(td3.reward_is_rate)
        self.assertAlmostEqual(td3.target_reference_dt, T, places=12)
        self.assertAlmostEqual(td3.beta, discount_rate * T, places=12)

        sac_env = DMCContinuousEnv("cartpole", "swingup", time_sampling="uniform",
                                   dt=T, episode_duration=0.1)
        self.addCleanup(sac_env.close)
        sac = CTSAC(
            env=sac_env,
            model=ActorQCriticModel(
                observation_space=sac_env.observation_space,
                action_space=sac_env.action_space,
                q_net_arch=[8], pi_net_arch=[8], device="cpu",
            ),
            device="cpu", learning_starts=10, batch_size=4, buffer_size=32,
            discount_rate=discount_rate, target_reference_dt=T, reward_is_rate=True,
        )
        rewards = th.tensor([[-0.4]], dtype=th.float64)
        rate = th.tensor([[-50.0]], dtype=th.float64)
        remaining = th.tensor([[4.0]], dtype=th.float64)
        anchor = th.tensor([[9.0]], dtype=th.float64)

        for h in (T, 0.5 * T, 3.0 * T):
            dt = th.tensor([[h]], dtype=th.float64)
            td3_target = td3._critic_target(
                self._batch(dt=dt, rewards=rewards, rate=rate, remaining=remaining),
                anchor, th.full((1, 1), float("nan"), dtype=th.float64),
            )
            with patch.object(sac, "_state_value", return_value=anchor):
                sac_target = sac._absorbing_failure_target(
                    th.zeros((1, sac_env.observation_space.shape[0]), dtype=th.float64),
                    rewards, dt, rate, remaining, th.tensor(0.1, dtype=th.float64),
                )
            th.testing.assert_close(td3_target, sac_target, rtol=1e-10, atol=1e-10)

    def test_reward_is_rate_requires_an_explicit_reference_interval(self):
        with self.assertRaises(ValueError):
            self._agent(gamma=0.98, reward_is_rate=True)
