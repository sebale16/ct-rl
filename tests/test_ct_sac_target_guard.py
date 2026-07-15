import unittest

import torch as th

from environment import DMCContinuousEnv
from algorithms.ct_sac import CTSAC, ModelBasedTargetNumericalError
from models.actor_q_critic import ActorQCriticModel


class _FrozenDynamics(th.nn.Module):
    """Non-trainable drift: the model-based target is ready immediately."""

    def drift(self, obs, action):
        return th.zeros_like(th.as_tensor(obs, dtype=th.float32))

    def diffusion(self, obs):
        return None


def _make_agent(*, generator_substeps=2, **guard):
    env = DMCContinuousEnv(
        "cartpole", "swingup", time_sampling="uniform", dt=0.02,
        episode_duration=0.1,
    )
    model = ActorQCriticModel(
        observation_space=env.observation_space,
        action_space=env.action_space,
        q_net_arch=[8],
        pi_net_arch=[8],
        v_net_arch=[8],
        device="cpu",
    )
    agent = CTSAC(
        env=env,
        model=model,
        device="cpu",
        learning_starts=10,
        batch_size=4,
        buffer_size=32,
        use_model_based_q=True,
        dynamics_model=_FrozenDynamics(),
        generator_substeps=generator_substeps,
        value_warmup=0,
        dynamics_warmup=0,
        **guard,
    )
    return env, agent


def _batch(agent, n=8):
    obs_dim = int(agent.env.observation_space.shape[0])
    act_dim = int(agent.env.action_space.shape[0])
    g = th.Generator().manual_seed(0)
    return (
        0.1 * th.randn(n, obs_dim, generator=g),
        th.zeros(n, act_dim),
        0.1 * th.randn(n, obs_dim, generator=g),
        th.ones(n, 1),
        th.zeros(n, 1),
        th.full((n, 1), 0.02),
        th.tensor(0.2),
    )


class TestTargetGuard(unittest.TestCase):
    def test_disabled_by_default_and_check_flag_is_inert_on_healthy_batch(self):
        _, agent = _make_agent()
        self.assertFalse(agent._target_guard_enabled)
        obs, act, nobs, rew, dn, dt, alpha = _batch(agent)
        t_checked = agent._model_based_target(obs, act, nobs, rew, dn, dt, alpha)
        t_raw = agent._model_based_target(
            obs, act, nobs, rew, dn, dt, alpha, check=False
        )
        self.assertTrue(th.equal(t_checked, t_raw))

    def test_guarded_first_order_generator_enables_observation_autograd(self):
        _, agent = _make_agent(generator_substeps=0, target_guard_kappa=4.0)
        target = agent._guarded_model_based_target(*_batch(agent))
        self.assertEqual(tuple(target.shape), (8, 1))
        self.assertTrue(th.isfinite(target).all())
        self.assertFalse(target.requires_grad)

    def test_empty_csv_cells_coerce_to_disabled(self):
        _, agent = _make_agent(target_guard_kappa="", target_guard_cap=None)
        self.assertFalse(agent._target_guard_enabled)

    def test_guard_parameters_must_be_finite_and_nonnegative(self):
        for name in ("target_guard_kappa", "target_guard_cap"):
            for bad in (-1.0, float("nan"), float("inf"), float("-inf")):
                with self.subTest(name=name, bad=bad):
                    with self.assertRaisesRegex(
                        ValueError, rf"{name} must be finite and >= 0"
                    ):
                        _make_agent(**{name: bad})

    def test_winsorize_clamps_only_the_outlier(self):
        _, agent = _make_agent(target_guard_kappa=4.0)
        obs, act, nobs, rew, dn, dt, alpha = _batch(agent, n=9)
        t_mf = th.linspace(-2.0, 2.0, 9).reshape(-1, 1)
        # consistent batch offset +0.5 with small spread, one huge outlier
        delta = 0.5 + 0.01 * th.arange(-4.0, 5.0).reshape(-1, 1)
        delta[8, 0] = 30.0
        agent._finite_difference_target = lambda *a, **k: t_mf
        agent._model_based_target = lambda *a, **k: t_mf + delta

        t = agent._guarded_model_based_target(obs, act, nobs, rew, dn, dt, alpha)
        # non-outliers pass through exactly (batch-consensus offset preserved)
        self.assertTrue(th.allclose(t[:8], (t_mf + delta)[:8], atol=1e-6))
        # the outlier is suppressed toward the consensus, direction kept
        self.assertLess(float(t[8]), float(t_mf[8]) + 1.0)
        self.assertGreater(float(t[8]), float(t_mf[8]))
        self.assertAlmostEqual(
            agent.logger.get_logger().name_to_value["train/guard_clamp_frac"], 1 / 9, places=6
        )
        self.assertEqual(
            agent.logger.get_logger().name_to_value["train/guard_cap_frac"], 0.0
        )

    def test_nonfinite_model_target_falls_to_anchor(self):
        _, agent = _make_agent(target_guard_kappa=4.0)
        obs, act, nobs, rew, dn, dt, alpha = _batch(agent, n=7)
        t_mf = th.linspace(-3.0, 3.0, 7).reshape(-1, 1)
        delta = th.full((7, 1), 0.5)
        delta[2, 0] = float("nan")
        agent._finite_difference_target = lambda *a, **k: t_mf
        agent._model_based_target = lambda *a, **k: t_mf + delta

        t = agent._guarded_model_based_target(obs, act, nobs, rew, dn, dt, alpha)
        self.assertTrue(th.isfinite(t).all())
        self.assertTrue(th.equal(t[2], t_mf[2]))
        good = th.arange(7) != 2
        self.assertTrue(th.allclose(t[good], (t_mf + delta)[good], atol=1e-6))
        self.assertAlmostEqual(
            agent.logger.get_logger().name_to_value["train/guard_nonfinite_frac"], 1 / 7,
            places=6,
        )
        self.assertEqual(
            agent.logger.get_logger().name_to_value["train/guard_clamp_frac"], 0.0
        )

    def test_all_nonfinite_model_targets_fall_to_anchor(self):
        _, agent = _make_agent(target_guard_kappa=4.0)
        obs, act, nobs, rew, dn, dt, alpha = _batch(agent, n=5)
        t_mf = th.linspace(-2.0, 2.0, 5).reshape(-1, 1)
        agent._finite_difference_target = lambda *a, **k: t_mf
        agent._model_based_target = lambda *a, **k: th.full_like(t_mf, float("nan"))

        t = agent._guarded_model_based_target(obs, act, nobs, rew, dn, dt, alpha)
        self.assertTrue(th.equal(t, t_mf))
        logged = agent.logger.get_logger().name_to_value
        self.assertEqual(logged["train/guard_nonfinite_frac"], 1.0)
        self.assertEqual(logged["train/guard_delta_med"], 0.0)
        self.assertEqual(logged["train/guard_delta_mad"], 0.0)
        self.assertEqual(logged["train/guard_clamp_frac"], 0.0)

    def test_typed_numerical_failure_falls_to_anchor_wholesale(self):
        _, agent = _make_agent(target_guard_kappa=4.0)
        obs, act, nobs, rew, dn, dt, alpha = _batch(agent, n=5)
        t_mf = th.full((5, 1), 2.0)
        agent._finite_difference_target = lambda *a, **k: t_mf

        def boom(*a, **k):
            raise ModelBasedTargetNumericalError("integration blew up")

        agent._model_based_target = boom
        t = agent._guarded_model_based_target(obs, act, nobs, rew, dn, dt, alpha)
        self.assertTrue(th.equal(t, t_mf))
        self.assertEqual(
            agent.logger.get_logger().name_to_value["train/guard_nonfinite_frac"], 1.0
        )

    def test_generic_runtime_and_oom_propagate_unchanged(self):
        for exc in (
            RuntimeError("synthetic programming failure"),
            th.OutOfMemoryError("synthetic OOM"),
        ):
            with self.subTest(exc_type=type(exc).__name__):
                _, agent = _make_agent(target_guard_kappa=4.0)
                obs, act, nobs, rew, dn, dt, alpha = _batch(agent, n=5)

                def boom(*args, _exc=exc, **kwargs):
                    raise _exc

                agent.dynamics_target_model.drift = boom
                with self.assertRaises(type(exc)) as caught:
                    agent._guarded_model_based_target(
                        obs, act, nobs, rew, dn, dt, alpha
                    )
                self.assertIs(caught.exception, exc)

    def test_model_target_shape_mismatch_propagates_as_configuration_error(self):
        _, agent = _make_agent(target_guard_kappa=4.0)
        obs, act, nobs, rew, dn, dt, alpha = _batch(agent, n=5)
        t_mf = th.ones(5, 1)
        agent._finite_difference_target = lambda *a, **k: t_mf
        agent._model_based_target = lambda *a, **k: th.ones(5)

        with self.assertRaisesRegex(ValueError, "shape must match its anchor"):
            agent._guarded_model_based_target(obs, act, nobs, rew, dn, dt, alpha)

    def test_nonfinite_anchor_still_raises(self):
        _, agent = _make_agent(target_guard_kappa=4.0)
        obs, act, nobs, rew, dn, dt, alpha = _batch(agent, n=4)
        agent._finite_difference_target = (
            lambda *a, **k: th.full((4, 1), float("nan"))
        )
        with self.assertRaisesRegex(
            ModelBasedTargetNumericalError, "component=guard_anchor"
        ):
            agent._guarded_model_based_target(obs, act, nobs, rew, dn, dt, alpha)

    def test_guard_arithmetic_cannot_return_nonfinite_target(self):
        _, agent = _make_agent(target_guard_kappa=4.0)
        obs, act, nobs, rew, dn, dt, alpha = _batch(agent, n=5)
        fmax = th.finfo(th.float32).max
        t_mf = th.tensor([[fmax], [0.0], [0.0], [0.0], [0.0]])
        t_model = th.tensor([[0.0], [fmax], [fmax], [fmax], [fmax]])
        self.assertTrue(th.isfinite(t_mf).all() and th.isfinite(t_model).all())
        agent._finite_difference_target = lambda *a, **k: t_mf
        agent._model_based_target = lambda *a, **k: t_model

        with self.assertRaisesRegex(
            ModelBasedTargetNumericalError, "component=guard_target"
        ):
            agent._guarded_model_based_target(obs, act, nobs, rew, dn, dt, alpha)

    def test_absolute_cap_bounds_the_target_scale(self):
        _, agent = _make_agent(target_guard_cap=50.0)
        self.assertTrue(agent._target_guard_enabled)
        obs, act, nobs, rew, dn, dt, alpha = _batch(agent, n=4)
        t_mf = th.full((4, 1), 400.0)  # runaway value scale in the anchor too
        agent._finite_difference_target = lambda *a, **k: t_mf
        agent._model_based_target = lambda *a, **k: t_mf.clone()

        t = agent._guarded_model_based_target(obs, act, nobs, rew, dn, dt, alpha)
        self.assertTrue(th.equal(t, th.full((4, 1), 50.0)))
        self.assertEqual(agent.logger.get_logger().name_to_value["train/guard_cap_frac"], 1.0)

    def test_guarded_train_smoke(self):
        _, agent = _make_agent(target_guard_kappa=6.0, target_guard_cap=150.0)
        self.assertTrue(agent._target_guard_enabled)
        obs_dim = int(agent.env.observation_space.shape[0])
        act_dim = int(agent.env.action_space.shape[0])
        for i in range(8):
            agent.replay_buffer.add(
                th.zeros(1, obs_dim).numpy(),
                th.zeros(1, act_dim).numpy(),
                th.zeros(1).numpy(),
                th.zeros(1).numpy(),
                th.zeros(1, obs_dim).numpy(),
                th.tensor([0.02 * i]).numpy(),
                th.tensor([0.02 * (i + 1)]).numpy(),
            )
        agent.train(gradient_steps=2, batch_size=4)
        logged = agent.logger.get_logger().name_to_value
        for key in (
            "train/guard_clamp_frac",
            "train/guard_cap_frac",
            "train/guard_nonfinite_frac",
            "train/guard_delta_med",
            "train/guard_delta_mad",
        ):
            self.assertIn(key, logged)
        self.assertEqual(logged["train/guard_nonfinite_frac"], 0.0)

    def test_guard_preserves_torch_rng_when_value_head_is_ready(self):
        _, agent = _make_agent(target_guard_kappa=6.0, target_guard_cap=150.0)
        self.assertTrue(agent._value_head_ready)
        args = _batch(agent)
        th.manual_seed(1234)
        before = th.random.get_rng_state().clone()
        target = agent._guarded_model_based_target(*args)
        self.assertTrue(th.isfinite(target).all())
        self.assertTrue(th.equal(th.random.get_rng_state(), before))


if __name__ == "__main__":
    unittest.main()
