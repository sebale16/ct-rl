import unittest
import torch as th

from environment import DMCContinuousEnv, VecContinuousEnv, Monitor
from algorithms.ct_sac import CTSAC
from models.actor_q_critic import ActorQCriticModel


class _ConstantLearnedDynamics(th.nn.Module):
    """Tiny trainable drift used to isolate CT-SAC's publication machinery."""

    def __init__(self, value=0.0):
        super().__init__()
        self.value = th.nn.Parameter(th.tensor(float(value)))

    def drift(self, obs, action):
        x = th.as_tensor(obs, dtype=th.float32, device=self.value.device)
        return th.ones_like(x) * self.value

    def diffusion(self, obs):
        return None

    def fit_step(
        self, obs, action, next_obs, dt, optimizer, **kwargs
    ):
        # Minimal optimizer-certified fit used to exercise CTSAC cadence rather
        # than dynamics-model numerics.
        optimizer.zero_grad()
        loss = self.value.square() * 0.0
        loss.backward()
        optimizer.step()
        self.last_fit_accepted = True
        self.last_fit_grad_norm = 0.0
        return float(loss.detach())


class _FrozenDynamics(th.nn.Module):
    def drift(self, obs, action):
        return th.zeros_like(th.as_tensor(obs, dtype=th.float32))

    def diffusion(self, obs):
        return None


class TestCTSAC(unittest.TestCase):
    def setUp(self):
        """
        Set up a small environment and a CTSAC agent to test.
        """
        self.env = DMCContinuousEnv(
            domain_name="cartpole",
            task_name="swingup",
            time_sampling="uniform",
            dt=0.02,
            episode_duration=0.1,  # short episodes
        )

        self.model = ActorQCriticModel(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            q_net_arch=[16, 16],
            pi_net_arch=[16, 16],
        )

        self.agent = CTSAC(
            env=self.env,
            model=self.model,
            learning_starts=10,
            batch_size=8,
            buffer_size=100,
            gradient_steps=2,
            train_freq=2,
            seed=123,
        )

    def test_learn_runs(self):
        """
        Test that the learn method runs for a few timesteps without crashing.
        """
        try:
            self.agent.learn(total_timesteps=20)
        except Exception as e:
            self.fail(f"agent.learn() raised an exception: {e}")

    def test_learn_runs_vectorized(self):
        """
        Test that the learn method runs with a vectorized environment.
        """
        n_envs = 3
        env_fns = [
            lambda: Monitor(
                DMCContinuousEnv("cartpole", "swingup", episode_duration=0.1, dt=0.02)
            )
            for _ in range(n_envs)
        ]
        vec_env = VecContinuousEnv(env_fns)

        agent = CTSAC(
            env=vec_env,
            model="ActorQCriticModel",
            model_kwargs={"q_net_arch": [16], "pi_net_arch": [16]},
            learning_starts=10,
            batch_size=8,
            buffer_size=100,
            seed=123,
        )

        try:
            agent.learn(total_timesteps=20)
        except Exception as e:
            self.fail(f"agent.learn() with vectorized env raised an exception: {e}")


class TestDynamicsPublication(unittest.TestCase):
    def _agent(
        self,
        *,
        dynamics=None,
        with_value=True,
        value_warmup=3,
        dynamics_warmup=2,
        target_tau=0.25,
        require_value=True,
        publish_interval=1,
        train_interval=1,
        rollout_interval=1,
        fit_horizon=1,
        fit_horizon_warmup=0,
    ):
        env = DMCContinuousEnv(
            "cartpole", "swingup", time_sampling="uniform", dt=0.02,
            episode_duration=0.1,
        )
        model = ActorQCriticModel(
            observation_space=env.observation_space,
            action_space=env.action_space,
            q_net_arch=[8],
            pi_net_arch=[8],
            v_net_arch=[8] if with_value else None,
            device="cpu",
        )
        dynamics = dynamics or _ConstantLearnedDynamics()
        agent = CTSAC(
            env=env,
            model=model,
            device="cpu",
            learning_starts=10,
            batch_size=4,
            buffer_size=32,
            use_model_based_q=True,
            dynamics_model=dynamics,
            generator_substeps=2,
            value_warmup=value_warmup,
            dynamics_warmup=dynamics_warmup,
            dynamics_target_tau=target_tau,
            dynamics_publish_max_flow_error_ratio=2.0,
            dynamics_publish_interval=publish_interval,
            dynamics_train_interval=train_interval,
            dynamics_rollout_interval=rollout_interval,
            dynamics_fit_horizon=fit_horizon,
            dynamics_fit_horizon_warmup=fit_horizon_warmup,
            dynamics_require_value_head=require_value,
        )
        return env, agent

    @staticmethod
    def _fill_replay(agent, n=8):
        obs_dim = int(agent.env.observation_space.shape[0])
        act_dim = int(agent.env.action_space.shape[0])
        for i in range(n):
            obs = th.zeros(1, obs_dim).numpy()
            action = th.zeros(1, act_dim).numpy()
            agent.replay_buffer.add(
                obs,
                action,
                th.zeros(1).numpy(),
                th.zeros(1).numpy(),
                obs.copy(),
                th.tensor([0.02 * i]).numpy(),
                th.tensor([0.02 * (i + 1)]).numpy(),
            )

    def test_publication_interval_controls_full_validation_in_train(self):
        _, agent = self._agent(
            publish_interval=3, dynamics_warmup=100, value_warmup=100
        )
        self._fill_replay(agent)
        calls = 0

        def accept(*args, **kwargs):
            nonlocal calls
            calls += 1
            return True, 0.0, "accepted"

        agent._post_fit_flow_quality = accept
        for _ in range(3):
            agent.train(gradient_steps=1, batch_size=4)
        self.assertEqual(agent._dynamics_updates, 3)
        self.assertEqual(calls, 1)
        self.assertEqual(agent._dynamics_publications, 1)

    def test_bad_fit_rolls_back_without_full_validation(self):
        _, agent = self._agent(publish_interval=10)
        self._fill_replay(agent)
        with th.no_grad():
            agent.dynamics_model.value.fill_(5.0)

        def skip(*args, **kwargs):
            agent.dynamics_model.last_fit_accepted = False
            return 0.0

        full_checks = 0

        def should_not_validate(*args, **kwargs):
            nonlocal full_checks
            full_checks += 1
            return True, 0.0, "accepted"

        agent.dynamics_model.fit_step = skip
        agent._post_fit_flow_quality = should_not_validate
        agent.train(gradient_steps=1, batch_size=4)
        self.assertEqual(full_checks, 0)
        self.assertEqual(agent._dynamics_rollbacks, 1)
        self.assertEqual(agent._dynamics_fit_rejections, 1)
        self.assertEqual(agent._dynamics_publish_rejections, 0)
        self.assertEqual(agent.dynamics_model.value.item(), 0.0)

    def test_publication_interval_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "publish_interval"):
            self._agent(publish_interval=0)

    def test_dynamics_fit_interval_decouples_live_model_from_critic_updates(self):
        _, agent = self._agent(
            train_interval=2,
            publish_interval=100,
            dynamics_warmup=100,
            value_warmup=100,
        )
        self._fill_replay(agent)
        fit_calls = 0
        original_fit = agent.dynamics_model.fit_step

        def count_fit(*args, **kwargs):
            nonlocal fit_calls
            fit_calls += 1
            return original_fit(*args, **kwargs)

        agent.dynamics_model.fit_step = count_fit
        agent.train(gradient_steps=5, batch_size=4)
        self.assertEqual(agent._n_updates, 5)
        self.assertEqual(agent._dynamics_updates, 3)
        self.assertEqual(fit_calls, 3)

    def test_rollout_horizon_is_periodic_after_local_curriculum(self):
        _, agent = self._agent(
            fit_horizon=4,
            fit_horizon_warmup=2,
            rollout_interval=3,
        )
        expected = (1, 1, 4, 1, 1, 4, 1)
        actual = []
        for update in range(len(expected)):
            agent._dynamics_updates = update
            actual.append(agent._current_dynamics_fit_horizon())
        self.assertEqual(tuple(actual), expected)

    def test_train_combines_fit_and_rollout_cadences(self):
        _, agent = self._agent(
            train_interval=2,
            fit_horizon=4,
            fit_horizon_warmup=0,
            rollout_interval=4,
            publish_interval=100,
            dynamics_warmup=100,
            value_warmup=100,
        )
        self._fill_replay(agent, n=12)
        horizons = []
        original_h1 = agent.dynamics_model.fit_step

        def h1(*args, **kwargs):
            horizons.append(1)
            return original_h1(*args, **kwargs)

        def h4(obs, actions, next_obs, dt, mask, optimizer, **kwargs):
            horizons.append(int(actions.shape[1]))
            return original_h1(
                obs,
                actions[:, 0],
                next_obs[:, 0],
                dt[:, 0],
                optimizer,
                **kwargs,
            )

        agent.dynamics_model.fit_step = h1
        agent.dynamics_model.fit_step_rollout = h4
        agent.train(gradient_steps=9, batch_size=4)
        self.assertEqual(horizons, [4, 1, 1, 1, 4])
        self.assertEqual(agent._dynamics_updates, 5)
        self.assertEqual(agent._n_updates, 9)

    def test_dynamics_fit_intervals_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "dynamics_train_interval"):
            self._agent(train_interval=0)
        with self.assertRaisesRegex(ValueError, "dynamics_rollout_interval"):
            self._agent(rollout_interval=0)

    def test_first_publication_is_hard_then_ema_and_frozen(self):
        _, agent = self._agent(target_tau=0.25)
        self.assertIsNot(agent.dynamics_model, agent.dynamics_target_model)
        self.assertTrue(all(not p.requires_grad for p in
                            agent.dynamics_target_model.parameters()))

        agent._dynamics_updates = 5
        with th.no_grad():
            agent.dynamics_model.value.fill_(2.0)
        agent._publish_dynamics_target()
        self.assertAlmostEqual(agent.dynamics_target_model.value.item(), 2.0)

        agent._dynamics_updates = 9
        with th.no_grad():
            agent.dynamics_model.value.fill_(4.0)
        agent._publish_dynamics_target()
        tau_eff = 1.0 - (1.0 - 0.25) ** 4
        self.assertAlmostEqual(
            agent.dynamics_target_model.value.item(),
            2.0 * (1.0 - tau_eff) + 4.0 * tau_eff,
        )
        self.assertEqual(agent._dynamics_publications, 2)

    def test_rollback_restarts_cadence_correct_ema_clock(self):
        _, agent = self._agent(target_tau=0.25)
        agent._dynamics_updates = 5
        with th.no_grad():
            agent.dynamics_model.value.fill_(2.0)
        agent._publish_dynamics_target()

        # A rejected trajectory may have consumed many updates, but rollback
        # synchronizes live back to target; those discarded updates must not make
        # the next accepted EMA artificially aggressive.
        agent._dynamics_updates = 20
        with th.no_grad():
            agent.dynamics_model.value.fill_(9.0)
        agent._rollback_live_dynamics()
        self.assertAlmostEqual(agent.dynamics_model.value.item(), 2.0)

        agent._dynamics_updates = 24
        with th.no_grad():
            agent.dynamics_model.value.fill_(4.0)
        agent._publish_dynamics_target()
        tau_eff = 1.0 - (1.0 - 0.25) ** 4
        self.assertAlmostEqual(
            agent.dynamics_target_model.value.item(),
            2.0 * (1.0 - tau_eff) + 4.0 * tau_eff,
        )

    def test_quality_check_and_nonfinite_rollback(self):
        env, agent = self._agent()
        batch = 4
        obs = th.zeros(batch, int(env.observation_space.shape[0]))
        actions = th.zeros(batch, int(env.action_space.shape[0]))
        dt = th.full((batch, 1), 0.02)
        with th.no_grad():
            agent.dynamics_model.value.fill_(1.0)
        next_obs = obs + 0.02
        accepted, ratio, _ = agent._post_fit_flow_quality(
            obs, actions, next_obs, dt, 0.0
        )
        self.assertTrue(accepted)
        self.assertLess(ratio, 1e-5)

        safe = agent.dynamics_target_model.value.detach().clone()
        agent.dynamics_optimizer.state[agent.dynamics_model.value]["sentinel"] = 1
        with th.no_grad():
            agent.dynamics_model.value.fill_(float("nan"))
        accepted, _, reason = agent._post_fit_flow_quality(
            obs, actions, next_obs, dt, 0.0
        )
        self.assertFalse(accepted)
        self.assertIn("non-finite model state", reason)
        agent._rollback_live_dynamics()
        self.assertTrue(th.equal(agent.dynamics_model.value, safe))
        self.assertEqual(len(agent.dynamics_optimizer.state), 0)
        self.assertEqual(agent._dynamics_rollbacks, 1)

    def test_quality_check_rejects_only_typed_flow_failures(self):
        env, agent = self._agent()
        batch = 4
        obs = th.zeros(batch, int(env.observation_space.shape[0]))
        actions = th.zeros(batch, int(env.action_space.shape[0]))
        next_obs = obs.clone()
        dt = th.full((batch, 1), 0.02)

        agent.dynamics_model.drift = (
            lambda x, _a: th.full_like(x, float("nan"))
        )
        accepted, _, reason = agent._post_fit_flow_quality(
            obs, actions, next_obs, dt, 0.0
        )
        self.assertFalse(accepted)
        self.assertIn("flow evaluation failed", reason)

        for exc in (
            RuntimeError("synthetic programming failure"),
            th.OutOfMemoryError("synthetic OOM"),
        ):
            with self.subTest(exc_type=type(exc).__name__):
                _, agent = self._agent()

                def boom(*args, _exc=exc, **kwargs):
                    raise _exc

                agent.dynamics_model.drift = boom
                with self.assertRaises(type(exc)) as caught:
                    agent._post_fit_flow_quality(
                        obs, actions, next_obs, dt, 0.0
                    )
                self.assertIs(caught.exception, exc)

    def test_skipped_optimizer_update_cannot_count_as_publication(self):
        env, agent = self._agent()
        batch = 4
        obs = th.zeros(batch, int(env.observation_space.shape[0]))
        actions = th.zeros(batch, int(env.action_space.shape[0]))
        dt = th.full((batch, 1), 0.02)
        next_obs = obs.clone()
        agent.dynamics_model.last_fit_accepted = False
        accepted, _, reason = agent._post_fit_flow_quality(
            obs, actions, next_obs, dt, 0.0
        )
        self.assertFalse(accepted)
        self.assertIn("optimizer update was skipped", reason)
        self.assertEqual(agent._dynamics_publications, 0)

    def test_optimizer_certificate_skips_redundant_state_scan(self):
        env, agent = self._agent()
        batch = 4
        obs = th.zeros(batch, int(env.observation_space.shape[0]))
        actions = th.zeros(batch, int(env.action_space.shape[0]))
        dt = th.full((batch, 1), 0.02)
        next_obs = obs.clone()
        agent.dynamics_model.last_fit_accepted = True

        def unexpected_scan(*args, **kwargs):
            raise AssertionError("certified fit should not rescan state_dict")

        agent.dynamics_model.state_dict = unexpected_scan
        accepted, _, _ = agent._post_fit_flow_quality(
            obs, actions, next_obs, dt, 0.0
        )
        self.assertTrue(accepted)

    def test_readiness_uses_update_warmup_one_publication_and_value_head(self):
        _, agent = self._agent(value_warmup=3, dynamics_warmup=2)
        agent._dynamics_updates = 1
        agent._dynamics_publications = 100
        self.assertFalse(agent._dynamics_ready)
        agent._dynamics_updates = 2
        agent._dynamics_publications = 0
        self.assertFalse(agent._dynamics_ready)
        agent._dynamics_publications = 1
        agent._value_updates = 2
        self.assertFalse(agent._dynamics_ready)
        agent._value_updates = 3
        self.assertTrue(agent._dynamics_ready)

    def test_learned_quadrature_requires_value_head_by_default(self):
        with self.assertRaisesRegex(ValueError, "requires an explicit V-head"):
            self._agent(with_value=False, require_value=True)

        # Parameterless oracle/custom dynamics keep their legacy behavior and do
        # not need a copied target or an explicit V-head.
        _, agent = self._agent(
            dynamics=_FrozenDynamics(), with_value=False, require_value=True
        )
        self.assertFalse(agent._train_dynamics)
        self.assertIs(agent.dynamics_model, agent.dynamics_target_model)
        self.assertTrue(agent._dynamics_ready)

    def test_rejected_live_model_cannot_poison_quadrature(self):
        env, agent = self._agent(value_warmup=0, dynamics_warmup=1)
        with th.no_grad():
            agent.dynamics_model.value.zero_()
        agent._publish_dynamics_target()
        with th.no_grad():
            agent.dynamics_model.value.fill_(float("nan"))

        obs = th.zeros(4, int(env.observation_space.shape[0]))
        actions = th.zeros(4, int(env.action_space.shape[0]))
        target = agent._substep_quadrature_target(
            obs, actions, th.zeros(4, 1), th.zeros(4, 1),
            th.tensor(float(agent.alpha)),
        )
        self.assertTrue(th.all(th.isfinite(target)))

    def test_quadrature_identifies_first_nonfinite_component(self):
        env, agent = self._agent(
            dynamics=_FrozenDynamics(), value_warmup=0, dynamics_warmup=0
        )
        obs = th.zeros(4, int(env.observation_space.shape[0]))
        actions = th.zeros(4, int(env.action_space.shape[0]))
        args = (
            obs, actions, th.zeros(4, 1), th.zeros(4, 1),
            th.tensor(float(agent.alpha)),
        )

        calls = 0
        original = agent._state_value

        def bad_second_value(x, alpha, use_target=True):
            nonlocal calls
            calls += 1
            if calls == 2:
                return th.full((x.shape[0], 1), float("nan"))
            return original(x, alpha, use_target=use_target)

        agent._state_value = bad_second_value
        with self.assertRaisesRegex(RuntimeError, "component=V_next"):
            agent._substep_quadrature_target(*args)
