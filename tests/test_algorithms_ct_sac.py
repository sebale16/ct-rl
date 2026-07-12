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
            dynamics_require_value_head=require_value,
        )
        return env, agent

    def test_first_publication_is_hard_then_ema_and_frozen(self):
        _, agent = self._agent(target_tau=0.25)
        self.assertIsNot(agent.dynamics_model, agent.dynamics_target_model)
        self.assertTrue(all(not p.requires_grad for p in
                            agent.dynamics_target_model.parameters()))

        with th.no_grad():
            agent.dynamics_model.value.fill_(2.0)
        agent._publish_dynamics_target()
        self.assertAlmostEqual(agent.dynamics_target_model.value.item(), 2.0)

        with th.no_grad():
            agent.dynamics_model.value.fill_(4.0)
        agent._publish_dynamics_target()
        self.assertAlmostEqual(agent.dynamics_target_model.value.item(), 2.5)
        self.assertEqual(agent._dynamics_publications, 2)

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

    def test_readiness_counts_publications_and_waits_for_value_head(self):
        _, agent = self._agent(value_warmup=3, dynamics_warmup=2)
        agent._dynamics_updates = 100
        self.assertFalse(agent._dynamics_ready)
        agent._dynamics_publications = 2
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
