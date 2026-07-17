import inspect
import unittest

import torch as th

from environment import DMCContinuousEnv
from common.utils import load_ct_hyperparams_from_table
from algorithms.ct_sac import (
    CTSAC,
    ModelBasedTargetNumericalError,
    reanchor_gate_statistics,
    reanchored_endpoint,
)
from models.actor_q_critic import ActorQCriticModel


class _ConstantDynamics(th.nn.Module):
    """Constant drift c: the Euler flow is exact, Phi_t(z) = z + c*t."""

    def __init__(self, value=0.0):
        super().__init__()
        self.value = float(value)

    def drift(self, obs, action):
        x = th.as_tensor(obs, dtype=th.float32)
        return th.ones_like(x) * self.value

    def diffusion(self, obs):
        return None


class _NaNDynamics(th.nn.Module):
    def drift(self, obs, action):
        x = th.as_tensor(obs, dtype=th.float32)
        return th.full_like(x, float("nan"))

    def diffusion(self, obs):
        return None


def _make_agent(
    *, dynamics=None, generator_substeps=2, value_warmup=0,
    v_net_arch=(8,), **kwargs
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
        v_net_arch=v_net_arch,
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
        dynamics_model=dynamics if dynamics is not None else _ConstantDynamics(),
        generator_substeps=generator_substeps,
        value_warmup=value_warmup,
        dynamics_warmup=0,
        **kwargs,
    )
    return env, agent


def _batch(agent, dts, next_obs=None):
    n = len(dts)
    obs_dim = int(agent.env.observation_space.shape[0])
    act_dim = int(agent.env.action_space.shape[0])
    g = th.Generator().manual_seed(0)
    obs = 0.1 * th.randn(n, obs_dim, generator=g)
    if next_obs is None:
        next_obs = 0.1 * th.randn(n, obs_dim, generator=g)
    return (
        obs,
        th.zeros(n, act_dim),
        next_obs,
        th.ones(n, 1),
        th.zeros(n, 1),
        th.tensor(dts, dtype=th.float32).reshape(-1, 1),
        th.tensor(0.2),
    )


class TestReanchoredEndpoint(unittest.TestCase):
    """Endpoint geometry against the exact flow of a constant drift."""

    def _check(self, c, dts):
        _, agent = _make_agent(dynamics=_ConstantDynamics(c))
        T = float(agent.dt_default)
        obs, act, nobs, _, _, dt, _ = _batch(agent, dts)
        x_re, e, ms = reanchored_endpoint(
            agent.dynamics_target_model.drift, obs, act, nobs, dt, T,
            max_step=agent._integration_max_step(),
        )
        for i, dti in enumerate(dts):
            e_exp = (
                th.zeros_like(nobs[i])
                if dti == T
                else nobs[i] - (obs[i] + c * dti)
            )
            th.testing.assert_close(e[i], e_exp, atol=1e-6, rtol=0)
            if dti <= T:
                x_exp = nobs[i] + c * (T - dti)
                self.assertAlmostEqual(float(ms[i]), T - dti, places=7)
            else:
                x_exp = (obs[i] + c * T) + (T / dti) * e_exp
                self.assertAlmostEqual(float(ms[i]), T, places=7)
            th.testing.assert_close(x_re[i], x_exp, atol=1e-6, rtol=0)
        return agent

    def test_short_nominal_and_long_durations(self):
        agent = self._check(c=0.7, dts=[0.002, 0.008, 0.01, 0.03])
        self.assertGreater(agent.dt_default, 0.002)
        self.assertLess(agent.dt_default, 0.03)

    def test_zero_drift_reduces_to_data(self):
        _, agent = _make_agent(dynamics=_ConstantDynamics(0.0))
        T = float(agent.dt_default)
        obs, act, nobs, _, _, dt, _ = _batch(agent, [0.002, T])
        x_re, e, _ = reanchored_endpoint(
            agent.dynamics_target_model.drift, obs, act, nobs, dt, T,
            max_step=agent._integration_max_step(),
        )
        # zero drift: the transported endpoint IS the observed next state
        th.testing.assert_close(x_re, nobs, atol=1e-7, rtol=0)
        th.testing.assert_close(e[0], nobs[0] - obs[0], atol=1e-7, rtol=0)
        th.testing.assert_close(e[1], th.zeros_like(e[1]), atol=1e-7, rtol=0)

    def test_scalar_duration_broadcasts_across_batch(self):
        _, agent = _make_agent(dynamics=_ConstantDynamics(0.0))
        T = float(agent.dt_default)
        obs, act, nobs, _, _, _, _ = _batch(agent, [T / 2] * 3)
        for scalar_dt in (T / 2, 2 * T):
            x_re, innovation, model_seconds = reanchored_endpoint(
                agent.dynamics_target_model.drift,
                obs,
                act,
                nobs,
                scalar_dt,
                T,
            )
            self.assertEqual(x_re.shape, obs.shape)
            self.assertEqual(innovation.shape, obs.shape)
            self.assertEqual(model_seconds.shape, (obs.shape[0], 1))

    def test_transport_rho_vanishes_as_duration_mismatch_vanishes(self):
        _, agent = _make_agent()
        T = float(agent.dt_default)
        durations = [T / 4, T / 2, T, 2 * T, 4 * T]
        obs, _, _, _, _, _, _ = _batch(agent, durations)
        dt = th.tensor(durations).reshape(-1, 1)
        # Equal realized and innovation rates make raw rho exactly one; only
        # the transport fraction should distinguish the rows.
        next_obs = obs + dt
        innovation = dt.expand_as(obs)
        rho, raw_rho, fraction = reanchor_gate_statistics(
            obs, next_obs, innovation, dt, T
        )
        th.testing.assert_close(raw_rho, th.ones_like(raw_rho), atol=1e-5, rtol=0)
        th.testing.assert_close(
            fraction,
            th.tensor([[0.75], [0.5], [0.0], [0.5], [0.75]]),
            atol=1e-6,
            rtol=0,
        )
        th.testing.assert_close(rho, fraction, atol=1e-5, rtol=0)
        self.assertTrue(bool(th.all((fraction >= 0.0) & (fraction <= 1.0))))

    def test_nominal_duration_skips_counterfactual_model_roll(self):
        _, agent = _make_agent()
        T = float(agent.dt_default)
        obs, act, nobs, _, _, _, _ = _batch(agent, [T] * 3)

        def raising_drift(_obs, _actions):
            raise RuntimeError("must not be evaluated")

        x_re, innovation, model_seconds = reanchored_endpoint(
            raising_drift, obs, act, nobs, T, T
        )
        th.testing.assert_close(x_re, nobs)
        th.testing.assert_close(innovation, th.zeros_like(innovation))
        th.testing.assert_close(model_seconds, th.zeros_like(model_seconds))


class TestReanchoredTarget(unittest.TestCase):
    def test_new_options_do_not_shift_existing_positional_cadence_arguments(self):
        params = list(inspect.signature(CTSAC.__init__).parameters)
        self.assertEqual(
            params[-5:],
            [
                "dynamics_publish_interval",
                "dynamics_train_interval",
                "dynamics_rollout_interval",
                "target_reanchor",
                "target_reanchor_gate_rho",
            ],
        )

    def test_fork_reanchor_mode_is_wired(self):
        _, _, _, algo_kwargs, _ = load_ct_hyperparams_from_table(
            "ct_sac", "cartpole-swingup", "fork_reanchor"
        )
        self.assertEqual(str(algo_kwargs["target_reanchor"]).lower(), "true")
        self.assertEqual(algo_kwargs["target_reanchor_gate_rho"], 1)

    def test_requires_quadrature_form(self):
        with self.assertRaisesRegex(ValueError, "generator_substeps"):
            _make_agent(generator_substeps=0, target_reanchor=True)

    def test_csv_coercions(self):
        _, agent = _make_agent(target_reanchor="", target_reanchor_gate_rho="")
        self.assertFalse(agent.target_reanchor)
        self.assertEqual(agent.target_reanchor_gate_rho, 0.0)
        _, agent = _make_agent(target_reanchor="True",
                               target_reanchor_gate_rho="1")
        self.assertTrue(agent.target_reanchor)
        self.assertEqual(agent.target_reanchor_gate_rho, 1.0)
        with self.assertRaisesRegex(ValueError, "requires target_reanchor"):
            _make_agent(target_reanchor_gate_rho=1.0)
        with self.assertRaisesRegex(ValueError, "target_reanchor_gate_rho"):
            _make_agent(target_reanchor_gate_rho=-1.0)

    def test_requires_explicit_ready_value_head(self):
        with self.assertRaisesRegex(ValueError, "explicit V-head"):
            _make_agent(v_net_arch=None, target_reanchor=True)

        _, agent = _make_agent(value_warmup=3, target_reanchor=True)
        self.assertFalse(agent._dynamics_ready)
        batch = _batch(agent, [0.002, 0.01, 0.03])
        state_before = th.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, "ready explicit V-head"):
            agent._model_based_target(*batch)
        self.assertTrue(th.equal(state_before, th.get_rng_state()))
        agent._value_updates = 3
        self.assertTrue(agent._dynamics_ready)

    def test_target_matches_manual_value_read(self):
        _, agent = _make_agent(dynamics=_ConstantDynamics(0.0),
                               target_reanchor=True)
        T = float(agent.dt_default)
        obs, act, nobs, rew, dn, dt, alpha = _batch(agent, [0.002, 0.008, T])
        y = agent._model_based_target(obs, act, nobs, rew, dn, dt, alpha)
        with th.no_grad():
            v_cur = agent.model.target_value(obs)
            v_re = agent.model.target_value(nobs)  # zero drift: x_re = x'
        y_exp = rew + (v_re - v_cur) + v_cur - agent.beta * v_cur - v_cur
        y_exp = rew + (1 - dn) * (v_cur + (v_re - v_cur) - agent.beta * v_cur)
        th.testing.assert_close(y, y_exp, atol=1e-6, rtol=0)

    def test_gate_trusts_model_when_innovation_zero(self):
        c = 0.5
        _, agent = _make_agent(dynamics=_ConstantDynamics(c),
                               target_reanchor=True,
                               target_reanchor_gate_rho=1.0)
        T = float(agent.dt_default)
        dts = [0.002, 0.008, T]
        obs, act, _, rew, dn, dt, alpha = _batch(agent, dts)
        nobs = obs + c * dt  # x' exactly on the model orbit -> innovation 0
        y = agent._model_based_target(obs, act, nobs, rew, dn, dt, alpha)
        agent.target_reanchor_gate_rho = 0.0
        y_pure = agent._model_based_target(obs, act, nobs, rew, dn, dt, alpha)
        th.testing.assert_close(y, y_pure, atol=1e-6, rtol=0)
        logged = agent.logger.get_logger().name_to_value
        self.assertGreater(logged["train/reanchor_lambda_mean"], 0.999)

    def test_gate_falls_to_anchor_when_innovation_huge(self):
        _, agent = _make_agent(dynamics=_ConstantDynamics(0.0),
                               target_reanchor=True,
                               target_reanchor_gate_rho=1.0)
        obs, act, _, rew, dn, dt, alpha = _batch(agent, [0.002, 0.002, 0.002])
        nobs = obs.clone()
        nobs[1] += 500.0  # off-orbit by ~5000 displacement scales
        y = agent._model_based_target(obs, act, nobs, rew, dn, dt, alpha)
        y_fd = agent._finite_difference_target(obs, nobs, rew, dn, dt, alpha)
        th.testing.assert_close(y[1], y_fd[1], atol=1e-5, rtol=0)

    def test_gate_trusts_measured_endpoint_at_nominal_duration(self):
        _, agent = _make_agent(dynamics=_ConstantDynamics(0.0),
                               target_reanchor=True,
                               target_reanchor_gate_rho=1.0)
        T = float(agent.dt_default)
        obs, act, _, rew, dn, dt, alpha = _batch(agent, [T, T, T, T])
        nobs = obs + 0.25  # deliberately far from the zero-drift model orbit
        y_gate = agent._model_based_target(obs, act, nobs, rew, dn, dt, alpha)
        logged = agent.logger.get_logger().name_to_value
        self.assertAlmostEqual(logged["train/reanchor_lambda_mean"], 1.0, places=7)
        self.assertAlmostEqual(logged["train/reanchor_rho_med"], 0.0, places=7)
        self.assertTrue(
            th.isnan(th.tensor(logged["train/reanchor_innovation_rho_med"]))
        )
        self.assertAlmostEqual(
            logged["train/reanchor_innovation_valid_frac"], 0.0, places=7
        )

        agent.target_reanchor_gate_rho = 0.0
        y_pure = agent._model_based_target(obs, act, nobs, rew, dn, dt, alpha)
        th.testing.assert_close(y_gate, y_pure, atol=1e-7, rtol=0)

    def test_nominal_rows_survive_a_nonfinite_model_in_a_mixed_batch(self):
        _, agent = _make_agent(
            dynamics=_NaNDynamics(),
            target_reanchor=True,
            target_reanchor_gate_rho=1.0,
        )
        T = float(agent.dt_default)
        obs, act, _, rew, dn, dt, alpha = _batch(agent, [T, T / 2])
        nobs = obs + 0.25

        y = agent._model_based_target(obs, act, nobs, rew, dn, dt, alpha)
        y_fd = agent._finite_difference_target(obs, nobs, rew, dn, dt, alpha)
        with th.no_grad():
            v_cur = agent.model.target_value(obs)
            v_data = agent.model.target_value(nobs)
            y_re = rew + (1 - dn) * (v_data - agent.beta * v_cur)

        # The exact-horizon row uses only the measured endpoint. The row that
        # needs transport sees the bad flow and falls back independently.
        th.testing.assert_close(y[0], y_re[0], atol=1e-7, rtol=0)
        th.testing.assert_close(y[1], y_fd[1], atol=1e-7, rtol=0)
        self.assertAlmostEqual(
            agent.logger.get_logger().name_to_value[
                "train/reanchor_nonfinite_frac"
            ],
            0.5,
        )

    def test_gate_off_nonfinite_raises_and_gate_on_recovers(self):
        _, agent = _make_agent(dynamics=_NaNDynamics(), target_reanchor=True)
        obs, act, nobs, rew, dn, dt, alpha = _batch(agent, [0.002, 0.008])
        with self.assertRaises(ModelBasedTargetNumericalError):
            agent._model_based_target(obs, act, nobs, rew, dn, dt, alpha)
        agent.target_reanchor_gate_rho = 1.0
        y = agent._model_based_target(obs, act, nobs, rew, dn, dt, alpha)
        y_fd = agent._finite_difference_target(obs, nobs, rew, dn, dt, alpha)
        th.testing.assert_close(y, y_fd, atol=1e-6, rtol=0)
        self.assertEqual(
            agent.logger.get_logger().name_to_value[
                "train/reanchor_nonfinite_frac"], 1.0,
        )

    def test_consumes_no_rng(self):
        _, agent = _make_agent(target_reanchor=True,
                               target_reanchor_gate_rho=1.0)
        batch = _batch(agent, [0.002, 0.01, 0.03])
        state_before = th.get_rng_state().clone()
        agent._model_based_target(*batch)
        self.assertTrue(th.equal(state_before, th.get_rng_state()))

    def test_guard_composes_over_reanchor(self):
        _, agent = _make_agent(target_reanchor=True,
                               target_reanchor_gate_rho=1.0,
                               target_guard_kappa=6.0,
                               target_guard_cap=150.0)
        self.assertTrue(agent._target_guard_enabled)
        obs, act, nobs, rew, dn, dt, alpha = _batch(agent, [0.002, 0.01, 0.03])
        y = agent._guarded_model_based_target(obs, act, nobs, rew, dn, dt, alpha)
        self.assertTrue(th.isfinite(y).all())
        logged = agent.logger.get_logger().name_to_value
        self.assertIn("train/guard_clamp_frac", logged)
        self.assertIn("train/reanchor_rho_med", logged)

    def test_train_smoke(self):
        _, agent = _make_agent(target_reanchor=True,
                               target_reanchor_gate_rho=1.0)
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
        for key in ("train/reanchor_rho_med", "train/reanchor_lambda_mean",
                    "train/reanchor_model_seconds_mean",
                    "train/reanchor_transport_seconds_mean",
                    "train/reanchor_mismatch_fraction_mean",
                    "train/reanchor_innovation_valid_frac",
                    "train/reanchor_long_frac"):
            self.assertIn(key, logged)


if __name__ == "__main__":
    unittest.main()
