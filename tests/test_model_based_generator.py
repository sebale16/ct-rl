import unittest

import numpy as np
import torch as th

from environment.dmc import DMCContinuousEnv
from algorithms.ct_sac import CTSAC
from models.actor_q_critic import ActorQCriticModel
from models.port_hamiltonian import PortHamiltonianModel


def _corr_slope(pred: th.Tensor, target: th.Tensor):
    """Pearson correlation and regression slope of pred onto target (flattened)."""
    p, t = pred.reshape(-1), target.reshape(-1)
    pm, tm = p.mean(), t.mean()
    cov = ((p - pm) * (t - tm)).mean()
    corr = (cov / (p.std(unbiased=False) * t.std(unbiased=False) + 1e-12)).item()
    slope = (cov / (t.var(unbiased=False) + 1e-12)).item()
    return corr, slope


def _collect_env(env, n: int):
    """Roll out random actions and return (obs, act, next_obs, dt) tensors."""
    env.action_space.seed(0)  # determinism
    O, A, NO, DT = [], [], [], []
    obs, _ = env.reset()
    for _ in range(n):
        a = env.action_space.sample()
        o, t, _, r, no, nt, term, trunc, _ = env.step_dt(a)
        O.append(o)
        A.append(a)
        NO.append(no)
        DT.append(nt - t)
        obs = no if not (term or trunc) else env.reset()[0]
    to = lambda x: th.as_tensor(np.asarray(x, dtype=np.float32))
    return to(O), to(A), to(NO), to(DT).reshape(-1, 1)


def _train_eval_phast(O, A, NO, DT, n_train, steps=1500, hidden=(128, 128)):
    """Fit a phast PortHamiltonianModel on a train split; return (ratio, corr) on
    the held-out split. ratio = one-step MSE / no-op MSE (<1 means it beats
    predicting no motion); corr = correlation of the learned drift with the true
    increment (x'-x)/dt."""
    od, ad = O.shape[1], A.shape[1]
    ph = PortHamiltonianModel(od, ad, mode="phast", hidden=hidden)
    opt = th.optim.Adam(ph.parameters(), lr=1e-3)
    for _ in range(steps):
        idx = th.randint(0, n_train, (128,))
        ph.fit_step(O[idx], A[idx], NO[idx], DT[idx], opt)
    with th.no_grad():
        b = ph.drift(O[n_train:], A[n_train:])
        mse = ((O[n_train:] + b * DT[n_train:] - NO[n_train:]) ** 2).mean().item()
        noop = ((O[n_train:] - NO[n_train:]) ** 2).mean().item()
        fd = (NO[n_train:] - O[n_train:]) / (DT[n_train:] + 1e-12)
        corr, _ = _corr_slope(b, fd)
    return ph, mse / noop, corr


class TestModelBasedGenerator(unittest.TestCase):
    """Validates the model-based generator (Milestone M0) on cheetah-run."""

    def setUp(self):
        th.manual_seed(0)
        np.random.seed(0)
        # Small dt: MuJoCo's Euler integrator damps joint velocities implicitly, so
        # the realized increment (x'-x)/dt approaches the true continuous drift only
        # as dt -> 0 (this O(u) gap is exactly what the model-based generator removes).
        self.env = DMCContinuousEnv(
            domain_name="cheetah",
            task_name="run",
            time_sampling="uniform",
            dt=0.001,
            physics_dt=0.001,
            episode_duration=5.0,
        )
        self.obs_dim = int(self.env.observation_space.shape[0])
        self.act_dim = int(self.env.action_space.shape[0])

        self.model = ActorQCriticModel(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            q_net_arch=[64, 64],
            pi_net_arch=[64, 64],
            device="cpu",
        )
        self.ph = PortHamiltonianModel(
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
            mode="mujoco",
            drift_fn=self.env.dynamics_terms,
            device="cpu",
        )
        self.agent = CTSAC(
            env=self.env,
            model=self.model,
            device="cpu",
            learning_starts=10,
            batch_size=32,
            buffer_size=4000,
            num_expectation_samples=8,
            seed=0,
            use_model_based_q=True,
            dynamics_model=self.ph,
        )

    def _collect(self, n: int = 500):
        O, A, NO, DT = _collect_env(self.env, n)
        for i in range(n):
            self.agent.replay_buffer.add(
                obs=O[i : i + 1].numpy(),
                next_obs=NO[i : i + 1].numpy(),
                action=A[i : i + 1].numpy(),
                reward=np.zeros((1,), dtype=np.float32),
                done=np.zeros((1,), dtype=np.float32),
                t=np.zeros((1,), dtype=np.float32),
                next_t=DT[i].numpy(),
            )

    def test_dynamics_terms_shape_and_finite(self):
        obs = np.stack([self.env.observation_space.sample() for _ in range(5)])
        act = np.stack([self.env.action_space.sample() for _ in range(5)])
        b = self.env.dynamics_terms(obs, act)
        self.assertEqual(b.shape, (5, self.obs_dim))
        self.assertTrue(np.all(np.isfinite(b)))

    def test_drift_matches_simulator(self):
        """The analytic drift b(x,a) must match the simulator increment (x'-x)/dt."""
        self._collect(500)
        batch = self.agent.replay_buffer.sample(256)
        b = self.agent.dynamics_model.drift(batch.observations, batch.actions)
        fd = (batch.next_observations - batch.observations) / (batch.dt + 1e-12)
        corr, slope = _corr_slope(b, fd)
        rel = ((b - fd).norm() / (fd.norm() + 1e-8)).item()
        print(f"\n[drift] corr={corr:.3f} slope={slope:.3f} rel_err={rel:.3f}")
        self.assertGreater(corr, 0.97)
        self.assertGreater(slope, 0.9)
        self.assertLess(slope, 1.12)

    def test_generator_units_linear_value(self):
        """With an exact linear value V(x)=w.x (zero curvature), the model-based
        generator dt_default*(b.grad V) - beta*V must match the finite-difference
        fraction. This isolates the time-unit handling from network nonlinearity.
        """
        self._collect(500)
        batch = self.agent.replay_buffer.sample(256)
        th.manual_seed(1)
        w = th.randn(self.obs_dim)

        with th.no_grad():
            v_cur = (batch.observations * w).sum(-1, keepdim=True)
            v_next = (batch.next_observations * w).sum(-1, keepdim=True)
            dt = batch.dt * self.agent.time_rescale
            gamma_dt = th.exp(-self.agent.beta * dt)
            fraction = (gamma_dt * v_next - v_cur) / (dt + 1e-8)

            b = self.agent.dynamics_model.drift(batch.observations, batch.actions)
            grad_v = w.unsqueeze(0).expand_as(b)
            lf = (
                self.agent.dt_default * (b * grad_v).sum(-1, keepdim=True)
                - self.agent.beta * v_cur
            )

        corr, slope = _corr_slope(lf, fraction)
        rel = ((lf - fraction).abs().mean() / (fraction.abs().mean() + 1e-6)).item()
        print(f"\n[generator] corr={corr:.3f} slope={slope:.3f} rel_err={rel:.3f}")
        self.assertGreater(corr, 0.97)
        self.assertGreater(slope, 0.9)
        self.assertLess(slope, 1.12)

    def test_model_based_target_finite(self):
        """The full model-based target path produces finite values of the right shape."""
        self._collect(200)
        batch = self.agent.replay_buffer.sample(32)
        alpha = th.tensor(float(self.agent.alpha))
        target = self.agent._model_based_target(
            batch.observations,
            batch.actions,
            batch.next_observations,
            batch.rewards,
            batch.dones,
            batch.dt,
            alpha,
        )
        self.assertEqual(tuple(target.shape), (32, 1))
        self.assertTrue(th.all(th.isfinite(target)))
        self.assertFalse(target.requires_grad)

    def test_gated_blend_target_finite(self):
        """The per-component |b*dt|-gated blend produces finite, right-shaped targets,
        and recovers the pure generator as the gate scale -> infinity."""
        self._collect(200)
        batch = self.agent.replay_buffer.sample(32)
        alpha = th.tensor(float(self.agent.alpha))

        args = (
            batch.observations,
            batch.actions,
            batch.next_observations,
            batch.rewards,
            batch.dones,
            batch.dt,
            alpha,
        )
        # Seed before each call: _value_expectation samples the policy, so V/grad V
        # are stochastic; fixing the seed isolates the drift as the only difference.
        th.manual_seed(0)
        self.agent.generator_gate_scale = 0.0
        pure = self.agent._model_based_target(*args)
        th.manual_seed(0)
        self.agent.generator_gate_scale = 0.3
        blended = self.agent._model_based_target(*args)
        th.manual_seed(0)
        self.agent.generator_gate_scale = 1e9  # gate -> 1 everywhere
        recovered = self.agent._model_based_target(*args)
        self.agent.generator_gate_scale = 0.0

        self.assertEqual(tuple(blended.shape), (32, 1))
        self.assertTrue(th.all(th.isfinite(blended)))
        self.assertFalse(blended.requires_grad)
        # large gate scale -> trust analytic drift everywhere -> pure generator
        self.assertLess((recovered - pure).abs().max().item(), 1e-3)

    def test_learn_model_based_runs(self):
        try:
            self.agent.learn(total_timesteps=40)
        except Exception as e:
            self.fail(f"model-based CT-SAC learn raised an exception: {e}")


class TestPhastLearnedDrift(unittest.TestCase):
    """Smoke test for the learned (phast-mode) port-Hamiltonian drift."""

    def test_shapes_and_fit_reduces_loss(self):
        th.manual_seed(0)
        obs_dim, act_dim = 17, 6
        ph = PortHamiltonianModel(
            obs_dim=obs_dim, action_dim=act_dim, mode="phast", hidden=(32, 32)
        )
        x = th.randn(8, obs_dim)
        a = th.randn(8, act_dim)
        b = ph.drift(x, a)
        self.assertEqual(tuple(b.shape), (8, obs_dim))

        opt = th.optim.Adam(ph.parameters(), lr=1e-3)
        xp = x + 0.01 * th.randn(8, obs_dim)
        first = ph.fit_step(x, a, xp, 0.01, opt)
        last = first
        for _ in range(60):
            last = ph.fit_step(x, a, xp, 0.01, opt)
        self.assertLess(last, first)


class TestPhastLearnedDynamics(unittest.TestCase):
    """Milestone M1: the learned (phast) port-Hamiltonian is trained from replay
    transitions and integrated into CT-SAC (warmup, then it takes over)."""

    def test_learns_smooth_dynamics_cartpole(self):
        """On a smooth (contact-free) system the learned drift partially fits:
        it beats the no-op baseline and tracks the true increment direction."""
        th.manual_seed(0)
        np.random.seed(0)
        env = DMCContinuousEnv(
            "cartpole", "swingup", time_sampling="uniform",
            dt=0.01, physics_dt=0.01, episode_duration=20.0, seed=0,
        )
        O, A, NO, DT = _collect_env(env, 2500)
        _, ratio, corr = _train_eval_phast(O, A, NO, DT, 2000, steps=3000)
        print(f"\n[phast-cartpole] ratio={ratio:.3f} drift_corr={corr:.3f}")
        self.assertLess(ratio, 0.85)   # beats no-op
        self.assertGreater(corr, 0.45)  # tracks the dynamics direction

    def test_partially_fits_contact_rich_cheetah(self):
        """Cheetah has ground contacts -> stiff, near-discontinuous accelerations
        that are much harder to regress than smooth systems. The learned drift
        still beats the no-op baseline. Richer fit (on-policy data, contact
        features, state-dependent damping, Strang substeps) is future work."""
        th.manual_seed(0)
        np.random.seed(0)
        env = DMCContinuousEnv(
            "cheetah", "run", time_sampling="uniform", dt=0.01, episode_duration=20.0, seed=0
        )
        O, A, NO, DT = _collect_env(env, 1500)
        _, ratio, corr = _train_eval_phast(O, A, NO, DT, 1200, steps=1000)
        print(f"\n[phast-cheetah] ratio={ratio:.3f} drift_corr={corr:.3f}")
        self.assertLess(ratio, 0.97)

    def test_ct_sac_phast_trains_dynamics_and_runs(self):
        """CT-SAC fits the learned dynamics online and runs through warmup."""
        th.manual_seed(0)
        np.random.seed(0)
        env = DMCContinuousEnv(
            "cheetah", "run", time_sampling="uniform", dt=0.01, episode_duration=20.0
        )
        od = int(env.observation_space.shape[0])
        ad = int(env.action_space.shape[0])
        model = ActorQCriticModel(
            observation_space=env.observation_space,
            action_space=env.action_space,
            q_net_arch=[32],
            pi_net_arch=[32],
            device="cpu",
        )
        ph = PortHamiltonianModel(od, ad, mode="phast", hidden=(32, 32))
        agent = CTSAC(
            env=env,
            model=model,
            device="cpu",
            learning_starts=10,
            batch_size=16,
            buffer_size=500,
            num_expectation_samples=2,
            seed=0,
            use_model_based_q=True,
            dynamics_model=ph,
            dynamics_warmup=5,
        )
        self.assertTrue(agent._train_dynamics)
        agent.learn(total_timesteps=80)
        self.assertGreater(agent._dynamics_updates, 5)


class TestValueHead(unittest.TestCase):
    """Explicit scalar V(s) head: clean, sample-free V and grad V for the generator."""

    def _make(self, v_net_arch):
        th.manual_seed(0)
        np.random.seed(0)
        env = DMCContinuousEnv(
            "cheetah", "run", time_sampling="uniform", dt=0.001, physics_dt=0.001,
            episode_duration=5.0,
        )
        od = int(env.observation_space.shape[0]); ad = int(env.action_space.shape[0])
        model = ActorQCriticModel(
            observation_space=env.observation_space,
            action_space=env.action_space,
            q_net_arch=[64, 64], pi_net_arch=[64, 64], v_net_arch=v_net_arch,
            device="cpu",
        )
        ph = PortHamiltonianModel(od, ad, mode="mujoco", drift_fn=env.dynamics_terms, device="cpu")
        agent = CTSAC(
            env=env, model=model, device="cpu", learning_starts=10, batch_size=32,
            buffer_size=4000, num_expectation_samples=8, seed=0,
            use_model_based_q=True, dynamics_model=ph,
        )
        O, A, NO, DT = _collect_env(env, 300)
        for i in range(300):
            agent.replay_buffer.add(
                obs=O[i:i+1].numpy(), next_obs=NO[i:i+1].numpy(), action=A[i:i+1].numpy(),
                reward=np.zeros((1,), np.float32), done=np.zeros((1,), np.float32),
                t=np.zeros((1,), np.float32), next_t=DT[i].numpy(),
            )
        return env, model, agent

    def test_head_off_by_default(self):
        """No v_net_arch -> no head, empty value params, generator unchanged."""
        _, model, agent = self._make(None)
        self.assertFalse(model.has_v_head)
        self.assertEqual(list(model.value_parameters), [])
        self.assertFalse(agent.use_value_head)
        self.assertIsNone(agent.value_optimizer)

    def test_head_built_and_wired(self):
        _, model, agent = self._make([64, 64])
        self.assertTrue(model.has_v_head)
        self.assertGreater(len(list(model.value_parameters)), 0)
        self.assertTrue(agent.use_value_head)
        self.assertIsNotNone(agent.value_optimizer)
        obs = th.as_tensor(
            np.stack([model.observation_space.sample() for _ in range(5)]), dtype=th.float32
        )
        self.assertEqual(tuple(model.value(obs).shape), (5, 1))
        self.assertEqual(tuple(model.target_value(obs).shape), (5, 1))

    def test_generator_target_is_sample_free(self):
        """With the V-head, the model-based target has no action-sampling noise:
        identical across different RNG seeds (vs the sampled E_a[Q~] path)."""
        _, _, agent = self._make([64, 64])
        b = agent.replay_buffer.sample(32); alpha = th.tensor(float(agent.alpha))
        args = (b.observations, b.actions, b.next_observations, b.rewards, b.dones, b.dt, alpha)
        th.manual_seed(1); t1 = agent._model_based_target(*args)
        th.manual_seed(2); t2 = agent._model_based_target(*args)
        self.assertTrue(th.all(th.isfinite(t1)))
        self.assertLess((t1 - t2).abs().max().item(), 1e-6)   # sample-free
        # contrast: the sampled path is stochastic across seeds
        agent.use_value_head = False
        th.manual_seed(1); s1 = agent._model_based_target(*args)
        th.manual_seed(2); s2 = agent._model_based_target(*args)
        self.assertGreater((s1 - s2).abs().max().item(), 1e-4)

    def test_value_head_trains(self):
        """A few direct train steps move the V-head params and log a finite loss."""
        _, model, agent = self._make([64, 64])
        before = [p.detach().clone() for p in model.value_parameters]
        agent.train(gradient_steps=5, batch_size=32)   # direct: no logger dump/clear
        after = list(model.value_parameters)
        changed = any((a - b).abs().max().item() > 0 for a, b in zip(after, before))
        self.assertTrue(changed, "value head parameters did not update")
        self.assertTrue(all(th.all(th.isfinite(p)) for p in after))


if __name__ == "__main__":
    unittest.main()
