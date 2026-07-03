import unittest

import numpy as np
import torch as th
from gymnasium import spaces

from environment.dmc import DMCContinuousEnv
from algorithms.ct_sac import CTSAC
from common.buffers import ReplayBuffer
from models.actor_q_critic import ActorQCriticModel
from models.port_hamiltonian import DOFLayout, PortHamiltonianModel


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

    def test_value_warmup_gates_head_read(self):
        """During warmup the generator uses the sampled path (stochastic); once
        the head has enough updates it switches to the sample-free head."""
        _, _, agent = self._make([64, 64])
        agent.value_warmup = 10
        agent._value_updates = 0
        self.assertFalse(agent._value_head_ready)
        b = agent.replay_buffer.sample(32); alpha = th.tensor(float(agent.alpha))
        args = (b.observations, b.actions, b.next_observations, b.rewards, b.dones, b.dt, alpha)
        th.manual_seed(1); w1 = agent._model_based_target(*args)
        th.manual_seed(2); w2 = agent._model_based_target(*args)
        self.assertGreater((w1 - w2).abs().max().item(), 1e-4)  # sampled fallback
        agent._value_updates = 10  # warmup reached
        self.assertTrue(agent._value_head_ready)
        th.manual_seed(1); r1 = agent._model_based_target(*args)
        th.manual_seed(2); r2 = agent._model_based_target(*args)
        self.assertLess((r1 - r2).abs().max().item(), 1e-6)     # head, sample-free

    def test_value_head_trains(self):
        """A few direct train steps move the V-head params and log a finite loss."""
        _, model, agent = self._make([64, 64])
        before = [p.detach().clone() for p in model.value_parameters]
        agent.train(gradient_steps=5, batch_size=32)   # direct: no logger dump/clear
        after = list(model.value_parameters)
        changed = any((a - b).abs().max().item() > 0 for a, b in zip(after, before))
        self.assertTrue(changed, "value head parameters did not update")
        self.assertTrue(all(th.all(th.isfinite(p)) for p in after))


class TestSubstepQuadrature(unittest.TestCase):
    """Sub-step quadrature generator target (generator_substeps = m), on cartpole
    (a smooth system where the learned drift fits well)."""

    def _cartpole_agent(self, substeps):
        th.manual_seed(0)
        np.random.seed(0)
        env = DMCContinuousEnv(
            "cartpole", "swingup", time_sampling="uniform",
            dt=0.01, physics_dt=0.01, episode_duration=10.0, seed=0,
        )
        od = int(env.observation_space.shape[0]); ad = int(env.action_space.shape[0])
        model = ActorQCriticModel(
            observation_space=env.observation_space, action_space=env.action_space,
            q_net_arch=[64, 64], pi_net_arch=[64, 64], v_net_arch=[64, 64], device="cpu",
        )
        ph = PortHamiltonianModel(od, ad, mode="phast", hidden=(64, 64), device="cpu")
        agent = CTSAC(
            env=env, model=model, device="cpu", learning_starts=10, batch_size=32,
            buffer_size=4000, num_expectation_samples=8, seed=0,
            use_model_based_q=True, dynamics_model=ph, dynamics_source="phast",
            dynamics_warmup=2, generator_substeps=substeps,
        )
        O, A, NO, DT = _collect_env(env, 300)
        for i in range(300):
            agent.replay_buffer.add(
                obs=O[i:i+1].numpy(), next_obs=NO[i:i+1].numpy(), action=A[i:i+1].numpy(),
                reward=np.zeros((1,), np.float32), done=np.zeros((1,), np.float32),
                t=np.zeros((1,), np.float32), next_t=DT[i].numpy(),
            )
        return env, model, ph, agent

    def test_m1_matches_finite_difference_over_states(self):
        """m=1 quadrature equals V(x + b*dt_default) - V(x) - beta*V(x)."""
        _, model, ph, agent = self._cartpole_agent(substeps=1)
        b = agent.replay_buffer.sample(16); alpha = th.tensor(float(agent.alpha))
        tq = agent._substep_quadrature_target(
            b.observations, b.actions, b.rewards, b.dones, alpha
        )
        with th.no_grad():
            drift = th.as_tensor(ph.drift(b.observations, b.actions))
            x_hat = b.observations + drift * agent.dt_default
            V_cur = model.target_value(b.observations)
            V_next = model.target_value(x_hat)
            manual = b.rewards + (1 - b.dones) * (V_cur + (V_next - V_cur) - agent.beta * V_cur)
        self.assertLess((tq - manual).abs().max().item(), 1e-6)

    def test_target_finite_and_sample_free(self):
        """m=4 quadrature target is finite, right-shaped, and sample-free (V-head)."""
        _, _, _, agent = self._cartpole_agent(substeps=4)
        b = agent.replay_buffer.sample(16); alpha = th.tensor(float(agent.alpha))
        args = (b.observations, b.actions, b.rewards, b.dones, alpha)
        th.manual_seed(1); t1 = agent._substep_quadrature_target(*args)
        th.manual_seed(2); t2 = agent._substep_quadrature_target(*args)
        self.assertEqual(tuple(t1.shape), (16, 1))
        self.assertTrue(th.all(th.isfinite(t1)))
        self.assertFalse(t1.requires_grad)
        self.assertLess((t1 - t2).abs().max().item(), 1e-6)

    def test_substeps_reduce_integration_error(self):
        """More sub-steps roll the model more accurately, so the target converges:
        target(m=8) is closer to a fine reference target(m=128) than target(m=1)."""
        env = DMCContinuousEnv(
            "cartpole", "swingup", time_sampling="uniform",
            dt=0.01, physics_dt=0.01, episode_duration=10.0, seed=0,
        )
        od = int(env.observation_space.shape[0]); ad = int(env.action_space.shape[0])
        th.manual_seed(0)

        class MockDynamics(th.nn.Module):
            """Linear drift b(x) = x A^T scaled so |b*dt_default| ~ O(1)."""
            def __init__(self, d):
                super().__init__()
                self.register_buffer("A", 60.0 * th.randn(d, d) / d**0.5)
            def drift(self, obs, action, prev_obs=None):
                x = th.as_tensor(obs, dtype=th.float32)
                return x @ self.A.t()
            def diffusion(self, obs):
                return None
            def to(self, device):
                return self

        model = ActorQCriticModel(
            observation_space=env.observation_space, action_space=env.action_space,
            q_net_arch=[64, 64], pi_net_arch=[64, 64], v_net_arch=[64, 64], device="cpu",
        )
        agent = CTSAC(
            env=env, model=model, device="cpu", learning_starts=10, batch_size=32,
            buffer_size=2000, seed=0, use_model_based_q=True,
            dynamics_model=MockDynamics(od), dynamics_source="phast", generator_substeps=1,
        )
        O, A, NO, DT = _collect_env(env, 64)
        obs, act = O[:32], A[:32]
        r = th.zeros(32, 1); done = th.zeros(32, 1); alpha = th.tensor(float(agent.alpha))

        def target(m):
            agent.generator_substeps = m
            return agent._substep_quadrature_target(obs, act, r, done, alpha)

        ref = target(128)
        err1 = (target(1) - ref).abs().mean().item()
        err8 = (target(8) - ref).abs().mean().item()
        self.assertLess(err8, err1)  # more sub-steps -> closer to the converged target


class TestReplayBufferPrevObs(unittest.TestCase):
    """The replay buffer reconstructs the previous observation (for the contact
    signal dv = obs - prev_obs), zeroing the jump across episode resets."""

    def test_prev_observation_reconstruction(self):
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        buf = ReplayBuffer(10, obs_space, act_space, device="cpu", n_envs=1)
        dones = [0, 0, 1, 0, 0]  # episode ends at t=2 -> t=3 is a fresh start
        for i in range(5):
            buf.add(
                obs=np.full((1, 3), i, np.float32),
                action=np.zeros((1, 2), np.float32),
                reward=np.zeros((1,), np.float32),
                done=np.array([dones[i]], np.float32),
                next_obs=np.full((1, 3), i + 1, np.float32),
                t=np.zeros((1,), np.float32),
                next_t=np.full((1,), 0.01, np.float32),
            )
        np.random.seed(0)
        batch = buf._get_samples(np.array([0, 1, 3, 4]))
        prev0 = batch.prev_observations.cpu().numpy()[:, 0]
        # idx0: no predecessor -> prev=obs(0)=0; idx1: obs(0)=0;
        # idx3: predecessor ended an episode -> prev=obs(3)=3; idx4: obs(3)=3
        self.assertEqual(list(prev0), [0.0, 0.0, 3.0, 3.0])


class TestContactAwareDynamics(unittest.TestCase):
    """Contact-aware damping R(x, dv): infer contact from the incoming velocity
    jump (#2) and route it through the PSD dissipation (#4). No MuJoCo reads."""

    def _model(self, contact_aware):
        th.manual_seed(0)
        return PortHamiltonianModel(
            6, 2, mode="phast", hidden=(32, 32), contact_aware=contact_aware
        )

    def test_off_has_no_beta_and_ignores_prev(self):
        m = self._model(False)
        self.assertFalse(hasattr(m, "beta_net"))
        x = th.randn(4, 6); a = th.randn(4, 2)
        with th.no_grad():
            b0 = m.drift(x, a)
            b1 = m.drift(x, a, prev_obs=x - 0.5)  # constant R ignores prev
        self.assertTrue(th.allclose(b0, b1))

    def test_on_builds_and_uses_prev(self):
        m = self._model(True)
        self.assertTrue(hasattr(m, "beta_net"))
        x = th.randn(4, 6); a = th.randn(4, 2)
        with th.no_grad():
            b_none = m.drift(x, a)                 # dx = 0
            b_prev = m.drift(x, a, prev_obs=x - 0.5)  # dx != 0 -> different damping
        self.assertEqual(tuple(b_prev.shape), (4, 6))
        self.assertTrue(th.all(th.isfinite(b_prev)))
        self.assertFalse(th.allclose(b_none, b_prev))

    def test_R_batch_is_psd(self):
        m = self._model(True)
        x = th.randn(5, 6); dx = th.randn(5, 6)
        with th.no_grad():
            R = m._R_batch(x, dx)
        self.assertEqual(tuple(R.shape), (5, 6, 6))
        self.assertLess((R - R.transpose(-1, -2)).abs().max().item(), 1e-5)  # symmetric
        self.assertGreaterEqual(th.linalg.eigvalsh(R).min().item(), -1e-4)   # PSD

    def test_fit_step_with_prev_reduces_loss(self):
        m = self._model(True)
        opt = th.optim.Adam(m.parameters(), lr=1e-3)
        x = th.randn(16, 6); a = th.randn(16, 2)
        prev = x - 0.1 * th.randn(16, 6); xp = x + 0.05 * th.randn(16, 6)
        first = m.fit_step(x, a, xp, 0.01, opt, prev_obs=prev)
        last = first
        for _ in range(60):
            last = m.fit_step(x, a, xp, 0.01, opt, prev_obs=prev)
        self.assertLess(last, first)


class TestContactAwareEndToEnd(unittest.TestCase):
    """Full path: buffer prev_obs -> fit_step / model-based target -> contact drift."""

    def test_ct_sac_contact_aware_runs(self):
        th.manual_seed(0); np.random.seed(0)
        env = DMCContinuousEnv(
            "cheetah", "run", time_sampling="uniform", dt=0.01, episode_duration=20.0
        )
        od = int(env.observation_space.shape[0]); ad = int(env.action_space.shape[0])
        model = ActorQCriticModel(
            observation_space=env.observation_space, action_space=env.action_space,
            q_net_arch=[32], pi_net_arch=[32], device="cpu",
        )
        ph = PortHamiltonianModel(od, ad, mode="phast", hidden=(32, 32), contact_aware=True)
        agent = CTSAC(
            env=env, model=model, device="cpu", learning_starts=10, batch_size=16,
            buffer_size=500, num_expectation_samples=2, seed=0,
            use_model_based_q=True, dynamics_model=ph, dynamics_warmup=5,
        )
        agent.learn(total_timesteps=80)
        self.assertGreater(agent._dynamics_updates, 5)


class TestDOFLayout(unittest.TestCase):
    """The DOF layout keeps the structured model domain-agnostic."""

    def test_cheetah_defaults(self):
        lay = DOFLayout.cheetah()
        self.assertEqual((lay.npos, lay.nv), (8, 9))
        self.assertEqual(lay.cyclic_cfg, (0,))
        self.assertEqual(lay.obs_pos_to_cfg, tuple(range(1, 9)))

    def test_validation_rejects_bad_layout(self):
        # an observed position mapping onto a cyclic DOF is inconsistent
        with self.assertRaises(AssertionError):
            DOFLayout(obs_dim=17, pos_slice=(0, 8), vel_slice=(8, 17),
                      cyclic_cfg=(0,), obs_pos_to_cfg=tuple(range(0, 8)))
        # slices must cover obs_dim
        with self.assertRaises(AssertionError):
            DOFLayout(obs_dim=17, pos_slice=(0, 7), vel_slice=(8, 17),
                      cyclic_cfg=(0,), obs_pos_to_cfg=tuple(range(1, 8)))


class TestStructuredDynamics(unittest.TestCase):
    """Structured port-Hamiltonian (DeLaN core + contact-gated D on momentum)."""

    def _model(self, contact_aware=False, layout=None):
        th.manual_seed(0)
        return PortHamiltonianModel(
            17, 6, mode="structured", structured_hidden=(32, 32),
            contact_aware=contact_aware, dof_layout=layout,
        )

    def test_drift_shape_finite_and_exact_position_block(self):
        m = self._model()
        x = th.randn(5, 17); a = th.randn(5, 6)
        b = m.drift(x, a)
        self.assertEqual(tuple(b.shape), (5, 17))
        self.assertTrue(th.all(th.isfinite(b)))
        # position-drift is exactly the velocity of the mapped config DOFs (qd[1:])
        self.assertTrue(th.allclose(b[:, :8], x[:, 9:17], atol=1e-5))

    def test_mass_matrix_is_spd(self):
        m = self._model()
        M = th.vmap(m._mass)(th.randn(6, 8))
        self.assertTrue(th.allclose(M, M.transpose(-1, -2), atol=1e-5))
        self.assertGreater(th.linalg.eigvalsh(M).min().item(), 0.0)

    def test_contact_gate_uses_prev_and_damping_psd(self):
        m = self._model(contact_aware=True)
        self.assertTrue(hasattr(m, "beta_net"))
        x = th.randn(5, 17); a = th.randn(5, 6)
        with th.no_grad():
            b0 = m.drift(x, a)                        # dv = 0
            b1 = m.drift(x, a, prev_obs=x - 0.5)      # dv != 0 -> different damping
            D = m._damping(x[:, :8], th.randn(5, 9))  # (B, nv, nv)
        self.assertFalse(th.allclose(b0, b1, atol=1e-6))
        self.assertLess((D - D.transpose(-1, -2)).abs().max().item(), 1e-5)     # symmetric
        self.assertGreaterEqual(th.linalg.eigvalsh(D).min().item(), -1e-4)      # PSD

    def test_runs_under_no_grad(self):
        # the CT-SAC quadrature target path calls drift under th.no_grad()
        m = self._model(contact_aware=True)
        with th.no_grad():
            b = m.drift(th.randn(4, 17), th.randn(4, 6), prev_obs=th.randn(4, 17))
        self.assertTrue(th.all(th.isfinite(b)))

    def test_fit_step_reduces_loss(self):
        m = self._model(contact_aware=True)
        opt = th.optim.Adam(m.parameters(), lr=1e-3)
        x = th.randn(16, 17); a = th.randn(16, 6)
        prev = x - 0.1 * th.randn(16, 17); xp = x + 0.05 * th.randn(16, 17)
        first = m.fit_step(x, a, xp, 0.01, opt, prev_obs=prev)
        last = first
        for _ in range(60):
            last = m.fit_step(x, a, xp, 0.01, opt, prev_obs=prev)
        self.assertLess(last, first)

    def test_energy_balance_matches_damping_power(self):
        """Independent ground truth for the accel/Coriolis sign: with zero action
        the energy dissipates exactly at the damping rate, dE/dt = -qd^T D qd (the
        Coriolis does no net work). A sign flip in the Coriolis or pdot breaks this
        identity, which the shape/finite tests cannot catch."""
        m = self._model(contact_aware=False)
        th.manual_seed(1)
        x = 0.5 * th.randn(6, 17)
        a = th.zeros(6, 6)
        xin = x.clone().requires_grad_(True)
        M = th.vmap(m._mass)(xin[:, :8])
        E = m._potential(xin[:, :8]) + 0.5 * th.einsum("na,nab,nb->n", xin[:, 8:], M, xin[:, 8:])
        (gE,) = th.autograd.grad(E.sum(), xin)          # dE/dobs
        with th.no_grad():
            dEdt = (gE * m.drift(x, a)).sum(-1)         # dE/dt along the drift
            D = m._damping(x[:, :8], th.zeros(6, 9))
            expected = -th.einsum("na,nab,nb->n", x[:, 8:], D, x[:, 8:])
        self.assertLess((dEdt - expected).abs().max().item(), 1e-2)

    def test_damping_contact_term_psd_at_scale(self):
        """The contact damping term must stay PSD even when scaled to dominate the
        diagonal base -- a dropped softplus on beta would make it indefinite, which
        the base otherwise masks."""
        m = self._model(contact_aware=True)
        th.manual_seed(2)
        with th.no_grad():
            m._damp_dirs.mul_(25.0)                     # let the contact term dominate
            for _ in range(5):
                D = m._damping(th.randn(8, 8), 3.0 * th.randn(8, 9))
                self.assertLess((D - D.transpose(-1, -2)).abs().max().item(), 1e-4)
                self.assertGreaterEqual(th.linalg.eigvalsh(D).min().item(), -1e-3)

    def test_custom_layout_sparse_actuation(self):
        # a small 2-pos / 3-vel system with one cyclic DOF and sparse actuation
        lay = DOFLayout(obs_dim=5, pos_slice=(0, 2), vel_slice=(2, 5),
                        cyclic_cfg=(0,), obs_pos_to_cfg=(1, 2), act_to_cfg=(1, 2))
        m = PortHamiltonianModel(5, 2, mode="structured", structured_hidden=(16,),
                                 dof_layout=lay)
        b = m.drift(th.randn(4, 5), th.randn(4, 2))
        self.assertEqual(tuple(b.shape), (4, 5))
        self.assertTrue(th.all(th.isfinite(b)))


class TestRolloutFit(unittest.TestCase):
    """Multi-step rollout fit (fit_step_rollout): backprop through the model's
    own Euler roll, per-step masked regression onto the stored window."""

    def _structured(self):
        th.manual_seed(0)
        return PortHamiltonianModel(
            17, 6, mode="structured", structured_hidden=(32, 32), contact_aware=True
        )

    def test_h1_full_mask_matches_fit_step(self):
        """Horizon 1 with a full mask is exactly the one-step fit."""
        import copy

        m = self._structured()
        state = copy.deepcopy(m.state_dict())
        th.manual_seed(1)
        x = th.randn(8, 17); a = th.randn(8, 6)
        xp = x + 0.05 * th.randn(8, 17); prev = x - 0.1 * th.randn(8, 17)

        opt = th.optim.Adam(m.parameters(), lr=1e-3)
        one = m.fit_step(x, a, xp, 0.01, opt, prev_obs=prev)

        m.load_state_dict(state)
        opt = th.optim.Adam(m.parameters(), lr=1e-3)
        roll = m.fit_step_rollout(
            x, a.unsqueeze(1), xp.unsqueeze(1), th.full((8, 1, 1), 0.01),
            th.ones(8, 1, 1), opt, prev_obs=prev,
        )
        self.assertLess(abs(one - roll), 1e-6)

    def test_mask_zeroes_invalid_tail(self):
        """A window whose tail is masked fits exactly like the truncated window."""
        import copy

        m = self._structured()
        state = copy.deepcopy(m.state_dict())
        th.manual_seed(2)
        x = th.randn(8, 17)
        A = th.randn(8, 3, 6); XP = th.randn(8, 3, 17)
        dt = th.full((8, 3, 1), 0.01)
        mask = th.tensor([1.0, 0.0, 0.0]).view(1, 3, 1).expand(8, 3, 1)

        opt = th.optim.Adam(m.parameters(), lr=1e-3)
        l_masked = m.fit_step_rollout(x, A, XP, dt, mask, opt)

        m.load_state_dict(state)
        opt = th.optim.Adam(m.parameters(), lr=1e-3)
        l_short = m.fit_step_rollout(
            x, A[:, :1], XP[:, :1], dt[:, :1], th.ones(8, 1, 1), opt
        )
        self.assertLess(abs(l_masked - l_short), 1e-6)

    def test_rollout_fit_reduces_loss(self):
        """Repeated rollout fits on consistent 3-step windows reduce the loss
        (BPTT through drift, mass Jacobian, and contact gate), for both learned
        modes."""
        th.manual_seed(3)
        for mode, kwargs in [
            ("structured", dict(structured_hidden=(32, 32), contact_aware=True)),
            ("phast", dict(hidden=(32, 32))),
        ]:
            m = PortHamiltonianModel(17, 6, mode=mode, **kwargs)
            # ground-truth linear dynamics rolled 3 steps from random starts
            Amat = 0.5 * th.randn(17, 17) / 17**0.5
            Bmat = 0.5 * th.randn(17, 6) / 6**0.5
            x0 = th.randn(16, 17); acts = th.randn(16, 3, 6)
            xs, xk = [], x0
            for k in range(3):
                xk = xk + (xk @ Amat.t() + acts[:, k] @ Bmat.t()) * 0.01
                xs.append(xk)
            targets = th.stack(xs, dim=1)  # (16, 3, 17)
            dt = th.full((16, 3, 1), 0.01); mask = th.ones(16, 3, 1)

            opt = th.optim.Adam(m.parameters(), lr=1e-3)
            first = m.fit_step_rollout(x0, acts, targets, dt, mask, opt)
            last = first
            for _ in range(60):
                last = m.fit_step_rollout(x0, acts, targets, dt, mask, opt)
            self.assertLess(last, first, f"mode={mode}")

    def test_ct_sac_rollout_fit_runs(self):
        """End-to-end: dynamics_fit_horizon > 1 routes the dynamics update through
        sample_sequences + fit_step_rollout inside CT-SAC training."""
        th.manual_seed(0); np.random.seed(0)
        env = DMCContinuousEnv(
            "cheetah", "run", time_sampling="uniform", dt=0.01, episode_duration=20.0
        )
        od = int(env.observation_space.shape[0]); ad = int(env.action_space.shape[0])
        model = ActorQCriticModel(
            observation_space=env.observation_space, action_space=env.action_space,
            q_net_arch=[32], pi_net_arch=[32], device="cpu",
        )
        ph = PortHamiltonianModel(
            od, ad, mode="structured", structured_hidden=(32, 32), contact_aware=True
        )
        agent = CTSAC(
            env=env, model=model, device="cpu", learning_starts=10, batch_size=16,
            buffer_size=500, num_expectation_samples=2, seed=0,
            use_model_based_q=True, dynamics_model=ph, dynamics_warmup=5,
            dynamics_fit_horizon=3,
        )
        self.assertEqual(agent.dynamics_fit_horizon, 3)
        agent.learn(total_timesteps=80)
        self.assertGreater(agent._dynamics_updates, 5)


class TestNumericalStabilityGuards(unittest.TestCase):
    """The failure mode observed in the mbq_*_roll benchmark run: the BPTT
    rollout fit explodes at high on-policy velocities, NaNs the dynamics model,
    and the NaN propagates through the critic target into every network — the
    agent pins at 0 return and never recovers. These tests pin the guards that
    make that impossible."""

    def _windows(self, batch=8, horizon=3):
        th.manual_seed(0)
        x = th.randn(batch, 17)
        A = th.randn(batch, horizon, 6)
        XP = x.unsqueeze(1) + 0.05 * th.randn(batch, horizon, 17)
        dt = th.full((batch, horizon, 1), 0.01)
        mask = th.ones(batch, horizon, 1)
        return x, A, XP, dt, mask

    def test_explosive_rollout_cannot_nan_parameters(self):
        """A model whose drift is astronomically wrong overflows the rollout
        loss; the update must be skipped and every parameter stay finite."""
        th.manual_seed(0)
        m = PortHamiltonianModel(17, 6, mode="structured", structured_hidden=(32, 32))
        with th.no_grad():
            m.G_a.weight.mul_(1e20)  # drift ~ 1e20 -> loss overflows to inf
        before = {k: v.clone() for k, v in m.state_dict().items()}
        opt = th.optim.Adam(m.parameters(), lr=1e-3)
        x, A, XP, dt, mask = self._windows()
        for _ in range(3):
            loss = m.fit_step_rollout(x, A, XP, dt, mask, opt)
        self.assertFalse(np.isfinite(loss))  # reported faithfully
        for k, v in m.state_dict().items():
            self.assertTrue(th.all(th.isfinite(v)), k)
            self.assertTrue(th.equal(v, before[k]), f"{k} updated on inf loss")

    def test_huge_but_finite_loss_takes_clipped_step(self):
        """A large finite loss must still update (clipped), not be skipped."""
        th.manual_seed(0)
        m = PortHamiltonianModel(17, 6, mode="structured", structured_hidden=(32, 32))
        with th.no_grad():
            m.G_a.weight.mul_(1e3)
        before = {k: v.clone() for k, v in m.state_dict().items()}
        opt = th.optim.Adam(m.parameters(), lr=1e-3)
        x, A, XP, dt, mask = self._windows()
        loss = m.fit_step_rollout(x, A, XP, dt, mask, opt)
        self.assertTrue(np.isfinite(loss))
        changed = any(
            not th.equal(v, before[k]) for k, v in m.state_dict().items()
        )
        self.assertTrue(changed, "no parameter moved on a finite loss")
        for k, v in m.state_dict().items():
            self.assertTrue(th.all(th.isfinite(v)), k)

    def test_compounding_steps_are_clamped(self):
        """The k>=1 Euler increments saturate at the data-derived bound, so a
        bad drift cannot compound the rolled state to overflow."""
        th.manual_seed(0)
        m = PortHamiltonianModel(17, 6, mode="structured", structured_hidden=(32, 32))
        with th.no_grad():
            m.G_a.weight.mul_(1e4)
        x, A, XP, dt, mask = self._windows(horizon=4)
        # reproduce the roll's forward pass bound: 5 * max observed step + 1e-3
        obs_steps = XP - th.cat([x.unsqueeze(1), XP[:, :-1]], dim=1)
        limit = 5.0 * obs_steps.abs().amax(dim=1) + 1e-3
        # worst possible rolled state: |x_hat| <= |x| + |step0| + (H-1)*limit,
        # with step0 the unclamped first increment
        with th.no_grad():
            b0 = m.drift(x, A[:, 0])
        bound = x.abs() + (b0 * 0.01).abs() + 3 * limit
        # run the fit and check the loss stayed finite (no overflow), which
        # only the clamp guarantees at this drift scale
        opt = th.optim.Adam(m.parameters(), lr=1e-3)
        loss = m.fit_step_rollout(x, A, XP, dt, mask, opt)
        worst = float(bound.max())
        self.assertTrue(np.isfinite(loss))
        self.assertLess((worst ** 2), np.finfo(np.float32).max)

    def test_nonfinite_target_fails_fast(self):
        """A dynamics model emitting NaN drift must terminate the run loudly
        (no model-free fallback, no silent critic poisoning): train() raises
        before the NaN target reaches any network."""
        th.manual_seed(0); np.random.seed(0)
        env = DMCContinuousEnv(
            "cheetah", "run", time_sampling="uniform", dt=0.01, episode_duration=20.0
        )

        class NaNDynamics(th.nn.Module):
            def drift(self, obs, action, prev_obs=None):
                return th.full_like(th.as_tensor(obs, dtype=th.float32), float("nan"))
            def diffusion(self, obs):
                return None

        model = ActorQCriticModel(
            observation_space=env.observation_space, action_space=env.action_space,
            q_net_arch=[32], pi_net_arch=[32], device="cpu",
        )
        agent = CTSAC(
            env=env, model=model, device="cpu", learning_starts=10, batch_size=16,
            buffer_size=500, num_expectation_samples=2, seed=0,
            use_model_based_q=True, dynamics_model=NaNDynamics(),
        )
        O, A, NO, DT = _collect_env(env, 60)
        for i in range(60):
            agent.replay_buffer.add(
                obs=O[i:i+1].numpy(), next_obs=NO[i:i+1].numpy(), action=A[i:i+1].numpy(),
                reward=np.zeros((1,), np.float32), done=np.zeros((1,), np.float32),
                t=np.zeros((1,), np.float32), next_t=DT[i].numpy(),
            )
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            agent.train(gradient_steps=1, batch_size=16)
        # the raise happened before any backward pass: every network is intact
        for p in agent.model.critic_parameters:
            self.assertTrue(th.all(th.isfinite(p)))
        for p in agent.model.actor.parameters():
            self.assertTrue(th.all(th.isfinite(p)))


class TestStructuredEndToEnd(unittest.TestCase):
    """Full path: buffer prev_obs -> structured drift -> model-based target / fit."""

    def test_ct_sac_structured_contact_runs(self):
        th.manual_seed(0); np.random.seed(0)
        env = DMCContinuousEnv(
            "cheetah", "run", time_sampling="uniform", dt=0.01, episode_duration=20.0
        )
        od = int(env.observation_space.shape[0]); ad = int(env.action_space.shape[0])
        model = ActorQCriticModel(
            observation_space=env.observation_space, action_space=env.action_space,
            q_net_arch=[32], pi_net_arch=[32], device="cpu",
        )
        ph = PortHamiltonianModel(
            od, ad, mode="structured", structured_hidden=(32, 32), contact_aware=True
        )
        agent = CTSAC(
            env=env, model=model, device="cpu", learning_starts=10, batch_size=16,
            buffer_size=500, num_expectation_samples=2, seed=0,
            use_model_based_q=True, dynamics_model=ph, dynamics_warmup=5,
        )
        agent.learn(total_timesteps=80)
        self.assertGreater(agent._dynamics_updates, 5)


if __name__ == "__main__":
    unittest.main()
