import unittest
from types import SimpleNamespace

import numpy as np
import torch as th

from benchmarks.run_ct_rl import (
    _pop_structured_model_kwargs,
    _select_structured_dof_layout,
)
from algorithms.ct_sac import CTSAC
from common.utils import load_ct_hyperparams_from_table
from evaluations.hamiltonian_recovery import (
    _select_recovery_dof_layout,
    energy_balance_report,
    learned_terms,
)
from models.port_hamiltonian import DOFLayout, PortHamiltonianModel, integrate_drift
from models.actor_q_critic import ActorQCriticModel

try:
    from environment import DMCContinuousEnv

    HAVE_DMC = True
except Exception:
    HAVE_DMC = False


class _Wrapper:
    def __init__(self, env):
        self.env = env


class _VecWrapper:
    def __init__(self, env):
        self.envs = [env]


class TestCartpoleLayoutSelection(unittest.TestCase):
    def test_runner_selects_cartpole_through_vec_and_wrapper_layers(self):
        calls = []

        class LayoutStub:
            @staticmethod
            def cartpole():
                calls.append(("cartpole", None))
                return "cartpole-layout"

            @staticmethod
            def raw_state(nv):
                calls.append(("raw", nv))
                return f"raw-{nv}"

        raw = SimpleNamespace(
            raw_state_obs=True, domain_name="cartpole"
        )
        selected = _select_structured_dof_layout(
            _VecWrapper(_Wrapper(raw)), 4, LayoutStub
        )
        self.assertEqual(selected, "cartpole-layout")
        self.assertEqual(calls, [("cartpole", None)])

    def test_runner_keeps_generic_raw_and_nonraw_fallbacks(self):
        class LayoutStub:
            @staticmethod
            def cartpole():
                raise AssertionError("cartpole layout should not be selected")

            @staticmethod
            def raw_state(nv):
                return ("raw", nv)

        acrobot = SimpleNamespace(
            raw_state_obs=True, domain_name="acrobot"
        )
        native = SimpleNamespace(
            raw_state_obs=False, domain_name="cheetah"
        )
        self.assertEqual(
            _select_structured_dof_layout(acrobot, 8, LayoutStub),
            ("raw", 4),
        )
        self.assertIsNone(
            _select_structured_dof_layout(native, 17, LayoutStub)
        )

    def test_guarded_cartpole_controls_parse_from_benchmark_row(self):
        _, _, _, algo, _ = load_ct_hyperparams_from_table(
            "ct_sac",
            "cartpole-swingup",
            "mbq_structured_quad_roll",
        )
        expected = {
            "dynamics_lr": 3e-4,
            "dynamics_warmup": 10000,
            "dynamics_fit_horizon_warmup": 5000,
            "dynamics_duration_balance": "True",
            "dynamics_target_tau": 0.02,
            "dynamics_publish_max_flow_error_ratio": 1.0,
            "dynamics_require_value_head": "True",
            "dynamics_mass_logdet_reg": 1e-5,
            "dynamics_mass_condition_reg": 1e-4,
            "dynamics_mass_condition_limit": 100,
        }
        for key, value in expected.items():
            self.assertEqual(algo.get(key), value, key)

        dynamics_kwargs = _pop_structured_model_kwargs(algo)
        self.assertEqual(
            dynamics_kwargs,
            {
                "mass_logdet_reg": 1e-5,
                "mass_condition_reg": 1e-4,
                "mass_condition_limit": 100.0,
            },
        )
        for key in (
            "dynamics_mass_logdet_reg",
            "dynamics_mass_condition_reg",
            "dynamics_mass_condition_limit",
        ):
            self.assertNotIn(key, algo)

    def test_recovery_uses_new_layout_but_can_load_legacy_sidecars(self):
        fresh = _select_recovery_dof_layout("cartpole", True, 4)
        legacy = _select_recovery_dof_layout(
            "cartpole", True, 4, {"G_a.weight": th.zeros(2, 1)}
        )
        current = _select_recovery_dof_layout(
            "cartpole", True, 4, {"G_a.weight": th.zeros(1, 1)}
        )
        self.assertEqual(fresh.act_to_cfg, (0,))
        self.assertEqual(current.act_to_cfg, (0,))
        self.assertIsNone(legacy.act_to_cfg)


class TestCartpoleStructuredLayout(unittest.TestCase):
    def _model(self):
        th.manual_seed(0)
        return PortHamiltonianModel(
            4,
            1,
            mode="structured",
            structured_hidden=(32, 32),
            dof_layout=DOFLayout.cartpole(),
        )

    def test_layout_enforces_invariance_periodicity_and_sparse_actuation(self):
        m = self._model()
        self.assertEqual(m.layout.act_to_cfg, (0,))
        self.assertEqual(m.G_a.out_features, 1)

        theta = 0.37
        pos = th.tensor(
            [
                [-1.5, theta],
                [1.5, theta],
                [0.0, theta + 2.0 * np.pi],
            ],
            dtype=th.float32,
        )
        with th.no_grad():
            mass = th.vmap(m._mass)(pos)
            potential = th.vmap(m._base_potential)(pos)

        # Translating the cart or adding one hinge revolution must not change
        # either learned base object. The separate rail-limit energy is allowed
        # to depend on cart position near the hard stops.
        self.assertTrue(th.allclose(mass[0], mass[1], atol=1e-6, rtol=1e-6))
        self.assertTrue(th.allclose(mass[0], mass[2], atol=1e-5, rtol=1e-5))
        self.assertTrue(
            th.allclose(potential[0], potential[1], atol=1e-6, rtol=1e-6)
        )
        self.assertTrue(
            th.allclose(potential[0], potential[2], atol=1e-5, rtol=1e-5)
        )

        # The cartpole is nearly conservative; the layout-specific initial
        # damping prior should not begin orders of magnitude above the plant.
        damping = th.nn.functional.softplus(m._log_d)
        self.assertLess(float(damping.detach().max()), 1e-3)

    def test_forbidden_cart_position_derivatives_are_structural_zeroes(self):
        m = self._model()
        pos = th.tensor(
            [[-1.0, -0.5], [0.0, 0.2], [1.0, 1.3]],
            dtype=th.float32,
        ).requires_grad_(True)

        mass = th.vmap(m._mass)(pos)
        potential = th.vmap(m._base_potential)(pos)
        (dm_dx,) = th.autograd.grad(
            mass.sum(), pos, retain_graph=True
        )
        (dv_dx,) = th.autograd.grad(potential.sum(), pos)
        self.assertLess(float(dm_dx[:, 0].abs().max()), 1e-8)
        self.assertLess(float(dv_dx[:, 0].abs().max()), 1e-8)

    def test_recovery_energy_ledger_includes_rail_limit_damping(self):
        m = self._model()
        obs = np.asarray(
            [
                [1.82, 0.2, 1.0, 0.1],
                [-1.82, -0.3, -1.0, -0.2],
                [1.79, 0.7, 0.5, -0.1],
                [-1.79, -0.8, -0.5, 0.2],
            ],
            dtype=np.float32,
        )
        rep = energy_balance_report(
            m, obs, np.zeros((len(obs), 1), dtype=np.float32)
        )
        self.assertLess(rep["residual_nrmse"], 1e-4)
        self.assertLess(rep["power_joint_limit_mean"], 0.0)
        self.assertEqual(rep["passivity_violation_frac"], 0.0)

        terms = learned_terms(m, obs)
        # Gravity/base-potential recovery stays translation invariant; rail
        # force is reported on its own constraint-force axis.
        self.assertLess(float(np.abs(terms["g_pot"][:, 0]).max()), 1e-7)
        self.assertIn("joint_limit_F", terms)
        self.assertGreater(float(np.linalg.norm(terms["joint_limit_F"])), 0.0)

    def test_duration_balance_removes_long_tail_quadratic_leverage(self):
        dt = th.tensor([[[0.002], [0.03]]])
        weight = PortHamiltonianModel._duration_balance_weights(dt, 0.002)
        self.assertAlmostEqual(float(weight[0, 0]), 1.0, places=6)
        self.assertAlmostEqual(
            float(weight[0, 1]), (0.002 / 0.03) ** 2, places=7
        )
        # Endpoint error from a fixed drift error scales with dt. Weighting its
        # square must give the two durations equal leverage.
        weighted = weight * dt.square()
        self.assertTrue(th.allclose(weighted[:, :1], weighted[:, 1:]))


@unittest.skipUnless(HAVE_DMC, "dm_control / DMCContinuousEnv not available")
class TestIrregularCartpoleFlowStress(unittest.TestCase):
    @staticmethod
    def _collect(seed=0, n=180):
        env = DMCContinuousEnv(
            "cartpole",
            "swingup",
            time_sampling="irregular",
            dt=0.01,
            physics_dt=0.002,
            min_dt=0.002,
            max_dt=0.03,
            max_steps=400,
            episode_duration=2.0,
            time_sampling_kwargs={"tail_p": 0.99, "tail_split": 0.9},
            seed=seed,
            raw_state_obs=True,
        )
        rng = np.random.default_rng(seed)
        env.reset(seed=seed)
        obs, actions, next_obs, durations, dones = [], [], [], [], []
        for _ in range(n):
            action = rng.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)
            o, t, a, _, no, nt, terminated, truncated, _ = env.step_dt(action)
            obs.append(o)
            actions.append(a)
            next_obs.append(no)
            durations.append(nt - t)
            dones.append(terminated or truncated)
            if terminated or truncated:
                break
        return tuple(
            np.asarray(v, np.float32)
            for v in (obs, actions, next_obs, durations, dones)
        )

    def test_h4_flow_fit_with_long_tail_remains_finite(self):
        th.manual_seed(0)
        O, A, NO, DT, DN = self._collect()
        self.assertTrue(np.any(np.isclose(DT, 0.002)))
        self.assertTrue(np.any(np.isclose(DT, 0.03)))

        horizon = 4
        starts = [
            i
            for i in range(len(O) - horizon + 1)
            if not DN[i : i + horizon].any()
            and np.any(np.isclose(DT[i : i + horizon], 0.03))
        ][:4]
        self.assertTrue(starts, "seed did not produce a valid long-tail window")

        x = th.as_tensor(O[starts])
        acts = th.stack(
            [th.as_tensor(A[i : i + horizon]) for i in starts]
        )
        targets = th.stack(
            [th.as_tensor(NO[i : i + horizon]) for i in starts]
        )
        dt = th.stack(
            [th.as_tensor(DT[i : i + horizon]) for i in starts]
        ).unsqueeze(-1)
        mask = th.ones(len(starts), horizon, 1)

        model = PortHamiltonianModel(
            4,
            1,
            mode="structured",
            structured_hidden=(32, 32),
            dof_layout=DOFLayout.cartpole(),
        )
        optimizer = th.optim.Adam(model.parameters(), lr=3e-4)
        for _ in range(3):
            loss = model.fit_step_rollout(
                x,
                acts,
                targets,
                dt,
                mask,
                optimizer,
                max_step=0.002,
                balance_dt=0.002,
            )
            self.assertTrue(np.isfinite(loss))

        for name, value in model.state_dict().items():
            self.assertTrue(th.all(th.isfinite(value)), name)

        # The unguarded critic path reads a nominal 10 ms flow, so pin that
        # separately from the guarded fit rollout.
        with th.no_grad():
            endpoint = integrate_drift(
                model.drift,
                x,
                acts[:, 0],
                0.01,
                max_step=0.002,
            )
        self.assertTrue(th.all(th.isfinite(endpoint)))

    def test_benchmark_controls_construct_h1_to_h4_guarded_agent(self):
        _, _, _, configured, _ = load_ct_hyperparams_from_table(
            "ct_sac",
            "cartpole-swingup",
            "mbq_structured_quad_roll",
        )
        env = DMCContinuousEnv(
            "cartpole",
            "swingup",
            time_sampling="irregular",
            dt=0.01,
            physics_dt=0.002,
            min_dt=0.002,
            max_dt=0.03,
            max_steps=100,
            episode_duration=0.5,
            time_sampling_kwargs={"tail_p": 0.99, "tail_split": 0.9},
            raw_state_obs=True,
            seed=0,
        )
        policy = ActorQCriticModel(
            observation_space=env.observation_space,
            action_space=env.action_space,
            q_net_arch=[16],
            pi_net_arch=[16],
            v_net_arch=[16],
            device="cpu",
        )
        dynamics = PortHamiltonianModel(
            4,
            1,
            mode="structured",
            structured_hidden=(16,),
            dof_layout=DOFLayout.cartpole(),
        )
        control_names = (
            "dynamics_lr",
            "dynamics_warmup",
            "dynamics_fit_horizon_warmup",
            "dynamics_duration_balance",
            "dynamics_target_tau",
            "dynamics_publish_max_flow_error_ratio",
            "dynamics_require_value_head",
        )
        controls = {name: configured[name] for name in control_names}
        agent = CTSAC(
            env=env,
            model=policy,
            device="cpu",
            buffer_size=100,
            batch_size=8,
            learning_starts=10,
            use_model_based_q=True,
            dynamics_model=dynamics,
            dynamics_fit_horizon=4,
            generator_substeps=4,
            value_warmup=5,
            **controls,
        )
        self.assertTrue(agent.dynamics_duration_balance)
        self.assertTrue(agent.dynamics_require_value_head)
        self.assertEqual(agent._current_dynamics_fit_horizon(), 1)
        self.assertAlmostEqual(agent._dynamics_balance_dt(), 0.002)
        agent._dynamics_updates = agent.dynamics_fit_horizon_warmup
        self.assertEqual(agent._current_dynamics_fit_horizon(), 4)
        self.assertIsNot(agent.dynamics_target_model, agent.dynamics_model)


if __name__ == "__main__":
    unittest.main()
