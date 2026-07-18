import unittest

import numpy as np
import torch as th
from gymnasium import spaces

from benchmarks.swingup_final_matrix import BACKEND_MODES, FINAL_MODES
from common.utils import load_ct_hyperparams_from_table
from models.actor_q_critic import ActorQCriticModel
from models.port_hamiltonian import DOFLayout, PortHamiltonianModel


ACROBOT_ENV_ID = "acrobot-swingup-v2"
ACROBOT_FINAL_CSV_MODES = tuple(
    dict.fromkeys((*FINAL_MODES, *BACKEND_MODES, "final_guarded_reanchor"))
)


class TestAcrobotStructuredRepresentation(unittest.TestCase):
    def setUp(self) -> None:
        th.manual_seed(7)
        self.layout = DOFLayout.acrobot()
        self.model = PortHamiltonianModel(
            4,
            1,
            mode="structured",
            structured_hidden=(16, 16),
            dof_layout=self.layout,
        )

    def test_layout_declares_acrobot_mechanics(self):
        layout = self.layout
        self.assertEqual((layout.obs_dim, layout.npos, layout.nv), (4, 2, 2))
        self.assertEqual(layout.pos_slice, (0, 2))
        self.assertEqual(layout.vel_slice, (2, 4))
        self.assertEqual(layout.cyclic_cfg, ())
        self.assertEqual(layout.obs_pos_to_cfg, (0, 1))
        self.assertEqual(layout.act_to_cfg, (1,))
        self.assertEqual(layout.m_invariant_pos, (0,))
        self.assertTrue(layout.enforce_m_invariance)
        self.assertEqual(layout.potential_invariant_pos, ())
        self.assertEqual(layout.periodic_pos, (0, 1))
        self.assertEqual(layout.joint_limits, ())
        damping = th.nn.functional.softplus(self.model._log_d)
        th.testing.assert_close(damping, th.full((2,), 0.05), atol=1e-7, rtol=1e-6)

    def test_sparse_port_applies_generalized_force_only_at_elbow(self):
        self.assertEqual(self.model.G_a.out_features, 1)
        with th.no_grad():
            # dm_control's normalized Acrobot motor has gear 2.  The port gain
            # is learned, but fixing it here makes the topology assertion exact.
            self.model.G_a.weight.fill_(2.0)
        action = th.tensor([[-0.4], [0.0], [0.7]])
        generalized_force = th.zeros(3, 2).index_add(
            1, self.model._act_to_cfg, self.model.G_a(action)
        )
        th.testing.assert_close(generalized_force[:, 0], th.zeros(3))
        th.testing.assert_close(generalized_force[:, 1], 2.0 * action[:, 0])

    def test_mass_invariance_and_periodic_mechanical_objects(self):
        pos = th.tensor(
            [[0.31, -0.73], [-1.2, 0.45], [2.1, -2.4]], dtype=th.float32
        )
        mass = th.vmap(self.model._mass)(pos)
        potential = th.vmap(self.model._potential)(pos)

        # Absolute shoulder rotation changes gravity but not rigid-body inertia.
        shoulder_rotated = pos.clone()
        shoulder_rotated[:, 0] += 0.83
        th.testing.assert_close(
            th.vmap(self.model._mass)(shoulder_rotated), mass, atol=0.0, rtol=0.0
        )
        mass_jacobian = th.func.jacfwd(self.model._mass)(pos[0])
        th.testing.assert_close(
            mass_jacobian[..., 0], th.zeros_like(mass_jacobian[..., 0]),
            atol=0.0, rtol=0.0,
        )

        for angle_index in (0, 1):
            wrapped = pos.clone()
            wrapped[:, angle_index] += 2.0 * np.pi
            th.testing.assert_close(
                th.vmap(self.model._mass)(wrapped), mass, atol=2e-6, rtol=2e-6
            )
            th.testing.assert_close(
                th.vmap(self.model._potential)(wrapped),
                potential,
                atol=2e-6,
                rtol=2e-6,
            )


class TestAcrobotPolicyRepresentation(unittest.TestCase):
    def test_actor_q_and_value_are_invariant_to_angle_winding(self):
        th.manual_seed(11)
        obs_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        model = ActorQCriticModel(
            observation_space=obs_space,
            action_space=action_space,
            q_net_arch=(16, 16),
            pi_net_arch=(16, 16),
            v_net_arch=(16, 16),
            n_critics=2,
            periodic_obs_indices=(0, 1),
            device="cpu",
        )
        obs = th.tensor(
            [[0.3, -0.7, 0.2, -0.1], [-2.0, 1.4, -0.3, 0.8]],
            dtype=th.float32,
        )
        action = th.tensor([[0.25], [-0.6]], dtype=th.float32)

        reference_features = model._process_obs(obs)
        reference_actor = model.act(obs, deterministic=True)
        reference_q = model.q_values(obs, action)
        reference_target_q = model.target_q_values(obs, action)
        reference_v = model.value(obs)
        reference_target_v = model.target_value(obs)
        self.assertEqual(tuple(reference_features.shape), (2, 6))

        shifts = ((2.0 * np.pi, 0.0), (0.0, 2.0 * np.pi), (2.0 * np.pi, -4.0 * np.pi))
        for shoulder_shift, elbow_shift in shifts:
            wound = obs.clone()
            wound[:, 0] += shoulder_shift
            wound[:, 1] += elbow_shift
            th.testing.assert_close(
                model._process_obs(wound), reference_features, atol=2e-6, rtol=2e-6
            )
            wound_actor = model.act(wound, deterministic=True)
            for actual, expected in zip(wound_actor, reference_actor):
                th.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
            for actual, expected in zip(model.q_values(wound, action), reference_q):
                th.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
            for actual, expected in zip(
                model.target_q_values(wound, action), reference_target_q
            ):
                th.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
            th.testing.assert_close(
                model.value(wound), reference_v, atol=2e-6, rtol=2e-6
            )
            th.testing.assert_close(
                model.target_value(wound), reference_target_v, atol=2e-6, rtol=2e-6
            )


class TestAcrobotFinalHyperparameters(unittest.TestCase):
    def _load(self, mode: str):
        return load_ct_hyperparams_from_table("ct_sac", ACROBOT_ENV_ID, mode)

    def test_all_final_modes_parse_periodic_observations_and_shared_parameters(self):
        self.assertEqual(
            set(ACROBOT_FINAL_CSV_MODES),
            {
                "final_mf",
                "final_mf_vhead",
                "final_oracle_loop",
                "final_oracle_rollout",
                "final_structured",
                "final_guarded",
                "final_reanchor",
                "final_guarded_reanchor",
            },
        )
        for mode in ACROBOT_FINAL_CSV_MODES:
            with self.subTest(mode=mode):
                total_steps, env, policy, algo, _ = self._load(mode)
                self.assertEqual(total_steps, 1_000_000)
                self.assertEqual(env["raw_state_obs"], "True")
                self.assertEqual(policy["periodic_obs_indices"], (0, 1))
                self.assertEqual(policy["log_std_init"], -1)
                self.assertEqual(algo["gamma"], 0.995)
                self.assertEqual(algo["learning_rate"], 3e-4)

    def test_final_mode_specific_generator_controls(self):
        _, _, _, model_free, _ = self._load("final_mf")
        _, _, model_free_v, model_free_v_algo, _ = self._load("final_mf_vhead")
        self.assertNotIn("use_model_based_q", model_free)
        self.assertEqual(model_free_v["v_net_arch"], [400, 300])
        self.assertNotIn("use_model_based_q", model_free_v_algo)

        for mode, backend in (
            ("final_oracle_loop", "loop"),
            ("final_oracle_rollout", "rollout"),
        ):
            with self.subTest(mode=mode):
                _, env, _, algo, _ = self._load(mode)
                self.assertEqual(env["drift_backend"], backend)
                self.assertEqual(env["drift_rollout_threads"], 4)
                self.assertEqual(algo["dynamics_source"], "mujoco")

        for mode in (
            "final_structured",
            "final_guarded",
            "final_reanchor",
            "final_guarded_reanchor",
        ):
            with self.subTest(mode=mode):
                _, _, policy, algo, _ = self._load(mode)
                self.assertEqual(policy["v_net_arch"], [400, 300])
                self.assertEqual(algo["dynamics_source"], "structured")
                self.assertEqual(algo["dynamics_mass_condition_limit"], 100)

        for mode in ("final_guarded", "final_guarded_reanchor"):
            _, _, _, algo, _ = self._load(mode)
            self.assertEqual(algo["target_guard_kappa"], 6)
            self.assertEqual(algo["target_guard_cap"], 600)
        for mode in ("final_reanchor", "final_guarded_reanchor"):
            _, _, _, algo, _ = self._load(mode)
            self.assertEqual(algo["target_reanchor"], "True")
            self.assertEqual(algo["target_reanchor_gate_rho"], 1)


if __name__ == "__main__":
    unittest.main()
