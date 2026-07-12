# tests/test_hamiltonian_recovery.py

import os
import tempfile
import unittest

import numpy as np
import torch as th

from algorithms.ct_sac import CTSAC
from environment.dmc import DMCContinuousEnv
from evaluations.hamiltonian_recovery import (
    collect,
    energy_balance_report,
    fit_model,
    generator_report,
    ground_truth,
    learned_terms,
    mujoco_transition,
    predictive_report,
    quadrature_report,
    recovery_report,
    sanity_check_truth,
)
from models.actor_q_critic import ActorQCriticModel
from models.port_hamiltonian import PortHamiltonianModel


def _cheetah(seed=0):
    return DMCContinuousEnv(
        "cheetah", "run", time_sampling="uniform", dt=0.01, physics_dt=0.002,
        episode_duration=20.0, seed=seed,
    )


# metrics that are legitimately NaN on some data (empty stratum, no contact
# edges, no valid rollout window) — excluded from the blanket finite checks
_NAN_OK = ("_strata", "contact_edge_offset_steps", "contact_edge_matched_frac",
           "rollout_rel_err")


def _assert_finite(testcase, node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            _assert_finite(testcase, v, f"{path}.{k}")
        return
    if isinstance(node, (bool, str)) or node is None:
        return
    arr = np.asarray(node, dtype=np.float64)
    if any(tag in path for tag in _NAN_OK):
        return
    testcase.assertTrue(np.all(np.isfinite(arr)), path)


class TestGroundTruthExtraction(unittest.TestCase):
    """The MuJoCo term extraction must be internally consistent."""

    def test_consistency_and_actuator_pattern(self):
        th.manual_seed(0); np.random.seed(0)
        env = _cheetah()
        O, A, NO, DT, DN = collect(env, 60, seed=0)
        obs = O[-20:].astype(np.float64)
        truth = ground_truth(env, obs)
        nq = int(env._env.physics.model.nq)
        # SPD mass, kinetic identity e_kin = 1/2 v^T M v, translation invariance
        sanity_check_truth(truth, obs, pos_width=nq - 1)
        # cheetah's three root DOFs are unactuated: G rows ~ 0 there, and each
        # actuator drives exactly one joint DOF with its gear value
        G = truth["G"]
        self.assertLess(np.abs(G[:3]).max(), 1e-9)
        self.assertEqual((np.abs(G) > 1e-9).sum(), G.shape[1])
        # Coriolis force at (nearly) zero velocity vanishes
        obs0 = obs.copy(); obs0[:, nq - 1:] = 0.0
        t0 = ground_truth(env, obs0)
        self.assertLess(np.abs(t0["coriolis"]).max(), 1e-9)


class TestConventionValidation(unittest.TestCase):
    """The model-side Coriolis/gradient formulas and the qfrc-based extraction
    must agree when both are built from the TRUE physics: finite-difference the
    true M(q) and E_pot(q) over the observed positions, push them through the
    same einsum/gradient conventions the model uses, and compare with the
    qfrc_bias/qfrc_passive extraction. Validates signs and frames end to end."""

    def test_true_coriolis_and_gradV_conventions(self):
        th.manual_seed(0); np.random.seed(0)
        env = _cheetah()
        O, *_ = collect(env, 80, seed=0)
        obs = O[-6:].astype(np.float64)
        nq = int(env._env.physics.model.nq)
        nv = int(env._env.physics.model.nv)
        B, npos, eps = obs.shape[0], nq - 1, 1e-5

        truth = ground_truth(env, obs)
        dM = np.zeros((B, nv, nv, nv))
        gV_fd = np.zeros((B, nv))
        for j in range(npos):
            op, om = obs.copy(), obs.copy()
            op[:, j] += eps; om[:, j] -= eps
            tp, tm = ground_truth(env, op), ground_truth(env, om)
            # observed position j is config DOF j+1 (root x is cyclic DOF 0)
            dM[:, :, :, j + 1] = (tp["M"] - tm["M"]) / (2 * eps)
            gV_fd[:, j + 1] = (tp["e_pot"] - tm["e_pot"]) / (2 * eps)

        v = obs[:, npos:]
        c_fd = (
            np.einsum("nabk,nk,nb->na", dM, v, v)
            - 0.5 * np.einsum("nabk,na,nb->nk", dM, v, v)
        )
        rel_c = np.abs(c_fd - truth["coriolis"]).max() / (
            np.abs(truth["coriolis"]).max() + 1e-12
        )
        rel_g = np.abs(gV_fd - truth["g_pot"]).max() / (
            np.abs(truth["g_pot"]).max() + 1e-12
        )
        self.assertLess(rel_c, 5e-2, "Coriolis convention mismatch")
        self.assertLess(rel_g, 5e-2, "potential-gradient convention mismatch")


class TestRecoveryMetrics(unittest.TestCase):
    """Feeding a rescaled copy of the truth through the report must recover the
    gauge and score ~perfectly on every metric."""

    def test_rescaled_truth_scores_perfectly(self):
        th.manual_seed(0); np.random.seed(0)
        env = _cheetah()
        O, A, *_ = collect(env, 60, seed=0)
        obs, act = O[-20:], A[-20:]
        truth = ground_truth(env, obs)
        nq = int(env._env.physics.model.nq)
        qd = obs[:, nq - 1:].astype(np.float64)
        s = 0.5  # pretend the learner found everything at half scale
        fake = dict(
            M=s * truth["M"],
            V=s * truth["e_pot"] + 3.0,           # offset gauge
            e_kin=s * truth["e_kin"],
            g_pot=s * truth["g_pot"],
            coriolis=s * truth["coriolis"],
            d_base=s * truth["dof_damping"],
            G=s * truth["G"],
            dM_mag=np.array([0.0] + [1.0] * 7),   # no z-dependence
            qd=qd,
            f_damp=s * truth["dof_damping"][None, :] * qd,
        )
        rep = recovery_report(truth, fake, actions=act)
        self.assertAlmostEqual(rep["gauge_scale_c"], 1.0 / s, places=5)
        # strict (scale-locked) recovery: near-zero error everywhere
        for k in ("mass_rel_frob_err", "mass_diag_rel_err",
                  "mass_offdiag_rel_err", "mass_eig_rel_err",
                  "mass_inverse_response_nrmse", "potential_locked_nrmse",
                  "kinetic_nrmse", "total_H_locked_nrmse", "damping_rel_err",
                  "gradV_force_nrmse", "coriolis_force_nrmse",
                  "damping_force_nrmse", "actuator_force_nrmse"):
            self.assertLess(rep[k], 1e-5, k)
        self.assertLess(rep["G_rel_frob_err"], 1e-6)
        self.assertAlmostEqual(rep["mass_cond_ratio"], 1.0, places=4)
        # shape diagnostics and correlations
        self.assertGreater(rep["mass_entry_corr"], 0.999)
        self.assertGreater(rep["mass_uppertri_corr"], 0.999)
        for k in ("potential_shape_R2", "kinetic_R2", "total_H_shape_R2",
                  "damping_locked_R2"):
            self.assertGreater(rep[k], 0.999, k)
        for k in ("gradV_force_corr", "coriolis_force_corr"):
            self.assertGreater(rep[k], 0.999, k)
        self.assertAlmostEqual(rep["potential_slope_ratio"], 1.0, places=4)
        # architectural probe: zero and flagged non-structural for a plain dict
        self.assertLess(rep["mass_dMdz_ratio"], 1e-9)
        self.assertFalse(rep["mass_dMdz_structural"])


class TestDynamicsSidecar(unittest.TestCase):
    """CTSAC checkpoints save/load the learned dynamics model in a sidecar."""

    def test_save_load_roundtrip(self):
        th.manual_seed(0); np.random.seed(0)
        env = _cheetah()
        od = int(env.observation_space.shape[0])
        ad = int(env.action_space.shape[0])
        model = ActorQCriticModel(
            observation_space=env.observation_space, action_space=env.action_space,
            q_net_arch=[16], pi_net_arch=[16], device="cpu",
        )
        ph = PortHamiltonianModel(od, ad, mode="structured", structured_hidden=(16,))
        agent = CTSAC(env=env, model=model, device="cpu", seed=0,
                      use_model_based_q=True, dynamics_model=ph)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "best_model.pth")
            agent.save(path)
            sidecar = os.path.join(d, "best_model.dynamics.pth")
            self.assertTrue(os.path.exists(sidecar))
            before = {k: v.clone() for k, v in ph.state_dict().items()}
            with th.no_grad():
                for p_ in ph.parameters():
                    p_.add_(1.0)
            agent.load(path)
            after = ph.state_dict()
            for k in before:
                self.assertTrue(th.allclose(before[k], after[k]), k)

    def test_oracle_has_no_sidecar(self):
        th.manual_seed(0); np.random.seed(0)
        env = _cheetah()
        model = ActorQCriticModel(
            observation_space=env.observation_space, action_space=env.action_space,
            q_net_arch=[16], pi_net_arch=[16], device="cpu",
        )
        ph = PortHamiltonianModel(
            int(env.observation_space.shape[0]), int(env.action_space.shape[0]),
            mode="mujoco", drift_fn=env.dynamics_terms,
        )
        agent = CTSAC(env=env, model=model, device="cpu", seed=0,
                      use_model_based_q=True, dynamics_model=ph)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.pth")
            agent.save(path)
            self.assertFalse(os.path.exists(os.path.join(d, "ckpt.dynamics.pth")))

    def test_sidecar_saves_published_target_not_rejected_live_candidate(self):
        th.manual_seed(0); np.random.seed(0)
        env = _cheetah()
        model = ActorQCriticModel(
            observation_space=env.observation_space,
            action_space=env.action_space,
            q_net_arch=[16], pi_net_arch=[16], v_net_arch=[16], device="cpu",
        )
        dynamics = PortHamiltonianModel(
            int(env.observation_space.shape[0]), int(env.action_space.shape[0]),
            mode="structured", structured_hidden=(16,),
        )
        agent = CTSAC(
            env=env, model=model, device="cpu", seed=0,
            use_model_based_q=True, dynamics_model=dynamics,
        )
        with th.no_grad():
            for p in agent.dynamics_target_model.parameters():
                p.fill_(1.0)
            for p in agent.dynamics_model.parameters():
                p.fill_(2.0)  # finite but not published
        agent._dynamics_publications = 1

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "accepted.pth")
            agent.save(path)
            saved = th.load(
                os.path.join(d, "accepted.dynamics.pth"), map_location="cpu"
            )
        for value in saved.values():
            if value.is_floating_point():
                self.assertFalse(th.any(value == 2.0))
        first_param = next(agent.dynamics_target_model.parameters())
        first_key = next(
            k for k, v in saved.items()
            if v.shape == first_param.shape and v.is_floating_point()
        )
        self.assertTrue(th.allclose(saved[first_key], first_param.detach()))


class TestRecoveryEndToEnd(unittest.TestCase):
    """A briefly fit structured model runs through the whole recovery pipeline."""

    def test_tiny_fit_produces_finite_report(self):
        th.manual_seed(0); np.random.seed(0)
        env = _cheetah()
        O, A, NO, DT, DN = collect(env, 300, seed=0)
        m = fit_model(env, O, A, NO, DT, DN, steps=50, horizon=2,
                      hidden=(32, 32), log_every=0)
        obs, act = O[-50:], A[-50:]
        truth = ground_truth(env, obs)
        rep = recovery_report(truth, learned_terms(m, obs), actions=act,
                              dt=DT[-50:], dones=DN[-50:])
        _assert_finite(self, rep)
        pred = predictive_report(m, obs, act, NO[-50:], DT[-50:], DN[-50:],
                                 max_step=env.physics_dt)
        _assert_finite(self, pred)
        eng = energy_balance_report(m, obs, act)
        _assert_finite(self, eng)
        # the energy balance is an identity of the model: the residual must be
        # numerics-small even on a barely fit model
        self.assertLess(eng["residual_nrmse"], 1e-2)

    def test_contact_port_combined_potential(self):
        """With the port active the report gains the combined-conservative-force
        metrics; on a fresh model the gaps are positive (+0.5 init), so the port
        is silent, the combined gradient equals grad V, and both correlations
        coincide."""
        th.manual_seed(0); np.random.seed(0)
        env = _cheetah()
        O, A, NO, DT, DN = collect(env, 300, seed=0)
        m = fit_model(env, O, A, NO, DT, DN, steps=5, horizon=1,
                      contact_force=2, hidden=(32, 32), log_every=0)
        obs = O[-50:]
        learned = learned_terms(m, obs)
        for key in ("g_pot_combined", "contact_gap", "contact_in_frac",
                    "contact_in_frac_per", "contact_spring_ratio",
                    "contact_kcm", "contact_F", "contact_power"):
            self.assertIn(key, learned)
        rep = recovery_report(ground_truth(env, obs), learned,
                              actions=A[-50:], dt=DT[-50:], dones=DN[-50:])
        self.assertIn("gradV_combined_corr", rep)
        # port parameters: k/c at gauge scale, mu raw; all positive (softplus),
        # one entry per contact point; per-contact activity + gap stats present
        for key in ("contact_k", "contact_c", "contact_mu",
                    "contact_in_frac_per"):
            self.assertEqual(len(rep[key]), 2, key)
        for key in ("contact_k", "contact_c", "contact_mu"):
            self.assertTrue(all(v > 0 for v in rep[key]), key)
        self.assertTrue(np.isfinite(rep["contact_gap_mean"]))
        self.assertGreaterEqual(rep["contact_gap_min"], -1e6)
        # 5 fit steps leave the +0.5 gap-bias intact: silent port, tiny ratio,
        # and the combined field is just grad V.
        if float(learned["contact_gap"].min()) > 0.1:
            self.assertLess(rep["contact_spring_ratio"], 1e-3)
            self.assertAlmostEqual(rep["gradV_combined_corr"],
                                   rep["gradV_force_corr"], places=3)
        # no-port models must not grow the new keys
        m0 = fit_model(env, O, A, NO, DT, DN, steps=1, horizon=1,
                       hidden=(32, 32), log_every=0)
        self.assertNotIn("g_pot_combined", learned_terms(m0, obs))


class TestRawStateRecovery(unittest.TestCase):
    """The generalized extraction on a raw-state env (cartpole): ground truth
    consistency and the full report through the raw DOFLayout."""

    def _cartpole(self, seed=0):
        return DMCContinuousEnv(
            "cartpole", "swingup", time_sampling="uniform", dt=0.01,
            physics_dt=0.002, episode_duration=20.0, seed=seed,
            raw_state_obs=True,
        )

    def test_ground_truth_consistency(self):
        th.manual_seed(0); np.random.seed(0)
        env = self._cartpole()
        O, *_ = collect(env, 60, seed=0)
        obs = O[-30:]
        truth = ground_truth(env, obs)
        nq = int(env._env.physics.model.nq)
        sanity_check_truth(truth, obs.astype(np.float64), pos_width=nq,
                           check_root_invariance=False)
        # cartpole-specific physics: gravity is vertical, the slider is
        # horizontal, so the gravity torque's slider component is zero
        self.assertLess(np.abs(truth["g_pot"][:, 0]).max(), 1e-9)
        # and the true M depends only on the pole angle, never the cart position
        self.assertEqual(truth["M"].shape, (30, 2, 2))

    def test_tiny_fit_produces_finite_report(self):
        th.manual_seed(0); np.random.seed(0)
        env = self._cartpole()
        O, A, NO, DT, DN = collect(env, 300, seed=0)
        from models.port_hamiltonian import DOFLayout
        m = fit_model(env, O, A, NO, DT, DN, steps=50, horizon=2,
                      hidden=(32, 32), log_every=0,
                      dof_layout=DOFLayout.raw_state(nv=2))
        obs = O[-50:]
        truth = ground_truth(env, obs)
        rep = recovery_report(truth, learned_terms(m, obs), actions=A[-50:],
                              dt=DT[-50:], dones=DN[-50:])
        _assert_finite(self, rep)


class TestMuJoCoTransition(unittest.TestCase):
    """The paired-rollout helper must reproduce the env's own transitions when
    integrating the recorded action over the recorded (uniform) duration."""

    def test_matches_recorded_transitions(self):
        th.manual_seed(0); np.random.seed(0)
        env = _cheetah()
        O, A, NO, DT, DN = collect(env, 40, seed=0)
        keep = DN[:-1] == 0.0  # exclude reset transitions
        obs, act, nxt = O[:-1][keep][:10], A[:-1][keep][:10], NO[:-1][keep][:10]
        pred = mujoco_transition(env, obs, act, float(DT[0]))
        err = np.abs(pred - nxt).max() / (np.abs(nxt).max() + 1e-12)
        self.assertLess(err, 1e-4, "mujoco_transition diverges from env.step")


class TestControlRelevantMetrics(unittest.TestCase):
    """Generator-projection and quadrature-label errors through a tiny V-head
    policy. With the oracle drift substituted, both errors must be ~zero."""

    def _tiny_policy(self, env):
        return ActorQCriticModel(
            observation_space=env.observation_space,
            action_space=env.action_space,
            q_net_arch=[16], pi_net_arch=[16], v_net_arch=[16], device="cpu",
        )

    def test_reports_finite_and_oracle_zero(self):
        th.manual_seed(0); np.random.seed(0)
        env = _cheetah()
        O, A, NO, DT, DN = collect(env, 120, seed=0)
        obs, act = O[-20:], A[-20:]
        policy = self._tiny_policy(env)
        self.assertTrue(policy.has_v_head)
        m = fit_model(env, O, A, NO, DT, DN, steps=5, horizon=1,
                      hidden=(16, 16), log_every=0)
        truth = ground_truth(env, obs)

        gen = generator_report(m, env, policy, obs, act,
                               contact_flag=truth["contact_flag"],
                               actions_pi=act[::-1].copy())
        _assert_finite(self, gen)
        self.assertIn("data_actions", gen)
        self.assertIn("policy_actions", gen)
        self.assertIn("err_vs_action_novelty_corr", gen)

        quad = quadrature_report(m, env, policy, obs, act,
                                 dt_default=float(env.dt_default),
                                 max_step=env.physics_dt,
                                 contact_flag=truth["contact_flag"])
        _assert_finite(self, quad)
        self.assertGreaterEqual(quad["sign_disagree_frac"], 0.0)
        self.assertLessEqual(quad["sign_disagree_frac"], 1.0)

        # oracle drift: the generator projection error must vanish identically
        oracle = PortHamiltonianModel(
            int(env.observation_space.shape[0]), int(env.action_space.shape[0]),
            mode="mujoco", drift_fn=env.dynamics_terms,
        )
        oracle.layout = m.layout  # generator_report slices velocities via layout
        gen0 = generator_report(oracle, env, policy, obs, act,
                                contact_flag=truth["contact_flag"])
        self.assertLess(gen0["data_actions"]["rmse"], 1e-6)


if __name__ == "__main__":
    unittest.main()
