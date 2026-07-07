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
    fit_model,
    ground_truth,
    learned_terms,
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
        sanity_check_truth(truth, obs, nq=nq)
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
        O, *_ = collect(env, 60, seed=0)
        obs = O[-20:]
        truth = ground_truth(env, obs)
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
        )
        rep = recovery_report(truth, fake)
        self.assertAlmostEqual(rep["gauge_scale_c"], 1.0 / s, places=5)
        self.assertLess(rep["mass_rel_frob_err"], 1e-5)
        self.assertGreater(rep["mass_entry_corr"], 0.999)
        self.assertLess(rep["mass_dMdz_ratio"], 1e-9)
        for k in ("potential_affine_R2", "kinetic_R2", "total_H_affine_R2",
                  "damping_affine_R2"):
            self.assertGreater(rep[k], 0.999, k)
        for k in ("gradV_force_corr", "coriolis_force_corr"):
            self.assertGreater(rep[k], 0.999, k)
        self.assertLess(rep["G_rel_frob_err"], 1e-6)


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


class TestRecoveryEndToEnd(unittest.TestCase):
    """A briefly fit structured model runs through the whole recovery pipeline."""

    def test_tiny_fit_produces_finite_report(self):
        th.manual_seed(0); np.random.seed(0)
        env = _cheetah()
        O, A, NO, DT, DN = collect(env, 300, seed=0)
        m = fit_model(env, O, A, NO, DT, DN, steps=50, horizon=2,
                      hidden=(32, 32), log_every=0)
        obs = O[-50:]
        truth = ground_truth(env, obs)
        rep = recovery_report(truth, learned_terms(m, obs))
        for k, v in rep.items():
            arr = np.asarray(v, dtype=np.float64)
            self.assertTrue(np.all(np.isfinite(arr)), k)

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
                    "contact_kcm"):
            self.assertIn(key, learned)
        rep = recovery_report(ground_truth(env, obs), learned)
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


if __name__ == "__main__":
    unittest.main()
