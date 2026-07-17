import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch as th

from algorithms.ct_sac import reanchor_gate_statistics
from evaluations.cartpole_reanchor_audit import build_targets, _chunked_gate, _estats
from tests.test_ct_sac_target_reanchor import _batch, _ConstantDynamics, _make_agent


class TestChunkedReanchorGate(unittest.TestCase):
    def test_uses_training_sized_batch_medians_and_finite_fallback(self):
        algo = SimpleNamespace(batch_size=2)
        T = 0.01
        dt = th.full((4, 1), 0.005)
        obs = th.zeros(4, 1)
        # Realized rates are [1, 3] and [10, 30]. Torch's median for each
        # two-row chunk is the lower value, so the two gate scales differ.
        next_obs = dt * th.tensor([[1.0], [3.0], [10.0], [30.0]])
        innovation = dt.clone()  # unit innovation rate on every row
        t_re = th.tensor([[10.0], [20.0], [float("nan")], [40.0]])
        t_fd = th.tensor([[1.0], [2.0], [3.0], [4.0]])

        target, aux = _chunked_gate(
            algo, obs, next_obs, innovation, dt, T, 1.0, t_re, t_fd,
        )

        expected_rho = th.cat([
            reanchor_gate_statistics(
                obs[:2], next_obs[:2], innovation[:2], dt[:2], T,
            )[0],
            reanchor_gate_statistics(
                obs[2:], next_obs[2:], innovation[2:], dt[2:], T,
            )[0],
        ])
        th.testing.assert_close(aux["rho"], expected_rho)
        self.assertEqual(aux["gate_batch_size"], 2)
        self.assertEqual(aux["gate_chunks"], 2)

        # The non-finite re-anchored row is restored exactly to its finite-
        # difference anchor, just as in the trainer.
        self.assertTrue(aux["gate_fallback"][2].item())
        self.assertEqual(aux["lam"][2].item(), 0.0)
        self.assertEqual(target[2].item(), t_fd[2].item())
        self.assertTrue(th.isfinite(target).all())

        # A whole-dataset median would produce a different first-row rho;
        # guard this specifically so the audit cannot regress to one giant
        # pseudo-minibatch.
        whole_rho = reanchor_gate_statistics(
            obs, next_obs, innovation, dt, T,
        )[0]
        self.assertNotEqual(aux["rho"][0].item(), whole_rho[0].item())

    def test_audit_gate_matches_trainer_over_the_same_minibatch_chunks(self):
        _, agent = _make_agent(
            dynamics=_ConstantDynamics(0.3),
            target_reanchor=True,
            target_reanchor_gate_rho=1.0,
        )
        dts = [0.002, 0.008, 0.02, 0.03, 0.004, 0.012, 0.025]
        batch = _batch(agent, dts)
        obs, actions, next_obs, rewards, dones, dt, alpha = batch

        trainer_chunks = []
        for start in range(0, len(dts), agent.batch_size):
            sl = slice(start, min(start + agent.batch_size, len(dts)))
            trainer_chunks.append(agent._model_based_target(
                obs[sl], actions[sl], next_obs[sl], rewards[sl], dones[sl],
                dt[sl], alpha,
            ))
        trainer_target = th.cat(trainer_chunks)

        with patch(
            "evaluations.cartpole_reanchor_audit.mujoco_transition",
            side_effect=lambda env, O, A, T: O,
        ):
            targets, aux = build_targets(
                agent,
                agent.env,
                obs.numpy(),
                actions.numpy(),
                rewards.numpy(),
                next_obs.numpy(),
                dones.numpy(),
                dt.numpy(),
                gate_rho=1.0,
            )

        th.testing.assert_close(targets["gate"], trainer_target)
        self.assertEqual(aux["gate_chunks"], 2)


class TestEndpointAuditStatistics(unittest.TestCase):
    def test_excludes_terminals_and_reports_nonfinite_fraction(self):
        eps = th.tensor([[1000.0], [2.0], [float("nan")], [4.0]])
        dv_or = th.tensor([[1000.0], [2.0], [10000.0], [4.0]])
        nonterminal = th.tensor([[False], [True], [True], [True]])
        result = {}

        _estats(eps, dv_or, nonterminal, "eps_re", result)

        # The large terminal error is absent; one of three nonterminal errors
        # is non-finite and the two finite errors determine the RMS.
        self.assertEqual(result["eps_re_finite_n"], 2)
        self.assertAlmostEqual(result["eps_re_nonfinite_frac"], 1.0 / 3.0, 6)
        self.assertEqual(result["eps_re_rms"], round(math.sqrt(10.0), 4))
        self.assertEqual(result["eps_re_rel"], 1.0)
        self.assertLess(result["eps_re_p99"], 4.1)


if __name__ == "__main__":
    unittest.main()
