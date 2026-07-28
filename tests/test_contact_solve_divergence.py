"""A diverged mass matrix must be a recoverable flow failure, not a crash.

Observed on the cheetah constraint chain (job 7877785): two of 48 cells died at
~12k steps with

    torch._C._LinAlgError: linalg.cholesky: (Batch element 2): ... not positive-definite

raised from ``_constraint_contact_solve`` while ``_post_fit_flow_quality`` was
validating a freshly fitted dynamics model. That check exists precisely to
reject a bad fit and roll the live model back -- it already catches
``FlowIntegrationError`` -- but a bare ``_LinAlgError`` is not something it
knows about, so the whole run terminated instead. The learned mass matrix was
visibly on its way there beforehand: condition number 2.5 at 10k steps, 3237 at
12k, 7359 at 14k on the same cell under the previous implementation.
"""

import unittest

import torch as th

from models.port_hamiltonian import (
    DOFLayout,
    FlowIntegrationError,
    PortHamiltonianModel,
    integrate_drift,
)


def _contact_model(**kwargs):
    th.manual_seed(0)
    return PortHamiltonianModel(
        obs_dim=17, action_dim=6, mode="structured", contact_force=4,
        dof_layout=DOFLayout.cheetah(), **kwargs
    )


class TestDivergedMassMatrixIsRecoverable(unittest.TestCase):
    def setUp(self):
        gen = th.Generator().manual_seed(1)
        self.x = th.randn(6, 17, generator=gen)
        self.a = th.randn(6, 6, generator=gen)

    def test_healthy_model_solves_normally(self):
        model = _contact_model()
        with th.no_grad():
            drift = model.drift(self.x, self.a)
        self.assertTrue(bool(th.all(th.isfinite(drift))))

    def test_diverged_mass_raises_flow_error_not_linalg_error(self):
        """The failure must arrive as the type the quality check recovers from."""
        for label, poison in (
            ("non-finite", float("nan")),
            ("overflowing", 1e30),
        ):
            with self.subTest(mass=label):
                model = _contact_model()
                with th.no_grad():
                    model.mass_net[-1].bias.fill_(poison)
                with self.assertRaises(FlowIntegrationError) as caught:
                    with th.no_grad():
                        model.drift(self.x, self.a)
                self.assertIn("positive-definite", str(caught.exception))

    def test_integrate_drift_surfaces_it_as_a_flow_failure(self):
        """End to end: the caller of integrate_drift sees a recoverable error."""
        model = _contact_model()
        with th.no_grad():
            model.mass_net[-1].bias.fill_(float("nan"))
        with self.assertRaises(FlowIntegrationError):
            with th.no_grad():
                integrate_drift(
                    model.drift, self.x, self.a, 0.01, max_step=0.002,
                    check_finite=True,
                )

    def test_compliant_solver_is_untouched(self):
        """Only the constraint solve factorizes a Delassus matrix."""
        model = _contact_model(contact_solver="compliant")
        with th.no_grad():
            drift = model.drift(self.x, self.a)
        self.assertTrue(bool(th.all(th.isfinite(drift))))

    def test_post_fit_check_rejects_and_asks_for_a_rollback(self):
        """The behaviour that matters: the run survives and the fit is undone.

        ``_post_fit_flow_quality`` must return not-accepted with a reason the
        caller routes to ``_rollback_live_dynamics`` (it rolls back on reasons
        starting with "flow evaluation failed"), rather than propagating.

        The failure is injected at the drift rather than provoked numerically:
        the real trigger is a finite but badly conditioned mass matrix, which
        the mass diagnostics (a finiteness check) correctly let through and
        which is chaotic to reproduce exactly. The tests above pin that a failed
        factorization raises ``FlowIntegrationError``; this one pins that such an
        error becomes a rejection plus rollback instead of killing the run.
        """
        import numpy as np
        from algorithms.ct_sac import CTSAC
        from environment.dmc import DMCContinuousEnv
        from models.actor_q_critic import ActorQCriticModel

        th.manual_seed(0)
        np.random.seed(0)
        # The cheetah benchmark rows leave env_raw_state_obs blank, which is the
        # 17-dim [qpos[1:]; qvel] observation DOFLayout.cheetah() expects.
        env = DMCContinuousEnv(
            "cheetah", "run", time_sampling="uniform", dt=0.01,
            episode_duration=20.0,
        )
        obs_dim = int(env.observation_space.shape[0])
        act_dim = int(env.action_space.shape[0])
        model = ActorQCriticModel(
            observation_space=env.observation_space,
            action_space=env.action_space,
            q_net_arch=[32], pi_net_arch=[32], device="cpu",
        )
        dynamics = PortHamiltonianModel(
            obs_dim, act_dim, mode="structured", contact_force=4,
            structured_hidden=(32, 32), dof_layout=DOFLayout.cheetah(),
        )
        agent = CTSAC(
            env=env, model=model, device="cpu", learning_starts=10,
            batch_size=16, buffer_size=500, seed=0, use_model_based_q=True,
            dynamics_model=dynamics, dynamics_warmup=5,
        )
        # A healthy model with a completed fit, so the cheap guards pass and the
        # check actually reaches its flow evaluation.
        agent.dynamics_model.last_fit_accepted = True
        healthy_drift = agent.dynamics_model.drift

        def diverged_drift(x, a):
            raise FlowIntegrationError(
                "contact Delassus matrix is not positive-definite for 1 of 8 "
                "samples; the learned mass matrix has diverged"
            )

        agent.dynamics_model.drift = diverged_drift
        self.addCleanup(setattr, agent.dynamics_model, "drift", healthy_drift)

        batch = 8
        gen = th.Generator().manual_seed(2)
        accepted, ratio, reason = agent._post_fit_flow_quality(
            th.randn(batch, obs_dim, generator=gen).numpy(),
            th.randn(batch, act_dim, generator=gen).numpy(),
            th.randn(batch, obs_dim, generator=gen).numpy(),
            np.full((batch, 1), 0.01, dtype=np.float32),
            0.5,
        )
        self.assertFalse(accepted)
        self.assertTrue(
            reason.startswith("flow evaluation failed"),
            f"reason must be one the caller rolls back on, got: {reason!r}",
        )
        self.assertIn("positive-definite", reason)


if __name__ == "__main__":
    unittest.main()
