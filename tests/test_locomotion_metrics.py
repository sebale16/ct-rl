import unittest

import numpy as np

from evaluations.locomotion_metrics import (
    CheetahRollout,
    _JOINT_NAMES,
    _autocorr_peak,
    _plv,
    _poincare_dispersion,
    _spectral_stats,
    _stride_stats,
    _phase,
    energy_metrics,
    evaluate_locomotion,
    gait_metrics,
    phase_portrait_data,
    rollout_cheetah,
)


def _synthetic_rollout(T=100, dt=0.01):
    """A rollout with analytically known actuator work.

    Six actuated DOFs (indices 3..8) carry constant generalized torque ``tau``
    and ramp linearly so each per-step angle increment ``c`` is constant. Then
    per-joint per-step work is exactly ``tau*c``.
    """
    nq = nv = 9
    ad = np.arange(3, 9)
    tau = np.array([10.0, -20.0, 30.0, -5.0, 0.0, 15.0])
    c = np.array([0.01, 0.01, -0.01, 0.01, 0.0, 0.02])

    steps = np.arange(1, T + 1)[:, None]
    qpos = np.zeros((T, nq))
    qpos[:, 0] = steps[:, 0] * 0.05                 # rootx: 5 m over 100 steps
    qpos[:, ad] = steps * c[None, :]                # linear ramp -> constant dq
    qfrc = np.zeros((T, nv))
    qfrc[:, ad] = tau[None, :]
    return CheetahRollout(
        dt=np.full(T, dt),
        time=np.cumsum(np.full(T, dt)),
        reward=np.ones(T),
        qpos=qpos,
        qvel=np.zeros((T, nv)),
        qfrc_act=qfrc,
        action=np.full((T, 6), 0.5),
        speed=np.full(T, 5.0),
        foot_z=np.zeros((T, 2)),
        qpos0=np.zeros(nq),
        act_dof=ad,
        gear=np.array([120.0, 90.0, 60.0, 90.0, 60.0, 30.0]),
        mass=14.0,
        g=9.81,
    ), tau, c


class TestEnergyMetrics(unittest.TestCase):
    def test_work_decomposition_is_exact(self):
        roll, tau, c = _synthetic_rollout(T=100)
        m = energy_metrics(roll)
        per_step = tau * c                          # per-joint work per step
        self.assertAlmostEqual(m["work_net_J"], 100 * per_step.sum(), places=6)
        self.assertAlmostEqual(m["work_pos_J"], 100 * per_step[per_step > 0].sum(), places=6)
        self.assertAlmostEqual(m["work_neg_J"], 100 * per_step[per_step < 0].sum(), places=6)
        self.assertAlmostEqual(m["work_abs_J"], 100 * np.abs(per_step).sum(), places=6)
        # pos + neg == net
        self.assertAlmostEqual(m["work_pos_J"] + m["work_neg_J"], m["work_net_J"], places=6)

    def test_derived_quantities(self):
        roll, tau, c = _synthetic_rollout(T=100)
        m = energy_metrics(roll)
        self.assertAlmostEqual(m["distance_m"], 5.0, places=6)
        self.assertAlmostEqual(m["duration_s"], 1.0, places=6)
        self.assertAlmostEqual(m["return"], 100.0, places=6)
        # control cost = sum_j a_j^2 * duration = 6*0.25*1.0
        self.assertAlmostEqual(m["control_cost"], 1.5, places=6)
        # CoT = E+/(m g d)
        self.assertAlmostEqual(
            m["cost_of_transport"], m["work_pos_J"] / (14.0 * 9.81 * 5.0), places=9
        )
        self.assertAlmostEqual(m["return_per_joule"], 100.0 / m["work_pos_J"], places=9)

    def test_backward_travel_gives_nan_cot(self):
        roll, _, _ = _synthetic_rollout(T=100)
        roll.qpos[:, 0] = -roll.qpos[:, 0]          # run backwards
        m = energy_metrics(roll)
        self.assertTrue(np.isnan(m["cost_of_transport"]))
        self.assertLess(m["distance_m"], 0.0)


class TestSpectral(unittest.TestCase):
    def setUp(self):
        self.fs = 100.0
        self.t = np.arange(0, 10, 1.0 / self.fs)
        self.f0 = 2.5
        self.sine = np.sin(2 * np.pi * self.f0 * self.t)
        self.noise = np.random.RandomState(0).randn(len(self.t))

    def test_sine_is_concentrated(self):
        s = _spectral_stats(self.sine, self.fs)
        self.assertAlmostEqual(s["dom_freq"], self.f0, delta=0.5)
        self.assertLess(s["spectral_entropy"], 0.5)
        self.assertGreater(s["peak_frac"], 0.8)
        self.assertGreater(s["band_frac"], 0.9)

    def test_noise_is_broadband(self):
        s = _spectral_stats(self.noise, self.fs)
        self.assertGreater(s["spectral_entropy"], 0.7)
        self.assertLess(s["peak_frac"], 0.4)
        self.assertLess(s["band_frac"], 0.4)          # little power in stride band

    def test_autocorr_peak(self):
        peak, lag = _autocorr_peak(self.sine, self.fs)
        self.assertGreater(peak, 0.9)
        self.assertAlmostEqual(lag, 1.0 / self.f0, delta=0.05)
        npk, _ = _autocorr_peak(self.noise, self.fs)
        self.assertLess(npk, peak)

    def test_stride_cv_low_for_constant_frequency(self):
        st = _stride_stats(_phase(self.sine), self.t)
        self.assertGreater(st["n_strides"], 20)
        self.assertLess(st["stride_cv"], 0.05)
        self.assertAlmostEqual(st["stride_period_s"], 1.0 / self.f0, delta=0.05)

    def test_plv(self):
        shifted = np.sin(2 * np.pi * self.f0 * self.t + 0.7)
        self.assertGreater(_plv(self.sine, shifted), 0.95)
        n2 = np.random.RandomState(1).randn(len(self.t))
        self.assertLess(_plv(self.noise, n2), 0.5)

    def test_poincare_limit_cycle_tighter_than_noise(self):
        deriv = 2 * np.pi * self.f0 * np.cos(2 * np.pi * self.f0 * self.t)
        cycle_state = np.column_stack([self.sine, deriv])
        disp_cycle, ncr = _poincare_dispersion(
            self.sine, cycle_state, cycle_state.std(axis=0)
        )
        rng = np.random.RandomState(2)
        noise_state = rng.randn(len(self.t), 2)
        disp_noise, _ = _poincare_dispersion(
            self.noise, noise_state, noise_state.std(axis=0)
        )
        self.assertGreater(ncr, 15)
        self.assertLess(disp_cycle, disp_noise)
        self.assertLess(disp_cycle, 0.5)


def _periodic_rollout(T=800, dt=0.01, f0=2.5, amp=0.6, noise=0.0, seed=0):
    """A rollout whose leg joints are clean sinusoids at a stride frequency."""
    rng = np.random.RandomState(seed)
    nq = nv = 9
    ad = np.arange(3, 9)
    t = np.arange(T) * dt
    qpos = np.zeros((T, nq))
    qpos[:, 0] = 6.0 * t                                   # forward travel
    qvel = np.zeros((T, nv))
    for k, j in enumerate(ad):
        phase = 0.0 if j < 6 else np.pi                    # back vs front offset
        sig = amp * np.sin(2 * np.pi * f0 * t + phase)
        qpos[:, j] = sig + noise * rng.randn(T)
        qvel[:, j] = amp * 2 * np.pi * f0 * np.cos(2 * np.pi * f0 * t + phase)
    return CheetahRollout(
        dt=np.full(T, dt), time=np.cumsum(np.full(T, dt)), reward=np.ones(T),
        qpos=qpos, qvel=qvel, qfrc_act=np.zeros((T, nv)),
        action=np.zeros((T, 6)), speed=np.full(T, 6.0), foot_z=np.zeros((T, 2)),
        qpos0=np.zeros(nq), act_dof=ad,
        gear=np.array([120.0, 90.0, 60.0, 90.0, 60.0, 30.0]), mass=14.0, g=9.81,
    )


class TestGaitRollout(unittest.TestCase):
    def test_clean_periodic_gait_is_detected(self):
        g = gait_metrics(_periodic_rollout(f0=2.5), warmup_s=0.5)
        self.assertEqual(g["gait_detected"], 1.0)
        self.assertGreater(g["autocorr_peak"], 0.8)
        self.assertLess(g["stride_cv"], 0.1)
        self.assertLess(g["poincare_dispersion"], 0.5)
        self.assertAlmostEqual(g["stride_freq_hz"], 2.5, delta=0.5)
        self.assertGreater(g["peak_power_frac_mean"], 0.7)
        self.assertGreater(g["band_power_frac_mean"], 0.7)

    def test_standing_policy_is_not_a_gait(self):
        # Joints held essentially constant -> no stride cycles -> not detected.
        roll = _periodic_rollout(amp=0.0, noise=1e-4)
        roll.qpos[:, 0] = 0.0                              # not moving forward
        g = gait_metrics(roll, warmup_s=0.5)
        self.assertEqual(g["gait_detected"], 0.0)
        self.assertTrue(np.isnan(g["autocorr_peak"]))
        self.assertLess(g["band_power_frac_mean"], 0.5)

    def test_noisy_gait_less_regular_than_clean(self):
        clean = gait_metrics(_periodic_rollout(noise=0.0), warmup_s=0.5)
        noisy = gait_metrics(_periodic_rollout(noise=0.4, seed=3), warmup_s=0.5)
        self.assertGreaterEqual(clean["autocorr_peak"], noisy["autocorr_peak"])
        self.assertLessEqual(clean["stride_cv"], noisy["stride_cv"])
        self.assertLessEqual(clean["poincare_dispersion"], noisy["poincare_dispersion"])
        self.assertLessEqual(clean["spectral_entropy_mean"], noisy["spectral_entropy_mean"])


class TestPhasePortrait(unittest.TestCase):
    def test_shapes_and_reference(self):
        pd = phase_portrait_data(_periodic_rollout(f0=2.5), warmup_s=0.5)
        self.assertIsNotNone(pd)
        n = len(pd["time"])
        self.assertEqual(len(pd["theta"]), n)
        self.assertEqual(len(pd["theta_dot"]), n)
        self.assertIn(pd["joint"], _JOINT_NAMES)
        self.assertGreaterEqual(len(pd["cross_theta_dot"]), 3)

    def test_return_map_tighter_for_clean_gait(self):
        clean = phase_portrait_data(_periodic_rollout(noise=0.0), warmup_s=0.5)
        loose = phase_portrait_data(_periodic_rollout(noise=0.4, seed=3), warmup_s=0.5)
        # Crossing-velocity spread (2-D shadow of the return map) is tighter clean.
        self.assertLess(clean["cross_theta_dot"].std(), loose["cross_theta_dot"].std())


class TestCheetahEndToEnd(unittest.TestCase):
    def test_random_policy_rollout(self):
        from environment.dmc import DMCContinuousEnv

        env = DMCContinuousEnv(
            domain_name="cheetah", task_name="run", seed=0,
            raw_state_obs=True, time_sampling="uniform", dt=0.01,
            episode_duration=3.0,
        )
        out = evaluate_locomotion(
            lambda _o: env.action_space.sample(), env,
            n_episodes=1, warmup_s=1.0, max_steps=300,
        )
        agg = out["aggregate"]
        # Energy quantities must be finite and physical.
        self.assertGreater(agg["work_abs_J_mean"], 0.0)
        self.assertGreater(agg["duration_s_mean"], 0.0)
        self.assertTrue(np.isfinite(agg["control_cost_mean"]))
        self.assertTrue(np.isfinite(agg["mean_abs_power_W_mean"]))
        # Gait keys are present (values may be nan for a flailing random policy).
        for k in ("gait_detected_mean", "autocorr_peak_mean", "stride_cv_mean",
                  "poincare_dispersion_mean", "limb_plv_mean", "stride_freq_hz_mean",
                  "spectral_entropy_mean_mean", "band_power_frac_mean_mean"):
            self.assertIn(k, agg)

    def test_rollout_records_expected_shapes(self):
        from environment.dmc import DMCContinuousEnv

        env = DMCContinuousEnv(
            domain_name="cheetah", task_name="run", seed=1,
            raw_state_obs=True, time_sampling="uniform", dt=0.01,
            episode_duration=1.0,
        )
        roll = rollout_cheetah(lambda _o: env.action_space.sample(), env, max_steps=80)
        T = roll.qpos.shape[0]
        self.assertEqual(roll.qfrc_act.shape, (T, 9))
        self.assertEqual(roll.action.shape[1], 6)
        self.assertEqual(list(roll.act_dof), [3, 4, 5, 6, 7, 8])
        # qfrc_actuator on actuated dofs == gear * clip(action)
        expected = roll.gear * np.clip(roll.action, -1, 1)
        np.testing.assert_allclose(roll.qfrc_act[:, roll.act_dof], expected, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
