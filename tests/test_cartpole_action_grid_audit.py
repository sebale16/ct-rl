import unittest

import numpy as np

try:
    from environment import DMCContinuousEnv
    from evaluations.cartpole_action_grid_audit import (
        action_grid,
        audit_state_set,
        mujoco_transition_reward,
        state_row,
        summarize_states,
    )
    from evaluations.hamiltonian_recovery import mujoco_transition

    HAVE_DMC = True
except Exception:
    HAVE_DMC = False


def _make_env(**overrides):
    kwargs = dict(
        domain_name="cartpole",
        task_name="swingup",
        seed=0,
        raw_state_obs=True,
        time_sampling="uniform",
        dt=0.01,
        physics_dt=0.002,
    )
    kwargs.update(overrides)
    return DMCContinuousEnv(**kwargs)


@unittest.skipUnless(HAVE_DMC, "dm_control not available")
class TestMujocoTransitionReward(unittest.TestCase):
    def setUp(self):
        self.env = _make_env()

    def _walk(self, n, seed=3):
        rng = np.random.default_rng(seed)
        O, A, R, NO = [], [], [], []
        self.env.reset(seed=seed)
        for _ in range(n):
            a = rng.uniform(-1, 1, size=(1,)).astype(np.float32)
            o, _, _, r, no, _, term, trunc, _ = self.env.step_dt(a)
            O.append(o), A.append(a), R.append(r), NO.append(no)
            if term or trunc:
                self.env.reset()
        f = lambda x: np.asarray(x, dtype=np.float32)
        return f(O), f(A), f(R), f(NO)

    def test_next_state_matches_mujoco_transition(self):
        O, A, _, _ = self._walk(8)
        x_ref = mujoco_transition(self.env, O, A, 0.01)
        x_new, rew = mujoco_transition_reward(self.env, O, A, 0.01)
        np.testing.assert_array_equal(x_new, x_ref)
        self.assertTrue(np.all(np.isfinite(rew)))
        self.assertTrue(np.all((rew >= 0.0) & (rew <= 1.0)))

    def test_reward_and_state_match_env_step(self):
        O, A, R, NO = self._walk(8)
        x_new, rew = mujoco_transition_reward(self.env, O, A, 0.01)
        # cartpole is constraint-free, so replaying (state, action, dt) must
        # reproduce the env's own transition and reward (float32 state storage
        # is the only noise source)
        np.testing.assert_allclose(x_new, NO, atol=1e-5)
        np.testing.assert_allclose(rew, R, atol=1e-5)

    def test_per_row_dt_array(self):
        O, A, _, _ = self._walk(4)
        dts = np.array([0.002, 0.01, 0.02, 0.03])
        x_arr, r_arr = mujoco_transition_reward(self.env, O, A, dts)
        for i, dt in enumerate(dts):
            x_i, r_i = mujoco_transition_reward(self.env, O[i:i + 1],
                                                A[i:i + 1], float(dt))
            np.testing.assert_array_equal(x_arr[i], x_i[0])
            self.assertEqual(r_arr[i], r_i[0])
        # longer horizons move the state further
        d_short = np.linalg.norm(x_arr[0] - O[0])
        d_long = np.linalg.norm(x_arr[3] - O[3])
        self.assertGreater(d_long, d_short)

    def test_live_physics_state_is_restored(self):
        self.env.reset(seed=5)
        before = self.env._raw_obs().copy()
        O = np.tile(np.array([[0.5, 2.0, -1.0, 3.0]], np.float32), (3, 1))
        A = np.full((3, 1), 0.7, np.float32)
        mujoco_transition_reward(self.env, O, A, 0.03)
        np.testing.assert_array_equal(self.env._raw_obs(), before)

    def test_action_grid_covers_box(self):
        grid = action_grid(self.env, 11)
        self.assertEqual(grid.shape, (11, 1))
        self.assertEqual(grid[0, 0], float(self.env.action_space.low[0]))
        self.assertEqual(grid[-1, 0], float(self.env.action_space.high[0]))


@unittest.skipUnless(HAVE_DMC, "dm_control not available")
class TestStateRowGeometry(unittest.TestCase):
    """state_row on synthetic tables with a known geometry: an oracle target
    peaked at a*=0.23 (off-grid sums -> no oracle ties on a 0.1-spaced grid)."""

    N_G, N_SAMPLES = 21, 2

    def _tab(self, t_l_fn, t_o_fn, q_fn, a_pi=0.0, h=0.1):
        grid = np.linspace(-1.0, 1.0, self.N_G)
        extras = np.array([a_pi, a_pi - h, a_pi + h, -0.4, 0.4])
        a = np.concatenate([grid, extras])[None, :]
        tab = {
            "t_l": t_l_fn(a), "t_o": t_o_fn(a), "q": q_fn(a),
            "s_score": None, "r": np.zeros_like(a), "a": a,
            "eps": t_l_fn(a) - t_o_fn(a), "v_cur": np.zeros(1),
        }
        return tab

    def test_perfect_agreement(self):
        f = lambda a: -((a - 0.23) ** 2)
        row = state_row(0, self._tab(f, f, f), self.N_G, self.N_SAMPLES,
                        dq_da=np.array([1.0]), k_top=5)
        self.assertEqual(row["spearman_tl_to"], 1.0)
        self.assertEqual(row["pairwise_agree"], 1.0)
        self.assertEqual(row["topk_overlap"], 1.0)
        self.assertEqual(row["argmax_disagree"], 0)
        self.assertEqual(row["regret_lgreedy"], 0.0)
        self.assertEqual(row["regret_qgreedy"], 0.0)
        self.assertEqual(row["best_a_oracle"], 0.2)  # nearest grid point to 0.23
        # slope of t_o at a_pi=0 is positive (peak to the right); dq_da=+1 agrees
        self.assertGreater(row["slope_o_pi"], 0.0)
        self.assertEqual(row["tslope_sign_agree"], 1.0)
        self.assertEqual(row["qslope_sign_agree"], 1.0)
        # regret of the policy action a_pi=0 under the oracle target
        self.assertAlmostEqual(row["regret_pi"], 0.23 ** 2 - 0.03 ** 2, places=9)

    def test_inverted_learned_target(self):
        f_o = lambda a: -((a - 0.23) ** 2)
        f_l = lambda a: +((a - 0.23) ** 2)
        row = state_row(0, self._tab(f_l, f_o, f_o), self.N_G, self.N_SAMPLES,
                        dq_da=np.array([1.0]), k_top=5)
        self.assertEqual(row["spearman_tl_to"], -1.0)
        self.assertEqual(row["pairwise_agree"], 0.0)
        self.assertEqual(row["argmax_disagree"], 1)
        # learned-greedy picks a=-1 (farthest from the peak): worst oracle action
        self.assertEqual(row["best_a_learned"], -1.0)
        self.assertEqual(row["regret_lgreedy_norm"], 1.0)
        self.assertEqual(row["tslope_sign_agree"], 0.0)
        # oracle-scored critic stays clean
        self.assertEqual(row["regret_qgreedy"], 0.0)
        self.assertEqual(row["spearman_q_to"], 1.0)

    def test_nonfinite_rows_are_flagged(self):
        f = lambda a: np.full_like(a, np.nan)
        row = state_row(0, self._tab(f, f, f), self.N_G, self.N_SAMPLES,
                        dq_da=np.array([np.nan]), k_top=5)
        self.assertEqual(row["finite_frac"], 0.0)
        self.assertNotIn("spearman_tl_to", row)
        summ = summarize_states([row])
        self.assertEqual(summ["n_states"], 1)
        self.assertTrue(np.isnan(summ["med_spearman_tl_to"]))


@unittest.skipUnless(HAVE_DMC, "dm_control not available")
class TestAuditSmoke(unittest.TestCase):
    """End-to-end audit on a freshly constructed (untrained) benchmark
    algorithm: exercises the exact code path the checkpointed audit runs."""

    N_SAMPLES = 2

    def test_audit_state_set_end_to_end(self):
        from evaluations.cartpole_critic_audit import build_algorithm

        algo, env = build_algorithm()
        obs, _ = env.reset(seed=11)
        S = [obs]
        for _ in range(6):
            _, _, _, _, no, _, term, trunc, _ = env.step_dt(
                np.array([0.3], np.float32))
            S.append(no)
            if term or trunc:
                no, _ = env.reset()
        S = np.asarray(S[:4], np.float32)

        grid = action_grid(env, 9)
        rows, summ, grid_mj = audit_state_set(
            algo, env, S, beta=float(algo.beta),
            dt_default=float(algo.dt_default), alpha=0.2, grid=grid,
            n_samples=self.N_SAMPLES, torch_seed=123, grad_h=0.05, k_top=3)

        self.assertEqual(len(rows), len(S))
        self.assertEqual(grid_mj[0].shape, (len(S) * len(grid), S.shape[1]))
        for row in rows:
            self.assertGreater(row["finite_frac"], 0.0)
            for key in ("spearman_tl_to", "pairwise_agree", "topk_overlap",
                        "regret_lgreedy_norm", "spearman_q_to",
                        "regret_qgreedy_norm", "spearman_s_to",
                        "regret_pi_norm", "regret_pi_sampled_norm",
                        "slope_q_pi", "r_to_range_ratio"):
                self.assertIn(key, row)
            for key in ("spearman_tl_to", "spearman_q_to", "spearman_s_to"):
                v = row[key]
                self.assertTrue(np.isnan(v) or -1.0 <= v <= 1.0, key)
            self.assertGreaterEqual(row["regret_pi"], -1e-6)
        self.assertIn("med_spearman_tl_to", summ)
        self.assertIn("p90_regret_lgreedy_norm", summ)
        self.assertEqual(summ["n_states"], len(S))

        # cached oracle grid reuse must reproduce the same metrics
        rows2, _, _ = audit_state_set(
            algo, env, S, beta=float(algo.beta),
            dt_default=float(algo.dt_default), alpha=0.2, grid=grid,
            n_samples=self.N_SAMPLES, torch_seed=123, grad_h=0.05, k_top=3,
            grid_mj=grid_mj)
        self.assertEqual(
            [r["spearman_tl_to"] for r in rows],
            [r["spearman_tl_to"] for r in rows2],
        )



if __name__ == "__main__":
    unittest.main()
