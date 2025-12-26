# environment/vec_env.py
import numpy as np
from typing import Callable, List, Tuple, Any, Dict, Optional


class VecContinuousEnv:
    def __init__(self, env_fns: List[Callable[[], Any]]):
        assert len(env_fns) > 0
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        # Delegate convenience attrs (used in some algos)
        self.dt_default = getattr(self.envs[0], "dt_default", 1.0)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obses, infos = [], []
        for i, env in enumerate(self.envs):
            s_i = None if seed is None else (seed + i)
            obs, info = env.reset(seed=s_i, options=options)
            obses.append(obs)
            infos.append(info)
        return np.stack(obses, axis=0), infos

    def step_dt(self, actions: np.ndarray):
        actions = np.asarray(actions)
        if actions.ndim == 1:
            actions = actions[None, :]  # (1, act_dim)

        obs_t_list, t_list = [], []
        rew_list, next_obs_list, next_t_list = [], [], []
        term_list, trunc_list, infos = [], [], []

        for i, env in enumerate(self.envs):
            obs_t, t, a, r, next_obs, next_t, terminated, truncated, info = env.step_dt(
                actions[i]
            )
            done = bool(terminated or truncated)

            # SB3-style terminal stash before auto-reset
            if done:
                info = dict(info)
                info["terminal_observation"] = next_obs
                info["terminal_next_t"] = next_t

                reset_obs, reset_info = env.reset()
                info["reset_info"] = reset_info
                next_obs = reset_obs
                next_t = 0.0

            obs_t_list.append(obs_t)
            t_list.append(t)
            rew_list.append(r)
            next_obs_list.append(next_obs)
            next_t_list.append(next_t)
            term_list.append(terminated)
            trunc_list.append(truncated)
            infos.append(info)

        return (
            np.stack(obs_t_list, axis=0),
            np.asarray(t_list, dtype=np.float32),
            actions,
            np.asarray(rew_list, dtype=np.float32),
            np.stack(next_obs_list, axis=0),
            np.asarray(next_t_list, dtype=np.float32),
            np.asarray(term_list, dtype=bool),
            np.asarray(trunc_list, dtype=bool),
            infos,
        )
