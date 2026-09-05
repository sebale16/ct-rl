"""Episode-wise Acrobot parameter randomization for Experiment 3."""

from dataclasses import asdict
import json
from pathlib import Path

from gymnasium import spaces
import numpy as np

from controllers.xin_kaneda import (
    AcrobotParams, Gains, XinKanedaController, kd_min, kp_min,
)
from environment.acrobot_xk import PlantScales, swingup_xk
from environment.dmc import DMCContinuousEnv


PARAMETER_ORDER = ("mass1", "mass2", "length1", "length2")


def parameter_context(scales):
    """Dimensionless physical parameters, centered at the nominal plant."""
    scales = PlantScales.coerce(scales)
    return np.array([getattr(scales, name) - 1.0 for name in PARAMETER_ORDER],
                    dtype=np.float32)


class RandomizedAcrobotEnv(DMCContinuousEnv):
    """Sample a finite training configuration uniformly at each reset.

    Parameter draws and reset-state draws have independent RNG streams. Context
    is appended to both actor and critic observations only in the conditioned
    arm. Rebuilding the task also recalibrates rewards and termination bounds.
    """

    def __init__(self, configurations, *, conditioned=False, seed=0,
                 episode_log=None, **env_kwargs):
        if not configurations:
            raise ValueError("at least one configuration is required")
        self.configurations = tuple(dict(config) for config in configurations)
        for config in self.configurations:
            PlantScales.coerce(config["scales"])
        self.conditioned = bool(conditioned)
        self._configuration_rng = np.random.default_rng(seed)
        self._reset_rng = np.random.default_rng(seed + 1000003)
        self.episode_log = Path(episode_log) if episode_log else None
        self.episode_index = 0
        self.current_configuration = self.configurations[0]
        self._task_template = dict(env_kwargs.pop("task_kwargs", {}))
        if "plant_scales" in self._task_template:
            raise ValueError("plant_scales must come from the configuration list")
        if float(self._task_template.get("damping", 0)) != 0:
            raise ValueError("Experiment 3 requires conservative (zero damping) plants")
        env_kwargs.pop("n_envs", None)
        env_kwargs.pop("raw_state_obs", None)
        self._environment_kwargs = dict(env_kwargs.get("environment_kwargs", {}))
        self._environment_kwargs.setdefault("flat_observation", True)
        task = {**self._task_template,
                "plant_scales": self.current_configuration["scales"]}
        super().__init__("acrobot", "swingup-xk", seed=seed, raw_state_obs=True,
                         task_kwargs=task, **env_kwargs)
        if self.conditioned:
            self.observation_space = spaces.Box(-np.inf, np.inf, (8,), np.float32)

    def _raw_obs(self):
        state = super()._raw_obs()
        if self.conditioned:
            return np.concatenate((state, parameter_context(
                self.current_configuration["scales"])))
        return state

    def _reset_physics(self, *, seed, options):
        if seed is not None:
            self._configuration_rng = np.random.default_rng(seed)
            self._reset_rng = np.random.default_rng(seed + 1000003)
        index = int(self._configuration_rng.integers(len(self.configurations)))
        config = self.configurations[index]
        reset_seed = int(self._reset_rng.integers(2**31))
        with self._drift_rollout_lock:
            self._close_drift_rollout()
            if config != self.current_configuration:
                replacement = swingup_xk(
                    **self._task_template, plant_scales=config["scales"],
                    random=reset_seed, environment_kwargs=self._environment_kwargs)
                replacement._step_limit = self.max_steps or int(1e9)
                replacement.physics.model.opt.timestep = self.physics_dt
                self._env.close()
                self._env = replacement
                self.current_configuration = config
            obs, info = super()._reset_physics(seed=reset_seed, options=options)
        info["configuration_id"] = config["id"]
        info["plant_scales"] = asdict(PlantScales.coerce(config["scales"]))
        if self.episode_log:
            with self.episode_log.open("a") as stream:
                stream.write(json.dumps({"episode": self.episode_index,
                                         "reset_seed": reset_seed,
                                         "configuration": config}) + "\n")
        self.episode_index += 1
        return obs, info

    def dynamics_terms(self, obs, action):
        raise NotImplementedError("Experiment 3 baselines use model-free CT-SAC")


class ConfigurationDemonstrator:
    """Warm-start controller rebuilt for the live plant, without replay imitation.

    Keep nominal gains when admissible; otherwise raise k_D/k_P to 1% above
    the plant-specific floors. Torque capacity remains fixed, so success is
    still empirical. The reward's gains remain those of the chosen base mode.
    """

    def __init__(self, env):
        self.env = env
        self._task = None
        self.controller = None

    def __call__(self, observation):
        task = self.env._env.task
        if task is not self._task:
            params = AcrobotParams.from_physics(self.env._env.physics)
            gains = Gains(k_v=task.k_v,
                          k_d=max(task.k_d, 1.01 * kd_min(params)),
                          k_p=max(task.k_p, 1.01 * kp_min(params)))
            self.controller = XinKanedaController(params, gains)
            self._task = task
        return self.controller(np.asarray(observation).reshape(-1)[:4])
