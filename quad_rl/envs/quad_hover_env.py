"""Gymnasium environment: quadrotor hover-at-target-position task.

Wraps quad_rl.envs.dynamics with a Gymnasium Env interface and registers
it as "QuadHover-v0" via gymnasium.register so it can be created with
gym.make("QuadHover-v0").

Observation (19 dims total, in this order -- document the layout here
since downstream code, e.g. VecNormalize stats or a policy's input
layer, will depend on it):
    position error to target      (3)  -- (state.position - target) / POSITION_ERROR_SCALE
    linear velocity                (3)  -- state.velocity / LINEAR_VELOCITY_SCALE
    orientation, 6D rotation rep   (6)  -- first two columns of the body->world
                                           rotation matrix, concatenated. Avoids
                                           the quaternion double-cover problem and
                                           Euler gimbal lock; standard choice in
                                           recent quadrotor RL work (Zhou et al.,
                                           "On the Continuity of Rotation
                                           Representations in Neural Networks").
                                           Already unit-scale (rotation matrix
                                           entries are in [-1, 1]) -- no separate
                                           normalization constant needed.
    angular velocity               (3)  -- state.angular_velocity / ANGULAR_VELOCITY_SCALE
    previous action                (4)  -- already in [-1, 1], the action space itself

reset(): spawns near a randomly sampled target position, with small
random position/velocity/orientation perturbation (not domain
randomization yet -- just enough variety that the policy can't memorize
a single trajectory).

Episode length: capped at max_steps (default 1000, i.e. 10 s at 100 Hz)
as a timeout (truncation), tracked internally so the cap holds even for
direct instantiation without gym.make()'s TimeLimit wrapper.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from quad_rl.config.loader import load_config
from quad_rl.envs import dynamics
from quad_rl.envs.rewards import HoverBonus, RewardFunction, StepContext

DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "default.yaml"

OBS_DIM = 19

# Observation normalization -- chosen to bring each component to roughly
# unit scale over the range the vehicle is expected to operate in before
# a crash/timeout ends the episode. These are fixed scaling constants,
# not physical parameters, so they live here rather than in default.yaml.
POSITION_ERROR_SCALE = 2.0    # m
LINEAR_VELOCITY_SCALE = 5.0   # m/s
ANGULAR_VELOCITY_SCALE = 5.0  # rad/s

# Height the target is centered around. Targets are sampled near this
# altitude (see reset()) rather than near z=0, so a typical target isn't
# sitting right at the ground-crash boundary.
TARGET_ALTITUDE = 1.5  # m


class QuadHoverEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config_path: str | Path | None = None,
        overrides: list[str] | None = None,
        render_mode: str | None = None,
    ):
        self.render_mode = render_mode

        self.env_config = load_config(config_path or DEFAULT_CONFIG_PATH, overrides=overrides)

        # Cached separately from env_config (rather than read through it
        # each call) since these are read on every dynamics.step() call, in
        # the hot per-step path.
        self.physics_params = self.env_config.physics.as_params()
        self.dt = self.env_config.simulation.dt

        self.reward_function = RewardFunction.from_config(self.env_config.reward)
        # within_hover_threshold is a general diagnostic (read by
        # train_ppo.py's HoverFractionCallback), not itself a reward term,
        # so it's derived from whichever hover_bonus term is configured --
        # if the config has none, it just reports False rather than raising.
        self._hover_threshold = next(
            (t.threshold for t in self.reward_function.terms if isinstance(t, HoverBonus)),
            None,
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(dynamics.ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)

        self.state = np.zeros(dynamics.STATE_DIM)
        self.target = np.zeros(3)
        self.prev_action = np.zeros(dynamics.ACTION_DIM)
        self.elapsed_steps = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        spawn_cfg = self.env_config.spawn
        self.target = np.array([0.0, 0.0, TARGET_ALTITUDE]) + self.np_random.uniform(
            -spawn_cfg.target_region, spawn_cfg.target_region, size=3
        )

        state = np.zeros(dynamics.STATE_DIM)
        state[dynamics.POS] = self.target + self.np_random.uniform(
            -spawn_cfg.position_perturbation, spawn_cfg.position_perturbation, size=3
        )
        state[dynamics.VEL] = self.np_random.uniform(
            -spawn_cfg.velocity_perturbation, spawn_cfg.velocity_perturbation, size=3
        )

        axis = self.np_random.normal(size=3)
        axis = axis / np.linalg.norm(axis)
        angle_deg = self.np_random.uniform(
            -spawn_cfg.orientation_perturbation_deg, spawn_cfg.orientation_perturbation_deg
        )
        rotvec = axis * np.radians(angle_deg)
        x, y, z, w = Rotation.from_rotvec(rotvec).as_quat()
        state[dynamics.QUAT] = [w, x, y, z]

        self.state = state
        self.prev_action = np.zeros(dynamics.ACTION_DIM)
        self.elapsed_steps = 0

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        motor_command = (action + 1.0) / 2.0  # [-1, 1] -> [0, 1]

        prev_state = self.state
        self.state = dynamics.step(self.state, motor_command, self.dt, self.physics_params)
        self.elapsed_steps += 1

        pos_error_norm = float(np.linalg.norm(self.state[dynamics.POS] - self.target))
        within_hover_threshold = (
            self._hover_threshold is not None and pos_error_norm < self._hover_threshold
        )

        crashed = self._check_crash()
        ctx = StepContext(
            state=self.state,
            prev_state=prev_state,
            action=action,
            prev_action=self.prev_action,
            target=self.target,
            dt=self.dt,
            crashed=crashed,
        )
        # reward_components always has the same keys regardless of crash,
        # so callers (e.g. eval_rollout.py) can stack a fixed set of series
        # across an episode without ragged per-step dicts.
        reward, reward_components = self.reward_function(ctx)
        terminated = crashed

        truncated = self.elapsed_steps >= self.env_config.episode.max_steps
        self.prev_action = action

        # Exposed for training/eval tooling: train_ppo.py's custom TensorBoard
        # metric averages "within_hover_threshold" over each episode, and
        # eval_rollout.py plots "reward_components" without needing to
        # duplicate the formula above.
        info = {
            "within_hover_threshold": within_hover_threshold,
            "reward_components": reward_components,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def _check_crash(self) -> bool:
        episode_cfg = self.env_config.episode
        altitude = self.state[dynamics.POS][2]
        if altitude < episode_cfg.crash_altitude:
            return True

        rotmat = dynamics.quat_to_rotmat(self.state[dynamics.QUAT])
        tilt_deg = np.degrees(np.arccos(np.clip(rotmat[2, 2], -1.0, 1.0)))
        if tilt_deg > episode_cfg.max_tilt_deg:
            return True

        if np.any(np.abs(self.state[dynamics.POS]) > episode_cfg.bounding_box):
            return True

        return False

    def _get_obs(self) -> np.ndarray:
        pos_error = self.state[dynamics.POS] - self.target
        velocity = self.state[dynamics.VEL]
        omega = self.state[dynamics.OMEGA]
        rotmat = dynamics.quat_to_rotmat(self.state[dynamics.QUAT])
        rot6d = np.concatenate([rotmat[:, 0], rotmat[:, 1]])

        obs = np.concatenate([
            pos_error / POSITION_ERROR_SCALE,
            velocity / LINEAR_VELOCITY_SCALE,
            rot6d,
            omega / ANGULAR_VELOCITY_SCALE,
            self.prev_action,
        ])
        return obs.astype(np.float32)


gym.register(
    id="QuadHover-v0",
    entry_point="quad_rl.envs.quad_hover_env:QuadHoverEnv",
)
