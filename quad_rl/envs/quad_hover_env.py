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
import yaml
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from quad_rl.envs import dynamics

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


def _load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _physics_params_from_config(config: dict) -> dict:
    """Bridge default.yaml's nested physics section into the flat params
    dict dynamics.step() expects (see dynamics.py's module docstring)."""
    phys = config["physics"]
    return {
        "mass": phys["mass"],
        "inertia": np.array(
            [phys["inertia"]["ixx"], phys["inertia"]["iyy"], phys["inertia"]["izz"]]
        ),
        "arm_length": phys["arm_length"],
        "thrust_coefficient": phys["thrust_coefficient"],
        "drag_coefficient": phys["drag_coefficient"],
        "yaw_torque_coefficient": phys["yaw_torque_coefficient"],
        "gravity": phys["gravity"],
        "motor_time_constant": phys["motor_time_constant"],
    }


class QuadHoverEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config_path: str | Path | None = None, render_mode: str | None = None):
        self.render_mode = render_mode

        config = _load_config(config_path or DEFAULT_CONFIG_PATH)
        self.physics_params = _physics_params_from_config(config)
        self.dt = config["simulation"]["dt"]

        episode_cfg = config["episode"]
        self.max_steps = episode_cfg["max_steps"]
        self.crash_altitude = episode_cfg["crash_altitude"]
        self.max_tilt_deg = episode_cfg["max_tilt_deg"]
        self.bounding_box = episode_cfg["bounding_box"]

        # Reward weights, named here rather than buried in _compute_reward's
        # arithmetic, and read from default.yaml so they stay tunable
        # without touching code.
        reward_cfg = config["reward"]
        self.w_position = reward_cfg["position_error_weight"]
        self.w_angular_velocity = reward_cfg["angular_velocity_weight"]
        self.w_action = reward_cfg["action_magnitude_weight"]
        self.w_action_rate = reward_cfg["action_rate_weight"]
        self.hover_bonus = reward_cfg["hover_bonus"]
        self.hover_threshold = reward_cfg["hover_threshold"]
        self.crash_penalty = reward_cfg["crash_penalty"]

        spawn_cfg = config["spawn"]
        self.position_perturbation = spawn_cfg["position_perturbation"]
        self.velocity_perturbation = spawn_cfg["velocity_perturbation"]
        self.orientation_perturbation_deg = spawn_cfg["orientation_perturbation_deg"]
        self.target_region = spawn_cfg["target_region"]

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(dynamics.ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)

        self.state = np.zeros(dynamics.STATE_DIM)
        self.target = np.zeros(3)
        self.prev_action = np.zeros(dynamics.ACTION_DIM)
        self.elapsed_steps = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.target = np.array([0.0, 0.0, TARGET_ALTITUDE]) + self.np_random.uniform(
            -self.target_region, self.target_region, size=3
        )

        state = np.zeros(dynamics.STATE_DIM)
        state[dynamics.POS] = self.target + self.np_random.uniform(
            -self.position_perturbation, self.position_perturbation, size=3
        )
        state[dynamics.VEL] = self.np_random.uniform(
            -self.velocity_perturbation, self.velocity_perturbation, size=3
        )

        axis = self.np_random.normal(size=3)
        axis = axis / np.linalg.norm(axis)
        angle_deg = self.np_random.uniform(
            -self.orientation_perturbation_deg, self.orientation_perturbation_deg
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

        self.state = dynamics.step(self.state, motor_command, self.dt, self.physics_params)
        self.elapsed_steps += 1

        pos_error_norm = float(np.linalg.norm(self.state[dynamics.POS] - self.target))
        within_hover_threshold = pos_error_norm < self.hover_threshold

        # reward_components always has the same keys regardless of crash,
        # so callers (e.g. eval_rollout.py) can stack a fixed set of series
        # across an episode without ragged per-step dicts.
        reward, reward_components = self._compute_reward(action, pos_error_norm)
        crashed = self._check_crash()
        if crashed:
            reward = -self.crash_penalty
            reward_components = {**{k: 0.0 for k in reward_components}, "crash_penalty": -self.crash_penalty}
            terminated = True
        else:
            reward_components["crash_penalty"] = 0.0
            terminated = False

        truncated = self.elapsed_steps >= self.max_steps
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

    def _compute_reward(self, action: np.ndarray, pos_error_norm: float) -> tuple[float, dict]:
        omega = self.state[dynamics.OMEGA]

        components = {
            "position": -self.w_position * pos_error_norm,
            "angular_velocity": -self.w_angular_velocity * np.linalg.norm(omega),
            "action_magnitude": -self.w_action * np.linalg.norm(action),
            "action_rate": -self.w_action_rate * np.linalg.norm(action - self.prev_action),
            "hover_bonus": self.hover_bonus if pos_error_norm < self.hover_threshold else 0.0,
        }
        return sum(components.values()), components

    def _check_crash(self) -> bool:
        altitude = self.state[dynamics.POS][2]
        if altitude < self.crash_altitude:
            return True

        rotmat = dynamics.quat_to_rotmat(self.state[dynamics.QUAT])
        tilt_deg = np.degrees(np.arccos(np.clip(rotmat[2, 2], -1.0, 1.0)))
        if tilt_deg > self.max_tilt_deg:
            return True

        if np.any(np.abs(self.state[dynamics.POS]) > self.bounding_box):
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
