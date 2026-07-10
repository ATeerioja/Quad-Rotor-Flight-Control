"""Records QuadHoverEnv rollouts to a .npz for offline 3D playback via
`python -m viz.animate`. Purely additive: a gym.Wrapper that reads
already-computed state after step()/reset() return -- never touches
quad_hover_env.py, dynamics.py, or reward code.

Usage (manual eval rollout only -- do NOT wrap training envs with this):
    import gymnasium as gym
    import quad_rl.envs  # noqa: F401  (registers QuadHover-v0)
    from viz.recorder import TrajectoryRecorder

    env = TrajectoryRecorder(gym.make("QuadHover-v0"))
    obs, info = env.reset(seed=0)
    terminated = truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()  # or a loaded policy's action
        obs, reward, terminated, truncated, info = env.step(action)
    env.save("rollout.npz")
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np

from quad_rl.envs import dynamics


class TrajectoryRecorder(gym.Wrapper):
    """Logs position/quaternion/time from self.unwrapped.state on every
    reset() and step(), without altering either. Call .save(path) after
    a rollout to write a .npz consumable by viz/animate.py."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._positions: list[np.ndarray] = []
        self._quats: list[np.ndarray] = []
        self._times: list[float] = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._positions.clear()
        self._quats.clear()
        self._times.clear()
        self._log(t=0.0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        t = self.unwrapped.elapsed_steps * self.unwrapped.dt
        self._log(t=t)
        return obs, reward, terminated, truncated, info

    def _log(self, t: float) -> None:
        state = self.unwrapped.state
        self._positions.append(state[dynamics.POS].copy())
        self._quats.append(state[dynamics.QUAT].copy())
        self._times.append(t)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            positions=np.array(self._positions),
            quaternions=np.array(self._quats),
            times=np.array(self._times),
            dt=np.array(self.unwrapped.dt),
        )
