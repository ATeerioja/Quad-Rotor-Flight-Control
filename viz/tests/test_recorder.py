"""Smoke test for viz.recorder.TrajectoryRecorder."""
import gymnasium as gym
import numpy as np

import quad_rl.envs  # noqa: F401
from viz.recorder import TrajectoryRecorder


def test_recorder_logs_reset_plus_each_step_and_round_trips(tmp_path):
    env = TrajectoryRecorder(gym.make("QuadHover-v0"))
    obs, _ = env.reset(seed=0)
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            break

    out = tmp_path / "rollout.npz"
    env.save(out)
    data = np.load(out)

    assert data["positions"].shape == (len(env._positions), 3)
    assert data["quaternions"].shape == (len(env._quats), 4)
    assert data["times"][0] == 0.0
    assert float(data["dt"]) == env.unwrapped.dt
