"""Standalone 3D trajectory animator for .npz files saved by
viz.recorder.TrajectoryRecorder. Draws the quadrotor as a rigid X-frame
cross (matching dynamics.py's X-configuration convention) rotated per
frame by the logged quaternion, plus a fading position trail.

Usage:
    python -m viz.animate --file rollout.npz --out rollout.mp4
    python -m viz.animate --file rollout.npz --out rollout.gif --fps 30 --trail 50

Example end-to-end (manual eval rollout, NOT SB3 training):
    import gymnasium as gym
    import quad_rl.envs  # noqa: F401
    from viz.recorder import TrajectoryRecorder

    env = TrajectoryRecorder(gym.make("QuadHover-v0"))
    obs, info = env.reset(seed=0)
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.save("rollout.npz")
    # then: python -m viz.animate --file rollout.npz --out rollout.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

from quad_rl.envs import dynamics

# Body-frame arm-tip directions for an X-configuration quad (dynamics.py
# lines 35-45): motors at +-45 deg from the body x-axis, front-right /
# rear-right / rear-left / front-left. Scaled by --arm-length at render
# time -- the .npz only carries kinematic state, not physics params, so
# this is a CLI-overridable visual constant, not a silent physics
# assumption baked into the recorded schema.
_SQRT2 = np.sqrt(2.0)
_BODY_ARM_DIRECTIONS = np.array([
    [1, -1, 0],
    [-1, -1, 0],
    [-1, 1, 0],
    [1, 1, 0],
]) / _SQRT2


def load_trajectory(path: Path) -> dict:
    data = np.load(path)
    return {
        "positions": data["positions"],
        "quaternions": data["quaternions"],
        "times": data["times"],
        "dt": float(data["dt"]),
    }


def _arm_tips_world(position: np.ndarray, quat: np.ndarray, arm_length: float) -> np.ndarray:
    rotmat = dynamics.quat_to_rotmat(quat)
    body_offsets = _BODY_ARM_DIRECTIONS * arm_length
    return position + body_offsets @ rotmat.T


def animate_trajectory(
    trajectory: dict,
    out_path: Path,
    trail_length: int = 50,
    fps: int = 30,
    arm_length: float = 0.17,
    writer: str | None = None,
) -> None:
    positions = trajectory["positions"]
    quats = trajectory["quaternions"]
    n_frames = len(positions)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    margin = 0.5
    lo, hi = positions.min(axis=0) - margin, positions.max(axis=0) + margin
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")

    trail_line, = ax.plot([], [], [], color="C0", alpha=0.6, linewidth=1.0)
    arm_lines = [ax.plot([], [], [], color="C1", linewidth=2.0)[0] for _ in range(2)]
    rotor_scatter = ax.scatter([], [], [], color="C3", s=30)

    def update(frame_idx: int):
        pos = positions[frame_idx]
        quat = quats[frame_idx]
        tips = _arm_tips_world(pos, quat, arm_length)  # (4, 3): FR, RR, RL, FL

        # Two crossed diagonal arms: FR<->RL, RR<->FL.
        arm_lines[0].set_data_3d([tips[0, 0], tips[2, 0]], [tips[0, 1], tips[2, 1]], [tips[0, 2], tips[2, 2]])
        arm_lines[1].set_data_3d([tips[1, 0], tips[3, 0]], [tips[1, 1], tips[3, 1]], [tips[1, 2], tips[3, 2]])
        rotor_scatter._offsets3d = (tips[:, 0], tips[:, 1], tips[:, 2])

        start = max(0, frame_idx - trail_length)
        trail = positions[start:frame_idx + 1]
        trail_line.set_data_3d(trail[:, 0], trail[:, 1], trail[:, 2])
        ax.set_title(f"t = {trajectory['times'][frame_idx]:.2f} s")
        return trail_line, *arm_lines, rotor_scatter

    writer_name = writer or ("pillow" if out_path.suffix.lower() == ".gif" else "ffmpeg")
    if not animation.writers.is_available(writer_name):
        plt.close(fig)
        raise SystemExit(
            f"matplotlib animation writer '{writer_name}' is not available "
            f"(needed to export {out_path.suffix}). Install ffmpeg (for .mp4) "
            f"or pillow (for .gif; `pip install pillow`), or pick the other format."
        )

    interval_ms = trajectory["dt"] * 1000.0
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=interval_ms, blit=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=writer_name, fps=fps)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, required=True, help="Input .npz from TrajectoryRecorder.save().")
    parser.add_argument("--out", type=Path, required=True, help="Output path; .mp4 or .gif, inferred from suffix.")
    parser.add_argument("--trail", type=int, default=50, help="Number of past positions in the fading trail.")
    parser.add_argument("--fps", type=int, default=30, help="Output video/gif frame rate.")
    parser.add_argument(
        "--arm-length", type=float, default=0.17,
        help="Visual arm length (m); matches base.yaml's default physics.arm_length.",
    )
    parser.add_argument(
        "--writer", choices=["ffmpeg", "pillow"], default=None,
        help="Override the writer inferred from --out's suffix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectory = load_trajectory(args.file)
    animate_trajectory(
        trajectory, args.out, trail_length=args.trail, fps=args.fps,
        arm_length=args.arm_length, writer=args.writer,
    )
    print(f"Saved animation to {args.out}")


if __name__ == "__main__":
    main()
