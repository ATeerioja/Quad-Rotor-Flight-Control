"""One-command, end-to-end generator for a 3D rollout animation: records a
QuadHoverEnv rollout with viz.recorder.TrajectoryRecorder (random actions
by default, or a trained SB3 checkpoint via --checkpoint), saves the
intermediate .npz, then renders it with viz.animate -- all in a single
run, no separate record/render steps.

Usage:
    python -m viz.generate_animation --out rollout.mp4
    python -m viz.generate_animation --out rollout.gif \\
        --checkpoint runs/my_run/final_model.zip \\
        --vecnormalize runs/my_run/final_vecnormalize.pkl --algo ppo

See docs/trajectory_visualization.md for a step-by-step guide.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym

import quad_rl.envs  # noqa: F401  (registers QuadHover-v0)
from viz.animate import animate_trajectory, load_trajectory
from viz.recorder import TrajectoryRecorder


def _load_checkpoint_policy(checkpoint: Path, vecnormalize: Path, algo: str):
    """Loads a trained SB3 checkpoint the same way quad_rl.training.evaluate
    does, returning a callable (obs) -> action. Reuses evaluate.py's own
    public loader rather than duplicating VecNormalize/ALGO_REGISTRY plumbing."""
    from quad_rl.config.loader import load_config
    from quad_rl.envs.quad_hover_env import DEFAULT_CONFIG_PATH
    from quad_rl.training.evaluate import load_policy

    env_config = load_config(DEFAULT_CONFIG_PATH)
    model, vecnorm = load_policy(checkpoint, vecnormalize, algo, env_config)

    def policy(obs):
        normalized_obs = vecnorm.normalize_obs(obs[None, :])
        action, _ = model.predict(normalized_obs, deterministic=True)
        return action[0]

    return policy, env_config


def record_rollout(
    out_npz: Path,
    checkpoint: Path | None,
    vecnormalize: Path | None,
    algo: str | None,
    seed: int,
    max_steps: int | None,
) -> None:
    env_config = None
    if checkpoint is not None:
        if vecnormalize is None or algo is None:
            raise SystemExit("--checkpoint requires both --vecnormalize and --algo.")
        policy, env_config = _load_checkpoint_policy(checkpoint, vecnormalize, algo)
    else:
        policy = None  # falls back to env.action_space.sample() below

    env = TrajectoryRecorder(gym.make("QuadHover-v0", env_config=env_config))
    obs, info = env.reset(seed=seed)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and (max_steps is None or steps < max_steps):
        action = policy(obs) if policy is not None else env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

    env.save(out_npz)
    print(f"Recorded {steps} steps -> {out_npz}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("rollout.mp4"), help="Output animation path (.mp4 or .gif).")
    parser.add_argument(
        "--npz", type=Path, default=None,
        help="Where to save the intermediate .npz (default: --out with a .npz suffix).",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="SB3 checkpoint .zip to fly instead of a random policy.")
    parser.add_argument("--vecnormalize", type=Path, default=None, help="VecNormalize .pkl matching --checkpoint.")
    parser.add_argument("--algo", default=None, choices=["ppo", "sac", "td3"], help="Algorithm that trained --checkpoint.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None, help="Stop early after this many steps (default: run the full episode).")
    parser.add_argument("--trail", type=int, default=50, help="Number of past positions in the fading trail.")
    parser.add_argument("--fps", type=int, default=30, help="Output video/gif frame rate.")
    parser.add_argument("--arm-length", type=float, default=0.17, help="Visual arm length (m).")
    parser.add_argument("--writer", choices=["ffmpeg", "pillow"], default=None, help="Override the writer inferred from --out's suffix.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = args.npz or args.out.with_suffix(".npz")

    record_rollout(npz_path, args.checkpoint, args.vecnormalize, args.algo, args.seed, args.max_steps)

    trajectory = load_trajectory(npz_path)
    animate_trajectory(
        trajectory, args.out, trail_length=args.trail, fps=args.fps,
        arm_length=args.arm_length, writer=args.writer,
    )
    print(f"Saved animation to {args.out}")


if __name__ == "__main__":
    main()
