"""Evaluate a trained PPO checkpoint on QuadHover-v0 and plot a rollout.

Loads a saved SB3 PPO model and its matching VecNormalize statistics,
runs one episode with the (deterministic, by default) policy on a plain
(non-vectorized) QuadHoverEnv, and plots position vs. target per axis
plus the reward components over time. Saves the figure to disk.

Usage:
    python -m quad_rl.training.eval_rollout --checkpoint runs/ppo_quadhover/final_model.zip
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from quad_rl.envs import dynamics
import quad_rl.envs  # noqa: F401  (registers QuadHover-v0)

CHECKPOINT_STEPS_RE = re.compile(r"^(?P<prefix>.+)_(?P<steps>\d+)_steps$")


def _infer_vecnormalize_path(checkpoint_path: Path) -> Path | None:
    """Match train_ppo.py's naming conventions:
    final_model.zip <-> final_vecnormalize.pkl
    <prefix>_<n>_steps.zip <-> <prefix>_vecnormalize_<n>_steps.pkl
    """
    stem = checkpoint_path.stem
    if stem == "final_model":
        candidate = checkpoint_path.with_name("final_vecnormalize.pkl")
        return candidate if candidate.exists() else None

    match = CHECKPOINT_STEPS_RE.match(stem)
    if match:
        candidate = checkpoint_path.with_name(
            f"{match['prefix']}_vecnormalize_{match['steps']}_steps.pkl"
        )
        return candidate if candidate.exists() else None

    return None


def load_policy(checkpoint_path: Path, vecnormalize_path: Path) -> tuple[PPO, VecNormalize]:
    model = PPO.load(str(checkpoint_path))

    # A throwaway single-env VecEnv, needed only so VecNormalize.load can
    # validate observation/action space shapes and hold the loaded running
    # statistics -- it is never stepped; the actual rollout runs on a
    # plain QuadHoverEnv below so we get direct, unambiguous state access
    # instead of unpicking VecEnv auto-reset/terminal-observation behavior.
    dummy_venv = DummyVecEnv([lambda: Monitor(gym.make("QuadHover-v0"))])
    vecnormalize = VecNormalize.load(str(vecnormalize_path), dummy_venv)
    vecnormalize.training = False
    return model, vecnormalize


def run_rollout(model: PPO, vecnormalize: VecNormalize, seed: int, deterministic: bool) -> dict:
    env = gym.make("QuadHover-v0")
    obs, _ = env.reset(seed=seed)
    raw_env = env.unwrapped

    times, positions, targets, rewards, actions = [], [], [], [], []
    reward_components: dict[str, list[float]] = {}

    t = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        normalized_obs = vecnormalize.normalize_obs(obs[None, :])
        action, _ = model.predict(normalized_obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action[0])
        t += raw_env.dt

        times.append(t)
        positions.append(raw_env.state[dynamics.POS].copy())
        targets.append(raw_env.target.copy())
        rewards.append(reward)
        actions.append(action[0].copy())
        for key, value in info["reward_components"].items():
            reward_components.setdefault(key, []).append(value)

    env.close()
    return {
        "times": np.array(times),
        "positions": np.array(positions),
        "targets": np.array(targets),
        "rewards": np.array(rewards),
        "actions": np.array(actions),
        "reward_components": {k: np.array(v) for k, v in reward_components.items()},
        "terminated": terminated,
        "truncated": truncated,
    }


def plot_rollout(result: dict, output_path: Path) -> None:
    times = result["times"]
    positions = result["positions"]
    targets = result["targets"]

    fig, axes = plt.subplots(5, 1, figsize=(10, 16), sharex=True)

    axis_labels = ["x", "y", "z"]
    for i, label in enumerate(axis_labels):
        ax = axes[i]
        ax.plot(times, positions[:, i], label=f"position {label}", color="C0")
        ax.plot(times, targets[:, i], label=f"target {label}", color="C1", linestyle="--")
        ax.set_ylabel(f"{label} (m)")
        ax.legend(loc="upper right", fontsize="small")
        ax.grid(alpha=0.3)

    action_ax = axes[3]
    actions = result["actions"]
    for i in range(actions.shape[1]):
        action_ax.plot(times, actions[:, i], label=f"motor {i + 1}", linewidth=1.0)
    action_ax.set_ylabel("action [-1, 1]")
    action_ax.set_ylim(-1.05, 1.05)
    action_ax.legend(loc="upper right", fontsize="small", ncol=4)
    action_ax.grid(alpha=0.3)

    reward_ax = axes[4]
    for key, values in result["reward_components"].items():
        reward_ax.plot(times, values, label=key)
    reward_ax.plot(times, result["rewards"], label="total", color="black", linewidth=1.5)
    reward_ax.set_ylabel("reward")
    reward_ax.set_xlabel("time (s)")
    reward_ax.legend(loc="lower right", fontsize="small", ncol=2)
    reward_ax.grid(alpha=0.3)

    outcome = "terminated (crash)" if result["terminated"] else "truncated (timeout)"
    fig.suptitle(f"QuadHover eval rollout -- {len(times)} steps, {outcome}")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vecnormalize", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of using the policy mean.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    vecnormalize_path = args.vecnormalize or _infer_vecnormalize_path(args.checkpoint)
    if vecnormalize_path is None:
        raise SystemExit(
            f"Could not infer a VecNormalize stats file next to {args.checkpoint}. "
            "Pass --vecnormalize explicitly."
        )

    output_path = args.output or args.checkpoint.parent / f"{args.checkpoint.stem}_eval.png"

    model, vecnormalize = load_policy(args.checkpoint, vecnormalize_path)
    result = run_rollout(model, vecnormalize, seed=args.seed, deterministic=not args.stochastic)
    plot_rollout(result, output_path)


if __name__ == "__main__":
    main()
