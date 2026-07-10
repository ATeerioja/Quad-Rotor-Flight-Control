"""Deprecated: use `python -m quad_rl.training.train --algo ppo` instead.

This is a thin shim forwarding to quad_rl.training.train so existing
`train_ppo.py` commands (see DEVELOPER.md) keep working rather than
silently breaking for anyone mid-run; it translates this script's old
PPO-specific flags into train.py's `algo.*` overrides.

Usage (unchanged from before Stage 1.6):
    python -m quad_rl.training.train_ppo --total-timesteps 300000
"""

from __future__ import annotations

import argparse
import warnings

from quad_rl.training.train import train as _train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--vec-env", choices=["subproc", "dummy"], default="subproc")
    parser.add_argument("--n-steps", type=int, default=2048, help="Rollout length per env, per update.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000, help="In environment timesteps.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", type=str, default="ppo_quadhover")
    return parser.parse_args()


def main() -> None:
    warnings.warn(
        "quad_rl.training.train_ppo is deprecated; use "
        "`python -m quad_rl.training.train --algo ppo` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    old_args = parse_args()

    train_args = argparse.Namespace(
        algo="ppo",
        env_config=None,
        algo_config=None,
        override=[
            f"algo.n_steps={old_args.n_steps}",
            f"algo.batch_size={old_args.batch_size}",
            f"algo.learning_rate={old_args.learning_rate}",
        ],
        total_timesteps=old_args.total_timesteps,
        n_envs=old_args.n_envs,
        vec_env=old_args.vec_env,
        checkpoint_freq=old_args.checkpoint_freq,
        seed=old_args.seed,
        run_name=old_args.run_name,
    )
    _train(train_args)


if __name__ == "__main__":
    main()
