"""Deprecated: use `python -m quad_rl.training.evaluate` instead.

This is a thin shim forwarding to quad_rl.training.evaluate so existing
`eval_rollout.py` commands (see DEVELOPER.md) keep working rather than
silently breaking for anyone mid-run. evaluate.py generalizes this
script's two hardcoded assumptions (PPO-only, exactly one episode) via
--algo (or config.yaml) and --n-episodes, but the single-episode default
reproduces this script's original plot exactly.

Usage (unchanged from before Stage 1.7):
    python -m quad_rl.training.eval_rollout --checkpoint runs/ppo_quadhover/final_model.zip
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from quad_rl.training.evaluate import evaluate as _evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vecnormalize", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of using the policy mean.")
    return parser.parse_args()


def main() -> None:
    warnings.warn(
        "quad_rl.training.eval_rollout is deprecated; use "
        "`python -m quad_rl.training.evaluate` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    old_args = parse_args()

    # This script predates config.yaml (Stage 1.6+); its only checkpoints
    # are PPO ones, so --algo falls back to "ppo" explicitly here rather
    # than requiring the caller to pass it through this old flag surface.
    # --eval-env-config/--record-video weren't part of this script's
    # surface either, so they're left at evaluate.py's own defaults.
    eval_args = argparse.Namespace(
        checkpoint=old_args.checkpoint,
        vecnormalize=old_args.vecnormalize,
        output=old_args.output,
        algo="ppo",
        n_episodes=1,
        eval_env_config=None,
        record_video=False,
        seed=old_args.seed,
        stochastic=old_args.stochastic,
    )
    _evaluate(eval_args)


if __name__ == "__main__":
    main()
