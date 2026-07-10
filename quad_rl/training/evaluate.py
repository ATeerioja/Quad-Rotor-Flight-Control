"""Evaluate a trained model on QuadHover-v0 over N episodes and report
aggregate metrics (Stage 1.7). Loads through
quad_rl.training.algorithms.ALGO_REGISTRY, so it isn't tied to PPO the way
eval_rollout.py was, and --algo is read back from the config.yaml that
train.py (Stage 1.6) writes into the run directory when available,
falling back to the --algo flag for older checkpoints that predate it
(e.g. Stage 0 runs).

Usage:
    python -m quad_rl.training.evaluate --checkpoint runs/my_run/final_model.zip
    python -m quad_rl.training.evaluate --checkpoint runs/my_run/final_model.zip --n-episodes 20
    python -m quad_rl.training.evaluate --checkpoint runs/my_run/final_model.zip --eval-env-config quad_rl/envs/configs/hover.yaml -o ...

--n-episodes 1 (the default) keeps the original 5-panel single-rollout
plot (position vs. target per axis, action trace, reward components).
For N>1, a summary table is printed and a single plot overlays all N
distance-to-target traces with a mean/+-sigma band.

--eval-env-config lets you evaluate under a different env config than the
one training used (the sim-to-sim generalization check) -- without it,
evaluation defaults to the run's own training-time config (from
config.yaml), not QuadHoverEnv's global default, so evaluation is
faithful to what was actually trained unless you deliberately ask
otherwise.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import yaml
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import quad_rl.envs  # noqa: F401  (registers QuadHover-v0)
from quad_rl.config.loader import load_config
from quad_rl.config.schema import EnvConfig
from quad_rl.envs import dynamics
from quad_rl.envs.quad_hover_env import DEFAULT_CONFIG_PATH
from quad_rl.envs.rewards import HoverBonus, RewardFunction
from quad_rl.training.algorithms import ALGO_REGISTRY

CHECKPOINT_STEPS_RE = re.compile(r"^(?P<prefix>.+)_(?P<steps>\d+)_steps$")


def _infer_vecnormalize_path(checkpoint_path: Path) -> Path | None:
    """Match train.py's naming conventions:
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


def _find_run_dir(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def _load_run_config(checkpoint_path: Path) -> dict | None:
    """Read back the config.yaml train.py writes into the run dir.
    Returns None if it doesn't exist (e.g. a Stage 0 checkpoint, trained
    before this file existed) -- callers fall back to CLI flags."""
    config_path = _find_run_dir(checkpoint_path) / "config.yaml"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        return yaml.safe_load(f)


def _hover_threshold_from_config(env_config: EnvConfig) -> float | None:
    """Same derivation QuadHoverEnv itself uses (see its __init__): find
    whichever hover_bonus term is configured and read its threshold, or
    None if the reward config doesn't include one."""
    reward_function = RewardFunction.from_config(env_config.reward)
    return next(
        (t.threshold for t in reward_function.terms if isinstance(t, HoverBonus)),
        None,
    )


def load_policy(checkpoint_path: Path, vecnormalize_path: Path, algo: str, env_config: EnvConfig):
    spec = ALGO_REGISTRY[algo]
    model = spec.cls.load(str(checkpoint_path))

    # A throwaway single-env VecEnv, needed only so VecNormalize.load can
    # validate observation/action space shapes and hold the loaded running
    # statistics -- it is never stepped; the actual rollout runs on a
    # plain QuadHoverEnv below so we get direct, unambiguous state access
    # instead of unpicking VecEnv auto-reset/terminal-observation behavior.
    dummy_venv = DummyVecEnv([lambda: Monitor(gym.make("QuadHover-v0", env_config=env_config))])
    vecnormalize = VecNormalize.load(str(vecnormalize_path), dummy_venv)
    vecnormalize.training = False
    return model, vecnormalize


def run_rollout(model, vecnormalize: VecNormalize, env_config: EnvConfig, seed: int, deterministic: bool) -> dict:
    env = gym.make("QuadHover-v0", env_config=env_config)
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
        "dt": raw_env.dt,
    }


def compute_aggregate_metrics(results: list[dict], hover_threshold: float | None) -> dict:
    """success rate: reached and held within hover_threshold for the
    final 2 seconds of the episode (a single "all steps in the final
    window are within threshold" check implies both -- being within
    threshold for the whole window necessarily means it was reached by
    the start of that window). None if no hover_bonus term is configured
    (there's nothing to measure "success" against)."""
    n = len(results)
    terminal_errors = []
    lengths = []
    smoothness_values = []
    successes = []
    crashed_count = 0

    for r in results:
        pos_error_norm = np.linalg.norm(r["positions"] - r["targets"], axis=1)
        terminal_errors.append(float(pos_error_norm[-1]))
        lengths.append(len(pos_error_norm))
        if r["terminated"]:
            crashed_count += 1

        if len(r["actions"]) > 1:
            diffs = np.linalg.norm(np.diff(r["actions"], axis=0), axis=1)
            smoothness_values.extend(diffs.tolist())

        if hover_threshold is not None:
            window = max(1, int(round(2.0 / r["dt"])))
            tail = pos_error_norm[-window:]
            successes.append(bool(np.all(tail < hover_threshold)))

    return {
        "n_episodes": n,
        "success_rate": float(np.mean(successes)) if hover_threshold is not None else None,
        "mean_terminal_position_error": float(np.mean(terminal_errors)),
        "crash_rate": crashed_count / n,
        "mean_episode_length": float(np.mean(lengths)),
        "action_smoothness": float(np.mean(smoothness_values)) if smoothness_values else float("nan"),
    }


def _format_summary(metrics: dict) -> str:
    success_rate = metrics["success_rate"]
    success_str = f"{success_rate:.2%}" if success_rate is not None else "N/A (no hover_bonus term configured)"
    return "\n".join([
        f"Aggregate metrics over {metrics['n_episodes']} episode(s):",
        f"  success rate (held within hover_threshold, final 2s): {success_str}",
        f"  mean terminal position error:                         {metrics['mean_terminal_position_error']:.4f} m",
        f"  crash rate:                                           {metrics['crash_rate']:.2%}",
        f"  mean episode length:                                  {metrics['mean_episode_length']:.1f} steps",
        f"  action smoothness (mean ||a_t - a_(t-1)||):            {metrics['action_smoothness']:.4f}",
    ])


def plot_single_episode(result: dict, output_path: Path) -> None:
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


def plot_multi_episode_summary(results: list[dict], metrics: dict, output_path: Path) -> None:
    dt = results[0]["dt"]
    max_len = max(len(r["times"]) for r in results)
    time_axis = np.arange(max_len) * dt

    # NaN-padded so episodes of different lengths (crash vs. timeout) each
    # contribute to the mean/sigma band only while they're still running,
    # rather than truncating everything to the shortest episode.
    padded = np.full((len(results), max_len), np.nan)
    for i, r in enumerate(results):
        pos_error_norm = np.linalg.norm(r["positions"] - r["targets"], axis=1)
        padded[i, : len(pos_error_norm)] = pos_error_norm

    mean = np.nanmean(padded, axis=0)
    std = np.nanstd(padded, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, r in enumerate(results):
        pos_error_norm = np.linalg.norm(r["positions"] - r["targets"], axis=1)
        ax.plot(r["times"], pos_error_norm, color="C0", alpha=0.2, linewidth=0.8)
    ax.plot(time_axis, mean, color="C0", linewidth=2.0, label="mean")
    ax.fill_between(time_axis, mean - std, mean + std, color="C0", alpha=0.25, label="+-1 sigma")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("distance to target (m)")
    ax.set_title(f"QuadHover eval -- {len(results)} episodes")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    fig.text(0.02, -0.02, _format_summary(metrics), fontsize=9, family="monospace", va="top")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vecnormalize", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--algo", choices=sorted(ALGO_REGISTRY), default=None,
        help="Falls back to this only if the run's config.yaml (written by train.py) isn't found.",
    )
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument(
        "--eval-env-config", type=Path, default=None,
        help="Evaluate under a different env config than training used (sim-to-sim generalization "
        "check). Defaults to the run's own training-time config from config.yaml, or QuadHoverEnv's "
        "own default if that isn't available.",
    )
    parser.add_argument(
        "--record-video", action="store_true",
        help="Not yet implemented -- reserved for a future stage.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of using the policy mean.")
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> None:
    vecnormalize_path = args.vecnormalize or _infer_vecnormalize_path(args.checkpoint)
    if vecnormalize_path is None:
        raise SystemExit(
            f"Could not infer a VecNormalize stats file next to {args.checkpoint}. "
            "Pass --vecnormalize explicitly."
        )

    run_config = _load_run_config(args.checkpoint)

    algo = (run_config or {}).get("algo") or args.algo
    if algo is None:
        raise SystemExit(
            f"Could not determine the algorithm: no config.yaml found next to {args.checkpoint} "
            "(an older checkpoint?) and --algo wasn't given."
        )
    if algo not in ALGO_REGISTRY:
        raise SystemExit(f"Unknown algo {algo!r}; choices: {sorted(ALGO_REGISTRY)}")

    if args.eval_env_config is not None:
        env_config = load_config(args.eval_env_config)
    elif run_config is not None and "env" in run_config:
        env_config = EnvConfig.from_dict(run_config["env"])
    else:
        env_config = load_config(DEFAULT_CONFIG_PATH)

    if args.record_video:
        print("--record-video is not yet implemented; ignoring.")

    model, vecnormalize = load_policy(args.checkpoint, vecnormalize_path, algo, env_config)

    results = [
        run_rollout(model, vecnormalize, env_config, seed=args.seed + i, deterministic=not args.stochastic)
        for i in range(args.n_episodes)
    ]

    hover_threshold = _hover_threshold_from_config(env_config)
    metrics = compute_aggregate_metrics(results, hover_threshold)
    print(_format_summary(metrics))

    output_path = args.output or args.checkpoint.parent / f"{args.checkpoint.stem}_eval.png"
    if args.n_episodes == 1:
        plot_single_episode(results[0], output_path)
    else:
        plot_multi_episode_summary(results, metrics, output_path)

    return metrics


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
