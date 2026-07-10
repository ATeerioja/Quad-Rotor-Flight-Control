"""Train an RL algorithm on the QuadHover-v0 environment. Algorithm-
agnostic: which SB3 class, default policy, and hyperparameters to use
come from quad_rl.training.algorithms.ALGO_REGISTRY, so a second (or
third) algorithm is a --algo config change, not a training-script rewrite.

Usage:
    python -m quad_rl.training.train --algo ppo --total-timesteps 300000
    python -m quad_rl.training.train --algo sac --total-timesteps 5000

Overrides (-o/--override, repeatable) are dotted key=value strings
prefixed "env." or "algo." to disambiguate which of the two independent
config objects they target, e.g. -o env.physics.mass=0.6 -o
algo.learning_rate=1e-4. Both reuse the same load_raw_config/
apply_overrides machinery introduced for env config in Stage 1.1.

Infrastructure:
    - SubprocVecEnv (or DummyVecEnv, via --vec-env) with N parallel
      QuadHover instances (--n-envs, default 8).
    - VecNormalize wraps the vec env for running mean/std normalization
      of observations, and of reward too when the chosen algorithm's
      AlgoSpec.supports_vecnormalize_reward is True (on-policy algorithms
      like PPO; off-policy algorithms' replay buffers make a
      continuously-renormalized reward a moving target, so it's off by
      default for SAC/TD3 -- encoded in the spec, not an `if algo ==
      "sac"` branch here).
    - Each sub-env is wrapped in SB3's Monitor, which populates the
      `rollout/ep_rew_mean` / `rollout/ep_len_mean` TensorBoard scalars.
      HoverFractionCallback and EpisodeMetricsCallback
      (quad_rl.training.callbacks) add per-component reward and
      hover/position-error diagnostics.
    - CheckpointCallback saves the model (and matching VecNormalize
      statistics) every --checkpoint-freq environment timesteps.
    - Policy selection (MlpPolicy vs. MultiInputPolicy) is driven by
      whether the env's observation_space is a gym.spaces.Dict (Stage
      1.4's expose_privileged), not a separate flag that could disagree
      with it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from quad_rl.config.loader import apply_overrides, load_raw_config
from quad_rl.training.algorithms import ALGO_REGISTRY
from quad_rl.training.callbacks import EpisodeMetricsCallback, HoverFractionCallback

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"


def make_env(rank: int, seed: int, env_config_path, env_overrides: list[str]):
    """Env factory for vec-env workers. Must do its own imports and be
    module-level (not a lambda/local closure holding unpicklable state)
    since SubprocVecEnv defaults to the 'forkserver'/'spawn' multiprocessing
    start methods, which run this in a fresh interpreter rather than
    inheriting the parent's already-registered gym env.
    """

    def _init():
        import quad_rl.envs  # noqa: F401  (registers QuadHover-v0)

        env = gym.make("QuadHover-v0", config_path=env_config_path, overrides=env_overrides)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def build_vec_env(n_envs, seed, vec_env_cls, env_config_path, env_overrides, norm_reward: bool) -> VecNormalize:
    env_fns = [make_env(rank, seed, env_config_path, env_overrides) for rank in range(n_envs)]
    vec_env = vec_env_cls(env_fns)
    return VecNormalize(vec_env, norm_obs=True, norm_reward=norm_reward, clip_obs=10.0, clip_reward=10.0)


def _split_overrides(overrides: list[str]) -> tuple[list[str], list[str]]:
    """Overrides are prefixed "env." or "algo." to disambiguate which of
    the two independent config objects (env config from Stage 1.1, algo
    hyperparameter config from this stage) they target."""
    env_overrides, algo_overrides = [], []
    for item in overrides:
        if item.startswith("env."):
            env_overrides.append(item[len("env."):])
        elif item.startswith("algo."):
            algo_overrides.append(item[len("algo."):])
        else:
            raise ValueError(f"Override must be prefixed 'env.' or 'algo.': {item!r}")
    return env_overrides, algo_overrides


def train(args: argparse.Namespace):
    if args.algo not in ALGO_REGISTRY:
        raise SystemExit(f"Unknown --algo {args.algo!r}; choices: {sorted(ALGO_REGISTRY)}")
    spec = ALGO_REGISTRY[args.algo]

    env_overrides, algo_overrides = _split_overrides(args.override or [])

    if args.algo_config is not None:
        hyperparams = load_raw_config(args.algo_config, overrides=algo_overrides)
    else:
        hyperparams = apply_overrides(spec.default_hyperparams, algo_overrides)
    hyperparams = spec.build_kwargs(hyperparams)

    vec_env_cls = SubprocVecEnv if args.vec_env == "subproc" else DummyVecEnv
    vec_env = build_vec_env(
        args.n_envs, args.seed, vec_env_cls, args.env_config, env_overrides,
        norm_reward=spec.supports_vecnormalize_reward,
    )

    run_dir = RUNS_DIR / args.run_name
    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir = RUNS_DIR / "tensorboard"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Driven by the env's actual observation_space, not a separate flag
    # that could disagree with it (Stage 1.4's expose_privileged is what
    # makes this a Dict space in the first place).
    policy = "MultiInputPolicy" if isinstance(vec_env.observation_space, spaces.Dict) else spec.default_policy

    model = spec.cls(
        policy=policy,
        env=vec_env,
        tensorboard_log=str(tensorboard_dir),
        seed=args.seed,
        verbose=1,
        device="cpu",
        **hyperparams,
    )

    # save_freq is counted in vec-env "steps" (one call to vec_env.step(),
    # which advances every sub-env by one env-step), not raw timesteps --
    # num_timesteps advances by n_envs per call. Divide so checkpoints
    # land roughly every `checkpoint_freq` environment timesteps.
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix=f"{args.algo}_quadhover",
        save_vecnormalize=True,
    )
    callback = CallbackList([checkpoint_callback, HoverFractionCallback(), EpisodeMetricsCallback()])

    model.learn(total_timesteps=args.total_timesteps, callback=callback, tb_log_name=args.run_name)

    model.save(str(run_dir / "final_model"))
    vec_env.save(str(run_dir / "final_vecnormalize.pkl"))
    vec_env.close()

    print(f"Training complete. Final model: {run_dir / 'final_model.zip'}")
    print(f"Final VecNormalize stats: {run_dir / 'final_vecnormalize.pkl'}")
    print(f"TensorBoard logs: {tensorboard_dir} (run: tensorboard --logdir {tensorboard_dir})")

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algo", choices=sorted(ALGO_REGISTRY), default="ppo")
    parser.add_argument(
        "--env-config", type=Path, default=None,
        help="Path to an env YAML config; defaults to QuadHoverEnv's own default (hover.yaml).",
    )
    parser.add_argument(
        "--algo-config", type=Path, default=None,
        help="Path to an algo hyperparameter YAML; defaults to the algo's shipped configs/algo/<algo>.yaml.",
    )
    parser.add_argument(
        "-o", "--override", action="append", default=[],
        help="Dotted key=value override, prefixed 'env.' or 'algo.' (repeatable).",
    )
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--vec-env", choices=["subproc", "dummy"], default="subproc")
    parser.add_argument("--checkpoint-freq", type=int, default=50_000, help="In environment timesteps.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None, help="Defaults to '<algo>_quadhover'.")
    args = parser.parse_args()
    if args.run_name is None:
        args.run_name = f"{args.algo}_quadhover"
    return args


if __name__ == "__main__":
    train(parse_args())
