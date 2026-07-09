"""Train PPO (Stable-Baselines3) on the QuadHover-v0 environment.

Usage:
    python -m quad_rl.training.train_ppo --total-timesteps 300000

Infrastructure:
    - SubprocVecEnv (or DummyVecEnv, via --vec-env) with N parallel
      QuadHover instances (--n-envs, default 8). Physics isn't
      randomized yet, but building the parallel-env plumbing now means
      Stage 3's domain randomization is a change to the env factory
      (make_env below), not a training-script rewrite.
    - VecNormalize wraps the vec env for running mean/std normalization
      of both observations and rewards.
    - Each sub-env is wrapped in SB3's Monitor, which is what populates
      the `rollout/ep_rew_mean` and `rollout/ep_len_mean` TensorBoard
      scalars automatically. A custom HoverFractionCallback adds one
      more: `custom/fraction_within_hover_threshold`, the fraction of
      each episode spent within the env's hover_threshold, read from the
      `within_hover_threshold` key QuadHoverEnv.step() puts in `info`.
    - CheckpointCallback saves the model (and matching VecNormalize
      statistics, via save_vecnormalize=True) every --checkpoint-freq
      environment timesteps.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"


def make_env(rank: int, seed: int = 0):
    """Env factory for vec-env workers. Must do its own imports and be
    module-level (not a lambda/local closure holding unpicklable state)
    since SubprocVecEnv defaults to the 'forkserver'/'spawn' multiprocessing
    start methods, which run this in a fresh interpreter rather than
    inheriting the parent's already-registered gym env.
    """

    def _init():
        import quad_rl.envs  # noqa: F401  (registers QuadHover-v0)

        env = gym.make("QuadHover-v0")
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


class HoverFractionCallback(BaseCallback):
    """Logs the mean, over the last 100 completed episodes, of the
    fraction of each episode's steps spent within the hover threshold.

    Reward alone doesn't distinguish "consistently close to target" from
    "oscillating through the target" -- this metric is a more direct
    read on whether the policy is actually learning to hover.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._in_progress: list[list[bool]] = []
        self._completed_fractions: deque[float] = deque(maxlen=100)

    def _on_training_start(self) -> None:
        self._in_progress = [[] for _ in range(self.training_env.num_envs)]

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]
        for i, (info, done) in enumerate(zip(infos, dones)):
            self._in_progress[i].append(info["within_hover_threshold"])
            if done:
                self._completed_fractions.append(float(np.mean(self._in_progress[i])))
                self._in_progress[i] = []

        if self._completed_fractions:
            self.logger.record(
                "custom/fraction_within_hover_threshold",
                float(np.mean(self._completed_fractions)),
            )
        return True


def build_vec_env(n_envs: int, seed: int, vec_env_cls) -> VecNormalize:
    env_fns = [make_env(rank, seed) for rank in range(n_envs)]
    vec_env = vec_env_cls(env_fns)
    return VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)


def train(args: argparse.Namespace) -> None:
    vec_env_cls = SubprocVecEnv if args.vec_env == "subproc" else DummyVecEnv
    vec_env = build_vec_env(args.n_envs, args.seed, vec_env_cls)

    run_dir = RUNS_DIR / args.run_name
    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir = RUNS_DIR / "tensorboard"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparameters, chosen for a small continuous-control problem
    # (19-dim obs, 4-dim action, 1000-step episodes at 100 Hz) rather
    # than left at SB3's defaults (which are tuned more toward Atari):
    #   - n_steps=2048 (per env): with n_envs=8 that's a 16384-step
    #     rollout buffer per update, long enough to span several full
    #     hover episodes and give GAE a stable advantage estimate before
    #     each policy update, in line with SB3-zoo's continuous-control
    #     (MuJoCo) defaults rather than its short discrete-action ones.
    #   - batch_size=256: divides the 16384-step buffer into 64
    #     minibatches per epoch -- small enough for frequent, low-variance
    #     gradient steps, large enough to keep the GPU/CPU batched
    #     efficiently; a common middle ground in continuous-control PPO
    #     configs (SB3's own 64 default is tuned for smaller buffers).
    #   - learning_rate=3e-4: the standard Adam starting point for PPO
    #     on continuous-control tasks (used across the original PPO paper
    #     and most SB3-zoo continuous-control configs); revisit with a
    #     linear decay schedule once a longer run needs the extra
    #     stability late in training.
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=str(tensorboard_dir),
        seed=args.seed,
        verbose=1,
        device="cpu"
    )

    # save_freq is counted in vec-env "steps" (one call to vec_env.step(),
    # which advances every sub-env by one env-step), not raw timesteps --
    # num_timesteps advances by n_envs per call. Divide so checkpoints
    # land roughly every `checkpoint_freq` environment timesteps.
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_quadhover",
        save_vecnormalize=True,
    )
    callback = CallbackList([checkpoint_callback, HoverFractionCallback()])

    model.learn(total_timesteps=args.total_timesteps, callback=callback, tb_log_name=args.run_name)

    model.save(str(run_dir / "final_model"))
    vec_env.save(str(run_dir / "final_vecnormalize.pkl"))
    vec_env.close()

    print(f"Training complete. Final model: {run_dir / 'final_model.zip'}")
    print(f"Final VecNormalize stats: {run_dir / 'final_vecnormalize.pkl'}")
    print(f"TensorBoard logs: {tensorboard_dir} (run: tensorboard --logdir {tensorboard_dir})")


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


if __name__ == "__main__":
    train(parse_args())
