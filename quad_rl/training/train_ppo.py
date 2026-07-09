"""Train PPO (Stable-Baselines3) on the QuadHover-v0 environment.

Planned contents:
    - Vectorized envs: SubprocVecEnv or DummyVecEnv with 8-16 parallel
      QuadHover instances. Parallel infrastructure goes in now, even
      without physics randomization yet, so later domain-randomization
      work is a drop-in change to env creation rather than a training
      script rewrite.
    - VecNormalize for observation and reward normalization.
    - TensorBoard logging: episode reward, episode length, and a custom
      metric for fraction of episode spent within the hover threshold.
    - Periodic checkpointing (every N steps) via SB3 callbacks.
    - PPO hyperparameters chosen for a ~19-dim observation, 4-dim
      continuous-action control problem (not SB3 defaults) -- batch
      size, n_steps, and learning rate to be justified in comments
      once implemented.

Run as a script, e.g.:
    python -m quad_rl.training.train_ppo
"""
