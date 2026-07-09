"""Evaluate a trained PPO checkpoint on QuadHover-v0 and plot a rollout.

Planned contents:
    - Load a saved SB3 PPO model (and its VecNormalize statistics).
    - Run one episode, recording position, target, and reward components
      at each step.
    - Plot position vs. target over time and the reward components,
      using matplotlib, and save the figure to disk.

Run as a script, e.g.:
    python -m quad_rl.training.eval_rollout --checkpoint <path>
"""
