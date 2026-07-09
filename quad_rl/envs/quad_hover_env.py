"""Gymnasium environment: quadrotor hover-at-target-position task.

Wraps quad_rl.envs.dynamics with a Gymnasium Env interface and registers
it as "QuadHover-v0" via gymnasium.register so it can be created with
gym.make().

Observation (19 dims total, each component normalized to roughly unit
scale):
    position error to target      (3)
    linear velocity                (3)
    orientation, 6D rotation rep   (6)  -- first two columns of the
                                           rotation matrix (avoids the
                                           quaternion double-cover
                                           problem; standard choice in
                                           recent quadrotor RL work)
    angular velocity               (3)
    previous action                (4)

Action space: Box(4,) in [-1, 1], mapped internally to motor commands
consumed by dynamics.py.

Reward per step (weights to be named constants at module top, not
buried in the formula):
    - negative position error norm
    - negative angular velocity norm (discourage spinning)
    - negative action magnitude (control effort penalty)
    - negative action-rate penalty: ||action - prev_action||
    - bonus when within a threshold distance of target
    - large fixed penalty + episode termination on crash (altitude < 0),
      excessive tilt (> 60 deg from upright), or leaving a bounding box

reset(): spawns near the target with small random position/velocity/
orientation perturbation (not domain randomization yet -- just enough
variety that the policy can't memorize a single trajectory), and samples
the target position within a small region so the agent learns
position-conditioned behavior.

Episode length: capped at 1000 steps (10 s at 100 Hz) as a timeout.
"""
