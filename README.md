# Quad-Rotor-Flight-Control
Using RL for Quadcopter flight control. Iterative approach with my own implementation of a MDP.

## Status

**Stage 0: complete.** A custom 6-DOF rigid-body quadrotor dynamics model
(quaternion state, RK4 integration) is wrapped in a Gymnasium hover-task
environment, and a PPO baseline demonstrably learns to hover: it converges
to within ~1cm of the target position and holds it, with smooth
(non-bang-bang) motor commands and a plateaued reward curve. See
[prompts/Stage0 prompts](prompts/Stage0%20prompts) for the stage roadmap
and [DEVELOPER.md](DEVELOPER.md) for commands and technical details.

Next: Stage 1 (reward shaping refinements, wind disturbance, sensor noise).

## Project structure

```
quad_rl/
  envs/
    dynamics.py        # rigid-body equations of motion, RK4 integrator
    quad_hover_env.py  # Gymnasium Env wrapping dynamics.py, registers "QuadHover-v0"
    configs/
      default.yaml      # physical parameters, reward weights, episode limits
  training/
    train_ppo.py         # Stable-Baselines3 PPO training script
    eval_rollout.py       # runs a trained policy, plots a trajectory
  tests/
    test_dynamics.py       # pytest suite for the physics model
requirements.txt
```

## Setup

Requires Python 3.10+ (developed against 3.14).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify the install:

```bash
python -c "import gymnasium, stable_baselines3; print(gymnasium.__version__, stable_baselines3.__version__)"
```

## Quick start

```bash
# Physics unit tests
pytest quad_rl/tests/ -v

# Train (see DEVELOPER.md for all options and hyperparameter notes)
python -m quad_rl.training.train_ppo --total-timesteps 3000000 --run-name my_run

# Watch progress
tensorboard --logdir runs/tensorboard

# Evaluate a checkpoint and plot a rollout
python -m quad_rl.training.eval_rollout --checkpoint runs/my_run/final_model.zip
```

For the full command reference, config file layout, and project conventions, see [DEVELOPER.md](DEVELOPER.md).
