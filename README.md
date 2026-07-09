# Quad-Rotor-Flight-Control
Using RL for Quadcopter flight control. Iterative approach with my own implementation of a MDP.

## Status

Stage 0.1: project scaffolding only. The custom physics model, Gym
environment, and PPO training loop are not implemented yet.

## Project structure

```
quad_rl/
  envs/
    dynamics.py        # rigid-body equations of motion, RK4 integrator
    quad_hover_env.py  # Gymnasium Env wrapping dynamics.py
    configs/
      default.yaml      # physical parameters, reward weights, episode limits
  training/
    train_ppo.py         # Stable-Baselines3 PPO training script
    eval_rollout.py       # runs a trained policy, logs/plots a trajectory
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

Run the (currently empty) test suite:

```bash
pytest quad_rl/tests/
```
