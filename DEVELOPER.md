# Developer guide

Technical reference for working on this repo: environment setup, commands,
config layout, and the conventions the code assumes. For project status
and roadmap, see [README.md](README.md).

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Requires Python 3.10+; developed and tested against 3.14.
- Core deps: numpy, scipy, gymnasium, stable-baselines3, torch, pyyaml,
  matplotlib, tensorboard, pytest (see `requirements.txt` for versions).
- If a CUDA GPU is present, PPO will use it by default and print a
  warning that MLP policies run better on CPU. For this env's tiny
  observation/action space, CPU is usually as fast or faster — pass
  `device="cpu"` to the `PPO(...)` constructor in `train_ppo.py` if you
  want to force it (already set there).

## Commands

### Run the physics unit tests

```bash
pytest quad_rl/tests/test_dynamics.py -v
```

Five tests, each targeting a specific way the dynamics model could be
subtly wrong: hover equilibrium, free-fall acceleration, roll/pitch/yaw
decoupling for a diagonal inertia tensor, quaternion-norm drift under
RK4 integration, and angular-momentum conservation for torque-free
motion. These are fast (~1s) and don't require training — run them
after touching anything in `quad_rl/envs/dynamics.py`.

### Train

```bash
python -m quad_rl.training.train_ppo \
  --total-timesteps 3000000 \
  --n-envs 8 \
  --run-name my_run
```

Full option list (`python -m quad_rl.training.train_ppo --help`):

| flag | default | meaning |
|---|---|---|
| `--total-timesteps` | 300000 | total environment steps across all parallel envs |
| `--n-envs` | 8 | parallel `QuadHoverEnv` instances |
| `--vec-env` | `subproc` | `subproc` (multiprocess, faster) or `dummy` (single-process, easier to debug/Ctrl+C) |
| `--n-steps` | 2048 | rollout length per env before each PPO update |
| `--batch-size` | 256 | SGD minibatch size |
| `--learning-rate` | 3e-4 | Adam learning rate |
| `--checkpoint-freq` | 50000 | environment steps between checkpoints |
| `--seed` | 0 | RNG seed (env spawn perturbations, PPO init) |
| `--run-name` | `ppo_quadhover` | names the output directory and TensorBoard run |

Outputs land under `runs/<run-name>/`:
- `checkpoints/ppo_quadhover_<N>_steps.zip` + matching
  `ppo_quadhover_vecnormalize_<N>_steps.pkl`, every `--checkpoint-freq` steps.
- `final_model.zip` + `final_vecnormalize.pkl`, written when training
  completes. Note the final step count can slightly exceed
  `--total-timesteps` — PPO always finishes its in-progress rollout
  before stopping, so it overshoots by up to `n_steps * n_envs`.

TensorBoard logs go to the shared `runs/tensorboard/` directory (each run
gets its own numbered subdirectory, e.g. `my_run_1`, incrementing if you
reuse a `--run-name`):

```bash
tensorboard --logdir runs/tensorboard
```

Scalars to watch:
- `rollout/ep_rew_mean` / `rollout/ep_len_mean` — standard SB3 episode
  stats (from the `Monitor` wrapper around each sub-env). Note
  `ep_rew_mean` is a **per-episode sum**, not a per-step average — as
  `ep_len_mean` grows (policy survives longer), the summed reward can
  look like it's getting worse even while per-step behavior improves.
  Divide `ep_rew_mean / ep_len_mean` by hand if the raw curve looks
  confusing early in training.
- `custom/fraction_within_hover_threshold` — mean, over the last 100
  completed episodes, of the fraction of each episode spent within
  `reward.hover_threshold` of the target. More direct evidence of
  "actually hovering" than reward alone.
- `train/explained_variance` — climbing toward 1.0 indicates the value
  function is fitting well.

### Evaluate a checkpoint

```bash
python -m quad_rl.training.eval_rollout --checkpoint runs/my_run/final_model.zip
```

Runs one deterministic episode on a plain (non-vectorized) `QuadHoverEnv`,
normalizing observations with the checkpoint's saved `VecNormalize`
statistics, and saves a plot to `<checkpoint_dir>/<checkpoint_stem>_eval.png`
with five panels: position vs. target for x/y/z, the 4-motor action
trace, and reward components over time.

Options:

| flag | default | meaning |
|---|---|---|
| `--checkpoint` | *required* | path to a `.zip` model file |
| `--vecnormalize` | inferred | path to the matching `.pkl` stats file; auto-inferred from the checkpoint filename if it follows `train_ppo.py`'s naming (see below) |
| `--output` | inferred | where to save the plot |
| `--seed` | 0 | episode seed (spawn point, target, perturbations) |
| `--stochastic` | off | sample actions instead of using the policy mean |

VecNormalize inference matches `train_ppo.py`'s naming convention:
`final_model.zip` → `final_vecnormalize.pkl`, and
`<prefix>_<N>_steps.zip` → `<prefix>_vecnormalize_<N>_steps.pkl`. If your
checkpoint doesn't follow that pattern, pass `--vecnormalize` explicitly
— running with mismatched or missing normalization stats will produce a
policy that looks broken because its inputs are scaled wrong, not
because the weights are bad.

## Config

`quad_rl/envs/configs/default.yaml` is the single source for every
physical parameter, reward weight, episode limit, and spawn range —
nothing physically meaningful is hardcoded in `dynamics.py` or
`quad_hover_env.py` (verified by grepping both files for the config's
literal values; this is a hard requirement for Stage 3's planned domain
randomization to be a config change, not a code change). Sections:

- `physics`: mass, inertia (diagonal), arm_length, thrust_coefficient,
  drag_coefficient, yaw_torque_coefficient, gravity, motor_time_constant
  (accepted but not yet used — no actuator-lag state in the current
  13-dim state vector).
- `simulation`: integration timestep (`dt`).
- `episode`: `max_steps`, crash conditions (`crash_altitude`,
  `max_tilt_deg`, `bounding_box`).
- `reward`: per-term weights, `hover_threshold`, `hover_bonus`,
  `crash_penalty`.
- `spawn`: reset-time randomization ranges (position/velocity/orientation
  perturbation, target sampling region).

`QuadHoverEnv.__init__` takes an optional `config_path` if you want to
point at an alternate YAML file instead of editing the default.

## Conventions (read before modifying `dynamics.py` or `quad_hover_env.py`)

- **State vector** (13-dim, `quad_rl/envs/dynamics.py`): position (3,
  world) | velocity (3, world) | quaternion (4, scalar-first `[w,x,y,z]`,
  body→world) | angular velocity (3, body frame). Quaternion, not Euler
  angles — avoids gimbal lock. Renormalized after every RK4 step.
- **Frames**: world is Z-up. Body frame at identity orientation is
  x-forward, y-left, z-up (FLU) — forced by x-forward + Z-up +
  right-handedness, not a free choice.
- **Motor layout**: X-configuration, motors numbered 1–4 as front-right,
  rear-right, rear-left, front-left. Diagonal pairs spin opposite
  directions. See the mixing matrix documented in `dynamics.py`'s module
  docstring before touching motor mixing.
- **Action space**: `Box(4,)` in `[-1, 1]` at the Gym env level, mapped
  to `[0, 1]` motor commands internally (`(action + 1) / 2`) before
  reaching `dynamics.step`.
- **Observation space** (19-dim, `quad_hover_env.py`): position error
  (3, ÷`POSITION_ERROR_SCALE`) | velocity (3, ÷`LINEAR_VELOCITY_SCALE`)
  | 6D rotation representation (6, first two columns of the body→world
  rotation matrix — avoids the quaternion double-cover problem, already
  unit-scale) | angular velocity (3, ÷`ANGULAR_VELOCITY_SCALE`) |
  previous action (4, already in `[-1, 1]`). The three `*_SCALE`
  constants and `TARGET_ALTITUDE` are env-design constants (not physical
  parameters), intentionally hardcoded at the top of `quad_hover_env.py`.
- **`info` dict** (`quad_hover_env.step`): `within_hover_threshold`
  (bool, drives the custom TensorBoard metric) and `reward_components`
  (dict, always the same keys — `position`, `angular_velocity`,
  `action_magnitude`, `action_rate`, `hover_bonus`, `crash_penalty` —
  regardless of whether the step crashed, so per-episode arrays stay
  rectangular for plotting).
- **Vec-env workers**: `train_ppo.py`'s `make_env` factory does its own
  imports inside the returned closure. `SubprocVecEnv` defaults to the
  `forkserver`/`spawn` multiprocessing start method (not `fork`), so
  worker processes start fresh and need `quad_rl.envs` imported again to
  register `QuadHover-v0` — don't hoist those imports to module level.

## Directory layout for run artifacts

`runs/` is gitignored entirely (checkpoints and TensorBoard logs are
large and not source). Structure created by `train_ppo.py`:

```
runs/
  <run-name>/
    final_model.zip
    final_vecnormalize.pkl
    final_model_eval.png           # from eval_rollout.py, if run on final_model.zip
    checkpoints/
      ppo_quadhover_<N>_steps.zip
      ppo_quadhover_vecnormalize_<N>_steps.pkl
      ppo_quadhover_<N>_steps_eval.png   # eval_rollout.py output lands next to
                                          # whichever checkpoint you pointed it at
  tensorboard/
    <run-name>_1/                  # SB3 auto-increments on reuse
```
