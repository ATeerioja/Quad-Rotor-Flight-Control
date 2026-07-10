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
- If a CUDA GPU is present, PPO/SAC/TD3 will print a warning that MLP
  policies run better on CPU. For this env's tiny observation/action
  space, CPU is usually as fast or faster — `train.py` already passes
  `device="cpu"` to the model constructor.

## Commands

### Run the tests

```bash
pytest quad_rl/tests/ -v
```

- `test_dynamics.py` — 7 tests: hover equilibrium, free-fall acceleration,
  roll/pitch/yaw decoupling for a diagonal inertia tensor, quaternion-norm
  drift under RK4, angular-momentum conservation, and the `external_force`
  parameter (adds pure translational acceleration, defaults to a no-op).
- `test_config.py` — config loader/schema round-trip, single-leaf
  overrides, unknown/missing-key validation, `defaults:` chain resolution.
- `test_rewards.py`, `test_disturbances.py`, `test_randomization.py` — the
  three pluggable registries (below) in isolation.
- `test_env.py` — the full env test suite: `check_env` for both
  `expose_privileged` settings, determinism, every crash condition firing
  independently, truncation timing, and a registry/config drift check
  (every `type:` string in every shipped YAML resolves in its registry).
- `test_algorithms.py`, `test_train.py`, `test_evaluate.py` — the
  algorithm registry, training entrypoint (including a fixed-seed PPO
  first-update loss reproducibility check and a `--algo sac` smoke test),
  and evaluation harness (including a regression check against the
  Stage 0 baseline checkpoint).

Most of these are fast (a few seconds total); `test_train.py` and the
Stage 0 checkpoint tests in `test_evaluate.py` actually train/run a model
briefly and take longer (well under a minute).

### Train

```bash
python -m quad_rl.training.train --algo ppo --total-timesteps 3000000 --run-name my_run
```

`--algo` selects from `quad_rl.training.algorithms.ALGO_REGISTRY`
(`ppo`, `sac`, `td3` — see [Adding a new algorithm](#adding-a-new-algorithm)
below). Full option list (`python -m quad_rl.training.train --help`):

| flag | default | meaning |
|---|---|---|
| `--algo` | `ppo` | `ppo`, `sac`, or `td3` |
| `--env-config` | *(none)* | path to an env YAML config; defaults to `QuadHoverEnv`'s own default (`hover.yaml`, via `default.yaml`) |
| `--algo-config` | *(none)* | path to an algo hyperparameter YAML; defaults to the algo's shipped `quad_rl/training/configs/algo/<algo>.yaml` |
| `-o`/`--override` | *(none)* | dotted `key=value` override, prefixed `env.` or `algo.` (repeatable — see [Config](#config) below) |
| `--total-timesteps` | 300000 | total environment steps across all parallel envs |
| `--n-envs` | 8 | parallel `QuadHoverEnv` instances |
| `--vec-env` | `subproc` | `subproc` (multiprocess, faster) or `dummy` (single-process, easier to debug/Ctrl+C) |
| `--checkpoint-freq` | 50000 | environment steps between checkpoints |
| `--seed` | 0 | RNG seed (env spawn perturbations, disturbance/randomization sampling, model init) |
| `--run-name` | `<algo>_quadhover` | names the output directory and TensorBoard run |

Per-algorithm hyperparameters (`learning_rate`, `n_steps`/`train_freq`,
`batch_size`, ...) are **not** CLI flags — they live in
`quad_rl/training/configs/algo/<algo>.yaml` and are overridden via
`-o algo.<key>=<value>`, not a growing list of algorithm-specific flags.
Off-policy algorithms (SAC, TD3) don't have PPO's `n_steps` concept; if you
pass `-o algo.n_steps=<N>` against one, it's translated into
`train_freq`/`gradient_steps=<N>` automatically (see
`quad_rl.training.algorithms._off_policy_kwargs`) rather than being
silently ignored.

Outputs land under `runs/<run-name>/`:
- `config.yaml` — the fully-resolved env config and algo hyperparameters
  actually used for this run (written before training starts, so it
  exists even if the run is interrupted). This is what `evaluate.py`
  reads `--algo` and the env config back from, and the source of truth
  for exactly reproducing a run later.
- `checkpoints/<algo>_quadhover_<N>_steps.zip` + matching
  `<algo>_quadhover_vecnormalize_<N>_steps.pkl`, every `--checkpoint-freq`
  steps.
- `final_model.zip` + `final_vecnormalize.pkl`, written when training
  completes. Note the final step count can slightly exceed
  `--total-timesteps` — PPO always finishes its in-progress rollout
  before stopping, so it overshoots by up to (rollout length) * `n_envs`.

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
  the configured `hover_bonus` term's `threshold` of the target (0 if no
  such term is configured). More direct evidence of "actually hovering"
  than reward alone. (`quad_rl.training.callbacks.HoverFractionCallback`)
- `custom/terminal_position_error_mean` — mean, over the last 100
  completed episodes, of the final step's distance to target in meters.
  (`quad_rl.training.callbacks.EpisodeMetricsCallback`)
- `reward_components/<name>_mean` — per-reward-term episode-mean, one
  scalar per configured term (`position`, `hover_bonus`, `crash_penalty`,
  ...). This is what makes reward shaping debuggable — the summed total
  reward alone can't tell you which term moved.
  (`quad_rl.training.callbacks.EpisodeMetricsCallback`)
- `train/explained_variance` — climbing toward 1.0 indicates the value
  function is fitting well (PPO only).

**Deprecated:** `python -m quad_rl.training.train_ppo ...` still works
(same flags as before Stage 1.6: `--n-steps`, `--batch-size`,
`--learning-rate`, etc.) — it's a thin shim that warns and forwards to
`train.py --algo ppo`, translating those flags into `algo.*` overrides.
Prefer `train.py` directly for anything new.

### Evaluate a checkpoint

```bash
python -m quad_rl.training.evaluate --checkpoint runs/my_run/final_model.zip --n-episodes 20
```

Loads through `ALGO_REGISTRY`, so it isn't tied to PPO. `--algo` is read
back from the run's own `config.yaml` automatically when available (any
run trained since Stage 1.6) — you only need to pass `--algo` explicitly
for older checkpoints that predate it (e.g. a Stage 0 run).

Options (`python -m quad_rl.training.evaluate --help`):

| flag | default | meaning |
|---|---|---|
| `--checkpoint` | *required* | path to a `.zip` model file |
| `--vecnormalize` | inferred | path to the matching `.pkl` stats file; auto-inferred from the checkpoint filename (see below) |
| `--output` | inferred | where to save the plot |
| `--algo` | *(none)* | falls back to this only if `config.yaml` isn't found next to the checkpoint |
| `--n-episodes` | 1 | number of episodes to roll out |
| `--eval-env-config` | *(none)* | evaluate under a different env config than training used (the sim-to-sim generalization check); defaults to the run's own training-time config from `config.yaml`, or `QuadHoverEnv`'s own default if that isn't available either |
| `--record-video` | off | not yet implemented — a stub reserved for a future stage |
| `--seed` | 0 | first episode's seed (subsequent episodes use `seed + 1`, `seed + 2`, ...) |
| `--stochastic` | off | sample actions instead of using the policy mean |

VecNormalize inference matches `train.py`'s naming convention:
`final_model.zip` → `final_vecnormalize.pkl`, and
`<prefix>_<N>_steps.zip` → `<prefix>_vecnormalize_<N>_steps.pkl`. If your
checkpoint doesn't follow that pattern, pass `--vecnormalize` explicitly —
running with mismatched or missing normalization stats will produce a
policy that looks broken because its inputs are scaled wrong, not because
the weights are bad.

Aggregate metrics reported (printed, and for `--n-episodes` > 1 also
embedded in the plot):
- **success rate** — reached and held within the configured
  `hover_bonus` term's threshold for the final 2 simulated seconds of the
  episode. `N/A` if no such term is configured.
- **mean terminal position error** — distance to target at each episode's
  final step, in meters.
- **crash rate** — fraction of episodes that ended via `terminated`
  (crash) rather than `truncated` (timeout).
- **mean episode length**, in steps.
- **action smoothness** — mean of ‖aₜ − aₜ₋₁‖ across all steps of all
  episodes.

`--n-episodes 1` (the default) produces the original 5-panel plot
(position vs. target per axis, the 4-motor action trace, reward
components over time). For N>1, a single plot instead overlays all N
distance-to-target traces with a mean/±σ band, annotated with the summary
table.

**Deprecated:** `python -m quad_rl.training.eval_rollout --checkpoint ...`
still works (same flags as before Stage 1.7) — a thin shim forwarding to
`evaluate.py --n-episodes 1 --algo ppo` (this script predates
`config.yaml`, so it always falls back to `ppo` rather than reading it
back). Prefer `evaluate.py` directly for anything new.

### Visualize a rollout in 3D

`viz/` is a separate, offline package — it never touches `quad_hover_env.py`
or `dynamics.py`, and nothing under `quad_rl/training/` imports it.

The quickest path is `viz/generate_animation.py`, which records a rollout
(random actions, or a trained checkpoint via `--checkpoint`) and renders it
in one command:

```bash
python -m viz.generate_animation --out rollout.mp4
python -m viz.generate_animation --out rollout.mp4 \
    --checkpoint runs/my_run/final_model.zip \
    --vecnormalize runs/my_run/final_vecnormalize.pkl --algo ppo
```

To record manually instead (e.g. a custom policy loop), wrap a manual eval
loop (not a training env) in `viz.recorder.TrajectoryRecorder`, then render
the saved `.npz` with `viz/animate.py` directly:

```bash
python -m viz.animate --file rollout.npz --out rollout.mp4
```

Options (`python -m viz.animate --help`):

| flag | default | meaning |
|---|---|---|
| `--file` | *required* | input `.npz` from `TrajectoryRecorder.save()` |
| `--out` | *required* | `.mp4` (ffmpeg writer) or `.gif` (pillow writer), inferred from the suffix |
| `--trail` | 50 | number of past positions in the fading trail |
| `--fps` | 30 | output frame rate |
| `--arm-length` | 0.17 | visual arm length (m); the `.npz` has no physics params, so this is a rendering constant, not read from config |
| `--writer` | inferred | force `ffmpeg` or `pillow` instead of inferring from `--out` |

`.mp4` needs a system `ffmpeg` binary on `PATH`; `.gif` works out of the
box via matplotlib's bundled `pillow` writer. See
[docs/trajectory_visualization.md](docs/trajectory_visualization.md) for
the recording example, the `.npz` schema, and how to view the output.

## Config

Config is hierarchical and validated (`quad_rl/config/`), not a single
flat file:

- `quad_rl/envs/configs/base.yaml` — `physics` + `simulation`. Root of the
  inheritance chain (no `defaults:` key).
- `quad_rl/envs/configs/hover.yaml` — `episode`, `reward`, `spawn`,
  `disturbance`, `randomization`, `expose_privileged`, `history_length`.
  Inherits `base.yaml` via a `defaults: [base.yaml]` key at the top —
  child values win over parent values, recursively merged per-section.
- `quad_rl/envs/configs/default.yaml` — a one-line alias,
  `defaults: [hover.yaml]`, kept for backward compatibility (so
  `config_path=None` keeps resolving to the full hover-task config).

`quad_rl.config.loader.load_config(path, overrides)` resolves the
`defaults:` chain, applies overrides, and validates against
`quad_rl.config.schema.EnvConfig` (raises on unknown or missing keys, or a
wrong-typed value). `load_raw_config`/`apply_overrides` are the
schema-free half of the same machinery, reused by
`quad_rl.training.algorithms` for algo hyperparameter YAMLs (which have no
single shared schema the way env sections do).

**Overrides** are dotted `key=value` strings, e.g. `physics.mass=0.6` or
`disturbance.force.type=ou_wind`. The value is parsed with `yaml.safe_load`
for free type coercion (`"0.6"` → `0.6`, `"true"` → `True`, `"{type: ou_wind,
sigma: 0.3}"` → a dict). In `train.py`, overrides are prefixed `env.` or
`algo.` to disambiguate which of the two independent config objects
(env vs. algorithm hyperparameters) they target.

Sections:
- `physics`: `mass`, `inertia` (nested `ixx`/`iyy`/`izz`, materialized as a
  flat `np.array` for `dynamics.step`), `arm_length`,
  `thrust_coefficient`, `drag_coefficient`, `yaw_torque_coefficient`,
  `gravity`, `motor_time_constant` (accepted but not yet used — no
  actuator-lag state in the current 13-dim state vector).
- `simulation`: integration timestep (`dt`).
- `episode`: `max_steps`, crash conditions (`crash_altitude`,
  `max_tilt_deg`, `bounding_box`).
- `reward`: a `terms:` list, e.g. `{type: hover_bonus, weight: 1.0,
  threshold: 0.1}` — see [Adding a new reward term](#adding-a-new-reward-term).
- `spawn`: reset-time randomization ranges (position/velocity/orientation
  perturbation, target sampling region).
- `disturbance`: `force` and `observation`, each `{type: ..., ...}` —
  defaults to `none`/`none` (no-op); see
  [Adding a new disturbance](#adding-a-new-disturbance).
- `randomization`: a `spec:` dict mapping dotted paths into `physics`
  (`mass`, `inertia.ixx`, `drag_coefficient`, ...) to a distribution
  (`{type: uniform, lo, hi}`, `{type: log_uniform, lo, hi}`, `{type:
  fixed, value}`). An empty spec (the shipped default) is a no-op — every
  episode uses the nominal `physics` values from `base.yaml` exactly.
- `expose_privileged` (bool): when true, `QuadHoverEnv.observation_space`
  becomes a `gym.spaces.Dict` — see
  [The `expose_privileged` contract](#the-expose_privileged-contract).
- `history_length` (int, ≥1): stacks the last N observations; 1 is a
  no-op. `observation_space`'s (or its `"observation"` key's, if
  `expose_privileged`) shape scales to `19 * history_length`.

`QuadHoverEnv.__init__` takes an optional `config_path`/`overrides` if you
want to point at an alternate YAML file instead of editing the default,
or an already-built `env_config: EnvConfig` directly (used by
`evaluate.py` to reconstruct a training run's exact config from its
dumped `config.yaml` without writing a temp file).

Every config dataclass has a `from_dict`/`asdict` pair that round-trips
exactly (`asdict()` output is plain-Python, safe to `yaml.safe_dump` —
this is what gets written to `runs/<name>/config.yaml`). See
`quad_rl/config/schema.py`'s module docstring before adding a new section.

## The three registries (and how to extend them)

Reward terms, disturbances, and algorithms are all "pick an implementation
by a `type`/`--algo` string" registries, built the same way so extending
any of them follows the same shape:

```python
REGISTRY: dict[str, type] = {}

def register(name: str):
    def decorator(cls):
        REGISTRY[name] = cls
        return cls
    return decorator

@register("my_thing")
class MyThing:
    def __init__(self, **kwargs_from_yaml): ...
```

Config supplies `{type: my_thing, ...kwargs}`; the registry's builder pops
`type` and constructs `REGISTRY[type](**rest)` — so a new implementation's
`__init__` parameter names must match its YAML config keys exactly.

### Adding a new reward term

In `quad_rl/envs/rewards.py`:
1. Add a class with a `name` attribute and `__call__(self, ctx:
   StepContext) -> float`, taking its weight (and any extra params, e.g.
   `HoverBonus`'s `threshold`) in `__init__`.
2. Decorate it `@register_reward("my_term")`.
3. Reference it in a config's `reward.terms` list:
   `{type: my_term, weight: 1.0, ...}`.

`RewardFunction.from_config` builds the list of term instances from
config; `RewardFunction.__call__` sums them and returns `(total,
components)`, with `crash_overrides=True` (the default) zeroing every
term except `CrashPenalty` on a crash step. `StepContext` carries `state`,
`prev_state`, `action`, `prev_action`, `target`, `dt`, `crashed` — not
every term needs every field, but new terms can rely on all of them being
present.

### Adding a new disturbance

In `quad_rl/envs/disturbances.py`:
- **Force** (wind-like): implement `force(self, state, t, rng) ->
  np.ndarray(3)`, decorate `@register_force("my_wind")`. Sampled once per
  env step (not per RK4 substage) and added to `dynamics.step`'s optional
  `external_force` parameter — `dynamics.py` has zero knowledge of what
  produces it.
- **Observation noise**: implement `apply(self, obs, rng) -> obs`,
  decorate `@register_observation("my_noise")`. Applied in `_get_obs`
  after scaling.

Reference via `disturbance.force`/`disturbance.observation`:
`{type: my_wind, ...params}`. Stateful disturbances (like
`OrnsteinUhlenbeckWind`) are rebuilt fresh every `reset()` — their
per-episode state (e.g. current wind value) restarts cleanly each
episode rather than carrying over.

### Adding a new algorithm

In `quad_rl/training/algorithms.py`:
1. Add a `quad_rl/training/configs/algo/<name>.yaml` with that
   algorithm's hyperparameters.
2. Add an `ALGO_REGISTRY["<name>"] = AlgoSpec(cls=..., default_policy=...,
   default_hyperparams=_load_default_hyperparams("<name>"),
   supports_vecnormalize_reward=..., build_kwargs=...)` entry.
   `build_kwargs` is only needed if the algorithm's constructor expects
   different parameter names than the shared/common ones (see
   `_off_policy_kwargs`'s `n_steps` → `train_freq`/`gradient_steps`
   translation for the pattern to follow — encode the translation in the
   spec, not an `if args.algo == "..."` branch in `train.py`).
3. `python -m quad_rl.training.train --algo <name>` now works.

### The `expose_privileged` contract

When `env_config.expose_privileged` is `true`:
- `QuadHoverEnv.observation_space` becomes `gym.spaces.Dict({"observation":
  Box(19 * history_length,), "privileged_obs": Box(len(PHYSICS_PARAM_NAMES),)})`.
- Every `reset()`/`step()` observation is a `{"observation": ..., "privileged_obs":
  ...}` dict, not a plain array.
- `info["physics_params"]` (every step) and `info["physics_param_names"]`
  (once, at `reset()`) expose the same vector regardless of
  `expose_privileged` — it's always in `info`, the Dict observation just
  also surfaces it to the policy directly.
- `train.py` selects SB3's `MultiInputPolicy` **because** the env's
  `observation_space` is a `Dict` — not a separate flag that could
  disagree with it. If you add a new way to produce a Dict observation
  space, policy selection picks it up automatically; don't add a second
  flag for this.
- When `false` (the default), `observation_space` is a plain `Box` exactly
  as before this feature existed — `MlpPolicy` keeps working untouched.

See [docs/domain_adaptation.md](docs/domain_adaptation.md) for what this
buys: domain-randomization-only, RMA two-phase teacher/student, and
explicit system identification all read the same flag and `privileged_obs`
shape.

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
- **`external_force`**: `dynamics.step`/`_state_derivative` accept an
  optional 3-vector (Newtons, world frame), added the same way as thrust,
  zero-order-held across RK4 substages like `action`. `None` by default —
  the disturbance seam is invisible unless a caller (the env, sampling
  once per step from `quad_rl.envs.disturbances`) opts in. `dynamics.py`
  itself has no knowledge of wind models.
- **Observation space** (19-dim before any `history_length` stacking,
  `quad_hover_env.py`): position error (3, ÷`POSITION_ERROR_SCALE`) |
  velocity (3, ÷`LINEAR_VELOCITY_SCALE`) | 6D rotation representation (6,
  first two columns of the body→world rotation matrix — avoids the
  quaternion double-cover problem, already unit-scale) | angular velocity
  (3, ÷`ANGULAR_VELOCITY_SCALE`) | previous action (4, already in
  `[-1, 1]`). The three `*_SCALE` constants and `TARGET_ALTITUDE` are
  env-design constants (not physical parameters), intentionally
  hardcoded at the top of `quad_hover_env.py`. Observation noise (if
  configured) is applied after scaling, per-block
  (`quad_rl.envs.disturbances.GaussianObsNoise`).
- **`info` dict** (`quad_hover_env.step`): `within_hover_threshold` (bool,
  drives `custom/fraction_within_hover_threshold`), `reward_components`
  (dict, always the same keys regardless of whether the step crashed, so
  per-episode arrays stay rectangular for plotting — keys come from
  whichever `reward.terms` are configured), `physics_params` (flat vector
  of the current episode's sampled physics parameters, every step),
  `pos_error_norm` (raw distance to target in meters, every step — read
  this rather than reverse-engineering it from a reward-term weight or a
  scaled/possibly-stacked observation). `physics_param_names` appears only
  in `reset()`'s info (static metadata, not repeated every step).
- **Vec-env workers**: `train.py`'s `make_env` factory does its own
  imports inside the returned closure. `SubprocVecEnv` defaults to the
  `forkserver`/`spawn` multiprocessing start method (not `fork`), so
  worker processes start fresh and need `quad_rl.envs` imported again to
  register `QuadHover-v0` — don't hoist those imports to module level.

## Directory layout for run artifacts

`runs/` is gitignored entirely (checkpoints and TensorBoard logs are
large and not source). Structure created by `train.py`:

```
runs/
  <run-name>/
    config.yaml                     # resolved env config + algo hyperparameters used
    final_model.zip
    final_vecnormalize.pkl
    final_model_eval.png            # from evaluate.py, if run on final_model.zip
    checkpoints/
      <algo>_quadhover_<N>_steps.zip
      <algo>_quadhover_vecnormalize_<N>_steps.pkl
      <algo>_quadhover_<N>_steps_eval.png   # evaluate.py output lands next to
                                             # whichever checkpoint you pointed it at
  tensorboard/
    <run-name>_1/                  # SB3 auto-increments on reuse
```
