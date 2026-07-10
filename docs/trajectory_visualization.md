# Trajectory visualization

`viz/` is a standalone, offline visualizer for a single rollout's flight
path — a 3D animation of the vehicle's position and attitude over time. It
is entirely decoupled from training: `TrajectoryRecorder` is a
`gym.Wrapper` you attach only around a manual eval rollout, and
`viz/animate.py` only ever reads a `.npz` file afterward. Nothing under
`quad_rl/training/` imports or knows about `viz/`, and `viz/` never
modifies `QuadHoverEnv`'s `step()`/`reset()`/reward logic — it only reads
`self.unwrapped.state` after the real env has already computed it.

## Quick start: one-command animation

`viz/generate_animation.py` records a rollout and renders it in a single
run — no separate record/render steps.

1. **Generate an animation from a random policy** (no checkpoint needed —
   good for a first smoke test):
   ```bash
   python -m viz.generate_animation --out rollout.gif
   ```
   This prints how many steps it recorded, then `Saved animation to
   rollout.gif`. It also writes `rollout.npz` next to it (the intermediate
   recording), so you can re-render later without re-running the sim.

2. **Or, generate it from a trained checkpoint** instead of random actions:
   ```bash
   python -m viz.generate_animation --out rollout.mp4 \
       --checkpoint runs/my_test_run/final_model.zip \
       --vecnormalize runs/my_test_run/final_vecnormalize.pkl \
       --algo ppo
   ```
   `--vecnormalize` and `--algo` are required together with `--checkpoint`
   (unlike `evaluate.py`, this script doesn't auto-infer them — pass the
   exact paths next to your checkpoint).

3. **View the result** — see [Viewing the output](#viewing-the-output) below.

4. **Tune the render** with the same flags `viz/animate.py` takes:
   `--trail` (fading-trail length, default 50), `--fps` (default 30),
   `--arm-length` (visual arm size in meters, default 0.17), `--writer`
   (`ffmpeg`/`pillow`, normally inferred from `--out`'s suffix), and
   `--seed`/`--max-steps` to control the recorded episode. Run
   `python -m viz.generate_animation --help` for the full list.

Use this script for a quick look at a rollout. For a custom recording loop
(e.g. injecting your own policy logic, or recording alongside other
analysis), record manually as described next.

## Manual usage

### 1. Record a rollout

```python
import gymnasium as gym
import quad_rl.envs  # noqa: F401  (registers QuadHover-v0)
from viz.recorder import TrajectoryRecorder

# Wrap ONLY for a manual eval rollout -- never during SB3 training.
env = TrajectoryRecorder(gym.make("QuadHover-v0"))
obs, info = env.reset(seed=0)
terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()  # or model.predict(obs)[0] for a loaded policy
    obs, reward, terminated, truncated, info = env.step(action)

env.save("rollout.npz")
```

To visualize a trained checkpoint instead of a random policy, load it the
same way `quad_rl/training/evaluate.py` does (`load_policy(checkpoint,
vecnormalize, algo, env_config)`) and call
`model.predict(vecnormalize.normalize_obs(obs[None]))` in place of
`env.action_space.sample()` above — this is exactly what
`viz/generate_animation.py`'s `--checkpoint` path does under the hood.

`TrajectoryRecorder` logs one frame at `reset()` (the initial pose) and one
per `step()` call, into an in-memory buffer that `.save(path)` writes to a
`.npz`:

| key | shape | dtype | contents |
|---|---|---|---|
| `positions` | `(N, 3)` | float64 | world-frame position (m), frame 0 = initial reset pose |
| `quaternions` | `(N, 4)` | float64 | scalar-first `[w, x, y, z]`, body→world, same convention as `quad_rl/envs/dynamics.py` |
| `times` | `(N,)` | float64 | seconds, `elapsed_steps * dt` |
| `dt` | scalar | float64 | the env's fixed timestep, used to pick the animation's default playback speed |

Calling `env.reset()` again on the same wrapped instance starts a fresh
recording (the buffers are cleared), so each `.npz` holds exactly one
episode.

### 2. Render the animation

```bash
python -m viz.animate --file rollout.npz --out rollout.mp4
# or, if ffmpeg isn't installed on your machine:
python -m viz.animate --file rollout.npz --out rollout.gif
```

The quadrotor is drawn as a rigid X-configuration cross (two crossed arm
lines + a rotor marker at each of the 4 tips), rotated every frame by the
logged quaternion via `dynamics.quat_to_rotmat`, with a fading trail of
past positions behind it.

| flag | default | meaning |
|---|---|---|
| `--file` | *required* | input `.npz` written by `TrajectoryRecorder.save()` |
| `--out` | *required* | output path; `.mp4` uses the `ffmpeg` writer, `.gif` uses the `pillow` writer (inferred from the suffix) |
| `--trail` | 50 | number of past positions shown in the fading trail |
| `--fps` | 30 | output frame rate |
| `--arm-length` | 0.17 | visual arm length in meters (matches `configs/base.yaml`'s default `physics.arm_length`; the `.npz` only stores kinematic state, so this is purely a rendering constant — override it if the rollout used a very different `arm_length`) |
| `--writer` | inferred from `--out` | force `ffmpeg` or `pillow` instead of inferring from the output suffix |

## Viewing the output

- **`.gif`** — opens in any browser or image viewer (`xdg-open rollout.gif` on
  Linux), and renders inline in a Jupyter notebook via
  `IPython.display.Image("rollout.gif")` or directly in a GitHub PR/README.
- **`.mp4`** — opens in any video player (`vlc rollout.mp4`,
  `xdg-open rollout.mp4`), or inline in a notebook via
  `IPython.display.Video("rollout.mp4")`.

## Dependencies

`matplotlib` (already a project dependency) ships the `pillow` writer's
requirements and the `mplot3d` toolkit, so `.gif` export works out of the
box. `.mp4` export needs a system `ffmpeg` binary on `PATH` (not a pip
package) — install it via your OS package manager (e.g. `apt install
ffmpeg`, `brew install ffmpeg`). Check what's available with:

```bash
python -c "import matplotlib.animation as a; print(a.writers.is_available('ffmpeg'), a.writers.is_available('pillow'))"
```

If you request a writer that isn't installed, both `viz/animate.py` and
`viz/generate_animation.py` exit with a clear error naming the missing
dependency rather than a matplotlib traceback.
