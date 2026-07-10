# Domain adaptation

Stage 1.4 added the seam this document describes; it does not itself
implement any of the three approaches below. The seam is two pieces:

- `quad_rl.envs.randomization.ParameterSampler` resamples a subset of
  `PhysicsConfig` (addressed by dotted path — `mass`, `inertia.ixx`,
  `drag_coefficient`, ...) from per-field `Distribution`s (`Uniform`,
  `LogUniform`, `Fixed`) every `reset()`. With an empty (or
  all-`Fixed`-at-nominal) `randomization.spec`, this is a no-op — Stage 1's
  setting.
- `env_config.expose_privileged: bool` — when `true`, `QuadHoverEnv`'s
  `observation_space` becomes a `gym.spaces.Dict` with an `"observation"`
  key (the usual 19-dim `Box`, unchanged) and a `"privileged_obs"` key: the
  sampled `PhysicsConfig` flattened to a fixed-order vector
  (`quad_rl.envs.randomization.PHYSICS_PARAM_NAMES`). The same vector is
  also exposed every step in `info["physics_params"]`, with
  `info["physics_param_names"]` given once at `reset()`. When `false`
  (the default), `observation_space` is a plain `Box` exactly as before
  this seam existed, so SB3's `MlpPolicy` keeps working untouched.

One flag (`expose_privileged`) plus one config block (`randomization.spec`)
covers three different domain-adaptation strategies without further env
surgery:

## 1. Domain randomization only

Train a single policy across the sampled parameter distribution and hope
it generalizes, with no privileged information used at all.

- **Config**: a non-trivial `randomization.spec` (e.g. `mass: {type:
  uniform, lo: 0.4, hi: 0.6}`), `expose_privileged: false`.
- **Training**: unchanged — `MlpPolicy` on the plain `Box` observation, no
  new heads or losses. The policy only ever sees the usual 19-dim
  observation; robustness comes purely from having encountered a range of
  dynamics during training.
- **Extra pieces needed**: none beyond what Stage 1.4 already ships. This
  is the cheapest strategy to try and a reasonable first baseline before
  reaching for the other two.
- **Limitation**: the policy can't specialize to the specific plant it's
  actually flying — it's forced to find one policy that's merely adequate
  across the whole sampled range, which typically costs some
  best-case performance relative to a policy trained (or adapted) with
  knowledge of the true parameters.

## 2. RMA-style two-phase adaptation

("Rapid Motor Adaptation" — train a teacher with privileged access to the
true physics parameters, then train a student that infers them from
proprioceptive history alone.)

- **Config**: a non-trivial `randomization.spec`, `expose_privileged: true`.
- **Phase 1 (teacher)**: policy input is `{"observation": ..., "privileged_obs":
  ...}` via `MultiInputPolicy`, trained normally (e.g. PPO) across the
  randomized distribution. The teacher directly conditions on the true
  sampled parameters, so it can specialize per-episode rather than
  averaging over the distribution.
- **Phase 2 (student)**: a small encoder is trained to regress an estimate
  of `privileged_obs` from a short window of past `(observation, action)`
  pairs (supervised regression against the teacher's `privileged_obs`
  target, collected via rollouts). The student then replaces
  `privileged_obs` with its own estimate at deployment time, when the true
  parameters aren't observable.
- **Extra pieces needed beyond Stage 1.4**:
  - Observation history / frame stacking so the student has something to
    regress from (a later Stage 1 prompt).
  - A second training loop / script for the student encoder, plus the
    supervised regression loss and its own data collection pass.
  - `MultiInputPolicy` wiring in the algorithm-selection code (a later
    Stage 1 prompt) so `expose_privileged: true` is picked up automatically.

## 3. Explicit system identification

Train an auxiliary head to predict the true physics parameters from
observation history, and either condition the policy on that prediction
or use it purely as an auxiliary/diagnostic loss.

- **Config**: same as RMA (non-trivial `randomization.spec`,
  `expose_privileged: true`) — the difference from RMA is entirely in how
  training is structured, not in the env/config seam.
- **Training**: a single network (or a shared trunk with two heads) is
  trained jointly: one head produces the control action, the other
  predicts `privileged_obs` from the policy's own input history, supervised
  directly against the env's `info["physics_params"]` at every step. Unlike
  RMA's two-phase teacher/student split, there's no separate teacher
  policy — the sysID head trains alongside the policy from the start.
- **Extra pieces needed beyond Stage 1.4**:
  - Observation history (same as RMA, since a single step's observation
    generally underdetermines dynamics parameters like mass or inertia).
  - A custom policy/algorithm wrapper adding the auxiliary prediction head
    and its loss term to the training loop (SB3's stock algorithms don't
    support this directly — this is the piece Stage 1.6's algorithm
    abstraction is meant to make tractable to add).
  - A decision on whether the predicted parameters feed back into the
    control head's input (closed-loop sysID) or stay a side-channel
    diagnostic/loss only (open-loop) — this changes the network
    architecture, not the env or config.

## Summary

| Approach | Uses `privileged_obs` at train time | Uses it at deploy time | Extra pieces |
|---|---|---|---|
| Domain randomization only | No | No | None |
| RMA two-phase | Yes (teacher) | No (student estimates it) | History, student training loop |
| Explicit system ID | Yes (as regression target) | Optional (open- vs. closed-loop) | History, auxiliary head + loss |

All three read the same `env_config.expose_privileged` flag and the same
`randomization.spec` format — the choice between them is a training-loop
decision, not an environment one.
