"""Pytest suite for quad_rl.envs.disturbances -- the domain-adaptation seam
(wind force + observation noise) wired into QuadHoverEnv/dynamics.step.

Tests:
    1. NoWind/NoNoise are true no-ops, and default (none/none) config
       consumes zero rng draws and matches hand-reproduced pre-seam
       arithmetic exactly, step by step.
    2. ConstantWind returns a defensive copy (external mutation of the
       returned force can't corrupt internal state).
    3. OrnsteinUhlenbeckWind: starts deterministically at mu with no rng
       draw, is reproducible under a seeded np.random.Generator, and has
       the right stationary variance over a long sample.
    4. GaussianObsNoise only perturbs configured blocks and rejects an
       unknown block name.
    5. Both registries contain the spec'd type strings, and the builders
       construct correctly from the spec's own example config dicts.
    6. Env-level: OU wind state restarts at mu each reset(), and an
       ou_wind rollout diverges from a none rollout under the same seed.
    7. Drift guard: the observation block slices tile the full 19-dim
       observation with no gaps or overlaps.
"""

from pathlib import Path

import numpy as np
import pytest

from quad_rl.envs import dynamics as dyn
from quad_rl.envs.disturbances import (
    _OBS_BLOCKS,
    FORCE_REGISTRY,
    OBSERVATION_REGISTRY,
    ConstantWind,
    GaussianObsNoise,
    NoNoise,
    NoWind,
    OrnsteinUhlenbeckWind,
    build_force_disturbance,
    build_observation_noise,
)
from quad_rl.envs.quad_hover_env import (
    ANGULAR_VELOCITY_SCALE,
    LINEAR_VELOCITY_SCALE,
    OBS_DIM,
    POSITION_ERROR_SCALE,
    QuadHoverEnv,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "envs" / "configs" / "default.yaml"


def test_no_wind_returns_zero_force():
    wind = NoWind()
    force = wind.force(np.zeros(dyn.STATE_DIM), 0.0, np.random.default_rng(0))
    assert np.array_equal(force, np.zeros(3))


def test_no_noise_is_identity():
    noise = NoNoise()
    obs = np.arange(19, dtype=np.float32)
    assert noise.apply(obs, np.random.default_rng(0)) is obs


def test_constant_wind_returns_a_copy():
    vector = np.array([1.0, 2.0, 3.0])
    wind = ConstantWind(vector)
    force = wind.force(np.zeros(dyn.STATE_DIM), 0.0, np.random.default_rng(0))
    force[0] = 999.0
    assert wind.vector[0] == 1.0


def test_no_op_disturbances_consume_no_rng_draws():
    env = QuadHoverEnv(config_path=CONFIG_PATH)
    env.reset(seed=0)
    state_before = env.np_random.bit_generator.state
    env.step(np.zeros(dyn.ACTION_DIM))
    state_after = env.np_random.bit_generator.state
    assert state_before == state_after


def test_no_wind_no_noise_matches_raw_dynamics_and_obs():
    env = QuadHoverEnv(config_path=CONFIG_PATH)
    env.reset(seed=123)
    action_rng = np.random.default_rng(999)

    state = env.state.copy()
    target = env.target.copy()

    for _ in range(20):
        action = action_rng.uniform(-1.0, 1.0, size=dyn.ACTION_DIM)
        obs_env, _, terminated, truncated, _ = env.step(action)

        clipped = np.clip(action, -1.0, 1.0)
        motor_command = (clipped + 1.0) / 2.0
        state = dyn.step(state, motor_command, env.dt, env.physics_params)

        pos_error = state[dyn.POS] - target
        velocity = state[dyn.VEL]
        omega = state[dyn.OMEGA]
        rotmat = dyn.quat_to_rotmat(state[dyn.QUAT])
        rot6d = np.concatenate([rotmat[:, 0], rotmat[:, 1]])
        expected_obs = np.concatenate([
            pos_error / POSITION_ERROR_SCALE,
            velocity / LINEAR_VELOCITY_SCALE,
            rot6d,
            omega / ANGULAR_VELOCITY_SCALE,
            clipped,
        ]).astype(np.float32)

        assert np.array_equal(obs_env, expected_obs)

        if terminated or truncated:
            break


def test_ou_wind_first_call_returns_mu_with_no_rng_draw():
    wind = OrnsteinUhlenbeckWind(theta=0.15, sigma=0.3, mu=[1.0, 2.0, 3.0])
    rng = np.random.default_rng(0)
    state_before = rng.bit_generator.state

    force = wind.force(np.zeros(dyn.STATE_DIM), 0.0, rng)

    assert np.array_equal(force, np.array([1.0, 2.0, 3.0]))
    assert rng.bit_generator.state == state_before


def test_ou_wind_reproducible_with_seeded_rng():
    wind1 = OrnsteinUhlenbeckWind(theta=0.15, sigma=0.3)
    wind2 = OrnsteinUhlenbeckWind(theta=0.15, sigma=0.3)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    state = np.zeros(dyn.STATE_DIM)

    for t in np.arange(0.0, 1.0, 0.01):
        f1 = wind1.force(state, t, rng1)
        f2 = wind2.force(state, t, rng2)
        assert np.array_equal(f1, f2)


def test_ou_wind_stationary_variance():
    theta, sigma, dt = 0.15, 0.3, 0.01
    wind = OrnsteinUhlenbeckWind(theta=theta, sigma=sigma)
    rng = np.random.default_rng(0)
    state = np.zeros(dyn.STATE_DIM)

    n_steps = 200_000
    values = np.empty((n_steps, 3))
    t = 0.0
    for i in range(n_steps):
        values[i] = wind.force(state, t, rng)
        t += dt

    burn_in = 20_000
    sample = values[burn_in:].reshape(-1)  # flatten all 3 independent axes together
    empirical_var = sample.var()
    theoretical_var = sigma**2 / (2 * theta)
    assert empirical_var == pytest.approx(theoretical_var, rel=0.15)


def test_gaussian_obs_noise_only_perturbs_configured_blocks():
    noise = GaussianObsNoise(std={"pos_error": 0.05})
    rng = np.random.default_rng(0)
    obs = np.arange(19, dtype=np.float32)

    noisy = noise.apply(obs, rng)

    assert not np.array_equal(noisy[0:3], obs[0:3])
    assert np.array_equal(noisy[3:], obs[3:])


def test_gaussian_obs_noise_rejects_unknown_block():
    with pytest.raises(ValueError):
        GaussianObsNoise(std={"bogus_block": 0.1})


def test_force_registry_has_spec_types():
    assert {"none", "constant", "ou_wind"} <= set(FORCE_REGISTRY)


def test_observation_registry_has_spec_types():
    assert {"none", "gaussian"} <= set(OBSERVATION_REGISTRY)


def test_build_force_disturbance_from_example_config():
    disturbance = build_force_disturbance({"type": "ou_wind", "sigma": 0.3, "theta": 0.15})
    assert isinstance(disturbance, OrnsteinUhlenbeckWind)


def test_build_observation_noise_from_example_config():
    noise = build_observation_noise(
        {"type": "gaussian", "std": {"pos_error": 0.01, "vel": 0.02, "omega": 0.05}}
    )
    assert isinstance(noise, GaussianObsNoise)


def _ou_wind_overrides():
    return [
        "disturbance.force.type=ou_wind",
        "disturbance.force.theta=0.15",
        "disturbance.force.sigma=0.3",
    ]


def test_ou_wind_restarts_at_mu_each_episode():
    env = QuadHoverEnv(config_path=CONFIG_PATH, overrides=_ou_wind_overrides())
    env.reset(seed=0)
    for _ in range(10):
        env.step(np.zeros(dyn.ACTION_DIM))
    assert not np.array_equal(env.force_disturbance._value, env.force_disturbance.mu)

    env.reset(seed=1)
    assert np.array_equal(env.force_disturbance._value, env.force_disturbance.mu)


def test_env_rollout_differs_with_wind_enabled():
    env_none = QuadHoverEnv(config_path=CONFIG_PATH)
    env_wind = QuadHoverEnv(config_path=CONFIG_PATH, overrides=_ou_wind_overrides())
    env_none.reset(seed=0)
    env_wind.reset(seed=0)

    action = np.zeros(dyn.ACTION_DIM)
    obs_none = obs_wind = None
    for _ in range(5):
        obs_none, *_ = env_none.step(action)
        obs_wind, *_ = env_wind.step(action)

    assert not np.array_equal(obs_none, obs_wind)


def test_obs_blocks_cover_full_observation():
    blocks = sorted(_OBS_BLOCKS.values(), key=lambda s: s.start)
    covered = 0
    for block in blocks:
        assert block.start == covered
        covered = block.stop
    assert covered == OBS_DIM
