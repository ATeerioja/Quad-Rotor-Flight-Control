"""Pytest suite for QuadHoverEnv -- currently covers Stage 1.4's
domain-adaptation seam (parameter sampling + privileged observation).
Grows into a full env test suite in a later stage (env_checker,
determinism, crash conditions, ...).

Tests:
    1. expose_privileged=false (the default): observation_space is a plain
       Box, and rollouts are unaffected by the sampler/privileged-obs
       machinery existing at all.
    2. With a non-trivial randomization spec, two episodes with different
       seeds get different mass, and the same seed gets the same mass.
    3. expose_privileged=true: observation_space becomes a Dict, obs is a
       dict with "observation"/"privileged_obs", and info exposes
       physics_params every step plus physics_param_names once at reset.
    4. Stage 1.5 history stacking: history_length=1 is bit-identical to no
       stacking; history_length=4 at reset returns the initial obs tiled
       4x; stepping shifts the window (oldest dropped, newest appended);
       privileged_obs stays unstacked regardless of history_length.
"""

from pathlib import Path

import numpy as np
from gymnasium import spaces

from quad_rl.envs.quad_hover_env import OBS_DIM, QuadHoverEnv
from quad_rl.envs.randomization import PHYSICS_PARAM_NAMES

CONFIG_PATH = Path(__file__).resolve().parents[1] / "envs" / "configs" / "default.yaml"


def _mass_override():
    return ["randomization.spec.mass={type: uniform, lo: 0.4, hi: 0.6}"]


def test_expose_privileged_false_uses_plain_box_space():
    env = QuadHoverEnv(config_path=CONFIG_PATH)
    assert isinstance(env.observation_space, spaces.Box)
    assert env.observation_space.shape == (OBS_DIM,)


def test_default_config_randomization_is_a_no_op():
    env = QuadHoverEnv(config_path=CONFIG_PATH)
    env.reset(seed=0)
    assert env.physics_config == env.env_config.physics


def test_rollout_unchanged_with_expose_privileged_false():
    env = QuadHoverEnv(config_path=CONFIG_PATH)
    obs, info = env.reset(seed=0)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (OBS_DIM,)

    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(np.zeros(4))
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (OBS_DIM,)
        if terminated or truncated:
            break


def test_different_seeds_different_mass_same_seed_same_mass():
    env1 = QuadHoverEnv(config_path=CONFIG_PATH, overrides=_mass_override())
    env2 = QuadHoverEnv(config_path=CONFIG_PATH, overrides=_mass_override())
    env3 = QuadHoverEnv(config_path=CONFIG_PATH, overrides=_mass_override())

    env1.reset(seed=1)
    env2.reset(seed=2)
    env3.reset(seed=1)

    assert env1.physics_config.mass != env2.physics_config.mass
    assert env1.physics_config.mass == env3.physics_config.mass


def test_expose_privileged_true_uses_dict_space_with_privileged_obs():
    env = QuadHoverEnv(config_path=CONFIG_PATH, overrides=["expose_privileged=true"])
    assert isinstance(env.observation_space, spaces.Dict)

    obs, info = env.reset(seed=0)
    assert isinstance(obs, dict)
    assert set(obs.keys()) == {"observation", "privileged_obs"}
    assert obs["observation"].shape == (OBS_DIM,)
    assert obs["privileged_obs"].shape == (len(PHYSICS_PARAM_NAMES),)
    assert info["physics_param_names"] == PHYSICS_PARAM_NAMES

    obs, reward, terminated, truncated, info = env.step(np.zeros(4))
    assert isinstance(obs, dict)
    assert set(obs.keys()) == {"observation", "privileged_obs"}


def test_info_exposes_physics_params_every_step_and_names_only_at_reset():
    env = QuadHoverEnv(config_path=CONFIG_PATH)
    obs, reset_info = env.reset(seed=0)
    assert "physics_params" in reset_info
    assert reset_info["physics_params"].shape == (len(PHYSICS_PARAM_NAMES),)
    assert reset_info["physics_param_names"] == PHYSICS_PARAM_NAMES

    obs, reward, terminated, truncated, step_info = env.step(np.zeros(4))
    assert "physics_params" in step_info
    assert step_info["physics_params"].shape == (len(PHYSICS_PARAM_NAMES),)
    assert "physics_param_names" not in step_info


def test_history_length_one_is_bit_identical_to_no_stacking():
    env = QuadHoverEnv(config_path=CONFIG_PATH, overrides=["history_length=1"])
    assert env.observation_space.shape == (OBS_DIM,)

    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert np.array_equal(obs, env._get_obs())

    obs, *_ = env.step(np.zeros(4))
    assert obs.shape == (OBS_DIM,)
    assert np.array_equal(obs, env._get_obs())


def test_history_length_four_reset_returns_initial_obs_tiled():
    env = QuadHoverEnv(config_path=CONFIG_PATH, overrides=["history_length=4"])
    assert env.observation_space.shape == (OBS_DIM * 4,)

    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM * 4,)

    single_frame = obs[:OBS_DIM]
    for i in range(4):
        assert np.array_equal(obs[i * OBS_DIM : (i + 1) * OBS_DIM], single_frame)


def test_history_length_four_step_shifts_window():
    env = QuadHoverEnv(config_path=CONFIG_PATH, overrides=["history_length=4"])
    obs0, info = env.reset(seed=0)

    obs1, *_ = env.step(np.zeros(4))

    # obs0 = [f0, f0, f0, f0]; obs1 should be [f0, f0, f0, f1] -- the
    # oldest frame dropped, the new one appended at the end.
    assert np.array_equal(obs1[: OBS_DIM * 3], obs0[OBS_DIM:])
    assert np.array_equal(obs1[OBS_DIM * 3 :], env._get_obs())


def test_expose_privileged_with_history_stacks_only_observation_block():
    env = QuadHoverEnv(
        config_path=CONFIG_PATH, overrides=["expose_privileged=true", "history_length=4"]
    )
    assert isinstance(env.observation_space, spaces.Dict)
    assert env.observation_space["observation"].shape == (OBS_DIM * 4,)
    assert env.observation_space["privileged_obs"].shape == (len(PHYSICS_PARAM_NAMES),)

    obs, info = env.reset(seed=0)
    assert obs["observation"].shape == (OBS_DIM * 4,)
    assert obs["privileged_obs"].shape == (len(PHYSICS_PARAM_NAMES),)
