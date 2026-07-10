"""Pytest suite for QuadHoverEnv -- the full env test suite (Stage 1.8),
built up incrementally alongside the domain-adaptation seam (Stage 1.4)
and history stacking (Stage 1.5).

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
    5. gymnasium.utils.env_checker.check_env passes for both
       expose_privileged settings.
    6. Determinism: same seed -> identical trajectory; different seed ->
       different trajectory.
    7. Each crash condition (crash_altitude, max_tilt_deg, bounding_box)
       fires when driven directly into it, and reward_components keys
       stay constant across crash and non-crash steps.
    8. truncated at exactly max_steps, terminated False at that moment.
    9. Registry round-trip: every type string in every shipped env config
       (reward terms, disturbance force/observation, randomization spec)
       resolves in its registry -- catches config/code drift, the main
       failure mode of a registry-heavy design.
"""

from pathlib import Path

import numpy as np
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from quad_rl.config.loader import load_raw_config
from quad_rl.envs import dynamics
from quad_rl.envs.disturbances import FORCE_REGISTRY, OBSERVATION_REGISTRY
from quad_rl.envs.quad_hover_env import OBS_DIM, QuadHoverEnv
from quad_rl.envs.randomization import DISTRIBUTION_REGISTRY, PHYSICS_PARAM_NAMES
from quad_rl.envs.rewards import REWARD_REGISTRY

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "envs" / "configs"
CONFIG_PATH = CONFIGS_DIR / "default.yaml"


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


def test_check_env_passes_expose_privileged_false():
    env = QuadHoverEnv(config_path=CONFIG_PATH)
    check_env(env, skip_render_check=True)


def test_check_env_passes_expose_privileged_true():
    env = QuadHoverEnv(config_path=CONFIG_PATH, overrides=["expose_privileged=true"])
    check_env(env, skip_render_check=True)


def test_same_seed_gives_identical_trajectory():
    env1 = QuadHoverEnv(config_path=CONFIG_PATH)
    env2 = QuadHoverEnv(config_path=CONFIG_PATH)
    obs1, info1 = env1.reset(seed=42)
    obs2, info2 = env2.reset(seed=42)
    assert np.array_equal(obs1, obs2)

    action_rng = np.random.default_rng(0)
    for _ in range(20):
        action = action_rng.uniform(-1.0, 1.0, size=4)
        o1, r1, term1, trunc1, _ = env1.step(action.copy())
        o2, r2, term2, trunc2, _ = env2.step(action.copy())
        assert np.array_equal(o1, o2)
        assert r1 == r2
        assert term1 == term2
        assert trunc1 == trunc2
        if term1 or trunc1:
            break


def test_different_seed_gives_different_trajectory():
    env1 = QuadHoverEnv(config_path=CONFIG_PATH)
    env2 = QuadHoverEnv(config_path=CONFIG_PATH)
    obs1, _ = env1.reset(seed=1)
    obs2, _ = env2.reset(seed=2)
    assert not np.array_equal(obs1, obs2)


def _crash_scenario_env(overrides=None):
    env = QuadHoverEnv(config_path=CONFIG_PATH, overrides=overrides)
    env.reset(seed=0)
    return env


def test_crash_altitude_condition_fires():
    env = _crash_scenario_env()
    env.state[dynamics.POS][2] = env.env_config.episode.crash_altitude - 10.0

    _, reward, terminated, truncated, info = env.step(np.zeros(4))

    assert terminated is True
    assert truncated is False
    assert info["reward_components"]["crash_penalty"] < 0.0


def test_max_tilt_condition_fires():
    env = _crash_scenario_env()
    # 90 degrees about the body x-axis -- well past the default 60 degree
    # max_tilt_deg, and rotational dynamics are far too slow over one
    # dt=0.01s step to recover before the crash check runs.
    env.state[dynamics.QUAT] = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]
    env.state[dynamics.POS] = env.target.copy()  # keep altitude/bbox uninvolved

    _, reward, terminated, truncated, info = env.step(np.zeros(4))

    assert terminated is True
    assert info["reward_components"]["crash_penalty"] < 0.0


def test_bounding_box_condition_fires():
    env = _crash_scenario_env()
    bbox = env.env_config.episode.bounding_box
    env.state[dynamics.POS] = np.array([bbox + 10.0, 0.0, 1.5])  # level, safe altitude

    _, reward, terminated, truncated, info = env.step(np.zeros(4))

    assert terminated is True
    assert info["reward_components"]["crash_penalty"] < 0.0


def test_reward_components_keys_constant_across_crash_and_normal_steps():
    env = _crash_scenario_env()
    _, _, terminated_normal, _, info_normal = env.step(np.zeros(4))
    assert terminated_normal is False
    normal_keys = set(info_normal["reward_components"])

    env.state[dynamics.POS][2] = env.env_config.episode.crash_altitude - 10.0
    _, _, terminated_crash, _, info_crash = env.step(np.zeros(4))
    assert terminated_crash is True
    crash_keys = set(info_crash["reward_components"])

    assert normal_keys == crash_keys


def test_truncated_at_exactly_max_steps_terminated_false():
    max_steps = 5
    env = QuadHoverEnv(config_path=CONFIG_PATH, overrides=[f"episode.max_steps={max_steps}"])
    env.reset(seed=0)

    # Near-hover thrust (action=0 -> motor_command=0.5, close to the
    # nominal hover command) over a handful of 0.01s steps from a small
    # spawn perturbation won't crash, isolating the timeout path.
    for i in range(max_steps):
        _, _, terminated, truncated, _ = env.step(np.zeros(4))
        if i < max_steps - 1:
            assert truncated is False
        else:
            assert truncated is True
            assert terminated is False


def test_registry_round_trip_for_all_shipped_env_configs():
    for name in ("default.yaml", "hover.yaml"):
        raw = load_raw_config(CONFIGS_DIR / name)

        for term in raw["reward"]["terms"]:
            assert term["type"] in REWARD_REGISTRY, f"{name}: unknown reward type {term['type']!r}"

        assert raw["disturbance"]["force"]["type"] in FORCE_REGISTRY, f"{name}: unknown force type"
        assert raw["disturbance"]["observation"]["type"] in OBSERVATION_REGISTRY, (
            f"{name}: unknown observation-noise type"
        )

        for path, dist_cfg in raw["randomization"]["spec"].items():
            assert dist_cfg["type"] in DISTRIBUTION_REGISTRY, (
                f"{name}: randomization.spec[{path!r}]: unknown distribution type {dist_cfg['type']!r}"
            )
