"""Pytest suite for quad_rl.envs.rewards -- the pluggable reward-term
registry and RewardFunction, which replaced QuadHoverEnv's old hardcoded
_compute_reward dict comprehension.

Tests:
    1. A config-built RewardFunction reproduces the exact numbers the old
       _compute_reward formulas would have produced, for a fixed
       synthetic state/action pair.
    2. On a crash step, every component except crash_penalty is zeroed and
       crash_penalty equals -weight -- the "crash overrides" invariant.
    3. reward_components has identical keys whether or not the step
       crashed (what eval_rollout.py's per-episode stacking depends on).
    4. All six spec'd reward types are present in REWARD_REGISTRY.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from quad_rl.config.loader import load_config
from quad_rl.envs import dynamics as dyn
from quad_rl.envs.rewards import REWARD_REGISTRY, RewardFunction, StepContext

CONFIG_PATH = Path(__file__).resolve().parents[1] / "envs" / "configs" / "default.yaml"


@pytest.fixture
def reward_config():
    return load_config(CONFIG_PATH).reward


@pytest.fixture
def ctx():
    state = np.zeros(dyn.STATE_DIM)
    state[dyn.QUAT] = [1.0, 0.0, 0.0, 0.0]
    state[dyn.POS] = [0.05, -0.02, 1.53]
    state[dyn.OMEGA] = [0.1, -0.2, 0.3]
    prev_state = np.zeros(dyn.STATE_DIM)
    prev_state[dyn.QUAT] = [1.0, 0.0, 0.0, 0.0]

    action = np.array([0.1, -0.1, 0.2, -0.2])
    prev_action = np.array([0.05, -0.05, 0.1, -0.1])
    target = np.array([0.0, 0.0, 1.5])

    return StepContext(
        state=state,
        prev_state=prev_state,
        action=action,
        prev_action=prev_action,
        target=target,
        dt=0.01,
        crashed=False,
    )


def _expected_components(ctx: StepContext) -> dict:
    """The old _compute_reward/step() arithmetic, hand-reproduced."""
    pos_error_norm = np.linalg.norm(ctx.state[dyn.POS] - ctx.target)
    omega = ctx.state[dyn.OMEGA]

    if ctx.crashed:
        return {
            "position": 0.0,
            "angular_velocity": 0.0,
            "action_magnitude": 0.0,
            "action_rate": 0.0,
            "hover_bonus": 0.0,
            "crash_penalty": -100.0,
        }

    return {
        "position": -1.0 * pos_error_norm,
        "angular_velocity": -0.1 * np.linalg.norm(omega),
        "action_magnitude": -0.01 * np.linalg.norm(ctx.action),
        "action_rate": -0.01 * np.linalg.norm(ctx.action - ctx.prev_action),
        "hover_bonus": 1.0 if pos_error_norm < 0.1 else 0.0,
        "crash_penalty": 0.0,
    }


def test_reward_function_reproduces_old_formula(reward_config, ctx):
    reward_function = RewardFunction.from_config(reward_config)
    total, components = reward_function(ctx)

    expected = _expected_components(ctx)
    assert components == pytest.approx(expected)
    assert total == pytest.approx(sum(expected.values()))


def test_crash_overrides_zeroes_all_other_terms(reward_config, ctx):
    reward_function = RewardFunction.from_config(reward_config)
    crashed_ctx = dataclasses.replace(ctx, crashed=True)
    total, components = reward_function(crashed_ctx)

    expected = _expected_components(crashed_ctx)
    assert components == pytest.approx(expected)
    assert components["crash_penalty"] == pytest.approx(-100.0)
    for key, value in components.items():
        if key != "crash_penalty":
            assert value == 0.0
    assert total == pytest.approx(-100.0)


def test_reward_components_keys_stable_across_crash(reward_config, ctx):
    reward_function = RewardFunction.from_config(reward_config)
    _, normal_components = reward_function(ctx)
    crashed_ctx = dataclasses.replace(ctx, crashed=True)
    _, crashed_components = reward_function(crashed_ctx)

    assert set(normal_components) == set(crashed_components)


def test_registry_has_all_spec_terms():
    expected_types = {
        "position",
        "angular_velocity",
        "action_magnitude",
        "action_rate",
        "hover_bonus",
        "crash_penalty",
    }
    assert expected_types <= set(REWARD_REGISTRY)
