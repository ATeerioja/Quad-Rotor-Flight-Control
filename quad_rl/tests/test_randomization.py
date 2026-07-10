"""Pytest suite for quad_rl.envs.randomization -- the per-episode physics
parameter sampler (domain-randomization/adaptation seam).

Tests:
    1. Uniform/LogUniform/Fixed sample within their expected ranges (or
       exactly, for Fixed).
    2. An empty spec is a no-op: sample() returns the base PhysicsConfig
       unchanged.
    3. A dotted path like "inertia.ixx" updates only that one component of
       PhysicsConfig.inertia, leaving iyy/izz untouched.
    4. ParameterSampler.from_config builds correctly from raw config dicts
       via DISTRIBUTION_REGISTRY.
    5. DISTRIBUTION_REGISTRY contains all three spec'd distribution types.
"""

from pathlib import Path

import numpy as np
import pytest

from quad_rl.config.loader import load_config
from quad_rl.envs.randomization import (
    DISTRIBUTION_REGISTRY,
    Fixed,
    LogUniform,
    ParameterSampler,
    Uniform,
    physics_config_to_vector,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "envs" / "configs" / "default.yaml"


@pytest.fixture
def base_physics():
    return load_config(CONFIG_PATH).physics


def test_uniform_samples_within_bounds():
    dist = Uniform(lo=0.4, hi=0.6)
    rng = np.random.default_rng(0)
    for _ in range(1000):
        value = dist.sample(rng)
        assert 0.4 <= value <= 0.6


def test_log_uniform_samples_within_bounds_and_is_positive():
    dist = LogUniform(lo=0.001, hi=0.1)
    rng = np.random.default_rng(0)
    for _ in range(1000):
        value = dist.sample(rng)
        assert 0.001 <= value <= 0.1


def test_fixed_always_returns_same_value_regardless_of_rng():
    dist = Fixed(value=0.5)
    rng = np.random.default_rng(0)
    assert dist.sample(rng) == 0.5
    assert dist.sample(np.random.default_rng(999)) == 0.5


def test_empty_spec_is_a_no_op(base_physics):
    sampler = ParameterSampler(spec={}, base_physics=base_physics)
    sampled = sampler.sample(np.random.default_rng(0))
    assert sampled == base_physics


def test_dotted_inertia_path_updates_only_that_component(base_physics):
    sampler = ParameterSampler(spec={"inertia.ixx": Fixed(0.999)}, base_physics=base_physics)
    sampled = sampler.sample(np.random.default_rng(0))

    assert sampled.inertia[0] == pytest.approx(0.999)
    assert sampled.inertia[1] == pytest.approx(base_physics.inertia[1])
    assert sampled.inertia[2] == pytest.approx(base_physics.inertia[2])
    assert sampled.mass == base_physics.mass


def test_mass_sampling_stays_within_configured_range(base_physics):
    sampler = ParameterSampler(spec={"mass": Uniform(lo=0.4, hi=0.6)}, base_physics=base_physics)
    rng = np.random.default_rng(0)
    for _ in range(100):
        sampled = sampler.sample(rng)
        assert 0.4 <= sampled.mass <= 0.6
        # Only mass was randomized; everything else stays at nominal.
        assert np.array_equal(sampled.inertia, base_physics.inertia)
        assert sampled.arm_length == base_physics.arm_length


def test_from_config_builds_sampler_from_raw_dicts(base_physics):
    from quad_rl.config.schema import RandomizationConfig

    randomization_config = RandomizationConfig.from_dict({
        "spec": {
            "mass": {"type": "uniform", "lo": 0.4, "hi": 0.6},
            "drag_coefficient": {"type": "fixed", "value": 0.2},
        }
    })
    sampler = ParameterSampler.from_config(randomization_config, base_physics)

    assert isinstance(sampler.spec["mass"], Uniform)
    assert isinstance(sampler.spec["drag_coefficient"], Fixed)

    sampled = sampler.sample(np.random.default_rng(0))
    assert 0.4 <= sampled.mass <= 0.6
    assert sampled.drag_coefficient == 0.2


def test_distribution_registry_has_spec_types():
    assert {"uniform", "log_uniform", "fixed"} <= set(DISTRIBUTION_REGISTRY)


def test_physics_config_to_vector_matches_field_order(base_physics):
    vector = physics_config_to_vector(base_physics)
    assert vector.shape == (10,)
    assert vector[0] == base_physics.mass
    assert vector[1] == base_physics.inertia[0]
    assert vector[2] == base_physics.inertia[1]
    assert vector[3] == base_physics.inertia[2]
    assert vector[9] == base_physics.motor_time_constant
