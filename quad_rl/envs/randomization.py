"""Per-episode physics parameter sampling: the domain-randomization/
adaptation seam. A ParameterSampler resamples a subset of PhysicsConfig's
fields (addressed by dotted path, e.g. "mass", "inertia.ixx") from
per-field Distributions each reset(); any field not named in the spec
keeps its nominal (config-file) value. An empty (or all-Fixed-at-nominal)
spec is a no-op -- exactly Stage 1's setting.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from quad_rl.config.schema import PhysicsConfig

# Fixed order for the flat "privileged" physics-parameter vector exposed
# in info["physics_params"] / the privileged_obs observation block.
PHYSICS_PARAM_NAMES = [
    "mass",
    "inertia.ixx",
    "inertia.iyy",
    "inertia.izz",
    "arm_length",
    "thrust_coefficient",
    "drag_coefficient",
    "yaw_torque_coefficient",
    "gravity",
    "motor_time_constant",
]


def physics_config_to_vector(physics: PhysicsConfig) -> np.ndarray:
    """Flatten a PhysicsConfig into the fixed-order vector described by
    PHYSICS_PARAM_NAMES -- the privileged signal domain-adaptation methods
    consume (RMA-style teacher input, sysID regression target, ...)."""
    return np.array([
        physics.mass,
        physics.inertia[0],
        physics.inertia[1],
        physics.inertia[2],
        physics.arm_length,
        physics.thrust_coefficient,
        physics.drag_coefficient,
        physics.yaw_torque_coefficient,
        physics.gravity,
        physics.motor_time_constant,
    ])


class Distribution(Protocol):
    def sample(self, rng: np.random.Generator) -> float: ...


DISTRIBUTION_REGISTRY: dict[str, type[Distribution]] = {}


def register_distribution(name: str):
    def decorator(cls):
        DISTRIBUTION_REGISTRY[name] = cls
        return cls

    return decorator


@register_distribution("uniform")
class Uniform:
    def __init__(self, lo: float, hi: float):
        self.lo = lo
        self.hi = hi

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.lo, self.hi))


@register_distribution("log_uniform")
class LogUniform:
    def __init__(self, lo: float, hi: float):
        self.lo = lo
        self.hi = hi

    def sample(self, rng: np.random.Generator) -> float:
        return float(np.exp(rng.uniform(np.log(self.lo), np.log(self.hi))))


@register_distribution("fixed")
class Fixed:
    def __init__(self, value: float):
        self.value = value

    def sample(self, rng: np.random.Generator) -> float:
        return self.value


def _set_dotted(values: dict, dotted_path: str, value: float) -> None:
    parts = dotted_path.split(".")
    node = values
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


class ParameterSampler:
    """spec maps dotted paths into PhysicsConfig (e.g. "mass",
    "inertia.ixx") to Distributions. Any field not named in spec keeps
    base_physics's nominal value every episode."""

    def __init__(self, spec: dict[str, Distribution], base_physics: PhysicsConfig):
        self.spec = spec
        self.base_physics = base_physics

    def sample(self, rng: np.random.Generator) -> PhysicsConfig:
        values = self.base_physics.asdict()
        for dotted_path, distribution in self.spec.items():
            _set_dotted(values, dotted_path, distribution.sample(rng))
        return PhysicsConfig.from_dict(values)

    @classmethod
    def from_config(cls, randomization_config, base_physics: PhysicsConfig) -> "ParameterSampler":
        spec = {}
        for dotted_path, dist_cfg in randomization_config.spec.items():
            dist_cfg = dict(dist_cfg)
            dist_cls = DISTRIBUTION_REGISTRY[dist_cfg.pop("type")]
            spec[dotted_path] = dist_cls(**dist_cfg)
        return cls(spec, base_physics)
