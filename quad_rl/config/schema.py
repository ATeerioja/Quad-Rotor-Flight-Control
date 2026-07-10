"""Typed, validated config sections for the QuadHover-v0 environment.

Each dataclass has a from_dict classmethod (raises ValueError on unknown
keys, missing keys, or a wrong-typed value) and an asdict method (plain
Python output, safe to yaml.safe_dump) -- distinct from stdlib
dataclasses.asdict, which is only correct here for the sections that have
no numpy fields. No `from __future__ import annotations` in this module:
typing.get_type_hints() needs real type objects, not strings.
"""

import dataclasses
import typing

import numpy as np

_PHYSICS_SCALAR_FIELDS = (
    "mass",
    "arm_length",
    "thrust_coefficient",
    "drag_coefficient",
    "yaw_torque_coefficient",
    "gravity",
    "motor_time_constant",
)
_INERTIA_KEYS = ("ixx", "iyy", "izz")


def _check_number(value, expected: type, label: str):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected {expected.__name__}, got {type(value).__name__}")
    if expected is float:
        return float(value)
    if not isinstance(value, int):
        raise ValueError(f"{label}: expected int, got {type(value).__name__}")
    return value


def _validate_and_build(cls, data: dict):
    """Build a flat dataclass instance from `data`. Raises ValueError on
    unknown keys, missing keys, or a value that doesn't match its
    declared (int/float) field type."""
    hints = typing.get_type_hints(cls)
    field_names = {f.name for f in dataclasses.fields(cls)}

    unknown = set(data) - field_names
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown key(s): {sorted(unknown)}")
    missing = field_names - set(data)
    if missing:
        raise ValueError(f"{cls.__name__}: missing required key(s): {sorted(missing)}")

    kwargs = {}
    for name in field_names:
        kwargs[name] = _check_number(data[name], hints[name], f"{cls.__name__}.{name}")
    return cls(**kwargs)


@dataclasses.dataclass(frozen=True)
class SimConfig:
    dt: float

    @classmethod
    def from_dict(cls, data: dict) -> "SimConfig":
        return _validate_and_build(cls, data)

    def asdict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class EpisodeConfig:
    max_steps: int
    crash_altitude: float
    max_tilt_deg: float
    bounding_box: float

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodeConfig":
        return _validate_and_build(cls, data)

    def asdict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class RewardConfig:
    """A list of {type, weight, ...extra} term configs, e.g.
    {"type": "hover_bonus", "weight": 1.0, "threshold": 0.1}. Only
    shallowly validated here (must be a list of dicts with a "type" key) --
    deeper per-type validation happens when quad_rl.envs.rewards builds
    concrete term instances via REWARD_REGISTRY."""

    terms: list[dict]

    @classmethod
    def from_dict(cls, data: dict) -> "RewardConfig":
        known = {"terms"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"RewardConfig: unknown key(s): {sorted(unknown)}")
        if "terms" not in data:
            raise ValueError("RewardConfig: missing required key(s): ['terms']")

        terms = data["terms"]
        if not isinstance(terms, list):
            raise ValueError("RewardConfig.terms: expected a list")
        for i, term in enumerate(terms):
            if not isinstance(term, dict) or "type" not in term:
                raise ValueError(f"RewardConfig.terms[{i}]: must be a dict with a 'type' key")
        return cls(terms=[dict(t) for t in terms])

    def asdict(self) -> dict:
        return {"terms": [dict(t) for t in self.terms]}


@dataclasses.dataclass(frozen=True)
class SpawnConfig:
    position_perturbation: float
    velocity_perturbation: float
    orientation_perturbation_deg: float
    target_region: float

    @classmethod
    def from_dict(cls, data: dict) -> "SpawnConfig":
        return _validate_and_build(cls, data)

    def asdict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, eq=False)
class PhysicsConfig:
    """Bridges default.yaml's nested physics section into the flat np.array
    inertia + scalar fields dynamics.step() expects (see dynamics.py's
    module docstring).

    eq=False + a hand-written __eq__ below: the dataclass-generated __eq__
    builds an `and`-chain across fields, and since `inertia` isn't the last
    field, evaluating that chain forces a truthiness check on an elementwise
    boolean ndarray mid-chain, raising ValueError on any `==` comparison.
    """

    mass: float
    inertia: np.ndarray  # [ixx, iyy, izz], kg*m^2
    arm_length: float
    thrust_coefficient: float
    drag_coefficient: float
    yaw_torque_coefficient: float
    gravity: float
    motor_time_constant: float

    @classmethod
    def from_dict(cls, data: dict) -> "PhysicsConfig":
        known = set(_PHYSICS_SCALAR_FIELDS) | {"inertia"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"PhysicsConfig: unknown key(s): {sorted(unknown)}")
        missing = known - set(data)
        if missing:
            raise ValueError(f"PhysicsConfig: missing required key(s): {sorted(missing)}")

        inertia_dict = data["inertia"]
        unknown_i = set(inertia_dict) - set(_INERTIA_KEYS)
        if unknown_i:
            raise ValueError(f"PhysicsConfig.inertia: unknown key(s): {sorted(unknown_i)}")
        missing_i = set(_INERTIA_KEYS) - set(inertia_dict)
        if missing_i:
            raise ValueError(f"PhysicsConfig.inertia: missing key(s): {sorted(missing_i)}")

        inertia = np.array(
            [_check_number(inertia_dict[k], float, f"PhysicsConfig.inertia.{k}") for k in _INERTIA_KEYS]
        )
        kwargs = {
            name: _check_number(data[name], float, f"PhysicsConfig.{name}")
            for name in _PHYSICS_SCALAR_FIELDS
        }
        return cls(inertia=inertia, **kwargs)

    def as_params(self) -> dict:
        """Flat dict for dynamics.step's `params` argument (inertia stays
        an ndarray) -- dataclasses.asdict deep-copies non-dataclass fields
        rather than converting them, so this is already the right shape."""
        return dataclasses.asdict(self)

    def asdict(self) -> dict:
        """Plain-Python, YAML-dumpable form: inertia goes back to nested
        {ixx, iyy, izz} for round-trip symmetry with from_dict. Deliberately
        NOT dataclasses.asdict(self) -- see as_params, which needs the
        ndarray form instead."""
        return {
            "mass": self.mass,
            "inertia": {
                "ixx": float(self.inertia[0]),
                "iyy": float(self.inertia[1]),
                "izz": float(self.inertia[2]),
            },
            "arm_length": self.arm_length,
            "thrust_coefficient": self.thrust_coefficient,
            "drag_coefficient": self.drag_coefficient,
            "yaw_torque_coefficient": self.yaw_torque_coefficient,
            "gravity": self.gravity,
            "motor_time_constant": self.motor_time_constant,
        }

    def __eq__(self, other):
        if not isinstance(other, PhysicsConfig):
            return NotImplemented
        return (
            self.mass == other.mass
            and np.array_equal(self.inertia, other.inertia)
            and self.arm_length == other.arm_length
            and self.thrust_coefficient == other.thrust_coefficient
            and self.drag_coefficient == other.drag_coefficient
            and self.yaw_torque_coefficient == other.yaw_torque_coefficient
            and self.gravity == other.gravity
            and self.motor_time_constant == other.motor_time_constant
        )


@dataclasses.dataclass(frozen=True)
class DisturbanceConfig:
    """Forward-compatible stub for the wind/sensor-noise seam (Stage 1.3).
    Only validated shallowly here (dict with a "type" key) since the actual
    force/observation registries don't exist yet."""

    force: dict
    observation: dict

    @classmethod
    def from_dict(cls, data: dict | None) -> "DisturbanceConfig":
        data = data or {}
        known = {"force", "observation"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"DisturbanceConfig: unknown key(s): {sorted(unknown)}")

        force = data.get("force", {"type": "none"})
        observation = data.get("observation", {"type": "none"})
        for name, block in (("force", force), ("observation", observation)):
            if not isinstance(block, dict) or "type" not in block:
                raise ValueError(f"DisturbanceConfig.{name}: must be a dict with a 'type' key")
        return cls(force=dict(force), observation=dict(observation))

    def asdict(self) -> dict:
        return {"force": dict(self.force), "observation": dict(self.observation)}


@dataclasses.dataclass(frozen=True)
class EnvConfig:
    physics: PhysicsConfig
    simulation: SimConfig
    episode: EpisodeConfig
    reward: RewardConfig
    spawn: SpawnConfig
    disturbance: DisturbanceConfig

    @classmethod
    def from_dict(cls, data: dict) -> "EnvConfig":
        known = {"physics", "simulation", "episode", "reward", "spawn", "disturbance"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"EnvConfig: unknown top-level key(s): {sorted(unknown)}")
        missing = known - set(data)
        if missing:
            raise ValueError(f"EnvConfig: missing required top-level key(s): {sorted(missing)}")
        return cls(
            physics=PhysicsConfig.from_dict(data["physics"]),
            simulation=SimConfig.from_dict(data["simulation"]),
            episode=EpisodeConfig.from_dict(data["episode"]),
            reward=RewardConfig.from_dict(data["reward"]),
            spawn=SpawnConfig.from_dict(data["spawn"]),
            disturbance=DisturbanceConfig.from_dict(data["disturbance"]),
        )

    def asdict(self) -> dict:
        # NOT dataclasses.asdict(self) -- that would recurse generically
        # into PhysicsConfig's raw fields and bypass its asdict() override,
        # leaving inertia as an ndarray. Must delegate to each sub-config.
        return {
            "physics": self.physics.asdict(),
            "simulation": self.simulation.asdict(),
            "episode": self.episode.asdict(),
            "reward": self.reward.asdict(),
            "spawn": self.spawn.asdict(),
            "disturbance": self.disturbance.asdict(),
        }
