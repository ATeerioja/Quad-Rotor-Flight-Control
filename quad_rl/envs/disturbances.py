"""Domain-adaptation seam: pluggable force disturbances (wind) and
observation noise (sensor error), registered the same way as rewards.py's
RewardTerm registry. dynamics.py imports nothing from here -- it only
gains an optional `external_force` parameter, with zero knowledge of
what produces it.

Stage 1 defaults are none/none (see configs/hover.yaml); Stage 2 turns
these on via config.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

# Mirrors quad_hover_env.py's observation layout docstring (pos_error(0:3),
# vel(3:6), rot6d(6:12), omega(12:15), prev_action(15:19)). Duplicated
# rather than imported to avoid a disturbances.py <-> quad_hover_env.py
# import cycle (quad_hover_env.py imports the builder functions below);
# test_disturbances.py cross-checks these slices against
# quad_hover_env.OBS_DIM to catch drift.
_OBS_BLOCKS: dict[str, slice] = {
    "pos_error": slice(0, 3),
    "vel": slice(3, 6),
    "rot6d": slice(6, 12),
    "omega": slice(12, 15),
    "prev_action": slice(15, 19),
}


class ForceDisturbance(Protocol):
    def force(self, state: np.ndarray, t: float, rng: np.random.Generator) -> np.ndarray: ...


class ObservationNoise(Protocol):
    def apply(self, obs: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...


FORCE_REGISTRY: dict[str, type[ForceDisturbance]] = {}
OBSERVATION_REGISTRY: dict[str, type[ObservationNoise]] = {}


def register_force(name: str):
    def decorator(cls):
        FORCE_REGISTRY[name] = cls
        return cls

    return decorator


def register_observation(name: str):
    def decorator(cls):
        OBSERVATION_REGISTRY[name] = cls
        return cls

    return decorator


@register_force("none")
class NoWind:
    def force(self, state: np.ndarray, t: float, rng: np.random.Generator) -> np.ndarray:
        return np.zeros(3)


@register_force("constant")
class ConstantWind:
    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=float)

    def force(self, state: np.ndarray, t: float, rng: np.random.Generator) -> np.ndarray:
        return self.vector.copy()


@register_force("ou_wind")
class OrnsteinUhlenbeckWind:
    """Euler-Maruyama Ornstein-Uhlenbeck process, sampled once per env
    step. Temporally correlated (unlike white noise, which a policy would
    trivially average out and learn nothing from). `t` (not an internally
    tracked step count) is used to derive dt between calls, so this
    doesn't need to know the env's dt directly.

    First call after construction initializes deterministically at `mu`
    (no rng draw). Combined with quad_hover_env.py rebuilding a fresh
    instance every reset(), each episode's wind restarts cleanly at its
    mean rather than carrying over state from the previous episode.
    """

    def __init__(self, theta: float, sigma: float, mu: float | list[float] = 0.0):
        self.theta = theta
        self.sigma = sigma
        self.mu = np.broadcast_to(np.asarray(mu, dtype=float), (3,)).copy()
        self._value = self.mu.copy()
        self._last_t: float | None = None

    def force(self, state: np.ndarray, t: float, rng: np.random.Generator) -> np.ndarray:
        if self._last_t is None:
            self._last_t = t
            return self._value.copy()

        dt = t - self._last_t
        self._last_t = t
        self._value = (
            self._value
            + self.theta * (self.mu - self._value) * dt
            + self.sigma * np.sqrt(dt) * rng.normal(size=3)
        )
        return self._value.copy()


@register_observation("none")
class NoNoise:
    def apply(self, obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return obs


@register_observation("gaussian")
class GaussianObsNoise:
    """std is specified per observation block since pos_error/vel/rot6d/
    omega/prev_action live on different post-scaling ranges. Blocks not
    present in `std` get zero noise."""

    def __init__(self, std: dict):
        unknown = set(std) - set(_OBS_BLOCKS)
        if unknown:
            raise ValueError(f"GaussianObsNoise: unknown block(s) in std: {sorted(unknown)}")
        self.std = {name: float(value) for name, value in std.items()}

    def apply(self, obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        noisy = obs.copy()
        for name, block in _OBS_BLOCKS.items():
            std = self.std.get(name, 0.0)
            if std:
                noisy[block] = noisy[block] + rng.normal(0.0, std, size=block.stop - block.start)
        return noisy


def build_force_disturbance(force_cfg: dict) -> ForceDisturbance:
    cfg = dict(force_cfg)
    cls = FORCE_REGISTRY[cfg.pop("type")]
    return cls(**cfg)


def build_observation_noise(observation_cfg: dict) -> ObservationNoise:
    cfg = dict(observation_cfg)
    cls = OBSERVATION_REGISTRY[cfg.pop("type")]
    return cls(**cfg)
