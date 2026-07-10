"""Pluggable reward terms for QuadHoverEnv.

A RewardTerm is any object with a `name` and a `__call__(ctx) -> float`.
Terms are registered under a config `type` string via @register_reward and
composed by RewardFunction, which sums their per-step values and returns
both the total and the per-term breakdown (see RewardFunction below).
"""

from __future__ import annotations

import dataclasses
from typing import Protocol

import numpy as np

from quad_rl.envs import dynamics


@dataclasses.dataclass(frozen=True)
class StepContext:
    """Everything a reward term might need for one step. Not every field is
    used by every term (e.g. prev_state, dt) -- they're carried here so
    future Stage 2 terms (rate-of-change, time-weighted, ...) don't require
    changing this shape."""

    state: np.ndarray
    prev_state: np.ndarray
    action: np.ndarray
    prev_action: np.ndarray
    target: np.ndarray
    dt: float
    crashed: bool


class RewardTerm(Protocol):
    name: str

    def __call__(self, ctx: StepContext) -> float: ...


REWARD_REGISTRY: dict[str, type[RewardTerm]] = {}


def register_reward(name: str):
    def decorator(cls):
        REWARD_REGISTRY[name] = cls
        return cls

    return decorator


@register_reward("position")
class PositionError:
    name = "position"

    def __init__(self, weight: float):
        self.weight = weight

    def __call__(self, ctx: StepContext) -> float:
        pos_error_norm = np.linalg.norm(ctx.state[dynamics.POS] - ctx.target)
        return -self.weight * pos_error_norm


@register_reward("angular_velocity")
class AngularVelocityPenalty:
    name = "angular_velocity"

    def __init__(self, weight: float):
        self.weight = weight

    def __call__(self, ctx: StepContext) -> float:
        return -self.weight * np.linalg.norm(ctx.state[dynamics.OMEGA])


@register_reward("action_magnitude")
class ActionMagnitude:
    name = "action_magnitude"

    def __init__(self, weight: float):
        self.weight = weight

    def __call__(self, ctx: StepContext) -> float:
        return -self.weight * np.linalg.norm(ctx.action)


@register_reward("action_rate")
class ActionRate:
    name = "action_rate"

    def __init__(self, weight: float):
        self.weight = weight

    def __call__(self, ctx: StepContext) -> float:
        return -self.weight * np.linalg.norm(ctx.action - ctx.prev_action)


@register_reward("hover_bonus")
class HoverBonus:
    name = "hover_bonus"

    def __init__(self, weight: float, threshold: float):
        self.weight = weight
        self.threshold = threshold

    def __call__(self, ctx: StepContext) -> float:
        pos_error_norm = np.linalg.norm(ctx.state[dynamics.POS] - ctx.target)
        return self.weight if pos_error_norm < self.threshold else 0.0


@register_reward("crash_penalty")
class CrashPenalty:
    name = "crash_penalty"

    def __init__(self, weight: float):
        self.weight = weight

    def __call__(self, ctx: StepContext) -> float:
        return -self.weight if ctx.crashed else 0.0


class RewardFunction:
    """Sums a list of RewardTerms. When crash_overrides is True and
    ctx.crashed, every term except CrashPenalty is zeroed for that step --
    CrashPenalty's own value already reflects the crash, so this reproduces
    the "crash zeroes all other terms" semantics without special-casing
    crash handling outside the terms themselves. Every configured term's
    name is always present in the returned components dict, crash or not,
    so callers can rely on a fixed set of keys across an episode."""

    def __init__(self, terms: list[RewardTerm], crash_overrides: bool = True):
        self.terms = terms
        self.crash_overrides = crash_overrides

    def __call__(self, ctx: StepContext) -> tuple[float, dict[str, float]]:
        if self.crash_overrides and ctx.crashed:
            components = {
                term.name: (term(ctx) if isinstance(term, CrashPenalty) else 0.0)
                for term in self.terms
            }
        else:
            components = {term.name: term(ctx) for term in self.terms}
        return sum(components.values()), components

    @classmethod
    def from_config(cls, reward_config, crash_overrides: bool = True) -> "RewardFunction":
        terms = []
        for term_cfg in reward_config.terms:
            term_cfg = dict(term_cfg)
            term_cls = REWARD_REGISTRY[term_cfg.pop("type")]
            terms.append(term_cls(**term_cfg))
        return cls(terms, crash_overrides=crash_overrides)
