"""Algorithm registry: decouples train.py from any single SB3 algorithm,
so adding a second algorithm is a config change (--algo sac) rather than a
training-script rewrite. Registered the same way rewards.py/
disturbances.py/randomization.py register their own pluggable pieces.

Each AlgoSpec's default_hyperparams is loaded once, at import time, from
this algorithm's own shipped configs/algo/<name>.yaml -- that file is the
single source of truth for the concrete hyperparameter values (not
duplicated as Python literals here), reusing the same load_raw_config
machinery introduced for env config in Stage 1.1.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable

from stable_baselines3 import PPO, SAC, TD3

from quad_rl.config.loader import load_raw_config

ALGO_CONFIGS_DIR = Path(__file__).parent / "configs" / "algo"


def _identity_kwargs(hyperparams: dict) -> dict:
    return dict(hyperparams)


def _off_policy_kwargs(hyperparams: dict) -> dict:
    """SAC/TD3 have no n_steps concept -- translate it into
    train_freq/gradient_steps if present, so a shared "how much data to
    collect before an update" knob works across on- and off-policy algos
    without scattering `if algo == "sac"` branches through train.py."""
    hyperparams = dict(hyperparams)
    if "n_steps" in hyperparams:
        n_steps = hyperparams.pop("n_steps")
        hyperparams.setdefault("train_freq", n_steps)
        hyperparams.setdefault("gradient_steps", n_steps)
    return hyperparams


@dataclasses.dataclass(frozen=True)
class AlgoSpec:
    cls: type
    default_policy: str
    default_hyperparams: dict
    supports_vecnormalize_reward: bool
    build_kwargs: Callable[[dict], dict] = _identity_kwargs


def _load_default_hyperparams(name: str) -> dict:
    return load_raw_config(ALGO_CONFIGS_DIR / f"{name}.yaml")


ALGO_REGISTRY: dict[str, AlgoSpec] = {
    "ppo": AlgoSpec(
        cls=PPO,
        default_policy="MlpPolicy",
        default_hyperparams=_load_default_hyperparams("ppo"),
        supports_vecnormalize_reward=True,
    ),
    "sac": AlgoSpec(
        cls=SAC,
        default_policy="MlpPolicy",
        default_hyperparams=_load_default_hyperparams("sac"),
        # Off-policy algorithms bootstrap from a replay buffer of raw
        # transitions; normalizing the reward on top would make the
        # buffer's stored rewards a moving target as the running
        # statistics update, unlike PPO's on-policy rollout-then-discard.
        supports_vecnormalize_reward=False,
        build_kwargs=_off_policy_kwargs,
    ),
    "td3": AlgoSpec(
        cls=TD3,
        default_policy="MlpPolicy",
        default_hyperparams=_load_default_hyperparams("td3"),
        supports_vecnormalize_reward=False,
        build_kwargs=_off_policy_kwargs,
    ),
}
