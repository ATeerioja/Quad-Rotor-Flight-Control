"""Hierarchical, overridable YAML config loading.

load_config(path, overrides) is the entry point most callers need;
_deep_merge and _read_yaml_with_defaults are exposed for direct testing of
the merge algorithm in isolation from schema validation.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from quad_rl.config.schema import EnvConfig


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` on top of `base`. Non-dict leaves in
    override replace the corresponding base value outright. Returns a new
    dict; does not mutate either input."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_yaml_with_defaults(path: Path, _seen: frozenset[Path] = frozenset()) -> dict:
    """Load `path`, resolve its `defaults:` chain (recursively, so
    default.yaml -> hover.yaml -> base.yaml works), and return one fully
    merged nested dict with the `defaults` key stripped."""
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"Circular `defaults:` chain detected at {path}")
    _seen = _seen | {path}

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    defaults = raw.pop("defaults", None)
    if defaults is None:
        parents = []
    elif isinstance(defaults, str):
        parents = [defaults]
    else:
        parents = list(defaults)

    merged: dict = {}
    for parent_name in parents:
        parent_path = path.parent / parent_name
        merged = _deep_merge(merged, _read_yaml_with_defaults(parent_path, _seen))

    return _deep_merge(merged, raw)


def _parse_override(item: str) -> tuple[list[str], object]:
    key, sep, value_str = item.partition("=")
    if sep != "=":
        raise ValueError(f"Invalid override (expected key=value): {item!r}")
    # Reuse yaml.safe_load for type coercion: "0.6" -> 0.6 (float),
    # "true" -> True (bool), "ou_wind" -> "ou_wind" (str), etc.
    value = yaml.safe_load(value_str.strip())
    return key.strip().split("."), value


def _apply_override(config: dict, key_path: list[str], value: object) -> None:
    node = config
    for part in key_path[:-1]:
        node = node.setdefault(part, {})
    node[key_path[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> EnvConfig:
    merged = _read_yaml_with_defaults(Path(path))
    for item in overrides or []:
        key_path, value = _parse_override(item)
        _apply_override(merged, key_path, value)
    return EnvConfig.from_dict(merged)
