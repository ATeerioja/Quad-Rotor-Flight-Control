"""Pytest suite for quad_rl.config -- the hierarchical, overridable YAML
config loader and its validated schema.

Tests:
    1. Round-trip: from_dict(asdict()) reproduces the original config.
    2. An override changes exactly one leaf and nothing else.
    3. An unknown key raises.
    4. A missing required key raises.
    5. The defaults: chain (default.yaml -> hover.yaml -> base.yaml)
       resolves correctly, and physics/simulation genuinely come from
       base.yaml rather than being duplicated in hover.yaml.
    6. The deep-merge algorithm itself, in isolation, across two levels
       of defaults: chaining.
"""

from pathlib import Path

import pytest
import yaml

from quad_rl.config.loader import _read_yaml_with_defaults, load_config
from quad_rl.config.schema import EnvConfig

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "envs" / "configs"
DEFAULT_YAML = CONFIGS_DIR / "default.yaml"
HOVER_YAML = CONFIGS_DIR / "hover.yaml"
BASE_YAML = CONFIGS_DIR / "base.yaml"


def _leaf_diffs(a: dict, b: dict, prefix=()):
    """{path_tuple: (old, new)} for every leaf where a and b differ."""
    diffs = {}
    for key in a:
        path = prefix + (key,)
        if isinstance(a[key], dict) and isinstance(b[key], dict):
            diffs.update(_leaf_diffs(a[key], b[key], path))
        elif a[key] != b[key]:
            diffs[path] = (a[key], b[key])
    return diffs


def test_round_trip_from_dict_asdict():
    cfg = load_config(DEFAULT_YAML)
    reparsed = EnvConfig.from_dict(cfg.asdict())
    assert reparsed.asdict() == cfg.asdict()


def test_override_changes_exactly_one_leaf():
    base_dict = load_config(DEFAULT_YAML).asdict()
    overridden_dict = load_config(DEFAULT_YAML, overrides=["physics.mass=0.6"]).asdict()

    diffs = _leaf_diffs(base_dict, overridden_dict)
    assert diffs == {("physics", "mass"): (base_dict["physics"]["mass"], 0.6)}


def test_unknown_key_raises():
    bad = load_config(DEFAULT_YAML).asdict()
    bad["physics"]["bogus_field"] = 1.0
    with pytest.raises(ValueError):
        EnvConfig.from_dict(bad)


def test_missing_key_raises():
    bad = load_config(DEFAULT_YAML).asdict()
    del bad["physics"]["mass"]
    with pytest.raises(ValueError):
        EnvConfig.from_dict(bad)


def test_defaults_chain_resolves_default_to_hover_to_base():
    default_cfg = load_config(DEFAULT_YAML)
    hover_cfg = load_config(HOVER_YAML)
    assert default_cfg.asdict() == hover_cfg.asdict()

    hover_raw = yaml.safe_load(HOVER_YAML.read_text())
    base_raw = yaml.safe_load(BASE_YAML.read_text())
    assert "physics" not in hover_raw and "simulation" not in hover_raw
    assert default_cfg.physics.mass == base_raw["physics"]["mass"]
    assert default_cfg.simulation.dt == base_raw["simulation"]["dt"]


def test_deep_merge_chaining_two_levels(tmp_path):
    """Isolated unit test of the merge algorithm itself (grandparent < parent
    < child precedence, keys surviving from every level), independent of the
    shipped YAML files."""
    (tmp_path / "grandparent.yaml").write_text("a: 1\nb: 1\n")
    (tmp_path / "parent.yaml").write_text("defaults: [grandparent.yaml]\nb: 2\nc: 2\n")
    (tmp_path / "child.yaml").write_text("defaults: [parent.yaml]\nc: 3\n")

    merged = _read_yaml_with_defaults(tmp_path / "child.yaml")
    assert merged == {"a": 1, "b": 2, "c": 3}
