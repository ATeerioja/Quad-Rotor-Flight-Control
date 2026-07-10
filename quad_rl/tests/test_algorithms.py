"""Pytest suite for quad_rl.training.algorithms -- the SB3-algorithm
registry (Stage 1.6).

Tests:
    1. ALGO_REGISTRY contains ppo, sac, td3.
    2. ppo's default_hyperparams (loaded from configs/algo/ppo.yaml)
       exactly match the values previously hardcoded in train_ppo.py, so
       the refactor doesn't silently change training behavior.
    3. build_kwargs: identity for ppo, n_steps -> train_freq/gradient_steps
       translation for the off-policy algorithms.
    4. supports_vecnormalize_reward is True only for ppo.
"""

from quad_rl.training.algorithms import ALGO_REGISTRY


def test_registry_has_ppo_sac_td3():
    assert {"ppo", "sac", "td3"} <= set(ALGO_REGISTRY)


def test_ppo_default_hyperparams_match_legacy_values():
    assert ALGO_REGISTRY["ppo"].default_hyperparams == {
        "learning_rate": 0.0003,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }


def test_ppo_build_kwargs_is_identity():
    hyperparams = {"n_steps": 64, "batch_size": 32}
    assert ALGO_REGISTRY["ppo"].build_kwargs(hyperparams) == hyperparams


def test_off_policy_build_kwargs_translates_n_steps():
    hyperparams = {"n_steps": 64, "batch_size": 32}
    for algo in ("sac", "td3"):
        result = ALGO_REGISTRY[algo].build_kwargs(hyperparams)
        assert result == {"train_freq": 64, "gradient_steps": 64, "batch_size": 32}
        assert "n_steps" not in result


def test_off_policy_build_kwargs_leaves_native_keys_alone():
    hyperparams = {"train_freq": 4, "gradient_steps": 4}
    for algo in ("sac", "td3"):
        assert ALGO_REGISTRY[algo].build_kwargs(hyperparams) == hyperparams


def test_supports_vecnormalize_reward_only_for_ppo():
    assert ALGO_REGISTRY["ppo"].supports_vecnormalize_reward is True
    assert ALGO_REGISTRY["sac"].supports_vecnormalize_reward is False
    assert ALGO_REGISTRY["td3"].supports_vecnormalize_reward is False


def test_default_policy_is_mlp_for_all_registered_algos():
    for spec in ALGO_REGISTRY.values():
        assert spec.default_policy == "MlpPolicy"
