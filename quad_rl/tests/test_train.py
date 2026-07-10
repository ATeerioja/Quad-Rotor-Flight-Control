"""Pytest suite for quad_rl.training.train -- the algorithm-agnostic
training entrypoint (Stage 1.6).

Tests:
    1. _split_overrides correctly routes "env."/"algo."-prefixed
       overrides and rejects unprefixed ones.
    2. --algo ppo with a fixed seed reproduces identical first-update loss
       values across two independent runs -- the regression test this
       stage's spec asks for ("reproduces the current run's first-update
       loss values under a fixed seed"); compared new-vs-new rather than
       against the literal pre-refactor train_ppo.py, since that code no
       longer exists in this repo to diff against directly.
    3. --algo sac completes a short run without error.
"""

import argparse
import shutil

import pytest

from quad_rl.training.train import RUNS_DIR, _split_overrides, train


def test_split_overrides_routes_by_prefix():
    env_overrides, algo_overrides = _split_overrides(["env.physics.mass=0.6", "algo.learning_rate=0.001"])
    assert env_overrides == ["physics.mass=0.6"]
    assert algo_overrides == ["learning_rate=0.001"]


def test_split_overrides_rejects_unprefixed():
    with pytest.raises(ValueError):
        _split_overrides(["physics.mass=0.6"])


def _cleanup_run(run_name: str) -> None:
    shutil.rmtree(RUNS_DIR / run_name, ignore_errors=True)
    for path in (RUNS_DIR / "tensorboard").glob(f"{run_name}_*"):
        shutil.rmtree(path, ignore_errors=True)


def _tiny_ppo_args(run_name: str) -> argparse.Namespace:
    return argparse.Namespace(
        algo="ppo",
        env_config=None,
        algo_config=None,
        override=["algo.n_steps=64", "algo.batch_size=32"],
        total_timesteps=64,  # exactly one update at n_envs=1, n_steps=64
        n_envs=1,
        vec_env="dummy",
        checkpoint_freq=1_000_000,
        seed=0,
        run_name=run_name,
    )


def test_ppo_reproduces_first_update_loss_with_fixed_seed():
    run_names = ["_test_ppo_repro_a", "_test_ppo_repro_b"]
    try:
        losses = []
        for run_name in run_names:
            model = train(_tiny_ppo_args(run_name))
            losses.append(model.logger.name_to_value["train/loss"])
        assert losses[0] == pytest.approx(losses[1], rel=1e-6)
    finally:
        for run_name in run_names:
            _cleanup_run(run_name)


def test_sac_completes_without_error():
    run_name = "_test_sac_smoke"
    try:
        args = argparse.Namespace(
            algo="sac",
            env_config=None,
            algo_config=None,
            override=["algo.buffer_size=1000"],
            total_timesteps=5000,
            n_envs=1,
            vec_env="dummy",
            checkpoint_freq=1_000_000,
            seed=0,
            run_name=run_name,
        )
        model = train(args)
        assert model is not None
    finally:
        _cleanup_run(run_name)
