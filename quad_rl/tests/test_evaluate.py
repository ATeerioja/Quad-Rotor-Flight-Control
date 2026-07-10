"""Pytest suite for quad_rl.training.evaluate -- the algorithm-agnostic,
multi-episode evaluation harness (Stage 1.7).

Tests:
    1. compute_aggregate_metrics: success rate ("held within
       hover_threshold for the final 2s"), crash rate, mean terminal
       position error, and action smoothness on synthetic rollouts with
       known answers.
    2. _find_run_dir / _load_run_config: correctly locate config.yaml next
       to a "final_model.zip"-style checkpoint and a
       "checkpoints/<prefix>_<n>_steps.zip"-style one, and return None
       (not raise) when it doesn't exist.
    3. _hover_threshold_from_config matches the default hover.yaml value.
    4. The required regression test: evaluating the existing Stage 0
       final_model.zip (runs/my_test_run/, no config.yaml -- falls back to
       --algo ppo) reproduces a plot and reports a success rate/terminal
       position error consistent with the README's ~1cm hover claim.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from quad_rl.config.loader import load_config
from quad_rl.training.evaluate import (
    _find_run_dir,
    _hover_threshold_from_config,
    _load_run_config,
    compute_aggregate_metrics,
    evaluate,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_CONFIG = REPO_ROOT / "quad_rl" / "envs" / "configs" / "default.yaml"
STAGE0_CHECKPOINT = REPO_ROOT / "runs" / "my_test_run" / "final_model.zip"


def _fake_result(positions, targets, dt=0.01, terminated=False, actions=None):
    positions = np.array(positions, dtype=float)
    targets = np.array(targets, dtype=float)
    n = len(positions)
    if actions is None:
        actions = np.zeros((n, 4))
    return {
        "times": np.arange(1, n + 1) * dt,
        "positions": positions,
        "targets": targets,
        "rewards": np.zeros(n),
        "actions": np.array(actions, dtype=float),
        "reward_components": {},
        "terminated": terminated,
        "truncated": not terminated,
        "dt": dt,
    }


def test_success_requires_holding_within_threshold_for_final_two_seconds():
    dt = 0.1
    n = 30  # 3.0s episode
    # Close to target only for the last 1.0s (10 steps) -- not long enough
    # to count as "held for the final 2s".
    positions = [[1.0, 0.0, 0.0]] * 20 + [[0.0, 0.0, 0.0]] * 10
    targets = [[0.0, 0.0, 0.0]] * n
    result = _fake_result(positions, targets, dt=dt)

    metrics = compute_aggregate_metrics([result], hover_threshold=0.1)
    assert metrics["success_rate"] == 0.0


def test_success_when_held_within_threshold_for_final_two_seconds():
    dt = 0.1
    n = 30
    positions = [[1.0, 0.0, 0.0]] * 10 + [[0.0, 0.0, 0.0]] * 20  # last 2.0s close
    targets = [[0.0, 0.0, 0.0]] * n
    result = _fake_result(positions, targets, dt=dt)

    metrics = compute_aggregate_metrics([result], hover_threshold=0.1)
    assert metrics["success_rate"] == 1.0


def test_success_rate_is_none_without_hover_threshold():
    result = _fake_result([[0.0, 0.0, 0.0]] * 5, [[0.0, 0.0, 0.0]] * 5)
    metrics = compute_aggregate_metrics([result], hover_threshold=None)
    assert metrics["success_rate"] is None


def test_crash_rate_and_terminal_position_error():
    crashed = _fake_result([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]] * 2, terminated=True)
    timed_out = _fake_result([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], [[0.0, 0.0, 0.0]] * 2, terminated=False)

    metrics = compute_aggregate_metrics([crashed, timed_out], hover_threshold=None)
    assert metrics["crash_rate"] == 0.5
    assert metrics["mean_terminal_position_error"] == pytest.approx((2.0 + 0.5) / 2)


def test_action_smoothness_is_mean_action_delta_norm():
    actions = [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    result = _fake_result([[0.0, 0.0, 0.0]] * 3, [[0.0, 0.0, 0.0]] * 3, actions=actions)

    metrics = compute_aggregate_metrics([result], hover_threshold=None)
    # |a1-a0| = 1.0, |a2-a1| = 0.0 -> mean 0.5
    assert metrics["action_smoothness"] == pytest.approx(0.5)


def test_find_run_dir_handles_final_model_and_checkpoints_subdir(tmp_path):
    run_dir = tmp_path / "my_run"
    (run_dir / "checkpoints").mkdir(parents=True)

    assert _find_run_dir(run_dir / "final_model.zip") == run_dir
    assert _find_run_dir(run_dir / "checkpoints" / "ppo_quadhover_1000_steps.zip") == run_dir


def test_load_run_config_returns_none_when_missing(tmp_path):
    run_dir = tmp_path / "my_run"
    run_dir.mkdir()
    assert _load_run_config(run_dir / "final_model.zip") is None


def test_load_run_config_reads_written_config(tmp_path):
    run_dir = tmp_path / "my_run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text("algo: sac\nenv: {}\n")

    config = _load_run_config(run_dir / "final_model.zip")
    assert config == {"algo": "sac", "env": {}}


def test_hover_threshold_from_config_matches_default_yaml():
    env_config = load_config(DEFAULT_ENV_CONFIG)
    assert _hover_threshold_from_config(env_config) == pytest.approx(0.1)


@pytest.mark.skipif(not STAGE0_CHECKPOINT.exists(), reason="Stage 0 checkpoint not present in this checkout")
def test_stage0_checkpoint_reproduces_plot_and_matches_readme_accuracy_claim(tmp_path):
    """README claims the Stage 0 PPO baseline 'converges to within ~1cm of
    the target position and holds it'. This checkpoint predates
    config.yaml (Stage 1.6+), so --algo must be given explicitly --
    evaluate.py should fall back to it rather than requiring the file."""
    assert _load_run_config(STAGE0_CHECKPOINT) is None  # confirms the fallback path is actually exercised

    output_path = tmp_path / "stage0_eval.png"
    args = _make_args(
        checkpoint=STAGE0_CHECKPOINT,
        algo="ppo",
        n_episodes=1,
        output=output_path,
        eval_env_config=None,
    )
    metrics = evaluate(args)

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    # "~1cm" with generous slack for a single-episode, non-cherry-picked
    # seed; well within hover_threshold=0.1m (10cm) either way.
    assert metrics["mean_terminal_position_error"] < 0.05
    assert metrics["success_rate"] == 1.0


@pytest.mark.skipif(not STAGE0_CHECKPOINT.exists(), reason="Stage 0 checkpoint not present in this checkout")
def test_stage0_checkpoint_multi_episode_summary(tmp_path):
    output_path = tmp_path / "stage0_summary.png"
    args = _make_args(
        checkpoint=STAGE0_CHECKPOINT,
        algo="ppo",
        n_episodes=5,
        output=output_path,
        eval_env_config=None,
    )
    metrics = evaluate(args)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert metrics["n_episodes"] == 5
    assert metrics["success_rate"] is not None


def _make_args(checkpoint, algo, n_episodes, output, eval_env_config):
    import argparse

    return argparse.Namespace(
        checkpoint=checkpoint,
        vecnormalize=None,
        output=output,
        algo=algo,
        n_episodes=n_episodes,
        eval_env_config=eval_env_config,
        record_video=False,
        seed=0,
        stochastic=False,
    )
