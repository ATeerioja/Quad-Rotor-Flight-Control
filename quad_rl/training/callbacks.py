"""Custom SB3 TensorBoard callbacks for QuadHover-v0 training, shared
across algorithms (moved out of train_ppo.py in Stage 1.6 so they aren't
tied to any one algorithm's training script).
"""

from __future__ import annotations

from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class HoverFractionCallback(BaseCallback):
    """Logs the mean, over the last 100 completed episodes, of the
    fraction of each episode's steps spent within the hover threshold.

    Reward alone doesn't distinguish "consistently close to target" from
    "oscillating through the target" -- this metric is a more direct
    read on whether the policy is actually learning to hover.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._in_progress: list[list[bool]] = []
        self._completed_fractions: deque[float] = deque(maxlen=100)

    def _on_training_start(self) -> None:
        self._in_progress = [[] for _ in range(self.training_env.num_envs)]

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]
        for i, (info, done) in enumerate(zip(infos, dones)):
            self._in_progress[i].append(info["within_hover_threshold"])
            if done:
                self._completed_fractions.append(float(np.mean(self._in_progress[i])))
                self._in_progress[i] = []

        if self._completed_fractions:
            self.logger.record(
                "custom/fraction_within_hover_threshold",
                float(np.mean(self._completed_fractions)),
            )
        return True


class EpisodeMetricsCallback(BaseCallback):
    """Logs, per completed episode (mean over the last 100): terminal
    position error (QuadHoverEnv's own info["pos_error_norm"] at the final
    step, in meters) and the mean of each reward component over the
    episode. Per-component TensorBoard scalars are what makes Stage 1.2's
    reward shaping debuggable -- without this, only the summed total
    reward is visible, which can't distinguish which term changed.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._component_running_sum: list[dict[str, float]] = []
        self._component_running_count: list[int] = []
        self._component_means: dict[str, deque[float]] = {}
        self._terminal_position_errors: deque[float] = deque(maxlen=100)

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._component_running_sum = [{} for _ in range(n)]
        self._component_running_count = [0 for _ in range(n)]

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for i, (info, done) in enumerate(zip(infos, dones)):
            running = self._component_running_sum[i]
            for key, value in info["reward_components"].items():
                running[key] = running.get(key, 0.0) + value
            self._component_running_count[i] += 1

            if done:
                count = self._component_running_count[i]
                for key, total in running.items():
                    self._component_means.setdefault(key, deque(maxlen=100)).append(total / count)
                self._terminal_position_errors.append(info["pos_error_norm"])
                self._component_running_sum[i] = {}
                self._component_running_count[i] = 0

        for key, values in self._component_means.items():
            if values:
                self.logger.record(f"reward_components/{key}_mean", float(np.mean(values)))
        if self._terminal_position_errors:
            self.logger.record(
                "custom/terminal_position_error_mean", float(np.mean(self._terminal_position_errors))
            )
        return True
