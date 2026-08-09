"""Standardised evaluation of named CoFedMaze training approaches.

This module deliberately accepts already-built environments and models.  It
does not own training, checkpoint loading, or federated orchestration; those
responsibilities remain in ``marl/``, ``utils/``, and ``scripts/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

from env.wrappers.pettingzoo_env import CoFedMazeParallelEnv
from marl.models.vdn import VDNModel

from evaluation.metrics import MetricsReport, compute_metrics


@dataclass(frozen=True)
class BenchmarkResult:
    """Evaluation result for one approach on a common set of maze seeds."""

    name: str
    metrics: MetricsReport
    convergence_episode: Optional[int] = None


def convergence_episode(
    history: Sequence[Mapping[str, float]],
    reward_threshold: float,
    window: int = 10,
) -> Optional[int]:
    """Return the first episode whose trailing mean reward reaches a threshold.

    ``LocalTrainer.run`` histories are accepted directly.  ``None`` means the
    supplied run never reached the threshold, which is clearer than reporting
    an invented terminal episode as its convergence speed.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    rewards = [float(item["total_reward"]) for item in history]
    if len(rewards) < window:
        return None
    for end in range(window, len(rewards) + 1):
        if sum(rewards[end - window:end]) / window >= reward_threshold:
            return int(history[end - 1]["episode"])
    return None


class BenchmarkRunner:
    """Compare models fairly using equivalent fresh environments and seeds."""

    def __init__(self, env_factory: Callable[[], CoFedMazeParallelEnv], seeds: Sequence[int]) -> None:
        if not seeds:
            raise ValueError("seeds must not be empty")
        self._env_factory = env_factory
        self._seeds = list(seeds)

    def evaluate(
        self,
        models: Mapping[str, VDNModel],
        histories: Optional[Mapping[str, Sequence[Mapping[str, float]]]] = None,
        reward_threshold: Optional[float] = None,
        convergence_window: int = 10,
    ) -> list[BenchmarkResult]:
        """Evaluate each named model on identical seeds.

        A separate environment is constructed for every model so mutable
        environment state cannot leak from one baseline into the next.
        """
        if not models:
            raise ValueError("models must not be empty")
        results = []
        for name, model in models.items():
            episode = None
            if histories is not None and reward_threshold is not None and name in histories:
                episode = convergence_episode(histories[name], reward_threshold, convergence_window)
            results.append(BenchmarkResult(name, compute_metrics(self._env_factory(), model, self._seeds), episode))
        return results
