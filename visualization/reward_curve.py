"""Plot local-training episode returns from ``LocalTrainer.run`` history."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence, Union

import matplotlib

# Plots are saved to files by this module; an interactive Tk window is neither
# needed nor reliable on headless Pi/test environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PathLike = Union[str, Path]


def plot_reward_curve(
    history: Sequence[Mapping[str, float]],
    output_path: PathLike,
    title: str = "CoFedMaze Training Reward",
    moving_average_window: int = 10,
) -> Path:
    """Save per-episode return and, when available, its trailing mean.

    ``history`` is the list returned by ``LocalTrainer.run``.  The function
    owns only plotting: callers remain responsible for training and history
    persistence.
    """
    if not history:
        raise ValueError("history must not be empty")
    if moving_average_window < 1:
        raise ValueError("moving_average_window must be at least 1")

    episodes = [int(item["episode"]) for item in history]
    rewards = [float(item["total_reward"]) for item in history]
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(episodes, rewards, alpha=0.35, color="tab:blue", label="Episode return")
    if len(rewards) >= moving_average_window:
        means = [
            sum(rewards[index - moving_average_window + 1:index + 1]) / moving_average_window
            for index in range(moving_average_window - 1, len(rewards))
        ]
        axis.plot(episodes[moving_average_window - 1:], means, color="tab:blue", linewidth=2, label=f"{moving_average_window}-episode mean")
    axis.set(title=title, xlabel="Episode", ylabel="Team return")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination
