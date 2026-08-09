"""Plot VDN training loss from ``LocalTrainer.run`` history."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PathLike = Union[str, Path]


def plot_loss_curve(
    history: Sequence[Mapping[str, float]],
    output_path: PathLike,
    title: str = "CoFedMaze VDN Training Loss",
    moving_average_window: int = 10,
) -> Path:
    """Save per-episode loss and an optional trailing mean to ``output_path``."""
    if not history:
        raise ValueError("history must not be empty")
    if moving_average_window < 1:
        raise ValueError("moving_average_window must be at least 1")

    episodes = [int(item["episode"]) for item in history]
    losses = [float(item["loss"]) for item in history]
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(episodes, losses, alpha=0.35, color="tab:orange", label="Episode loss")
    if len(losses) >= moving_average_window:
        means = [
            sum(losses[index - moving_average_window + 1:index + 1]) / moving_average_window
            for index in range(moving_average_window - 1, len(losses))
        ]
        axis.plot(episodes[moving_average_window - 1:], means, color="tab:orange", linewidth=2, label=f"{moving_average_window}-episode mean")
    axis.set(title=title, xlabel="Episode", ylabel="VDN TD loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination
