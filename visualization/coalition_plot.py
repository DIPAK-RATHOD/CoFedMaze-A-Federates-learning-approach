"""Visualize each node's coalition size over simulation rounds."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PathLike = Union[str, Path]


def plot_coalition_history(
    history: Sequence[Mapping[str, object]], output_path: PathLike,
    title: str = "CoFedMaze Coalition Membership",
) -> Path:
    """Save coalition-size trajectories from per-round membership snapshots.

    Each history item must have ``round`` and ``coalitions`` keys, where
    ``coalitions`` maps a node id to its current iterable of member ids.  This
    intentionally accepts plain snapshots so the plotting layer does not need
    to depend on the scheduler or coalition manager.
    """
    if not history:
        raise ValueError("history must not be empty")

    try:
        rounds = [int(item["round"]) for item in history]
        nodes = sorted({node for item in history for node in item["coalitions"]})
        sizes = {
            node: [len(item["coalitions"].get(node, ())) for item in history]
            for node in nodes
        }
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("each history item must contain a round and coalition mapping") from error
    if not nodes:
        raise ValueError("coalition history must contain at least one node")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for node in nodes:
        axis.step(rounds, sizes[node], where="post", linewidth=2, label=node)
    axis.set(title=title, xlabel="Round", ylabel="Coalition size", ylim=(0.8, max(max(values) for values in sizes.values()) + 0.2))
    axis.set_yticks(range(1, max(max(values) for values in sizes.values()) + 1))
    axis.grid(alpha=0.25)
    axis.legend(title="Node")
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination
