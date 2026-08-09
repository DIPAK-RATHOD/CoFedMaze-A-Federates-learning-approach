"""Render the fixed physical communication topology from ``PhysicalGraph``."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from federation.topology.physical_graph import PhysicalGraph

PathLike = Union[str, Path]


def plot_topology(graph: PhysicalGraph, output_path: PathLike) -> Path:
    """Save a deterministic diagram of physical links for reports and debugging."""
    nodes = graph.nodes
    if not nodes:
        raise ValueError("physical graph has no nodes")
    positions = {
        node: (math.cos(2 * math.pi * index / len(nodes)), math.sin(2 * math.pi * index / len(nodes)))
        for index, node in enumerate(nodes)
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6, 6))
    for node in nodes:
        for neighbor in graph.neighbors(node):
            if node < neighbor:
                x1, y1 = positions[node]
                x2, y2 = positions[neighbor]
                axis.plot((x1, x2), (y1, y2), color="tab:gray", linewidth=2, zorder=1)
    for node, (x, y) in positions.items():
        axis.scatter(x, y, s=900, color="white", edgecolors="black", zorder=2)
        axis.text(x, y, node, ha="center", va="center", zorder=3, fontweight="bold")
    axis.set(title=f"Physical Topology ({graph.topology_type})", xlim=(-1.35, 1.35), ylim=(-1.35, 1.35))
    axis.set_aspect("equal")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination
