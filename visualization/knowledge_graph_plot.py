"""Render one node's directed, KS-weighted knowledge-graph view."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from knowledge_graph.directed_graph import DirectedKnowledgeGraph

PathLike = Union[str, Path]


def plot_knowledge_graph(graph: DirectedKnowledgeGraph, output_path: PathLike) -> Path:
    """Save the physical neighborhood and active inbound KS edges for one node.

    Gray lines show physically reachable peers.  Colored arrows show active
    directed edges from a neighbor into ``graph.own_node_id``; their labels
    are the current EMA-smoothed knowledge score.
    """
    nodes = graph.physical_graph.nodes
    if not nodes:
        raise ValueError("knowledge graph has no physical nodes")
    positions = {
        node: (math.cos(2 * math.pi * index / len(nodes)), math.sin(2 * math.pi * index / len(nodes)))
        for index, node in enumerate(nodes)
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6, 6))
    for node in nodes:
        for neighbor in graph.physical_graph.neighbors(node):
            if node < neighbor:
                x1, y1 = positions[node]
                x2, y2 = positions[neighbor]
                axis.plot((x1, x2), (y1, y2), color="0.8", linewidth=1, zorder=1)

    for source in graph.active_neighbors():
        x1, y1 = positions[source]
        x2, y2 = positions[graph.own_node_id]
        score = graph.ks_bar(source)
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=16,
                                color="tab:blue", linewidth=1.5 + 3 * score,
                                connectionstyle="arc3,rad=0.12", zorder=2)
        axis.add_patch(arrow)
        axis.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.08, f"KS={score:.2f}", color="tab:blue", ha="center", fontsize=9)

    for node, (x, y) in positions.items():
        color = "tab:green" if node == graph.own_node_id else "white"
        axis.scatter(x, y, s=900, color=color, edgecolors="black", zorder=3)
        axis.text(x, y, node, ha="center", va="center", zorder=4, fontweight="bold")
    axis.set(title=f"Knowledge Graph View: {graph.own_node_id}", xlim=(-1.35, 1.35), ylim=(-1.35, 1.35))
    axis.set_aspect("equal")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination
