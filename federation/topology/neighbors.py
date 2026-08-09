"""
neighbors.py

Small utility: given a PhysicalGraph and a node id, return that node's
physically-reachable neighbours. Kept separate from physical_graph.py
so federation/ and knowledge_graph/ modules that only need "who is X's
neighbour" can import this thin function without pulling in
PhysicalGraph's yaml-loading machinery, per the Directory Structure
Reference's stated rationale for splitting these two files.
"""

from typing import Set

from federation.topology.physical_graph import PhysicalGraph


def get_neighbors(graph: PhysicalGraph, node: str) -> Set[str]:
    """Thin pass-through to PhysicalGraph.neighbors() -- see that method's docstring."""
    return graph.neighbors(node)


if __name__ == "__main__":
    graph = PhysicalGraph(
        nodes=["N1", "N2", "N3", "N4", "N5"],
        links=[("N1", "N2"), ("N2", "N3"), ("N3", "N4"), ("N4", "N5"), ("N5", "N1")],
        topology_type="ring",
    )
    print("N2 neighbors:", get_neighbors(graph, "N2"))
    assert get_neighbors(graph, "N2") == {"N1", "N3"}
    print("OK")
