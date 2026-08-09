"""
physical_graph.py

Loads and exposes configs/topology.yaml as the fixed physical adjacency
of N1-N5. This is the gate that must run before any Knowledge Score
computation happens (edge-efficiency strategy doc, Principle 2): a node
never even evaluates a neighbour it cannot physically reach.

Current topology: RING (see configs/topology.yaml), per the workplan
Section 2.3 -- every node has exactly two direct physical neighbours.
Backed by networkx.Graph specifically because a ring makes multi-hop
routing a real, common case (most node pairs are NOT directly linked,
unlike a fully-connected topology) -- reusing networkx's shortest-path
implementation here avoids writing a second graph-search algorithm
alongside env/validators/bfs_validator.py's.
"""

from pathlib import Path
from typing import List, Set, Tuple, Union

import networkx as nx
import yaml

PathLike = Union[str, Path]


class PhysicalGraph:
    """
    Undirected graph of which nodes can physically communicate.
    """

    def __init__(
        self, nodes: List[str], links: List[Tuple[str, str]], topology_type: str = "unspecified"
    ) -> None:
        """
        Args:
            nodes: All node ids (e.g. ["N1", ..., "N5"]).
            links: Undirected physical links as (node_a, node_b) pairs.
            topology_type: Free-form label from config (e.g. "ring") --
                stored for logging/debugging only. The actual topology
                shape is fully determined by `links`, not this label.

        Raises:
            ValueError: If nodes is empty, or a link references a node
                not in `nodes`.
        """
        if not nodes:
            raise ValueError("nodes must not be empty")

        self.topology_type = topology_type
        self._graph = nx.Graph()
        self._graph.add_nodes_from(nodes)
        for a, b in links:
            if a not in nodes or b not in nodes:
                raise ValueError(f"Link ({a}, {b}) references a node not in {nodes}")
            self._graph.add_edge(a, b)

    @classmethod
    def from_yaml(cls, path: PathLike) -> "PhysicalGraph":
        """
        Raises:
            FileNotFoundError: If the config file doesn't exist.
            ValueError: If 'nodes' or 'links' is missing from the file.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Topology config not found at {path}")
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        if "nodes" not in config or "links" not in config:
            raise ValueError(f"{path} must define both 'nodes' and 'links'")

        nodes = config["nodes"]
        links = [tuple(link) for link in config["links"]]
        topology_type = config.get("topology_type", "unspecified")
        return cls(nodes=nodes, links=links, topology_type=topology_type)

    def is_connected(self, node_a: str, node_b: str) -> bool:
        """Whether node_a and node_b share a DIRECT physical link (one hop)."""
        return self._graph.has_edge(node_a, node_b)

    def neighbors(self, node: str) -> Set[str]:
        """
        Directly physically-reachable neighbours of `node` (one hop
        only). See federation/topology/neighbors.py for a thin
        standalone function wrapping this.

        Raises:
            ValueError: If `node` is not in this graph.
        """
        if node not in self._graph:
            raise ValueError(f"Unknown node {node!r}; known nodes: {self.nodes}")
        return set(self._graph.neighbors(node))

    def shortest_path(self, source: str, target: str) -> List[str]:
        """
        Shortest multi-hop path from source to target over the physical
        graph -- meaningful for a ring, where most node pairs are NOT
        directly linked and require relaying through intermediate nodes.

        Raises:
            ValueError: If source/target is unknown, or no path exists
                (a disconnected graph).
        """
        if source not in self._graph or target not in self._graph:
            raise ValueError(f"Unknown node in ({source}, {target}); known nodes: {self.nodes}")
        try:
            return nx.shortest_path(self._graph, source, target)
        except nx.NetworkXNoPath:
            raise ValueError(f"No path exists between {source} and {target} in this topology")

    @property
    def nodes(self) -> List[str]:
        return sorted(self._graph.nodes)

    def __repr__(self) -> str:
        return (
            f"PhysicalGraph(topology_type={self.topology_type!r}, "
            f"nodes={self.nodes}, edges={self._graph.number_of_edges()})"
        )


if __name__ == "__main__":
    graph = PhysicalGraph(
        nodes=["N1", "N2", "N3", "N4", "N5"],
        links=[("N1", "N2"), ("N2", "N3"), ("N3", "N4"), ("N4", "N5"), ("N5", "N1")],
        topology_type="ring",
    )
    print(graph)
    print("N1 neighbors:", graph.neighbors("N1"))
    print("N1 directly connected to N3?", graph.is_connected("N1", "N3"))
    print("Shortest path N1 -> N3:", graph.shortest_path("N1", "N3"))
    print("Shortest path N1 -> N4:", graph.shortest_path("N1", "N4"))

    # Loading from the real config file
    loaded = PhysicalGraph.from_yaml("configs/topology.yaml")
    print("Loaded from yaml:", loaded)
