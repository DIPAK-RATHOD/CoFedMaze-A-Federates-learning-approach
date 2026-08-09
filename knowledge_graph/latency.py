"""
latency.py

Computes the normalized latency (L) criterion for a directed edge.

SIMULATION NOTE: no real network exists yet (federation/communication/
transport.py is an in-process simulation -- see that file's own
docstring), so there is no real latency to measure. Rather than invent
an arbitrary number, latency here is estimated from HOP COUNT over the
physical ring topology (federation/topology/physical_graph.py's
shortest_path()) -- more hops genuinely does correlate with more
real-world delay on an actual deployment, so this is a grounded proxy
computed from a real, already-built quantity, not a made-up
placeholder. Swap this for a real round-trip-time measurement once
transport.py has a real-networking implementation.
"""

from federation.topology.physical_graph import PhysicalGraph
from knowledge_graph.normalization import clip_normalize

# PLACEHOLDER, not measured: assumed per-hop latency in milliseconds on
# a real Pi-to-Pi LAN. Retune once actual hardware round-trip times are
# measured.
MS_PER_HOP = 5.0
# PLACEHOLDER normalization range: 0 to a generously large 10-hop
# worst case (this project's ring never exceeds 2 hops at N=5, but the
# range should not silently break if the topology is later widened,
# per the topology-comparison plan in the workplan).
_LATENCY_RANGE_MS = (0.0, 10 * MS_PER_HOP)


def compute_latency(graph: PhysicalGraph, source: str, target: str) -> float:
    """
    Normalized latency estimate for source -> target, based on the
    number of physical hops between them (shortest_path length - 1).

    Raises:
        ValueError: Propagated from PhysicalGraph.shortest_path() if
            source/target is unknown or no path exists.
    """
    path = graph.shortest_path(source, target)
    hop_count = len(path) - 1
    estimated_ms = hop_count * MS_PER_HOP
    return clip_normalize(estimated_ms, *_LATENCY_RANGE_MS)


if __name__ == "__main__":
    graph = PhysicalGraph.from_yaml("configs/topology.yaml")

    l_adjacent = compute_latency(graph, "N1", "N2")  # direct ring neighbor, 1 hop
    l_far = compute_latency(graph, "N1", "N3")  # 2 hops around the ring
    print("Latency N1->N2 (1 hop):", l_adjacent)
    print("Latency N1->N3 (2 hops):", l_far)
    assert l_adjacent < l_far
    print("OK: more hops -> higher normalized latency")
