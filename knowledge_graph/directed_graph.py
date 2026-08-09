"""
directed_graph.py

Maintains the directed, weighted Knowledge Graph edge set over
PHYSICALLY-REACHABLE node pairs only (federation/topology/physical_graph.py
gates this before any KS computation happens, per the edge-efficiency
strategy doc's Principle 2 -- a node never even evaluates a neighbor it
cannot physically reach). Applies threshold gating (create/strengthen/
prune) from each edge's KS-bar (knowledge_graph/knowledge_score.py),
per Section 3.7.10's edge creation rule.

Sparse-by-construction bound: for the RING topology this project
actually uses (configs/topology.yaml), each node has exactly 2
physical neighbors, so the maximum possible directed edge count for 5
nodes is 5*2=10 -- NOT the 20 the KG/Coalition Implementation Strategy
doc's "at most 20 directed edges" figure assumes, which was computed
for a FULLY-CONNECTED topology (5*4=20). max_possible_edges() below
computes this from the PhysicalGraph directly rather than hardcoding
either number, so it stays correct if the topology is later changed
(the workplan explicitly plans to compare line/star/random/full-mesh
topologies).
"""

from typing import Dict, List, Set, Tuple

from federation.topology.physical_graph import PhysicalGraph
from knowledge_graph.knowledge_score import KnowledgeScoreTracker


class DirectedKnowledgeGraph:
    """
    One node's view of the directed knowledge graph: a
    KnowledgeScoreTracker per physically-reachable INCOMING edge (for
    node X, one tracker per physical neighbor Y that could send X an
    update), plus threshold-gated membership in the "active" edge set
    X actually trusts.
    """

    def __init__(
        self,
        own_node_id: str,
        physical_graph: PhysicalGraph,
        tau_form: float = 0.50,
        tau_break: float = 0.30,
        alpha: float = 0.30,
    ) -> None:
        """
        Args:
            own_node_id: This node's id.
            physical_graph: Gates which neighbors are ever tracked at
                all -- only own_node_id's PHYSICAL neighbors get a
                KnowledgeScoreTracker.
            tau_form: KS-bar threshold to CREATE/strengthen an edge
                (KG/Coalition Implementation Strategy doc's
                hyperparameter table default: 0.50).
            tau_break: KS-bar threshold to PRUNE an edge (default:
                0.30). tau_break < tau_form gives hysteresis: an edge,
                once formed, tolerates some KS-bar decline before being
                pruned, rather than flickering create/prune on every
                small fluctuation around one threshold.
            alpha: EMA smoothing factor, forwarded to every
                KnowledgeScoreTracker this graph creates.

        Raises:
            ValueError: If own_node_id is not in physical_graph, or
                tau_break >= tau_form (would defeat the hysteresis the
                two-threshold design exists for).
        """
        if own_node_id not in physical_graph.nodes:
            raise ValueError(f"{own_node_id!r} is not a node in the given physical graph")
        if tau_break >= tau_form:
            raise ValueError(
                f"tau_break ({tau_break}) must be < tau_form ({tau_form}) for hysteresis to work"
            )

        self.own_node_id = own_node_id
        self.physical_graph = physical_graph
        self.tau_form = tau_form
        self.tau_break = tau_break

        self._trackers: Dict[str, KnowledgeScoreTracker] = {
            neighbor: KnowledgeScoreTracker(alpha=alpha)
            for neighbor in physical_graph.neighbors(own_node_id)
        }
        self._active_edges: Set[str] = set()

    def update_edge(self, source_node_id: str, ks: float) -> float:
        """
        Record one round's raw KS for the edge source_node_id ->
        own_node_id, update its EMA (KS-bar), and apply threshold
        gating: create the edge if KS-bar rises above tau_form, prune
        it if KS-bar falls below tau_break. An edge strictly between
        the two thresholds keeps its current state (hysteresis).

        Returns:
            The updated KS-bar.

        Raises:
            ValueError: If source_node_id is not a physical neighbor of
                own_node_id (Principle 2: never even evaluate who you
                can't physically reach).
        """
        if source_node_id not in self._trackers:
            raise ValueError(
                f"{source_node_id!r} is not a physical neighbor of {self.own_node_id!r} -- "
                f"only physically-reachable nodes can have a tracked edge"
            )

        ks_bar = self._trackers[source_node_id].update(ks)

        if ks_bar > self.tau_form:
            self._active_edges.add(source_node_id)
        elif ks_bar < self.tau_break:
            self._active_edges.discard(source_node_id)

        return ks_bar

    def active_neighbors(self) -> List[str]:
        """Physical neighbors whose edge is currently ACTIVE (trusted)."""
        return sorted(self._active_edges)

    def ks_bar(self, source_node_id: str) -> float:
        """
        Raises:
            ValueError: If source_node_id is not a tracked physical neighbor.
        """
        if source_node_id not in self._trackers:
            raise ValueError(f"{source_node_id!r} is not a tracked physical neighbor")
        return self._trackers[source_node_id].ks_bar

    def max_possible_edges(self) -> int:
        """
        The sparse-by-construction bound for the CURRENT physical
        topology: sum of every node's physical-neighbor count. See
        module docstring -- computed live, never hardcoded.
        """
        return sum(len(self.physical_graph.neighbors(n)) for n in self.physical_graph.nodes)


if __name__ == "__main__":
    graph = PhysicalGraph.from_yaml("configs/topology.yaml")
    dkg = DirectedKnowledgeGraph(own_node_id="N1", physical_graph=graph, tau_form=0.50, tau_break=0.30)

    print("N1's physical neighbors:", graph.neighbors("N1"))
    print("Max possible edges for this ring topology (should be 10, not 20):", dkg.max_possible_edges())
    assert dkg.max_possible_edges() == 10

    # Wrong-neighbor rejection: N3 is NOT a physical neighbor of N1 in a ring.
    try:
        dkg.update_edge("N3", ks=0.9)
        print("FAIL: should have raised")
    except ValueError as e:
        print("OK, non-physical-neighbor rejected:", e)

    # Threshold gating + hysteresis walkthrough for N2 -> N1.
    dkg.update_edge("N2", ks=0.9)  # high KS -> should cross tau_form and activate
    print("After high KS: active?", "N2" in dkg.active_neighbors(), " ks_bar:", dkg.ks_bar("N2"))
    assert "N2" in dkg.active_neighbors()

    dkg.update_edge("N2", ks=0.4)  # drops, but EMA-smoothed value should still be > tau_break -> hysteresis holds
    print("After moderate drop: still active (hysteresis)?", "N2" in dkg.active_neighbors(), " ks_bar:", dkg.ks_bar("N2"))
    assert "N2" in dkg.active_neighbors()  # hysteresis: still above tau_break

    for _ in range(10):
        dkg.update_edge("N2", ks=0.0)  # sustained low KS -> should eventually prune
    print("After sustained low KS: pruned?", "N2" not in dkg.active_neighbors(), " ks_bar:", dkg.ks_bar("N2"))
    assert "N2" not in dkg.active_neighbors()

    print("OK")
