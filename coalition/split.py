"""
split.py

Coalition break/dissolve logic, applied once a coalition's dwell timer
(dwell_timer.py) has expired AND a periodic health check fails. Kept
separate from merge.py since merge and split have different triggering
conditions (patience+Pareto vs. dwell-expired+health-failure) and
different failure modes worth testing independently.

Health check itself is just "is every current member's edge KS-bar
still above tau_break" -- reusing knowledge_graph/directed_graph.py's
own threshold, not a new one. A size-3 coalition failing its health
check goes through leave_one_out.py's expulsion logic instead of a
full dissolve (see coalition_manager.py); a size-2 coalition failing
its health check has no meaningful "expel one member" case (that would
just leave a singleton, which is what dissolve already produces), so it
dissolves directly.
"""

from typing import Dict, Set

from knowledge_graph.directed_graph import DirectedKnowledgeGraph


def health_check(graph: DirectedKnowledgeGraph, coalition_members: Set[str], own_node_id: str) -> bool:
    """
    A coalition is healthy if every OTHER member's edge into this node
    still has KS-bar above tau_break (i.e. none have been pruned from
    the underlying knowledge graph since the coalition formed).

    Args:
        graph: This node's DirectedKnowledgeGraph.
        coalition_members: Every node id in the coalition, INCLUDING
            own_node_id.
        own_node_id: This node's own id -- excluded from the check
            (a node is never its own "edge").

    Returns:
        True if every other member's edge is still above tau_break.
    """
    for member_id in coalition_members:
        if member_id == own_node_id:
            continue
        if graph.ks_bar(member_id) < graph.tau_break:
            return False
    return True


def should_dissolve(dwell_active: bool, coalition_size: int, is_healthy: bool) -> bool:
    """
    A coalition should fully dissolve (back to every member being its
    own singleton) if: the dwell timer has expired (dwell_active is
    False -- a dwelling coalition can never be broken, regardless of
    health), the health check failed, AND the coalition has only 2
    members (a 3-member unhealthy coalition goes through leave_one_out
    expulsion instead -- see coalition_manager.py -- rather than a full
    dissolve, since expelling one member is a lighter-weight fix that
    might restore health without discarding a still-good pairing).

    Raises:
        ValueError: If coalition_size is not a positive integer, or
            exceeds 3 (the project's max_coalition_size).
    """
    if coalition_size <= 0:
        raise ValueError(f"coalition_size must be positive, got {coalition_size}")
    if coalition_size > 3:
        raise ValueError(f"coalition_size must be <= 3 (max_coalition_size), got {coalition_size}")

    if dwell_active:
        return False
    if is_healthy:
        return False
    return coalition_size <= 2


if __name__ == "__main__":
    from federation.topology.physical_graph import PhysicalGraph

    graph = PhysicalGraph.from_yaml("configs/topology.yaml")
    dkg = DirectedKnowledgeGraph(own_node_id="N1", physical_graph=graph, tau_form=0.50, tau_break=0.30)
    dkg.update_edge("N2", ks=0.9)  # healthy
    dkg.update_edge("N5", ks=0.9)  # healthy

    healthy = health_check(dkg, coalition_members={"N1", "N2", "N5"}, own_node_id="N1")
    print("Health check with two strong edges:", healthy)
    assert healthy

    # Degrade N2's edge below tau_break
    for _ in range(10):
        dkg.update_edge("N2", ks=0.0)
    unhealthy = health_check(dkg, coalition_members={"N1", "N2", "N5"}, own_node_id="N1")
    print("Health check after N2's edge degrades:", unhealthy)
    assert not unhealthy

    # should_dissolve semantics
    assert should_dissolve(dwell_active=True, coalition_size=2, is_healthy=False) is False
    print("Dwelling coalition never dissolves, even if unhealthy: OK")

    assert should_dissolve(dwell_active=False, coalition_size=2, is_healthy=False) is True
    print("Size-2, dwell expired, unhealthy -> dissolves: OK")

    assert should_dissolve(dwell_active=False, coalition_size=3, is_healthy=False) is False
    print("Size-3, dwell expired, unhealthy -> does NOT dissolve (goes to leave-one-out instead): OK")

    assert should_dissolve(dwell_active=False, coalition_size=2, is_healthy=True) is False
    print("Healthy coalition never dissolves: OK")

    try:
        should_dissolve(dwell_active=False, coalition_size=4, is_healthy=False)
        print("FAIL: should have raised")
    except ValueError as e:
        print("OK, coalition_size > 3 rejected:", e)

    print("OK")
