"""
weighted.py

KS-weighted averaging of accepted updates: the ACTUAL aggregation
method used in coalitions per Chapter 3.7.12 -- members with a higher
smoothed Knowledge Score (KS-bar) contribute more strongly to the
aggregated model. fedavg.py's unweighted mean is the special case
where every weight is equal.
"""

from typing import List

import torch

from federation.aggregation.fedavg import SharedState, validate_matching_keys_and_shapes

_REQUIRED_COMPONENTS = ("encoder", "memory")


def weighted_average(shared_states: List[SharedState], weights: List[float]) -> SharedState:
    """
    Weighted mean of `shared_states`, weighted elementwise by
    `weights` (same order, same length) -- e.g. each member's smoothed
    Knowledge Score KS-bar (knowledge_graph/knowledge_score.py, not
    yet built).

    Weights are NORMALIZED internally (divided by their sum) before
    use, so callers can pass raw, un-normalized KS-bar values directly
    without summing them to 1 themselves first.

    Raises:
        ValueError: If shared_states is empty, len(shared_states) !=
            len(weights), any weight is negative, all weights are zero
            (nothing to weight by), or member states have mismatched
            keys/shapes (same check federated_average uses).
    """
    if not shared_states:
        raise ValueError("shared_states must not be empty")
    if len(shared_states) != len(weights):
        raise ValueError(
            f"shared_states has {len(shared_states)} entries but weights has {len(weights)}"
        )
    if any(w < 0 for w in weights):
        raise ValueError(f"weights must be non-negative, got {weights}")
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("weights sum to zero -- nothing to weight by")

    validate_matching_keys_and_shapes(shared_states)

    normalized = [w / total_weight for w in weights]

    aggregated: SharedState = {"encoder": {}, "memory": {}}
    for component in _REQUIRED_COMPONENTS:
        for key in shared_states[0][component]:
            weighted_sum = sum(
                w * s[component][key].float() for w, s in zip(normalized, shared_states)
            )
            aggregated[component][key] = weighted_sum
    return aggregated


if __name__ == "__main__":
    from env.core.actions import NUM_ACTIONS
    from env.core.observations import NUM_CHANNELS
    from federation.aggregation.fedavg import federated_average
    from federation.validation.transfer_validation import extract_shared_state
    from marl.models.vdn import VDNModel

    models = [
        VDNModel(in_channels=NUM_CHANNELS, window_size=5, num_actions=NUM_ACTIONS, num_agents=2)
        for _ in range(3)
    ]
    shared_states = [extract_shared_state(m) for m in models]

    # Equal weights must match federated_average exactly.
    equal_result = weighted_average(shared_states, weights=[1.0, 1.0, 1.0])
    fedavg_result = federated_average(shared_states)
    key = list(equal_result["encoder"].keys())[0]
    matches_fedavg = torch.allclose(equal_result["encoder"][key], fedavg_result["encoder"][key])
    print("Equal weights matches federated_average:", matches_fedavg)
    assert matches_fedavg

    # All weight on one member must reproduce that member exactly.
    single_result = weighted_average(shared_states, weights=[1.0, 0.0, 0.0])
    matches_member_0 = torch.allclose(single_result["encoder"][key], shared_states[0]["encoder"][key])
    print("All-weight-on-member-0 reproduces member 0 exactly:", matches_member_0)
    assert matches_member_0

    print("OK")
