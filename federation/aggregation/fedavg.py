"""
fedavg.py

Standard unweighted federated averaging: a simple mean across all
members' SHARED state (encoder + memory only -- never the private
head, matching the Shared vs. Private Model Components glossary term
and the same pattern federation/validation/transfer_validation.py
already uses). Kept as a baseline to compare KS-weighted aggregation
(weighted.py) against, to demonstrate empirically that weighting by
Knowledge Score actually helps -- the project's core novelty claim.

Operates on plain shared-state dicts, not VDNModel instances directly
-- callers extract/apply state themselves via
federation.validation.transfer_validation.extract_shared_state /
apply_shared_state. This keeps this module's only job "given N shared
states, compute one aggregate," with zero VDNModel-specific coupling,
so it stays trivially testable with plain tensors.
"""

from typing import Dict, List

import torch

SharedState = Dict[str, Dict[str, torch.Tensor]]

_REQUIRED_COMPONENTS = ("encoder", "memory")


def federated_average(shared_states: List[SharedState]) -> SharedState:
    """
    Simple (unweighted) mean of `shared_states` across every parameter
    tensor in both the "encoder" and "memory" sub-dicts.

    Raises:
        ValueError: If shared_states is empty, or the member states
            don't all have identical parameter keys/shapes (e.g. from
            architecturally mismatched models -- averaging those would
            silently produce nonsense rather than a meaningful model).
    """
    if not shared_states:
        raise ValueError("shared_states must not be empty")

    validate_matching_keys_and_shapes(shared_states)

    aggregated: SharedState = {"encoder": {}, "memory": {}}
    for component in _REQUIRED_COMPONENTS:
        for key in shared_states[0][component]:
            stacked = torch.stack([s[component][key].float() for s in shared_states], dim=0)
            aggregated[component][key] = stacked.mean(dim=0)
    return aggregated


def validate_matching_keys_and_shapes(shared_states: List[SharedState]) -> None:
    """
    Shared validation used by both fedavg.py and weighted.py: every
    member's shared state must have the same components, same
    parameter keys within each component, and identical tensor shapes
    -- otherwise "averaging" them is meaningless (or would crash on the
    stack() call with a much less informative error).

    Raises:
        ValueError: On any mismatch, naming exactly which member and
            which key/component disagreed.
    """
    reference = shared_states[0]
    for component in _REQUIRED_COMPONENTS:
        if component not in reference:
            raise ValueError(f"shared_states[0] is missing required component {component!r}")
        ref_keys = set(reference[component].keys())
        for i, state in enumerate(shared_states[1:], start=1):
            if component not in state:
                raise ValueError(f"shared_states[{i}] is missing required component {component!r}")
            if set(state[component].keys()) != ref_keys:
                raise ValueError(
                    f"shared_states[{i}]['{component}'] has different parameter keys than "
                    f"shared_states[0] -- cannot aggregate architecturally mismatched models"
                )
            for key in ref_keys:
                if state[component][key].shape != reference[component][key].shape:
                    raise ValueError(
                        f"shared_states[{i}]['{component}']['{key}'] shape "
                        f"{tuple(state[component][key].shape)} != shared_states[0]'s "
                        f"{tuple(reference[component][key].shape)}"
                    )


if __name__ == "__main__":
    from env.core.actions import NUM_ACTIONS
    from env.core.observations import NUM_CHANNELS
    from federation.validation.transfer_validation import extract_shared_state
    from marl.models.vdn import VDNModel

    models = [
        VDNModel(in_channels=NUM_CHANNELS, window_size=5, num_actions=NUM_ACTIONS, num_agents=2)
        for _ in range(3)
    ]
    shared_states = [extract_shared_state(m) for m in models]

    aggregated = federated_average(shared_states)
    print("Aggregated encoder keys:", list(aggregated["encoder"].keys())[:3], "...")

    # Sanity: the mean of 3 models' first conv weight must lie strictly
    # between the min and max of the 3 individual values, elementwise.
    key = list(aggregated["encoder"].keys())[0]
    stacked = torch.stack([s["encoder"][key] for s in shared_states])
    within_range = (
        (aggregated["encoder"][key] >= stacked.min(dim=0).values - 1e-6).all()
        and (aggregated["encoder"][key] <= stacked.max(dim=0).values + 1e-6).all()
    )
    print("Aggregated values lie within [min, max] of member values:", within_range.item())
    assert within_range
    print("OK")
