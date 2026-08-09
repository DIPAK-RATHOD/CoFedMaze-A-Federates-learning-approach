"""
leave_one_out.py

For a size-3 coalition that fails its health check: test each member's
removal on the shared small validation subset, and expel whichever
member's absence helps the group most. Directly implements Step 6 of
the KG/Coalition Implementation Strategy doc's Coalition Formation
section. Shares the validation subset with merge.py's Pareto check and
federation/validation/transfer_validation.py's transfer-benefit test,
rather than each maintaining its own separate evaluation cost.

Reuses federation.aggregation.weighted.weighted_average() to compute
"what would the coalition's aggregate look like WITHOUT member X" --
the same aggregation primitive used for actually forming the
coalition's shared state in the first place, so "coalition minus one
member" is computed with the identical method as "coalition as it
currently stands," not a separately-written variant.
"""

from typing import Dict, List

from env.wrappers.pettingzoo_env import CoFedMazeParallelEnv
from federation.aggregation.fedavg import SharedState
from federation.aggregation.weighted import weighted_average
from federation.validation.transfer_validation import apply_shared_state, evaluate_model, extract_shared_state
from marl.models.vdn import VDNModel


def find_member_to_expel(
    coalition_model: VDNModel,
    member_shared_states: Dict[str, SharedState],
    member_weights: Dict[str, float],
    env: CoFedMazeParallelEnv,
    validation_seeds: List[int],
) -> str:
    """
    For each member, compute what the coalition's aggregate would be
    WITHOUT that member (weighted_average over the remaining two), and
    evaluate it. Returns the member whose removal yields the HIGHEST
    resulting performance -- i.e. the member currently dragging the
    coalition's aggregate down the most.

    Args:
        coalition_model: The coalition's model, used as scratch space
            for each candidate evaluation and ALWAYS restored to its
            original (all-three-members) state before returning --
            same try/finally discipline as merge.py's pareto_check()
            and transfer_validation.compute_transfer_benefit().
        member_shared_states: {member_id: that member's own shared
            state}, for exactly the 3 current coalition members.
        member_weights: {member_id: that member's current KS-bar (or
            other weighting basis)}, same keys as member_shared_states.
        validation_seeds: The shared small validation subset.

    Returns:
        The member_id to expel.

    Raises:
        ValueError: If member_shared_states has fewer than 3 entries
            (leave-one-out expulsion is only defined for size-3
            coalitions -- a size-2 coalition that fails health goes
            through split.py's dissolve path instead, not this
            function), or member_weights' keys don't match
            member_shared_states'.
    """
    if len(member_shared_states) != 3:
        raise ValueError(
            f"find_member_to_expel requires exactly 3 members, got {len(member_shared_states)} "
            f"-- a size-2 coalition should use split.should_dissolve() instead"
        )
    if set(member_shared_states.keys()) != set(member_weights.keys()):
        raise ValueError(
            f"member_shared_states keys {set(member_shared_states.keys())} != "
            f"member_weights keys {set(member_weights.keys())}"
        )

    original_state = extract_shared_state(coalition_model)
    scores: Dict[str, float] = {}

    try:
        for excluded_member in member_shared_states:
            remaining_ids = [m for m in member_shared_states if m != excluded_member]
            remaining_states = [member_shared_states[m] for m in remaining_ids]
            remaining_weights = [member_weights[m] for m in remaining_ids]

            without_member_state = weighted_average(remaining_states, remaining_weights)
            apply_shared_state(coalition_model, without_member_state)
            scores[excluded_member] = evaluate_model(env, coalition_model, validation_seeds)
    finally:
        apply_shared_state(coalition_model, original_state)

    # The member whose EXCLUSION produced the highest score is the one
    # currently hurting the coalition most -- expel them.
    return max(scores, key=scores.get)


if __name__ == "__main__":
    from env.core.actions import NUM_ACTIONS
    from env.core.observations import NUM_CHANNELS

    env = CoFedMazeParallelEnv(rows=9, cols=9, algorithm="recursive_backtracking", window_size=5, max_episode_steps=30)

    coalition_model = VDNModel(in_channels=NUM_CHANNELS, window_size=5, num_actions=NUM_ACTIONS, num_agents=2)
    original_snapshot = extract_shared_state(coalition_model)

    members = ["N1", "N2", "N5"]
    member_states = {
        m: extract_shared_state(VDNModel(in_channels=NUM_CHANNELS, window_size=5, num_actions=NUM_ACTIONS, num_agents=2))
        for m in members
    }
    member_weights = {"N1": 0.6, "N2": 0.7, "N5": 0.5}

    expelled = find_member_to_expel(coalition_model, member_states, member_weights, env, validation_seeds=[1, 2])
    print("Member selected for expulsion:", expelled)
    assert expelled in members

    reverted = all(
        __import__("torch").equal(original_snapshot["encoder"][k], coalition_model.encoder.state_dict()[k])
        for k in original_snapshot["encoder"]
    )
    print("coalition_model reverted to its original (all-3-members) state after the test:", reverted)
    assert reverted

    try:
        find_member_to_expel(coalition_model, {"N1": member_states["N1"], "N2": member_states["N2"]},
                              {"N1": 0.5, "N2": 0.5}, env, [1])
        print("FAIL: should have raised")
    except ValueError as e:
        print("OK, size-2 input rejected:", e)

    print("OK")
