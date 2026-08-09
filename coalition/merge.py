"""
merge.py

Merge confirmation logic: tau_form gating (already enforced by
knowledge_graph/directed_graph.py's active_neighbors() -- not
reimplemented here) plus P=2 consecutive confirmation rounds, and a
Pareto check verifying an existing coalition's performance wouldn't get
WORSE by admitting a candidate. Directly implements Steps 2-4 of the
KG/Coalition Implementation Strategy doc's Coalition Formation section.

Reuses federation.validation.transfer_validation's
extract_shared_state/apply_shared_state/evaluate_model -- specifically
their swap-evaluate-revert pattern -- rather than writing a second
"temporarily apply a candidate model and measure the effect"
implementation. The try/finally revert-on-crash discipline from
compute_transfer_benefit() is followed here for the same reason it
exists there: a Pareto check must never leave the coalition's model
corrupted, even if evaluation raises partway through.
"""

from typing import Dict, List

from env.wrappers.pettingzoo_env import CoFedMazeParallelEnv
from federation.aggregation.fedavg import SharedState
from federation.validation.transfer_validation import (
    apply_shared_state,
    evaluate_model,
    extract_shared_state,
)
from marl.models.vdn import VDNModel

DEFAULT_PATIENCE = 2  # P, per the KG/Coalition Implementation Strategy doc's hyperparameter table


class MergeConfirmationTracker:
    """
    Per-candidate consecutive-confirmation counter. One instance per
    node, tracking every current merge candidate (a node's
    active_neighbors() not already in its coalition) at once.
    """

    def __init__(self, patience: int = DEFAULT_PATIENCE) -> None:
        """
        Raises:
            ValueError: If patience is not a positive integer.
        """
        if patience <= 0:
            raise ValueError(f"patience must be positive, got {patience}")
        self.patience = patience
        self._consecutive_rounds: Dict[str, int] = {}

    def record_round(self, candidate_id: str, above_tau_form: bool) -> int:
        """
        Record whether `candidate_id`'s KS-bar was above tau_form this
        round. A round where it is NOT above tau_form resets the streak
        to 0 -- confirmation requires P CONSECUTIVE rounds ("stay above
        tau_form for P consecutive rounds," not "P cumulative rounds
        above, with gaps allowed"), so a single dip breaks the streak
        rather than merely pausing it.

        Returns:
            The candidate's current consecutive-round streak count.
        """
        if above_tau_form:
            self._consecutive_rounds[candidate_id] = self._consecutive_rounds.get(candidate_id, 0) + 1
        else:
            self._consecutive_rounds[candidate_id] = 0
        return self._consecutive_rounds[candidate_id]

    def is_patience_satisfied(self, candidate_id: str) -> bool:
        """Whether `candidate_id` has reached P consecutive above-tau_form rounds."""
        return self._consecutive_rounds.get(candidate_id, 0) >= self.patience

    def reset(self, candidate_id: str) -> None:
        """
        Clear a candidate's streak -- call once it's actually merged
        (patience tracking no longer applies to a member, only to
        candidates), or if it should no longer be tracked for any other
        reason (e.g. its physical edge itself got pruned).
        """
        self._consecutive_rounds.pop(candidate_id, None)


def pareto_check(
    coalition_model: VDNModel,
    candidate_shared_state: SharedState,
    env: CoFedMazeParallelEnv,
    validation_seeds: List[int],
    tolerance: float = 0.0,
) -> bool:
    """
    Verify admitting `candidate_shared_state` into the coalition would
    not make the coalition's OWN current performance worse. "Cheap" per
    the doc: reuses the small validation subset, a single before/after
    comparison -- not a full retraining or multi-metric analysis.

    ALWAYS reverts coalition_model back to its pre-check state before
    returning, whether the check passes, fails, or raises -- a caller
    must be able to trust that calling this function never has a
    lasting side effect on coalition_model.

    Args:
        coalition_model: The coalition's CURRENT aggregated model, used
            (and always restored) in place.
        candidate_shared_state: What the coalition's shared state would
            become if the candidate is admitted -- typically the output
            of federation.aggregation.weighted.weighted_average() run
            WITH the candidate included, computed by the caller
            (coalition_manager.py) before calling this function.
        validation_seeds: The shared small validation subset (same
            seeds used elsewhere for transfer-benefit/leave-one-out
            testing, per the doc's "reuse the same validation subset"
            principle).
        tolerance: How much performance drop is still acceptable.
            PLACEHOLDER, not specified in any project doc -- defaults
            to 0.0 (candidate must not make coalition performance any
            worse at all, no slack for evaluation noise). Retune once
            real validation-reward variance across repeated evaluations
            of the SAME model is measured.

    Returns:
        True if the candidate passes the Pareto check (safe to admit).
    """
    before = evaluate_model(env, coalition_model, validation_seeds)

    original_state = extract_shared_state(coalition_model)
    try:
        apply_shared_state(coalition_model, candidate_shared_state)
        after = evaluate_model(env, coalition_model, validation_seeds)
    finally:
        apply_shared_state(coalition_model, original_state)

    return after >= (before - tolerance)


def should_merge(
    tracker: MergeConfirmationTracker,
    candidate_id: str,
    above_tau_form: bool,
    coalition_model: VDNModel,
    candidate_shared_state: SharedState,
    env: CoFedMazeParallelEnv,
    validation_seeds: List[int],
) -> bool:
    """
    One call per round, per candidate: records this round's
    above/below-tau_form result, and only if patience is now satisfied,
    runs the (expensive, real-episode) Pareto check. The Pareto check
    deliberately does NOT run every round -- only once the cheap
    patience gate has already been cleared, since there is no point
    paying evaluation cost for a candidate that hasn't even held a
    strong edge for P rounds yet.

    Returns:
        True if `candidate_id` should be merged into the coalition this round.
    """
    tracker.record_round(candidate_id, above_tau_form)
    if not tracker.is_patience_satisfied(candidate_id):
        return False
    return pareto_check(coalition_model, candidate_shared_state, env, validation_seeds)


if __name__ == "__main__":
    from env.core.actions import NUM_ACTIONS
    from env.core.observations import NUM_CHANNELS

    # --- MergeConfirmationTracker: consecutive-streak semantics ---
    tracker = MergeConfirmationTracker(patience=2)
    assert tracker.record_round("N2", True) == 1
    assert not tracker.is_patience_satisfied("N2")
    assert tracker.record_round("N2", True) == 2
    assert tracker.is_patience_satisfied("N2")
    print("Patience satisfied after 2 consecutive above-tau_form rounds: OK")

    # A dip resets the streak, not just pauses it
    tracker2 = MergeConfirmationTracker(patience=2)
    tracker2.record_round("N5", True)
    tracker2.record_round("N5", False)  # dip -- should reset to 0
    count = tracker2.record_round("N5", True)
    print("After True, False, True: streak =", count, "(should be 1, not 2 -- dip resets)")
    assert count == 1
    assert not tracker2.is_patience_satisfied("N5")

    tracker2.reset("N5")
    assert not tracker2.is_patience_satisfied("N5")
    print("reset() clears streak: OK")

    # --- pareto_check: always reverts, even for a trivial self-swap ---
    env = CoFedMazeParallelEnv(rows=9, cols=9, algorithm="recursive_backtracking", window_size=5, max_episode_steps=30)
    model = VDNModel(in_channels=NUM_CHANNELS, window_size=5, num_actions=NUM_ACTIONS, num_agents=2)
    original_snapshot = extract_shared_state(model)

    # Swapping in the coalition's OWN current state should trivially
    # pass (before == after exactly), and must leave the model
    # byte-identical afterward.
    passed = pareto_check(model, original_snapshot, env, validation_seeds=[1, 2])
    print("Self-swap Pareto check passes (before == after):", passed)
    assert passed

    reverted = all(
        __import__("torch").equal(original_snapshot["encoder"][k], model.encoder.state_dict()[k])
        for k in original_snapshot["encoder"]
    )
    print("Model reverted exactly after pareto_check:", reverted)
    assert reverted

    print("OK")
