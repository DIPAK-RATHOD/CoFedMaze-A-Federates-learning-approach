"""
reputation.py

Slower-moving node-level trust score, built from a fixed-size rolling
log of (node_id, round, action, resulting_ks_bar) entries -- distinct
from knowledge_graph/'s per-EDGE KS-bar (which tracks "how useful has
THIS specific neighbor been to me"), reputation.py tracks "how has this
node behaved ACROSS all the edges/coalitions it's been part of, over a
longer window." A node could have one currently-strong edge (high
KS-bar with one neighbor right now) while having a poor longer-run
reputation (a history of edges that got pruned, or coalitions it was
expelled from) -- these are genuinely different signals, which is why
this is a separate file rather than just reading KS-bar directly.

Fixed-size log (not unbounded): matches the same "O(1)-ish memory, no
growing history" discipline knowledge_graph/knowledge_score.py's
KnowledgeScoreTracker already follows for KS-bar itself.
"""

from collections import deque
from typing import Deque, List, NamedTuple

DEFAULT_LOG_CAPACITY = 200  # PLACEHOLDER, not specified in project docs -- chosen as a
# round number large enough to smooth over many rounds of noise without
# growing unbounded; retune once real multi-node run lengths are known.

# PLACEHOLDER action-to-score mapping, not specified in project docs.
# Reasoning: a merge/confirm is a positive signal (this node was worth
# forming/keeping a coalition with); an expel/dissolve is negative
# (this node's coalition membership didn't work out); a routine
# health-check pass is mildly positive (steady, unremarkable good
# behavior); a prune (edge-level, not coalition-level) is mildly
# negative. These are small relative to merge/expel so a single prune
# doesn't dominate a node's reputation the way a full coalition
# expulsion should.
ACTION_SCORES = {
    "merge": 1.0,
    "confirm": 0.5,
    "health_check_pass": 0.2,
    "prune": -0.3,
    "expel": -1.0,
    "dissolve": -0.5,
}


class ReputationEntry(NamedTuple):
    node_id: str
    round: int
    action: str
    resulting_ks_bar: float


class ReputationTracker:
    """
    Maintains ONE fixed-size rolling log and derives a single node-level
    reputation score from it: the mean of ACTION_SCORES over whatever
    entries are currently in the log (older entries are silently
    dropped once the log is full -- the log IS the memory window, there
    is no separate decay/smoothing parameter).
    """

    def __init__(self, capacity: int = DEFAULT_LOG_CAPACITY) -> None:
        """
        Raises:
            ValueError: If capacity is not a positive integer.
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self._log: Deque[ReputationEntry] = deque(maxlen=capacity)

    def record(self, node_id: str, round_number: int, action: str, resulting_ks_bar: float) -> None:
        """
        Append one entry. If the log is already at capacity, the
        oldest entry is silently evicted (deque's own behavior) --
        matching the same bounded rolling-log discipline
        marl/replay/replay_buffer.py already uses for trajectories.

        Raises:
            ValueError: If `action` is not one of ACTION_SCORES' keys
                -- an unrecognized action string is almost certainly a
                typo, not a legitimate new action type nobody defined a
                score for yet.
        """
        if action not in ACTION_SCORES:
            raise ValueError(
                f"Unknown action {action!r}; expected one of {sorted(ACTION_SCORES.keys())}"
            )
        self._log.append(ReputationEntry(node_id, round_number, action, resulting_ks_bar))

    def reputation_for(self, node_id: str) -> float:
        """
        Mean ACTION_SCORE across every logged entry for `node_id`
        currently in the rolling log.

        Returns:
            0.0 (neutral) if `node_id` has no entries in the log yet --
            an unknown node isn't assumed good OR bad, matching
            knowledge_graph/knowledge_score.py's KnowledgeScoreTracker
            defaulting a never-updated edge to 0.0 for the same reason.
        """
        entries = [e for e in self._log if e.node_id == node_id]
        if not entries:
            return 0.0
        return sum(ACTION_SCORES[e.action] for e in entries) / len(entries)

    def recent_entries_for(self, node_id: str) -> List[ReputationEntry]:
        """All currently-logged entries for `node_id`, in chronological order."""
        return [e for e in self._log if e.node_id == node_id]

    def __len__(self) -> int:
        return len(self._log)


if __name__ == "__main__":
    tracker = ReputationTracker(capacity=5)

    print("Reputation for a never-seen node (should be neutral 0.0):", tracker.reputation_for("N2"))
    assert tracker.reputation_for("N2") == 0.0

    tracker.record("N2", round_number=1, action="confirm", resulting_ks_bar=0.6)
    tracker.record("N2", round_number=2, action="merge", resulting_ks_bar=0.7)
    rep = tracker.reputation_for("N2")
    print("Reputation for N2 after confirm+merge:", rep)
    assert abs(rep - (0.5 + 1.0) / 2) < 1e-9

    tracker.record("N2", round_number=3, action="expel", resulting_ks_bar=0.1)
    rep_after_expel = tracker.reputation_for("N2")
    print("Reputation for N2 drops after expel:", rep_after_expel)
    assert rep_after_expel < rep

    # Fixed-size rolling: fill past capacity, confirm oldest entries evicted
    for i in range(4, 10):
        tracker.record("N2", round_number=i, action="health_check_pass", resulting_ks_bar=0.5)
    print("Log size stays capped at capacity:", len(tracker))
    assert len(tracker) == 5

    try:
        tracker.record("N2", round_number=99, action="not_a_real_action", resulting_ks_bar=0.5)
        print("FAIL: should have raised")
    except ValueError as e:
        print("OK, unknown action rejected:", e)

    print("OK")
