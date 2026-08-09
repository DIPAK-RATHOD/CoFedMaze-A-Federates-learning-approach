"""
knowledge_score.py

Combines the five normalized criteria (TB, TS, MS, L, E) into a single
Knowledge Score via the weighted formula from Section 3.7.7/3.7.9, then
applies EMA smoothing to produce KS-bar -- the only value
directed_graph.py (not yet built) actually reads for threshold gating.

DEFAULT_WEIGHTS and DEFAULT_ALPHA match Section 3.7.9's worked example
and the KG/Coalition Implementation Strategy doc's hyperparameter table
EXACTLY -- these are the one set of numbers in the whole KS pipeline
that ARE specified in project docs, not placeholders (unlike
transfer_benefit.py's sigmoid steepness, or the downstream
tau_form/tau_break thresholds, which are).
"""

from typing import Dict, Optional

DEFAULT_WEIGHTS: Dict[str, float] = {"tb": 0.40, "ts": 0.25, "ms": 0.20, "l": 0.10, "e": 0.05}
DEFAULT_ALPHA = 0.30  # EMA smoothing factor, KG/Coalition Implementation Strategy doc's hyperparameter table

_REQUIRED_KEYS = frozenset({"tb", "ts", "ms", "l", "e"})


def compute_knowledge_score(
    tb: float, ts: float, ms: float, l: float, e: float,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    KS = w1*TB + w2*TS + w3*MS - w4*L - w5*E (Section 3.7.9's formula
    exactly). L and E are SUBTRACTED, not added: high latency/energy
    cost make a neighbor LESS desirable, unlike TB/TS/MS where higher
    is better.

    Args:
        tb, ts, ms, l, e: All expected already normalized to [0,1] --
            this function does not normalize anything itself; that is
            each criterion's own file's job (transfer_benefit.py for
            TB, etc.).
        weights: Overrides DEFAULT_WEIGHTS. Must have exactly the keys
            {"tb","ts","ms","l","e"}.

    Raises:
        ValueError: If any of tb/ts/ms/l/e is outside [0,1], or
            weights doesn't have exactly the required keys.
    """
    w = weights if weights is not None else DEFAULT_WEIGHTS
    if set(w.keys()) != _REQUIRED_KEYS:
        raise ValueError(f"weights must have exactly keys {_REQUIRED_KEYS}, got {set(w.keys())}")
    for name, value in (("tb", tb), ("ts", ts), ("ms", ms), ("l", l), ("e", e)):
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {value}")

    return w["tb"] * tb + w["ts"] * ts + w["ms"] * ms - w["l"] * l - w["e"] * e


class KnowledgeScoreTracker:
    """
    Maintains the EMA-smoothed KS-bar for ONE directed edge (e.g.
    N2 -> N1) across successive rounds. One instance per directed edge
    -- directed_graph.py (not yet built) will own a
    KnowledgeScoreTracker per physically-reachable neighbor pair.

    O(1) memory by design: only the current KS-bar float is retained,
    matching the edge-efficiency strategy doc's explicit "no
    per-episode history buffer" principle.
    """

    def __init__(self, alpha: float = DEFAULT_ALPHA, initial_ks_bar: float = 0.0) -> None:
        """
        Args:
            alpha: EMA smoothing factor in (0, 1]. Higher = more weight
                on the newest KS, less smoothing.
            initial_ks_bar: Only used as a display/query value before
                the first update() call -- discarded (not blended) on
                that first call, so an arbitrary default never dilutes
                the first real measurement. Defaults to 0.0 (neutral):
                an edge with no evidence yet shouldn't start as if
                already trusted or already distrusted.

        Raises:
            ValueError: If alpha is not in (0, 1].
        """
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self.ks_bar: float = initial_ks_bar
        self._has_update = False

    def update(self, ks: float) -> float:
        """
        Apply one round's raw KS to the EMA:
        KS-bar_new = alpha*KS + (1-alpha)*KS-bar_old.

        The VERY FIRST call sets KS-bar directly to `ks` rather than
        blending with initial_ks_bar -- otherwise the first real
        measurement would be diluted by a value that was never actually
        observed.

        Returns:
            The new KS-bar (also stored in self.ks_bar).
        """
        if not self._has_update:
            self.ks_bar = ks
            self._has_update = True
        else:
            self.ks_bar = self.alpha * ks + (1 - self.alpha) * self.ks_bar
        return self.ks_bar


if __name__ == "__main__":
    # Section 3.7.9's worked example inputs, with default weights.
    ks = compute_knowledge_score(tb=0.18, ts=0.90, ms=0.82, l=0.20, e=0.10)
    print("KS from Section 3.7.9's worked example inputs:", ks)
    expected = 0.40 * 0.18 + 0.25 * 0.90 + 0.20 * 0.82 - 0.10 * 0.20 - 0.05 * 0.10
    assert abs(ks - expected) < 1e-9

    tracker = KnowledgeScoreTracker(alpha=0.30)
    print("Before any update, ks_bar:", tracker.ks_bar)
    first = tracker.update(0.8)
    print("After first update (should equal 0.8 exactly, no blending):", first)
    assert first == 0.8
    second = tracker.update(0.2)
    expected_second = 0.30 * 0.2 + 0.70 * 0.8
    print("After second update:", second, " expected:", expected_second)
    assert abs(second - expected_second) < 1e-9
    print("OK")
