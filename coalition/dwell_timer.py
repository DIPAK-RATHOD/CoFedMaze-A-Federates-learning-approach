"""
dwell_timer.py

Tracks the minimum dwell time (D = 2-3 episodes, per the KG/Coalition
Implementation Strategy doc's hyperparameter table) after a coalition
forms, during which it cannot be broken. Kept as its own small,
reusable file because dwell-timer logic is shared by two different
callers: merge.py's post-merge lock (start a timer the moment a
coalition confirms) and coalition_manager.py's periodic health-check
gate (skip health checks entirely while dwelling) -- a single
implementation here means both call sites can't drift out of sync with
each other about what "still dwelling" means.

Deliberately trivial and stateless-per-call: a DwellTimer instance
tracks ONE coalition's countdown, in episodes, with no dependency on
wall-clock time or any other coalition's state.
"""


class DwellTimer:
    """
    Counts down from D episodes after start(); is_active() reports
    whether the dwell protection window is still in effect.
    """

    def __init__(self, dwell_episodes: int = 2) -> None:
        """
        Args:
            dwell_episodes: D, the number of episodes a coalition is
                protected from being broken after forming. Per the
                KG/Coalition Implementation Strategy doc's
                hyperparameter table, D is in the 2-3 range; 2 is used
                here as the default (the lower, more conservative end
                of that range -- shorter protection means a genuinely
                bad merge gets caught and undone sooner).

        Raises:
            ValueError: If dwell_episodes is not a positive integer.
        """
        if dwell_episodes <= 0:
            raise ValueError(f"dwell_episodes must be positive, got {dwell_episodes}")
        self.dwell_episodes = dwell_episodes
        self._remaining = 0  # 0 = not dwelling (either never started, or already expired)

    def start(self) -> None:
        """
        (Re)start the countdown at dwell_episodes. Called once, at the
        moment a coalition merge is confirmed (merge.py) -- calling
        this again while already dwelling resets the countdown rather
        than stacking, since there's only ever one dwell window per
        coalition at a time, not one per merge event.
        """
        self._remaining = self.dwell_episodes

    def tick(self) -> None:
        """
        Advance by one episode. Call exactly once per episode/round
        from the caller's own event loop (coalition_manager.py) --
        this class has no notion of episode boundaries itself, it only
        counts calls to tick().
        """
        if self._remaining > 0:
            self._remaining -= 1

    def is_active(self) -> bool:
        """True if the coalition is still within its post-merge protection window."""
        return self._remaining > 0

    @property
    def remaining(self) -> int:
        """Episodes left in the current dwell window (0 if not dwelling)."""
        return self._remaining

    def __repr__(self) -> str:
        return f"DwellTimer(dwell_episodes={self.dwell_episodes}, remaining={self._remaining})"


if __name__ == "__main__":
    timer = DwellTimer(dwell_episodes=3)
    print("Before start:", timer, "is_active:", timer.is_active())
    assert not timer.is_active()

    timer.start()
    print("After start:", timer, "is_active:", timer.is_active())
    assert timer.is_active()
    assert timer.remaining == 3

    timer.tick()
    timer.tick()
    print("After 2 ticks:", timer, "is_active:", timer.is_active())
    assert timer.is_active()
    assert timer.remaining == 1

    timer.tick()
    print("After 3rd tick (dwell fully elapsed):", timer, "is_active:", timer.is_active())
    assert not timer.is_active()
    assert timer.remaining == 0

    # Ticking further while already expired must not go negative
    timer.tick()
    timer.tick()
    print("Further ticks after expiry stay at 0, not negative:", timer.remaining)
    assert timer.remaining == 0

    # Restart resets the countdown
    timer.start()
    assert timer.remaining == 3
    print("Restart resets countdown correctly: OK")

    try:
        DwellTimer(dwell_episodes=0)
        print("FAIL: should have raised")
    except ValueError as e:
        print("OK, non-positive dwell_episodes rejected:", e)

    print("OK")
