"""
actions.py

Defines the discrete action space available to agents: four movement
actions, reusing constants.DIRECTIONS (so this file is never a second,
possibly-drifting definition of "what UP means"), plus an INTERACT
action for the key-collection / checkpoint-marking / exit-use mechanics
env/objects/ already implements.

This closes a gap flagged during the VDN work: marl/models/vdn.py and
marl/agents/ currently take num_actions as a bare constructor
parameter (tests used num_actions=5), with nothing anywhere actually
defining what those 5 actions mean. The (not yet built) PettingZoo
wrapper should import NUM_ACTIONS and direction_for() from here rather
than any code re-deriving or hardcoding the action space independently.
"""

from typing import Dict, Tuple

from env.core.constants import DIRECTIONS, DOWN, LEFT, RIGHT, UP

# Movement actions are indices 0-3 into DIRECTIONS, in the exact order
# DIRECTIONS is already defined in constants.py. Reusing that order
# (rather than redefining direction tuples here) is what keeps this
# file from ever silently disagreeing with constants.py about what
# "action 0" means.
MOVE_UP: int = 0
MOVE_DOWN: int = 1
MOVE_LEFT: int = 2
MOVE_RIGHT: int = 3
INTERACT: int = 4

NUM_ACTIONS: int = 5

_MOVEMENT_ACTIONS = (MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT)

ACTION_NAMES: Dict[int, str] = {
    MOVE_UP: "MOVE_UP",
    MOVE_DOWN: "MOVE_DOWN",
    MOVE_LEFT: "MOVE_LEFT",
    MOVE_RIGHT: "MOVE_RIGHT",
    INTERACT: "INTERACT",
}

# Sanity check, asserted at import time rather than trusted silently:
# DIRECTIONS must stay in the exact [UP, DOWN, LEFT, RIGHT] order the
# MOVE_* indices above assume. A reorder in constants.py would
# otherwise desync action indices from actual movement direction with
# no error anywhere — exactly the kind of silent-drift bug this
# codebase has already hit multiple times (see
# tests/env/test_coordinate_consistency.py for the coordinate-space
# equivalent of this same failure mode).
assert DIRECTIONS == [UP, DOWN, LEFT, RIGHT], (
    "env.core.constants.DIRECTIONS order changed — actions.py's MOVE_* index "
    "assignments assume [UP, DOWN, LEFT, RIGHT] and must be updated to match"
)


def direction_for(action: int) -> Tuple[int, int]:
    """
    Return the (delta_row, delta_col) direction for a movement action.

    Args:
        action: One of MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT.

    Returns:
        The (delta_row, delta_col) tuple from constants.DIRECTIONS.

    Raises:
        ValueError: If `action` is not a movement action (e.g. is
            INTERACT, or out of range).
    """
    if action not in _MOVEMENT_ACTIONS:
        raise ValueError(
            f"action {action} is not a movement action (expected one of "
            f"{list(_MOVEMENT_ACTIONS)}); INTERACT and out-of-range indices "
            "have no direction"
        )
    return DIRECTIONS[action]


def action_name(action: int) -> str:
    """
    Human-readable name for an action index, for logging/debugging.

    Raises:
        ValueError: If `action` is not in [0, NUM_ACTIONS).
    """
    if action not in ACTION_NAMES:
        raise ValueError(f"Unknown action index {action}; valid range is [0, {NUM_ACTIONS})")
    return ACTION_NAMES[action]


if __name__ == "__main__":
    print("NUM_ACTIONS:", NUM_ACTIONS)
    for a in range(NUM_ACTIONS):
        print(a, action_name(a))

    print("Direction for MOVE_UP:", direction_for(MOVE_UP))
    print("Direction for MOVE_RIGHT:", direction_for(MOVE_RIGHT))

    try:
        direction_for(INTERACT)
        print("FAIL: should have raised")
    except ValueError as e:
        print("OK:", e)

    try:
        action_name(99)
        print("FAIL: should have raised")
    except ValueError as e:
        print("OK:", e)
