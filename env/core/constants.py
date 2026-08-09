"""
constants.py

Centralized constants for the maze environment.

This module is the single source of truth for cell kinds, cell traversal
states, placeable object types, and movement directions. Every other
module in `env/core` (and later, generators, validators, renderers, and
the multi-agent RL layer) should import from here rather than redefining
or hardcoding these values.
"""

from typing import List, Tuple

# ---------------------------------------------------------------------------
# Cell Kinds (double-resolution grid structure)
# ---------------------------------------------------------------------------
# Every Cell in a Grid is one of these two structural kinds, determined
# purely by whether its (row, col) grid indices are even or odd:
#   - LOGICAL cells sit at (even, even) indices. They are the actual
#     rooms/positions an agent can occupy and are NEVER walls.
#   - WALL_SLOT cells sit at any index where row is odd, col is odd, or
#     both. An (even, odd) or (odd, even) wall-slot is an *edge*
#     wall-slot between two logical neighbors, which a generator carves
#     (flips to open) to create a passage. An (odd, odd) wall-slot is a
#     *pillar* — a permanent structural corner that is never carved.
# See env/core/grid.py for the full double-resolution design and the
# Grid methods (get_logical_neighbors, get_wall_between, carve_passage)
# that operate on this structure.
LOGICAL: str = "LOGICAL"
WALL_SLOT: str = "WALL_SLOT"

CELL_KINDS: Tuple[str, ...] = (LOGICAL, WALL_SLOT)

# ---------------------------------------------------------------------------
# Cell Traversal States
# ---------------------------------------------------------------------------
# Human-readable labels for a Cell's `is_wall` boolean. Used for debug
# output, ASCII/matplotlib rendering, and logging — NOT as the field
# type itself (`is_wall` stays a plain bool for fast checks). This is
# independent of CELL_KINDS above: a WALL_SLOT cell can be in either
# traversal state (WALL before carving, PATH after), while a LOGICAL
# cell is always PATH.
WALL: str = "WALL"
PATH: str = "PATH"

CELL_TYPES: Tuple[str, ...] = (WALL, PATH)

# ---------------------------------------------------------------------------
# Object Types
# ---------------------------------------------------------------------------
# Entities that can occupy a LOGICAL (non-wall) cell once the maze
# skeleton exists. Kept as independent string constants rather than a
# single Enum class so that new object types (e.g. a second KEY variant,
# a TRAP) can be added later without touching or renumbering existing
# values.
START: str = "START"
EXIT: str = "EXIT"
KEY: str = "KEY"
DOOR: str = "DOOR"
CHECKPOINT: str = "CHECKPOINT"
OBSTACLE: str = "OBSTACLE"
AGENT_A: str = "AGENT_A"
AGENT_B: str = "AGENT_B"

OBJECT_TYPES: Tuple[str, ...] = (
    START,
    EXIT,
    KEY,
    DOOR,
    CHECKPOINT,
    OBSTACLE,
    AGENT_A,
    AGENT_B,
)

# ---------------------------------------------------------------------------
# Directions
# ---------------------------------------------------------------------------
# Each direction is a (delta_row, delta_col) tuple. Applied directly
# (delta = 1 step) to raw grid coordinates for full-resolution traversal
# (Grid.get_neighbors), and applied at 2x scale (delta = 1 logical step
# = 2 raw grid steps) for logical-space generation/adjacency
# (Grid.get_logical_neighbors). This representation is algorithm-agnostic:
# it works the same way for maze generation (Recursive Backtracking /
# Prim / Kruskal), BFS reachability validation, agent movement, and a
# future RL discrete action space (index 0-3 into DIRECTIONS).
UP: Tuple[int, int] = (-1, 0)
DOWN: Tuple[int, int] = (1, 0)
LEFT: Tuple[int, int] = (0, -1)
RIGHT: Tuple[int, int] = (0, 1)

DIRECTIONS: List[Tuple[int, int]] = [UP, DOWN, LEFT, RIGHT]


if __name__ == "__main__":
    # Minimal smoke test / usage example.
    print("Cell kinds   :", CELL_KINDS)
    print("Cell types   :", CELL_TYPES)
    print("Object types :", OBJECT_TYPES)
    print("Directions   :", DIRECTIONS)

    row, col = 5, 5
    dr, dc = UP
    print(f"Moving UP from ({row}, {col}) -> ({row + dr}, {col + dc})")