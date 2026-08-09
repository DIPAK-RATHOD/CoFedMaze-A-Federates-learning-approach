"""
cell.py

Defines the Cell class: the atomic unit of the maze grid.

A Cell is a pure data holder. It knows nothing about maze generation
algorithms, rendering, or game rules — it simply describes the state of
one (row, col) position. Higher-level modules (Grid, Maze, generators,
validators, renderers) read and mutate Cell instances but a Cell never
reaches back into any of them. This keeps the dependency direction
one-way: constants -> cell -> grid -> maze.
"""

from dataclasses import dataclass, field

from env.core.constants import LOGICAL


@dataclass
class Cell:
    """
    Represents a single position in the double-resolution maze grid.

    Attributes:
        row: Zero-indexed row coordinate within the raw grid.
        col: Zero-indexed column coordinate within the raw grid.
        kind: Structural role of this cell — LOGICAL (an actual
            room/position, always at even row/col indices) or WALL_SLOT
            (an edge wall between two logical neighbors, or a permanent
            pillar at an odd/odd index). See constants.py and grid.py
            for the full double-resolution design.
        visited: Whether this cell has been visited. Primarily used by
            maze-generation algorithms (Recursive Backtracking, Prim's,
            Kruskal's) during carving, and by traversal/validation logic
            (e.g. BFS reachability checks) afterwards. `Grid.reset_visits()`
            clears this flag independently of wall/object state so the same
            grid can be re-validated multiple times.
        is_wall: Whether this cell currently blocks movement. LOGICAL
            cells are always False (a room is never a wall). WALL_SLOT
            cells start True and a generator carves a passage by
            flipping an *edge* wall-slot to False via
            `Grid.carve_passage()`; pillar wall-slots are never carved
            and stay True permanently.

        contains_agent: Placeholder for a future AGENT_A / AGENT_B
            occupant. Left as Optional[str] rather than a bool so it can
            store *which* agent occupies the cell without a second field.
        contains_start: Whether this is the maze's start cell. Mirrors
            Maze.start_position — Maze is the single source of truth and
            always writes this flag through when start_position changes,
            so code holding only a Cell reference (e.g. a renderer) can
            still tell if it's looking at the start.
        contains_exit: Same as contains_start, but for Maze.exit_position.
        contains_key: Placeholder for a KEY object.
        contains_door: Placeholder for a DOOR object.
        contains_checkpoint: Placeholder for a CHECKPOINT object.
        contains_obstacle: Placeholder for an OBSTACLE object.

    Design note:
        These "contains_*" fields are intentionally simple flags rather
        than references to full object instances (e.g. an `Agent` or
        `Key` class). Phase 1 only needs to reserve the shape of the
        data model — the `env/objects` package (agent.py, key.py,
        door.py, etc.) will define the actual object classes later and
        can either populate these fields with richer objects or replace
        them, without requiring a change to Grid or Maze in the meantime.

        `kind` defaults to LOGICAL and `is_wall` defaults to True only as
        conservative fallbacks for a Cell built outside of Grid (e.g. in
        a unit test). In normal use, Grid.__init__ is the only place
        that constructs Cells, and it always sets `kind` and `is_wall`
        explicitly and consistently.
    """

    row: int
    col: int
    kind: str = LOGICAL
    visited: bool = False
    is_wall: bool = True

    # --- Reserved for future object placement (env/objects package) ---
    contains_agent: str | None = field(default=None)
    contains_start: bool = field(default=False)
    contains_exit: bool = field(default=False)
    contains_key: bool = field(default=False)
    contains_door: bool = field(default=False)
    contains_checkpoint: bool = field(default=False)
    contains_obstacle: bool = field(default=False)

    def __repr__(self) -> str:
        """
        Compact, debug-friendly representation.

        Shows kind, traversal state, and visited flag — the fields that
        matter for quickly scanning a grid dump in a console or log.
        Object-placement flags are omitted here since in Phase 1 they're
        usually at their defaults; once objects are placed, str(cell)
        output can be extended without changing this class's public
        interface.
        """
        state = "WALL" if self.is_wall else "PATH"
        visited_flag = "V" if self.visited else " "
        return f"Cell({self.row},{self.col})[{self.kind}][{state}][{visited_flag}]"


if __name__ == "__main__":
    # Minimal smoke test / usage example.
    c = Cell(row=2, col=2, kind=LOGICAL, is_wall=False)
    print("New logical cell:", c)

    c.visited = True
    print("After visiting:", c)

    c.contains_key = True
    print("After placing a key:", c)
    print("Fields still accessible:", c.row, c.col, c.kind, c.contains_key)