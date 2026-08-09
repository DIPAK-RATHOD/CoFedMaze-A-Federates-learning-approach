"""
door.py

Defines the Door class: a locked-by-default object used by the
"key-and-door" task variant. A Door blocks movement while locked and
stops blocking once unlocked by a matching Key (see key.py).

Coordinate space: see agent.py's module docstring.

Known limitation: Cell.contains_door (cell.py) is a plain boolean, not
an identifier -- a maze with more than one Door cannot tell from Cell
flags alone WHICH door occupies a cell, only that some door does.
Disambiguating requires holding the actual Door instance (this class),
which is the real source of truth for door identity; Cell's flag is
only a cheap presence marker (matching contains_start/contains_exit's
role on Maze). Not fixed here to avoid speculative scope growth --
matches cell.py's own note that Phase 1 only reserves the data model's
shape.
"""

from typing import Optional, Tuple

from env.core.maze import Maze


class Door:
    """
    A single door placed on a logical cell. Locked by default -- a
    locked-door variant with a door that starts unlocked wouldn't need
    a door at all.
    """

    def __init__(self, door_id: str) -> None:
        """
        Args:
            door_id: Caller-assigned identifier. A Key must be
                constructed with the matching door_id to unlock this
                door -- see key.py.
        """
        self.door_id: str = door_id
        self.position: Optional[Tuple[int, int]] = None  # LOGICAL coordinates
        self.is_locked: bool = True

    @property
    def blocks_movement(self) -> bool:
        """
        Whether this door currently blocks movement onto its cell.

        Deliberately NOT implemented via Cell.is_wall -- Grid's
        double-resolution invariant guarantees every LOGICAL cell is
        always structurally open (see grid.py); is_wall represents only
        the fixed maze skeleton, not dynamic gameplay state. A locked
        door is a behavioral/object-level block, checked here, meant to
        be composed with structural passability by whatever validates a
        proposed move (Agent.move_to() checks structural passability
        only -- see its module docstring for why object-level blocking
        is intentionally a separate, not-yet-wired-up concern).
        """
        return self.is_locked

    def place_at(self, maze: Maze, logical_row: int, logical_col: int) -> None:
        """
        Raises:
            IndexError: If the logical position is out of bounds.
        """
        if not (
            0 <= logical_row < maze.grid.logical_rows
            and 0 <= logical_col < maze.grid.logical_cols
        ):
            raise IndexError(
                f"Door position (logical {logical_row}, {logical_col}) is out of bounds "
                f"for a {maze.grid.logical_rows}x{maze.grid.logical_cols} logical maze"
            )

        if self.position is not None:
            self._clear_cell(maze, self.position)

        maze.grid.get_logical_cell(logical_row, logical_col).contains_door = True
        self.position = (logical_row, logical_col)

    def remove(self, maze: Maze) -> None:
        """
        Raises:
            ValueError: If this door is not currently placed.
        """
        if self.position is None:
            raise ValueError(f"Door {self.door_id!r} is not currently placed")
        self._clear_cell(maze, self.position)
        self.position = None

    def unlock(self) -> None:
        """Unlock this door (typically called after a matching Key.use())."""
        self.is_locked = False

    def lock(self) -> None:
        """Re-lock this door."""
        self.is_locked = True

    def _clear_cell(self, maze: Maze, logical_position: Tuple[int, int]) -> None:
        maze.grid.get_logical_cell(*logical_position).contains_door = False

    def __repr__(self) -> str:
        state = "locked" if self.is_locked else "unlocked"
        return f"Door(id={self.door_id!r}, position={self.position}, {state})"


if __name__ == "__main__":
    from env.core.maze import Maze
    from env.generator.generator_factory import create_generator

    maze = Maze(rows=9, cols=9, random_seed=1)
    create_generator("recursive_backtracking", random_seed=1).generate(maze)

    door = Door("door_1")
    door.place_at(maze, 2, 2)
    print("After placement:", door, "blocks_movement:", door.blocks_movement)

    door.unlock()
    print("After unlock:", door, "blocks_movement:", door.blocks_movement)

    door.lock()
    print("After re-lock:", door, "blocks_movement:", door.blocks_movement)

    door.remove(maze)
    print("After removal:", door)
