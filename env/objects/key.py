"""
key.py

Defines the Key class: paired with a specific Door (see door.py) via a
shared door_id, a Key unlocks that door when used.

Coordinate space: see agent.py's module docstring.
"""

from typing import Optional, Tuple

from env.core.maze import Maze
from env.objects.door import Door


class Key:
    """
    A single key placed on a logical cell, linked to exactly one Door
    by door_id.
    """

    def __init__(self, door_id: str) -> None:
        """
        Args:
            door_id: Identifier of the Door this key unlocks. Must
                match that Door's own door_id exactly -- use() raises
                on a mismatched Door rather than silently doing
                nothing, since a silent no-op here would look like a
                working key that mysteriously doesn't open its door.
        """
        self.door_id: str = door_id
        self.position: Optional[Tuple[int, int]] = None  # LOGICAL coordinates
        self.is_collected: bool = False

    def place_at(self, maze: Maze, logical_row: int, logical_col: int) -> None:
        """
        Raises:
            IndexError: If the logical position is out of bounds.
            ValueError: If this key has already been collected.
        """
        if self.is_collected:
            raise ValueError(
                f"Key for door {self.door_id!r} has already been collected and cannot "
                "be placed on the grid again"
            )
        if not (
            0 <= logical_row < maze.grid.logical_rows
            and 0 <= logical_col < maze.grid.logical_cols
        ):
            raise IndexError(
                f"Key position (logical {logical_row}, {logical_col}) is out of bounds "
                f"for a {maze.grid.logical_rows}x{maze.grid.logical_cols} logical maze"
            )

        if self.position is not None:
            self._clear_cell(maze, self.position)

        maze.grid.get_logical_cell(logical_row, logical_col).contains_key = True
        self.position = (logical_row, logical_col)

    def collect(self, maze: Maze) -> None:
        """
        Pick up this key: clears it from the grid, marks it collected.
        Typically called by whatever detects an agent stepping onto
        this key's cell (the not-yet-built PettingZoo wrapper) -- Key
        has no way to detect that itself, since it doesn't know where
        any Agent is.

        Raises:
            ValueError: If not currently placed, or already collected.
        """
        if self.is_collected:
            raise ValueError(f"Key for door {self.door_id!r} has already been collected")
        if self.position is None:
            raise ValueError(f"Key for door {self.door_id!r} is not currently placed")

        self._clear_cell(maze, self.position)
        self.position = None
        self.is_collected = True

    def use(self, door: Door) -> None:
        """
        Unlock a Door with this key.

        Raises:
            ValueError: If not yet collected, or door_id doesn't match.
        """
        if not self.is_collected:
            raise ValueError(f"Key for door {self.door_id!r} must be collected before use")
        if door.door_id != self.door_id:
            raise ValueError(f"Key for door {self.door_id!r} does not match door {door.door_id!r}")
        door.unlock()

    def _clear_cell(self, maze: Maze, logical_position: Tuple[int, int]) -> None:
        maze.grid.get_logical_cell(*logical_position).contains_key = False

    def __repr__(self) -> str:
        state = "collected" if self.is_collected else "on grid"
        return f"Key(door_id={self.door_id!r}, position={self.position}, {state})"


if __name__ == "__main__":
    from env.core.maze import Maze
    from env.generator.generator_factory import create_generator

    maze = Maze(rows=9, cols=9, random_seed=1)
    create_generator("recursive_backtracking", random_seed=1).generate(maze)

    door = Door("door_1")
    door.place_at(maze, 2, 2)
    wrong_door = Door("door_2")
    wrong_door.place_at(maze, 3, 3)

    key = Key("door_1")
    key.place_at(maze, 0, 0)
    print("After placement:", key)

    key.collect(maze)
    print("After collection:", key)
    print("Cell cleared:", maze.grid.get_logical_cell(0, 0).contains_key)

    key2 = Key("door_1")
    key2.place_at(maze, 4, 4)
    key2.collect(maze)
    try:
        key2.use(wrong_door)
        print("FAIL: should have raised")
    except ValueError as e:
        print("OK:", e)

    key.use(door)
    print("After correct use:", door)
