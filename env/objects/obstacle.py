"""
obstacle.py

Defines the Obstacle class: a static, always-blocking maze object. Base
class other obstacle behaviors (e.g. a moving obstacle) can extend --
kept as a plain (non-frozen) class specifically so subclassing is
straightforward later.

Coordinate space: see agent.py's module docstring.
"""

from typing import Optional, Tuple

from env.core.maze import Maze


class Obstacle:
    """
    A single static obstacle placed on a logical cell. Always blocks
    movement while placed -- see blocks_movement.
    """

    def __init__(self, obstacle_id: str) -> None:
        """
        Args:
            obstacle_id: Caller-assigned identifier distinguishing this
                obstacle from others in the same maze.
        """
        self.obstacle_id: str = obstacle_id
        self.position: Optional[Tuple[int, int]] = None  # LOGICAL coordinates

    @property
    def blocks_movement(self) -> bool:
        """
        Whether this obstacle currently blocks movement onto its cell.

        Always True while placed for a static Obstacle. A subclass
        modeling a moving/intermittent obstacle (the "moving-obstacle"
        task variant) should override this property rather than the
        shared placement/removal machinery below. See door.py's
        blocks_movement property for why this is not implemented via
        Cell.is_wall.
        """
        return self.position is not None

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
                f"Obstacle position (logical {logical_row}, {logical_col}) is out of "
                f"bounds for a {maze.grid.logical_rows}x{maze.grid.logical_cols} logical maze"
            )

        if self.position is not None:
            self._clear_cell(maze, self.position)

        maze.grid.get_logical_cell(logical_row, logical_col).contains_obstacle = True
        self.position = (logical_row, logical_col)

    def remove(self, maze: Maze) -> None:
        """
        Raises:
            ValueError: If this obstacle is not currently placed.
        """
        if self.position is None:
            raise ValueError(f"Obstacle {self.obstacle_id!r} is not currently placed")
        self._clear_cell(maze, self.position)
        self.position = None

    def _clear_cell(self, maze: Maze, logical_position: Tuple[int, int]) -> None:
        maze.grid.get_logical_cell(*logical_position).contains_obstacle = False

    def __repr__(self) -> str:
        return f"Obstacle(id={self.obstacle_id!r}, position={self.position})"


if __name__ == "__main__":
    from env.generator.generator_factory import create_generator

    maze = Maze(rows=9, cols=9, random_seed=1)
    create_generator("recursive_backtracking", random_seed=1).generate(maze)

    obstacle = Obstacle("obstacle_1")
    print("Before placement, blocks_movement:", obstacle.blocks_movement)

    obstacle.place_at(maze, 2, 2)
    print("After placement:", obstacle, "blocks_movement:", obstacle.blocks_movement)
    print("Cell flag set:", maze.grid.get_logical_cell(2, 2).contains_obstacle)

    obstacle.remove(maze)
    print("After removal:", obstacle, "blocks_movement:", obstacle.blocks_movement)
