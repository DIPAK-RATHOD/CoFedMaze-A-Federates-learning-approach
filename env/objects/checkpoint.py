"""
checkpoint.py

Defines the Checkpoint class: a passive maze marker used by the
"checkpoints" task variant. Checkpoints never block movement -- they
only track whether an agent has passed through, for task logic (e.g.
"exit unlocks once every checkpoint is reached") that composes on top
rather than being hardcoded here.

Coordinate space: see agent.py's module docstring.
"""

from typing import Optional, Tuple

from env.core.maze import Maze


class Checkpoint:
    """
    A single checkpoint marker placed on a logical cell.

    Reaching a checkpoint is not detected by this class -- that requires
    knowing when an agent's position coincides with a checkpoint's
    position, which is the (not yet built) PettingZoo wrapper's job each
    step, calling mark_reached() here. Checkpoint only tracks the
    resulting state.
    """

    def __init__(self, checkpoint_id: str) -> None:
        """
        Args:
            checkpoint_id: Caller-assigned identifier. Not validated
                against a fixed set (unlike Agent.agent_id) -- a maze
                may have any number of checkpoints.
        """
        self.checkpoint_id: str = checkpoint_id
        self.position: Optional[Tuple[int, int]] = None  # LOGICAL coordinates
        self.is_reached: bool = False

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
                f"Checkpoint position (logical {logical_row}, {logical_col}) is out of "
                f"bounds for a {maze.grid.logical_rows}x{maze.grid.logical_cols} logical maze"
            )

        if self.position is not None:
            self._clear_cell(maze, self.position)

        maze.grid.get_logical_cell(logical_row, logical_col).contains_checkpoint = True
        self.position = (logical_row, logical_col)

    def remove(self, maze: Maze) -> None:
        """
        Raises:
            ValueError: If this checkpoint is not currently placed.
        """
        if self.position is None:
            raise ValueError(f"Checkpoint {self.checkpoint_id!r} is not currently placed")
        self._clear_cell(maze, self.position)
        self.position = None

    def mark_reached(self) -> None:
        """Mark this checkpoint as reached by an agent."""
        self.is_reached = True

    def reset(self) -> None:
        """
        Reset reached state (not placement) -- for a new episode on the
        same maze layout, mirroring Maze.reset()'s "clear episode state,
        keep structure" contract.
        """
        self.is_reached = False

    def _clear_cell(self, maze: Maze, logical_position: Tuple[int, int]) -> None:
        maze.grid.get_logical_cell(*logical_position).contains_checkpoint = False

    def __repr__(self) -> str:
        state = "reached" if self.is_reached else "unreached"
        return f"Checkpoint(id={self.checkpoint_id!r}, position={self.position}, {state})"


if __name__ == "__main__":
    from env.core.maze import Maze
    from env.generator.generator_factory import create_generator

    maze = Maze(rows=9, cols=9, random_seed=1)
    create_generator("recursive_backtracking", random_seed=1).generate(maze)

    cp = Checkpoint("checkpoint_1")
    cp.place_at(maze, 1, 1)
    print("After placement:", cp)
    print("Cell flag set:", maze.grid.get_logical_cell(1, 1).contains_checkpoint)

    cp.mark_reached()
    print("After reaching:", cp)

    cp.reset()
    print("After reset (position kept, reached cleared):", cp)

    cp.remove(maze)
    print("After removal:", cp)
    print("Cell flag cleared:", maze.grid.get_logical_cell(1, 1).contains_checkpoint)
