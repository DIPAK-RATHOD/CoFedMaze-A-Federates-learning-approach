"""
agent.py

Defines the Agent class: one of the two cooperating agents placed on a
node's maze.

Coordinate space: Agent tracks position in LOGICAL coordinates only,
converting to raw grid coordinates exclusively via Grid's own
get_logical_cell()/get_wall_between() -- never by hand.

Scope boundary: move_to() only validates STRUCTURAL passability
(bounds, logical adjacency, whether the wall-slot between two cells has
been carved). It does NOT check object-level blocking (a locked Door,
a static Obstacle) on the destination cell -- that requires a registry
of which Door/Obstacle instance sits where, which doesn't exist yet (no
task-variant setup or PettingZoo wrapper built). Compose that with this
method later; don't bypass it.
"""

from typing import Optional, Tuple

from env.core.constants import AGENT_A, AGENT_B
from env.core.maze import Maze

VALID_AGENT_IDS = (AGENT_A, AGENT_B)


class Agent:
    """
    A single agent's identity and position within a Maze. Not bound to
    one Maze permanently -- place_at()/remove() take the Maze as a
    parameter, so the same Agent instance can be reset onto the same or
    a different Maze across episodes.
    """

    def __init__(self, agent_id: str) -> None:
        """
        Raises:
            ValueError: If agent_id is not AGENT_A or AGENT_B.
        """
        if agent_id not in VALID_AGENT_IDS:
            raise ValueError(f"agent_id must be one of {VALID_AGENT_IDS}, got {agent_id!r}")
        self.agent_id: str = agent_id
        self.position: Optional[Tuple[int, int]] = None  # LOGICAL coordinates

    def place_at(self, maze: Maze, logical_row: int, logical_col: int) -> None:
        """
        Place this agent at a logical cell. No adjacency requirement --
        for spawn/episode reset. Use move_to() for in-episode movement.

        Raises:
            IndexError: If the logical position is out of bounds.
            ValueError: If the target cell already holds a DIFFERENT
                agent (Cell.contains_agent holds only one id at a time).
        """
        if not (
            0 <= logical_row < maze.grid.logical_rows
            and 0 <= logical_col < maze.grid.logical_cols
        ):
            raise IndexError(
                f"Agent position (logical {logical_row}, {logical_col}) is out of bounds "
                f"for a {maze.grid.logical_rows}x{maze.grid.logical_cols} logical maze"
            )

        target_cell = maze.grid.get_logical_cell(logical_row, logical_col)
        is_exit_cell = (logical_row, logical_col) == maze.exit_position
        if (
            not is_exit_cell
            and target_cell.contains_agent is not None
            and target_cell.contains_agent != self.agent_id
        ):
            raise ValueError(
                f"Cell (logical {logical_row}, {logical_col}) is already occupied by "
                f"agent {target_cell.contains_agent!r}"
            )

        if self.position is not None:
            self._clear_cell(maze, self.position)

        target_cell.contains_agent = self.agent_id
        self.position = (logical_row, logical_col)

    def move_to(self, maze: Maze, logical_row: int, logical_col: int) -> None:
        """
        Move to an orthogonally adjacent logical cell, validating
        structural passability only (see module docstring).

        Raises:
            ValueError: If not yet placed, or if the wall-slot between
                current and target has not been carved.
            IndexError: If the target is out of bounds, or (propagated
                from Grid.get_wall_between()) not exactly one logical
                step from the current position.
        """
        if self.position is None:
            raise ValueError(f"Agent {self.agent_id} has not been placed yet -- call place_at() first")
        if not (
            0 <= logical_row < maze.grid.logical_rows
            and 0 <= logical_col < maze.grid.logical_cols
        ):
            raise IndexError(
                f"Move target (logical {logical_row}, {logical_col}) is out of bounds "
                f"for a {maze.grid.logical_rows}x{maze.grid.logical_cols} logical maze"
            )

        current_cell = maze.grid.get_logical_cell(*self.position)
        target_cell = maze.grid.get_logical_cell(logical_row, logical_col)

        wall = maze.grid.get_wall_between(current_cell, target_cell)
        if wall.is_wall:
            raise ValueError(
                f"Cannot move agent {self.agent_id} from {self.position} to "
                f"({logical_row}, {logical_col}) -- no passage carved between them"
            )

        self.place_at(maze, logical_row, logical_col)

    def remove(self, maze: Maze) -> None:
        """
        Raises:
            ValueError: If this agent is not currently placed.
        """
        if self.position is None:
            raise ValueError(f"Agent {self.agent_id} is not currently placed on any maze")
        self._clear_cell(maze, self.position)
        self.position = None

    def _clear_cell(self, maze: Maze, logical_position: Tuple[int, int]) -> None:
        cell = maze.grid.get_logical_cell(*logical_position)
        if cell.contains_agent == self.agent_id:
            cell.contains_agent = None

    def __repr__(self) -> str:
        return f"Agent(id={self.agent_id!r}, position={self.position})"


if __name__ == "__main__":
    from env.generator.generator_factory import create_generator

    maze = Maze(rows=9, cols=9, random_seed=1)
    create_generator("recursive_backtracking", random_seed=1).generate(maze)

    agent_a = Agent(AGENT_A)
    agent_a.place_at(maze, 0, 0)
    print("After placement:", agent_a)
    print("Cell reflects occupancy:", maze.grid.get_logical_cell(0, 0).contains_agent)

    agent_b = Agent(AGENT_B)
    try:
        agent_b.place_at(maze, 0, 0)
        print("FAIL: should have raised")
    except ValueError as e:
        print("OK:", e)

    origin = maze.grid.get_logical_cell(0, 0)
    neighbor_cell = next(
        n for n in maze.grid.get_logical_neighbors(origin)
        if not maze.grid.get_wall_between(origin, n).is_wall
    )
    neighbor = (neighbor_cell.row // 2, neighbor_cell.col // 2)
    agent_a.move_to(maze, *neighbor)
    print("After move:", agent_a)
    print("Old cell cleared:", maze.grid.get_logical_cell(0, 0).contains_agent)

    agent_a.remove(maze)
    print("After removal:", agent_a)
