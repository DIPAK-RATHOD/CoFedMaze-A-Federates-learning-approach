"""
exit.py

Defines the Exit class: behavioral wrapper around the maze's exit
position, adding task-variant hooks (e.g. a locked exit) without
duplicating position tracking.

Deliberately does NOT store its own position. Maze already owns
exit_position/exit_grid_position as the single source of truth -- a
second, independent position on this class would be exactly the kind
of duplicated-source-of-truth bug this codebase has already hit three
times (generator_factory.py's _finalize, and all three validators).
Exit.place() delegates to Maze.set_exit() rather than reimplementing
placement.
"""

from typing import Tuple

from env.core.maze import Maze


class Exit:
    """
    Behavioral wrapper around a Maze's exit: adds an is_locked flag for
    task variants where the exit isn't usable until some condition is
    met (all checkpoints reached, a key collected, etc). Does not
    decide WHAT that condition is -- task-specific behavior composes on
    top rather than being hardcoded here.
    """

    def __init__(self) -> None:
        self.is_locked: bool = False

    def place(self, maze: Maze, logical_row: int, logical_col: int) -> None:
        """
        Set the maze's exit position. Thin delegation to
        Maze.set_exit() -- see module docstring.

        Raises:
            IndexError: If out of bounds (propagated from Maze.set_exit()).
            ValueError: If equal to the current start position
                (propagated from Maze.set_exit()).
        """
        maze.set_exit(logical_row, logical_col)

    def position(self, maze: Maze) -> Tuple[int, int]:
        """Convenience passthrough -- returns maze.exit_position."""
        return maze.exit_position

    def lock(self) -> None:
        """Lock the exit -- an agent cannot use it while locked."""
        self.is_locked = True

    def unlock(self) -> None:
        """Unlock the exit."""
        self.is_locked = False

    def is_usable_by(self, maze: Maze, agent_position: Tuple[int, int]) -> bool:
        """
        Is `agent_position` (LOGICAL coordinates) exactly this maze's
        exit position AND is the exit unlocked?

        Does not check pathfinding reachability -- that's
        env/validators/path_checker.py's job (has_path_to_exit()), a
        structural question independent of any agent's current position
        or this exit's lock state. This answers a narrower, per-step
        question: "is the agent standing on a usable exit right now."
        """
        return (not self.is_locked) and agent_position == maze.exit_position

    def __repr__(self) -> str:
        state = "locked" if self.is_locked else "unlocked"
        return f"Exit({state})"


if __name__ == "__main__":
    from env.generator.generator_factory import create_generator

    maze = Maze(rows=9, cols=9, random_seed=1)
    create_generator("recursive_backtracking", random_seed=1).generate(maze)

    exit_obj = Exit()
    exit_obj.place(maze, 3, 3)
    print("After placement:", exit_obj, "maze.exit_position:", maze.exit_position)

    print("Usable by agent at exit (unlocked)?", exit_obj.is_usable_by(maze, (3, 3)))

    exit_obj.lock()
    print("Usable by agent at exit (locked)?", exit_obj.is_usable_by(maze, (3, 3)))

    exit_obj.unlock()
    print("Usable by agent elsewhere?", exit_obj.is_usable_by(maze, (0, 0)))
