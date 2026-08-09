"""
path_checker.py

Start-to-exit solvability checks, built on top of BFSValidator.

Where connectivity.py asks a structural question ("is the whole maze
one connected region"), this module asks the question that actually
matters for an RL agent: can you get from start_position to
exit_position at all, and if so, how? A maze can technically be "fully
connected" per connectivity.py while still being useless if start or
exit were never set -- that's a separate failure mode this module
guards against explicitly.
"""

from typing import List, Tuple

from env.core.maze import Maze
from env.validators.bfs_validator import BFSValidator

Coordinate = Tuple[int, int]


def _require_start_and_exit(maze: Maze) -> None:
    """Shared precondition check used by every function in this module."""
    if maze.start_position is None:
        raise ValueError("maze.start_position must be set before checking path solvability")
    if maze.exit_position is None:
        raise ValueError("maze.exit_position must be set before checking path solvability")


def has_path_to_exit(maze: Maze) -> bool:
    """
    Check whether maze.exit_position is reachable from
    maze.start_position.

    Args:
        maze: The Maze to check. Must have both start_position and
            exit_position set.

    Returns:
        True if a path exists from start to exit, else False.

    Raises:
        ValueError: If start_position or exit_position is not set.
    """
    _require_start_and_exit(maze)

    validator = BFSValidator(maze)
    reachable = validator.reachable_cells(maze.start_grid_position)
    return maze.exit_grid_position in reachable


def shortest_path(maze: Maze) -> List[Coordinate]:
    """
    Compute the shortest path (fewest cells) from start to exit, in RAW
    grid coordinates (i.e. including the wall-slot cells the path
    physically passes through between logical cells -- a step count in
    raw space is 2 per logical move; use shortest_path_length() for the
    logical move count).

    BFS guarantees shortest-path-by-cell-count on an unweighted grid
    like this one, so no separate pathfinding algorithm (e.g.
    Dijkstra/A*) is needed here -- reusing BFSValidator is sufficient
    and keeps this module free of a second graph-search implementation.

    Args:
        maze: The Maze to check. Must have both start_position and
            exit_position set.

    Returns:
        A list of RAW (row, col) coordinates from start to exit
        inclusive, in order. Empty list if no path exists.

    Raises:
        ValueError: If start_position or exit_position is not set.
    """
    _require_start_and_exit(maze)

    validator = BFSValidator(maze)
    parents = validator.traverse(maze.start_grid_position)
    return validator.reconstruct_path(parents, maze.exit_grid_position)


def shortest_path_length(maze: Maze) -> int:
    """
    Compute the shortest-path length from start to exit, measured in
    moves (edges) between RAW grid cells, not cells visited.

    Note this counts raw-grid steps, not logical moves: because of the
    double-resolution grid, one logical move (room to adjacent room)
    corresponds to two raw-grid steps (room -> wall-slot -> room), so
    this value is exactly double the number of logical moves an agent
    would actually take.

    Args:
        maze: The Maze to check. Must have both start_position and
            exit_position set.

    Returns:
        The number of raw-grid moves required to reach exit from start
        via the shortest path, or -1 if no path exists.

    Raises:
        ValueError: If start_position or exit_position is not set.
    """
    path = shortest_path(maze)
    if not path:
        return -1
    return len(path) - 1  # N cells visited = N - 1 moves between them.


if __name__ == "__main__":
    from env.generator.generator_factory import create_generator

    maze = Maze(rows=15, cols=21, random_seed=11)
    generator = create_generator("kruskal", random_seed=11)
    generator.generate(maze)

    print("Start (logical):", maze.start_position, "Exit (logical):", maze.exit_position)
    print("Start (raw):", maze.start_grid_position, "Exit (raw):", maze.exit_grid_position)
    print("Has path to exit?", has_path_to_exit(maze))
    print("Shortest path length (raw-grid moves):", shortest_path_length(maze))
    print("Shortest path:", shortest_path(maze))

    # Deliberately sever the only connection to the exit corner to
    # confirm has_path_to_exit correctly reports False rather than
    # trivially passing. Uses exit_grid_position (raw coords) so this
    # actually isolates the real exit cell.
    exit_row, exit_col = maze.exit_grid_position
    maze.grid.get_cell(exit_row - 1, exit_col).is_wall = True
    maze.grid.get_cell(exit_row, exit_col - 1).is_wall = True

    print("\nAfter isolating the exit cell:")
    print("Has path to exit?", has_path_to_exit(maze))
    print("Shortest path length (raw-grid moves):", shortest_path_length(maze))
