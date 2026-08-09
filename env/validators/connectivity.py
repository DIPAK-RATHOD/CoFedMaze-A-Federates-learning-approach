"""
connectivity.py

Maze-wide structural connectivity checks, built on top of BFSValidator.

Where bfs_validator.py only knows how to traverse from a single point,
this module answers whole-maze questions: is every walkable cell part
of one connected region, are there isolated pockets left behind by a
buggy or manually-edited maze, and so on. It has no opinion about
start/exit specifically — that's path_checker.py's concern.
"""

from typing import Set, Tuple

from env.core.maze import Maze
from env.validators.bfs_validator import BFSValidator

Coordinate = Tuple[int, int]


def _all_path_cells(maze: Maze) -> Set[Coordinate]:
    """Return every non-wall (row, col) coordinate in the maze's grid."""
    cells: Set[Coordinate] = set()
    for row in range(maze.rows):
        for col in range(maze.cols):
            if not maze.grid.get_cell(row, col).is_wall:
                cells.add((row, col))
    return cells


def is_fully_connected(maze: Maze) -> bool:
    """
    Check whether every walkable cell in the maze is reachable from
    every other walkable cell (i.e. the maze forms a single connected
    region, with no isolated pockets).

    Uses maze.start_position as the traversal origin if it's set
    (cheaper — reuses a position the caller already cares about);
    falls back to an arbitrary walkable cell otherwise, since
    connectivity as a structural property doesn't actually depend on
    which cell the search starts from.

    Args:
        maze: The Maze to check.

    Returns:
        True if the maze has zero or one walkable cells, or if all
        walkable cells belong to a single connected region. False if
        there are two or more disconnected pockets of walkable cells.
    """
    path_cells = _all_path_cells(maze)
    if len(path_cells) <= 1:
        return True

    origin = maze.start_position if maze.start_position in path_cells else next(iter(path_cells))

    validator = BFSValidator(maze)
    reachable = validator.reachable_cells(origin)

    return reachable == path_cells


def unreachable_path_cells(maze: Maze) -> Set[Coordinate]:
    """
    Return every walkable cell that is NOT reachable from
    maze.start_position.

    Useful for diagnosing exactly *where* a broken maze's disconnected
    regions are (e.g. after a maze has been manually edited, or if a
    generator bug is suspected), rather than just knowing that a
    problem exists.

    Args:
        maze: The Maze to check. Must have start_position set.

    Returns:
        The set of unreachable walkable cells. Empty if the maze is
        fully connected from its start.

    Raises:
        ValueError: If maze.start_position is not set.
    """
    if maze.start_position is None:
        raise ValueError("maze.start_position must be set to compute unreachable cells")

    path_cells = _all_path_cells(maze)
    validator = BFSValidator(maze)
    reachable = validator.reachable_cells(maze.start_position)

    return path_cells - reachable


def count_connected_components(maze: Maze) -> int:
    """
    Count the number of disjoint connected regions of walkable cells
    in the maze.

    A perfect maze produced by any of the env/generator algorithms
    should always return exactly 1 (or 0 for a maze with no walkable
    cells at all). A value greater than 1 indicates the maze is broken
    — e.g. a manually edited maze that walled off part of itself, or
    (if this ever fires on generator output) a genuine bug in that
    generator's carving logic.

    Args:
        maze: The Maze to check.

    Returns:
        The number of connected components among walkable cells.
    """
    remaining = _all_path_cells(maze)
    validator = BFSValidator(maze)

    components = 0
    while remaining:
        seed = next(iter(remaining))
        reached = validator.reachable_cells(seed)
        remaining -= reached
        components += 1

    return components


if __name__ == "__main__":
    from env.generator.generator_factory import create_generator

    maze = Maze(rows=9, cols=9, random_seed=5)
    generator = create_generator("recursive_backtracking", random_seed=5)
    generator.generate(maze)

    print("Fully connected (valid maze)?", is_fully_connected(maze))
    print("Connected components (valid maze)?", count_connected_components(maze))
    print("Unreachable cells (valid maze)?", unreachable_path_cells(maze))

    # Deliberately break the maze: wall off an interior passage cell
    # that is currently the only connection between two regions, to
    # confirm the checks actually detect a broken maze rather than
    # trivially passing.
    broken_cell = maze.grid.get_cell(4, 3)
    broken_cell.is_wall = True

    print("\nAfter deliberately breaking a passage:")
    print("Fully connected (broken maze)?", is_fully_connected(maze))
    print("Connected components (broken maze)?", count_connected_components(maze))
    print("Unreachable cells (broken maze)?", unreachable_path_cells(maze))