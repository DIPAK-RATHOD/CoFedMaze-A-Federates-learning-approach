"""
bfs_validator.py

Implements BFSValidator: a reusable breadth-first traversal engine over
a Maze's Grid.

This module deliberately contains no maze-wide "is this a valid maze"
logic and no start-to-exit-specific logic — it only knows how to walk
the Grid from a given cell and report what it found. connectivity.py
and path_checker.py both build their higher-level checks on top of
this traversal engine, so the actual graph-search code exists in
exactly one place.
"""

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from env.core.grid import Grid
from env.core.maze import Maze

Coordinate = Tuple[int, int]


class BFSValidator:
    """
    Performs breadth-first search over the walkable (non-wall) cells of
    a Maze's Grid.

    A BFSValidator is bound to one Maze at construction time and can
    run multiple traversals from different start points against the
    same underlying Grid.
    """

    def __init__(self, maze: Maze) -> None:
        """
        Args:
            maze: The Maze whose Grid this validator will traverse.
        """
        self.maze: Maze = maze
        self.grid: Grid = maze.grid

    def traverse(self, start: Coordinate) -> Dict[Coordinate, Optional[Coordinate]]:
        """
        Run a breadth-first search from `start` over walkable cells.

        Args:
            start: (row, col) coordinate to start the search from.

        Returns:
            A parent map: {cell: parent_cell} for every cell reached,
            where `start` maps to None. This is the minimal data needed
            to both list reachable cells (the map's keys) and
            reconstruct a shortest path to any reached cell (by walking
            parent pointers backward), so both connectivity.py and
            path_checker.py can be built on the same return value.

        Raises:
            ValueError: If `start` is out of bounds or is a wall cell
                (BFS cannot begin inside a wall).
        """
        if not self.grid.is_valid(*start):
            raise ValueError(f"BFS start position {start} is out of bounds for this maze's grid")

        start_cell = self.grid.get_cell(*start)
        if start_cell.is_wall:
            raise ValueError(f"BFS start position {start} is a wall cell — cannot start traversal there")

        parents: Dict[Coordinate, Optional[Coordinate]] = {start: None}
        queue: deque = deque([start])

        while queue:
            current = queue.popleft()
            current_cell = self.grid.get_cell(*current)

            for neighbor_cell in self.grid.get_neighbors(current_cell):
                neighbor_coord = (neighbor_cell.row, neighbor_cell.col)
                if neighbor_cell.is_wall or neighbor_coord in parents:
                    continue
                parents[neighbor_coord] = current
                queue.append(neighbor_coord)

        return parents

    def reachable_cells(self, start: Coordinate) -> Set[Coordinate]:
        """
        Return the set of every walkable cell reachable from `start`,
        including `start` itself.
        """
        return set(self.traverse(start).keys())

    @staticmethod
    def reconstruct_path(
        parents: Dict[Coordinate, Optional[Coordinate]], target: Coordinate
    ) -> List[Coordinate]:
        """
        Reconstruct the path from a traversal's start to `target` by
        walking the parent map backward.

        Args:
            parents: The parent map returned by `traverse()`.
            target: The coordinate to reconstruct the path to.

        Returns:
            A list of coordinates from start to target, inclusive, in
            order. Empty list if `target` was not reached by the
            traversal that produced `parents`.
        """
        if target not in parents:
            return []

        path: List[Coordinate] = []
        current: Optional[Coordinate] = target
        while current is not None:
            path.append(current)
            current = parents[current]
        path.reverse()
        return path


if __name__ == "__main__":
    from env.generator.generator_factory import create_generator

    maze = Maze(rows=9, cols=9, random_seed=5)
    generator = create_generator("recursive_backtracking", random_seed=5)
    generator.generate(maze)

    validator = BFSValidator(maze)
    parents = validator.traverse(maze.start_position)

    print("Cells reached from start:", len(parents))
    print("Exit reached?", maze.exit_position in parents)

    path = validator.reconstruct_path(parents, maze.exit_position)
    print("Path length (cells):", len(path))
    print("Path:", path)