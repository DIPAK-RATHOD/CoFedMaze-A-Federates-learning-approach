"""
recursive_backtracking.py

Implements Recursive Backtracking maze generation on top of the
Grid/Cell model, via the MazeGenerator interface.
"""

from typing import List, Set, Tuple

from env.core.constants import DIRECTIONS
from env.core.maze import Maze
from env.generator.generator_factory import MazeGenerator


class RecursiveBacktrackingGenerator(MazeGenerator):
    """
    Generates a perfect maze (a spanning tree over the logical grid —
    exactly one path between any two cells, no loops) using randomized
    depth-first search with backtracking.

    Algorithm:
        1. Start at a logical cell, mark it visited.
        2. From the current cell, look at unvisited logical neighbors.
           - If at least one exists: pick one at random, carve the
             passage to it, mark it visited, and move to it.
           - If none exist: backtrack to the previous cell on the
             stack and repeat.
        3. Stop when the stack is empty (every reachable cell visited).

    Because it always commits fully to a direction before backtracking,
    this algorithm tends to produce long, winding corridors with
    comparatively few short dead ends, versus Prim's more uniformly
    branching result.

    Traversal state (`visited`, `stack`) is kept as plain logical-
    coordinate tuples for cheap set/stack membership checks; only the
    two MazeGenerator helpers (`_mark_visited`, `_carve_logical_edge`)
    cross over into Grid's Cell-based API when structure actually needs
    to change.
    """

    def generate(self, maze: Maze) -> Maze:
        """
        Carve a maze into `maze` using Recursive Backtracking.

        Args:
            maze: A Maze with odd rows and odd cols.

        Returns:
            The same Maze instance, carved in place.
        """
        self._validate_odd_dimensions(maze)
        logical_rows, logical_cols = maze.grid.logical_rows, maze.grid.logical_cols
        self._validate_minimum_size(logical_rows, logical_cols)
        rng = self._get_rng(maze)

        start: Tuple[int, int] = (0, 0)
        visited: Set[Tuple[int, int]] = {start}
        stack: List[Tuple[int, int]] = [start]
        self._mark_visited(maze, start)

        while stack:
            current = stack[-1]
            candidates = self._unvisited_neighbors(
                current, visited, logical_rows, logical_cols
            )

            if not candidates:
                stack.pop()  # Dead end reached — backtrack.
                continue

            next_cell = rng.choice(candidates)
            self._carve_logical_edge(maze, current, next_cell)
            visited.add(next_cell)
            stack.append(next_cell)

        self._finalize(maze, "recursive_backtracking", logical_rows, logical_cols)
        return maze

    @staticmethod
    def _unvisited_neighbors(
        cell: Tuple[int, int],
        visited: Set[Tuple[int, int]],
        logical_rows: int,
        logical_cols: int,
    ) -> List[Tuple[int, int]]:
        """Return `cell`'s orthogonal logical neighbors not yet visited."""
        neighbors: List[Tuple[int, int]] = []
        for dr, dc in DIRECTIONS:
            candidate = (cell[0] + dr, cell[1] + dc)
            if (
                0 <= candidate[0] < logical_rows
                and 0 <= candidate[1] < logical_cols
                and candidate not in visited
            ):
                neighbors.append(candidate)
        return neighbors


if __name__ == "__main__":
    maze = Maze(rows=9, cols=9, random_seed=7)
    generator = RecursiveBacktrackingGenerator(random_seed=7)
    generator.generate(maze)

    print(maze)
    print(maze.to_numpy())