"""
prim.py

Implements randomized Prim's algorithm for maze generation on top of
the Grid/Cell model, via the MazeGenerator interface.
"""

from typing import List, Set, Tuple

from env.core.constants import DIRECTIONS
from env.core.maze import Maze
from env.generator.generator_factory import MazeGenerator

# A frontier entry records a candidate passage: the already-visited
# logical cell it would come from, and the unvisited logical cell it
# would lead to.
FrontierEdge = Tuple[Tuple[int, int], Tuple[int, int]]


class PrimGenerator(MazeGenerator):
    """
    Generates a perfect maze using randomized Prim's algorithm.

    Algorithm:
        1. Start at a logical cell, mark it visited, and add all of its
           unvisited neighbors to a frontier list.
        2. Repeatedly pick a random edge from the frontier.
           - If the edge's target cell is still unvisited: carve the
             passage, mark the target visited, and add its unvisited
             neighbors to the frontier.
           - If the target was already visited in the meantime (it can
             be added to the frontier more than once, from different
             sides): discard the edge and try another.
        3. Stop when the frontier is empty.

    Because it grows outward from many active edges at once rather than
    committing to one path at a time, Prim's tends to produce mazes
    with more short branches and a more uniform, less corridor-heavy
    texture than Recursive Backtracking.
    """

    def generate(self, maze: Maze) -> Maze:
        """
        Carve a maze into `maze` using randomized Prim's algorithm.

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
        self._mark_visited(maze, start)

        frontier: List[FrontierEdge] = []
        self._extend_frontier(frontier, start, visited, logical_rows, logical_cols)

        while frontier:
            index = rng.randrange(len(frontier))
            from_cell, to_cell = frontier.pop(index)

            if to_cell in visited:
                # This target was reached via a different frontier edge
                # since this one was added — skip it rather than
                # re-carving (carving is idempotent, but skipping keeps
                # the algorithm's intent explicit: only unvisited
                # targets should trigger a new passage).
                continue

            self._carve_logical_edge(maze, from_cell, to_cell)
            visited.add(to_cell)
            self._extend_frontier(frontier, to_cell, visited, logical_rows, logical_cols)

        self._finalize(maze, "prim", logical_rows, logical_cols)
        return maze

    @staticmethod
    def _extend_frontier(
        frontier: List[FrontierEdge],
        cell: Tuple[int, int],
        visited: Set[Tuple[int, int]],
        logical_rows: int,
        logical_cols: int,
    ) -> None:
        """Add all of `cell`'s unvisited-neighbor edges to the frontier."""
        for dr, dc in DIRECTIONS:
            neighbor = (cell[0] + dr, cell[1] + dc)
            if (
                0 <= neighbor[0] < logical_rows
                and 0 <= neighbor[1] < logical_cols
                and neighbor not in visited
            ):
                frontier.append((cell, neighbor))


if __name__ == "__main__":
    maze = Maze(rows=9, cols=9, random_seed=7)
    generator = PrimGenerator(random_seed=7)
    generator.generate(maze)

    print(maze)
    print(maze.to_numpy())