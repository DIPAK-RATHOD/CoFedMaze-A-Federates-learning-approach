"""
kruskal.py

Implements randomized Kruskal's algorithm for maze generation on top of
the Grid/Cell model, via the MazeGenerator interface.
"""

from typing import List, Tuple

from networkx.utils import UnionFind

from env.core.maze import Maze
from env.generator.generator_factory import MazeGenerator

LogicalEdge = Tuple[Tuple[int, int], Tuple[int, int]]


class KruskalGenerator(MazeGenerator):
    """
    Generates a perfect maze using randomized Kruskal's algorithm.

    Algorithm:
        1. Every logical cell is already open by construction (Grid
           creates all logical cells with is_wall=False) — unlike
           Recursive Backtracking or Prim's, Kruskal's doesn't grow
           outward from one start point, so every cell "starts in the
           maze"; the only decisions are which walls *between* cells to
           carve. Each cell is marked visited up front purely for
           bookkeeping consistency with the other two generators.
        2. Build the full list of candidate edges (walls between
           orthogonally adjacent logical cells) and shuffle it.
        3. For each edge, in shuffled order: if its two endpoints
           belong to different disjoint sets (i.e. carving this wall
           wouldn't create a loop), remove the wall and union the two
           sets. If they're already in the same set, skip it — carving
           it would connect two cells that are already connected via
           some other path, creating a cycle rather than a tree.
        4. Stop once every edge has been considered.

    A disjoint-set / union-find structure (networkx.utils.UnionFind) is
    used to answer "are these two cells already connected?" in
    near-constant time, which is what makes this approach efficient —
    the alternative (a graph search from scratch per edge) would be far
    more expensive for larger mazes.

    Kruskal's produces mazes with a more uniformly random texture than
    either Recursive Backtracking or Prim's, since the decision order is
    driven purely by a global shuffle rather than growing from a
    frontier or a stack.
    """

    def generate(self, maze: Maze) -> Maze:
        """
        Carve a maze into `maze` using randomized Kruskal's algorithm.

        Args:
            maze: A Maze with odd rows and odd cols.

        Returns:
            The same Maze instance, carved in place.
        """
        self._validate_odd_dimensions(maze)
        logical_rows, logical_cols = maze.grid.logical_rows, maze.grid.logical_cols
        self._validate_minimum_size(logical_rows, logical_cols)
        rng = self._get_rng(maze)

        all_cells: List[Tuple[int, int]] = [
            (r, c) for r in range(logical_rows) for c in range(logical_cols)
        ]
        for cell in all_cells:
            self._mark_visited(maze, cell)

        edges = self._build_edges(logical_rows, logical_cols)
        rng.shuffle(edges)

        disjoint_sets: UnionFind = UnionFind(all_cells)

        for cell_a, cell_b in edges:
            if disjoint_sets[cell_a] != disjoint_sets[cell_b]:
                disjoint_sets.union(cell_a, cell_b)
                self._carve_logical_edge(maze, cell_a, cell_b)

        self._finalize(maze, "kruskal", logical_rows, logical_cols)
        return maze

    @staticmethod
    def _build_edges(logical_rows: int, logical_cols: int) -> List[LogicalEdge]:
        """
        Build the full list of candidate edges between orthogonally
        adjacent logical cells.

        Only "down" and "right" edges are generated per cell (rather
        than checking all four DIRECTIONS) so that each edge between a
        given pair of cells appears exactly once in the list, instead
        of twice (once from each side) — Kruskal's needs a single
        unambiguous edge set to shuffle and process, not a directed
        adjacency list.
        """
        edges: List[LogicalEdge] = []
        for r in range(logical_rows):
            for c in range(logical_cols):
                if r + 1 < logical_rows:
                    edges.append(((r, c), (r + 1, c)))
                if c + 1 < logical_cols:
                    edges.append(((r, c), (r, c + 1)))
        return edges


if __name__ == "__main__":
    maze = Maze(rows=9, cols=9, random_seed=7)
    generator = KruskalGenerator(random_seed=7)
    generator.generate(maze)

    print(maze)
    print(maze.to_numpy())