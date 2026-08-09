"""
generator_factory.py

Defines the MazeGenerator abstract base class shared by every maze
generation algorithm, plus a factory function for instantiating a
generator by name.

Design note — why the interface lives in this file:
    The directory layout for env/generator only specifies four files:
    recursive_backtracking.py, prim.py, kruskal.py, and
    generator_factory.py. Rather than introduce a fifth (base.py) that
    wasn't part of the agreed structure, the shared abstract interface
    and its common helper methods live here, since the factory already
    needs to know the shape every generator must conform to in order to
    return one. The three concrete algorithm files import MazeGenerator
    from this module.

Design note — generators orchestrate, Grid mutates:
    Grid (env/core/grid.py) already owns the double-resolution model:
    logical cells at even (row, col) indices are always open, and the
    wall-slot cells between them are what a generator carves via
    `Grid.carve_passage()`. This file previously reimplemented that
    coordinate math itself (a separate `_to_grid`/`_carve_logical`/
    `_connect_logical` set of helpers). That duplication is removed —
    every helper below that touches grid structure now delegates to
    Grid's own logical-space API (`get_logical_cell`, `carve_passage`),
    which also means adjacency and kind validation (previously absent
    here) now come for free instead of being a duplicated,
    un-enforced precondition on every caller.
"""

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Type

from env.core.maze import Maze


class MazeGenerator(ABC):
    """
    Abstract base class for all maze generation algorithms.

    Concrete subclasses (RecursiveBacktrackingGenerator, PrimGenerator,
    KruskalGenerator) implement `generate()` with their own traversal
    strategy over LOGICAL cell coordinates (plain (row, col) tuples,
    for cheap set/stack/frontier bookkeeping), and call the shared
    helpers below whenever that traversal needs to touch grid
    structure (marking a cell visited, carving a passage) or finalize
    the maze.
    """

    def __init__(self, random_seed: Optional[int] = None) -> None:
        """
        Args:
            random_seed: Seed for this generator's private random number
                generator. If None, `_get_rng()` falls back to the
                seed stored on the Maze being generated (if any),
                keeping a single source of truth for reproducibility
                when the caller has already set Maze.random_seed.
        """
        self.random_seed: Optional[int] = random_seed

    @abstractmethod
    def generate(self, maze: Maze) -> Maze:
        """
        Carve a maze into the given Maze's Grid, in place.

        Args:
            maze: A Maze with odd rows and odd cols.

        Returns:
            The same Maze instance, now carved.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _get_rng(self, maze: Maze) -> random.Random:
        """
        Build a private random.Random instance for this generation run.

        A local Random instance (rather than the global `random` module)
        is used deliberately: this framework will eventually run
        multiple generators across multiple federated nodes, potentially
        concurrently. Relying on the global random module would let one
        node's generation affect another's random stream; a private
        instance keeps each generation run fully isolated and
        reproducible from its seed alone.
        """
        seed = self.random_seed if self.random_seed is not None else maze.random_seed
        return random.Random(seed)

    @staticmethod
    def _validate_odd_dimensions(maze: Maze) -> None:
        """
        Ensure the Maze's Grid dimensions support double-resolution
        carving.

        In practice this can never fire: Maze.__init__ always builds a
        Grid immediately, and Grid.__init__ already rejects even
        dimensions, so a Maze object with even rows/cols cannot exist
        by the time a generator sees it. Kept anyway as defense-in-depth
        in case Maze ever gains a path to wrap a pre-built Grid.

        Raises:
            ValueError: If rows or cols is even.
        """
        if maze.rows % 2 == 0 or maze.cols % 2 == 0:
            raise ValueError(
                "Maze generation requires odd rows and odd cols "
                f"(got rows={maze.rows}, cols={maze.cols})."
            )

    @staticmethod
    def _validate_minimum_size(logical_rows: int, logical_cols: int) -> None:
        """
        Ensure the logical maze has at least two cells, so a distinct
        start and exit position can exist.

        Without this check, a 1x1 logical maze would compute
        start == exit == (0, 0) in `_finalize()`, and `Maze.set_exit()`
        would raise a ValueError from deep inside finalize with a
        message that doesn't explain *why* generation itself was
        invalid. Failing fast here, before any carving happens, gives a
        clearer error at the actual point of the problem.

        Raises:
            ValueError: If the logical maze has fewer than 2 cells.
        """
        if logical_rows * logical_cols < 2:
            raise ValueError(
                "Maze must have at least 2 logical cells to have distinct start "
                f"and exit positions (got a {logical_rows}x{logical_cols} logical maze)"
            )

    @staticmethod
    def _mark_visited(maze: Maze, logical_cell: Tuple[int, int]) -> None:
        """
        Mark a single logical cell's underlying Cell as visited.

        Logical cells are always open (is_wall=False) by construction in
        Grid, so there is nothing to "carve" for a lone cell — this only
        updates the `visited` bookkeeping flag, for consistency with
        `Grid.carve_passage()` (called via `_carve_logical_edge` below),
        which marks both of its endpoints visited as a side effect.
        Algorithms call this directly for the very first cell of a
        traversal (which has no predecessor edge to carve into it), and
        Kruskal's calls it for every cell up front, since every logical
        cell participates in the maze regardless of which edges are
        eventually carved.

        Args:
            maze: The Maze being generated.
            logical_cell: (logical_row, logical_col) tuple.
        """
        maze.grid.get_logical_cell(*logical_cell).visited = True

    @staticmethod
    def _carve_logical_edge(
        maze: Maze, cell_a: Tuple[int, int], cell_b: Tuple[int, int]
    ) -> None:
        """
        Carve a passage between two adjacent logical cells.

        Delegates directly to `Grid.carve_passage()`, which validates
        that both cells are LOGICAL and exactly one logical step apart
        before mutating anything, and marks both endpoints visited as
        part of the same call. `cell_a`/`cell_b` are plain (row, col)
        logical-coordinate tuples for the caller's convenience (matching
        the tuple-based visited-set/stack/frontier bookkeeping every
        concrete generator already uses); this method is the one place
        that converts them to Cell objects via `Grid.get_logical_cell()`.

        Args:
            maze: The Maze being generated.
            cell_a: (logical_row, logical_col) of the first cell.
            cell_b: (logical_row, logical_col) of the second cell,
                adjacent to cell_a.

        Raises:
            ValueError: If cell_a and cell_b are not adjacent logical
                cells (propagated from Grid.get_wall_between()).
        """
        grid = maze.grid
        grid.carve_passage(grid.get_logical_cell(*cell_a), grid.get_logical_cell(*cell_b))

    def _finalize(
        self,
        maze: Maze,
        algorithm_name: str,
        logical_rows: int,
        logical_cols: int,
    ) -> None:
        """
        Record generation metadata and set default start/exit positions.

        Start defaults to the top-left logical cell and exit to the
        bottom-right logical cell — the conventional opposite-corner
        layout. Both are guaranteed distinct because `generate()` calls
        `_validate_minimum_size()` before any carving happens. Callers
        that want a different layout can call maze.set_start()/
        maze.set_exit() again afterwards to override this default.

        Note: these are passed as LOGICAL coordinates directly —
        Maze.set_start()/set_exit() take logical coordinates and do
        their own bounds/kind handling internally, so no grid-coordinate
        conversion belongs here.
        """
        maze.algorithm_name = algorithm_name
        if self.random_seed is not None:
            maze.random_seed = self.random_seed

        maze.set_start(0, 0)
        maze.set_exit(logical_rows - 1, logical_cols - 1)


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------

_REGISTRY_CACHE: Optional[Dict[str, Type[MazeGenerator]]] = None


def _load_registry() -> Dict[str, Type[MazeGenerator]]:
    """
    Build (and cache) the name -> generator-class registry.

    The imports of the concrete generator classes are deferred to
    inside this function, rather than placed at module level, because
    each concrete generator module imports MazeGenerator from *this*
    file. An eager module-level import here would create a circular
    import (generator_factory -> recursive_backtracking ->
    generator_factory -> ...). Deferring the import to first call time
    breaks the cycle while still only doing the import work once,
    thanks to the cache.
    """
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        from env.generator.recursive_backtracking import RecursiveBacktrackingGenerator
        from env.generator.prim import PrimGenerator
        from env.generator.kruskal import KruskalGenerator

        _REGISTRY_CACHE = {
            "recursive_backtracking": RecursiveBacktrackingGenerator,
            "prim": PrimGenerator,
            "kruskal": KruskalGenerator,
        }
    return _REGISTRY_CACHE


def create_generator(name: str, random_seed: Optional[int] = None) -> MazeGenerator:
    """
    Instantiate a MazeGenerator by name.

    Args:
        name: One of "recursive_backtracking", "prim", "kruskal"
            (case-insensitive).
        random_seed: Seed forwarded to the generator's constructor.

    Returns:
        A new instance of the requested generator.

    Raises:
        ValueError: If `name` doesn't match a registered algorithm.
    """
    registry = _load_registry()
    key = name.lower()
    if key not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unknown generator algorithm {name!r}. Available: {available}")
    return registry[key](random_seed=random_seed)


def available_generators() -> List[str]:
    """Return the sorted list of registered generator algorithm names."""
    return sorted(_load_registry().keys())


if __name__ == "__main__":
    print("Available generators:", available_generators())

    maze = Maze(rows=9, cols=9, random_seed=1)
    generator = create_generator("recursive_backtracking", random_seed=1)
    generator.generate(maze)

    print(maze)
    print(maze.to_numpy())