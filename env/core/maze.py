"""
maze.py

Defines the Maze class: the top-level owner of a single maze instance.

Maze does not generate mazes and does not know how to carve paths — it
owns a Grid (the structural data), plus metadata about how that Grid came
to exist (algorithm_name, random_seed) and where the two most important
special positions are (start_position, exit_position). Generation
algorithms, validators, and renderers are all expected to operate *on* a
Maze's Grid, not be owned by it — keeping Maze a thin coordinator rather
than a place where generation logic accumulates.
"""

from typing import Optional, Tuple

import numpy as np

from env.core.grid import Grid


class Maze:
    """
    Represents a single maze instance: a Grid plus identifying metadata.

    Responsibilities:
        - Own a Grid object (the actual cell data).
        - Track how this maze was produced: algorithm_name, random_seed.
          This matters for reproducibility in a research context — every
          maze used in an experiment should be traceable back to the
          exact algorithm and seed that generated it.
        - Track start_position and exit_position, given and stored in
          LOGICAL coordinates (room position, not raw grid index) since
          that's the space agents/RL/renderers reason in. Maze is the
          single source of truth for these and always writes through to
          the corresponding Cell's contains_start/contains_exit flag, so
          the two never disagree.
        - Provide reset() to clear per-episode traversal state without
          discarding the maze layout, and to_numpy() for a numeric view.

    Maze intentionally does NOT:
        - Implement any maze-generation algorithm. A generator (e.g.
          RecursiveBacktracking, in env/generator) is expected to accept
          a Maze (or its Grid) and carve it via Grid.carve_passage();
          Maze does not call out to generators itself, avoiding a
          circular dependency between env/core and env/generator.
        - Validate connectivity or solvability. That is the job of
          env/validators (bfs_validator.py, connectivity.py,
          path_checker.py), which will operate on a Maze's Grid.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        random_seed: Optional[int] = None,
        algorithm_name: Optional[str] = None,
    ) -> None:
        """
        Create a new, ungenerated Maze.

        The Grid is constructed immediately (double-resolution, fully
        walled between logical cells, per Grid's own defaults) — this
        also means `rows`/`cols` must be odd, or Grid raises ValueError
        before Maze is even partially constructed.

        Args:
            rows: Raw number of rows for the underlying Grid (must be odd).
            cols: Raw number of columns for the underlying Grid (must be odd).
            random_seed: The seed used (or to be used) for reproducible
                generation. Stored here rather than only inside the
                generator so that a Maze object alone is enough to
                describe/reproduce a given layout, e.g. for logging or
                a federated-learning experiment manifest.
            algorithm_name: Name of the generation algorithm associated
                with this maze (e.g. "recursive_backtracking", "prim",
                "kruskal"). Stored as a plain string rather than an Enum
                so new algorithms can be added without touching this
                class.
        """
        self.rows: int = rows
        self.cols: int = cols
        self.random_seed: Optional[int] = random_seed
        self.algorithm_name: Optional[str] = algorithm_name

        self.grid: Grid = Grid(rows=rows, cols=cols)

        # Stored in LOGICAL coordinates. Unset until set_start()/
        # set_exit() are called, typically by a generator or a setup
        # step after generation.
        self.start_position: Optional[Tuple[int, int]] = None
        self.exit_position: Optional[Tuple[int, int]] = None

    def set_start(self, logical_row: int, logical_col: int) -> None:
        """
        Set the maze's start position, given in LOGICAL coordinates.

        Logical cells are never walls by construction (see Grid), so
        unlike a single-resolution grid there is no "is this cell
        actually open" check to perform here beyond bounds — only
        out-of-bounds and start==exit collisions are guarded against.

        Args:
            logical_row: Logical row index of the start room.
            logical_col: Logical column index of the start room.

        Raises:
            IndexError: If (logical_row, logical_col) is out of bounds
                for this maze's logical grid.
            ValueError: If the position is identical to the current exit
                position.
        """
        if not (
            0 <= logical_row < self.grid.logical_rows
            and 0 <= logical_col < self.grid.logical_cols
        ):
            raise IndexError(
                f"Start position (logical {logical_row}, {logical_col}) is out of "
                f"bounds for a {self.grid.logical_rows}x{self.grid.logical_cols} "
                "logical maze"
            )
        if self.exit_position == (logical_row, logical_col):
            raise ValueError(
                f"Start position cannot equal the current exit position "
                f"({logical_row}, {logical_col})"
            )

        if self.start_position is not None:
            self.grid.get_logical_cell(*self.start_position).contains_start = False

        self.grid.get_logical_cell(logical_row, logical_col).contains_start = True
        self.start_position = (logical_row, logical_col)

    def set_exit(self, logical_row: int, logical_col: int) -> None:
        """
        Set the maze's exit position, given in LOGICAL coordinates.

        Mirrors set_start(); see its docstring for the no-wall-check
        rationale.

        Args:
            logical_row: Logical row index of the exit room.
            logical_col: Logical column index of the exit room.

        Raises:
            IndexError: If (logical_row, logical_col) is out of bounds
                for this maze's logical grid.
            ValueError: If the position is identical to the current
                start position.
        """
        if not (
            0 <= logical_row < self.grid.logical_rows
            and 0 <= logical_col < self.grid.logical_cols
        ):
            raise IndexError(
                f"Exit position (logical {logical_row}, {logical_col}) is out of "
                f"bounds for a {self.grid.logical_rows}x{self.grid.logical_cols} "
                "logical maze"
            )
        if self.start_position == (logical_row, logical_col):
            raise ValueError(
                f"Exit position cannot equal the current start position "
                f"({logical_row}, {logical_col})"
            )

        if self.exit_position is not None:
            self.grid.get_logical_cell(*self.exit_position).contains_exit = False

        self.grid.get_logical_cell(logical_row, logical_col).contains_exit = True
        self.exit_position = (logical_row, logical_col)

    @property
    def start_grid_position(self) -> Optional[Tuple[int, int]]:
        """
        The start position in RAW grid coordinates, derived from
        start_position (logical). Returns None if start_position is
        unset.

        This is the single conversion point from logical to raw
        coordinates for start — any code that needs to index directly
        into Grid (e.g. BFS traversal) should use this rather than
        re-deriving raw coordinates from start_position itself, which
        is a common source of bugs: start_position and raw grid
        coordinates look identical in shape (both (int, int) tuples)
        but are NOT interchangeable except by coincidence at (0, 0).
        """
        if self.start_position is None:
            return None
        cell = self.grid.get_logical_cell(*self.start_position)
        return (cell.row, cell.col)

    @property
    def exit_grid_position(self) -> Optional[Tuple[int, int]]:
        """
        The exit position in RAW grid coordinates, derived from
        exit_position (logical). Returns None if exit_position is
        unset. See start_grid_position for why this conversion point
        exists rather than letting callers do the math themselves.
        """
        if self.exit_position is None:
            return None
        cell = self.grid.get_logical_cell(*self.exit_position)
        return (cell.row, cell.col)

    def reset(self) -> None:
        """
        Reset per-episode state without discarding the maze layout.

        Delegates to Grid.reset_visits(), which clears `visited` on every
        cell but leaves `is_wall` untouched. start_position and
        exit_position (and their mirrored Cell flags) are intentionally
        NOT cleared here — those are structural properties of this maze
        instance, not transient episode state (analogous to how is_wall
        survives a reset while visited does not).
        """
        self.grid.reset_visits()

    def to_numpy(self) -> np.ndarray:
        """
        Export the maze's structural layout as a numpy array.

        Delegates directly to Grid.to_numpy() (1 = wall, 0 = path).
        Maze does not add start/exit/object encoding here, for the same
        reason Grid doesn't: a richer multi-channel encoding is a
        downstream concern (e.g. an RL observation builder) that
        shouldn't be baked into this foundational data model.

        Returns:
            A (rows, cols) numpy array of dtype int8.
        """
        return self.grid.to_numpy()

    def __repr__(self) -> str:
        """Compact representation showing key identifying metadata."""
        return (
            f"Maze(rows={self.rows}, cols={self.cols}, "
            f"algorithm={self.algorithm_name!r}, seed={self.random_seed}, "
            f"start={self.start_position}, exit={self.exit_position})"
        )


if __name__ == "__main__":
    # Minimal smoke test / usage example on a 4x4-logical maze (7x7 raw).
    maze = Maze(rows=7, cols=7, random_seed=42, algorithm_name="recursive_backtracking")
    print(maze)

    maze.set_start(0, 0)
    maze.set_exit(3, 3)
    print("After setting start/exit:", maze)

    # Simulate a generator carving one passage directly on the Grid (no
    # generation algorithm is implemented yet — this just proves the
    # Maze -> Grid -> Cell chain, including carve_passage, is wired
    # correctly).
    a = maze.grid.get_logical_cell(0, 0)
    b = maze.grid.get_logical_cell(0, 1)
    maze.grid.carve_passage(a, b)

    print("Numpy view before reset:\n", maze.to_numpy())

    maze.reset()
    print("Start/exit survive reset:", maze.start_position, maze.exit_position)
    print("A.visited after reset:", a.visited)

    try:
        maze.set_exit(0, 0)
    except ValueError as e:
        print("Correctly rejected duplicate start/exit:", e)