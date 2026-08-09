"""
grid.py

Defines the Grid class: a 2D container of Cell objects implementing the
double-resolution maze representation.

Double-resolution technique
----------------------------
Grid dimensions (`rows`, `cols`) must always be odd. This is not a
stylistic choice — it is what makes the double-resolution encoding
well-defined:

    - Cells at EVEN (row, col) indices are LOGICAL cells: the actual
      rooms/positions an agent can occupy. They are never walls.
    - Cells at any ODD row or ODD col index are WALL_SLOT cells. An
      (even, odd) or (odd, even) wall-slot sits *between* two logical
      neighbors — an edge wall a generator can carve open. An (odd, odd)
      wall-slot is a pillar at the intersection of four logical cells
      and is never carved.

This lets two different traversal models share one Grid:
    - Generation and logical-adjacency logic works in LOGICAL space,
      stepping by 2 raw grid units (see `get_logical_neighbors`,
      `get_wall_between`, `carve_passage`).
    - Full-resolution traversal (BFS validation, agent movement) works
      on the RAW grid, stepping by 1 and simply respecting `is_wall`
      (see `get_neighbors`) — a wall-slot cell blocks movement exactly
      like any other wall, with no special-casing required by the
      caller.

Grid is a pure data structure and utility layer — it knows how to create,
index, and query cells, but it has no knowledge of maze *generation*
(no carving-order or algorithm-specific state beyond the `carve_passage`
primitive) and no knowledge of higher-level Maze concepts like start/exit
positions or random seeds. That separation is what lets Recursive
Backtracking, Prim's, and Kruskal's all share the exact same Grid class
without any of them being coupled to each other.
"""

from typing import List

import numpy as np

from env.core.cell import Cell
from env.core.constants import DIRECTIONS, LOGICAL, WALL_SLOT


class Grid:
    """
    A 2D double-resolution grid of Cell objects, indexed by (row, col).

    Responsibilities:
        - Construct and hold a rows x cols matrix of Cell instances,
          with LOGICAL cells at even indices and WALL_SLOT cells at odd
          indices.
        - Provide safe, bounds-checked access to individual cells, in
          both raw and logical coordinate spaces.
        - Compute raw geometric neighbors (wall-agnostic, for
          traversal) and logical neighbors (for generation/adjacency).
        - Provide `carve_passage()` as the single mutation point for
          opening a passage between two logical cells.
        - Reset per-run traversal state (`visited`) without touching
          structural state (`is_wall`).
        - Export the grid as a numpy array for renderers, validators,
          or (later) RL observation construction.

    Grid intentionally does NOT:
        - Generate mazes (no carving-order/algorithm logic beyond the
          carve_passage primitive).
        - Know about start/exit positions (that belongs to Maze).
        - Know about agents or other placeable objects beyond exposing
          the Cell fields that already reserve space for them.
    """

    def __init__(self, rows: int, cols: int) -> None:
        """
        Create a new double-resolution Grid of the given raw dimensions.

        Every LOGICAL cell (even, even) is created open (is_wall=False);
        every WALL_SLOT cell (odd row and/or odd col) is created blocked
        (is_wall=True), ready for a generation algorithm to carve edge
        wall-slots open via `carve_passage()`.

        Args:
            rows: Raw number of rows. Must be a positive ODD integer —
                a logical maze of N rooms per axis needs a raw dimension
                of 2N - 1 to fit the wall-slots between and around them.
            cols: Raw number of columns. Same odd-dimension requirement
                as rows.

        Raises:
            ValueError: If rows or cols is not a positive integer, or if
                either is even.
        """
        if rows <= 0 or cols <= 0:
            raise ValueError(f"rows and cols must be positive, got rows={rows}, cols={cols}")
        if rows % 2 == 0 or cols % 2 == 0:
            raise ValueError(
                "rows and cols must be odd for the double-resolution grid "
                "(logical cells at even indices, walls at odd indices); "
                f"got rows={rows}, cols={cols}"
            )

        self.rows: int = rows
        self.cols: int = cols
        # Logical maze size: the number of actual rooms/positions per axis.
        self.logical_rows: int = (rows + 1) // 2
        self.logical_cols: int = (cols + 1) // 2

        self._cells: List[List[Cell]] = []
        for r in range(rows):
            row_cells: List[Cell] = []
            for c in range(cols):
                if r % 2 == 0 and c % 2 == 0:
                    row_cells.append(Cell(row=r, col=c, kind=LOGICAL, is_wall=False))
                else:
                    row_cells.append(Cell(row=r, col=c, kind=WALL_SLOT, is_wall=True))
            self._cells.append(row_cells)

    def get_cell(self, row: int, col: int) -> Cell:
        """
        Retrieve the Cell at the given RAW grid coordinates.

        Args:
            row: Row index.
            col: Column index.

        Returns:
            The Cell at (row, col).

        Raises:
            IndexError: If (row, col) is out of bounds.
        """
        if not self.is_valid(row, col):
            raise IndexError(
                f"Cell ({row}, {col}) is out of bounds for a {self.rows}x{self.cols} grid"
            )
        return self._cells[row][col]

    def is_valid(self, row: int, col: int) -> bool:
        """
        Check whether (row, col) lies within the grid's raw bounds.

        This is deliberately a pure bounds check — it says nothing about
        whether the cell is a wall, or whether it's a LOGICAL vs.
        WALL_SLOT cell. Callers that care use `is_logical()` or inspect
        `cell.is_wall` / `cell.kind` separately.

        Args:
            row: Row index to check.
            col: Column index to check.

        Returns:
            True if (row, col) is within [0, rows) x [0, cols), else False.
        """
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_logical(self, row: int, col: int) -> bool:
        """
        Check whether RAW coordinates (row, col) address an in-bounds
        LOGICAL cell (both indices even).

        Args:
            row: Raw row index to check.
            col: Raw column index to check.

        Returns:
            True if (row, col) is in-bounds and both indices are even.
        """
        return self.is_valid(row, col) and row % 2 == 0 and col % 2 == 0

    def get_logical_cell(self, logical_row: int, logical_col: int) -> Cell:
        """
        Retrieve a logical cell by its LOGICAL coordinates (room
        position — e.g. (0, 0) is the first room, not raw grid index).

        Args:
            logical_row: Logical row index, in [0, logical_rows).
            logical_col: Logical column index, in [0, logical_cols).

        Returns:
            The Cell at raw coordinates (logical_row * 2, logical_col * 2).

        Raises:
            IndexError: If the logical coordinates are out of bounds.
        """
        if not (0 <= logical_row < self.logical_rows and 0 <= logical_col < self.logical_cols):
            raise IndexError(
                f"Logical cell ({logical_row}, {logical_col}) is out of bounds for a "
                f"{self.logical_rows}x{self.logical_cols} logical grid"
            )
        return self._cells[logical_row * 2][logical_col * 2]

    def get_neighbors(self, cell: Cell) -> List[Cell]:
        """
        Return the in-bounds RAW geometric neighbors of the given cell,
        one step in each direction from constants.DIRECTIONS.

        Note: this returns *geometric* neighbors regardless of wall
        state or cell kind. This is the correct primitive for
        full-resolution traversal (BFS validation, agent movement): a
        caller walking the raw grid one step at a time and checking
        `neighbor.is_wall` gets correct double-resolution behavior for
        free, since wall-slots are just ordinary Cells with
        is_wall=True/False. Use `get_logical_neighbors()` instead for
        generation/adjacency decisions in logical space.

        Args:
            cell: The cell whose raw neighbors should be found.

        Returns:
            A list of in-bounds neighboring Cell objects (up to 4).
        """
        neighbors: List[Cell] = []
        for dr, dc in DIRECTIONS:
            nr, nc = cell.row + dr, cell.col + dc
            if self.is_valid(nr, nc):
                neighbors.append(self._cells[nr][nc])
        return neighbors

    def get_logical_neighbors(self, cell: Cell) -> List[Cell]:
        """
        Return the in-bounds LOGICAL neighbors of a logical cell (one
        logical step = two raw grid steps in each direction).

        This is the adjacency primitive generation algorithms should use
        for carving decisions (which room to visit/link next).

        Args:
            cell: The LOGICAL cell whose logical neighbors should be found.

        Returns:
            A list of in-bounds neighboring LOGICAL Cell objects (up to 4).

        Raises:
            ValueError: If `cell` is not a LOGICAL cell.
        """
        if cell.kind != LOGICAL:
            raise ValueError(
                f"get_logical_neighbors() requires a LOGICAL cell, got kind={cell.kind!r} "
                f"at ({cell.row}, {cell.col})"
            )
        neighbors: List[Cell] = []
        for dr, dc in DIRECTIONS:
            nr, nc = cell.row + dr * 2, cell.col + dc * 2
            if self.is_valid(nr, nc):
                neighbors.append(self._cells[nr][nc])
        return neighbors

    def get_wall_between(self, cell_a: Cell, cell_b: Cell) -> Cell:
        """
        Return the WALL_SLOT cell sitting between two adjacent LOGICAL
        cells. This is the cell a generator flips to `is_wall=False` to
        carve a passage between them.

        Args:
            cell_a: First LOGICAL cell.
            cell_b: Second LOGICAL cell, exactly one logical step from
                cell_a along a single axis.

        Returns:
            The WALL_SLOT Cell at the midpoint between cell_a and cell_b.

        Raises:
            ValueError: If either cell is not LOGICAL, or the two cells
                are not exactly one logical step apart along one axis.
        """
        if cell_a.kind != LOGICAL or cell_b.kind != LOGICAL:
            raise ValueError("get_wall_between() requires two LOGICAL cells")

        row_delta = cell_b.row - cell_a.row
        col_delta = cell_b.col - cell_a.col
        if abs(row_delta) + abs(col_delta) != 2 or (row_delta != 0 and col_delta != 0):
            raise ValueError(
                f"Cells ({cell_a.row}, {cell_a.col}) and ({cell_b.row}, {cell_b.col}) "
                "are not adjacent logical cells (must differ by exactly one logical "
                "step along a single axis)"
            )

        wall_row = cell_a.row + row_delta // 2
        wall_col = cell_a.col + col_delta // 2
        return self._cells[wall_row][wall_col]

    def carve_passage(self, cell_a: Cell, cell_b: Cell) -> None:
        """
        Carve a passage between two adjacent logical cells: flips the
        edge wall-slot between them to `is_wall=False` and marks both
        logical cells visited.

        This is the single mutation point every generation algorithm
        (recursive backtracking, Prim's, Kruskal's) should call rather
        than reaching into `Cell.is_wall` directly — centralizing it
        here enforces the double-resolution invariant that only *edge*
        wall-slots between logical neighbors ever get carved (pillar
        wall-slots at odd/odd indices are structurally unreachable
        through this method, since `get_wall_between` only ever returns
        an edge wall-slot for two valid logical neighbors).

        Args:
            cell_a: First LOGICAL cell.
            cell_b: Second LOGICAL cell, adjacent to cell_a.

        Raises:
            ValueError: If cell_a and cell_b are not adjacent logical cells.
        """
        wall = self.get_wall_between(cell_a, cell_b)
        wall.is_wall = False
        cell_a.visited = True
        cell_b.visited = True

    def reset_visits(self) -> None:
        """
        Clear the `visited` flag on every cell in the grid.

        Deliberately leaves `is_wall` untouched — this resets traversal
        state (useful for re-running BFS validation or restarting an
        agent episode on the same maze layout) without regenerating or
        altering the maze structure itself.
        """
        for row in self._cells:
            for cell in row:
                cell.visited = False

    def to_numpy(self) -> np.ndarray:
        """
        Export the grid as a 2D numpy array of integers.

        Encoding: 1 = wall, 0 = path. This is intentionally the
        minimal structural encoding (wall/path only) — object
        placement (agents, keys, doors, etc.) is not encoded here,
        since Grid has no opinion on how those should be represented
        numerically. A richer multi-channel encoding for RL
        observations can be built on top of this later, likely in
        Maze or a dedicated observation-builder module.

        Returns:
            A (rows, cols) numpy array of dtype int8, where 1 marks a
            wall and 0 marks a path.
        """
        return np.array(
            [[1 if cell.is_wall else 0 for cell in row] for row in self._cells],
            dtype=np.int8,
        )

    def __repr__(self) -> str:
        """Compact representation showing raw and logical dimensions."""
        return (
            f"Grid(rows={self.rows}, cols={self.cols}, "
            f"logical={self.logical_rows}x{self.logical_cols})"
        )


if __name__ == "__main__":
    # Minimal smoke test / usage example on a 4x4-logical maze (7x7 raw).
    grid = Grid(rows=7, cols=7)
    print(grid)

    a = grid.get_logical_cell(1, 1)
    b = grid.get_logical_cell(1, 2)
    print("Logical cell A:", a)
    print("Logical cell B:", b)
    print("Logical neighbors of A:", grid.get_logical_neighbors(a))

    wall = grid.get_wall_between(a, b)
    print("Wall-slot between A and B (before carve):", wall)

    grid.carve_passage(a, b)
    print("Wall-slot between A and B (after carve):", grid.get_wall_between(a, b))
    print("Raw neighbors of that wall-slot:", grid.get_neighbors(wall))

    grid.reset_visits()
    print("A.visited after reset_visits:", a.visited)

    print("Numpy view (1=wall, 0=open):\n", grid.to_numpy())

    try:
        Grid(rows=6, cols=7)
    except ValueError as e:
        print("Correctly rejected even dimension:", e)