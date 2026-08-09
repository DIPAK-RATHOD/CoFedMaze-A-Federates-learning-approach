"""
test_coordinate_consistency.py

Regression guard against the logical-vs-raw coordinate bug that has
appeared FOUR times in this codebase:
    1. env/generator/generator_factory.py's _finalize() (fed raw
       coordinates into Maze.set_start/set_exit, which expect logical)
    2-4. env/validators/bfs_validator.py, connectivity.py,
       path_checker.py (fed Maze.start_position/exit_position -- LOGICAL
       -- directly into Grid methods expecting RAW coordinates)
    5. env/render/ascii_renderer.py and matplotlib_renderer.py (same
       mistake as 2-4, in the rendering layer)

Every instance was SILENT — no crash, because the mismatched coordinate
almost always happens to be a valid, real, open cell in the grid (just
the wrong one). That's what makes this bug class dangerous and worth a
standing regression guard rather than relying on catching it by eye
every time new code touches Maze.start_position/exit_position.

These tests don't just check "the code runs" — they specifically assert
that logical and raw coordinates are DIFFERENT for a non-trivial maze
(the canary), and that every consumer resolves to the cell that is
ACTUALLY flagged (contains_start/contains_exit=True) on the Grid, not
merely a plausible-looking cell.
"""

import pytest

from env.core.maze import Maze
from env.generator.generator_factory import available_generators, create_generator
from env.render.ascii_renderer import AsciiRenderer, EXIT_CHAR, START_CHAR
from env.validators.path_checker import has_path_to_exit, shortest_path

GRID_SIZES = [(9, 9), (15, 21), (7, 13)]
SEEDS = [1, 2, 3, 42]


def _generated_mazes():
    """Yield (label, Maze) for every algorithm x seed x size combination."""
    for algo in available_generators():
        for seed in SEEDS:
            for rows, cols in GRID_SIZES:
                maze = Maze(rows=rows, cols=cols, random_seed=seed)
                create_generator(algo, random_seed=seed).generate(maze)
                label = f"{algo}-seed{seed}-{rows}x{cols}"
                yield label, maze


_CASES = list(_generated_mazes())
_IDS = [label for label, _ in _CASES]


@pytest.mark.parametrize("label,maze", _CASES, ids=_IDS)
def test_logical_and_raw_exit_positions_actually_differ(label, maze):
    """
    Canary: for any non-degenerate maze, exit_position (logical) and
    exit_grid_position (raw) must be different tuples. If this ever
    starts failing, Grid's logical<->raw mapping changed in a way that
    could silently re-collapse the exact distinction these tests exist
    to protect — investigate that before touching anything else.
    """
    assert maze.exit_position != maze.exit_grid_position, (
        "exit_position and exit_grid_position coincide — either this maze is "
        "degenerate, or Grid's coordinate mapping changed"
    )


@pytest.mark.parametrize("label,maze", _CASES, ids=_IDS)
def test_exit_grid_position_is_the_true_flagged_cell(label, maze):
    """
    exit_grid_position must point at the Cell with contains_exit=True.
    If exit_position (the LOGICAL tuple) is fed into Grid.get_cell()
    directly — the exact mistake this guards against — it must NOT
    resolve to a cell that is wrongly treated as the exit.
    """
    true_cell = maze.grid.get_cell(*maze.exit_grid_position)
    assert true_cell.contains_exit is True

    if maze.grid.is_valid(*maze.exit_position):
        wrong_cell = maze.grid.get_cell(*maze.exit_position)
        assert wrong_cell.contains_exit is False


@pytest.mark.parametrize("label,maze", _CASES, ids=_IDS)
def test_path_checker_terminates_at_true_start_and_exit(label, maze):
    """
    path_checker.shortest_path() must start/end at the RAW grid
    positions, not silently at the logical-coordinate lookalikes.
    """
    assert has_path_to_exit(maze)
    path = shortest_path(maze)
    assert path[0] == maze.start_grid_position
    assert path[-1] == maze.exit_grid_position


@pytest.mark.parametrize("label,maze", _CASES, ids=_IDS)
def test_ascii_renderer_marks_true_start_and_exit(label, maze):
    """
    The S/E characters in the ASCII render must land on the true raw
    start/exit cells, not the logical-coordinate lookalikes.
    """
    rendered = AsciiRenderer().render(maze)
    lines = rendered.split("\n")
    s_pos = next((r, line.index(START_CHAR)) for r, line in enumerate(lines) if START_CHAR in line)
    e_pos = next((r, line.index(EXIT_CHAR)) for r, line in enumerate(lines) if EXIT_CHAR in line)
    assert s_pos == maze.start_grid_position
    assert e_pos == maze.exit_grid_position
