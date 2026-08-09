"""
observations.py

Builds the egocentric, multi-channel observation array consumed by
marl/models/components/encoder.py: a fixed-size (C, H, W) window of
LOGICAL cells centered on one agent.

THIS FILE IS THE HIGHEST-RISK FILE IN THIS CODEBASE, because it
translates between three different coordinate spaces in one place:
    1. WINDOW-RELATIVE coordinates: (wr, wc) in [0, window_size), the
       index into the output array itself.
    2. LOGICAL maze coordinates: (logical_row, logical_col) -- the
       space Maze.start_position/exit_position, Agent.position, and
       every object in env/objects/ live in.
    3. RAW grid coordinates: (row, col) -- the space Grid.get_cell()
       indexes into.

Every prior coordinate bug in this codebase (env/generator/generator_factory.py's
_finalize, all three env/validators/ files, both env/render/ files --
see tests/env/test_coordinate_consistency.py) came from silently
treating LOGICAL coordinates as RAW, or vice versa, with no crash,
because the mismatched coordinate was almost always still a valid cell
-- just the wrong one. To make that mistake structurally hard here: the
ONLY conversion out of window-relative space is _window_to_logical()
below, and the ONLY way this module ever touches raw grid coordinates
is through Grid.get_logical_cell() / Grid.get_wall_between() -- never
hand-rolled *2 arithmetic.

Known limitation (documented, not silently worked around): Cell.contains_door
and Cell.contains_checkpoint (cell.py) are plain booleans -- they cannot
express a door's lock state or a checkpoint's reached state, only that
*something* of that type is present. That state lives on the actual
Door/Checkpoint instances (env/objects/), not the Cell. This module
therefore accepts OPTIONAL registries (LOGICAL position -> instance) for
those two object types; without a registry, it falls back to the
Cell-only boolean (a locked-or-unlocked door reports identically, and a
reached-or-unreached checkpoint reports identically) -- a real but
bounded loss of information, not silently fabricated state.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from env.core.constants import DIRECTIONS
from env.core.maze import Maze
from env.objects.checkpoint import Checkpoint
from env.objects.door import Door

# ---------------------------------------------------------------------------
# Channel layout
# ---------------------------------------------------------------------------
# Whatever constructs a VDNModel (marl/models/vdn.py) MUST use
# in_channels=NUM_CHANNELS, or the encoder's input-shape validation will
# reject every observation this module produces.
CAN_MOVE_UP = 0
CAN_MOVE_DOWN = 1
CAN_MOVE_LEFT = 2
CAN_MOVE_RIGHT = 3
CONTAINS_OTHER_AGENT = 4
CONTAINS_EXIT = 5
CONTAINS_KEY = 6
DOOR_BLOCKING = 7
CHECKPOINT_UNREACHED = 8
CONTAINS_OBSTACLE = 9

NUM_CHANNELS = 10

MIN_WINDOW_SIZE = 3


def _window_to_logical(
    agent_position: Tuple[int, int], window_row: int, window_col: int, half: int
) -> Tuple[int, int]:
    """
    The single conversion point from WINDOW-RELATIVE to LOGICAL
    coordinates. Every other function in this module reaches logical
    coordinates only through this function, so there is exactly one
    place the window-centering arithmetic can be wrong.

    Args:
        agent_position: The LOGICAL position the window is centered on.
        window_row, window_col: Window-relative indices in [0, window_size).
        half: window_size // 2.

    Returns:
        (logical_row, logical_col) for that window cell. May be out of
        the maze's logical bounds near the maze edge -- callers must
        check with maze.grid.is_valid-equivalent bounds before indexing
        Grid (see _passability_channels / _object_channels below).
    """
    return (agent_position[0] + window_row - half, agent_position[1] + window_col - half)


def _in_logical_bounds(maze: Maze, logical_position: Tuple[int, int]) -> bool:
    """Bounds check in LOGICAL space (maze.grid.logical_rows/cols)."""
    row, col = logical_position
    return 0 <= row < maze.grid.logical_rows and 0 <= col < maze.grid.logical_cols


def _passability(maze: Maze, logical_position: Tuple[int, int]) -> Tuple[bool, bool, bool, bool]:
    """
    Compute (can_move_up, can_move_down, can_move_left, can_move_right)
    for the LOGICAL cell at `logical_position`.

    Reuses Grid.get_logical_cell() + Grid.get_wall_between() exclusively
    -- never derives raw coordinates by hand -- so adjacency/kind
    validation already built into Grid applies here for free.

    A direction is False if the neighbor in that direction would fall
    outside the maze's logical bounds (there is nothing to move into),
    or if the wall-slot between the two logical cells has not been
    carved.
    """
    cell = maze.grid.get_logical_cell(*logical_position)
    results = []
    for dlr, dlc in DIRECTIONS:  # DIRECTIONS order is [UP, DOWN, LEFT, RIGHT]
        neighbor_logical = (logical_position[0] + dlr, logical_position[1] + dlc)
        if not _in_logical_bounds(maze, neighbor_logical):
            results.append(False)
            continue
        neighbor_cell = maze.grid.get_logical_cell(*neighbor_logical)
        wall = maze.grid.get_wall_between(cell, neighbor_cell)
        results.append(not wall.is_wall)
    return tuple(results)  # type: ignore[return-value]


def build_observation(
    maze: Maze,
    agent_position: Tuple[int, int],
    self_agent_id: str,
    window_size: int = 5,
    door_registry: Optional[Dict[Tuple[int, int], Door]] = None,
    checkpoint_registry: Optional[Dict[Tuple[int, int], Checkpoint]] = None,
) -> np.ndarray:
    """
    Build the (NUM_CHANNELS, window_size, window_size) egocentric
    observation array for one agent.

    Args:
        maze: The Maze to observe.
        agent_position: This agent's current LOGICAL position (e.g.
            from Agent.position) -- the window is centered here.
        self_agent_id: This agent's id (AGENT_A/AGENT_B), used only to
            distinguish "the other agent" from "myself" when reading
            Cell.contains_agent -- self is always at the window's exact
            center by construction and is not separately encoded.
        window_size: Odd, >= MIN_WINDOW_SIZE. Must match the
            window_size the SharedEncoder consuming this array was
            constructed with.
        door_registry: Optional {LOGICAL position: Door instance} map.
            Without it, DOOR_BLOCKING falls back to the raw
            Cell.contains_door boolean (see module docstring's "Known
            limitation").
        checkpoint_registry: Optional {LOGICAL position: Checkpoint
            instance} map. Without it, CHECKPOINT_UNREACHED falls back
            to the raw Cell.contains_checkpoint boolean.

    Returns:
        A (NUM_CHANNELS, window_size, window_size) float32 array, ready
        to feed into SharedEncoder after adding a batch dimension.

    Raises:
        ValueError: If window_size is even or < MIN_WINDOW_SIZE.
        IndexError: If agent_position is out of the maze's logical bounds.
    """
    if window_size % 2 == 0 or window_size < MIN_WINDOW_SIZE:
        raise ValueError(
            f"window_size must be odd and >= {MIN_WINDOW_SIZE}, got {window_size}"
        )
    if not _in_logical_bounds(maze, agent_position):
        raise IndexError(
            f"agent_position (logical) {agent_position} is out of bounds for a "
            f"{maze.grid.logical_rows}x{maze.grid.logical_cols} logical maze"
        )

    half = window_size // 2
    obs = np.zeros((NUM_CHANNELS, window_size, window_size), dtype=np.float32)

    for wr in range(window_size):
        for wc in range(window_size):
            logical_position = _window_to_logical(agent_position, wr, wc, half)
            if not _in_logical_bounds(maze, logical_position):
                continue  # stays all-zero: off the maze edge, nothing there

            up, down, left, right = _passability(maze, logical_position)
            obs[CAN_MOVE_UP, wr, wc] = up
            obs[CAN_MOVE_DOWN, wr, wc] = down
            obs[CAN_MOVE_LEFT, wr, wc] = left
            obs[CAN_MOVE_RIGHT, wr, wc] = right

            cell = maze.grid.get_logical_cell(*logical_position)

            other_agent = cell.contains_agent is not None and cell.contains_agent != self_agent_id
            obs[CONTAINS_OTHER_AGENT, wr, wc] = other_agent

            obs[CONTAINS_EXIT, wr, wc] = (
                maze.exit_position is not None and logical_position == maze.exit_position
            )

            obs[CONTAINS_KEY, wr, wc] = cell.contains_key

            if door_registry is not None and logical_position in door_registry:
                obs[DOOR_BLOCKING, wr, wc] = door_registry[logical_position].blocks_movement
            else:
                obs[DOOR_BLOCKING, wr, wc] = cell.contains_door

            if checkpoint_registry is not None and logical_position in checkpoint_registry:
                obs[CHECKPOINT_UNREACHED, wr, wc] = not checkpoint_registry[logical_position].is_reached
            else:
                obs[CHECKPOINT_UNREACHED, wr, wc] = cell.contains_checkpoint

            obs[CONTAINS_OBSTACLE, wr, wc] = cell.contains_obstacle

    return obs


if __name__ == "__main__":
    from env.core.constants import AGENT_A, AGENT_B
    from env.generator.generator_factory import create_generator
    from env.objects.agent import Agent

    maze = Maze(rows=9, cols=9, random_seed=1)
    create_generator("recursive_backtracking", random_seed=1).generate(maze)

    agent_a = Agent(AGENT_A)
    agent_a.place_at(maze, 2, 2)
    agent_b = Agent(AGENT_B)
    agent_b.place_at(maze, 2, 3)

    obs = build_observation(maze, agent_position=(2, 2), self_agent_id=AGENT_A, window_size=5)
    print("Observation shape:", obs.shape)
    print("Center cell can_move_right (should match agent_a<->agent_b's carved state):",
          obs[CAN_MOVE_RIGHT, 2, 2])
    print("Other-agent channel, cell just right of center:", obs[CONTAINS_OTHER_AGENT, 2, 3])

    # Off-the-edge padding check: agent at logical (0, 0), corner of the maze.
    obs_corner = build_observation(maze, agent_position=(0, 0), self_agent_id=AGENT_A, window_size=5)
    print("Off-maze window cell (top-left) is all zero:", obs_corner[:, 0, 0].sum() == 0)
