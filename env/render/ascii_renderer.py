"""
ascii_renderer.py

Renders a Maze as a plain-text ASCII grid — a zero-dependency way to
inspect a maze directly in a terminal or log file, with no plotting
library required.

Coordinate space note: this renderer iterates RAW grid coordinates
(range(maze.rows) x range(maze.cols)), so any comparison against
start/exit must use maze.start_grid_position / maze.exit_grid_position
— NOT maze.start_position / maze.exit_position, which are LOGICAL
coordinates and only coincidentally equal their raw counterparts at
(0, 0). See Maze.start_grid_position's docstring for why this
distinction exists. (Object presence — contains_agent/contains_key/etc.
— is read directly off each iterated Cell and is unaffected by this;
only the start/exit marker comparison needed fixing.)
"""

from env.core.constants import AGENT_A, AGENT_B
from env.core.maze import Maze

# Visual vocabulary for ASCII rendering. This is a rendering-layer
# concern, deliberately kept separate from env.core.constants: the
# core constants define what a cell *means* (WALL, PATH, AGENT_A...),
# while these characters define how ascii_renderer chooses to *display*
# that meaning. A different renderer is free to use a different
# vocabulary without touching env.core at all.
WALL_CHAR = "#"
PATH_CHAR = " "
START_CHAR = "S"
EXIT_CHAR = "E"
AGENT_A_CHAR = "A"
AGENT_B_CHAR = "B"
KEY_CHAR = "K"
DOOR_CHAR = "D"
CHECKPOINT_CHAR = "C"
OBSTACLE_CHAR = "X"


class AsciiRenderer:
    """
    Renders a Maze to a plain-text string, one character per Grid cell.

    Rendering precedence per cell (first match wins, since only one
    character can be drawn per cell): wall, start, exit, agent, key,
    door, checkpoint, obstacle, path. This ordering only matters if
    multiple features are ever placed on the same cell simultaneously;
    in normal use each cell has at most one non-wall feature.
    """

    def render(self, maze: Maze) -> str:
        """
        Build the ASCII representation of `maze` as a single string,
        one row per line, ready to print or write to a log file.
        """
        lines = []
        for row in range(maze.rows):
            line_chars = [self._char_for(maze, row, col) for col in range(maze.cols)]
            lines.append("".join(line_chars))
        return "\n".join(lines)

    def print_maze(self, maze: Maze) -> None:
        """Convenience method: render and print directly to stdout."""
        print(self.render(maze))

    @staticmethod
    def _char_for(maze: Maze, row: int, col: int) -> str:
        """Determine the single display character for one grid cell."""
        cell = maze.grid.get_cell(row, col)

        if cell.is_wall:
            return WALL_CHAR
        if maze.start_grid_position == (row, col):
            return START_CHAR
        if maze.exit_grid_position == (row, col):
            return EXIT_CHAR
        if cell.contains_agent == AGENT_A:
            return AGENT_A_CHAR
        if cell.contains_agent == AGENT_B:
            return AGENT_B_CHAR
        if cell.contains_key:
            return KEY_CHAR
        if cell.contains_door:
            return DOOR_CHAR
        if cell.contains_checkpoint:
            return CHECKPOINT_CHAR
        if cell.contains_obstacle:
            return OBSTACLE_CHAR
        return PATH_CHAR


if __name__ == "__main__":
    from env.generator.generator_factory import create_generator

    maze = Maze(rows=15, cols=21, random_seed=3)
    generator = create_generator("recursive_backtracking", random_seed=3)
    generator.generate(maze)

    renderer = AsciiRenderer()
    renderer.print_maze(maze)
