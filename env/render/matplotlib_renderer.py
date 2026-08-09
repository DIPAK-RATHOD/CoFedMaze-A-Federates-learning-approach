"""
matplotlib_renderer.py

Renders a Maze as an image using matplotlib — for visual inspection,
notebook display (notebooks/maze_visualization.ipynb), and saving
generated mazes to disk for a report or paper figure.

Coordinate space note: maze.to_numpy() (via Grid.to_numpy()) is a RAW
grid array, and ax.scatter() plots directly against that array's pixel
coordinates. Marker positions therefore must use
maze.start_grid_position / maze.exit_grid_position — NOT
maze.start_position / maze.exit_position, which are LOGICAL coordinates
that only coincidentally align with raw pixels at (0, 0). See
Maze.start_grid_position's docstring for why this distinction exists.
"""

from typing import Optional

import matplotlib

matplotlib.use("Agg")  # Headless-safe default; a caller running in a
# notebook or with an interactive backend already configured can set
# their own backend *before* importing this module, since matplotlib
# only allows the backend to be set once per process.

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from env.core.constants import AGENT_A, AGENT_B
from env.core.maze import Maze

AGENT_COLORS = {AGENT_A: "tab:blue", AGENT_B: "tab:orange"}


class MatplotlibRenderer:
    """
    Renders a Maze as a matplotlib image: walls and paths drawn as a
    black/white grid (via Grid.to_numpy()), with start, exit, and any
    placed agents overlaid as colored markers.
    """

    def render(
        self,
        maze: Maze,
        show: bool = False,
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> Figure:
        """
        Build a matplotlib Figure visualizing `maze`.

        Args:
            maze: The Maze to render.
            show: If True, calls plt.show(). Only produces a visible
                window under an interactive backend; harmless no-op
                under the default headless "Agg" backend.
            save_path: If given, saves the figure to this path
                (format inferred from the file extension, e.g. ".png").
            dpi: Resolution used when saving.

        Returns:
            The matplotlib Figure, so the caller can further customize
            it (e.g. in a notebook) before displaying or saving.
        """
        array = maze.to_numpy()  # 1 = wall, 0 = path — RAW grid array

        fig, ax = plt.subplots(figsize=(max(maze.cols / 3, 3), max(maze.rows / 3, 3)))
        # "gray_r" maps 1 (wall) -> black, 0 (path) -> white — the
        # reverse of matplotlib's default "gray" colormap, chosen so
        # the numeric wall encoding needs no inversion before display.
        ax.imshow(array, cmap="gray_r", interpolation="nearest", vmin=0, vmax=1)

        self._plot_marker(ax, maze.start_grid_position, color="tab:green", label="Start", marker="o")
        self._plot_marker(ax, maze.exit_grid_position, color="tab:red", label="Exit", marker="X")
        self._plot_agents(ax, maze)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{maze.algorithm_name or 'maze'}  "
            f"({maze.rows}x{maze.cols}, seed={maze.random_seed})"
        )

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles, labels, loc="upper center",
                bbox_to_anchor=(0.5, -0.03), ncol=len(labels), frameon=False,
            )

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    @staticmethod
    def _plot_marker(ax: Axes, position, color: str, label: str, marker: str) -> None:
        """
        Overlay a single marker (e.g. start or exit) if its position is
        set. `position` must already be in RAW grid coordinates — see
        module docstring.
        """
        if position is None:
            return
        row, col = position
        ax.scatter(
            [col], [row], c=color, s=140, marker=marker,
            edgecolors="black", linewidths=1.2, label=label, zorder=3,
        )

    @staticmethod
    def _plot_agents(ax: Axes, maze: Maze) -> None:
        """
        Overlay markers for every cell currently flagged with an agent.
        Iterates RAW grid coordinates and reads contains_agent directly
        off each Cell — unaffected by the logical/raw distinction, since
        it never compares against a stored position tuple.
        """
        seen_labels = set()
        for row in range(maze.rows):
            for col in range(maze.cols):
                agent = maze.grid.get_cell(row, col).contains_agent
                if agent not in AGENT_COLORS:
                    continue
                # Only attach a legend label the first time each agent
                # type appears, so the legend doesn't repeat "AGENT_A"
                # once per occupied cell.
                label = agent if agent not in seen_labels else None
                seen_labels.add(agent)
                ax.scatter(
                    [col], [row], c=AGENT_COLORS[agent], s=100, marker="s",
                    edgecolors="black", linewidths=1.0, label=label, zorder=3,
                )


if __name__ == "__main__":
    from env.generator.generator_factory import create_generator

    maze = Maze(rows=15, cols=21, random_seed=3)
    generator = create_generator("recursive_backtracking", random_seed=3)
    generator.generate(maze)

    renderer = MatplotlibRenderer()
    renderer.render(maze, save_path="maze_preview.png")
    print("Saved maze_preview.png")
