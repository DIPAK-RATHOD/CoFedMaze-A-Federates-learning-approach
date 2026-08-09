"""Report-facing maze-figure helper that reuses the environment renderer."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt

from env.core.maze import Maze
from env.render.matplotlib_renderer import MatplotlibRenderer

PathLike = Union[str, Path]


def save_maze_plot(maze: Maze, output_path: PathLike, dpi: int = 150) -> Path:
    """Save a generated maze figure without duplicating environment drawing logic."""
    if dpi < 1:
        raise ValueError("dpi must be positive")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = MatplotlibRenderer().render(maze, save_path=str(destination), dpi=dpi)
    plt.close(figure)
    return destination
