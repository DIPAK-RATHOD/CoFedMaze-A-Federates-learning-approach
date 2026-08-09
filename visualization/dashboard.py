"""Compose saved CoFedMaze plots into one report/dashboard image."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

PathLike = Union[str, Path]


def create_dashboard(
    panels: Mapping[str, PathLike], output_path: PathLike, title: str = "CoFedMaze Experiment Dashboard",
    columns: int = 2,
) -> Path:
    """Combine already-generated plot files without duplicating their rendering logic."""
    if not panels:
        raise ValueError("panels must not be empty")
    if columns < 1:
        raise ValueError("columns must be positive")
    paths = {name: Path(path) for name, path in panels.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"dashboard panel files do not exist: {', '.join(missing)}")

    rows = math.ceil(len(paths) / columns)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4.5 * rows), squeeze=False)
    for axis, (name, path) in zip(axes.flat, paths.items()):
        axis.imshow(mpimg.imread(path))
        axis.set_title(name)
        axis.axis("off")
    for axis in list(axes.flat)[len(paths):]:
        axis.axis("off")
    figure.suptitle(title, fontsize=16)
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination
