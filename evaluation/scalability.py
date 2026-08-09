"""Node-count and coalition-size experiment bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class ScalePoint:
    """One requested configuration for a scalability experiment."""

    node_count: int
    max_coalition_size: int

    def __post_init__(self) -> None:
        if self.node_count < 1:
            raise ValueError("node_count must be positive")
        if not 1 <= self.max_coalition_size <= self.node_count:
            raise ValueError("max_coalition_size must be between 1 and node_count")


@dataclass(frozen=True)
class ScalabilityResult:
    point: ScalePoint
    result: object


def default_scale_points(node_counts: Iterable[int] = (2, 3, 5), coalition_sizes: Iterable[int] = (1, 2, 3)) -> list[ScalePoint]:
    """Build valid Cartesian scale points; N=5 keeps the workplan baseline."""
    return [ScalePoint(n, c) for n in node_counts for c in coalition_sizes if c <= n]


def run_scalability(experiment: Callable[[ScalePoint], T], points: Iterable[ScalePoint] | None = None) -> list[ScalabilityResult]:
    """Execute caller-owned simulations for each scale point in stable order."""
    selected = list(default_scale_points() if points is None else points)
    return [ScalabilityResult(point, experiment(point)) for point in selected]
