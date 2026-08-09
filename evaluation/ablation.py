"""Knowledge-Score criterion ablation definitions and execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, TypeVar

from knowledge_graph.knowledge_score import DEFAULT_WEIGHTS


CRITERIA = ("tb", "ts", "ms", "l", "e")
T = TypeVar("T")


@dataclass(frozen=True)
class AblationVariant:
    """A named KS configuration, with one criterion optionally omitted."""

    omitted_criterion: str | None
    weights: Dict[str, float]

    @property
    def name(self) -> str:
        return "full_ks" if self.omitted_criterion is None else f"without_{self.omitted_criterion}"


def ablation_variants(base_weights: Dict[str, float] | None = None) -> list[AblationVariant]:
    """Return full KS plus five leave-one-criterion-out, renormalised variants.

    Renormalisation retains the original score scale, making thresholds and
    outcomes comparable between the full and ablated experiments.
    """
    weights = dict(DEFAULT_WEIGHTS if base_weights is None else base_weights)
    if set(weights) != set(CRITERIA) or any(value < 0 for value in weights.values()):
        raise ValueError(f"weights must contain non-negative values for exactly {CRITERIA}")
    variants = [AblationVariant(None, weights)]
    for criterion in CRITERIA:
        remaining_total = sum(value for key, value in weights.items() if key != criterion)
        if remaining_total <= 0:
            raise ValueError(f"cannot ablate {criterion}: remaining weights sum to zero")
        variants.append(AblationVariant(
            criterion,
            {key: (0.0 if key == criterion else value / remaining_total) for key, value in weights.items()},
        ))
    return variants


def run_ablation(experiment: Callable[[AblationVariant], T], variants: Iterable[AblationVariant] | None = None) -> Dict[str, T]:
    """Run caller-owned training/evaluation once for every supplied variant."""
    selected = list(ablation_variants() if variants is None else variants)
    return {variant.name: experiment(variant) for variant in selected}
