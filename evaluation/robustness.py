"""Repeatable noisy-validation and dropped-message robustness experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class FaultProfile:
    """Fault rates applied by a simulation runner, each constrained to [0, 1]."""

    validation_noise_std: float = 0.0
    message_drop_rate: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.validation_noise_std < 0:
            raise ValueError("validation_noise_std must be non-negative")
        if not 0.0 <= self.message_drop_rate <= 1.0:
            raise ValueError("message_drop_rate must be in [0, 1]")


class FaultInjector:
    """Seeded helpers that make a faulted run reproducible and auditable."""

    def __init__(self, profile: FaultProfile) -> None:
        self.profile = profile
        self._rng = random.Random(profile.seed)
        self.sent = 0
        self.dropped = 0

    def perturb_validation_reward(self, reward: float) -> float:
        return reward + self._rng.gauss(0.0, self.profile.validation_noise_std)

    def should_drop_message(self) -> bool:
        self.sent += 1
        dropped = self._rng.random() < self.profile.message_drop_rate
        self.dropped += int(dropped)
        return dropped


def default_fault_profiles() -> list[FaultProfile]:
    return [FaultProfile(), FaultProfile(validation_noise_std=0.1), FaultProfile(message_drop_rate=0.1), FaultProfile(validation_noise_std=0.1, message_drop_rate=0.1)]


def run_robustness(experiment: Callable[[FaultInjector], T], profiles: Iterable[FaultProfile] | None = None) -> dict[FaultProfile, T]:
    """Run a simulation for each fault profile; the runner applies faults itself."""
    return {profile: experiment(FaultInjector(profile)) for profile in (default_fault_profiles() if profiles is None else profiles)}
