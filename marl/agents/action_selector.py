"""
action_selector.py

Epsilon-greedy action selection over a Q-value vector. Kept separate
from policy.py so exploration strategy can be swapped (e.g. for
Boltzmann/softmax exploration later) without touching the model or
policy definitions — matching the Directory Structure Reference's
stated rationale for this file's existence.
"""

import random
from typing import Optional

import torch


class EpsilonGreedySelector:
    """
    Selects the greedy (argmax) action with probability (1 - epsilon),
    and a uniformly random action with probability epsilon.
    """

    def __init__(self, epsilon: float = 1.0, rng: Optional[random.Random] = None) -> None:
        """
        Args:
            epsilon: Initial exploration rate, in [0, 1]. 1.0 = fully
                random, 0.0 = fully greedy.
            rng: Optional private random.Random instance for
                reproducibility, matching the same rationale as
                MazeGenerator._get_rng() (env/generator/generator_factory.py)
                — a private instance keeps this agent's exploration
                stream independent of any other node's or agent's.
                Defaults to a fresh, unseeded Random() if not given.

        Raises:
            ValueError: If epsilon is not in [0, 1].
        """
        self._validate_epsilon(epsilon)
        self.epsilon: float = epsilon
        self._rng: random.Random = rng if rng is not None else random.Random()

    @staticmethod
    def _validate_epsilon(epsilon: float) -> None:
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")

    def set_epsilon(self, epsilon: float) -> None:
        """Update epsilon (e.g. from an external decay schedule)."""
        self._validate_epsilon(epsilon)
        self.epsilon = epsilon

    def select(self, q_values: torch.Tensor) -> int:
        """
        Args:
            q_values: 1-D tensor of shape (num_actions,) — Q-values for
                a single agent at a single timestep (batch already
                stripped by the caller; see VDNAgent.act()).

        Returns:
            The selected action index.

        Raises:
            ValueError: If q_values is not 1-dimensional.
        """
        if q_values.dim() != 1:
            raise ValueError(
                f"Expected q_values to be 1-D (num_actions,), got shape {tuple(q_values.shape)}"
            )

        num_actions = q_values.shape[0]
        if self._rng.random() < self.epsilon:
            return self._rng.randrange(num_actions)
        return int(torch.argmax(q_values).item())


if __name__ == "__main__":
    selector = EpsilonGreedySelector(epsilon=0.0, rng=random.Random(0))
    q = torch.tensor([0.1, 0.9, 0.3, -0.2, 0.05])
    action = selector.select(q)
    print("Greedy action (epsilon=0):", action)
    assert action == 1

    selector.set_epsilon(1.0)
    actions = {selector.select(q) for _ in range(50)}
    print("Random actions seen (epsilon=1):", sorted(actions))
    assert len(actions) > 1  # should see variety, not always the greedy action
    print("OK")
