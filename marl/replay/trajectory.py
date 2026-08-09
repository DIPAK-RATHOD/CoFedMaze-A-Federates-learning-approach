"""
trajectory.py

Defines Transition (one shared-team timestep of experience) and
Trajectory (a full episode's worth of Transitions).
"""

from typing import Iterator, List, NamedTuple, Optional, Tuple

import numpy as np


class Transition(NamedTuple):
    """
    One shared-team timestep of experience.
    """
    obs: Tuple[np.ndarray, np.ndarray]        # (obs_a, obs_b) at time t
    actions: Tuple[int, int]                   # (action_a, action_b) at time t
    reward: float                              # shared team reward at time t
    next_obs: Tuple[np.ndarray, np.ndarray]    # (obs_a, obs_b) at time t+1
    done: bool                                 # terminated OR truncated at t+1


class Trajectory:
    """
    A full episode's worth of Transitions, plus metadata (seed, algorithm,
    goal_reached, timeout, evaluation flag).
    """

    def __init__(self, seed: Optional[int] = None, algorithm: Optional[str] = None) -> None:
        self.seed = seed
        self.algorithm = algorithm
        self.transitions: List[Transition] = []
        self.goal_reached: bool = False
        self.timeout: bool = False
        self.evaluation: bool = False

    def append(self, transition: Transition) -> None:
        """Append one Transition to this trajectory, in time order."""
        self.transitions.append(transition)

    def total_reward(self) -> float:
        """Sum of shared team rewards across the whole episode."""
        return sum(t.reward for t in self.transitions)

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, index: int) -> Transition:
        return self.transitions[index]

    def __iter__(self) -> Iterator[Transition]:
        return iter(self.transitions)

    def __repr__(self) -> str:
        return (
            f"Trajectory(len={len(self.transitions)}, seed={self.seed}, "
            f"algorithm={self.algorithm!r}, total_reward={self.total_reward():.3f}, "
            f"goal_reached={self.goal_reached}, timeout={self.timeout})"
        )


if __name__ == "__main__":
    dummy_obs = np.zeros((10, 5, 5), dtype=np.float32)
    traj = Trajectory(seed=1, algorithm="recursive_backtracking")
    for step in range(3):
        traj.append(Transition(
            obs=(dummy_obs, dummy_obs),
            actions=(0, 1),
            reward=-0.01,
            next_obs=(dummy_obs, dummy_obs),
            done=(step == 2),
        ))
    traj.goal_reached = True
    assert len(traj) == 3
    assert traj.goal_reached is True
    print("OK")
