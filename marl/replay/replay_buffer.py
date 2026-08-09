"""
replay_buffer.py

Bounded, rotating store of complete Trajectories (not individual
transitions -- see trajectory.py's docstring for why). Backs
data/nodeN/replay_buffer/ per the Directory Structure Reference.

Deliberately does NOT implement any sampling strategy itself --
sampler.py consumes a ReplayBuffer instance, so sampling strategy
(uniform here; prioritized or otherwise later) can change without
touching storage logic, matching the Directory Structure Reference's
stated rationale for keeping these as separate files.
"""

from collections import deque
from typing import Deque, Iterator

from marl.replay.trajectory import Trajectory


class ReplayBuffer:
    """
    Fixed-capacity, rotating buffer of Trajectory objects.

    Rotation is implemented via collections.deque(maxlen=capacity):
    once full, adding a new trajectory automatically evicts the oldest
    one in O(1), with no manual bookkeeping -- this is the "rotating
    chunk" design the Directory Structure Reference calls for (rather
    than one unbounded, monolithic store) to bound RAM on constrained
    Pi hardware.
    """

    def __init__(self, capacity: int) -> None:
        """
        Args:
            capacity: Maximum number of trajectories retained.

        Raises:
            ValueError: If capacity is not a positive integer.
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self._trajectories: Deque[Trajectory] = deque(maxlen=capacity)

    def add(self, trajectory: Trajectory) -> None:
        """
        Add a completed Trajectory. If the buffer is already at
        capacity, the oldest trajectory is silently evicted.

        Raises:
            ValueError: If `trajectory` has zero transitions -- an
                empty trajectory has nothing to train on and almost
                certainly indicates a bug upstream (e.g. an episode
                that ended before a single step was recorded), so this
                fails loudly rather than silently storing a useless
                entry.
        """
        if len(trajectory) == 0:
            raise ValueError("Cannot add an empty Trajectory (zero transitions) to the buffer")
        self._trajectories.append(trajectory)

    def is_ready(self, min_size: int) -> bool:
        """Convenience check: has the buffer accumulated at least `min_size` trajectories?"""
        return len(self._trajectories) >= min_size

    def __len__(self) -> int:
        return len(self._trajectories)

    def __iter__(self) -> Iterator[Trajectory]:
        return iter(self._trajectories)

    def __getitem__(self, index: int) -> Trajectory:
        return self._trajectories[index]

    def __repr__(self) -> str:
        return f"ReplayBuffer(size={len(self)}/{self.capacity})"


if __name__ == "__main__":
    from marl.replay.trajectory import Transition
    import numpy as np

    dummy_obs = np.zeros((10, 5, 5), dtype=np.float32)

    def make_trajectory(length: int, seed: int) -> Trajectory:
        traj = Trajectory(seed=seed, algorithm="recursive_backtracking")
        for step in range(length):
            traj.append(Transition(
                obs=(dummy_obs, dummy_obs), actions=(0, 1), reward=-0.01,
                next_obs=(dummy_obs, dummy_obs), done=(step == length - 1),
            ))
        return traj

    buf = ReplayBuffer(capacity=3)
    for i in range(5):
        buf.add(make_trajectory(length=4, seed=i))
        print(buf)

    # Only the last 3 (capacity) should survive -- seeds 2, 3, 4.
    surviving_seeds = [t.seed for t in buf]
    print("Surviving seeds:", surviving_seeds)
    assert surviving_seeds == [2, 3, 4]
    assert len(buf) == 3
    assert buf.is_ready(3) and not buf.is_ready(4)

    try:
        buf.add(Trajectory())  # empty
        print("FAIL: should have raised")
    except ValueError as e:
        print("OK:", e)
