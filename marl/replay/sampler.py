"""
sampler.py

Uniform random trajectory sampling with fixed-length sequence
truncation/padding, producing batches shaped for recurrent (BPTT)
training against marl/models/vdn.py's forward_team().

Kept separate from replay_buffer.py per the Directory Structure
Reference's stated rationale: sampling strategy (uniform here) can be
swapped for prioritized or another strategy later without touching
buffer storage logic.
"""

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from marl.replay.replay_buffer import ReplayBuffer
from marl.replay.trajectory import Trajectory, Transition


@dataclass
class TimestepBatch:
    """
    One timestep's worth of data across a batch of sampled episode
    sub-sequences, shaped for direct use with VDNModel.forward_team():
    obs/next_obs are (obs_a, obs_b) tuples, each a (batch_size, C, H, W)
    float32 array; actions is an (action_a, action_b) tuple, each a
    (batch_size,) int64 array; reward/done/mask are each (batch_size,).

    mask is 1.0 where this batch element has real data at this
    timestep, 0.0 where it's padding beyond that episode's true length
    (see Sampler._sample_window). A trainer must exclude mask=0
    positions from the loss -- their obs/actions/reward/done values are
    filler (zeros/False), not meaningful.
    """
    obs: Tuple[np.ndarray, np.ndarray]
    actions: Tuple[np.ndarray, np.ndarray]
    reward: np.ndarray
    next_obs: Tuple[np.ndarray, np.ndarray]
    done: np.ndarray
    mask: np.ndarray


class Sampler:
    """
    Samples `batch_size` trajectories (with replacement, uniform) from
    a ReplayBuffer, and returns a length-`sequence_length` list of
    TimestepBatch, ready for a trainer to iterate over with BPTT,
    carrying GRU hidden state from one TimestepBatch to the next.
    """

    def __init__(
        self, buffer: ReplayBuffer, sequence_length: int, rng: Optional[random.Random] = None
    ) -> None:
        """
        Args:
            buffer: The ReplayBuffer to sample from.
            sequence_length: Fixed timesteps per sampled sub-sequence.
                Trajectories shorter than this are zero-padded (with
                mask=0 on the padded steps). Trajectories longer than
                this have a random contiguous window of this length
                selected -- not always the first `sequence_length`
                steps, so training sees a variety of episode phases
                (early exploration vs. late-episode behavior) rather
                than always the same window every time that trajectory
                is sampled.
            rng: Optional private random.Random for reproducibility,
                matching the same rationale as
                MazeGenerator._get_rng()/EpsilonGreedySelector --
                keeps this sampler's random stream independent and
                seedable rather than relying on the shared global
                `random` module.

        Raises:
            ValueError: If sequence_length is not a positive integer.
        """
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
        self.buffer = buffer
        self.sequence_length = sequence_length
        self._rng = rng if rng is not None else random.Random()

    def sample(self, batch_size: int) -> List[TimestepBatch]:
        """
        Raises:
            ValueError: If the buffer is empty, or batch_size is not
                a positive integer.
        """
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from an empty ReplayBuffer")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        windows: List[List[Optional[Transition]]] = [
            self._sample_window(self.buffer[self._rng.randrange(len(self.buffer))])
            for _ in range(batch_size)
        ]
        return [self._build_timestep_batch(windows, t) for t in range(self.sequence_length)]

    def _sample_window(self, trajectory: Trajectory) -> List[Optional[Transition]]:
        """
        Return a list of length self.sequence_length: a contiguous
        slice of `trajectory` (random start point if longer than
        sequence_length), padded with None at the end if `trajectory`
        is shorter than sequence_length.
        """
        length = len(trajectory)
        if length >= self.sequence_length:
            start = self._rng.randrange(length - self.sequence_length + 1)
            return [trajectory[start + i] for i in range(self.sequence_length)]
        return [trajectory[i] for i in range(length)] + [None] * (self.sequence_length - length)

    def _build_timestep_batch(
        self, windows: List[List[Optional[Transition]]], t: int
    ) -> TimestepBatch:
        obs_a, obs_b, next_obs_a, next_obs_b = [], [], [], []
        act_a, act_b, rewards, dones, mask = [], [], [], [], []

        sample_shape = next(
            tr.obs[0].shape for window in windows for tr in window if tr is not None
        )

        for window in windows:
            tr = window[t]
            if tr is None:
                obs_a.append(np.zeros(sample_shape, dtype=np.float32))
                obs_b.append(np.zeros(sample_shape, dtype=np.float32))
                act_a.append(0)
                act_b.append(0)
                rewards.append(0.0)
                next_obs_a.append(np.zeros(sample_shape, dtype=np.float32))
                next_obs_b.append(np.zeros(sample_shape, dtype=np.float32))
                dones.append(True)
                mask.append(0.0)
            else:
                obs_a.append(tr.obs[0])
                obs_b.append(tr.obs[1])
                act_a.append(tr.actions[0])
                act_b.append(tr.actions[1])
                rewards.append(tr.reward)
                next_obs_a.append(tr.next_obs[0])
                next_obs_b.append(tr.next_obs[1])
                dones.append(tr.done)
                mask.append(1.0)

        return TimestepBatch(
            obs=(np.stack(obs_a), np.stack(obs_b)),
            actions=(np.array(act_a, dtype=np.int64), np.array(act_b, dtype=np.int64)),
            reward=np.array(rewards, dtype=np.float32),
            next_obs=(np.stack(next_obs_a), np.stack(next_obs_b)),
            done=np.array(dones, dtype=bool),
            mask=np.array(mask, dtype=np.float32),
        )


if __name__ == "__main__":
    import numpy as np

    from marl.replay.replay_buffer import ReplayBuffer
    from marl.replay.trajectory import Trajectory, Transition

    def make_trajectory(length: int, seed: int) -> Trajectory:
        traj = Trajectory(seed=seed)
        for step in range(length):
            obs = np.full((10, 5, 5), fill_value=step, dtype=np.float32)  # step-tagged, for verification
            traj.append(Transition(
                obs=(obs, obs), actions=(step % 5, (step + 1) % 5), reward=float(step),
                next_obs=(obs, obs), done=(step == length - 1),
            ))
        return traj

    buf = ReplayBuffer(capacity=10)
    buf.add(make_trajectory(length=3, seed=1))   # shorter than sequence_length=5 -> padded
    buf.add(make_trajectory(length=8, seed=2))   # longer -> random window

    sampler = Sampler(buf, sequence_length=5, rng=__import__("random").Random(0))
    batch = sampler.sample(batch_size=4)

    print("Batch length (timesteps):", len(batch))
    print("Timestep 0 obs_a shape:", batch[0].obs[0].shape)
    print("Timestep 0 mask:", batch[0].mask)
    print("Timestep 4 mask (may include padding from the short trajectory):", batch[4].mask)

    assert len(batch) == 5
    assert batch[0].obs[0].shape == (4, 10, 5, 5)
    assert batch[0].mask.shape == (4,)
    print("OK")
