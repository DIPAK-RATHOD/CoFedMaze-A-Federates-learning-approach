"""
vdn_loss.py

The TD-error loss computed over the summed team Q-value (Q_tot), using
a target network (hard-updated, not soft/Polyak) and Double Q-learning
-- the three design decisions made in conversation, matching PyMARL's
VDN convention (the reference implementation named in the Implementation
Requirements doc).

Isolated from trainer.py so the loss function itself can be tested and
verified independently, per the Directory Structure Reference's stated
rationale for this file's existence. trainer.py (not yet built) owns
the target network's update SCHEDULE (how often to call
hard_update_target_network below); this file only knows how to compute
the loss and how to perform one hard update when asked.

============================================================================
Recurrent TD-target correctness: why this file processes obs_t AND
next_obs_t through BOTH networks every step, not just next_obs_t
============================================================================
Computing Q(s_t, ·) with a GRU requires the hidden state accumulated
from every observation up to and including obs_t: h_t = GRU(encode(obs_t), h_{t-1}).
Computing the bootstrap Q(s_{t+1}, ·) needed for the TD target requires
h_{t+1} = GRU(encode(obs_{t+1}), h_t) -- i.e. ONE MORE recurrent step
starting from h_t, not a freshly zero-initialized hidden state fed
next_obs_t in isolation. Since next_obs_t == obs_{t+1} for a contiguous
sampled window (see the assumption note below), the correct sequence is
a SINGLE rollout per network across [obs_0, ..., obs_{T-1}, next_obs_{T-1}],
not two independent rollouts.

Crucially, the ONLINE and TARGET networks have different weights, so
they produce DIFFERENT hidden-state trajectories from the same inputs.
Each network's own h_t (computed via its own weights) must be used as
the base for that same network's evaluation of next_obs_t -- using
online_hidden as the base for a target-network evaluation (or vice
versa) silently produces a wrong bootstrap target with no error raised.
This was caught and fixed while writing this file, before any testing
-- flagged here so it's not re-introduced by a future edit.

Assumption this file depends on (owned by sampler.py, not re-verified
here): a sampled window never spans more than one episode -- padding
(mask=0) only ever appears after a trajectory's true end, never in the
middle of real data. If that assumption is ever violated, hidden state
would incorrectly carry across an episode boundary.
============================================================================
"""

from typing import List, Tuple

import numpy as np
import torch

from marl.models.vdn import VDNModel
from marl.replay.sampler import TimestepBatch


def compute_vdn_loss(
    online_model: VDNModel,
    target_model: VDNModel,
    batch: List[TimestepBatch],
    gamma: float = 0.99,
) -> torch.Tensor:
    """
    Args:
        online_model: The model being trained (gradients flow into this
            one only).
        target_model: A separate VDNModel instance with the same
            architecture, used ONLY to compute the bootstrap target
            (wrapped in torch.no_grad() throughout -- never trained
            directly). Kept in sync with online_model via
            hard_update_target_network(), called by trainer.py on
            whatever schedule it decides, not by this function.
        batch: A length-sequence_length list of TimestepBatch, as
            returned by marl.replay.sampler.Sampler.sample() -- see
            that module for the exact shape contract.
        gamma: Discount factor.

    Returns:
        A scalar loss tensor (mean squared TD error over every
        mask=1 timestep in the batch), ready for loss.backward().

    Raises:
        ValueError: If `batch` is empty.
    """
    if not batch:
        raise ValueError("batch must contain at least one TimestepBatch")

    sequence_length = len(batch)
    batch_size = batch[0].mask.shape[0]
    device = next(online_model.parameters()).device
    num_agents = online_model.num_agents

    online_hidden = online_model.init_hidden(batch_size, device)
    target_hidden = target_model.init_hidden(batch_size, device)

    total_loss = torch.zeros((), device=device)
    total_mask = torch.zeros((), device=device)

    for t in range(sequence_length):
        tb = batch[t]
        mask = torch.from_numpy(tb.mask).to(device)
        reward = torch.from_numpy(tb.reward).to(device)
        done = torch.from_numpy(tb.done.astype(np.float32)).to(device)
        actions = [torch.from_numpy(tb.actions[i]).to(device) for i in range(num_agents)]
        obs_t = [torch.from_numpy(tb.obs[i]).to(device) for i in range(num_agents)]
        next_obs_t = [torch.from_numpy(tb.next_obs[i]).to(device) for i in range(num_agents)]

        # Advance the ONLINE network through obs_t -- this is the real,
        # gradient-tracked forward pass whose Q_tot trains against the
        # TD target below. online_hidden becomes each agent's h_t.
        _, online_hidden, q_tot_online = online_model.forward_team(obs_t, online_hidden, actions)

        with torch.no_grad():
            # Advance the TARGET network through the SAME obs_t, using
            # its OWN (different-weight) hidden state -- see module
            # docstring for why this must not reuse online_hidden.
            target_hidden = [
                target_model.forward_agent(obs_t[i], target_hidden[i], i)[1]
                for i in range(num_agents)
            ]

            # Double Q-learning: select each agent's greedy next action
            # using the ONLINE network evaluated at next_obs_t (from
            # online's own just-advanced h_t)...
            next_q_online = [
                online_model.forward_agent(next_obs_t[i], online_hidden[i], i)[0]
                for i in range(num_agents)
            ]
            greedy_actions = [torch.argmax(q, dim=1) for q in next_q_online]

            # ...but EVALUATE that action's value using the TARGET
            # network (from target's own just-advanced h_t) -- this
            # selection/evaluation split is what "Double Q" means, and
            # is what reduces the overestimation bias of vanilla
            # max-Q bootstrapping.
            next_q_target = [
                target_model.forward_agent(next_obs_t[i], target_hidden[i], i)[0]
                for i in range(num_agents)
            ]
            chosen_target_q = [
                next_q_target[i].gather(1, greedy_actions[i].unsqueeze(1)).squeeze(1)
                for i in range(num_agents)
            ]
            # VDNMixer has no learnable parameters -- either model's
            # instance computes the identical sum; online_model's is
            # used only because it's guaranteed to exist.
            q_tot_target_next = online_model.mixer(chosen_target_q)
            td_target = reward + gamma * (1.0 - done) * q_tot_target_next

        step_loss = (q_tot_online - td_target) ** 2
        total_loss = total_loss + (step_loss * mask).sum()
        total_mask = total_mask + mask.sum()

    return total_loss / total_mask.clamp(min=1.0)


def hard_update_target_network(online_model: VDNModel, target_model: VDNModel) -> None:
    """
    Copy online_model's weights into target_model exactly (hard
    update). trainer.py (not yet built) is responsible for deciding
    HOW OFTEN to call this -- e.g. every N episodes, per the hard-update
    convention decided for this project. Calling this every step would
    make target_model identical to online_model at all times, defeating
    the entire purpose of having a separate target network.
    """
    target_model.load_state_dict(online_model.state_dict())


if __name__ == "__main__":
    import random

    from marl.replay.replay_buffer import ReplayBuffer
    from marl.replay.sampler import Sampler
    from marl.replay.trajectory import Trajectory, Transition

    def make_trajectory(length: int, seed: int) -> Trajectory:
        traj = Trajectory(seed=seed)
        obs_seq = [np.random.rand(8, 5, 5).astype(np.float32) for _ in range(length + 1)]
        for step in range(length):
            traj.append(Transition(
                obs=(obs_seq[step], obs_seq[step]),
                actions=(step % 5, (step + 1) % 5),
                reward=-0.01,
                next_obs=(obs_seq[step + 1], obs_seq[step + 1]),
                done=(step == length - 1),
            ))
        return traj

    buf = ReplayBuffer(capacity=20)
    for i in range(10):
        buf.add(make_trajectory(length=random.randint(3, 12), seed=i))
    sampler = Sampler(buf, sequence_length=6, rng=random.Random(0))

    online = VDNModel(in_channels=8, window_size=5, num_actions=5, num_agents=2)
    target = VDNModel(in_channels=8, window_size=5, num_actions=5, num_agents=2)
    hard_update_target_network(online, target)

    batch = sampler.sample(batch_size=16)
    loss = compute_vdn_loss(online, target, batch, gamma=0.99)
    print("Loss:", loss.item())

    loss.backward()
    online_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in online.parameters())
    target_has_grad = any(p.grad is not None for p in target.parameters())
    print("Online network received gradients:", online_has_grad)
    print("Target network received NO gradients:", not target_has_grad)
    assert online_has_grad and not target_has_grad
    print("OK")
