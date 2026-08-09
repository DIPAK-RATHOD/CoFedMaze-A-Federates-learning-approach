"""
messages.py

Message schema for exchanging shared-component updates between nodes:
compact metadata (node ID, round, update norm, validation reward,
timestamp) plus the actual payload -- matching the KG/Coalition
Implementation Strategy doc's Step 1 spec exactly.

update_norm is computed here (not left to the caller) so every message
producer computes it identically -- a cheap, transport-agnostic signal
a receiver can inspect (e.g. flag an anomalously large update) without
touching the full payload.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict

import torch

SharedState = Dict[str, Dict[str, torch.Tensor]]


def compute_update_norm(shared_state: SharedState) -> float:
    """
    L2 norm of every tensor in both the "encoder" and "memory"
    sub-dicts, flattened and concatenated -- a single scalar summarizing
    the overall magnitude of a shared-component update.
    """
    pieces = []
    for component in ("encoder", "memory"):
        for tensor in shared_state[component].values():
            pieces.append(tensor.flatten().float())
    if not pieces:
        return 0.0
    return torch.cat(pieces).norm().item()


@dataclass
class UpdateMessage:
    """
    One node's shared-component update, ready to serialize and send to
    a peer.

    Attributes:
        node_id: Sending node's id (e.g. "N1").
        round: Training round/episode counter this update corresponds
            to -- lets a receiver detect and discard stale updates.
        update_norm: See compute_update_norm(). Computed automatically
            in __post_init__ if not supplied explicitly.
        validation_reward: The sender's own local validation-set
            average reward at send time -- this is what the receiver's
            transfer-benefit test (federation/validation/transfer_validation.py)
            compares its own local R_old against.
        timestamp: Unix time the message was constructed (time.time()),
            for staleness checks and logging.
        payload: The actual shared state (encoder + memory) being sent.
            Whether this is raw or quantized (see compression.py, not
            yet built) is opaque to this schema -- messages.py doesn't
            care which form payload is in, only that it's present.
    """
    node_id: str
    round: int
    validation_reward: float
    payload: SharedState
    update_norm: float = field(default=None)  # computed in __post_init__ if not given
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.update_norm is None:
            self.update_norm = compute_update_norm(self.payload)


if __name__ == "__main__":
    from env.core.actions import NUM_ACTIONS
    from env.core.observations import NUM_CHANNELS
    from federation.validation.transfer_validation import extract_shared_state
    from marl.models.vdn import VDNModel

    model = VDNModel(in_channels=NUM_CHANNELS, window_size=5, num_actions=NUM_ACTIONS, num_agents=2)
    shared_state = extract_shared_state(model)

    msg = UpdateMessage(node_id="N1", round=42, validation_reward=3.5, payload=shared_state)
    print(msg.node_id, msg.round, msg.validation_reward, "update_norm=", msg.update_norm)
    assert msg.update_norm > 0
    assert msg.timestamp > 0
    print("OK")
