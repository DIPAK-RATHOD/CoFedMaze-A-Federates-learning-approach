"""
heads.py

Private, task-specific output heads: map a GRU hidden state (plus an
optional agent-identity signal) to per-action Q-values for one agent.

Explicitly PRIVATE (see encoder.py's module docstring for the
shared/private distinction) — never quantized or transmitted to peers.
Isolating this in its own file makes it unambiguous what
federation/communication/compression.py (not yet built) is and is not
allowed to touch: encoder.py and gru.py, never this file.
"""

from typing import Optional

import torch
from torch import nn


class QHead(nn.Module):
    """
    Two-layer MLP head producing per-action Q-values from a hidden
    state.

    Optionally accepts a one-hot agent-identity vector, concatenated
    onto the hidden state before the first linear layer. This is what
    lets a single parameter-shared encoder+GRU (see vdn.py) still
    produce agent-specific behavior: the encoder/GRU treat both agents
    identically, but this head can specialize per agent using the
    identity signal alone, rather than requiring two separate networks.
    """

    def __init__(self, hidden_dim: int, num_actions: int, agent_id_dim: int = 0) -> None:
        """
        Args:
            hidden_dim: Size of the incoming GRU hidden state.
            num_actions: Size of the discrete action space.
            agent_id_dim: Size of the one-hot agent-identity vector
                concatenated onto the hidden state. 0 disables this
                (both agents then produce identical Q-values for
                identical hidden states — usually not what you want
                with parameter sharing, but left configurable).

        Raises:
            ValueError: If hidden_dim, num_actions is not positive, or
                agent_id_dim is negative.
        """
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_actions <= 0:
            raise ValueError(f"num_actions must be positive, got {num_actions}")
        if agent_id_dim < 0:
            raise ValueError(f"agent_id_dim must be non-negative, got {agent_id_dim}")

        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.agent_id_dim = agent_id_dim

        input_dim = hidden_dim + agent_id_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(
        self, hidden: torch.Tensor, agent_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden: (batch, hidden_dim) GRU hidden state.
            agent_id: (batch, agent_id_dim) one-hot agent-identity
                tensor. Required if agent_id_dim > 0 at construction,
                must be omitted (None) if agent_id_dim == 0.

        Returns:
            Q-values of shape (batch, num_actions).

        Raises:
            ValueError: If agent_id's presence/shape doesn't match
                agent_id_dim, or hidden's shape is wrong.
        """
        if hidden.dim() != 2 or hidden.shape[1] != self.hidden_dim:
            raise ValueError(
                f"Expected hidden shape (batch, {self.hidden_dim}), got {tuple(hidden.shape)}"
            )

        if self.agent_id_dim == 0:
            if agent_id is not None:
                raise ValueError("agent_id_dim=0 was configured, but an agent_id tensor was passed")
            combined = hidden
        else:
            if agent_id is None:
                raise ValueError(f"agent_id_dim={self.agent_id_dim} requires an agent_id tensor")
            if agent_id.shape != (hidden.shape[0], self.agent_id_dim):
                raise ValueError(
                    f"Expected agent_id shape ({hidden.shape[0]}, {self.agent_id_dim}), "
                    f"got {tuple(agent_id.shape)}"
                )
            combined = torch.cat([hidden, agent_id], dim=1)

        return self.mlp(combined)


if __name__ == "__main__":
    head = QHead(hidden_dim=128, num_actions=5, agent_id_dim=2)
    dummy_hidden = torch.rand(4, 128)
    dummy_agent_id = torch.eye(2)[torch.zeros(4, dtype=torch.long)]  # agent 0, one-hot
    q_values = head(dummy_hidden, dummy_agent_id)
    print("Q-values shape:", q_values.shape)
    assert q_values.shape == (4, 5)
    print("OK")
