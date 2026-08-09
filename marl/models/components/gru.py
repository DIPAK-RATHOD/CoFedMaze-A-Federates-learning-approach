"""
gru.py

The shared GRU recurrent-memory component, supporting the
partial-observability task variant by letting an agent integrate
information across steps rather than acting on a single frame alone.

Also a SHARED component (see encoder.py's module docstring for the
shared/private distinction) — the recurrent memory reasons over the
embeddings the shared encoder produces, so it travels with the encoder
in federated exchange.

Implemented with nn.GRUCell (single-step) rather than nn.GRU
(full-sequence), because the primary consumer is an online agent
stepping through an episode one observation at a time (see
marl/agents/vdn_agent.py), where the agent must carry hidden state
forward between individual environment steps rather than process a
whole episode at once. A sequence-batched training loop (marl/training/
— not yet built) can still use this same GRUCell in a manual per-step
loop over a batch of trajectories; it does not require nn.GRU.
"""

from typing import Optional

import torch
from torch import nn


class RecurrentMemory(nn.Module):
    """
    Single-step GRU memory: (embedding, previous_hidden) -> new_hidden.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        """
        Args:
            input_dim: Size of the incoming embedding vector (must match
                SharedEncoder's embedding_dim).
            hidden_dim: Size of the hidden state carried between steps.

        Raises:
            ValueError: If input_dim or hidden_dim is not positive.
        """
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell = nn.GRUCell(input_dim, hidden_dim)

    def init_hidden(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Create a zeroed hidden state for the start of a new episode.

        Args:
            batch_size: Number of parallel sequences (usually 1 for a
                single agent stepping through one live episode; larger
                during batched training).
            device: Device to allocate the tensor on. Defaults to this
                module's own parameter device.

        Returns:
            A (batch_size, hidden_dim) zero tensor.
        """
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, embedding: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding: (batch, input_dim) tensor from SharedEncoder.
            hidden: (batch, hidden_dim) previous hidden state — from
                init_hidden() at episode start, or the previous call's
                return value on subsequent steps.

        Returns:
            The new (batch, hidden_dim) hidden state.

        Raises:
            ValueError: If embedding/hidden shapes don't match this
                module's configured dimensions, or their batch sizes
                disagree.
        """
        if embedding.dim() != 2 or embedding.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected embedding shape (batch, {self.input_dim}), got {tuple(embedding.shape)}"
            )
        if hidden.dim() != 2 or hidden.shape[1] != self.hidden_dim:
            raise ValueError(
                f"Expected hidden shape (batch, {self.hidden_dim}), got {tuple(hidden.shape)}"
            )
        if embedding.shape[0] != hidden.shape[0]:
            raise ValueError(
                f"Batch size mismatch between embedding ({embedding.shape[0]}) "
                f"and hidden state ({hidden.shape[0]})"
            )
        return self.cell(embedding, hidden)


if __name__ == "__main__":
    memory = RecurrentMemory(input_dim=128, hidden_dim=128)
    hidden = memory.init_hidden(batch_size=4)
    print("Initial hidden shape:", hidden.shape)

    dummy_embedding = torch.rand(4, 128)
    hidden = memory(dummy_embedding, hidden)
    print("Hidden shape after one step:", hidden.shape)
    assert hidden.shape == (4, 128)
    print("OK")
