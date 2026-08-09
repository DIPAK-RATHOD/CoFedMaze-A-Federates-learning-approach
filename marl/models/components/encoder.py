"""
encoder.py

The shared maze encoder: a small CNN that maps an egocentric,
multi-channel local-view observation to a fixed-size embedding vector.

This is the SHARED component (per the Shared vs. Private Model
Components glossary term) — the one actually quantized to 8-bit and
exchanged between federated nodes (see the KG/Coalition Implementation
Strategy doc, Section 3 Step 1). Nothing task-specific lives here;
task-specific behavior belongs in heads.py (private, never transmitted).

Input contract (provisional — see module-level note below):
    A tensor of shape (batch, in_channels, window, window):
        - `window` x `window` is a fixed-size EGOCENTRIC local view
          (the agent's logical cell at the center), not the full maze.
          Fixed size keeps the observation — and therefore this
          encoder's input shape — independent of maze size or task
          variant, which matters since N1-N5 run different mazes.
        - Channels are planes over that window. The first 4 are
          movement-passability planes (can-move up/down/left/right from
          each cell in the window, derived from wall-slot state — NOT a
          redundant "is this cell a wall" plane, since logical cells in
          this project's double-resolution Grid are never walls and
          such a plane would be constant/uninformative). Remaining
          channels are object-presence planes (goal, other agent, key,
          door, checkpoint, obstacle, etc.) and are configurable via
          `in_channels`, since not every node's task variant uses every
          object type.

    NOTE: env/core/observations.py (the module that will actually
    construct this tensor from a Maze) has not been built yet — it is
    marked "NEXT" in the project's directory structure reference. This
    encoder is built against the input contract above and tested with
    synthetic tensors; wiring it to real observations is a follow-up
    step once observations.py exists.
"""

from typing import Tuple

import torch
from torch import nn


class SharedEncoder(nn.Module):
    """
    Small CNN encoder: Conv -> Conv -> flatten -> FC -> embedding.

    Kept intentionally shallow (two conv layers, no pooling) since this
    runs on Raspberry Pi CPU with no accelerator, and — per the
    edge-efficiency strategy doc — is quantized to 8-bit before every
    transmission, so keeping the parameter count small also keeps the
    per-round communication payload small.
    """

    def __init__(
        self,
        in_channels: int,
        window_size: int,
        embedding_dim: int = 128,
        conv_channels: Tuple[int, int] = (16, 32),
    ) -> None:
        """
        Args:
            in_channels: Number of input planes (movement-passability +
                object-presence channels — see module docstring).
            window_size: Height/width of the square egocentric window.
                Must be >= 3 so two stride-1, padding-1, 3x3 conv layers
                leave a non-empty spatial map to flatten.
            embedding_dim: Size of the output embedding vector — this is
                what feeds into the GRU memory layer (gru.py).
            conv_channels: Output channel counts for the two conv
                layers, in order.

        Raises:
            ValueError: If in_channels, window_size, or embedding_dim is
                not a positive integer, or window_size < 3.
        """
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if window_size < 3:
            raise ValueError(
                f"window_size must be >= 3 (need room for two 3x3 conv layers), "
                f"got {window_size}"
            )
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")

        self.in_channels = in_channels
        self.window_size = window_size
        self.embedding_dim = embedding_dim

        c1, c2 = conv_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        flattened_size = c2 * window_size * window_size
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observation: Tensor of shape
                (batch, in_channels, window_size, window_size).

        Returns:
            Embedding tensor of shape (batch, embedding_dim).

        Raises:
            ValueError: If `observation`'s shape doesn't match
                (*, in_channels, window_size, window_size).
        """
        if observation.dim() != 4 or observation.shape[1:] != (
            self.in_channels,
            self.window_size,
            self.window_size,
        ):
            raise ValueError(
                f"Expected observation shape (batch, {self.in_channels}, "
                f"{self.window_size}, {self.window_size}), got {tuple(observation.shape)}"
            )
        features = self.conv(observation)
        flattened = features.flatten(start_dim=1)
        return self.fc(flattened)


if __name__ == "__main__":
    # Minimal smoke test / usage example with a synthetic observation.
    encoder = SharedEncoder(in_channels=8, window_size=5, embedding_dim=128)
    dummy_obs = torch.rand(4, 8, 5, 5)  # batch of 4
    embedding = encoder(dummy_obs)
    print("Embedding shape:", embedding.shape)
    assert embedding.shape == (4, 128)
    print("OK")
