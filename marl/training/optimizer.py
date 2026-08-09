"""
optimizer.py

Optimizer construction/wrapping from config, isolated so optimizer
choice/hyperparameters are swappable without editing trainer.py.
"""

from typing import Iterable

import torch


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """
    Args:
        parameters: Typically model.parameters() -- pass the ONLINE
            model's parameters, never the target model's (the target
            network is never trained directly, only hard-synced -- see
            marl/losses/vdn_loss.py).
        learning_rate: Adam learning rate.
        weight_decay: L2 regularization coefficient (0.0 disables it).

    Returns:
        A configured torch.optim.Adam instance.

    Raises:
        ValueError: If learning_rate or weight_decay is negative.
    """
    if learning_rate < 0:
        raise ValueError(f"learning_rate must be non-negative, got {learning_rate}")
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
    return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)


if __name__ == "__main__":
    model = torch.nn.Linear(4, 2)
    opt = build_optimizer(model.parameters(), learning_rate=1e-3)
    print(opt)
    assert isinstance(opt, torch.optim.Adam)
    print("OK")
