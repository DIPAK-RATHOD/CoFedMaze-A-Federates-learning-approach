"""
scheduler.py

Learning-rate scheduling, kept distinct from optimizer.py since
schedule and optimizer algorithm are independent design choices.
"""

from typing import Optional

import torch


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    schedule_type: str = "none",
    **kwargs,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """
    Args:
        optimizer: The optimizer to attach a schedule to.
        schedule_type: One of "none", "step", "exponential".
        **kwargs: Forwarded to the underlying scheduler constructor
            (e.g. step_size/gamma for "step"; gamma for "exponential").

    Returns:
        None if schedule_type == "none" (constant learning rate,
        trainer.py should simply not call .step() on anything), else
        the constructed scheduler.

    Raises:
        ValueError: If schedule_type is not recognized.
    """
    if schedule_type == "none":
        return None
    if schedule_type == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    if schedule_type == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    raise ValueError(
        f"Unknown schedule_type {schedule_type!r}; expected 'none', 'step', or 'exponential'"
    )


if __name__ == "__main__":
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    assert build_scheduler(opt, "none") is None
    print("none -> None: OK")

    sched = build_scheduler(opt, "step", step_size=10, gamma=0.5)
    print("step scheduler:", sched)
    assert isinstance(sched, torch.optim.lr_scheduler.StepLR)

    try:
        build_scheduler(opt, "bogus")
        print("FAIL: should have raised")
    except ValueError as e:
        print("OK:", e)
