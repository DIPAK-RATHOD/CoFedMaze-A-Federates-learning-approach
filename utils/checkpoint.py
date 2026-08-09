"""
checkpoint.py

Sole owner of checkpoint save/load and the current.pt/previous.pt
two-slot rollback rotation, per the Directory Structure Reference.

Supports full state persistence: model weights, target network weights,
optimizer state, episode count, total env steps, and custom metadata.
"""

import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

PathLike = Union[str, Path]

CURRENT_FILENAME = "current.pt"
PREVIOUS_FILENAME = "previous.pt"

_VALID_SLOTS = ("current", "previous")


def save_checkpoint(
    directory: PathLike,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    episode_count: int = 0,
    total_env_steps: int = 0,
    target_model: Optional[torch.nn.Module] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a new checkpoint, rotating the existing current.pt (if any)
    into previous.pt first.

    Args:
        directory: state/nodeN/checkpoints/-style directory. Created if
            it doesn't exist.
        model: The online model to save (model.state_dict() is stored).
        optimizer: If given, optimizer.state_dict() is also stored.
        episode_count: Training completed episode count.
        total_env_steps: Total environment steps taken across all episodes.
        target_model: Optional target model (target_model.state_dict() is stored).
        metadata: Free-form extra info (e.g. maze seed, node_id, restart_count).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    current_path = directory / CURRENT_FILENAME
    previous_path = directory / PREVIOUS_FILENAME

    if current_path.exists():
        shutil.copy2(current_path, previous_path)

    payload = {
        "model_state": model.state_dict(),
        "target_model_state": target_model.state_dict() if target_model is not None else None,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "episode_count": episode_count,
        "total_env_steps": total_env_steps,
        "metadata": metadata or {},
    }

    tmp_path = directory / f".{CURRENT_FILENAME}.tmp"
    torch.save(payload, tmp_path)
    tmp_path.replace(current_path)


def load_checkpoint(directory: PathLike, slot: str = "current") -> Dict[str, Any]:
    """
    Load a full checkpoint payload (model_state, target_model_state,
    optimizer_state, episode_count, total_env_steps, metadata) from slot.

    Args:
        directory: state/nodeN/checkpoints/-style directory.
        slot: "current" or "previous".

    Raises:
        ValueError: If slot is not "current" or "previous".
        FileNotFoundError: If that slot has no checkpoint saved yet.
    """
    if slot not in _VALID_SLOTS:
        raise ValueError(f"slot must be one of {_VALID_SLOTS}, got {slot!r}")
    directory = Path(directory)
    filename = CURRENT_FILENAME if slot == "current" else PREVIOUS_FILENAME
    path = directory / filename
    if not path.exists():
        raise FileNotFoundError(f"No {slot!r} checkpoint found at {path}")
    return torch.load(path, map_location="cpu")


def load_model_state(directory: PathLike, slot: str = "current") -> Dict[str, torch.Tensor]:
    """
    Convenience wrapper: just the model's state_dict.
    """
    return load_checkpoint(directory, slot=slot)["model_state"]


def rollback(directory: PathLike) -> None:
    """
    Revert current.pt to previous.pt's content.
    """
    directory = Path(directory)
    previous_path = directory / PREVIOUS_FILENAME
    current_path = directory / CURRENT_FILENAME
    if not previous_path.exists():
        raise FileNotFoundError(f"No previous checkpoint to roll back to at {previous_path}")
    shutil.copy2(previous_path, current_path)


def has_checkpoint(directory: PathLike, slot: str = "current") -> bool:
    """Check whether a given slot has a saved checkpoint, without raising."""
    if slot not in _VALID_SLOTS:
        raise ValueError(f"slot must be one of {_VALID_SLOTS}, got {slot!r}")
    directory = Path(directory)
    filename = CURRENT_FILENAME if slot == "current" else PREVIOUS_FILENAME
    return (directory / filename).exists()


if __name__ == "__main__":
    import tempfile
    import torch.nn as nn

    with tempfile.TemporaryDirectory() as tmp:
        model = nn.Linear(4, 2)
        target = nn.Linear(4, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        assert not has_checkpoint(tmp)

        save_checkpoint(tmp, model, optimizer, episode_count=5, total_env_steps=120, target_model=target)
        assert has_checkpoint(tmp)

        loaded = load_checkpoint(tmp, slot="current")
        assert loaded["episode_count"] == 5
        assert loaded["total_env_steps"] == 120
        assert "target_model_state" in loaded
        print("utils/checkpoint.py self-test OK")
