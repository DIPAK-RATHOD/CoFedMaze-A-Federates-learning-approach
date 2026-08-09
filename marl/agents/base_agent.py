"""
base_agent.py

Abstract agent interface shared by every concrete agent implementation,
so vdn_agent.py and any future agent type present a common contract to
the (not yet built) PettingZoo wrapper — matching the Directory
Structure Reference's description of this file's role.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch


class BaseAgent(ABC):
    """
    Common contract every agent implementation must satisfy: act,
    observe, update.
    """

    @abstractmethod
    def act(self, observation: torch.Tensor) -> Tuple[int, Any]:
        """
        Choose an action given the current observation, advancing this
        agent's internal recurrent state as a side effect.

        Args:
            observation: This agent's observation for the current step,
                shape (1, in_channels, window, window) — single-agent,
                single-environment inference (batch size 1). Batched
                inference for training happens through the model
                directly (see marl/models/vdn.py), not through this
                per-step agent interface.

        Returns:
            (action, info) — `action` is the chosen discrete action
            index; `info` is implementation-defined extra data (e.g.
            the full Q-value vector) callers may want for logging.
        """
        raise NotImplementedError

    @abstractmethod
    def observe(self, *args: Any, **kwargs: Any) -> None:
        """
        Record a transition (observation, action, reward, next
        observation, done) into this agent's replay buffer.

        Not implemented by any concrete agent yet: marl/replay/ (the
        replay buffer implementation) has not been built. Concrete
        subclasses should raise NotImplementedError with a message
        pointing here until that layer exists, rather than silently
        no-op — a silent no-op would look like working code while
        quietly discarding every transition.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> Any:
        """
        Run one training step (sample a batch, compute VDN TD loss,
        step the optimizer).

        Not implemented by any concrete agent yet: marl/training/
        (trainer.py, optimizer.py, losses/vdn_loss.py) has not been
        built. Concrete subclasses should raise NotImplementedError
        pointing here rather than silently no-op, for the same reason
        as observe() above.
        """
        raise NotImplementedError

    @abstractmethod
    def reset_hidden(self, batch_size: int = 1) -> None:
        """
        Reset this agent's recurrent hidden state to zero — call at the
        start of every new episode, since GRU hidden state must not
        leak across episode boundaries.
        """
        raise NotImplementedError
