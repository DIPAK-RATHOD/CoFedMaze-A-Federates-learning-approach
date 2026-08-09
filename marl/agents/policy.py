"""
policy.py

High-level policy interface used by the training loop and evaluation.
Decouples "how an action is chosen" (action_selector.py) from "what the
agent's overall behavior contract is" (this file) — matching the
Directory Structure Reference's stated rationale.

A Policy wraps a VDNModel (for one specific agent slot) and an
EpsilonGreedySelector, and exposes both exploratory and greedy action
selection without the caller needing to know about hidden-state
plumbing or Q-value tensors directly.
"""

from typing import Optional, Tuple

import torch

from marl.agents.action_selector import EpsilonGreedySelector
from marl.models.vdn import VDNModel


class Policy:
    """
    Per-agent policy: wraps a shared VDNModel plus an
    EpsilonGreedySelector to answer "what action does agent i take
    given this observation and hidden state."

    A Policy does NOT own hidden state itself — hidden state belongs to
    whatever is running the episode (VDNAgent), since a Policy may be
    asked to act for the same agent across many independent parallel
    episodes (e.g. during batched evaluation), each with its own hidden
    state. Keeping Policy stateless with respect to hidden state avoids
    that ambiguity.
    """

    def __init__(self, model: VDNModel, agent_index: int, selector: EpsilonGreedySelector) -> None:
        """
        Args:
            model: The (parameter-shared) VDNModel this policy queries.
            agent_index: Which of the model's agents this policy acts
                for (0-indexed, < model.num_agents).
            selector: The exploration strategy to use for act(); ignored
                by act_greedy().

        Raises:
            ValueError: If agent_index is out of range for the model.
        """
        if not (0 <= agent_index < model.num_agents):
            raise ValueError(
                f"agent_index must be in [0, {model.num_agents}), got {agent_index}"
            )
        self.model = model
        self.agent_index = agent_index
        self.selector = selector

    def act(
        self, observation: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Choose an action using the wrapped exploration strategy.

        Args:
            observation: (1, in_channels, window, window) — single-step,
                single-environment observation for this agent.
            hidden: (1, hidden_dim) previous hidden state for this
                agent.

        Returns:
            (action, q_values, new_hidden):
                action: chosen action index (int).
                q_values: (num_actions,) full Q-vector, batch dim
                    squeezed off for the caller's convenience.
                new_hidden: (1, hidden_dim) updated hidden state to pass
                    into the next call.
        """
        with torch.no_grad():
            q_values_batched, new_hidden = self.model.forward_agent(
                observation, hidden, self.agent_index
            )
        q_values = q_values_batched.squeeze(0)
        action = self.selector.select(q_values)
        return action, q_values, new_hidden

    def act_greedy(
        self, observation: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Choose the greedy (argmax) action, bypassing exploration —
        intended for evaluation/deployment rather than training
        rollouts.

        Args/Returns: same shape contract as act().
        """
        with torch.no_grad():
            q_values_batched, new_hidden = self.model.forward_agent(
                observation, hidden, self.agent_index
            )
        q_values = q_values_batched.squeeze(0)
        action = int(torch.argmax(q_values).item())
        return action, q_values, new_hidden


if __name__ == "__main__":
    import random

    model = VDNModel(in_channels=8, window_size=5, num_actions=5, num_agents=2)
    selector = EpsilonGreedySelector(epsilon=0.0, rng=random.Random(0))
    policy = Policy(model, agent_index=0, selector=selector)

    hidden = model.init_hidden(batch_size=1)[0]
    observation = torch.rand(1, 8, 5, 5)

    action, q_values, new_hidden = policy.act(observation, hidden)
    print("Action:", action, "Q-values shape:", q_values.shape, "New hidden shape:", new_hidden.shape)
    assert 0 <= action < 5
    assert q_values.shape == (5,)
    assert new_hidden.shape == (1, 128)

    greedy_action, _, _ = policy.act_greedy(observation, hidden)
    print("Greedy action matches epsilon=0 action:", greedy_action == action)
    assert greedy_action == action
    print("OK")
