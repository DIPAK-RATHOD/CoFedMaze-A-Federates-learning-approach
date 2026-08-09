"""
vdn_agent.py

Concrete agent using the VDN-decomposed per-agent Q-function. One of
the two cooperating agents per node, per the Directory Structure
Reference's description of this file's role.

A VDNAgent owns exactly one thing that Policy deliberately does not:
its own hidden state across the lifetime of an episode. It wraps a
Policy (for action selection) and exposes the BaseAgent contract that
the (not yet built) PettingZoo wrapper will call into.
"""

from typing import Any, Optional, Tuple

import torch

from marl.agents.base_agent import BaseAgent
from marl.agents.policy import Policy


class VDNAgent(BaseAgent):
    """
    Stateful per-episode wrapper around a Policy: owns this agent's GRU
    hidden state and exposes act()/observe()/update()/reset_hidden().
    """

    def __init__(self, policy: Policy, device: Optional[torch.device] = None) -> None:
        """
        Args:
            policy: The Policy this agent uses for action selection.
            device: Device to allocate this agent's hidden state on.
                Defaults to the policy's model's own parameter device.
        """
        self.policy = policy
        self._device = device
        self._hidden: torch.Tensor = self._zero_hidden()

    def _zero_hidden(self, batch_size: int = 1) -> torch.Tensor:
        return self.policy.model.memory.init_hidden(batch_size, self._device)

    def reset_hidden(self, batch_size: int = 1) -> None:
        """
        Reset this agent's hidden state to zero. Must be called at the
        start of every new episode — GRU hidden state carries
        information across steps within an episode and must not leak
        into the next one.
        """
        self._hidden = self._zero_hidden(batch_size)

    def act(self, observation: torch.Tensor) -> Tuple[int, Any]:
        """
        Choose an action for the current step using the wrapped
        Policy's exploration strategy, advancing this agent's hidden
        state as a side effect.

        Args:
            observation: (1, in_channels, window, window) tensor for
                this agent's current step.

        Returns:
            (action, info) where info is a dict containing the full
            Q-value vector under key "q_values", for callers that want
            it for logging without a second forward pass.
        """
        action, q_values, new_hidden = self.policy.act(observation, self._hidden)
        self._hidden = new_hidden
        return action, {"q_values": q_values}

    def act_greedy(self, observation: torch.Tensor) -> Tuple[int, Any]:
        """
        Same as act(), but bypasses exploration (greedy action only).
        Intended for evaluation/deployment rollouts.
        """
        action, q_values, new_hidden = self.policy.act_greedy(observation, self._hidden)
        self._hidden = new_hidden
        return action, {"q_values": q_values}

    def observe(self, *args: Any, **kwargs: Any) -> None:
        """
        Not yet implemented: marl/replay/replay_buffer.py does not
        exist yet. Raises NotImplementedError rather than silently
        discarding the transition — see BaseAgent.observe()'s docstring
        for why a silent no-op would be worse than an explicit error
        here.
        """
        raise NotImplementedError(
            "VDNAgent.observe() requires marl/replay/replay_buffer.py, which has not "
            "been built yet. This agent can currently act() but not learn from "
            "experience."
        )

    def update(self, *args: Any, **kwargs: Any) -> Any:
        """
        Not yet implemented: marl/training/trainer.py and
        marl/losses/vdn_loss.py do not exist yet. Raises
        NotImplementedError for the same reason as observe().
        """
        raise NotImplementedError(
            "VDNAgent.update() requires marl/training/trainer.py and "
            "marl/losses/vdn_loss.py, which have not been built yet."
        )


if __name__ == "__main__":
    import random

    from marl.agents.action_selector import EpsilonGreedySelector
    from marl.models.vdn import VDNModel

    model = VDNModel(in_channels=8, window_size=5, num_actions=5, num_agents=2)
    selector = EpsilonGreedySelector(epsilon=0.2, rng=random.Random(0))
    policy = Policy(model, agent_index=0, selector=selector)
    agent = VDNAgent(policy)

    # Simulate a 3-step episode, confirming hidden state actually
    # changes between steps (i.e. memory is really being carried).
    hidden_snapshots = [agent._hidden.clone()]
    for step in range(3):
        observation = torch.rand(1, 8, 5, 5)
        action, info = agent.act(observation)
        print(f"Step {step}: action={action}, q_values={info['q_values'].tolist()}")
        hidden_snapshots.append(agent._hidden.clone())

    changed = [
        not torch.allclose(hidden_snapshots[i], hidden_snapshots[i + 1])
        for i in range(len(hidden_snapshots) - 1)
    ]
    print("Hidden state changed each step:", changed)
    assert all(changed)

    agent.reset_hidden()
    print("Hidden state reset to zero:", torch.allclose(agent._hidden, torch.zeros_like(agent._hidden)))
    assert torch.allclose(agent._hidden, torch.zeros_like(agent._hidden))

    try:
        agent.observe()
        print("FAIL: should have raised")
    except NotImplementedError as e:
        print("OK:", e)

    try:
        agent.update()
        print("FAIL: should have raised")
    except NotImplementedError as e:
        print("OK:", e)

    print("OK")
