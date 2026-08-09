"""
vdn.py

Assembles encoder + gru + heads + mixing into the full local two-agent
VDN model for one node. This is the single class the trainer and
checkpoint system will actually save/load (per the Directory Structure
Reference's description of vdn.py's role).

Parameter-sharing design:
    The encoder and GRU are shared (one instance, used by both agents)
    rather than each agent getting its own copy. This matches the
    Shared vs. Private Model Components glossary term and standard
    VDN/QMIX practice for homogeneous agents: sharing improves sample
    efficiency (both agents' experience trains the same maze-encoding
    weights), and per-agent behavior differences still come through via
    the private QHead, which optionally takes a one-hot agent-identity
    vector to break symmetry. If the two agents in a node ever need
    genuinely different observation encoders (e.g. structurally
    different roles, not just different identity), that would be a
    reason to revisit this and give each agent its own SharedEncoder
    instance instead — flagged here as the assumption to challenge if
    agent roles diverge.
"""

from typing import List, Optional, Tuple

import torch
from torch import nn

from marl.models.components.encoder import SharedEncoder
from marl.models.components.gru import RecurrentMemory
from marl.models.components.heads import QHead
from marl.models.components.mixing import VDNMixer


class VDNModel(nn.Module):
    """
    The full local model for one node: two agents sharing an encoder
    and GRU memory, each with a private Q-head, combined via VDN
    summation for team-level TD training.
    """

    def __init__(
        self,
        in_channels: int,
        window_size: int,
        num_actions: int,
        num_agents: int = 2,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        use_agent_id: bool = True,
    ) -> None:
        """
        Args:
            in_channels: Observation channel count (see encoder.py).
            window_size: Observation window height/width (see encoder.py).
            num_actions: Size of the discrete action space.
            num_agents: Number of agents sharing this model. The
                project's glossary specifies two agents per node; kept
                configurable rather than hardcoded to 2 in case that
                changes.
            embedding_dim: SharedEncoder output size.
            hidden_dim: RecurrentMemory hidden state size.
            use_agent_id: Whether QHead receives a one-hot agent-identity
                vector (of size num_agents) to break symmetry between
                agents sharing the encoder/GRU. See module docstring.

        Raises:
            ValueError: If num_agents is not a positive integer.
        """
        super().__init__()
        if num_agents <= 0:
            raise ValueError(f"num_agents must be positive, got {num_agents}")

        self.num_agents = num_agents
        self.use_agent_id = use_agent_id
        self.hidden_dim = hidden_dim

        self.encoder = SharedEncoder(
            in_channels=in_channels, window_size=window_size, embedding_dim=embedding_dim
        )
        self.memory = RecurrentMemory(input_dim=embedding_dim, hidden_dim=hidden_dim)
        agent_id_dim = num_agents if use_agent_id else 0
        self.head = QHead(hidden_dim=hidden_dim, num_actions=num_actions, agent_id_dim=agent_id_dim)
        self.mixer = VDNMixer()

        # Precompute the identity matrix used to build one-hot agent-id
        # vectors on demand; registered as a buffer so it moves with
        # the model across devices (.to(device)) automatically.
        if use_agent_id:
            self.register_buffer("_agent_id_eye", torch.eye(num_agents))

    def init_hidden(self, batch_size: int, device: Optional[torch.device] = None) -> List[torch.Tensor]:
        """
        Create zeroed hidden states for all agents at episode start.

        Returns:
            A list of length num_agents, each a (batch_size, hidden_dim)
            zero tensor.
        """
        return [self.memory.init_hidden(batch_size, device) for _ in range(self.num_agents)]

    def forward_agent(
        self, observation: torch.Tensor, hidden: torch.Tensor, agent_index: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run one agent's forward pass for a single step.

        Args:
            observation: (batch, in_channels, window, window) tensor
                for this agent.
            hidden: (batch, hidden_dim) previous hidden state for this
                agent (from init_hidden() or a prior call's return).
            agent_index: Which agent this is (0-indexed, < num_agents) —
                used to build the one-hot identity vector if
                use_agent_id is enabled.

        Returns:
            (q_values, new_hidden): q_values has shape
            (batch, num_actions); new_hidden has shape
            (batch, hidden_dim).

        Raises:
            ValueError: If agent_index is out of range [0, num_agents).
        """
        if not (0 <= agent_index < self.num_agents):
            raise ValueError(
                f"agent_index must be in [0, {self.num_agents}), got {agent_index}"
            )

        embedding = self.encoder(observation)
        new_hidden = self.memory(embedding, hidden)

        agent_id_tensor: Optional[torch.Tensor] = None
        if self.use_agent_id:
            batch_size = observation.shape[0]
            agent_id_tensor = self._agent_id_eye[agent_index].unsqueeze(0).expand(batch_size, -1)

        q_values = self.head(new_hidden, agent_id_tensor)
        return q_values, new_hidden

    def forward_team(
        self,
        observations: List[torch.Tensor],
        hiddens: List[torch.Tensor],
        actions: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], Optional[torch.Tensor]]:
        """
        Run all agents' forward passes for a single step, and optionally
        compute the team Q_tot for a given joint action (for TD-loss
        computation in the — not yet built — training loop).

        Args:
            observations: List of length num_agents, each
                (batch, in_channels, window, window).
            hiddens: List of length num_agents, each (batch, hidden_dim).
            actions: Optional list of length num_agents, each
                (batch,) int64 tensor of chosen action indices. If
                provided, Q_tot is computed by gathering each agent's
                Q-value for its chosen action and summing via VDNMixer.

        Returns:
            (q_values_list, new_hiddens_list, q_tot):
                q_values_list: per-agent full Q-vectors,
                    each (batch, num_actions).
                new_hiddens_list: per-agent new hidden states,
                    each (batch, hidden_dim).
                q_tot: (batch,) team Q-value if `actions` was given,
                    else None.

        Raises:
            ValueError: If `observations` or `hiddens` doesn't have
                exactly num_agents entries, or (when provided) `actions`
                doesn't either.
        """
        if len(observations) != self.num_agents:
            raise ValueError(
                f"Expected {self.num_agents} observations, got {len(observations)}"
            )
        if len(hiddens) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} hidden states, got {len(hiddens)}")
        if actions is not None and len(actions) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} action tensors, got {len(actions)}")

        q_values_list: List[torch.Tensor] = []
        new_hiddens_list: List[torch.Tensor] = []
        for i in range(self.num_agents):
            q_values, new_hidden = self.forward_agent(observations[i], hiddens[i], i)
            q_values_list.append(q_values)
            new_hiddens_list.append(new_hidden)

        q_tot: Optional[torch.Tensor] = None
        if actions is not None:
            chosen = [
                q.gather(1, actions[i].unsqueeze(1)).squeeze(1)
                for i, q in enumerate(q_values_list)
            ]
            q_tot = self.mixer(chosen)

        return q_values_list, new_hiddens_list, q_tot


if __name__ == "__main__":
    model = VDNModel(in_channels=8, window_size=5, num_actions=5, num_agents=2)

    batch_size = 4
    hiddens = model.init_hidden(batch_size)
    observations = [torch.rand(batch_size, 8, 5, 5) for _ in range(2)]
    actions = [torch.randint(0, 5, (batch_size,)) for _ in range(2)]

    q_values_list, new_hiddens, q_tot = model.forward_team(observations, hiddens, actions)
    print("Per-agent Q-value shapes:", [q.shape for q in q_values_list])
    print("Per-agent hidden shapes:", [h.shape for h in new_hiddens])
    print("Q_tot shape:", q_tot.shape)

    assert all(q.shape == (batch_size, 5) for q in q_values_list)
    assert all(h.shape == (batch_size, 128) for h in new_hiddens)
    assert q_tot.shape == (batch_size,)
    print("OK")
