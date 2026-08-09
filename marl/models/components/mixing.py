"""
mixing.py

The VDN summation layer: combines each agent's Q-value for its chosen
action into a single team Q-value.

This is literally what makes VDN "VDN" — the additive decomposition
from Sunehag et al., Value-Decomposition Networks For Cooperative
Multi-Agent Learning. It is the one component genuinely unique to
value-decomposition methods versus vanilla independent Q-learning, so
it is kept as its own file/class even though the operation itself is a
simple sum with no learnable parameters — that separation is what would
let a future variant (e.g. QMIX's learned monotonic mixing network)
be swapped in as a drop-in replacement without touching encoder.py,
gru.py, or heads.py.
"""

from typing import List

import torch
from torch import nn


class VDNMixer(nn.Module):
    """
    Sums per-agent chosen-action Q-values into a team Q-value.

    Deliberately has no learnable parameters: Q_tot(s, a) = sum_i
    Q_i(s_i, a_i). This is the whole idea of VDN — team value
    decomposes additively across agents, so training the summed value
    against a shared team TD-target still produces individually useful
    per-agent Q-functions.
    """

    def forward(self, agent_q_values: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            agent_q_values: A list of per-agent Q-values for their
                CHOSEN actions (already gathered/indexed — not full
                per-action Q-vectors), one tensor of shape (batch,) per
                agent. All tensors must share the same shape.

        Returns:
            Team Q-value of shape (batch,) — the elementwise sum across
            agents.

        Raises:
            ValueError: If `agent_q_values` is empty, any tensor isn't
                1-dimensional, or the tensors' shapes disagree.
        """
        if not agent_q_values:
            raise ValueError("agent_q_values must contain at least one agent's Q-values")

        expected_shape = agent_q_values[0].shape
        for i, q in enumerate(agent_q_values):
            if q.dim() != 1:
                raise ValueError(
                    f"Expected each agent's Q-values to be 1-D (batch,), got "
                    f"{tuple(q.shape)} for agent index {i} — pass gathered/chosen-action "
                    "Q-values, not full per-action Q-vectors"
                )
            if q.shape != expected_shape:
                raise ValueError(
                    f"Shape mismatch: agent 0 has shape {tuple(expected_shape)}, "
                    f"agent {i} has shape {tuple(q.shape)}"
                )

        return torch.stack(agent_q_values, dim=0).sum(dim=0)


if __name__ == "__main__":
    mixer = VDNMixer()
    q_agent_0 = torch.tensor([1.0, 2.0, 3.0])
    q_agent_1 = torch.tensor([0.5, -1.0, 4.0])
    q_tot = mixer([q_agent_0, q_agent_1])
    print("Q_tot:", q_tot)
    assert torch.allclose(q_tot, torch.tensor([1.5, 1.0, 7.0]))
    print("OK")
