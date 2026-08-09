"""
model_similarity.py

Computes Model Similarity (MS) via a lightweight cosine similarity over
the shared encoder weights ONLY (never the full model, never
memory/head) -- per Section 3.7.8's "Model Similarity" derivation.
"""

from typing import Dict
import torch

EncoderState = Dict[str, torch.Tensor]


def _flatten(state: EncoderState) -> torch.Tensor:
    return torch.cat([t.flatten().float() for t in state.values()])


def compute_model_similarity(encoder_a: EncoderState, encoder_b: EncoderState) -> float:
    """
    Cosine similarity between two nodes' shared ENCODER weights only.
    Remapped from cosine similarity's natural [-1, 1] range onto [0, 1]
    via (cos_sim + 1) / 2. Returns 0.5 (neutral) if either encoder vector
    is all-zero / uninitialized rather than raising a fatal exception.
    """
    if set(encoder_a.keys()) != set(encoder_b.keys()):
        raise ValueError(
            f"encoder_a and encoder_b have different parameter keys: "
            f"{set(encoder_a.keys())} vs {set(encoder_b.keys())}"
        )
    for key in encoder_a:
        if encoder_a[key].shape != encoder_b[key].shape:
            raise ValueError(
                f"encoder_a['{key}'] shape {tuple(encoder_a[key].shape)} != "
                f"encoder_b['{key}'] shape {tuple(encoder_b[key].shape)}"
            )

    vec_a = _flatten(encoder_a)
    vec_b = _flatten(encoder_b)

    norm_a = vec_a.norm().item()
    norm_b = vec_b.norm().item()
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.5  # Neutral fallback for uninitialized or zeroed state vectors

    cos_sim = torch.dot(vec_a, vec_b).item() / (norm_a * norm_b)
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return (cos_sim + 1.0) / 2.0


if __name__ == "__main__":
    from env.core.actions import NUM_ACTIONS
    from env.core.observations import NUM_CHANNELS
    from federation.validation.transfer_validation import extract_shared_state
    from marl.models.vdn import VDNModel

    model_a = VDNModel(in_channels=NUM_CHANNELS, window_size=5, num_actions=NUM_ACTIONS, num_agents=2)
    model_b = VDNModel(in_channels=NUM_CHANNELS, window_size=5, num_actions=NUM_ACTIONS, num_agents=2)

    encoder_a = extract_shared_state(model_a)["encoder"]
    encoder_b = extract_shared_state(model_b)["encoder"]

    ms_different = compute_model_similarity(encoder_a, encoder_b)
    print("MS between two independently-initialized models:", ms_different)
    assert 0.0 <= ms_different <= 1.0

    ms_identical = compute_model_similarity(encoder_a, encoder_a)
    print("MS of a model against itself (should be exactly 1.0):", ms_identical)
    assert abs(ms_identical - 1.0) < 1e-6

    print("OK")
