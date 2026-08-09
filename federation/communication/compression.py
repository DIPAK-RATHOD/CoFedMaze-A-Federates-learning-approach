"""
compression.py

8-bit quantization of a shared-component update, applied to a DELTA
from a base state rather than the raw state itself -- implements the
delta/quantized exchange principle from the edge-efficiency strategy
doc (roughly a 4x payload reduction vs. float32), critical for
Pi-to-Pi bandwidth limits.

Stateless by design: given a `current` state and a `base` state (the
last version actually sent to a given neighbor), this file only knows
how to compute/quantize the delta and how to reconstruct from a
quantized delta + that same base. TRACKING which base each neighbor
was last sent -- so the right base gets passed in here -- is a
node-level orchestration concern (node/scheduler.py, not yet built),
not this file's job.

Per-tensor affine quantization (one scale + zero-point per parameter
tensor, not per-channel): the simplest scheme that still gets the full
4x reduction, matching this project's established "minimum computation
per round" philosophy (e.g. the KS EMA is O(1) per edge, not a stored
history).

Lossy by construction: decompress_delta() reconstructs an
APPROXIMATION of the sender's original state, not a bit-identical copy
-- see compression_ratio() for the actual bandwidth/precision tradeoff
achieved, rather than assuming the textbook "4x" figure holds exactly.
"""

from typing import Dict, NamedTuple, Tuple

import torch

from federation.aggregation.fedavg import validate_matching_keys_and_shapes

SharedState = Dict[str, Dict[str, torch.Tensor]]

_REQUIRED_COMPONENTS = ("encoder", "memory")


class QuantizedTensor(NamedTuple):
    """
    An 8-bit quantized representation of one float tensor. q_values are
    uint8 in [0, 255]; scale/zero_point recover the approximate
    original range via dequantize_tensor().
    """
    q_values: torch.Tensor
    scale: float
    zero_point: float
    shape: Tuple[int, ...]


QuantizedState = Dict[str, Dict[str, QuantizedTensor]]


def quantize_tensor(x: torch.Tensor) -> QuantizedTensor:
    """
    Affine 8-bit quantization: maps x's [min, max] range onto [0, 255].

    A constant tensor (min == max, e.g. an all-zero delta where a
    parameter didn't change at all) is quantized with scale=0 rather
    than dividing by a zero range -- dequantize_tensor() special-cases
    this back to the exact constant value, with zero error (the one
    case where quantization is lossless: nothing to lose).
    """
    x = x.float()
    x_min = x.min().item()
    x_max = x.max().item()
    if x_max == x_min:
        q = torch.zeros_like(x, dtype=torch.uint8)
        return QuantizedTensor(q_values=q, scale=0.0, zero_point=x_min, shape=tuple(x.shape))

    scale = (x_max - x_min) / 255.0
    q = torch.round((x - x_min) / scale).clamp(0, 255).to(torch.uint8)
    return QuantizedTensor(q_values=q, scale=scale, zero_point=x_min, shape=tuple(x.shape))


def dequantize_tensor(q: QuantizedTensor) -> torch.Tensor:
    """Inverse of quantize_tensor() -- approximate for scale > 0, exact for the constant-tensor (scale=0) case."""
    if q.scale == 0.0:
        return torch.full(q.shape, q.zero_point, dtype=torch.float32)
    return q.q_values.float() * q.scale + q.zero_point


def compress_delta(current: SharedState, base: SharedState) -> QuantizedState:
    """
    Compute (current - base) for every parameter tensor, then quantize
    each delta tensor to 8-bit.

    Raises:
        ValueError: If current/base have mismatched keys or shapes
            (reuses federated_average's own validation, so a mismatch
            is caught with the same clear error rather than a cryptic
            subtraction failure).
    """
    validate_matching_keys_and_shapes([current, base])

    result: QuantizedState = {"encoder": {}, "memory": {}}
    for component in _REQUIRED_COMPONENTS:
        for key in current[component]:
            delta = current[component][key].float() - base[component][key].float()
            result[component][key] = quantize_tensor(delta)
    return result


def decompress_delta(quantized_delta: QuantizedState, base: SharedState) -> SharedState:
    """Reconstruct an approximate current state: base + dequantize(delta)."""
    result: SharedState = {"encoder": {}, "memory": {}}
    for component in _REQUIRED_COMPONENTS:
        for key in quantized_delta[component]:
            delta = dequantize_tensor(quantized_delta[component][key])
            result[component][key] = base[component][key].float() + delta
    return result


def compression_ratio(original: SharedState, quantized: QuantizedState) -> float:
    """
    Actual achieved compression ratio (original float32 bytes /
    quantized bytes) -- computed from real byte counts rather than
    assumed, since per-tensor scale/zero_point overhead makes the real
    figure very slightly below a pure 4x.
    """
    original_bytes = sum(
        t.numel() * 4  # float32 = 4 bytes/element
        for component in _REQUIRED_COMPONENTS
        for t in original[component].values()
    )
    quantized_bytes = sum(
        q.q_values.numel() * 1 + 16  # uint8 = 1 byte/element, +16 bytes generously estimated for scale/zero_point
        for component in _REQUIRED_COMPONENTS
        for q in quantized[component].values()
    )
    return original_bytes / quantized_bytes


if __name__ == "__main__":
    from env.core.actions import NUM_ACTIONS
    from env.core.observations import NUM_CHANNELS
    from federation.validation.transfer_validation import extract_shared_state
    from marl.models.vdn import VDNModel

    base_model = VDNModel(in_channels=NUM_CHANNELS, window_size=5, num_actions=NUM_ACTIONS, num_agents=2)
    current_model = VDNModel(in_channels=NUM_CHANNELS, window_size=5, num_actions=NUM_ACTIONS, num_agents=2)

    base_state = extract_shared_state(base_model)
    current_state = extract_shared_state(current_model)

    quantized = compress_delta(current_state, base_state)
    reconstructed = decompress_delta(quantized, base_state)

    ratio = compression_ratio(current_state, quantized)
    print(f"Compression ratio: {ratio:.2f}x (textbook target ~4x)")
    assert 3.5 < ratio < 4.0

    key = list(current_state["encoder"].keys())[0]
    max_error = (reconstructed["encoder"][key] - current_state["encoder"][key]).abs().max().item()
    print(f"Max reconstruction error on '{key}': {max_error:.6f}")
    assert max_error < 0.1  # lossy but should be small, not wildly off

    print("OK")
