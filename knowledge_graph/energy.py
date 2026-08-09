"""
energy.py

Computes the normalized energy cost (E) criterion for a directed edge.

SIMULATION NOTE: no real energy sensor exists yet (the Implementation
Requirements doc lists an INA219 power sensor as OPTIONAL hardware, not
yet integrated). Energy here is estimated from the SERIALIZED MESSAGE
SIZE in bytes (federation/communication/serializer.py's actual
output) -- transmission energy on real hardware genuinely scales with
payload size, so this is a grounded proxy computed from a real,
already-built quantity, not an arbitrary number. Swap this for real
INA219 measurements once that hardware integration exists.
"""

from federation.communication.messages import UpdateMessage
from federation.communication.serializer import serialize_message
from knowledge_graph.normalization import clip_normalize

# PLACEHOLDER, not measured: assumed energy cost in joules per byte
# transmitted on real Pi-to-Pi hardware. Retune once real INA219
# measurements exist.
JOULES_PER_BYTE = 1e-6
# PLACEHOLDER normalization range: 0 to a generously large 2MB message
# (a full uncompressed encoder+memory state is comfortably under this).
_MAX_MESSAGE_BYTES = 2_000_000
_ENERGY_RANGE_J = (0.0, _MAX_MESSAGE_BYTES * JOULES_PER_BYTE)


def compute_energy_from_size(size_bytes: int) -> float:
    """
    Normalized energy estimate from a raw byte count directly -- avoids
    re-serializing a message when the caller already has serialized
    bytes on hand (e.g. node/scheduler.py, which receives raw bytes
    from transport.py's receive_all() before any deserialization
    happens; calling compute_energy(message) there would mean
    serializing the SAME payload a second time just to measure it).
    """
    estimated_joules = size_bytes * JOULES_PER_BYTE
    return clip_normalize(estimated_joules, *_ENERGY_RANGE_J)


def compute_energy(message: UpdateMessage) -> float:
    """Normalized energy estimate for transmitting `message`, based on its actual serialized byte size."""
    size_bytes = len(serialize_message(message))
    return compute_energy_from_size(size_bytes)


if __name__ == "__main__":
    from env.core.actions import NUM_ACTIONS
    from env.core.observations import NUM_CHANNELS
    from federation.communication.compression import compress_delta
    from federation.validation.transfer_validation import extract_shared_state
    from marl.models.vdn import VDNModel

    model = VDNModel(in_channels=NUM_CHANNELS, window_size=5, num_actions=NUM_ACTIONS, num_agents=2)
    state = extract_shared_state(model)

    # Uncompressed (raw float32 state used directly as payload) vs
    # compressed (8-bit quantized delta from a zero base, i.e. the
    # actual wire format federation/communication/protocol.py sends).
    raw_message = UpdateMessage(node_id="N1", round=1, validation_reward=1.0, payload=state)

    zero_base = {c: {k: v * 0 for k, v in tensors.items()} for c, tensors in state.items()}
    quantized = compress_delta(state, zero_base)
    # update_norm must be supplied explicitly for a quantized payload --
    # UpdateMessage's auto-compute assumes raw tensors (see
    # federation/communication/protocol.py's module docstring for the
    # full explanation of this same issue, fixed there the same way).
    compressed_message = UpdateMessage(
        node_id="N1", round=1, validation_reward=1.0, payload=quantized, update_norm=0.0
    )

    e_raw = compute_energy(raw_message)
    e_compressed = compute_energy(compressed_message)
    print("Energy (uncompressed payload):", e_raw)
    print("Energy (compressed/quantized payload):", e_compressed)
    assert e_compressed < e_raw
    print("OK: compression genuinely reduces the estimated energy cost, not just message size")
