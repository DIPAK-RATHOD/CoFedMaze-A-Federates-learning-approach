"""
serializer.py

Serializes/deserializes UpdateMessage (tensors + metadata) to/from
bytes for transmission. Needed regardless of transport choice (in-
process simulation now, ZeroMQ/gRPC later per federation.yaml) -- kept
transport-agnostic so switching transport.py's implementation later
never requires touching this file.

============================================================================
Security note: why this does NOT just torch.save(message) as a whole
============================================================================
PyTorch's torch.load() defaults to weights_only=True specifically to
prevent arbitrary code execution from untrusted pickled data -- and
this file's whole job is deserializing data that (once transport.py
does real networking) arrives from OTHER NODES, which is exactly the
"untrusted source" weights_only=True exists to protect against. Naively
pickling the whole UpdateMessage dataclass fails under weights_only=True
(PyTorch doesn't allowlist arbitrary custom classes by default), and
the tempting fix -- passing weights_only=False -- would silently
reintroduce that exact vulnerability for the one file in this project
that actually needs to guard against it. utils/checkpoint.py's use of
torch.load is a different situation (loading YOUR OWN previously-saved
local file, not data received from a peer), which is why it doesn't
need this same treatment.

Instead: metadata (node_id, round, update_norm, validation_reward,
timestamp -- all plain strings/numbers) is serialized as JSON, no
pickle involved at all. Only the payload (a plain
Dict[str, Dict[str, Tensor]] with no custom classes) goes through
torch.load, WITH weights_only=True kept on -- nested dicts of tensors
are exactly what weights_only=True is designed to support safely.
============================================================================
"""

import io
import json
import struct

import torch

from federation.communication.messages import UpdateMessage

_LENGTH_HEADER_FORMAT = ">I"  # 4-byte big-endian unsigned int
_LENGTH_HEADER_SIZE = struct.calcsize(_LENGTH_HEADER_FORMAT)


def serialize_message(message: UpdateMessage) -> bytes:
    """
    Layout: [4-byte metadata length][JSON metadata bytes][torch-saved payload bytes].
    """
    metadata = {
        "node_id": message.node_id,
        "round": message.round,
        "update_norm": message.update_norm,
        "validation_reward": message.validation_reward,
        "timestamp": message.timestamp,
    }
    metadata_bytes = json.dumps(metadata).encode("utf-8")

    payload_buffer = io.BytesIO()
    torch.save(message.payload, payload_buffer)  # payload is a plain tensor dict -- no custom classes
    payload_bytes = payload_buffer.getvalue()

    header = struct.pack(_LENGTH_HEADER_FORMAT, len(metadata_bytes))
    return header + metadata_bytes + payload_bytes


def deserialize_message(data: bytes) -> UpdateMessage:
    """
    Raises:
        ValueError: If `data` is too short to contain a valid header,
            or the embedded length doesn't fit within the given bytes
            (a truncated or corrupted message).
    """
    if len(data) < _LENGTH_HEADER_SIZE:
        raise ValueError(
            f"data is too short to contain a valid message header "
            f"({len(data)} bytes, need at least {_LENGTH_HEADER_SIZE})"
        )

    (metadata_len,) = struct.unpack(_LENGTH_HEADER_FORMAT, data[:_LENGTH_HEADER_SIZE])
    metadata_end = _LENGTH_HEADER_SIZE + metadata_len
    if metadata_end > len(data):
        raise ValueError(
            f"Corrupted or truncated message: header claims {metadata_len} metadata "
            f"bytes but only {len(data) - _LENGTH_HEADER_SIZE} bytes remain"
        )

    metadata = json.loads(data[_LENGTH_HEADER_SIZE:metadata_end].decode("utf-8"))

    payload_buffer = io.BytesIO(data[metadata_end:])
    # weights_only=True kept ON deliberately -- see module docstring.
    payload = torch.load(payload_buffer, map_location="cpu", weights_only=True)

    return UpdateMessage(
        node_id=metadata["node_id"],
        round=metadata["round"],
        validation_reward=metadata["validation_reward"],
        payload=payload,
        update_norm=metadata["update_norm"],
        timestamp=metadata["timestamp"],
    )


if __name__ == "__main__":
    from env.core.actions import NUM_ACTIONS
    from env.core.observations import NUM_CHANNELS
    from federation.validation.transfer_validation import extract_shared_state
    from marl.models.vdn import VDNModel

    model = VDNModel(in_channels=NUM_CHANNELS, window_size=5, num_actions=NUM_ACTIONS, num_agents=2)
    shared_state = extract_shared_state(model)
    original = UpdateMessage(node_id="N1", round=1, validation_reward=2.0, payload=shared_state)

    data = serialize_message(original)
    print("Serialized size (bytes):", len(data))

    restored = deserialize_message(data)
    print("Round-tripped node_id/round/reward:", restored.node_id, restored.round, restored.validation_reward)

    assert restored.node_id == original.node_id
    assert restored.round == original.round
    assert restored.validation_reward == original.validation_reward
    assert restored.update_norm == original.update_norm

    exact_match = all(
        torch.equal(original.payload[c][k], restored.payload[c][k])
        for c in ("encoder", "memory")
        for k in original.payload[c]
    )
    print("Every tensor round-trips EXACTLY (no precision loss):", exact_match)
    assert exact_match

    try:
        deserialize_message(data[:2])
        print("FAIL: should have raised")
    except ValueError as e:
        print("OK, truncated header rejected:", e)
    try:
        deserialize_message(data[:20])
        print("FAIL: should have raised")
    except ValueError as e:
        print("OK, truncated body rejected:", e)

    print("OK")
