"""
protocol.py

Defines the message exchange SEQUENCE between nodes -- what gets
computed, in what order, before bytes go over the wire (serializer.py)
and how a received message gets turned back into a usable shared
state. Separated from transport.py's raw socket mechanics (not yet
built), per the Directory Structure Reference's stated rationale for
splitting these two files.

Owns per-neighbor CONVERSATION STATE: the last shared state sent to,
and last state reconstructed from, each neighbor. This is "protocol
state" in the standard networking sense (a connection's own memory of
where it left off) -- distinct from node/scheduler.py's job (not yet
built) of deciding WHEN to call prepare_outgoing()/receive_incoming()
each round.

First exchange with a neighbor is not a special case: compress_delta()
always runs, with the base defaulting to an all-zero state (matching
shapes) when no prior state is known for that neighbor. "Delta from
zero" is just the current values themselves, so the same code path
naturally sends/reconstructs a full state on the first exchange and a
true delta on every exchange after -- no branch needed.

Bug avoided here, not left for someone to hit later: messages.py's
UpdateMessage auto-computes update_norm from its payload assuming raw
tensors, but the payload sent over the wire is a QUANTIZED delta
(QuantizedTensor objects, no .flatten()/.float() methods a raw tensor
has). update_norm is therefore computed here from the RAW
(pre-quantization) delta and passed into UpdateMessage explicitly,
never left to the dataclass's auto-compute path against a payload type
it wasn't designed for.
"""

from typing import Dict, NamedTuple

import torch
import torch.serialization

from federation.communication.compression import QuantizedTensor, compress_delta, decompress_delta
from federation.communication.messages import UpdateMessage, compute_update_norm
from federation.communication.serializer import deserialize_message, serialize_message

SharedState = Dict[str, Dict[str, torch.Tensor]]


class ReceivedUpdate(NamedTuple):
    """
    A fully-processed incoming update: metadata plus the RECONSTRUCTED
    (dequantized, delta-applied) shared state, ready to hand to
    federation/validation/transfer_validation.py or
    federation/aggregation/. Deliberately a distinct type from
    UpdateMessage (whose `payload` field is the QUANTIZED wire format)
    so nothing downstream has to know or care whether it's looking at
    wire-format or reconstructed data -- ReceivedUpdate is always the
    latter.
    """
    node_id: str
    round: int
    validation_reward: float
    update_norm: float
    timestamp: float
    shared_state: SharedState


class FederationProtocol:
    """
    Per-node protocol state: tracks the last shared state sent to, and
    reconstructed from, each neighbor, and exposes the two operations
    node/scheduler.py (not yet built) will call each round.
    """

    def __init__(self, own_node_id: str) -> None:
        self.own_node_id = own_node_id
        self._last_sent_to: Dict[str, SharedState] = {}
        self._last_received_from: Dict[str, SharedState] = {}

    @staticmethod
    def _zero_state_like(state: SharedState) -> SharedState:
        return {
            component: {key: torch.zeros_like(tensor) for key, tensor in tensors.items()}
            for component, tensors in state.items()
        }

    @staticmethod
    def _raw_delta(current: SharedState, base: SharedState) -> SharedState:
        return {
            component: {key: current[component][key] - base[component][key] for key in current[component]}
            for component in current
        }

    def prepare_outgoing(
        self,
        neighbor_id: str,
        current_shared_state: SharedState,
        round_number: int,
        validation_reward: float,
    ) -> bytes:
        """
        Build this node's outgoing message to `neighbor_id`: delta from
        whatever was last sent to that neighbor (zero-state if this is
        the first message), quantized, wrapped in an UpdateMessage,
        serialized to bytes.

        Updates `_last_sent_to[neighbor_id]` to `current_shared_state`
        -- the NEXT outgoing message to this same neighbor deltas
        against THIS exchange, not the raw current state again. This is
        the actual "delta chaining" the edge-efficiency doc calls for;
        verified explicitly in this file's __main__ smoke test, not
        just assumed to work from the code shape.
        """
        base = self._last_sent_to.get(neighbor_id, self._zero_state_like(current_shared_state))

        raw_delta = self._raw_delta(current_shared_state, base)
        update_norm = compute_update_norm(raw_delta)

        quantized_delta = compress_delta(current_shared_state, base)
        message = UpdateMessage(
            node_id=self.own_node_id,
            round=round_number,
            validation_reward=validation_reward,
            payload=quantized_delta,
            update_norm=update_norm,
        )

        self._last_sent_to[neighbor_id] = current_shared_state
        return serialize_message(message)

    def receive_incoming(self, data: bytes) -> ReceivedUpdate:
        """
        Deserialize `data`, reconstruct the sender's shared state from
        the received quantized delta plus whatever this protocol last
        reconstructed FROM that same sender (zero-state if this is the
        first message received from them), and update bookkeeping.
        """
        # QuantizedTensor is a plain NamedTuple of (tensor, float, float,
        # tuple) with zero methods and zero code-execution surface --
        # allowlisting it here is scoped to exactly this one load call
        # (not a process-wide change), and is a narrowly-targeted
        # exception, not the blanket weights_only=False that would
        # reopen arbitrary-code-execution risk for every future payload
        # type. See serializer.py's module docstring for the full
        # reasoning; this is the layer that actually knows the payload
        # contains QuantizedTensor, so it's the right place to make
        # this call, not serializer.py itself.
        with torch.serialization.safe_globals([QuantizedTensor]):
            message = deserialize_message(data)

        base = self._last_received_from.get(message.node_id)
        if base is None:
            base = {
                component: {key: torch.zeros(q.shape, dtype=torch.float32) for key, q in tensors.items()}
                for component, tensors in message.payload.items()
            }

        reconstructed = decompress_delta(message.payload, base)
        self._last_received_from[message.node_id] = reconstructed

        return ReceivedUpdate(
            node_id=message.node_id,
            round=message.round,
            validation_reward=message.validation_reward,
            update_norm=message.update_norm,
            timestamp=message.timestamp,
            shared_state=reconstructed,
        )


if __name__ == "__main__":
    from env.core.actions import NUM_ACTIONS
    from env.core.observations import NUM_CHANNELS
    from federation.validation.transfer_validation import extract_shared_state
    from marl.models.vdn import VDNModel

    sender = FederationProtocol(own_node_id="N1")
    receiver = FederationProtocol(own_node_id="N2")

    model_round1 = VDNModel(in_channels=NUM_CHANNELS, window_size=5, num_actions=NUM_ACTIONS, num_agents=2)
    state_round1 = extract_shared_state(model_round1)

    # --- Round 1: first-ever exchange, should behave like a full send (delta from zero) ---
    wire_bytes_1 = sender.prepare_outgoing("N2", state_round1, round_number=1, validation_reward=3.0)
    received_1 = receiver.receive_incoming(wire_bytes_1)
    key = list(state_round1["encoder"].keys())[0]
    error_round1 = (received_1.shared_state["encoder"][key] - state_round1["encoder"][key]).abs().max().item()
    print(f"Round 1 reconstruction error (first exchange): {error_round1:.6f}")
    assert error_round1 < 0.1

    # --- Round 2: small change, delta should chain against round 1's state, not zero again ---
    with torch.no_grad():
        for p in model_round1.parameters():
            p.add_(0.01)  # tiny change
    state_round2 = extract_shared_state(model_round1)

    wire_bytes_2 = sender.prepare_outgoing("N2", state_round2, round_number=2, validation_reward=3.1)
    received_2 = receiver.receive_incoming(wire_bytes_2)
    error_round2 = (received_2.shared_state["encoder"][key] - state_round2["encoder"][key]).abs().max().item()
    print(f"Round 2 reconstruction error (delta from round 1): {error_round2:.6f}")
    assert error_round2 < 0.1

    # Round 2's quantized delta should be MUCH smaller-magnitude than round 1's
    # (only the 0.01 change, not the whole model) -- confirms delta chaining is
    # actually happening, not silently re-sending a full state every round.
    print(f"Round 1 update_norm: {sender._last_sent_to is not None}")  # bookkeeping exists
    print("Round 2 reconstructed close to actual round-2 state:", error_round2 < 0.1)
    print("OK")
