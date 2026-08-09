"""
transport.py

Low-level P2P send/receive. This implementation is an IN-PROCESS
SIMULATION -- all "nodes" run as Python objects in one process, and
"sending" a message means directly enqueueing bytes into the
recipient's inbox, no real sockets involved. This matches the
workplan's own phased plan: a development-workstation software
simulation of all 5 virtual nodes BEFORE Raspberry Pi hardware
deployment (Implementation Requirements doc, Sec 1.1) -- this file is
that simulation phase, not a placeholder standing in for "real work
later."

Swapping to real networking later (ZeroMQ, per federation.yaml) means
implementing a new class satisfying the same Transport interface below.
Callers (protocol.py, node/scheduler.py -- not yet built) interact only
through send()/receive()/pending_count(), never through this file's
internals, so that swap should not require touching any calling code.
"""

from abc import ABC, abstractmethod
from collections import deque
import socket
import struct
import threading
import time
from typing import Deque, Dict, List, Mapping, Optional, Tuple


class Transport(ABC):
    """Abstract interface every transport implementation (simulated or real) must satisfy."""

    @abstractmethod
    def send(self, to_node: str, data: bytes) -> None:
        """Send raw bytes to `to_node`. Must not block indefinitely."""
        raise NotImplementedError

    @abstractmethod
    def receive(self, node_id: str) -> Optional[bytes]:
        """
        Pop and return the next pending message addressed to
        `node_id`, or None if nothing is waiting. A non-blocking poll,
        not a blocking receive -- matches the fast-loop design where a
        node should never stall waiting on a neighbor that hasn't sent
        anything this round.
        """
        raise NotImplementedError

    @abstractmethod
    def pending_count(self, node_id: str) -> int:
        """How many messages are currently queued for `node_id`."""
        raise NotImplementedError

    @abstractmethod
    def receive_all(self, node_id: str) -> List[bytes]:
        """Drain every currently pending message for `node_id` in FIFO order."""
        raise NotImplementedError


class InProcessTransport(Transport):
    """
    Simulated P2P transport: a shared registry of per-node inboxes, all
    living in this one Python process. Every "node" that wants to send
    or receive must first register() with the SAME InProcessTransport
    instance -- the stand-in here for "nodes on the same physical LAN"
    is one shared object, not actual network discovery.
    """

    def __init__(self) -> None:
        self._inboxes: Dict[str, Deque[bytes]] = {}

    def register(self, node_id: str) -> None:
        """
        Create an inbox for `node_id` if one doesn't already exist.
        Idempotent: calling this again for an already-registered node
        is a harmless no-op (e.g. a simulated node "restarting"
        shouldn't need special-case handling here).
        """
        self._inboxes.setdefault(node_id, deque())

    def is_registered(self, node_id: str) -> bool:
        return node_id in self._inboxes

    def send(self, to_node: str, data: bytes) -> None:
        """
        Raises:
            ValueError: If `to_node` was never register()'ed. Sending
                to an unknown node is almost certainly a bug (typo'd
                node id, or a node that hasn't started yet) -- silently
                dropping the message would hide that bug rather than
                surface it.
        """
        self._require_registered(to_node)
        self._inboxes[to_node].append(data)

    def receive(self, node_id: str) -> Optional[bytes]:
        """
        Raises:
            ValueError: If `node_id` was never register()'ed.
        """
        self._require_registered(node_id)
        inbox = self._inboxes[node_id]
        return inbox.popleft() if inbox else None

    def receive_all(self, node_id: str) -> List[bytes]:
        """Drain every currently-pending message for `node_id` at once, in arrival (FIFO) order."""
        self._require_registered(node_id)
        inbox = self._inboxes[node_id]
        drained = list(inbox)
        inbox.clear()
        return drained

    def pending_count(self, node_id: str) -> int:
        self._require_registered(node_id)
        return len(self._inboxes[node_id])

    def _require_registered(self, node_id: str) -> None:
        if node_id not in self._inboxes:
            raise ValueError(f"Unknown node {node_id!r} -- must register() before send/receive")


class TCPTransport(Transport):
    """Server-capable transport using one short-lived TCP connection per message.

    Each node process creates one instance bound to its own address and supplies
    the fixed ``node_id -> (host, port)`` address map for its physical peers.
    The listener thread accepts a length-prefixed byte payload and queues it;
    ``receive``/``receive_all`` remain non-blocking just like the simulation
    transport, so NodeScheduler's fast loop does not change.
    """

    _FRAME_HEADER = struct.Struct("!I")
    _MAX_MESSAGE_BYTES = 64 * 1024 * 1024

    def __init__(
        self,
        own_node_id: str,
        bind_address: Tuple[str, int],
        peer_addresses: Mapping[str, Tuple[str, int]],
        connect_timeout_s: float = 1.0,
        max_retries: int = 3,
        retry_delay_s: float = 0.5,
    ) -> None:
        if own_node_id not in peer_addresses:
            raise ValueError("peer_addresses must include this node's own address")
        peer_host, peer_port = peer_addresses[own_node_id]
        bind_host, bind_port = bind_address
        if bind_port != peer_port:
            raise ValueError(f"bind_address port ({bind_port}) must match peer_addresses port ({peer_port})")
        if connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be positive")
        self.own_node_id = own_node_id
        self._peers = dict(peer_addresses)
        self._connect_timeout_s = connect_timeout_s
        self.max_retries = max_retries
        self.retry_delay_s = retry_delay_s
        self._inbox: Deque[bytes] = deque()
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._listener.bind(bind_address)
        except OSError:
            # Fallback to binding all network interfaces ("0.0.0.0", port) if Linux rejects explicit IP binding
            self._listener.bind(("0.0.0.0", bind_port))
        self._listener.listen()
        self._listener.settimeout(0.2)
        self._thread = threading.Thread(target=self._listen, name=f"cofedmaze-{own_node_id}-transport", daemon=True)
        self._thread.start()

    def send(self, to_node: str, data: bytes) -> None:
        if self._closed.is_set():
            raise RuntimeError("TCPTransport is closed")
        if to_node not in self._peers:
            raise ValueError(f"Unknown peer {to_node!r}")
        if len(data) > self._MAX_MESSAGE_BYTES:
            raise ValueError(f"Message exceeds {self._MAX_MESSAGE_BYTES} byte transport limit")
        
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with socket.create_connection(self._peers[to_node], timeout=self._connect_timeout_s) as connection:
                    connection.sendall(self._FRAME_HEADER.pack(len(data)) + data)
                    return  # Successfully sent
            except OSError as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_s)

        # Log warning if peer is not reachable yet, but do not crash process
        print(f"[{self.own_node_id}] Warning: Peer {to_node} at {self._peers[to_node]} not reachable this round ({last_error}). Continuing...")

    def receive(self, node_id: str) -> Optional[bytes]:
        self._require_own_node(node_id)
        with self._lock:
            return self._inbox.popleft() if self._inbox else None

    def receive_all(self, node_id: str) -> List[bytes]:
        self._require_own_node(node_id)
        with self._lock:
            messages = list(self._inbox)
            self._inbox.clear()
            return messages

    def pending_count(self, node_id: str) -> int:
        self._require_own_node(node_id)
        with self._lock:
            return len(self._inbox)

    def close(self) -> None:
        """Stop the listener thread and release this node's server socket."""
        if self._closed.is_set():
            return
        self._closed.set()
        self._listener.close()
        self._thread.join(timeout=1.0)

    def _require_own_node(self, node_id: str) -> None:
        if node_id != self.own_node_id:
            raise ValueError(f"TCPTransport for {self.own_node_id!r} cannot receive for {node_id!r}")

    def _listen(self) -> None:
        while not self._closed.is_set():
            try:
                connection, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with connection:
                connection.settimeout(self._connect_timeout_s)
                try:
                    header = self._read_exact(connection, self._FRAME_HEADER.size)
                    (length,) = self._FRAME_HEADER.unpack(header)
                    if length > self._MAX_MESSAGE_BYTES:
                        continue
                    payload = self._read_exact(connection, length)
                except (OSError, ValueError):
                    continue
                with self._lock:
                    self._inbox.append(payload)

    @staticmethod
    def _read_exact(connection: socket.socket, size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            chunk = connection.recv(size - len(chunks))
            if not chunk:
                raise ValueError("connection closed before complete message was received")
            chunks.extend(chunk)
        return bytes(chunks)


if __name__ == "__main__":
    from env.core.actions import NUM_ACTIONS
    from env.core.observations import NUM_CHANNELS
    from federation.communication.protocol import FederationProtocol
    from federation.validation.transfer_validation import extract_shared_state
    from marl.models.vdn import VDNModel

    transport = InProcessTransport()
    transport.register("N1")
    transport.register("N2")

    protocol_n1 = FederationProtocol(own_node_id="N1")
    protocol_n2 = FederationProtocol(own_node_id="N2")

    model = VDNModel(in_channels=NUM_CHANNELS, window_size=5, num_actions=NUM_ACTIONS, num_agents=2)
    state = extract_shared_state(model)

    wire_bytes = protocol_n1.prepare_outgoing("N2", state, round_number=1, validation_reward=1.0)
    transport.send("N2", wire_bytes)

    assert transport.pending_count("N2") == 1
    incoming = transport.receive("N2")
    assert incoming is not None
    received = protocol_n2.receive_incoming(incoming)
    assert received.node_id == "N1"
    assert transport.pending_count("N2") == 0

    print("OK")
