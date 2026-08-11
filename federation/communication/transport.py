"""
transport.py

Low-level P2P send/receive. Supports both in-process simulation (InProcessTransport)
and multi-machine TCP socket networking (TCPTransport).
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
        Pop and return the next pending message addressed to `node_id`,
        or None if nothing is waiting.
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
    living in this one Python process.
    """

    def __init__(self) -> None:
        self._inboxes: Dict[str, Deque[bytes]] = {}

    def register(self, node_id: str) -> None:
        self._inboxes.setdefault(node_id, deque())

    def is_registered(self, node_id: str) -> bool:
        return node_id in self._inboxes

    def send(self, to_node: str, data: bytes) -> None:
        if not self.is_registered(to_node):
            return
        self._inboxes[to_node].append(data)

    def receive(self, node_id: str) -> Optional[bytes]:
        if not self.is_registered(node_id):
            return None
        inbox = self._inboxes[node_id]
        return inbox.popleft() if inbox else None

    def receive_all(self, node_id: str) -> List[bytes]:
        if not self.is_registered(node_id):
            return []
        inbox = self._inboxes[node_id]
        drained = list(inbox)
        inbox.clear()
        return drained

    def pending_count(self, node_id: str) -> int:
        if not self.is_registered(node_id):
            return 0
        return len(self._inboxes[node_id])


class TCPTransport(Transport):
    """Server-capable transport using one short-lived TCP connection per message.

    Each node process creates one instance bound to its own address and supplies
    the fixed ``node_id -> (host, port)`` address map for its physical peers.
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
                    return
            except OSError as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_s)

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
        pass

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
