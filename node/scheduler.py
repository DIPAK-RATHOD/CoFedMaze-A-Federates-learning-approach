"""
scheduler.py

Implements the event schedule for ONE node: each round runs a local
training episode (marl/training/trainer.py, already built/tested),
then the federation fast-loop steps -- quantized exchange with
physical neighbors, a small-subset transfer-benefit test, the KS-bar
update, and a coalition candidate/health check.
"""

import argparse
from pathlib import Path
import time
from typing import Callable, List, Optional
import yaml

from env.core.constants import AGENT_A
from federation.communication.transport import InProcessTransport, TCPTransport, Transport
from federation.topology.physical_graph import PhysicalGraph
from federation.validation.transfer_validation import compute_transfer_benefit, extract_shared_state
from node.node_config import NodeConfig
from node.services import NodeServices, build_services
from visualization.auto_evaluator import generate_node_evaluation_report
from visualization.live_view import LiveTerminalView

DEFAULT_VALIDATION_SEEDS: List[int] = [101, 102, 103]


class NodeScheduler:
    """Drives one node's per-round event schedule."""

    def __init__(
        self,
        services: NodeServices,
        transport: Transport,
        validation_seeds: Optional[List[int]] = None,
    ) -> None:
        self.services = services
        self.transport = transport
        self.validation_seeds = validation_seeds if validation_seeds is not None else DEFAULT_VALIDATION_SEEDS
        self.round = 0
        self._known_states = {}

    def run_round(self, on_step: Optional[Callable] = None) -> None:
        """One full round: local training, then the federation fast-loop steps."""
        self.round += 1

        # --- Local learning ---
        trajectory = self.services.trainer.run_episode(on_step=on_step)
        self.services.trainer.buffer.add(trajectory)
        self.services.trainer.train_step()

        # Save checkpoint per round
        if self.services.trainer.checkpoint_dir is not None:
            self.services.trainer.save_checkpoint(self.services.trainer.checkpoint_dir)

        # --- Fast loop: send this node's shared state to every physical neighbor ---
        my_state = extract_shared_state(self.services.trainer.online_model)
        self._known_states[self.services.config.node_id] = my_state
        for neighbor_id in self.services.physical_graph.neighbors(self.services.config.node_id):
            wire_bytes = self.services.protocol.prepare_outgoing(
                neighbor_id, my_state, round_number=self.round,
                validation_reward=trajectory.total_reward(),
            )
            self.transport.send(neighbor_id, wire_bytes)

        # --- Fast loop: process everything received this round ---
        incoming = self.transport.receive_all(self.services.config.node_id)
        for data in incoming:
            received = self.services.protocol.receive_incoming(data)
            self._known_states[received.node_id] = received.shared_state

            tb_result = compute_transfer_benefit(
                self.services.trainer.online_model, received.shared_state,
                self.services.env, self.validation_seeds,
            )

            ks_bar = self.services.knowledge_updater.update(
                neighbor_id=received.node_id,
                tb_result=tb_result,
                own_encoder_state=my_state["encoder"],
                neighbor_task_features=self.services.knowledge_updater.own_task_features,
                neighbor_encoder_state=received.shared_state["encoder"],
                neighbor_message_size_bytes=len(data),
            )

            member_weights = {m: 1.0 for m in self._known_states}
            member_weights[received.node_id] = ks_bar
            self.services.coalition_manager.step(
                coalition_model=self.services.trainer.online_model,
                member_shared_states=self._known_states,
                member_weights=member_weights,
                env=self.services.env,
                validation_seeds=self.validation_seeds,
            )


def main():
    parser = argparse.ArgumentParser(description="Run a single CoFedMaze node scheduler.")
    parser.add_argument("--node-id", type=str, required=True, help="Node ID (e.g. N1, N2, N3).")
    parser.add_argument("--config", type=Path, required=True, help="Path to node config YAML.")
    parser.add_argument("--topology", type=Path, default=Path("configs/topology.yaml"), help="Path to topology YAML.")
    parser.add_argument("--rounds", type=int, default=10, help="Number of training rounds.")
    parser.add_argument("--round-delay", type=float, default=2.0, help="Inter-round synchronization delay in seconds (default: 2.0s).")
    parser.add_argument("--mode", choices=["tcp", "sim"], default="tcp", help="Transport mode: tcp for distributed or sim for in-process.")
    parser.add_argument("--render", action="store_true", help="Enable live terminal rendering of maze and agent steps.")
    args = parser.parse_args()

    config = NodeConfig.load(args.config, topology_config_path=args.topology)

    if args.mode == "tcp":
        topo_data = yaml.safe_load(args.topology.read_text(encoding="utf-8"))
        raw_addresses = topo_data.get("addresses", {})
        if not raw_addresses or args.node_id not in raw_addresses:
            raise ValueError(f"Topology config {args.topology} must contain addresses mapping for {args.node_id}")

        addresses = {nid: (info["host"], int(info["port"])) for nid, info in raw_addresses.items()}
        bind_address = addresses[args.node_id]

        print(f"[{args.node_id}] Initializing TCPTransport on {bind_address[0]}:{bind_address[1]}...")
        transport = TCPTransport(
            own_node_id=args.node_id,
            bind_address=bind_address,
            peer_addresses=addresses,
        )
    else:
        transport = InProcessTransport()
        transport.register(args.node_id)
        physical_graph = PhysicalGraph.from_yaml(args.topology)
        for nbr in physical_graph.neighbors(args.node_id):
            transport.register(nbr)

    services = build_services(config, transport)
    scheduler = NodeScheduler(services, transport)
    live_view = LiveTerminalView(node_id=args.node_id, mode=args.mode, enabled=args.render)

    coalition_history = []
    print(f"[{args.node_id}] Starting node execution for {args.rounds} rounds (inter-round delay: {args.round_delay}s)...")

    try:
        for r in range(1, args.rounds + 1):
            def step_cb(t):
                current_members = sorted(services.coalition_manager.members)
                live_view.update(t, current_round=r, total_rounds=args.rounds, coalition_members=current_members)

            scheduler.run_round(on_step=step_cb if args.render else None)
            members = sorted(services.coalition_manager.members)
            coalition_history.append({"round": r, "coalitions": {args.node_id: members}})
            print(f"[{args.node_id}] Round {r}/{args.rounds} complete | Active Coalition: {members}")
            if args.round_delay > 0 and r < args.rounds:
                time.sleep(args.round_delay)
    finally:
        if isinstance(transport, TCPTransport):
            transport.close()

    # Automatically generate post-training plots and report in node's evaluation directory
    eval_dir = generate_node_evaluation_report(
        node_id=args.node_id,
        coalition_history=coalition_history,
        env=services.env,
        log_dir=services.step_logger.log_dir,
    )
    print(f"\n[{args.node_id}] All evaluation plots & dashboard saved to: {eval_dir}")


if __name__ == "__main__":
    main()
