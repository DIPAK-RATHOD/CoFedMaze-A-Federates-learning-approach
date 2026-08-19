"""
scheduler.py

Entry point for running a single node's training schedule across N rounds.
Orchestrates local VDN training, P2P state exchange via Transport/Protocol,
and coalition formation via CoalitionManager. Includes CLI argparse support,
live ASCII terminal rendering, and live Matplotlib graphical GUI rendering.
"""

import argparse
from pathlib import Path
import time
from typing import Callable, List, Optional
import yaml

from federation.communication.transport import InProcessTransport, TCPTransport, Transport
from federation.topology.physical_graph import PhysicalGraph
from federation.validation.transfer_validation import extract_shared_state
from knowledge_graph.model_similarity import compute_model_similarity
from node.node_config import NodeConfig
from node.services import NodeServices, build_services
from visualization.live_view import LiveTerminalView
from visualization.matplotlib_live_view import MatplotlibLiveView


class NodeScheduler:
    """
    Orchestrates one node's round-by-round execution flow:
      1. Run one local training episode (run_episode) + 1 gradient step (train_step).
      2. Save model checkpoint per round.
      3. Fast loop: send this node's shared state to physical neighbors.
      4. Fast loop: receive incoming states from physical neighbors.
      5. Coalition evaluation: update model similarities, evaluate coalitions.
    """

    def __init__(self, services: NodeServices, transport: Transport) -> None:
        self.services = services
        self.transport = transport
        self.round = 0
        self._known_states = {}
        self.active_coalition: List[str] = [services.config.node_id]

    def run_round(self, on_step: Optional[Callable[["LocalTrainer"], None]] = None) -> List[str]:
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
            if "encoder" in my_state and "encoder" in received.shared_state:
                ms = compute_model_similarity(my_state["encoder"], received.shared_state["encoder"])
                self.services.knowledge_updater.graph.update_edge(received.node_id, ms)

        # --- Coalition evaluation ---
        self.active_coalition = self.services.coalition_manager.evaluate_and_update(
            current_round=self.round,
            known_states=self._known_states,
            my_trajectory=trajectory,
        )

        # Periodic deterministic evaluation (epsilon = 0.0) every 10 rounds
        if self.round % 10 == 0:
            self.services.trainer.evaluate(current_round=self.round)

        return self.active_coalition


def main():
    parser = argparse.ArgumentParser(description="Run a single CoFedMaze node scheduler.")
    parser.add_argument("--node-id", type=str, required=True, help="Node ID (e.g. N1, N2, N3).")
    parser.add_argument("--config", type=Path, required=True, help="Path to node config YAML.")
    parser.add_argument("--topology", type=Path, default=Path("configs/topology.yaml"), help="Path to topology YAML.")
    parser.add_argument("--rounds", type=int, default=10, help="Number of training rounds.")
    parser.add_argument("--round-delay", type=float, default=2.0, help="Inter-round synchronization delay in seconds (default: 2.0s).")
    parser.add_argument("--mode", choices=["tcp", "sim"], default="tcp", help="Transport mode: tcp for distributed or sim for in-process.")
    parser.add_argument("--render", action="store_true", help="Enable live ASCII terminal rendering of maze and agent steps.")
    parser.add_argument("--gui", "--matplotlib", dest="gui", action="store_true", help="Enable live Matplotlib graphical GUI rendering window.")
    args = parser.parse_args()

    config = NodeConfig.load(args.config, topology_config_path=args.topology)
    config.node_id = args.node_id

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
    gui_view = MatplotlibLiveView(node_id=args.node_id, enabled=args.gui)

    def step_cb(trainer):
        if args.render:
            live_view.render_step(trainer.env, trainer.episode_count, trainer.total_env_steps)
        if args.gui:
            goal_reached = False
            if hasattr(trainer.env, "_exit_obj") and hasattr(trainer.env, "_agent_objs"):
                goal_reached = all(
                    trainer.env._exit_obj.is_usable_by(trainer.env.maze, trainer.env._agent_objs[aid].position)
                    for aid in trainer.env.possible_agents
                )
            step_count = getattr(trainer.env, "_step_count", 0)
            max_steps = getattr(trainer.env, "max_episode_steps", 100)

            gui_view.render_step(
                trainer.env,
                episode_count=trainer.episode_count,
                total_env_steps=trainer.total_env_steps,
                loss=trainer.last_loss,
                reward=step_count * -0.01 + (10.0 if goal_reached else 0.0),
                epsilon=trainer._epsilon_for_episode(trainer.episode_count),
                coalition=scheduler.active_coalition,
                goal_reached=goal_reached,
                timeout=step_count >= max_steps and not goal_reached,
            )

    services.trainer.print_configuration_summary(federation_enabled=(args.mode == "tcp"), alpha=0.25)
    print(f"[{args.node_id}] Starting node execution for {args.rounds} rounds (inter-round delay: {args.round_delay:.1f}s)...")
    for r in range(args.rounds):
        coalition = scheduler.run_round(on_step=step_cb if (args.render or args.gui) else None)
        print(f"[{args.node_id}] Round {r+1}/{args.rounds} complete | Active Coalition: {coalition}")
        if r < args.rounds - 1 and args.round_delay > 0:
            time.sleep(args.round_delay)

    if args.mode == "tcp":
        transport.close()
    if args.gui:
        gui_view.close()

    # Automatically generate evaluation summary & visualization PNGs upon training completion
    from visualization.auto_evaluator import generate_node_evaluation_report
    generate_node_evaluation_report(args.node_id)


if __name__ == "__main__":
    main()
