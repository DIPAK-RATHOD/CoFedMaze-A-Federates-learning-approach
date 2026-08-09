"""
run_simulation.py

Dev-workstation entry point that spins up all 5 virtual nodes (N1-N5)
before real hardware deployment, per the Implementation Requirements
doc's explicit two-phase plan (software simulation first, Pi hardware
second). Each node imports from node/, federation/, knowledge_graph/,
coalition/, marl/ -- this script itself contains no logic beyond
wiring 5 NodeSchedulers to one shared InProcessTransport and stepping
them together in lockstep, per the scripts/ package's stated
philosophy: "thin CLI entry points... imports from the real subpackages
rather than containing logic itself."

Lockstep, not truly concurrent: every node runs its round_run() call
sequentially within one Python process (matching InProcessTransport's
own single-process simulation -- see that file's docstring). This is
faithful to the workplan's "software simulation" phase; genuine
concurrency only becomes meaningful once transport.py has a real
networking implementation (separate physical/OS processes), which
doesn't exist yet.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

from federation.communication.transport import InProcessTransport
from node.node_config import NodeConfig
from node.scheduler import NodeScheduler
from node.services import NodeServices, build_services

NODE_IDS = ["N1", "N2", "N3", "N4", "N5"]
DATA_DIR = Path("data")


def build_all_nodes(master_seed: int = 0) -> Dict[str, NodeScheduler]:
    """
    Load every node's data/nodeN/config.yaml, build its NodeServices
    against one SHARED InProcessTransport, and wrap each in a
    NodeScheduler.

    Raises:
        FileNotFoundError: Propagated from NodeConfig.load() if any
            node's config.yaml is missing.
        ValueError: If window_size differs across any two nodes.
            window_size determines the shared encoder's architecture
            (its flattened FC layer scales with window_size^2) -- two
            nodes with different window_size have INCOMPATIBLE encoder
            shapes, and weight exchange between them crashes with a
            state_dict mismatch the first time either tries to load
            the other's shared state. This was discovered the hard way
            while building this exact function (N4's original config
            used window_size=3 while every other node used 5) -- caught
            here, once, before any node runs, rather than deep inside a
            training round's transfer-benefit test.
    """
    configs: Dict[str, NodeConfig] = {}
    for node_id in NODE_IDS:
        config_path = DATA_DIR / node_id.lower().replace("n", "node") / "config.yaml"
        configs[node_id] = NodeConfig.load(config_path)

    window_sizes = {node_id: cfg.window_size for node_id, cfg in configs.items()}
    distinct_sizes = set(window_sizes.values())
    if len(distinct_sizes) > 1:
        raise ValueError(
            f"All nodes must use the same window_size for shared-encoder weight exchange "
            f"to be architecturally valid, but got mismatched values: {window_sizes}. "
            f"See data/node4/config.yaml's comment for the full explanation of why this "
            f"matters."
        )

    transport = InProcessTransport()
    schedulers: Dict[str, NodeScheduler] = {}
    for i, node_id in enumerate(NODE_IDS):
        services = build_services(configs[node_id], transport, master_seed=master_seed + i)
        schedulers[node_id] = NodeScheduler(services, transport)

    return schedulers


def run(num_rounds: int, master_seed: int = 0, verbose: bool = True) -> Dict[str, NodeScheduler]:
    """
    Run all 5 nodes for `num_rounds`, one round at a time, in a fixed
    node order each round (N1..N5) -- matching the ring's own N1-N2-
    N3-N4-N5-N1 ordering, though the actual per-round work each node
    does is independent of this ordering (every node sends before any
    node in that round reads its inbox is NOT guaranteed -- see the
    note in the loop below).
    """
    schedulers = build_all_nodes(master_seed=master_seed)

    for round_num in range(1, num_rounds + 1):
        for node_id in NODE_IDS:
            # NOTE: because InProcessTransport delivers synchronously
            # and NodeScheduler.run_round() both sends AND immediately
            # drains its inbox within one call, a node processes
            # whatever arrived from EARLIER-in-this-round senders
            # immediately, but messages from LATER-in-this-round
            # senders won't be seen until next round. This is a real,
            # order-dependent asymmetry in a single-process lockstep
            # simulation -- not a bug, but worth knowing: N1 (first in
            # NODE_IDS) never sees same-round messages from N5 until
            # round N+1, while N5 (last) sees same-round messages from
            # everyone before it immediately. A true concurrent/real
            # transport would not have this asymmetry.
            schedulers[node_id].run_round()

        if verbose:
            summary = ", ".join(
                f"{nid}:{sorted(schedulers[nid].services.coalition_manager.members)}"
                for nid in NODE_IDS
            )
            print(f"Round {round_num}: {summary}")

    return schedulers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CoFedMaze 5-node software simulation.")
    parser.add_argument("--rounds", type=int, default=10, help="Number of rounds to simulate (default: 10).")
    parser.add_argument("--seed", type=int, default=0, help="Master seed (default: 0).")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    schedulers = run(num_rounds=args.rounds, master_seed=args.seed)

    print("\nFinal coalition membership across all 5 nodes:")
    for node_id in NODE_IDS:
        members = sorted(schedulers[node_id].services.coalition_manager.members)
        print(f"  {node_id}: {members}")
