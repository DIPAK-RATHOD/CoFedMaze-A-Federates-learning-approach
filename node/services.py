"""
services.py

Wires together the concrete instances (env, trainer, federation
protocol, knowledge graph updater, coalition manager) that
scheduler.py operates on for ONE node -- keeping dependency
construction/wiring separate from the scheduling logic itself.
"""

from dataclasses import dataclass
from pathlib import Path

from coalition.coalition_manager import CoalitionManager
from env.core.actions import NUM_ACTIONS
from env.core.observations import NUM_CHANNELS
from env.wrappers.pettingzoo_env import CoFedMazeParallelEnv
from federation.communication.protocol import FederationProtocol
from federation.communication.transport import Transport
from federation.topology.physical_graph import PhysicalGraph
from knowledge_graph.task_similarity import TaskFeatures
from knowledge_graph.updater import KnowledgeGraphUpdater
from marl.training.trainer import LocalTrainer
from node.node_config import NodeConfig
from utils.logger import StepLogger


@dataclass
class NodeServices:
    """Every constructed runtime object one node needs, bundled together."""
    config: NodeConfig
    env: CoFedMazeParallelEnv
    trainer: LocalTrainer
    physical_graph: PhysicalGraph
    protocol: FederationProtocol
    knowledge_updater: KnowledgeGraphUpdater
    coalition_manager: CoalitionManager
    step_logger: StepLogger


def build_services(config: NodeConfig, transport: Transport, master_seed: int = None) -> NodeServices:
    """
    Construct every runtime object for one node from its resolved
    NodeConfig.
    """
    env = CoFedMazeParallelEnv(
        rows=config.maze_rows, cols=config.maze_cols, algorithm=config.maze_algorithm,
        window_size=config.window_size, num_checkpoints=config.num_checkpoints,
        num_obstacles=config.num_obstacles, num_key_door_pairs=config.num_key_door_pairs,
    )

    node_dir_name = config.node_id.lower().replace("n", "node")
    log_dir = Path("state") / node_dir_name / "logs"
    checkpoint_dir = Path("state") / node_dir_name / "checkpoints"

    step_logger = StepLogger(log_dir=log_dir, node_id=config.node_id)

    trainer = LocalTrainer(
        env=env, in_channels=NUM_CHANNELS, num_actions=NUM_ACTIONS,
        master_seed=master_seed, step_logger=step_logger,
        checkpoint_dir=checkpoint_dir, auto_resume=True,
    )

    physical_graph = PhysicalGraph.from_yaml(config.topology_path)
    register = getattr(transport, "register", None)
    if register is not None:
        register(config.node_id)
    protocol = FederationProtocol(own_node_id=config.node_id)

    own_task_features = TaskFeatures.from_env(env)
    knowledge_updater = KnowledgeGraphUpdater(
        own_node_id=config.node_id,
        physical_graph=physical_graph,
        own_task_features=own_task_features,
        tau_form=config.tau_form,
        tau_break=config.tau_break,
    )

    coalition_manager = CoalitionManager(
        own_node_id=config.node_id,
        directed_graph=knowledge_updater.graph,
        patience=config.patience,
        dwell_episodes=config.dwell_episodes,
        health_check_interval=config.health_check_interval,
    )

    return NodeServices(
        config=config, env=env, trainer=trainer, physical_graph=physical_graph,
        protocol=protocol, knowledge_updater=knowledge_updater, coalition_manager=coalition_manager,
        step_logger=step_logger,
    )


if __name__ == "__main__":
    from federation.communication.transport import InProcessTransport
    from node.node_config import NodeConfig

    transport = InProcessTransport()
    config = NodeConfig.load("data/node1/config.yaml")
    services = build_services(config, transport, master_seed=0)

    same_graph = services.coalition_manager.directed_graph is services.knowledge_updater.graph
    assert same_graph
    print("node/services.py self-test OK")
