"""
services.py

Container object (NodeServices) + factory function (build_services) that
instantiates and wires together every single component one node needs to run:

  - CoFedMazeParallelEnv
  - StepLogger
  - LocalTrainer (VDN)
  - PhysicalGraph
  - FederationProtocol
  - TaskFeatures
  - KnowledgeGraphUpdater
  - CoalitionManager

Services is a pure dependency-injection container -- it holds references to all these
objects so node/scheduler.py can drive them without needing to know how to construct them.
"""

from dataclasses import dataclass
from typing import Optional

from env.core.observations import NUM_CHANNELS
from env.wrappers.pettingzoo_env import CoFedMazeParallelEnv
from env.core.actions import NUM_ACTIONS

from federation.communication.protocol import FederationProtocol
from federation.communication.transport import Transport
from federation.topology.physical_graph import PhysicalGraph
from knowledge_graph.task_similarity import TaskFeatures
from knowledge_graph.updater import KnowledgeGraphUpdater
from marl.training.trainer import LocalTrainer
from coalition.coalition_manager import CoalitionManager
from node.node_config import NodeConfig
from utils.logger import StepLogger


@dataclass
class NodeServices:
    config: NodeConfig
    env: CoFedMazeParallelEnv
    trainer: LocalTrainer
    physical_graph: PhysicalGraph
    protocol: FederationProtocol
    knowledge_updater: KnowledgeGraphUpdater
    coalition_manager: CoalitionManager
    step_logger: StepLogger


def build_services(config: NodeConfig, transport: Transport) -> NodeServices:
    """Instantiate and wire up all per-node services."""
    env = CoFedMazeParallelEnv(
        rows=config.maze_rows,
        cols=config.maze_cols,
        algorithm=config.maze_algorithm,
        window_size=config.window_size,
        num_checkpoints=config.num_checkpoints,
        num_obstacles=config.num_obstacles,
        num_key_door_pairs=config.num_key_door_pairs,
    )

    log_dir = f"state/{config.node_id.lower()}/logs"
    step_logger = StepLogger(log_dir=log_dir, node_id=config.node_id)
    checkpoint_dir = f"state/{config.node_id.lower()}/checkpoints"

    trainer = LocalTrainer(
        env=env,
        in_channels=NUM_CHANNELS,
        num_actions=NUM_ACTIONS,
        step_logger=step_logger,
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
    coalition_manager.set_trainer(trainer)

    return NodeServices(
        config=config, env=env, trainer=trainer, physical_graph=physical_graph,
        protocol=protocol, knowledge_updater=knowledge_updater, coalition_manager=coalition_manager,
        step_logger=step_logger,
    )


if __name__ == "__main__":
    from federation.communication.transport import InProcessTransport
    from node.node_config import NodeConfig

    config = NodeConfig.load("data/node1/config.yaml")
    transport = InProcessTransport()
    services = build_services(config, transport)
    print("NodeServices created successfully for node:", services.config.node_id)
    assert services.config.node_id == "N1"
    print("OK")
