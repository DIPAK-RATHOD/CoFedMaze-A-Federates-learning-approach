"""
node_config.py

Loads data/nodeN/config.yaml plus the relevant configs/*.yaml files
(topology.yaml, coalition.yaml) and resolves them into ONE runtime
NodeConfig object -- the single place "which task variant am I, what
are my hyperparameters, where does my topology config live" all get
resolved together, since both services.py and scheduler.py need this
resolved view.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import yaml

PathLike = Union[str, Path]


@dataclass
class NodeConfig:
    node_id: str
    task_variant: str

    maze_rows: int
    maze_cols: int
    maze_algorithm: str
    window_size: int

    num_checkpoints: int
    num_obstacles: int
    num_key_door_pairs: int

    topology_path: str

    tau_form: float
    tau_break: float
    patience: int
    dwell_episodes: int
    max_coalition_size: int
    health_check_interval: int

    @classmethod
    def load(
        cls,
        node_config_path: PathLike,
        coalition_config_path: PathLike = "configs/coalition.yaml",
        topology_config_path: PathLike = "configs/topology.yaml",
    ) -> "NodeConfig":
        """
        Raises:
            FileNotFoundError: If node_config_path or
                coalition_config_path doesn't exist. (topology_path is
                only stored here, not read -- PhysicalGraph.from_yaml()
                is the one place that actually opens it, so a bad
                topology path fails there, not here.)
            KeyError: If a required key is missing from either config file.
        """
        node_config_path = Path(node_config_path)
        coalition_config_path = Path(coalition_config_path)
        if not node_config_path.exists():
            raise FileNotFoundError(f"Node config not found at {node_config_path}")
        if not coalition_config_path.exists():
            raise FileNotFoundError(f"Coalition config not found at {coalition_config_path}")

        node_cfg = yaml.safe_load(node_config_path.read_text())
        coalition_cfg = yaml.safe_load(coalition_config_path.read_text())

        maze = node_cfg["maze"]
        task_params = node_cfg.get("task_params", {})

        return cls(
            node_id=node_cfg["node_id"],
            task_variant=node_cfg["task_variant"],
            maze_rows=maze["rows"],
            maze_cols=maze["cols"],
            maze_algorithm=maze["algorithm"],
            window_size=maze["window_size"],
            num_checkpoints=task_params.get("num_checkpoints", 0),
            num_obstacles=task_params.get("num_obstacles", 0),
            num_key_door_pairs=task_params.get("num_key_door_pairs", 0),
            topology_path=str(topology_config_path),
            tau_form=coalition_cfg["tau_form"],
            tau_break=coalition_cfg["tau_break"],
            patience=coalition_cfg["patience"],
            dwell_episodes=coalition_cfg["dwell_episodes"],
            max_coalition_size=coalition_cfg["max_coalition_size"],
            health_check_interval=coalition_cfg["health_check_interval"],
        )


if __name__ == "__main__":
    config = NodeConfig.load("data/node1/config.yaml")
    print(config)
    assert config.node_id == "N1"
    assert config.task_variant == "checkpoints"
    assert config.num_checkpoints == 2
    assert config.tau_form == 0.50
    print("OK")
