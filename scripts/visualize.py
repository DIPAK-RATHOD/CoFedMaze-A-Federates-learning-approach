"""Generate report figures from CoFedMaze configuration and training history.

Examples::

    python -m scripts.visualize --output-dir outputs/figures
    python -m scripts.visualize --node-config data/node1/config.yaml --maze-seed 42 \
        --history outputs/node1_history.json --dashboard
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from env.core.maze import Maze
from env.generator.generator_factory import create_generator
from federation.topology.physical_graph import PhysicalGraph
from node.node_config import NodeConfig
from visualization.dashboard import create_dashboard
from visualization.loss_curve import plot_loss_curve
from visualization.maze_plot import save_maze_plot
from visualization.reward_curve import plot_reward_curve
from visualization.topology_plot import plot_topology


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate CoFedMaze report figures.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"), help="Figure output directory.")
    parser.add_argument("--topology-config", type=Path, default=Path("configs/topology.yaml"), help="Physical topology YAML path.")
    parser.add_argument("--node-config", type=Path, default=None, help="Optional node config for a maze figure.")
    parser.add_argument("--maze-seed", type=int, default=None, help="Maze seed; requires --node-config.")
    parser.add_argument("--history", type=Path, default=None, help="JSON array saved from LocalTrainer.run().")
    parser.add_argument("--dashboard", action="store_true", help="Combine generated figures into dashboard.png.")
    return parser


def _load_history(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Training history not found at {path}")
    history = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(history, list):
        raise ValueError("training history JSON must be an array")
    return history


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.node_config is None) != (args.maze_seed is None):
        raise ValueError("--node-config and --maze-seed must be supplied together")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    panels: dict[str, Path] = {}
    topology_path = args.output_dir / "physical_topology.png"
    plot_topology(PhysicalGraph.from_yaml(args.topology_config), topology_path)
    panels["Physical topology"] = topology_path

    if args.node_config is not None:
        config = NodeConfig.load(args.node_config)
        maze = Maze(rows=config.maze_rows, cols=config.maze_cols, random_seed=args.maze_seed)
        create_generator(config.maze_algorithm, random_seed=args.maze_seed).generate(maze)
        maze_path = args.output_dir / f"{config.node_id.lower()}_maze_seed_{args.maze_seed}.png"
        save_maze_plot(maze, maze_path)
        panels["Maze"] = maze_path

    if args.history is not None:
        history = _load_history(args.history)
        reward_path = args.output_dir / "reward_curve.png"
        loss_path = args.output_dir / "loss_curve.png"
        plot_reward_curve(history, reward_path)
        plot_loss_curve(history, loss_path)
        panels["Training reward"] = reward_path
        panels["Training loss"] = loss_path

    if args.dashboard:
        dashboard_path = args.output_dir / "dashboard.png"
        create_dashboard(panels, dashboard_path)
        print(f"Wrote dashboard to {dashboard_path}")
    for name, path in panels.items():
        print(f"Wrote {name.lower()} figure to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
