"""Train one CoFedMaze node locally and persist artifacts for evaluation.

Example::

    python -m scripts.train_local --node-config data/node1/config.yaml --episodes 200 --render
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from env.core.actions import NUM_ACTIONS
from env.core.observations import NUM_CHANNELS
from env.wrappers.pettingzoo_env import CoFedMazeParallelEnv
from marl.training.trainer import LocalTrainer
from node.node_config import NodeConfig
from utils.logger import StepLogger
from visualization.auto_evaluator import generate_node_evaluation_report
from visualization.live_view import LiveTerminalView


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one CoFedMaze node without federation.")
    parser.add_argument("--node-config", type=Path, required=True, help="Path to data/nodeN/config.yaml.")
    parser.add_argument("--episodes", type=int, default=200, help="Training episodes (default: 200).")
    parser.add_argument("--seed", type=int, default=0, help="Trainer master seed (default: 0).")
    parser.add_argument("--max-episode-steps", type=int, default=100, help="Environment timeout per episode (default: 100).")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Checkpoint directory; defaults to state/nodeN/checkpoints.")
    parser.add_argument("--history-output", type=Path, default=None, help="History JSON path; defaults to outputs/nodeN/training_history.json.")
    parser.add_argument("--render", action="store_true", help="Enable live terminal rendering of maze and agent steps.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-episode training output.")
    return parser


def _default_checkpoint_dir(config: NodeConfig) -> Path:
    return Path("state") / config.node_id.lower().replace("n", "node", 1) / "checkpoints"


def _default_history_output(config: NodeConfig) -> Path:
    return Path("outputs") / config.node_id.lower().replace("n", "node", 1) / "training_history.json"


def build_environment(config: NodeConfig, max_episode_steps: int) -> CoFedMazeParallelEnv:
    return CoFedMazeParallelEnv(
        rows=config.maze_rows, cols=config.maze_cols, algorithm=config.maze_algorithm,
        window_size=config.window_size, max_episode_steps=max_episode_steps,
        num_checkpoints=config.num_checkpoints, num_obstacles=config.num_obstacles,
        num_key_door_pairs=config.num_key_door_pairs,
    )


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if args.max_episode_steps < 1:
        raise ValueError("--max-episode-steps must be positive")

    config = NodeConfig.load(args.node_config)
    checkpoint_dir = args.checkpoint_dir or _default_checkpoint_dir(config)
    node_dir_name = config.node_id.lower().replace("n", "node", 1)
    log_dir = Path("state") / node_dir_name / "logs"
    step_logger = StepLogger(log_dir=log_dir, node_id=config.node_id)

    env = build_environment(config, args.max_episode_steps)
    trainer = LocalTrainer(
        env=env,
        in_channels=NUM_CHANNELS,
        num_actions=NUM_ACTIONS,
        master_seed=args.seed,
        step_logger=step_logger,
        checkpoint_dir=checkpoint_dir,
        auto_resume=True,
    )

    live_view = LiveTerminalView(node_id=config.node_id, mode="local", enabled=args.render)

    history = trainer.run(
        args.episodes,
        verbose=not args.quiet,
        on_step=lambda t: live_view.update(t, current_round=1, total_rounds=1) if args.render else None,
    )

    trainer.save_checkpoint(
        checkpoint_dir,
        metadata={"node_id": config.node_id, "task_variant": config.task_variant, "master_seed": args.seed},
    )
    history_path = args.history_output or _default_history_output(config)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"Saved checkpoint to {checkpoint_dir / 'current.pt'}")
    print(f"Saved training history to {history_path}")

    # Generate post-training evaluation report and graphs
    eval_dir = generate_node_evaluation_report(
        node_id=config.node_id,
        history=history,
        env=env,
        log_dir=log_dir,
    )
    print(f"[{config.node_id}] All evaluation plots & dashboard saved to: {eval_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
