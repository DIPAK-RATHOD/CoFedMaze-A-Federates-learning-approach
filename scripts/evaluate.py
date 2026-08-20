"""Evaluate a saved local VDN checkpoint on reproducible maze seeds.

Run from the repository root::

    python -m scripts.evaluate --node-config data/node1/config.yaml \
        --checkpoint-dir state/node1/checkpoints --seeds 101 102 103

The script intentionally evaluates only; it never trains or modifies a
checkpoint.  More involved comparisons belong to the reusable helpers in
``evaluation/benchmark.py``, ``ablation.py``, ``scalability.py``, and
``robustness.py``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import yaml

from env.core.actions import NUM_ACTIONS
from env.core.observations import NUM_CHANNELS
from env.wrappers.pettingzoo_env import CoFedMazeParallelEnv
from evaluation.metrics import compute_metrics
from marl.models.vdn import VDNModel
from node.node_config import NodeConfig
from utils.checkpoint import load_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved CoFedMaze VDN model.")
    parser.add_argument("--node-config", type=Path, required=True, help="Path to data/nodeN/config.yaml.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Directory containing current.pt/previous.pt.")
    parser.add_argument("--slot", choices=("current", "previous", "best"), default="current", help="Checkpoint slot to evaluate.")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Epsilon exploration noise during evaluation (default: 0.0 for deterministic).")
    parser.add_argument("--evaluation-config", type=Path, default=Path("configs/evaluation.yaml"), help="Evaluation policy YAML path.")
    parser.add_argument("--seed-set", choices=("validation", "test"), default="test", help="Manifest split used when --seeds is omitted (default: test).")
    parser.add_argument("--seed-manifest", type=Path, default=None, help="Explicit seed-manifest JSON path; overrides the node's default split manifest.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Explicit evaluation maze seeds; overrides --seed-set and --seed-manifest.")
    parser.add_argument("--max-episode-steps", type=int, default=None, help="Evaluation timeout per episode; overrides evaluation config.")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Checkpoint encoder embedding dimension.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Checkpoint GRU hidden dimension.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path for the JSON report.")
    return parser


def make_environment(config: NodeConfig, max_episode_steps: int) -> CoFedMazeParallelEnv:
    return CoFedMazeParallelEnv(
        rows=config.maze_rows,
        cols=config.maze_cols,
        algorithm=config.maze_algorithm,
        window_size=config.window_size,
        max_episode_steps=max_episode_steps,
        num_checkpoints=config.num_checkpoints,
        num_obstacles=config.num_obstacles,
        num_key_door_pairs=config.num_key_door_pairs,
    )


def load_evaluation_config(path: Path) -> dict:
    """Load the small, shared evaluation policy with useful validation errors."""
    if not path.exists():
        raise FileNotFoundError(f"Evaluation config not found at {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("evaluation config must contain a YAML mapping")
    for key in ("routine_validation_subset_size", "max_episode_steps"):
        if key not in config:
            raise KeyError(f"evaluation config is missing required key {key!r}")
    return config


def load_seed_manifest(path: Path, config: NodeConfig, expected_split: str) -> list[int]:
    """Load and validate one node's split manifest before evaluating it."""
    if not path.exists():
        raise FileNotFoundError(
            f"Seed manifest not found at {path}. Generate it with "
            "python -m scripts.generate_mazes before running a default evaluation."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("node_id") != config.node_id:
        raise ValueError(f"Seed manifest is for {manifest.get('node_id')!r}, not {config.node_id!r}")
    if manifest.get("split") != expected_split:
        raise ValueError(f"Seed manifest split is {manifest.get('split')!r}, expected {expected_split!r}")
    seeds = manifest.get("seeds")
    if not isinstance(seeds, list) or not seeds or not all(isinstance(seed, int) for seed in seeds):
        raise ValueError(f"Seed manifest {path} must contain a non-empty integer 'seeds' list")
    return seeds


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluation_config = load_evaluation_config(args.evaluation_config)
    config = NodeConfig.load(args.node_config)
    manifest_path = args.seed_manifest or args.node_config.parent / "seeds" / f"{args.seed_set}.json"
    seeds = args.seeds if args.seeds is not None else load_seed_manifest(manifest_path, config, args.seed_set)
    max_episode_steps = args.max_episode_steps if args.max_episode_steps is not None else evaluation_config["max_episode_steps"]
    if not seeds:
        raise ValueError("evaluation seeds must not be empty")
    if max_episode_steps < 1:
        raise ValueError("--max-episode-steps must be positive")

    model = VDNModel(
        in_channels=NUM_CHANNELS,
        window_size=config.window_size,
        num_actions=NUM_ACTIONS,
        num_agents=2,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    )
    checkpoint = load_checkpoint(args.checkpoint_dir, slot=args.slot)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    report = compute_metrics(make_environment(config, max_episode_steps), model, seeds)
    payload = {
        "node_id": config.node_id,
        "task_variant": config.task_variant,
        "checkpoint_slot": args.slot,
        "checkpoint_episode": checkpoint["episode_count"],
        "seeds": seeds,
        "metrics": asdict(report),
    }
    print(json.dumps(payload, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Saved evaluation report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
