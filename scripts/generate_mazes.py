"""Create reproducible per-node train/validation/test maze-seed manifests.

Seeds are stored instead of materialised mazes: the environment regenerates a
maze deterministically when a seed is evaluated or trained on.

Example::

    python -m scripts.generate_mazes --node-config data/node1/config.yaml \
        --train-count 100 --validation-count 10 --test-count 20 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

from env.core.maze import Maze
from env.generator.generator_factory import create_generator
from env.validators.path_checker import has_path_to_exit
from node.node_config import NodeConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate reproducible CoFedMaze seed manifests.")
    parser.add_argument("--node-config", type=Path, required=True, help="Path to data/nodeN/config.yaml.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory; defaults to data/nodeN/seeds.")
    parser.add_argument("--train-count", type=int, default=100, help="Number of training seeds (default: 100).")
    parser.add_argument("--validation-count", type=int, default=10, help="Number of validation seeds (default: 10).")
    parser.add_argument("--test-count", type=int, default=20, help="Number of held-out test seeds (default: 20).")
    parser.add_argument("--seed", type=int, default=0, help="Manifest random seed (default: 0).")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacement of existing manifest files.")
    return parser


def _default_output_dir(node_config_path: Path) -> Path:
    return node_config_path.parent / "seeds"


def _is_solvable(config: NodeConfig, seed: int) -> bool:
    maze = Maze(rows=config.maze_rows, cols=config.maze_cols, random_seed=seed)
    create_generator(config.maze_algorithm, random_seed=seed).generate(maze)
    return has_path_to_exit(maze)


def generate_seeds(config: NodeConfig, count: int, rng: random.Random, used: set[int]) -> list[int]:
    """Generate distinct, validated seeds without accepting an invalid maze."""
    if count < 0:
        raise ValueError("seed counts must be non-negative")
    seeds: list[int] = []
    while len(seeds) < count:
        candidate = rng.randrange(2**31)
        if candidate in used or not _is_solvable(config, candidate):
            continue
        used.add(candidate)
        seeds.append(candidate)
    return seeds


def _manifest(config: NodeConfig, split: str, seeds: list[int], manifest_seed: int) -> dict:
    return {
        "node_id": config.node_id,
        "task_variant": config.task_variant,
        "split": split,
        "generator": config.maze_algorithm,
        "rows": config.maze_rows,
        "cols": config.maze_cols,
        "manifest_seed": manifest_seed,
        "seeds": seeds,
    }


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = NodeConfig.load(args.node_config)
    output_dir = args.output_dir or _default_output_dir(args.node_config)
    counts = {"train": args.train_count, "validation": args.validation_count, "test": args.test_count}
    if any(count < 0 for count in counts.values()):
        raise ValueError("all seed counts must be non-negative")

    paths = {split: output_dir / f"{split}.json" for split in counts}
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Seed manifests already exist: {joined}. Pass --overwrite to replace them.")

    rng = random.Random(args.seed)
    used: set[int] = set()
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, count in counts.items():
        seeds = generate_seeds(config, count, rng, used)
        paths[split].write_text(json.dumps(_manifest(config, split, seeds, args.seed), indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {len(seeds)} {split} seeds to {paths[split]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
