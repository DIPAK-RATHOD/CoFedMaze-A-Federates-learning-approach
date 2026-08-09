"""Run the in-process federated simulation and save every node's model.

This is a software-simulation entry point, not a deployment script. It uses
the existing lockstep ``scripts.run_simulation`` runtime and persists the
resulting local models for evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from scripts.run_simulation import run as run_simulation
from utils.checkpoint import save_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CoFedMaze's federated software simulation.")
    parser.add_argument("--rounds", type=int, default=10, help="Federated training rounds (default: 10).")
    parser.add_argument("--seed", type=int, default=0, help="Simulation master seed (default: 0).")
    parser.add_argument("--state-dir", type=Path, default=Path("state"), help="Directory for per-node checkpoints.")
    parser.add_argument("--summary-output", type=Path, default=Path("outputs/federated_summary.json"), help="JSON simulation-summary path.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-round coalition summaries.")
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.rounds < 1:
        raise ValueError("--rounds must be positive")

    schedulers = run_simulation(args.rounds, master_seed=args.seed, verbose=not args.quiet)
    summary = {"rounds": args.rounds, "master_seed": args.seed, "nodes": {}}
    for node_id, scheduler in schedulers.items():
        trainer = scheduler.services.trainer
        checkpoint_dir = args.state_dir / node_id.lower().replace("n", "node", 1) / "checkpoints"
        save_checkpoint(
            checkpoint_dir, trainer.online_model, trainer.optimizer,
            episode_count=trainer.episode_count,
            total_env_steps=trainer.total_env_steps,
            target_model=trainer.target_model,
            metadata={"node_id": node_id, "master_seed": args.seed, "federated_rounds": args.rounds},
        )
        summary["nodes"][node_id] = {
            "checkpoint": str(checkpoint_dir / "current.pt"),
            "episode_count": trainer.episode_count,
            "total_env_steps": trainer.total_env_steps,
            "coalition_members": sorted(scheduler.services.coalition_manager.members),
        }
        print(f"Saved {node_id} checkpoint to {checkpoint_dir / 'current.pt'}")

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved federated summary to {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
