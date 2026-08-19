"""
post_train_best_selector.py

One-time post-training model selection utility for CoFedMaze.

For each node (N1, N2, N3):
1. Loads current.pt and previous.pt deterministically (epsilon = 0.0).
2. Evaluates both candidates on the SAME fixed 20 OOD validation seeds (1001-1020).
3. Selects the best model using:
   - PRIMARY CRITERION: highest validation_success_rate
   - SECONDARY CRITERION: lower average_steps_to_goal (tie-breaker)
4. Copies the winning checkpoint byte-for-byte to best.pt without modifying current.pt or previous.pt.
5. Writes best_metadata.json with full validation metrics and selection rationale.

Usage:
    python utils/post_train_best_selector.py
    python utils/post_train_best_selector.py --node-id N1
"""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env.core.observations import NUM_CHANNELS
from env.wrappers.pettingzoo_env import CoFedMazeParallelEnv
from marl.training.trainer import LocalTrainer
from node.node_config import NodeConfig
from utils.checkpoint import BEST_FILENAME, BEST_METADATA_FILENAME, CURRENT_FILENAME, PREVIOUS_FILENAME, has_checkpoint

DEFAULT_VALIDATION_SEEDS = list(range(1001, 1021))


def parse_args():
    parser = argparse.ArgumentParser(description="Post-training model selection utility for CoFedMaze nodes.")
    parser.add_argument("--node-id", type=str, default=None, help="Specific node ID to process (e.g. N1, N2, N3). If omitted, processes all available nodes.")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated list of validation seeds (default: 1001-1020).")
    parser.add_argument("--save-summary", type=str, default="visualization/post_train_selection_summary.json", help="Path to save full summary JSON.")
    return parser.parse_args()


def evaluate_checkpoint_candidate(
    trainer: LocalTrainer,
    ckpt_dir: Path,
    slot: str,
    seeds: List[int],
    node_id: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Load candidate checkpoint (current or previous) and evaluate deterministically
    on the exact list of validation seeds.
    """
    trainer.load_checkpoint(ckpt_dir, slot=slot)

    successful_steps = []
    eval_rewards = []
    successes = 0
    trial_records = []

    eval_id = f"eval_post_{node_id.lower()}_{slot}"

    for trial_idx, seed in enumerate(seeds, start=1):
        traj = trainer.run_episode(eval_mode=True, seed=seed)
        tot_rew = traj.total_reward()
        eval_rewards.append(tot_rew)
        goal_reached = traj.goal_reached

        if goal_reached:
            successes += 1
            successful_steps.append(len(traj))

        trial_records.append({
            "node_id": node_id,
            "checkpoint_name": slot,
            "evaluation_id": eval_id,
            "eval_trial": trial_idx,
            "maze_seed": seed,
            "success": 1 if goal_reached else 0,
            "goal_reached": goal_reached,
            "steps_to_goal": len(traj) if goal_reached else None,
            "steps": len(traj),
            "timeout": traj.timeout,
            "total_reward": tot_rew,
        })

    num_trials = len(seeds)
    success_rate = (successes / num_trials) * 100.0
    timeout_rate = ((num_trials - successes) / num_trials) * 100.0
    avg_steps = (sum(successful_steps) / len(successful_steps)) if successful_steps else float("inf")
    avg_reward = sum(eval_rewards) / num_trials

    metrics = {
        "node_id": node_id,
        "slot": slot,
        "successes": successes,
        "total_trials": num_trials,
        "validation_success_rate": success_rate,
        "timeout_rate": timeout_rate,
        "average_steps_to_goal": avg_steps if avg_steps != float("inf") else None,
        "average_reward": avg_reward,
        "checkpoint_episode": trainer.episode_count,
        "total_env_steps": trainer.total_env_steps,
    }

    return metrics, trial_records


def compute_file_hash(path: Path) -> str:
    """Compute MD5 checksum of a file to verify byte-for-byte copying."""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def process_node(node_id: str, seeds: List[int]) -> Dict[str, Any]:
    """
    Evaluate current.pt vs previous.pt for a node and select the best model.
    """
    config_path = f"data/{node_id.lower()}/config.yaml"
    if not os.path.exists(config_path):
        config_path = "data/node1/config.yaml"

    config = NodeConfig.load(config_path)
    config.node_id = node_id

    ckpt_dir = Path(f"state/{node_id.lower()}/checkpoints")
    if not ckpt_dir.exists() and Path(f"state/checkpoint_{node_id}").exists():
        ckpt_dir = Path(f"state/checkpoint_{node_id}")

    if not ckpt_dir.exists():
        print(f"[{node_id}] SKIPPED: Checkpoint directory {ckpt_dir} does not exist.")
        return {}

    has_curr = has_checkpoint(ckpt_dir, slot="current")
    has_prev = has_checkpoint(ckpt_dir, slot="previous")

    if not has_curr and not has_prev:
        print(f"[{node_id}] SKIPPED: No current.pt or previous.pt found in {ckpt_dir}.")
        return {}

    env = CoFedMazeParallelEnv(
        rows=config.maze_rows,
        cols=config.maze_cols,
        algorithm=config.maze_algorithm,
        window_size=config.window_size,
        num_checkpoints=config.num_checkpoints,
        num_obstacles=config.num_obstacles,
        num_key_door_pairs=config.num_key_door_pairs,
    )

    trainer = LocalTrainer(
        env=env,
        in_channels=NUM_CHANNELS,
        num_actions=5,
        checkpoint_dir=ckpt_dir,
        validation_seeds=seeds,
        auto_resume=False,
    )

    curr_metrics: Optional[Dict[str, Any]] = None
    curr_trials: List[Dict[str, Any]] = []
    if has_curr:
        curr_metrics, curr_trials = evaluate_checkpoint_candidate(trainer, ckpt_dir, "current", seeds, node_id)

    prev_metrics: Optional[Dict[str, Any]] = None
    prev_trials: List[Dict[str, Any]] = []
    if has_prev:
        prev_metrics, prev_trials = evaluate_checkpoint_candidate(trainer, ckpt_dir, "previous", seeds, node_id)

    # Calculate initial hash values to verify current.pt & previous.pt are not modified
    curr_hash_before = compute_file_hash(ckpt_dir / CURRENT_FILENAME) if has_curr else None
    prev_hash_before = compute_file_hash(ckpt_dir / PREVIOUS_FILENAME) if has_prev else None

    # Apply Best Model Selection Rule
    selected_slot = "current"
    reason = ""

    if curr_metrics and not prev_metrics:
        selected_slot = "current"
        reason = "Only current.pt existed for evaluation."
    elif prev_metrics and not curr_metrics:
        selected_slot = "previous"
        reason = "Only previous.pt existed for evaluation."
    else:
        rate_curr = curr_metrics["validation_success_rate"]
        rate_prev = prev_metrics["validation_success_rate"]
        steps_curr = curr_metrics["average_steps_to_goal"] if curr_metrics["average_steps_to_goal"] is not None else float("inf")
        steps_prev = prev_metrics["average_steps_to_goal"] if prev_metrics["average_steps_to_goal"] is not None else float("inf")

        if rate_curr > rate_prev:
            selected_slot = "current"
            reason = f"Higher validation success rate ({rate_curr:.1f}% vs {rate_prev:.1f}%)."
        elif rate_prev > rate_curr:
            selected_slot = "previous"
            reason = f"Higher validation success rate ({rate_prev:.1f}% vs {rate_curr:.1f}%)."
        else:
            # Success rates are equal -> secondary tie-breaker (lower average_steps_to_goal)
            if steps_curr < steps_prev:
                selected_slot = "current"
                reason = f"Equal success rate ({rate_curr:.1f}%), lower average steps to goal ({steps_curr:.2f} vs {steps_prev:.2f})."
            elif steps_prev < steps_curr:
                selected_slot = "previous"
                reason = f"Equal success rate ({rate_prev:.1f}%), lower average steps to goal ({steps_prev:.2f} vs {steps_curr:.2f})."
            else:
                selected_slot = "current"
                reason = f"Equal success rate ({rate_curr:.1f}%) and equal average steps to goal. Defaulted to current.pt."

    winning_metrics = curr_metrics if selected_slot == "current" else prev_metrics
    source_filename = CURRENT_FILENAME if selected_slot == "current" else PREVIOUS_FILENAME
    source_path = ckpt_dir / source_filename
    best_path = ckpt_dir / BEST_FILENAME

    # Copy selected checkpoint byte-for-byte to best.pt
    shutil.copy2(source_path, best_path)

    # Verify hashes after copy
    curr_hash_after = compute_file_hash(ckpt_dir / CURRENT_FILENAME) if has_curr else None
    prev_hash_after = compute_file_hash(ckpt_dir / PREVIOUS_FILENAME) if has_prev else None
    best_hash = compute_file_hash(best_path)

    assert curr_hash_before == curr_hash_after, f"ERROR: current.pt was mutated during post-training selection!"
    assert prev_hash_before == prev_hash_after, f"ERROR: previous.pt was mutated during post-training selection!"

    # Write best_metadata.json
    metadata = {
        "node_id": node_id,
        "source_checkpoint": selected_slot,
        "source_filename": source_filename,
        "validation_success_rate": winning_metrics["validation_success_rate"],
        "timeout_rate": winning_metrics["timeout_rate"],
        "average_steps_to_goal": winning_metrics["average_steps_to_goal"],
        "average_reward": winning_metrics["average_reward"],
        "validation_episodes": winning_metrics["total_trials"],
        "validation_seeds": seeds,
        "selection_reason": reason,
        "selected_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "checkpoint_episode": winning_metrics["checkpoint_episode"],
        "total_env_steps": winning_metrics["total_env_steps"],
    }

    meta_path = ckpt_dir / BEST_METADATA_FILENAME
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # Print requested formatted comparison
    print("\n" + "=" * 60)
    print(f"Node {node_id}")
    print("-" * 60)

    if curr_metrics:
        avg_s_str = f"{curr_metrics['average_steps_to_goal']:.2f}" if curr_metrics['average_steps_to_goal'] is not None else "None"
        print("current.pt:")
        print(f"  success = {curr_metrics['successes']}/{curr_metrics['total_trials']}")
        print(f"  success_rate = {curr_metrics['validation_success_rate']:.1f}%")
        print(f"  avg_steps = {avg_s_str}")
        print(f"  timeout = {curr_metrics['timeout_rate']:.1f}%")
    else:
        print("current.pt:\n  [Not Available]")

    print()
    if prev_metrics:
        avg_s_str = f"{prev_metrics['average_steps_to_goal']:.2f}" if prev_metrics['average_steps_to_goal'] is not None else "None"
        print("previous.pt:")
        print(f"  success = {prev_metrics['successes']}/{prev_metrics['total_trials']}")
        print(f"  success_rate = {prev_metrics['validation_success_rate']:.1f}%")
        print(f"  avg_steps = {avg_s_str}")
        print(f"  timeout = {prev_metrics['timeout_rate']:.1f}%")
    else:
        print("previous.pt:\n  [Not Available]")

    print("\nSELECTED:")
    print(f"  {source_filename} ({selected_slot})")
    print(f"  Reason: {reason}")
    print(f"  Saved best.pt to: {best_path}")
    print(f"  Saved metadata to: {meta_path}")
    print("=" * 60)

    return {
        "node_id": node_id,
        "selected_slot": selected_slot,
        "selected_filename": source_filename,
        "metadata": metadata,
        "current_metrics": curr_metrics,
        "previous_metrics": prev_metrics,
        "trials": curr_trials + prev_trials,
    }


def main():
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()] if args.seeds else DEFAULT_VALIDATION_SEEDS

    nodes_to_process = [args.node_id] if args.node_id else ["N1", "N2", "N3"]
    results = {}

    print(f"Starting Post-Training Model Selection Utility")
    print(f"Validation Seeds ({len(seeds)} total): {seeds}")

    for nid in nodes_to_process:
        res = process_node(nid, seeds)
        if res:
            results[nid] = res

    os.makedirs(Path(args.save_summary).parent, exist_ok=True)
    with open(args.save_summary, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\nCompleted post-training selection for all nodes. Full report written to {args.save_summary}")


if __name__ == "__main__":
    main()
