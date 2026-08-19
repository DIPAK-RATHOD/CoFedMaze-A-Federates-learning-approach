"""
test_eval_manual.py

Standalone manual evaluation tool for CoFedMaze.
Loads a trained model checkpoint from disk and evaluates it deterministically (epsilon = 0.0)
step-by-step on specified maze seeds, printing full trajectory metrics and saving a summary JSON.

Usage:
    python visualization/test_eval_manual.py --node-id N1 --episodes 100 --seeds 101,102,103,999
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env.core.observations import NUM_CHANNELS
from env.wrappers.pettingzoo_env import CoFedMazeParallelEnv
from marl.training.trainer import LocalTrainer
from node.node_config import NodeConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Manual evaluation tool for trained CoFedMaze checkpoints.")
    parser.add_argument("--node-id", type=str, default="N1", help="Node ID (e.g. N1, N2, N3)")
    parser.add_argument("--config", type=str, default=None, help="Path to node config YAML")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--seeds", type=str, default="101,102,103,999", help="Comma-separated list of maze seeds to evaluate")
    parser.add_argument("--episodes", type=int, default=100, help="Train episodes to run first if no checkpoint exists")
    parser.add_argument("--save-summary", type=str, default="visualization/manual_eval_summary.json", help="Summary JSON output path")
    return parser.parse_args()


def run_manual_evaluation():
    args = parse_args()
    
    config_path = args.config or f"data/{args.node_id.lower()}/config.yaml"
    if not os.path.exists(config_path):
        config_path = "data/node1/config.yaml"

    config = NodeConfig.load(config_path)
    config.node_id = args.node_id

    ckpt_dir = args.checkpoint_dir or f"state/checkpoint_{args.node_id}"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs("visualization", exist_ok=True)

    env = CoFedMazeParallelEnv(
        rows=config.maze_rows,
        cols=config.maze_cols,
        algorithm=config.maze_algorithm,
        num_checkpoints=config.num_checkpoints,
        num_obstacles=config.num_obstacles,
        num_key_door_pairs=config.num_key_door_pairs,
    )

    trainer = LocalTrainer(
        env=env,
        in_channels=NUM_CHANNELS,
        num_actions=5,
        checkpoint_dir=ckpt_dir,
        auto_resume=True,
    )

    # Check if checkpoint exists, otherwise train for requested episodes
    from utils.checkpoint import has_checkpoint
    if not has_checkpoint(ckpt_dir):
        print(f"[{args.node_id}] No existing checkpoint found at {ckpt_dir}. Running fresh training for {args.episodes} episodes...")
        for ep in range(1, args.episodes + 1):
            trainer.run_episode()
        trainer.save_checkpoint(ckpt_dir)
        print(f"[{args.node_id}] Checkpoint saved after {args.episodes} episodes.")
    else:
        loaded = trainer.load_checkpoint(ckpt_dir)
        print(f"[{args.node_id}] Successfully loaded checkpoint from {ckpt_dir} (Episode {trainer.episode_count}, Steps {trainer.total_env_steps}).")

    # Set deterministic policy (epsilon = 0.0)
    for selector in trainer._selectors.values():
        selector.set_epsilon(0.0)

    seeds_to_test: List[int] = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    results = []

    print("\n" + "=" * 70)
    print(f"  COFEDMAZE MANUAL DETERMINISTIC EVALUATION (Node {args.node_id} | Epsilon = 0.0)")
    print("=" * 70)

    for seed in seeds_to_test:
        obs, _ = env.reset(seed=seed)
        for agent in trainer._agents.values():
            agent.reset_hidden()

        done = False
        step = 0
        total_reward = 0.0
        
        print(f"\n--- Running Evaluation on Seed {seed} ---")
        while not done and step < env.max_episode_steps:
            step += 1
            actions = {}
            for aid in env.agents:
                obs_tensor = trainer._to_tensor(obs[aid])
                action, _ = trainer._agents[aid].act(obs_tensor)
                actions[aid] = action

            obs, rewards, terminations, truncations, infos = env.step(actions)
            step_reward = sum(rewards.values()) / len(rewards)
            total_reward += step_reward

            done = all(terminations.values()) or all(truncations.values())
            goal_reached = all(
                env._exit_obj.is_usable_by(env.maze, env._agent_objs[aid].position)
                for aid in env.possible_agents
            )

            if step <= 5 or done or step % 25 == 0:
                pos_a = env._agent_objs["AGENT_A"].position if "AGENT_A" in env._agent_objs else None
                pos_b = env._agent_objs["AGENT_B"].position if "AGENT_B" in env._agent_objs else None
                print(f"  Step {step:03d} | Agent A: {pos_a} | Agent B: {pos_b} | Step Rew: {step_reward:+.2f} | Goal: {goal_reached}")

        outcome = "SUCCESS" if goal_reached else "TIMEOUT"
        print(f"  => SEED {seed} RESULT: {outcome} | Steps: {step} | Total Reward: {total_reward:.2f}")

        results.append({
            "seed": seed,
            "outcome": outcome,
            "goal_reached": goal_reached,
            "steps": step,
            "total_reward": total_reward,
        })

    # Save summary report
    summary = {
        "node_id": args.node_id,
        "checkpoint_episode": trainer.episode_count,
        "total_env_steps": trainer.total_env_steps,
        "num_trials": len(seeds_to_test),
        "success_rate": (sum(1 for r in results if r["goal_reached"]) / len(results)) * 100.0,
        "trials": results,
    }

    with open(args.save_summary, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  SUMMARY: Success Rate = {summary['success_rate']:.1f}% | Saved report to {args.save_summary}")
    print("=" * 70)


if __name__ == "__main__":
    run_manual_evaluation()
