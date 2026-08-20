"""
scripts/analyze_trajectory.py

Detailed diagnostic trajectory analyzer comparing successful vs failed validation episodes.
Exposes step-by-step positions, actions, Q-values, milestone progress, reward components,
and termination reasons under deterministic (epsilon=0.0) and exploratory (epsilon=0.1) evaluation.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env.core.actions import NUM_ACTIONS
from env.core.constants import AGENT_A, AGENT_B
from env.core.observations import NUM_CHANNELS
from env.wrappers.pettingzoo_env import CoFedMazeParallelEnv
from marl.agents.action_selector import EpsilonGreedySelector
from marl.agents.policy import Policy
from marl.agents.vdn_agent import VDNAgent
from marl.models.vdn import VDNModel
from node.node_config import NodeConfig
from utils.checkpoint import load_checkpoint


def analyze_seed_trajectory(
    env: CoFedMazeParallelEnv,
    model: VDNModel,
    seed: int,
    epsilon: float = 0.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    selectors = {
        AGENT_A: EpsilonGreedySelector(epsilon=epsilon),
        AGENT_B: EpsilonGreedySelector(epsilon=epsilon),
    }
    policies = {
        AGENT_A: Policy(model, agent_index=0, selector=selectors[AGENT_A]),
        AGENT_B: Policy(model, agent_index=1, selector=selectors[AGENT_B]),
    }
    agents = {
        AGENT_A: VDNAgent(policies[AGENT_A]),
        AGENT_B: VDNAgent(policies[AGENT_B]),
    }

    obs, _ = env.reset(seed=seed)
    for agent in agents.values():
        agent.reset_hidden()

    trajectory_trace = []
    step_num = 0
    total_reward = 0.0
    collisions = 0
    goal_reached = False

    if verbose:
        print(f"\n" + "=" * 90)
        print(f"  DIAGNOSTIC TRAJECTORY TRACE — SEED {seed} | EPSILON = {epsilon:.1f}")
        print("=" * 90)
        print(f"  {'Step':<5} | {'Pos Agent A':<12} | {'Pos Agent B':<12} | {'Action A/B':<12} | {'Max Q A/B':<16} | {'Reward':<8} | {'Status'}")
        print("  " + "-" * 90)

    while env.agents:
        step_num += 1
        actions = {}
        q_values_dict = {}

        pos_a = env._agent_objs[AGENT_A].position if AGENT_A in env._agent_objs else (0, 0)
        pos_b = env._agent_objs[AGENT_B].position if AGENT_B in env._agent_objs else (0, 0)

        for agent_id in env.agents:
            obs_tensor = torch.from_numpy(obs[agent_id]).unsqueeze(0)
            agent_idx = 0 if agent_id == AGENT_A else 1

            # Query Q-values from policy model
            with torch.no_grad():
                q_vals, _ = model.forward_agent(obs_tensor, agents[agent_id].hidden_state, agent_idx)
                q_values_dict[agent_id] = q_vals.squeeze(0).cpu().numpy().tolist()

            if epsilon == 0.0:
                action, _ = agents[agent_id].act_greedy(obs_tensor)
            else:
                action, _ = agents[agent_id].act(obs_tensor)
            actions[agent_id] = action

        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        step_reward = rewards[AGENT_A]
        total_reward += step_reward

        done = any(terminations.values()) or any(truncations.values())
        if any(terminations.values()):
            goal_reached = True

        q_a_max = max(q_values_dict.get(AGENT_A, [0.0]))
        q_b_max = max(q_values_dict.get(AGENT_B, [0.0]))
        action_a = actions.get(AGENT_A, -1)
        action_b = actions.get(AGENT_B, -1)

        step_info = {
            "step": step_num,
            "pos_a": list(pos_a),
            "pos_b": list(pos_b),
            "action_a": action_a,
            "action_b": action_b,
            "q_values_a": q_values_dict.get(AGENT_A, []),
            "q_values_b": q_values_dict.get(AGENT_B, []),
            "step_reward": round(step_reward, 4),
            "cumulative_reward": round(total_reward, 4),
            "done": done,
            "goal_reached": goal_reached,
        }
        trajectory_trace.append(step_info)

        if verbose:
            pos_a_str = f"({pos_a[0]},{pos_a[1]})"
            pos_b_str = f"({pos_b[0]},{pos_b[1]})"
            act_str = f"{action_a} / {action_b}"
            q_str = f"{q_a_max:.2f} / {q_b_max:.2f}"
            status = "REACHED GOAL" if goal_reached else ("TIMED OUT" if done else "IN_PROGRESS")
            print(f"  {step_num:<5} | {pos_a_str:<12} | {pos_b_str:<12} | {act_str:<12} | {q_str:<16} | {step_reward:+8.3f} | {status}")

        obs = next_obs

    summary = {
        "seed": seed,
        "epsilon": epsilon,
        "goal_reached": goal_reached,
        "timeout": not goal_reached,
        "total_steps": step_num,
        "total_reward": round(total_reward, 4),
        "steps_to_goal": step_num if goal_reached else None,
        "trace_length": len(trajectory_trace),
    }

    if verbose:
        print("  " + "-" * 90)
        res_str = "SUCCESS" if goal_reached else "FAILURE (TIMEOUT)"
        print(f"  RESULT: {res_str} | Total Steps: {step_num} | Total Reward: {total_reward:+.4f}")
        print("=" * 90)

    return {"summary": summary, "trace": trajectory_trace}


def main():
    parser = argparse.ArgumentParser(description="Detailed diagnostic trajectory analyzer.")
    parser.add_argument("--node-config", type=Path, required=True, help="Path to data/nodeN/config.yaml.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Checkpoint directory.")
    parser.add_argument("--slot", choices=("current", "previous", "best"), default="best")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1001, 1005])
    parser.add_argument("--compare-epsilon", action="store_true", help="Compare epsilon=0.0 vs epsilon=0.1 on all seeds.")
    args = parser.parse_args()

    config = NodeConfig.load(args.node_config)
    env = CoFedMazeParallelEnv(
        rows=config.maze_rows,
        cols=config.maze_cols,
        algorithm=config.maze_algorithm,
        window_size=config.window_size,
        max_episode_steps=100,
        num_checkpoints=config.num_checkpoints,
        num_obstacles=config.num_obstacles,
        num_key_door_pairs=config.num_key_door_pairs,
    )

    model = VDNModel(
        in_channels=NUM_CHANNELS,
        window_size=config.window_size,
        num_actions=NUM_ACTIONS,
        num_agents=2,
    )
    checkpoint = load_checkpoint(args.checkpoint_dir, slot=args.slot)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    for seed in args.seeds:
        analyze_seed_trajectory(env, model, seed=seed, epsilon=0.0, verbose=True)
        if args.compare_epsilon:
            analyze_seed_trajectory(env, model, seed=seed, epsilon=0.1, verbose=True)


if __name__ == "__main__":
    main()
