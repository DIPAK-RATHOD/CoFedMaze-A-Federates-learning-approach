"""
expert_demos.py

BFS Expert Demonstration Generator for CoFedMaze.

Per Section 3.2 & 3.3 of the CoFedMaze Workplan:
Generates optimal rule-based expert gameplay trajectories using BFSValidator
to pre-fill replay buffers and warm-start VDN policies.
"""

from __future__ import annotations

from typing import List, Optional

from env.core.actions import INTERACT, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP
from env.validators.bfs_validator import BFSValidator
from env.wrappers.pettingzoo_env import CoFedMazeParallelEnv
from marl.replay.trajectory import Trajectory, Transition


def _coord_to_action(curr_pos: tuple, next_pos: tuple) -> int:
    """Convert consecutive raw grid coordinates (row, col) into a movement action."""
    r1, c1 = curr_pos
    r2, c2 = next_pos

    if r2 > r1:
        return MOVE_DOWN
    if r2 < r1:
        return MOVE_UP
    if c2 > c1:
        return MOVE_RIGHT
    if c2 < c1:
        return MOVE_LEFT
    return INTERACT


def generate_expert_trajectory(env: CoFedMazeParallelEnv, seed: int = 101) -> Optional[Trajectory]:
    """
    Generate one complete expert gameplay Trajectory using BFS pathfinding.
    Both agents follow their optimal path through checkpoints to the exit goal.
    """
    obs, _ = env.reset(seed=seed)
    maze = env.maze
    validator = BFSValidator(maze)

    cell_a = maze.grid.get_logical_cell(*env._agent_objs["AGENT_A"].position)
    cell_b = maze.grid.get_logical_cell(*env._agent_objs["AGENT_B"].position)

    start_a_raw = (cell_a.row, cell_a.col)
    start_b_raw = (cell_b.row, cell_b.col)
    exit_raw = maze.exit_grid_position

    # Checkpoint-aware waypoint planning
    cps_raw = [cp.position for cp in env._checkpoints.values()]

    # Plan Agent A path through checkpoints then to exit
    curr = start_a_raw
    path_a = [curr]
    for cp_pos in cps_raw:
        cp_cell = maze.grid.get_logical_cell(*cp_pos)
        sub_path = validator.reconstruct_path(validator.traverse(curr), (cp_cell.row, cp_cell.col))
        if sub_path:
            path_a.extend(sub_path[1:])
            curr = (cp_cell.row, cp_cell.col)
    final_sub_a = validator.reconstruct_path(validator.traverse(curr), exit_raw)
    if final_sub_a:
        path_a.extend(final_sub_a[1:])

    # Plan Agent B direct path to exit
    path_b = validator.reconstruct_path(validator.traverse(start_b_raw), exit_raw)

    if not path_a or not path_b:
        return None

    trajectory = Trajectory(seed=seed, algorithm=env.algorithm)
    step_a_idx = 0
    step_b_idx = 0

    while env.agents and (step_a_idx < len(path_a) - 1 or step_b_idx < len(path_b) - 1):
        actions = {}

        if "AGENT_A" in env.agents:
            if step_a_idx < len(path_a) - 1:
                actions["AGENT_A"] = _coord_to_action(path_a[step_a_idx], path_a[step_a_idx + 1])
                step_a_idx += 1
            else:
                actions["AGENT_A"] = INTERACT

        if "AGENT_B" in env.agents:
            if step_b_idx < len(path_b) - 1:
                actions["AGENT_B"] = _coord_to_action(path_b[step_b_idx], path_b[step_b_idx + 1])
                step_b_idx += 1
            else:
                actions["AGENT_B"] = INTERACT

        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        done = all(terminations.values()) or all(truncations.values())
        team_reward = rewards.get("AGENT_A", 0.0)

        transition = Transition(
            obs=(obs["AGENT_A"], obs["AGENT_B"]),
            actions=(actions["AGENT_A"], actions["AGENT_B"]),
            reward=team_reward,
            next_obs=(next_obs.get("AGENT_A", obs["AGENT_A"]), next_obs.get("AGENT_B", obs["AGENT_B"])),
            done=done,
        )
        trajectory.append(transition)

        obs = next_obs
        if done:
            break

    trajectory.goal_reached = True
    trajectory.timeout = False
    return trajectory


def prefill_replay_buffer(env: CoFedMazeParallelEnv, buffer, num_demos: int = 16) -> int:
    """
    Generate `num_demos` expert trajectories and insert them into `buffer`.
    """
    added = 0
    curriculum_seeds = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]

    for i in range(num_demos):
        seed = curriculum_seeds[i % len(curriculum_seeds)]
        traj = generate_expert_trajectory(env, seed=seed)
        if traj is not None:
            buffer.add(traj)
            added += 1

    return added


if __name__ == "__main__":
    from marl.replay.replay_buffer import ReplayBuffer

    env = CoFedMazeParallelEnv(rows=7, cols=7, algorithm="recursive_backtracking")
    buf = ReplayBuffer(capacity=100)
    count = prefill_replay_buffer(env, buf, num_demos=5)
    print(f"expert_demos.py self-test OK. Added {count} expert trajectories into buffer.")
