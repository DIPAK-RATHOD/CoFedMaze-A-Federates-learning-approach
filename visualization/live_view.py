"""
live_view.py

Provides a live, interactive ASCII terminal display for CoFedMaze nodes during training.
Renders per-step agent movements, episode metrics, loss, epsilon, and active coalition state
directly in the terminal tab in real-time.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from marl.training.trainer import LocalTrainer

# ANSI Color Codes for Terminal Rendering
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"
BLUE = "\033[34m"
GRAY = "\033[90m"


def render_ascii_maze(env) -> str:
    """
    Renders an ASCII grid representation of the CoFedMaze environment.
    """
    maze = env.maze
    walls_array = maze.to_numpy()
    rows, cols = walls_array.shape
    grid = [["." for _ in range(cols)] for _ in range(rows)]

    # Draw walls
    for r in range(rows):
        for c in range(cols):
            if walls_array[r, c] == 1:
                grid[r][c] = f"{GRAY}█{RESET}"

    # Draw Agent A and Agent B
    if hasattr(env, "_agent_objs"):
        if "agent_0" in env._agent_objs:
            pos_a = env._agent_objs["agent_0"].pos
            if 0 <= pos_a[0] < rows and 0 <= pos_a[1] < cols:
                grid[pos_a[0]][pos_a[1]] = f"{CYAN}{BOLD}A{RESET}"
        if "agent_1" in env._agent_objs:
            pos_b = env._agent_objs["agent_1"].pos
            if 0 <= pos_b[0] < rows and 0 <= pos_b[1] < cols:
                grid[pos_b[0]][pos_b[1]] = f"{MAGENTA}{BOLD}B{RESET}"

    lines = [" ".join(row) for row in grid]
    return "\n".join(lines)


class LiveTerminalView:
    """
    Manages live terminal tab output during training.
    """

    def __init__(self, node_id: str = "N1", mode: str = "tcp", enabled: bool = True) -> None:
        self.node_id = node_id
        self.mode = mode
        self.enabled = enabled
        self._last_round = 0

    def update(self, trainer: "LocalTrainer", current_round: int = 1, total_rounds: int = 10, coalition_members: Optional[list] = None) -> None:
        """
        Render the live status frame in the terminal tab.
        """
        if not self.enabled:
            return

        env = trainer.env
        ep = trainer.episode_count
        total_steps = trainer.total_env_steps
        eps = trainer._epsilon_for_episode(ep)
        loss_str = f"{trainer.last_loss:.4f}" if trainer.last_loss is not None else "N/A (buffering)"
        coal_str = str(coalition_members if coalition_members else [self.node_id])

        ascii_grid = render_ascii_maze(env)

        clear_str = "\033[H\033[J" if os.name != "nt" else ""
        header = (
            f"{clear_str}"
            f"================================================================================\n"
            f"{BOLD}{CYAN} COFEDMAZE LIVE NODE TRAINER {RESET}| Node: {BOLD}{self.node_id}{RESET} | Mode: {self.mode.upper()}\n"
            f"================================================================================\n"
            f" Task Variant: {BOLD}{env.algorithm}{RESET} ({env.rows}x{env.cols})\n"
            f" Round: {current_round}/{total_rounds} | Episode: {ep} | Total Env Steps: {total_steps}\n"
            f" Epsilon (ε): {eps:.3f} | Last Loss: {loss_str} | Active Coalition: {GREEN}{coal_str}{RESET}\n"
            f"--------------------------------------------------------------------------------\n"
            f" LIVE MAZE VIEW:\n"
            f"{ascii_grid}\n"
            f"--------------------------------------------------------------------------------\n"
            f" Legend: {CYAN}A{RESET}=Agent 0 | {MAGENTA}B{RESET}=Agent 1 | {GRAY}█{RESET}=Wall | {GREEN}.{RESET}=Path\n"
            f"================================================================================\n"
        )

        sys.stdout.write(header)
        sys.stdout.flush()


if __name__ == "__main__":
    from env.wrappers.pettingzoo_env import CoFedMazeParallelEnv

    env = CoFedMazeParallelEnv(rows=7, cols=7, algorithm="recursive_backtracking")
    env.reset(seed=42)
    print("ASCII Rendering Test:\n" + render_ascii_maze(env))
    print("live_view.py self-test OK")
