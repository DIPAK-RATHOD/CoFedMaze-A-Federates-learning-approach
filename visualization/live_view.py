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
    maze = getattr(env, "maze", None)
    if maze is None:
        return "[Maze not initialized]"

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
        for aid, color, symbol in [("agent_0", CYAN, "A"), ("agent_1", MAGENTA, "B")]:
            if aid in env._agent_objs:
                obj = env._agent_objs[aid]
                pos = getattr(obj, "position", getattr(obj, "pos", None))
                if pos is not None and len(pos) == 2:
                    r, c = pos
                    if 0 <= r < rows and 0 <= c < cols:
                        grid[r][c] = f"{color}{BOLD}{symbol}{RESET}"

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

    def render_step(self, env, episode_count: int = 0, total_env_steps: int = 0) -> None:
        """
        Render a live step frame during episode rollout.
        """
        if not self.enabled:
            return

        ascii_grid = render_ascii_maze(env)
        clear_str = "\033[H\033[J" if os.name != "nt" else ""
        header = (
            f"{clear_str}"
            f"================================================================================\n"
            f"{BOLD}{CYAN} COFEDMAZE LIVE STEP VIEW {RESET}| Node: {BOLD}{self.node_id}{RESET} | Mode: {self.mode.upper()}\n"
            f"================================================================================\n"
            f" Episode: {episode_count} | Total Env Steps: {total_env_steps}\n"
            f"--------------------------------------------------------------------------------\n"
            f" LIVE MAZE VIEW:\n"
            f"{ascii_grid}\n"
            f"--------------------------------------------------------------------------------\n"
            f" Legend: {CYAN}A{RESET}=Agent 0 | {MAGENTA}B{RESET}=Agent 1 | {GRAY}█{RESET}=Wall | {GREEN}.{RESET}=Path\n"
            f"================================================================================\n"
        )
        sys.stdout.write(header)
        sys.stdout.flush()

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
    view = LiveTerminalView(node_id="N1", mode="tcp", enabled=True)
    view.render_step(env, episode_count=1, total_env_steps=5)
    print("live_view.py self-test OK")
