"""
matplotlib_live_view.py

Real-time Matplotlib Graphical Renderer for CoFedMaze.

Displays live 2D grid maze rendering with animated agent movements (Agent A & B),
exit goal, checkpoints, keys, doors, obstacles, TWO separate plots for VDN Loss and Reward
(current + 10-episode moving avg), and a SPECIAL graph indicating whether each maze episode was SOLVED.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np


class MatplotlibLiveView:
    """
    Live Matplotlib GUI window for real-time maze, agent trajectory, separate Loss & Reward graphs,
    and a dedicated Goal Solved outcome graph during training.
    """

    def __init__(self, node_id: str = "N1", enabled: bool = True) -> None:
        self.node_id = node_id
        self.enabled = enabled
        self.fig = None
        self.ax_maze = None
        self.ax_loss = None
        self.ax_reward = None
        self.ax_solved = None

        self.img_plot = None
        self.scatter_a = None
        self.scatter_b = None
        self.scatter_exit = None

        # History tracking for Loss, Reward, and Solved outcomes
        self.episode_history: List[int] = []
        self.loss_history: List[float] = []
        self.reward_history: List[float] = []
        self.ma10_reward_history: List[float] = []
        self.solved_history: List[float] = []  # 100.0 if solved, 0.0 if timeout/unsolved
        self.succ_rate_history: List[float] = []

        self._last_rendered_episode = -1

        self.line_loss = None
        self.line_reward = None
        self.line_ma10 = None
        self.line_succ_rate = None

        self.is_closed = False

    def _init_figure(self, rows: int, cols: int):
        import os
        import matplotlib

        self.is_headless = False
        current_backend = matplotlib.get_backend().lower()
        if current_backend in ["agg", "template", "cairo"]:
            for backend in ["TkAgg", "Qt5Agg", "GTK3Agg", "WXAgg"]:
                try:
                    matplotlib.use(backend)
                    break
                except Exception:
                    pass

        import matplotlib.pyplot as plt

        if matplotlib.get_backend().lower() in ["agg", "template"]:
            if not os.environ.get("DISPLAY"):
                print(
                    f"[{self.node_id}] Headless Linux environment detected without active $DISPLAY. "
                    f"Pop-up GUI window disabled, but final 4-panel GUI plot will be saved to visualization/training_gui_plot_{self.node_id}.png upon training completion."
                )
            self.is_headless = True

        if not self.is_headless:
            plt.ion()  # Enable interactive GUI mode

        self.fig = plt.figure(figsize=(14, 8), num=f"CoFedMaze Live Training & Evaluation — Node {self.node_id}")
        if not self.is_headless:
            self.fig.canvas.mpl_connect("close_event", self._on_close)

        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], height_ratios=[1.0, 1.0])

        # Subplot 1: 2D Grid Maze View (Top-Left)
        self.ax_maze = self.fig.add_subplot(gs[0, 0])
        self.ax_maze.set_title(f"Node {self.node_id} — Live Agents & Maze Movement", fontsize=11, fontweight="bold")
        self.ax_maze.axis("off")

        # Subplot 2: VDN Loss Graph (Top-Right)
        self.ax_loss = self.fig.add_subplot(gs[0, 1])
        self.ax_loss.set_title("VDN Neural Network Loss", fontsize=10, fontweight="bold")
        self.ax_loss.set_xlabel("Episode", fontsize=8)
        self.ax_loss.set_ylabel("Loss", fontsize=8)
        self.ax_loss.grid(True, linestyle="--", alpha=0.5)
        (self.line_loss,) = self.ax_loss.plot([], [], color="#fb923c", linewidth=2.0, label="VDN Loss")
        self.ax_loss.legend(loc="upper right", fontsize=8)

        # Subplot 3: Reward Graph (Bottom-Left)
        self.ax_reward = self.fig.add_subplot(gs[1, 0])
        self.ax_reward.set_title("Episode Reward (Current vs 10-Ep Moving Avg)", fontsize=10, fontweight="bold")
        self.ax_reward.set_xlabel("Episode", fontsize=8)
        self.ax_reward.set_ylabel("Reward", fontsize=8)
        self.ax_reward.grid(True, linestyle="--", alpha=0.5)
        (self.line_reward,) = self.ax_reward.plot([], [], color="#38bdf8", linewidth=1.5, alpha=0.7, label="Current Reward")
        (self.line_ma10,) = self.ax_reward.plot([], [], color="#facc15", linewidth=2.2, label="10-Ep Moving Avg")
        self.ax_reward.legend(loc="upper left", fontsize=8)

        # Subplot 4: SPECIAL GRAPH — Maze Solved / Goal Reached Outcome (Bottom-Right)
        self.ax_solved = self.fig.add_subplot(gs[1, 1])
        self.ax_solved.set_title("SPECIAL GRAPH — Maze Solved Outcome (100% Solved vs 0% Timeout)", fontsize=10, fontweight="bold")
        self.ax_solved.set_xlabel("Episode", fontsize=8)
        self.ax_solved.set_ylabel("Outcome (%)", fontsize=8)
        self.ax_solved.set_ylim(-5, 115)
        self.ax_solved.grid(True, linestyle="--", alpha=0.5)

        (self.line_succ_rate,) = self.ax_solved.plot([], [], color="#4ade80", linewidth=2.2, linestyle="-", label="Cumulative Success Rate (%)")
        self.ax_solved.legend(loc="upper left", fontsize=8)

        # 2D Grid Initializer
        dummy_grid = np.zeros((rows, cols))
        self.img_plot = self.ax_maze.imshow(dummy_grid, cmap="binary_r", vmin=0, vmax=1, zorder=1)

        # Agent markers with high zorder so they are ALWAYS clearly visible over grid walls
        (self.scatter_a,) = self.ax_maze.plot([], [], "co", markersize=14, label="Agent A (0)", markeredgecolor="black", markeredgewidth=2.0, zorder=10)
        (self.scatter_b,) = self.ax_maze.plot([], [], "mo", markersize=14, label="Agent B (1)", markeredgecolor="black", markeredgewidth=2.0, zorder=10)
        (self.scatter_exit,) = self.ax_maze.plot([], [], "g*", markersize=22, label="Exit Goal", markeredgecolor="yellow", markeredgewidth=2.0, zorder=9)

        self.ax_maze.legend(loc="upper right", fontsize=8, framealpha=0.85)
        self.fig.tight_layout()

    def _on_close(self, event):
        self.is_closed = True

    def render_step(
        self,
        env,
        episode_count: int = 0,
        total_env_steps: int = 0,
        loss: Optional[float] = None,
        reward: Optional[float] = None,
        epsilon: Optional[float] = None,
        coalition: Optional[List[str]] = None,
        goal_reached: Optional[bool] = None,
        timeout: Optional[bool] = None,
    ) -> None:
        """
        Update the live Matplotlib GUI window with animated agent positions, separate Loss & Reward plots, and Solved status.
        """
        if not self.enabled or self.is_closed:
            return

        import matplotlib.pyplot as plt

        maze = getattr(env, "maze", None)
        if maze is None:
            return

        walls_array = maze.to_numpy()
        rows, cols = walls_array.shape

        if self.fig is None or not plt.fignum_exists(self.fig.number):
            try:
                self._init_figure(rows, cols)
            except Exception:
                self.enabled = False
                return

        if not self.enabled or self.img_plot is None:
            return

        # 1. Update Grid Image (0 = Path/White, 1 = Wall/Black)
        self.img_plot.set_data(walls_array)

        # 2. Update Exit Goal Position
        if hasattr(maze, "exit_grid_position"):
            exit_r, exit_c = maze.exit_grid_position
            self.scatter_exit.set_data([exit_c], [exit_r])

        # 3. Update Live Agent Positions (Convert logical pos -> raw grid pos via maze.grid)
        if hasattr(env, "_agent_objs") and len(env._agent_objs) >= 2:
            agent_objs = list(env._agent_objs.values())
            
            # Agent A
            pos_a = getattr(agent_objs[0], "position", None)
            if pos_a is not None and hasattr(maze, "grid"):
                try:
                    cell_a = maze.grid.get_logical_cell(*pos_a)
                    self.scatter_a.set_data([cell_a.col], [cell_a.row])
                except Exception:
                    pass

            # Agent B
            pos_b = getattr(agent_objs[1], "position", None)
            if pos_b is not None and hasattr(maze, "grid"):
                try:
                    cell_b = maze.grid.get_logical_cell(*pos_b)
                    self.scatter_b.set_data([cell_b.col], [cell_b.row])
                except Exception:
                    pass

        # 4. Update Loss, Reward, and Solved Graphs on Episode Boundary
        if episode_count > 0 and episode_count != self._last_rendered_episode:
            self._last_rendered_episode = episode_count
            self.episode_history.append(episode_count)

            cur_loss = loss if loss is not None else (self.loss_history[-1] if self.loss_history else 0.0)
            cur_rew = reward if reward is not None else (self.reward_history[-1] if self.reward_history else 0.0)
            is_solved = 100.0 if goal_reached else 0.0

            self.loss_history.append(cur_loss)
            self.reward_history.append(cur_rew)
            self.solved_history.append(is_solved)

            # Compute last 10 episode average reward
            last_10 = self.reward_history[-10:]
            avg_10 = sum(last_10) / len(last_10)
            self.ma10_reward_history.append(avg_10)

            # Cumulative success rate
            succ_count = sum(1 for s in self.solved_history if s == 100.0)
            self.succ_rate_history.append((succ_count / len(self.solved_history)) * 100.0)

            # Update Plot Lines
            self.line_loss.set_data(self.episode_history, self.loss_history)
            self.line_reward.set_data(self.episode_history, self.reward_history)
            self.line_ma10.set_data(self.episode_history, self.ma10_reward_history)

            # Update Solved Outcome Bar Plot
            self.ax_solved.clear()
            self.ax_solved.set_title("SPECIAL GRAPH — Maze Solved Outcome (100% Solved vs 0% Timeout)", fontsize=10, fontweight="bold")
            self.ax_solved.set_xlabel("Episode", fontsize=8)
            self.ax_solved.set_ylabel("Outcome (%)", fontsize=8)
            self.ax_solved.set_ylim(-5, 115)
            self.ax_solved.grid(True, linestyle="--", alpha=0.5)

            bar_colors = ["#4ade80" if s == 100.0 else "#f87171" for s in self.solved_history]
            self.ax_solved.bar(self.episode_history, self.solved_history, color=bar_colors, width=0.6, alpha=0.7)
            (self.line_succ_rate,) = self.ax_solved.plot(self.episode_history, self.succ_rate_history, color="#3b82f6", linewidth=2.2, label="Cumulative Success Rate (%)")
            self.ax_solved.legend(loc="upper left", fontsize=8)

            for ax in (self.ax_loss, self.ax_reward):
                ax.relim()
                ax.autoscale_view()

        # 5. Determine Maze Solved Status String & Banner Color
        step_count = getattr(env, "_step_count", 0)
        max_steps = getattr(env, "max_episode_steps", 100)

        if goal_reached:
            status_str = f"MAZE SOLVED! [Both Agents Reached Exit at Step {step_count}]"
            status_color = "#16a34a"  # Green
        elif timeout or step_count >= max_steps:
            status_str = f"TIMEOUT [Step Limit {max_steps} Reached]"
            status_color = "#dc2626"  # Red
        else:
            status_str = f"SEARCHING EXIT... [Step {step_count}/{max_steps}]"
            status_color = "#2563eb"  # Blue

        eps_val = epsilon if epsilon is not None else 1.0
        self.ax_maze.set_title(
            f"Node {self.node_id} - {status_str}\nEpisode: {episode_count} | Steps: {total_env_steps} | Epsilon: {eps_val:.2f}",
            fontsize=10, fontweight="bold", color=status_color
        )

        # 6. Redraw GUI Canvas
        if not getattr(self, "is_headless", False):
            try:
                plt.pause(0.005)  # Process Matplotlib GUI event loop
            except Exception:
                pass

    def save_plot(self, output_path: Optional[str] = None) -> str:
        """
        Save the final GUI figure plot as a PNG image inside the visualization folder.
        """
        if self.fig is None:
            return ""
        if output_path is None:
            import os
            os.makedirs("visualization", exist_ok=True)
            output_path = os.path.join("visualization", f"training_gui_plot_{self.node_id}.png")
        else:
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            self.fig.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"[{self.node_id}] Saved final GUI training plot to {output_path}")
            return output_path
        except Exception as e:
            print(f"[{self.node_id}] Warning: Could not save GUI plot: {e}")
            return ""

    def close(self) -> None:
        if self.fig is not None:
            self.save_plot()
            import matplotlib.pyplot as plt
            try:
                plt.close(self.fig)
            except Exception:
                pass


if __name__ == "__main__":
    from env.wrappers.pettingzoo_env import CoFedMazeParallelEnv

    env = CoFedMazeParallelEnv(rows=7, cols=7, algorithm="recursive_backtracking")
    env.reset(seed=42)
    view = MatplotlibLiveView(node_id="N1", enabled=True)
    view.render_step(env, episode_count=1, total_env_steps=10, loss=0.025, reward=-0.1, epsilon=0.99, goal_reached=False)
    print("visualization/matplotlib_live_view.py self-test OK")
