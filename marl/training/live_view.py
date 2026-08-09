"""
live_view.py

Updated version

Improvements
------------
✓ Fixed repeated twinx() creation
✓ Added moving-average reward curve
✓ Added loss curve (if available)
✓ Prevented duplicate legends
✓ Cleaner plotting
✓ More efficient redraws
"""

import matplotlib

_BACKEND = None
for _candidate in ("TkAgg", "Qt5Agg", "MacOSX"):
    try:
        matplotlib.use(_candidate)
        _BACKEND = _candidate
        break
    except Exception:
        continue

if _BACKEND is None:
    raise RuntimeError(
        "No interactive matplotlib backend available "
        "(TkAgg / Qt5Agg / MacOSX)."
    )

import matplotlib.pyplot as plt

from env.core.constants import AGENT_A, AGENT_B
from marl.training.trainer import LocalTrainer

AGENT_COLORS = {
    AGENT_A: "tab:blue",
    AGENT_B: "tab:orange",
}


class LiveTrainerView:
    """
    Live visualization for LocalTrainer.

    Left:
        Maze animation.

    Right:
        Reward
        Moving-average reward
        Loss
        Epsilon
    """

    def __init__(
        self,
        trainer: LocalTrainer,
        render_every_n_steps: int = 1,
        pause_seconds: float = 0.03,
        max_history: int = 200,
        moving_average_window: int = 10,
    ):

        if render_every_n_steps <= 0:
            raise ValueError(
                "render_every_n_steps must be positive."
            )

        self.trainer = trainer
        self.render_every_n_steps = render_every_n_steps
        self.pause_seconds = pause_seconds
        self.max_history = max_history
        self.ma_window = moving_average_window

        self.reward_history = []
        self.loss_history = []
        self.epsilon_history = []

        plt.ion()

        self.fig, (self.ax_maze, self.ax_curves) = plt.subplots(
            1,
            2,
            figsize=(12, 5)
        )

        # IMPORTANT:
        # Create twinx() ONLY ONCE.
        self.ax_eps = self.ax_curves.twinx()

        try:
            self.fig.canvas.manager.set_window_title(
                "CoFedMaze - Live Training"
            )
        except Exception:
            pass

    def _moving_average(self, values):
        """
        Compute moving average.
        """
        if not values:
            return []

        out = []

        for i in range(len(values)):
            start = max(0, i - self.ma_window + 1)

            out.append(
                sum(values[start:i + 1]) /
                (i - start + 1)
            )

        return out

    def _trim_histories(self):
        """
        Keep only recent history.
        """

        self.reward_history = self.reward_history[-self.max_history:]
        self.loss_history = self.loss_history[-self.max_history:]
        self.epsilon_history = self.epsilon_history[-self.max_history:]

    def _render_maze(self):
        """
        Draw the current maze and agent positions.
        """
        maze = self.trainer.env.maze

        if maze is None:
            return

        array = maze.to_numpy()

        self.ax_maze.clear()

        self.ax_maze.imshow(
            array,
            cmap="gray_r",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )

        # Draw exit
        if maze.exit_grid_position is not None:
            row, col = maze.exit_grid_position

            self.ax_maze.scatter(
                col,
                row,
                marker="X",
                s=150,
                c="tab:red",
                edgecolors="black",
                linewidths=1.2,
                label="Exit",
                zorder=3,
            )

        # Draw agents
        for agent_id, color in AGENT_COLORS.items():

            agent = self.trainer.env._agent_objs.get(agent_id)

            if agent is None or agent.position is None:
                continue

            cell = maze.grid.get_logical_cell(*agent.position)

            self.ax_maze.scatter(
                cell.col,
                cell.row,
                s=130,
                c=color,
                edgecolors="black",
                linewidths=1.2,
                label=agent_id,
                zorder=4,
            )

        self.ax_maze.set_xticks([])
        self.ax_maze.set_yticks([])

        eps = self.trainer._epsilon_for_episode(
            self.trainer.episode_count
        )

        self.ax_maze.set_title(
            f"Episode {self.trainer.episode_count + 1}"
            f" | Steps {self.trainer.total_env_steps}"
            f" | ε={eps:.3f}"
        )

        # Remove duplicate legend entries
        handles, labels = self.ax_maze.get_legend_handles_labels()

        unique = {}

        for h, l in zip(handles, labels):
            if l not in unique:
                unique[l] = h

        if unique:
            self.ax_maze.legend(
                unique.values(),
                unique.keys(),
                fontsize=8,
                loc="upper right",
            )

    def _render_curves(self):
        """
        Draw reward, moving average, loss and epsilon.
        """

        self.ax_curves.clear()
        self.ax_eps.clear()

        episodes = list(
            range(
                1,
                len(self.reward_history) + 1,
            )
        )

        reward_avg = self._moving_average(
            self.reward_history
        )

        # Raw reward
        self.ax_curves.plot(
            episodes,
            self.reward_history,
            color="tab:green",
            alpha=0.35,
            linewidth=1.2,
            label="Reward",
        )

        # Moving-average reward
        self.ax_curves.plot(
            episodes,
            reward_avg,
            color="tab:green",
            linewidth=2.5,
            label=f"{self.ma_window}-Episode Avg",
        )

        # Loss
        if len(self.loss_history) == len(episodes):

            self.ax_curves.plot(
                episodes,
                self.loss_history,
                color="tab:red",
                linewidth=1.5,
                label="Loss",
            )

        self.ax_curves.set_xlabel(
            f"Episode (Last {self.max_history})"
        )

        self.ax_curves.set_ylabel(
            "Reward / Loss"
        )

        self.ax_curves.grid(
            True,
            linestyle="--",
            alpha=0.35,
        )

        # Epsilon (secondary axis)
        self.ax_eps.plot(
            episodes,
            self.epsilon_history,
            "--",
            color="tab:purple",
            linewidth=2,
            label="Epsilon",
        )

        self.ax_eps.set_ylabel(
            "Epsilon",
            color="tab:purple",
        )

        self.ax_eps.tick_params(
            axis="y",
            labelcolor="tab:purple",
        )

        # Combined legend
        h1, l1 = self.ax_curves.get_legend_handles_labels()
        h2, l2 = self.ax_eps.get_legend_handles_labels()

        self.ax_curves.legend(
            h1 + h2,
            l1 + l2,
            fontsize=8,
            loc="best",
        )

        self.ax_curves.set_title(
            "Learning Progress"
        )

    def _on_step(self, trainer: LocalTrainer) -> None:
        """
        Callback executed after every environment step.
        """

        if trainer.total_env_steps % self.render_every_n_steps != 0:
            return

        self._render_maze()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        plt.pause(self.pause_seconds)

    def _on_episode_end(
        self,
        trainer: LocalTrainer,
        summary: dict,
    ) -> None:
        """
        Callback executed after each episode.
        """

        self.reward_history.append(
            summary.get("total_reward", 0)
        )

        self.epsilon_history.append(
            summary.get("epsilon", 0)
        )

        # Works even if trainer doesn't report loss.
        self.loss_history.append(
            summary.get("loss", 0)
        )

        self._trim_histories()

        self._render_curves()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def run(
        self,
        num_episodes: int,
        verbose: bool = True,
    ):
        """
        Run training with live visualization.

        Returns the same history returned by
        LocalTrainer.run().
        """

        history = self.trainer.run(
            num_episodes=num_episodes,
            verbose=verbose,
            on_step=self._on_step,
            on_episode_end=self._on_episode_end,
        )

        plt.ioff()
        plt.show()

        return history


if __name__ == "__main__":

    from env.core.actions import NUM_ACTIONS
    from env.core.observations import NUM_CHANNELS
    from env.wrappers.pettingzoo_env import (
        CoFedMazeParallelEnv,
    )

    env = CoFedMazeParallelEnv(
        rows=9,
        cols=9,
        algorithm="recursive_backtracking",
        window_size=5,
        max_episode_steps=100,
    )

    trainer = LocalTrainer(
        env=env,
        in_channels=NUM_CHANNELS,
        num_actions=NUM_ACTIONS,
        buffer_capacity=100,
        sequence_length=15,
        batch_size=8,
        min_buffer_size=4,
        target_update_interval_episodes=10,
        epsilon_decay_episodes=100,
        master_seed=0,
    )

    view = LiveTrainerView(
        trainer,
        render_every_n_steps=1,
        pause_seconds=0.02,
        max_history=200,
        moving_average_window=10,
    )

    view.run(num_episodes=50)