"""
auto_evaluator.py

Automated Post-Training Visualization & Evaluation Generator for CoFedMaze nodes.

When training completes for a node, this module automatically generates and persists
all evaluation plots and summary statistics into each node's evaluation folder:
  `outputs/{node_id}/evaluation/`

Generated Artifacts:
  - reward_curve.png: Episode vs Team Return
  - loss_curve.png: Episode vs VDN Loss
  - epsilon_curve.png: Episode vs Epsilon Decay
  - coalition_history.png: Round vs Coalition Size (if available)
  - maze_layout.png: Rendered Maze Structure Map
  - dashboard.png: Combined Multi-Panel Report Dashboard
  - evaluation_summary.json: Structured Node Metrics Summary JSON
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualization.coalition_plot import plot_coalition_history
from visualization.dashboard import create_dashboard
from visualization.loss_curve import plot_loss_curve
from visualization.maze_plot import save_maze_plot
from visualization.reward_curve import plot_reward_curve

PathLike = Union[str, Path]


def plot_epsilon_curve(
    history: List[Dict[str, Any]],
    output_path: PathLike,
    title: str = "CoFedMaze Epsilon Decay",
) -> Path:
    """Plot epsilon decay schedule across training episodes."""
    episodes = [int(item["episode"]) for item in history]
    epsilons = [float(item["epsilon"]) for item in history]

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(episodes, epsilons, color="tab:green", linewidth=2, label="Epsilon (ε)")
    axis.set(title=title, xlabel="Episode", ylabel="Epsilon")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def generate_node_evaluation_report(
    node_id: str,
    output_dir: Optional[PathLike] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    coalition_history: Optional[List[Dict[str, Any]]] = None,
    env: Optional[Any] = None,
    log_dir: Optional[PathLike] = None,
) -> Path:
    """
    Generate all post-training plots and summary artifacts for a node,
    saving everything into `outputs/{node_id}/evaluation/`.
    """
    node_name = node_id.lower().replace("n", "node", 1)
    if output_dir is None:
        eval_dir = Path("outputs") / node_name / "evaluation"
    else:
        eval_dir = Path(output_dir)

    eval_dir.mkdir(parents=True, exist_ok=True)

    # Read history from log_dir if not directly provided
    if (history is None or len(history) == 0) and log_dir is not None:
        log_file = Path(log_dir) / f"episode_summary_{node_id.lower()}.jsonl"
        if log_file.exists():
            history = []
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            history.append(json.loads(line))
            except Exception:
                pass

    panels: Dict[str, Path] = {}

    # 1. Reward Curve
    if history and len(history) > 0:
        reward_path = eval_dir / "reward_curve.png"
        plot_reward_curve(history, reward_path, title=f"Node {node_id} Training Return")
        panels["Reward Curve"] = reward_path

        # 2. Loss Curve
        loss_path = eval_dir / "loss_curve.png"
        plot_loss_curve(history, loss_path, title=f"Node {node_id} VDN Training Loss")
        panels["Loss Curve"] = loss_path

        # 3. Epsilon Curve
        eps_path = eval_dir / "epsilon_curve.png"
        plot_epsilon_curve(history, eps_path, title=f"Node {node_id} Epsilon Decay")
        panels["Epsilon Decay"] = eps_path

    # 4. Coalition History
    if coalition_history and len(coalition_history) > 0:
        coalition_path = eval_dir / "coalition_history.png"
        try:
            plot_coalition_history(coalition_history, coalition_path, title=f"Node {node_id} Coalition History")
            panels["Coalition History"] = coalition_path
        except Exception:
            pass

    # 5. Maze Layout Map
    if env is not None and hasattr(env, "maze"):
        maze_path = eval_dir / "maze_layout.png"
        save_maze_plot(env.maze, maze_path)
        panels["Maze Structure"] = maze_path

    # 6. Combined Dashboard Image
    if panels:
        dashboard_path = eval_dir / "dashboard.png"
        create_dashboard(panels, dashboard_path, title=f"Node {node_id} Evaluation Dashboard")
        print(f"\n[{node_id}] Generated node evaluation dashboard: {dashboard_path}")

    # 7. Summary JSON Report
    if history and len(history) > 0:
        rewards = [float(h["total_reward"]) for h in history]
        last_10 = rewards[-10:] if len(rewards) >= 10 else rewards
        summary_data = {
            "node_id": node_id,
            "total_episodes": len(history),
            "final_episode": history[-1].get("episode", len(history)),
            "final_reward": rewards[-1],
            "mean_reward_last_10": sum(last_10) / len(last_10),
            "final_epsilon": float(history[-1].get("epsilon", 0.05)),
            "final_loss": float(history[-1].get("loss", 0.0)),
            "saved_plots": [str(p) for p in panels.values()],
        }
        summary_json_path = eval_dir / "evaluation_summary.json"
        summary_json_path.write_text(json.dumps(summary_data, indent=2) + "\n", encoding="utf-8")
        print(f"[{node_id}] Saved evaluation summary JSON: {summary_json_path}")

    return eval_dir


if __name__ == "__main__":
    from env.wrappers.pettingzoo_env import CoFedMazeParallelEnv

    env = CoFedMazeParallelEnv(rows=7, cols=7, algorithm="recursive_backtracking")
    env.reset(seed=42)

    sample_history = [
        {"episode": i, "total_reward": -0.5 + i * 0.1, "loss": 0.5 / (i + 1), "epsilon": max(0.05, 1.0 - i * 0.1)}
        for i in range(1, 15)
    ]

    out = generate_node_evaluation_report(node_id="N1", history=sample_history, env=env)
    print(f"auto_evaluator.py self-test OK. Created report at {out}")
