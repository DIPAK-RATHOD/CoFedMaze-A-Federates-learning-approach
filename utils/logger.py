"""
logger.py

Structured step-level and episode-level logging system for CoFedMaze.

Performs immediate atomic logging after EVERY training step and episode to disk
(formatted as JSON Lines .jsonl). Includes goal_reached, success, steps_to_goal,
timeout, and evaluation flags for research-grade verification of maze solving.
"""

from dataclasses import asdict, dataclass
import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

PathLike = Union[str, Path]


@dataclass
class StepLogEntry:
    """Represents a single environment/training step record."""

    timestamp: str
    node_id: str
    run_id: str
    restart_count: int
    episode: int
    step: int
    total_env_steps: int
    reward: float
    loss: Optional[float]
    epsilon: float
    actions: Dict[str, int]
    done: bool
    goal_reached: Optional[bool] = None
    success: Optional[int] = None
    steps_to_goal: Optional[int] = None
    timeout: Optional[bool] = None
    evaluation: bool = False


class StepLogger:
    """
    Logs step-by-step and episode-by-episode metrics directly to disk.
    """

    def __init__(
        self,
        log_dir: PathLike,
        node_id: str = "N1",
        filename_prefix: str = "step_metrics",
        auto_flush: bool = True,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.node_id = node_id
        self.auto_flush = auto_flush

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.step_log_path = self.log_dir / f"{filename_prefix}_{node_id.lower()}.jsonl"
        self.episode_log_path = self.log_dir / f"episode_summary_{node_id.lower()}.jsonl"

        max_restart = 0
        if self.step_log_path.exists():
            try:
                with open(self.step_log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                record = json.loads(line)
                                rc = record.get("restart_count", 0)
                                if rc > max_restart:
                                    max_restart = rc
                            except Exception:
                                pass
            except Exception:
                pass

        self.restart_count = max_restart + 1
        self.run_id = f"run_{self.restart_count}"

        self._step_file = open(self.step_log_path, "a", encoding="utf-8")
        self._episode_file = open(self.episode_log_path, "a", encoding="utf-8")

    def log_step(
        self,
        episode: int,
        step: int,
        total_env_steps: int,
        reward: float,
        loss: Optional[float],
        epsilon: float,
        actions: Dict[str, int],
        done: bool,
        goal_reached: Optional[bool] = None,
        success: Optional[int] = None,
        steps_to_goal: Optional[int] = None,
        timeout: Optional[bool] = None,
        evaluation: bool = False,
    ) -> StepLogEntry:
        """Record and persist a single step metric to disk."""
        entry = StepLogEntry(
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            node_id=self.node_id,
            run_id=self.run_id,
            restart_count=self.restart_count,
            episode=episode,
            step=step,
            total_env_steps=total_env_steps,
            reward=float(reward),
            loss=float(loss) if loss is not None else None,
            epsilon=float(epsilon),
            actions={k: int(v) for k, v in actions.items()},
            done=bool(done),
            goal_reached=goal_reached,
            success=success if success is not None else (1 if goal_reached else 0 if goal_reached is not None else None),
            steps_to_goal=steps_to_goal,
            timeout=timeout,
            evaluation=evaluation,
        )

        self._step_file.write(json.dumps(asdict(entry)) + "\n")
        if self.auto_flush:
            self._step_file.flush()

        return entry

    def log_episode_summary(self, summary: Dict[str, Any]) -> None:
        """Record and persist an end-of-episode summary."""
        record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "node_id": self.node_id,
            "run_id": self.run_id,
            "restart_count": self.restart_count,
            **summary,
        }
        self._episode_file.write(json.dumps(record) + "\n")
        if self.auto_flush:
            self._episode_file.flush()

    def flush(self) -> None:
        if not self._step_file.closed:
            self._step_file.flush()
        if not self._episode_file.closed:
            self._episode_file.flush()

    def close(self) -> None:
        self.flush()
        if not self._step_file.closed:
            self._step_file.close()
        if not self._episode_file.closed:
            self._episode_file.close()

    def __enter__(self) -> "StepLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        logger = StepLogger(log_dir=tmp_dir, node_id="N1")
        logger.log_step(1, 1, 1, -0.01, 0.05, 0.99, {"agent_0": 0, "agent_1": 2}, False, goal_reached=False)
        logger.close()
        print("utils/logger.py self-test OK")
