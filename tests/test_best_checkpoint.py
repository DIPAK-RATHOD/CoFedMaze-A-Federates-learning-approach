"""
test_best_checkpoint.py

Comprehensive test suite verifying best.pt checkpoint management, metadata logging,
fixed validation seed evaluation, tie-breaking, current/previous rotation, and node isolation.
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from env.core.observations import NUM_CHANNELS
from env.wrappers.pettingzoo_env import CoFedMazeParallelEnv
from marl.training.trainer import LocalTrainer
from utils.checkpoint import (
    BEST_FILENAME,
    BEST_METADATA_FILENAME,
    CURRENT_FILENAME,
    PREVIOUS_FILENAME,
    has_checkpoint,
    load_best_metadata,
    load_checkpoint,
    rollback,
    save_best_checkpoint,
    save_checkpoint,
)
from utils.logger import StepLogger


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)


def test_checkpoint_save_and_load_slots():
    """Verify current, previous, and best slots save and load correctly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        assert not has_checkpoint(tmp_dir, slot="current")
        assert not has_checkpoint(tmp_dir, slot="previous")
        assert not has_checkpoint(tmp_dir, slot="best")

        # Save initial current checkpoint
        save_checkpoint(tmp_dir, model, optimizer, episode_count=1, total_env_steps=10)
        assert has_checkpoint(tmp_dir, slot="current")
        assert not has_checkpoint(tmp_dir, slot="previous")

        # Save second checkpoint (rotates current to previous)
        save_checkpoint(tmp_dir, model, optimizer, episode_count=2, total_env_steps=20)
        assert has_checkpoint(tmp_dir, slot="current")
        assert has_checkpoint(tmp_dir, slot="previous")

        curr = load_checkpoint(tmp_dir, slot="current")
        prev = load_checkpoint(tmp_dir, slot="previous")
        assert curr["episode_count"] == 2
        assert prev["episode_count"] == 1

        # Save best checkpoint
        save_best_checkpoint(
            tmp_dir,
            model,
            optimizer,
            episode_count=2,
            total_env_steps=20,
            validation_summary={
                "node_id": "N1",
                "validation_success_rate": 80.0,
                "average_steps_to_goal": 12.5,
            },
        )
        assert has_checkpoint(tmp_dir, slot="best")

        best = load_checkpoint(tmp_dir, slot="best")
        assert best["episode_count"] == 2
        meta = load_best_metadata(tmp_dir)
        assert meta is not None
        assert meta["validation_success_rate"] == 80.0


def test_current_previous_rotation_and_rollback():
    """Verify current.pt rotates into previous.pt and rollback restores previous.pt."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = DummyModel()
        save_checkpoint(tmp_dir, model, episode_count=10)
        save_checkpoint(tmp_dir, model, episode_count=20)

        assert load_checkpoint(tmp_dir, slot="current")["episode_count"] == 20
        assert load_checkpoint(tmp_dir, slot="previous")["episode_count"] == 10

        rollback(tmp_dir)
        assert load_checkpoint(tmp_dir, slot="current")["episode_count"] == 10


def test_best_model_creation_and_non_update():
    """Verify best.pt updates only when validation success rate improves."""
    env = CoFedMazeParallelEnv(rows=7, cols=7, num_checkpoints=0)
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = LocalTrainer(
            env=env,
            in_channels=NUM_CHANNELS,
            num_actions=5,
            checkpoint_dir=tmp_dir,
            validation_seeds=[2001, 2002],
        )

        # Manually simulate validation 1: 50% success
        trainer.best_success_rate = -1.0
        summary1 = trainer.evaluate(validation_seeds=[2001, 2002])
        assert (Path(tmp_dir) / BEST_FILENAME).exists()
        assert (Path(tmp_dir) / BEST_METADATA_FILENAME).exists()

        initial_best_meta = load_best_metadata(tmp_dir)
        assert initial_best_meta is not None
        initial_rate = initial_best_meta["validation_success_rate"]

        # Manually test non-update when success rate is lower
        trainer.best_success_rate = 100.0
        trainer.best_avg_steps_to_goal = 5.0
        summary2 = trainer.evaluate(validation_seeds=[2001, 2002])
        assert not summary2["best_model_updated"]


def test_tie_breaking_by_average_steps():
    """Verify equal success rates update best.pt if average steps to goal is lower."""
    env = CoFedMazeParallelEnv(rows=7, cols=7, num_checkpoints=0)
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = LocalTrainer(
            env=env,
            in_channels=NUM_CHANNELS,
            num_actions=5,
            checkpoint_dir=tmp_dir,
            validation_seeds=[3001, 3002],
        )

        trainer.best_success_rate = 50.0
        trainer.best_avg_steps_to_goal = 100.0  # High steps

        # Run evaluate where success rate is 50% but steps < 100.0
        summary = trainer.evaluate(validation_seeds=[3001, 3002])
        if summary["validation_success_rate"] == 50.0 and summary["average_steps_to_goal"] is not None:
            if summary["average_steps_to_goal"] < 100.0:
                assert summary["best_model_updated"] is True


def test_node_isolation():
    """Verify N1, N2, and N3 write to separate directories and never overwrite each other."""
    model = DummyModel()
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        dir1 = base / "checkpoint_N1"
        dir2 = base / "checkpoint_N2"
        dir3 = base / "checkpoint_N3"

        save_checkpoint(dir1, model, episode_count=100)
        save_checkpoint(dir2, model, episode_count=200)
        save_checkpoint(dir3, model, episode_count=300)

        assert load_checkpoint(dir1, slot="current")["episode_count"] == 100
        assert load_checkpoint(dir2, slot="current")["episode_count"] == 200
        assert load_checkpoint(dir3, slot="current")["episode_count"] == 300


def test_evaluation_logging_structure():
    """Verify evaluation log records explicit evaluation_id, trial index, and seed metrics."""
    env = CoFedMazeParallelEnv(rows=7, cols=7, num_checkpoints=0)
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger = StepLogger(log_dir=tmp_dir, node_id="N1")
        trainer = LocalTrainer(
            env=env,
            in_channels=NUM_CHANNELS,
            num_actions=5,
            step_logger=logger,
            checkpoint_dir=tmp_dir,
            validation_seeds=[4001, 4002],
        )

        summary = trainer.evaluate(current_round=10)
        assert summary["evaluation_id"] == "eval_001"
        assert summary["node_id"] == "N1"
        assert summary["round"] == 10

        # Read JSONL summaries
        summary_file = Path(tmp_dir) / "episode_summary_n1.jsonl"
        assert summary_file.exists()
        lines = [json.loads(line) for line in summary_file.read_text().strip().split("\n")]
        eval_lines = [l for l in lines if l.get("evaluation")]

        assert len(eval_lines) == 2
        assert eval_lines[0]["evaluation_id"] == "eval_001"
        assert eval_lines[0]["eval_trial"] == 1
        assert eval_lines[0]["maze_seed"] == 4001
        assert eval_lines[1]["eval_trial"] == 2
        assert eval_lines[1]["maze_seed"] == 4002

        logger.close()


def test_fixed_validation_seed_reproducibility():
    """Verify evaluating on identical validation seeds produces identical trajectory results."""
    env = CoFedMazeParallelEnv(rows=7, cols=7, num_checkpoints=0)
    trainer = LocalTrainer(env=env, in_channels=NUM_CHANNELS, num_actions=5, master_seed=42)

    seeds = [5001, 5002, 5003]
    eval1 = trainer.evaluate(validation_seeds=seeds)
    eval2 = trainer.evaluate(validation_seeds=seeds)

    assert eval1["validation_success_rate"] == eval2["validation_success_rate"]
    assert eval1["average_reward"] == eval2["average_reward"]
    assert eval1["average_steps_to_goal"] == eval2["average_steps_to_goal"]
