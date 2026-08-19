"""
test_post_train_selector.py

Unit test suite for the post-training model selection utility.
Verifies byte-for-byte exact checkpoint copying, unmutated current/previous files,
correct metadata calculation, and fixed validation seed consistency.
"""

import hashlib
import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from utils.checkpoint import BEST_FILENAME, BEST_METADATA_FILENAME, CURRENT_FILENAME, PREVIOUS_FILENAME, save_checkpoint
from utils.post_train_best_selector import DEFAULT_VALIDATION_SEEDS, compute_file_hash, process_node


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)


def test_post_train_selector_execution_and_integrity(monkeypatch):
    """
    Test that process_node correctly evaluates candidates, copies the best byte-for-byte,
    leaves current.pt and previous.pt unchanged, and writes matching metadata.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        ckpt_dir = base_dir / "n1" / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        model = DummyModel()

        # Save dummy current.pt and previous.pt
        save_checkpoint(ckpt_dir, model, episode_count=100, total_env_steps=1000)
        save_checkpoint(ckpt_dir, model, episode_count=200, total_env_steps=2000)

        current_path = ckpt_dir / CURRENT_FILENAME
        previous_path = ckpt_dir / PREVIOUS_FILENAME
        best_path = ckpt_dir / BEST_FILENAME
        meta_path = ckpt_dir / BEST_METADATA_FILENAME

        curr_hash_before = compute_file_hash(current_path)
        prev_hash_before = compute_file_hash(previous_path)

        # Mock NodeConfig.load and LocalTrainer.run_episode to avoid heavy training
        class MockTraj:
            def __init__(self, goal_reached=True, length=10):
                self.goal_reached = goal_reached
                self.timeout = not goal_reached
                self._length = length

            def total_reward(self):
                return 15.0 if self.goal_reached else 0.0

            def __len__(self):
                return self._length

        def mock_evaluate_candidate(trainer, dir_path, slot, seeds, node_id):
            if slot == "current":
                # 100% success, avg steps 8.0
                metrics = {
                    "node_id": node_id,
                    "slot": slot,
                    "successes": len(seeds),
                    "total_trials": len(seeds),
                    "validation_success_rate": 100.0,
                    "timeout_rate": 0.0,
                    "average_steps_to_goal": 8.0,
                    "average_reward": 15.0,
                    "checkpoint_episode": 200,
                    "total_env_steps": 2000,
                }
            else:
                # 80% success, avg steps 12.0
                metrics = {
                    "node_id": node_id,
                    "slot": slot,
                    "successes": int(len(seeds) * 0.8),
                    "total_trials": len(seeds),
                    "validation_success_rate": 80.0,
                    "timeout_rate": 20.0,
                    "average_steps_to_goal": 12.0,
                    "average_reward": 10.0,
                    "checkpoint_episode": 100,
                    "total_env_steps": 1000,
                }
            return metrics, []

        monkeypatch.setattr("utils.post_train_best_selector.evaluate_checkpoint_candidate", mock_evaluate_candidate)

        # Mock NodeConfig.load
        class MockConfig:
            maze_rows = 7
            maze_cols = 7
            maze_algorithm = "recursive_backtracking"
            window_size = 5
            num_checkpoints = 2
            num_obstacles = 0
            num_key_door_pairs = 0

        monkeypatch.setattr("node.node_config.NodeConfig.load", lambda path: MockConfig())

        # Monkeypatch Path in process_node to look inside tmp_dir
        monkeypatch.setattr("utils.post_train_best_selector.Path", lambda p: ckpt_dir if "state" in str(p) else Path(p))

        result = process_node("N1", DEFAULT_VALIDATION_SEEDS)

        # 1. Verify best.pt exists
        assert best_path.exists()

        # 2. Verify selected slot is "current"
        assert result["selected_slot"] == "current"

        # 3. Verify best.pt is exact byte-for-byte copy of current.pt
        best_hash = compute_file_hash(best_path)
        assert best_hash == curr_hash_before

        # 4. Verify current.pt and previous.pt were NOT modified
        curr_hash_after = compute_file_hash(current_path)
        prev_hash_after = compute_file_hash(previous_path)
        assert curr_hash_after == curr_hash_before
        assert prev_hash_after == prev_hash_before

        # 5. Verify best_metadata.json metadata matches
        assert meta_path.exists()
        with open(meta_path, "r") as f:
            meta = json.load(f)

        assert meta["node_id"] == "N1"
        assert meta["source_checkpoint"] == "current"
        assert meta["validation_success_rate"] == 100.0
        assert meta["average_steps_to_goal"] == 8.0
        assert meta["validation_seeds"] == DEFAULT_VALIDATION_SEEDS
