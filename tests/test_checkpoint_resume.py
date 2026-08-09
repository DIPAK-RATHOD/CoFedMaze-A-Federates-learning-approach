"""
test_checkpoint_resume.py

Unit test for full state persistence, auto-resume, and restart tracking.
"""

from pathlib import Path
import tempfile

from env.core.actions import NUM_ACTIONS
from env.core.observations import NUM_CHANNELS
from env.wrappers.pettingzoo_env import CoFedMazeParallelEnv
from marl.training.trainer import LocalTrainer
from utils.checkpoint import has_checkpoint, load_checkpoint
from utils.logger import StepLogger


def test_checkpoint_resume_and_restart_count():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        ckpt_dir = tmp_path / "checkpoints"
        log_dir = tmp_path / "logs"

        env1 = CoFedMazeParallelEnv(rows=9, cols=9, algorithm="recursive_backtracking", max_episode_steps=20)
        logger1 = StepLogger(log_dir=log_dir, node_id="N1")
        assert logger1.restart_count == 1
        assert logger1.run_id == "run_1"

        trainer1 = LocalTrainer(
            env=env1,
            in_channels=NUM_CHANNELS,
            num_actions=NUM_ACTIONS,
            buffer_capacity=20,
            sequence_length=5,
            batch_size=2,
            min_buffer_size=2,
            master_seed=42,
            step_logger=logger1,
            checkpoint_dir=ckpt_dir,
            auto_resume=False,
        )

        # Run 5 episodes
        history1 = trainer1.run(num_episodes=5, verbose=False)
        assert trainer1.episode_count == 5
        assert trainer1.total_env_steps > 0
        eps_after_ep5 = trainer1._epsilon_for_episode(trainer1.episode_count)
        steps_after_ep5 = trainer1.total_env_steps
        logger1.close()

        assert has_checkpoint(ckpt_dir)
        ckpt_data = load_checkpoint(ckpt_dir)
        assert ckpt_data["episode_count"] == 5
        assert ckpt_data["total_env_steps"] == steps_after_ep5

        # Simulate process restart (Kaggle session restart)
        logger2 = StepLogger(log_dir=log_dir, node_id="N1")
        assert logger2.restart_count == 2
        assert logger2.run_id == "run_2"

        env2 = CoFedMazeParallelEnv(rows=9, cols=9, algorithm="recursive_backtracking", max_episode_steps=20)
        trainer2 = LocalTrainer(
            env=env2,
            in_channels=NUM_CHANNELS,
            num_actions=NUM_ACTIONS,
            buffer_capacity=20,
            sequence_length=5,
            batch_size=2,
            min_buffer_size=2,
            master_seed=42,
            step_logger=logger2,
            checkpoint_dir=ckpt_dir,
            auto_resume=True,  # Should auto-restore episode 5 and total_env_steps
        )

        assert trainer2.episode_count == 5
        assert trainer2.total_env_steps == steps_after_ep5
        assert abs(trainer2._epsilon_for_episode(trainer2.episode_count) - eps_after_ep5) < 1e-6

        # Run 1 more episode -- should be episode 5 (0-indexed before run_episode completes, resulting in episode_count=6)
        history2 = trainer2.run(num_episodes=1, verbose=False)
        assert trainer2.episode_count == 6
        assert trainer2.total_env_steps > steps_after_ep5
        assert history2[0]["episode"] == 6
        logger2.close()


if __name__ == "__main__":
    test_checkpoint_resume_and_restart_count()
    print("test_checkpoint_resume.py OK")
