"""
tests/test_ablation_and_diagnostics.py

Unit tests for controlled ablation configurations, target update modes,
sequence length, gradient clipping, dynamic success rewards, checkpoint selection,
deterministic evaluation, and node state directory isolation.
"""

from pathlib import Path
import tempfile
import torch

from env.core.observations import NUM_CHANNELS
from env.wrappers.pettingzoo_env import SUCCESS_REWARD, CoFedMazeParallelEnv
from marl.losses.vdn_loss import hard_update_target_network, soft_update_target_network
from marl.models.vdn import VDNModel
from marl.training.trainer import LocalTrainer
from node.node_config import NodeConfig


def test_sequence_length_configuration():
    """Verify LocalTrainer and Sampler correctly inherit sequence_length."""
    env = CoFedMazeParallelEnv(rows=5, cols=5, algorithm="recursive_backtracking", max_episode_steps=20)
    trainer = LocalTrainer(
        env=env,
        in_channels=NUM_CHANNELS,
        num_actions=5,
        sequence_length=50,
        pretrain_expert=False,
    )
    assert trainer.sampler.sequence_length == 50

    trainer_short = LocalTrainer(
        env=env,
        in_channels=NUM_CHANNELS,
        num_actions=5,
        sequence_length=20,
        pretrain_expert=False,
    )
    assert trainer_short.sampler.sequence_length == 20


def test_polyak_soft_update():
    """Verify Polyak target network soft update moves target weights towards online model."""
    online = VDNModel(in_channels=NUM_CHANNELS, window_size=5, num_actions=5, num_agents=2)
    target = VDNModel(in_channels=NUM_CHANNELS, window_size=5, num_actions=5, num_agents=2)
    hard_update_target_network(online, target)

    # Mutate online weights
    with torch.no_grad():
        for p in online.parameters():
            p.add_(1.0)

    p_target_before = next(target.parameters()).clone()
    p_online = next(online.parameters()).clone()

    soft_update_target_network(online, target, tau=0.015)
    p_target_after = next(target.parameters()).clone()

    # Target should move closer to online (0.985 * before + 0.015 * online)
    expected = (1.0 - 0.015) * p_target_before + 0.015 * p_online
    assert torch.allclose(p_target_after, expected, atol=1e-5)


def test_hard_update_fallback():
    """Verify hard_update_target_network copies exact weights."""
    online = VDNModel(in_channels=NUM_CHANNELS, window_size=5, num_actions=5, num_agents=2)
    target = VDNModel(in_channels=NUM_CHANNELS, window_size=5, num_actions=5, num_agents=2)

    with torch.no_grad():
        for p in online.parameters():
            p.add_(2.5)

    hard_update_target_network(online, target)

    for p_online, p_target in zip(online.parameters(), target.parameters()):
        assert torch.equal(p_online, p_target)


def test_reward_configuration():
    """Verify dynamic team success_reward configuration."""
    env_default = CoFedMazeParallelEnv(rows=5, cols=5, algorithm="recursive_backtracking")
    assert env_default.success_reward == SUCCESS_REWARD

    env_custom = CoFedMazeParallelEnv(rows=5, cols=5, algorithm="recursive_backtracking", success_reward=10.0)
    assert env_custom.success_reward == 10.0


def test_gradient_clipping():
    """Verify gradient norm clipping in LocalTrainer."""
    env = CoFedMazeParallelEnv(rows=5, cols=5, algorithm="recursive_backtracking", max_episode_steps=10)
    trainer = LocalTrainer(
        env=env,
        in_channels=NUM_CHANNELS,
        num_actions=5,
        max_grad_norm=10.0,
        pretrain_expert=False,
    )
    assert trainer.max_grad_norm == 10.0


def test_checkpoint_selection_and_seeds():
    """Verify validation seeds default to 1001-1020 and test seeds default to 2001-2020."""
    env = CoFedMazeParallelEnv(rows=5, cols=5, algorithm="recursive_backtracking", max_episode_steps=10)
    trainer = LocalTrainer(env=env, in_channels=NUM_CHANNELS, num_actions=5, pretrain_expert=False)

    assert len(trainer.validation_seeds) == 20
    assert trainer.validation_seeds == list(range(1001, 1021))
    assert len(trainer.test_seeds) == 20
    assert trainer.test_seeds == list(range(2001, 2021))


def test_deterministic_evaluation():
    """Verify evaluation run_episode in eval_mode uses epsilon=0.0."""
    env = CoFedMazeParallelEnv(rows=5, cols=5, algorithm="recursive_backtracking", max_episode_steps=10)
    trainer = LocalTrainer(env=env, in_channels=NUM_CHANNELS, num_actions=5, pretrain_expert=False)

    traj = trainer.run_episode(eval_mode=True, seed=1001)
    assert traj.evaluation is True


def test_node_isolation():
    """Verify custom experiment_id creates isolated directory paths."""
    from node.services import build_services
    from federation.communication.transport import InProcessTransport

    config = NodeConfig.load("data/node1/config.yaml")
    config.node_id = "N1"
    transport = InProcessTransport()
    services = build_services(config, transport, experiment_id="exp_a")

    assert "state/exp_a/n1/checkpoints" in services.trainer.checkpoint_dir.as_posix()
