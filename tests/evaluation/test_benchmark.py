from evaluation.benchmark import convergence_episode


def test_convergence_episode_uses_trailing_reward_window():
    history = [
        {"episode": 1, "total_reward": 0.0},
        {"episode": 2, "total_reward": 2.0},
        {"episode": 3, "total_reward": 2.0},
    ]

    assert convergence_episode(history, reward_threshold=1.0, window=2) == 2
    assert convergence_episode(history, reward_threshold=3.0, window=2) is None
