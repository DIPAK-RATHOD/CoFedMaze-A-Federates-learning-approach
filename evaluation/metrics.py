"""
metrics.py

Computes the core per-episode/per-node metrics table from the
glossary: Success Rate, Average Episode Return, Average Episode
Length, Collision Count, Timeout Rate. The baseline metric computation
every other evaluation/ file (not yet built: benchmark.py, ablation.py,
scalability.py, robustness.py) would build on.

Runs its OWN evaluation episodes directly against env+model (greedy
policy, same rollout pattern already established in
federation/validation/transfer_validation.py and coalition/merge.py),
rather than deriving metrics from already-collected training
Trajectories -- marl/replay/trajectory.py's Transition doesn't
distinguish SUCCESS from TIMEOUT/TRUNCATION (both just set done=True),
so success-rate-style metrics genuinely need a dedicated evaluation
pass with that distinction captured directly, not a post-hoc read of
training data that was never recorded with this level of detail.

Escape Time and Convergence Speed are deliberately NOT computed here:
    - Escape Time (wall-clock time to reach the exit) would need real
      timing instrumentation this project doesn't have yet
      (utils/timer.py is listed in the Directory Structure Reference
      but not built) -- Average Episode Length (step count) is
      reported instead, as the closest available proxy.
    - Convergence Speed (episodes/rounds needed to learn a good
      policy) is a property of a TRAINING RUN's trajectory over time,
      not a single evaluation pass -- it belongs in benchmark.py (not
      yet built), which would compare metrics across checkpoints taken
      during training, not in this file.

Assumption stated explicitly: with the current environment design (no
"agent death" mechanic), an episode ends either by SUCCESS (both
agents reach an unlocked exit) or by TIMEOUT (max_episode_steps
reached) -- there is no third outcome. success_rate + timeout_rate
therefore always sums to exactly 1.0; if a third termination mode is
ever added to env/wrappers/pettingzoo_env.py, this file's timeout_rate
computation (currently just "not success") would need to change too.
"""

from dataclasses import dataclass
from typing import List

import torch

from env.core.constants import AGENT_A, AGENT_B
from env.wrappers.pettingzoo_env import CoFedMazeParallelEnv
from marl.agents.action_selector import EpsilonGreedySelector
from marl.agents.policy import Policy
from marl.agents.vdn_agent import VDNAgent
from marl.models.vdn import VDNModel


@dataclass
class EpisodeResult:
    """Raw per-episode outcome, before aggregation into MetricsReport."""
    success: bool
    total_reward: float
    length: int
    collisions: int


@dataclass
class MetricsReport:
    """The glossary's metrics table, aggregated across N evaluation episodes."""
    num_episodes: int
    success_rate: float
    average_episode_return: float
    average_episode_length: float
    average_collision_count: float
    timeout_rate: float

    def __repr__(self) -> str:
        return (
            f"MetricsReport(n={self.num_episodes}, success_rate={self.success_rate:.1%}, "
            f"avg_return={self.average_episode_return:.3f}, avg_length={self.average_episode_length:.1f}, "
            f"avg_collisions={self.average_collision_count:.2f}, timeout_rate={self.timeout_rate:.1%})"
        )


def run_evaluation_episode(env: CoFedMazeParallelEnv, model: VDNModel, seed: int) -> EpisodeResult:
    """
    Play one full episode with GREEDY action selection on a FIXED seed,
    capturing success/truncation distinctly. Also counts collisions via
    each step's info dict, which env/wrappers/pettingzoo_env.py's
    step() already populates with "collided": bool per agent -- reused
    directly here rather than re-deriving collision detection.
    """
    selectors = {AGENT_A: EpsilonGreedySelector(epsilon=0.0), AGENT_B: EpsilonGreedySelector(epsilon=0.0)}
    policies = {
        AGENT_A: Policy(model, agent_index=0, selector=selectors[AGENT_A]),
        AGENT_B: Policy(model, agent_index=1, selector=selectors[AGENT_B]),
    }
    agents = {AGENT_A: VDNAgent(policies[AGENT_A]), AGENT_B: VDNAgent(policies[AGENT_B])}

    obs, _ = env.reset(seed=seed)
    for agent in agents.values():
        agent.reset_hidden()

    total_reward = 0.0
    length = 0
    collisions = 0
    success = False

    while env.agents:
        actions = {}
        for agent_id in env.agents:
            obs_tensor = torch.from_numpy(obs[agent_id]).unsqueeze(0)
            action, _ = agents[agent_id].act_greedy(obs_tensor)
            actions[agent_id] = action

        obs, rewards, terminations, truncations, infos = env.step(actions)
        total_reward += rewards[AGENT_A]
        length += 1
        collisions += sum(1 for info in infos.values() if info.get("collided", False))

        if any(terminations.values()):
            success = True

    return EpisodeResult(success=success, total_reward=total_reward, length=length, collisions=collisions)


def compute_metrics(env: CoFedMazeParallelEnv, model: VDNModel, seeds: List[int]) -> MetricsReport:
    """
    Run one evaluation episode per seed and aggregate into a MetricsReport.

    Raises:
        ValueError: If seeds is empty.
    """
    if not seeds:
        raise ValueError("seeds must not be empty")

    results = [run_evaluation_episode(env, model, seed) for seed in seeds]
    n = len(results)

    return MetricsReport(
        num_episodes=n,
        success_rate=sum(r.success for r in results) / n,
        average_episode_return=sum(r.total_reward for r in results) / n,
        average_episode_length=sum(r.length for r in results) / n,
        average_collision_count=sum(r.collisions for r in results) / n,
        timeout_rate=sum(not r.success for r in results) / n,
    )


if __name__ == "__main__":
    from env.core.actions import NUM_ACTIONS
    from env.core.observations import NUM_CHANNELS

    env = CoFedMazeParallelEnv(rows=9, cols=9, algorithm="recursive_backtracking", window_size=5, max_episode_steps=30)
    model = VDNModel(in_channels=NUM_CHANNELS, window_size=5, num_actions=NUM_ACTIONS, num_agents=2)

    report = compute_metrics(env, model, seeds=[1, 2, 3, 4, 5])
    print(report)

    assert report.num_episodes == 5
    assert 0.0 <= report.success_rate <= 1.0
    assert 0.0 <= report.timeout_rate <= 1.0
    # Stated assumption: with no third termination mode, these must sum to exactly 1.0.
    assert abs((report.success_rate + report.timeout_rate) - 1.0) < 1e-9
    assert report.average_episode_length > 0
    assert report.average_collision_count >= 0

    try:
        compute_metrics(env, model, seeds=[])
        print("FAIL: should have raised")
    except ValueError as e:
        print("OK, empty seeds rejected:", e)

    print("OK")
