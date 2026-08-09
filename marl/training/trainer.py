"""
trainer.py

The local VDN training loop: runs episodes against a
CoFedMazeParallelEnv, stores each episode as a joint Trajectory,
periodically samples a batch and takes a gradient step, and
periodically hard-syncs the target network.
"""

from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch

from env.core.constants import AGENT_A, AGENT_B
from env.wrappers.pettingzoo_env import CoFedMazeParallelEnv
from marl.agents.action_selector import EpsilonGreedySelector
from marl.agents.policy import Policy
from marl.agents.vdn_agent import VDNAgent
from marl.losses.vdn_loss import compute_vdn_loss, hard_update_target_network
from marl.models.vdn import VDNModel
from marl.replay.replay_buffer import ReplayBuffer
from marl.replay.sampler import Sampler
from marl.replay.trajectory import Trajectory, Transition
from marl.training.optimizer import build_optimizer
from utils.logger import StepLogger


class LocalTrainer:
    """
    Owns one node's local (non-federated) VDN training loop: a single
    shared VDNModel used by both agents, a shared ReplayBuffer of joint
    Trajectories, and the online/target network pair the loss function
    needs.
    """

    def __init__(
        self,
        env: CoFedMazeParallelEnv,
        in_channels: int,
        num_actions: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        buffer_capacity: int = 200,
        sequence_length: int = 20,
        batch_size: int = 16,
        min_buffer_size: int = 8,
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        target_update_interval_episodes: int = 10,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 200,
        device: Optional[torch.device] = None,
        master_seed: Optional[int] = None,
        step_logger: Optional[StepLogger] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        auto_resume: bool = True,
    ) -> None:
        self.env = env
        self.device = device or torch.device("cpu")
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.target_update_interval_episodes = target_update_interval_episodes
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.step_logger = step_logger
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.last_loss: Optional[float] = None

        self.online_model = VDNModel(
            in_channels=in_channels, window_size=env.window_size, num_actions=num_actions,
            num_agents=2, embedding_dim=embedding_dim, hidden_dim=hidden_dim,
        ).to(self.device)
        self.target_model = VDNModel(
            in_channels=in_channels, window_size=env.window_size, num_actions=num_actions,
            num_agents=2, embedding_dim=embedding_dim, hidden_dim=hidden_dim,
        ).to(self.device)
        hard_update_target_network(self.online_model, self.target_model)

        self.optimizer = build_optimizer(self.online_model.parameters(), learning_rate=learning_rate)

        self._master_rng = random.Random(master_seed)
        self._selectors: Dict[str, EpsilonGreedySelector] = {
            AGENT_A: EpsilonGreedySelector(epsilon=epsilon_start, rng=self._child_rng()),
            AGENT_B: EpsilonGreedySelector(epsilon=epsilon_start, rng=self._child_rng()),
        }
        self._policies = {
            AGENT_A: Policy(self.online_model, agent_index=0, selector=self._selectors[AGENT_A]),
            AGENT_B: Policy(self.online_model, agent_index=1, selector=self._selectors[AGENT_B]),
        }
        self._agents: Dict[str, VDNAgent] = {
            aid: VDNAgent(self._policies[aid], device=self.device) for aid in (AGENT_A, AGENT_B)
        }

        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        self.sampler = Sampler(self.buffer, sequence_length=sequence_length, rng=self._child_rng())

        self.episode_count = 0
        self.total_env_steps = 0

        # Auto-resume from checkpoint if available
        if self.checkpoint_dir is not None:
            from utils.checkpoint import has_checkpoint
            if auto_resume:
                self.load_checkpoint(self.checkpoint_dir)
            elif has_checkpoint(self.checkpoint_dir):
                node_name = getattr(self.step_logger, "node_id", "Local")
                print(
                    f"[{node_name}] WARNING: Checkpoint file exists at {self.checkpoint_dir} "
                    f"but auto_resume=False. Starting fresh from episode 0, step 1, epsilon {self.epsilon_start:.2f}!"
                )

    def load_checkpoint(self, checkpoint_dir: Union[str, Path], slot: str = "current") -> bool:
        """
        Restore trainer state from checkpoint_dir if a checkpoint exists.
        """
        from utils.checkpoint import has_checkpoint, load_checkpoint as load_ckpt

        checkpoint_dir = Path(checkpoint_dir)
        if not has_checkpoint(checkpoint_dir, slot=slot):
            node_name = getattr(self.step_logger, "node_id", "Local")
            if checkpoint_dir.exists():
                print(
                    f"[{node_name}] WARNING: No checkpoint file found at {checkpoint_dir}. "
                    f"Starting fresh training from episode 0, step 1, epsilon {self.epsilon_start:.2f}."
                )
            return False

        payload = load_ckpt(checkpoint_dir, slot=slot)

        self.online_model.load_state_dict(payload["model_state"])

        if payload.get("target_model_state") is not None:
            self.target_model.load_state_dict(payload["target_model_state"])
        else:
            hard_update_target_network(self.online_model, self.target_model)

        if payload.get("optimizer_state") is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(payload["optimizer_state"])

        self.episode_count = payload.get("episode_count", 0)
        self.total_env_steps = payload.get("total_env_steps", 0)

        restored_eps = self._epsilon_for_episode(self.episode_count)
        for selector in self._selectors.values():
            selector.set_epsilon(restored_eps)

        node_name = getattr(self.step_logger, "node_id", "Local")
        print(
            f"[{node_name}] Resumed from checkpoint: episode={self.episode_count}, "
            f"total_env_steps={self.total_env_steps}, epsilon={restored_eps:.4f}"
        )
        return True

    def save_checkpoint(self, checkpoint_dir: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Persist full trainer state to checkpoint_dir.
        """
        from utils.checkpoint import save_checkpoint as save_ckpt

        meta = dict(metadata or {})
        if self.step_logger is not None:
            meta.setdefault("restart_count", self.step_logger.restart_count)
            meta.setdefault("run_id", self.step_logger.run_id)

        save_ckpt(
            directory=checkpoint_dir,
            model=self.online_model,
            optimizer=self.optimizer,
            episode_count=self.episode_count,
            total_env_steps=self.total_env_steps,
            target_model=self.target_model,
            metadata=meta,
        )

    def _child_rng(self) -> random.Random:
        return random.Random(self._master_rng.randrange(2**31))

    def _epsilon_for_episode(self, episode: int) -> float:
        if episode >= self.epsilon_decay_episodes:
            return self.epsilon_end
        frac = episode / self.epsilon_decay_episodes
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def _to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(obs).unsqueeze(0).to(self.device)

    def run_episode(
        self,
        on_step: Optional[Callable[["LocalTrainer"], None]] = None,
        eval_mode: bool = False,
    ) -> Trajectory:
        """
        Play one full episode against self.env using current online network.
        If eval_mode is True, sets epsilon = 0.0 for deterministic evaluation.
        """
        epsilon = 0.0 if eval_mode else self._epsilon_for_episode(self.episode_count)
        for selector in self._selectors.values():
            selector.set_epsilon(epsilon)

        seed = self._master_rng.randrange(2**31)
        obs, _ = self.env.reset(seed=seed)
        for agent in self._agents.values():
            agent.reset_hidden()

        trajectory = Trajectory(seed=seed, algorithm=self.env.algorithm)
        trajectory.evaluation = eval_mode
        step_in_ep = 0
        goal_reached = False

        while self.env.agents:
            actions = {}
            for agent_id in self.env.agents:
                obs_tensor = self._to_tensor(obs[agent_id])
                action, _ = self._agents[agent_id].act(obs_tensor)
                actions[agent_id] = action

            next_obs, rewards, terminations, truncations, infos = self.env.step(actions)
            done = any(terminations.values()) or any(truncations.values())
            reward = rewards[AGENT_A]

            if any(terminations.values()):
                goal_reached = True

            trajectory.append(Transition(
                obs=(obs[AGENT_A], obs[AGENT_B]),
                actions=(actions[AGENT_A], actions[AGENT_B]),
                reward=reward,
                next_obs=(next_obs[AGENT_A], next_obs[AGENT_B]),
                done=done,
            ))

            obs = next_obs
            if not eval_mode:
                self.total_env_steps += 1
            step_in_ep += 1

            if self.step_logger is not None:
                self.step_logger.log_step(
                    episode=self.episode_count,
                    step=step_in_ep,
                    total_env_steps=self.total_env_steps,
                    reward=reward,
                    loss=self.last_loss,
                    epsilon=epsilon,
                    actions=actions,
                    done=done,
                    goal_reached=goal_reached,
                    success=1 if goal_reached else 0,
                    steps_to_goal=step_in_ep if goal_reached else None,
                    timeout=done and not goal_reached,
                    evaluation=eval_mode,
                )

            if on_step is not None:
                on_step(self)

        trajectory.goal_reached = goal_reached
        trajectory.timeout = not goal_reached

        if not eval_mode:
            self.episode_count += 1
            if self.checkpoint_dir is not None:
                self.save_checkpoint(self.checkpoint_dir)

        return trajectory

    def evaluate(self, num_episodes: int = 5) -> Dict[str, Any]:
        """
        Run evaluation episodes with epsilon = 0.0 (deterministic policy)
        and record evaluation metrics.
        """
        successful_steps = []
        eval_rewards = []
        successes = 0

        for _ in range(num_episodes):
            traj = self.run_episode(eval_mode=True)
            tot_rew = traj.total_reward()
            eval_rewards.append(tot_rew)
            if traj.goal_reached:
                successes += 1
                successful_steps.append(len(traj))

        success_rate = (successes / num_episodes) * 100.0
        avg_steps = (sum(successful_steps) / len(successful_steps)) if successful_steps else None
        avg_reward = sum(eval_rewards) / num_episodes
        timeout_rate = ((num_episodes - successes) / num_episodes) * 100.0

        eval_summary = {
            "episode": self.episode_count,
            "evaluation": True,
            "success_rate": success_rate,
            "avg_steps_to_goal": avg_steps,
            "evaluation_reward": avg_reward,
            "timeout_rate": timeout_rate,
            "num_eval_episodes": num_episodes,
        }

        if self.step_logger is not None:
            self.step_logger.log_episode_summary(eval_summary)

        return eval_summary

    def train_step(self) -> Optional[float]:
        """One gradient step against a freshly sampled batch."""
        if not self.buffer.is_ready(self.min_buffer_size):
            return None
        batch = self.sampler.sample(self.batch_size)
        loss = compute_vdn_loss(self.online_model, self.target_model, batch, gamma=self.gamma)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.last_loss = loss.item()
        return self.last_loss

    def run(
        self,
        num_episodes: int,
        verbose: bool = True,
        on_step: Optional[Callable[["LocalTrainer"], None]] = None,
        on_episode_end: Optional[Callable[["LocalTrainer", dict], None]] = None,
        eval_interval: int = 10,
    ) -> List[dict]:
        """
        Run num_episodes of training with periodic evaluation.
        """
        history: List[dict] = []
        for _ in range(num_episodes):
            trajectory = self.run_episode(on_step=on_step, eval_mode=False)
            self.buffer.add(trajectory)
            loss = self.train_step()

            if self.episode_count % self.target_update_interval_episodes == 0:
                hard_update_target_network(self.online_model, self.target_model)

            summary = {
                "episode": self.episode_count,
                "length": len(trajectory),
                "total_reward": trajectory.total_reward(),
                "epsilon": self._epsilon_for_episode(self.episode_count),
                "loss": 0.0 if loss is None else float(loss),
                "goal_reached": trajectory.goal_reached,
                "success": 1 if trajectory.goal_reached else 0,
                "steps_to_goal": len(trajectory) if trajectory.goal_reached else None,
                "timeout": trajectory.timeout,
                "evaluation": False,
            }
            history.append(summary)

            if self.step_logger is not None:
                self.step_logger.log_episode_summary(summary)

            # Periodic evaluation with deterministic epsilon = 0 policy
            if eval_interval > 0 and self.episode_count % eval_interval == 0:
                eval_metrics = self.evaluate(num_episodes=3)
                summary["eval_success_rate"] = eval_metrics["success_rate"]
                summary["eval_reward"] = eval_metrics["evaluation_reward"]

            if verbose:
                goal_str = "SOLVED" if trajectory.goal_reached else "TIMEOUT"
                print(
                    f"Episode {summary['episode']:4d} | "
                    f"Status: {goal_str:7s} | "
                    f"Length: {summary['length']:3d} | "
                    f"Reward: {summary['total_reward']:8.3f} | "
                    f"Loss: {summary['loss']:.4f} | "
                    f"Epsilon: {summary['epsilon']:.3f}"
                )
            if on_episode_end is not None:
                on_episode_end(self, summary)
        return history


if __name__ == "__main__":
    from env.core.observations import NUM_CHANNELS
    from env.core.actions import NUM_ACTIONS

    env = CoFedMazeParallelEnv(rows=9, cols=9, algorithm="recursive_backtracking", window_size=5, max_episode_steps=60)
    trainer = LocalTrainer(
        env=env,
        in_channels=NUM_CHANNELS,
        num_actions=NUM_ACTIONS,
        buffer_capacity=50,
        sequence_length=10,
        batch_size=4,
        min_buffer_size=3,
        target_update_interval_episodes=5,
        epsilon_decay_episodes=20,
        master_seed=0,
    )
    trainer.run(num_episodes=5)
    print("marl/training/trainer.py self-test OK")