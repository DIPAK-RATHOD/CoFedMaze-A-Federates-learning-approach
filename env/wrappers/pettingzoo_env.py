"""
pettingzoo_env.py

Wraps a Maze + two Agents + task-variant objects (checkpoints, key/door
pairs, obstacles) as a PettingZoo Parallel API environment. This is the
single integration point env/ and marl/ meet at -- marl/ never touches
Maze/Grid/Cell directly, and env/ never knows about VDN/agents.

============================================================================
REWARD POLICY -- STRUCTURE DECIDED, MAGNITUDES STILL PLACEHOLDERS
============================================================================
Project memory listed reward policy as unresolved. The STRUCTURE below
was decided in conversation: shared team reward (required by VDN's
single Q_tot target, and matches the workplan's "team rewards" wording),
lightly shaped via potential-based distance-to-exit shaping (chosen
because a purely sparse reward makes the Transfer-Benefit metric's
R_old denominator numerically unstable on small validation subsets),
an extra penalty on collision, and an immediate one-time bonus per
checkpoint/key/door event. See the reward constants and
_default_reward_fn below. The MAGNITUDES of those constants are NOT
tuned or validated -- only their sign and rough relative scale are
deliberate. Override via the `reward_fn` constructor parameter (note:
its signature changed to accept a StepEvent, not a bare success bool --
see StepEvent below).
============================================================================

Design decisions made here (previously open, now resolved -- flag any
of these if they're wrong):

  - Two-agent spawn: AGENT_A spawns at maze.start_position. AGENT_B
    spawns at a random unoccupied logical neighbor of start if one
    exists, else a random free cell elsewhere. (This was explicitly
    flagged as unresolved in env/objects/agent.py's docstring -- this
    file is where that decision now lives.)
  - Success condition: BOTH agents simultaneously occupy an UNLOCKED
    exit. If num_checkpoints > 0, the exit starts locked and unlocks
    automatically once every checkpoint has been reached by either
    agent (team-wide completion, not per-agent).
  - Key/checkpoint pickup is AUTOMATIC on entering their cell (neither
    blocks movement, so this is safe) -- INTERACT's only job is
    unlocking an adjacent locked door using a held matching key.
  - Illegal moves (wall, out-of-bounds, blocked by object, or the other
    agent's current cell) are silently absorbed as no-ops, never raised
    out of step() -- standard practice for RL environments.
  - Collisions: agents are processed in a fixed order (AGENT_A, then
    AGENT_B) each step. A later agent trying to move into a cell the
    earlier agent just vacated-or-claimed this same step will find it
    already resolved by Agent.place_at()'s existing occupancy check --
    no separate collision bookkeeping was needed. Swap attempts (each
    agent moving into the other's current cell) both fail for the same
    reason. This means AGENT_A has a structural first-mover advantage
    in same-cell contention -- a known simplification, not a claimed
    fair tie-break.
  - Obstacles are static only. A "moving obstacle" variant would be a
    subclass of Obstacle (per its own docstring) with its own per-step
    update hook -- not implemented here.
  - "Limited-communication" and "partial-observability" task variants
    are not separately implemented: partial observability already
    falls out of the egocentric window observation for free; limited
    communication is a federation-layer concern (knowledge_graph/,
    coalition/), not something a single node's environment enforces.

Scope boundary carried over from door.py/obstacle.py: this file is
exactly the "whatever validates a move" composition point those files'
docstrings pointed to -- object-level blocking (locked doors, static
obstacles) is checked HERE, composed with Agent.move_to()'s structural
passability check, not inside Agent itself.
"""

import functools
import random
from typing import Callable, Dict, List, NamedTuple, Optional, Set, Tuple

import gymnasium
import numpy as np
from pettingzoo import ParallelEnv

from env.core.actions import INTERACT, NUM_ACTIONS, direction_for
from env.core.constants import AGENT_A, AGENT_B, DIRECTIONS
from env.core.maze import Maze
from env.core.observations import NUM_CHANNELS, build_observation
from env.generator.generator_factory import create_generator
from env.objects.agent import Agent
from env.objects.checkpoint import Checkpoint
from env.objects.door import Door
from env.objects.exit import Exit
from env.objects.key import Key
from env.objects.obstacle import Obstacle
from env.validators.bfs_validator import BFSValidator

AgentID = str
LogicalPos = Tuple[int, int]


class StepEvent(NamedTuple):
    """
    Everything a reward_fn needs to know about what happened in one
    step -- passed instead of a single 'success' bool now that the
    reward policy is a real decision (see module docstring), not a
    placeholder. This is a BREAKING CHANGE to reward_fn's signature
    from the earlier placeholder version (env, success: bool) -> ...;
    it is now (env, event: StepEvent) -> ....

    collided_agents: agent ids whose attempted move this step was
        rejected (wall, obstacle, locked door, or the other agent's cell).
    newly_reached_checkpoints / newly_collected_keys / newly_unlocked_doors:
        ids of objects that changed state to "done" THIS step specifically
        (not objects that were already done before this step), so a
        one-time bonus fires exactly once per object, not every step.
    success: both agents simultaneously on an unlocked exit this step.
    potential_before / potential_after: team-level shaping potential
        (see _team_potential()) before and after this step's moves,
        for potential-based shaping (Ng et al. 1999): reward includes
        gamma * potential_after - potential_before, which is provably
        policy-invariant (doesn't change the optimal policy) unlike
        arbitrary shaping terms.
    """
    collided_agents: Set[AgentID]
    newly_reached_checkpoints: List[str]
    newly_collected_keys: List[str]
    newly_unlocked_doors: List[str]
    newly_reached_exits: Set[AgentID]
    success: bool
    potential_before: float
    potential_after: float


RewardFn = Callable[["CoFedMazeParallelEnv", StepEvent], Dict[AgentID, float]]

# ---------------------------------------------------------------------------
# Reward policy -- decided per the discussion in this conversation:
#   - Shared TEAM reward (required by VDN's single Q_tot TD target).
#   - Lightly shaped via potential-based distance-to-exit shaping.
#   - Extra penalty on collision (wall/obstacle/locked-door/other-agent).
#   - Immediate one-time bonus for each checkpoint/key/door/exit event.
# ---------------------------------------------------------------------------
STEP_PENALTY = -0.01
COLLISION_PENALTY = -0.05
CHECKPOINT_BONUS = 1.0
KEY_BONUS = 0.5
DOOR_BONUS = 1.0
INDIVIDUAL_GOAL_BONUS = 3.0
SUCCESS_REWARD = 25.0
SHAPING_GAMMA = 0.99


def _default_reward_fn(env: "CoFedMazeParallelEnv", event: StepEvent) -> Dict[AgentID, float]:
    """
    Shaped team reward policy with one-time individual milestone bonuses & parking support.
    """
    shaping = SHAPING_GAMMA * event.potential_after - event.potential_before
    rewards = {}

    for aid in env.possible_agents:
        is_at_exit = False
        if hasattr(env, "_exit_obj") and hasattr(env, "_agent_objs") and aid in env._agent_objs:
            is_at_exit = env._exit_obj.is_usable_by(env.maze, env._agent_objs[aid].position)

        # Suppress step penalty if parked at exit goal waiting for partner
        val = 0.0 if is_at_exit else STEP_PENALTY

        if aid in event.collided_agents:
            val += COLLISION_PENALTY

        val += CHECKPOINT_BONUS * len(event.newly_reached_checkpoints)
        val += KEY_BONUS * len(event.newly_collected_keys)
        val += DOOR_BONUS * len(event.newly_unlocked_doors)

        # ONE-TIME individual goal bonus when an agent NEWLY reaches the exit goal!
        if aid in event.newly_reached_exits:
            val += INDIVIDUAL_GOAL_BONUS

        val += shaping

        if event.success:
            val += SUCCESS_REWARD

        rewards[aid] = val

    return rewards


class CoFedMazeParallelEnv(ParallelEnv):
    """
    PettingZoo Parallel API environment wrapping one CoFedMaze node's
    maze, two agents, and optional task-variant objects.
    """

    metadata = {"render_modes": ["ascii", None], "name": "cofedmaze_v0"}

    def __init__(
        self,
        rows: int = 9,
        cols: int = 9,
        algorithm: str = "recursive_backtracking",
        window_size: int = 5,
        max_episode_steps: int = 200,
        num_checkpoints: int = 0,
        num_obstacles: int = 0,
        num_key_door_pairs: int = 0,
        reward_fn: Optional[RewardFn] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        """
        Args:
            rows, cols: Raw grid dimensions (must be odd -- see Grid).
            algorithm: One of env.generator.generator_factory.available_generators().
            window_size: Odd, >=3 -- passed through to build_observation().
            max_episode_steps: Truncation limit.
            num_checkpoints, num_obstacles, num_key_door_pairs: How many
                of each object to randomly place at reset(). Set to 0
                to disable a variant entirely (the base task is just
                "both agents reach the exit").
            reward_fn: Overrides _default_reward_fn -- see the
                module-level warning about reward policy being
                unresolved. Signature: (env, success: bool) -> {agent_id: float}.
            render_mode: "ascii" or None.
        """
        self.rows = rows
        self.cols = cols
        self.algorithm = algorithm
        self.window_size = window_size
        self.max_episode_steps = max_episode_steps
        self.num_checkpoints = num_checkpoints
        self.num_obstacles = num_obstacles
        self.num_key_door_pairs = num_key_door_pairs
        self.reward_fn: RewardFn = reward_fn if reward_fn is not None else _default_reward_fn
        self.render_mode = render_mode

        self.possible_agents: List[AgentID] = [AGENT_A, AGENT_B]
        self.agents: List[AgentID] = []

        self.maze: Optional[Maze] = None
        self._agent_objs: Dict[AgentID, Agent] = {}
        self._checkpoints: Dict[LogicalPos, Checkpoint] = {}
        self._doors: Dict[LogicalPos, Door] = {}
        self._keys: Dict[LogicalPos, Key] = {}
        self._obstacles: Dict[LogicalPos, Obstacle] = {}
        self._held_keys: Dict[AgentID, Dict[str, Key]] = {}
        self._exit_obj: Optional[Exit] = None
        self._step_count = 0
        self._rng = random.Random()
        self._distance_to_exit: Dict[LogicalPos, int] = {}

    # ------------------------------------------------------------------
    # PettingZoo required interface
    # ------------------------------------------------------------------

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        # lru_cache is required, not just an optimization: PettingZoo's
        # own compliance test (pettingzoo.test.parallel_api_test) asserts
        # observation_space(agent) returns the SAME object (identity,
        # `is`, not just `==`) across calls. Caching per (self, agent)
        # is safe here because window_size/NUM_CHANNELS never change
        # after construction.
        return gymnasium.spaces.Box(
            low=0.0, high=1.0,
            shape=(NUM_CHANNELS, self.window_size, self.window_size),
            dtype=np.float32,
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Discrete(NUM_ACTIONS)

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[AgentID, np.ndarray], Dict[AgentID, dict]]:
        seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        self._rng = random.Random(seed)

        self.maze = Maze(rows=self.rows, cols=self.cols, random_seed=seed)
        create_generator(self.algorithm, random_seed=seed).generate(self.maze)
        # generator's _finalize() already set start_position/exit_position.

        reserved: Set[LogicalPos] = {self.maze.start_position, self.maze.exit_position}

        self._checkpoints = {}
        self._doors = {}
        self._keys = {}
        self._obstacles = {}
        self._held_keys = {AGENT_A: {}, AGENT_B: {}}

        for i, pos in enumerate(self._pick_free_cells(self.num_checkpoints, reserved)):
            cp = Checkpoint(f"checkpoint_{i}")
            cp.place_at(self.maze, *pos)
            self._checkpoints[pos] = cp
            reserved.add(pos)

        for i, pos in enumerate(self._pick_free_cells(self.num_obstacles, reserved)):
            obstacle = Obstacle(f"obstacle_{i}")
            obstacle.place_at(self.maze, *pos)
            self._obstacles[pos] = obstacle
            reserved.add(pos)

        for i in range(self.num_key_door_pairs):
            door_pos, key_pos = self._pick_free_cells(2, reserved)
            door_id = f"door_{i}"
            door = Door(door_id)
            door.place_at(self.maze, *door_pos)
            self._doors[door_pos] = door
            reserved.add(door_pos)

            key = Key(door_id)
            key.place_at(self.maze, *key_pos)
            self._keys[key_pos] = key
            reserved.add(key_pos)

        self._exit_obj = Exit()
        if self.num_checkpoints > 0:
            self._exit_obj.lock()

        self._agent_objs = {}
        start = self.maze.start_position
        agent_a = Agent(AGENT_A)
        agent_a.place_at(self.maze, *start)
        self._agent_objs[AGENT_A] = agent_a

        b_spawn = self._pick_spawn_for_second_agent(start, reserved)
        agent_b = Agent(AGENT_B)
        agent_b.place_at(self.maze, *b_spawn)
        self._agent_objs[AGENT_B] = agent_b

        self._step_count = 0
        self.agents = list(self.possible_agents)
        self._agents_at_exit: Set[AgentID] = set()

        self._distance_to_exit = self._compute_distance_to_exit()

        observations = {aid: self._build_obs(aid) for aid in self.agents}
        infos = {aid: {} for aid in self.agents}
        return observations, infos

    def step(
        self, actions: Dict[AgentID, int]
    ) -> Tuple[
        Dict[AgentID, np.ndarray], Dict[AgentID, float],
        Dict[AgentID, bool], Dict[AgentID, bool], Dict[AgentID, dict],
    ]:
        active_agents = list(self.agents)  # snapshot -- returned dicts cover these

        potential_before = self._team_potential()

        collided_agents: Set[AgentID] = set()
        newly_collected_keys: List[str] = []
        newly_reached_checkpoints: List[str] = []
        newly_unlocked_doors: List[str] = []

        for agent_id in self.possible_agents:  # fixed order: AGENT_A, then AGENT_B
            if agent_id in actions:
                collided, unlocked = self._attempt_action(agent_id, actions[agent_id])
                if collided:
                    collided_agents.add(agent_id)
                if unlocked is not None:
                    newly_unlocked_doors.append(unlocked)

        for agent_id in self.possible_agents:
            key_collected, checkpoint_reached = self._process_auto_pickup(agent_id)
            if key_collected is not None:
                newly_collected_keys.append(key_collected)
            if checkpoint_reached is not None:
                newly_reached_checkpoints.append(checkpoint_reached)

        newly_reached_exits: Set[AgentID] = set()
        for aid in self.possible_agents:
            if self._exit_obj.is_usable_by(self.maze, self._agent_objs[aid].position):
                if aid not in self._agents_at_exit:
                    self._agents_at_exit.add(aid)
                    newly_reached_exits.add(aid)

        if self._checkpoints and all(cp.is_reached for cp in self._checkpoints.values()):
            self._exit_obj.unlock()

        self._step_count += 1
        success = all(
            self._exit_obj.is_usable_by(self.maze, self._agent_objs[aid].position)
            for aid in self.possible_agents
        )
        truncated = self._step_count >= self.max_episode_steps

        potential_after = self._team_potential()
        event = StepEvent(
            collided_agents=collided_agents,
            newly_reached_checkpoints=newly_reached_checkpoints,
            newly_collected_keys=newly_collected_keys,
            newly_unlocked_doors=newly_unlocked_doors,
            newly_reached_exits=newly_reached_exits,
            success=success,
            potential_before=potential_before,
            potential_after=potential_after,
        )
        rewards = self.reward_fn(self, event)

        terminations = {aid: success for aid in active_agents}
        truncations = {aid: (truncated and not success) for aid in active_agents}
        infos = {
            aid: {
                "step_count": self._step_count,
                "held_keys": list(self._held_keys.get(aid, {}).keys()),
                "collided": aid in collided_agents,
            }
            for aid in active_agents
        }
        observations = {aid: self._build_obs(aid) for aid in active_agents}

        if success or truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode == "ascii":
            from env.render.ascii_renderer import AsciiRenderer
            AsciiRenderer().print_maze(self.maze)
        return None

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_distance_to_exit(self) -> Dict[LogicalPos, int]:
        """
        Precompute raw-grid-step distance from every LOGICAL cell to
        the exit, once per episode, by reusing BFSValidator (the same
        tested traversal engine env/validators/ already uses) rather
        than writing a second BFS implementation here.

        Simplification, stated plainly: this is STRUCTURAL distance
        only -- it ignores locked doors and obstacles (neither affects
        Cell.is_wall, so BFSValidator can't see them). Shaping near a
        locked door or obstacle may therefore be slightly optimistic
        (it "sees through" a temporary block). Recomputing dynamically
        whenever a door unlocks would fix this but wasn't judged worth
        the added per-step cost for a placeholder reward policy.
        """
        validator = BFSValidator(self.maze)
        parents = validator.traverse(self.maze.exit_grid_position)
        distances: Dict[LogicalPos, int] = {}
        for lr in range(self.maze.grid.logical_rows):
            for lc in range(self.maze.grid.logical_cols):
                cell = self.maze.grid.get_logical_cell(lr, lc)
                raw = (cell.row, cell.col)
                if raw in parents:
                    distances[(lr, lc)] = len(validator.reconstruct_path(parents, raw)) - 1
        return distances

    def _team_potential(self) -> float:
        """
        Team-level shaping potential: negative AVERAGE distance-to-exit
        across both agents (average, not min/sum, since success requires
        BOTH agents to reach the exit -- shaping should reward progress
        by either agent, weighted equally). Higher potential = closer
        to goal, matching standard potential-based-shaping convention.

        Falls back to a fixed large-negative value for a position this
        episode's distance map doesn't cover (should not happen for a
        connected generated maze; defensive only).
        """
        fallback = -(self.maze.grid.logical_rows + self.maze.grid.logical_cols)
        total = 0.0
        for aid in self.possible_agents:
            pos = self._agent_objs[aid].position
            distance = self._distance_to_exit.get(pos)
            total += -distance if distance is not None else fallback
        return total / len(self.possible_agents)

    def _build_obs(self, agent_id: AgentID) -> np.ndarray:
        return build_observation(
            self.maze,
            agent_position=self._agent_objs[agent_id].position,
            self_agent_id=agent_id,
            window_size=self.window_size,
            door_registry=self._doors,
            checkpoint_registry=self._checkpoints,
        )

    def _pick_free_cells(self, count: int, reserved: Set[LogicalPos]) -> List[LogicalPos]:
        """
        Sample `count` distinct LOGICAL cells not in `reserved`, using
        this episode's seeded RNG for reproducibility.

        Raises:
            ValueError: If fewer than `count` free cells exist.
        """
        if count == 0:
            return []
        all_cells = [
            (r, c)
            for r in range(self.maze.grid.logical_rows)
            for c in range(self.maze.grid.logical_cols)
            if (r, c) not in reserved
        ]
        if len(all_cells) < count:
            raise ValueError(
                f"Requested {count} free cells but only {len(all_cells)} are available "
                f"(maze logical size {self.maze.grid.logical_rows}x{self.maze.grid.logical_cols}, "
                f"{len(reserved)} cells already reserved)"
            )
        return self._rng.sample(all_cells, count)

    def _pick_spawn_for_second_agent(
        self, start: LogicalPos, reserved: Set[LogicalPos]
    ) -> LogicalPos:
        """
        AGENT_B spawns at a random unoccupied logical neighbor of start
        if one exists, else a random free cell elsewhere. See module
        docstring -- this resolves the previously-open "where does the
        second agent spawn" question from env/objects/agent.py.
        """
        start_cell = self.maze.grid.get_logical_cell(*start)
        candidates = []
        for dr, dc in DIRECTIONS:
            neighbor = (start[0] + dr, start[1] + dc)
            if (
                0 <= neighbor[0] < self.maze.grid.logical_rows
                and 0 <= neighbor[1] < self.maze.grid.logical_cols
                and neighbor not in reserved
            ):
                neighbor_cell = self.maze.grid.get_logical_cell(*neighbor)
                if not self.maze.grid.get_wall_between(start_cell, neighbor_cell).is_wall:
                    candidates.append(neighbor)
        if candidates:
            return self._rng.choice(candidates)
        return self._pick_free_cells(1, reserved | {start})[0]

    def _attempt_action(self, agent_id: AgentID, action: int) -> Tuple[bool, Optional[str]]:
        """
        Returns (collided, unlocked_door_id).

        collided is True only for a REJECTED movement attempt (wall,
        obstacle, locked door, or the other agent's cell) -- used by
        step() to apply COLLISION_PENALTY. INTERACT is never a
        collision, whether or not it found a door to unlock.
        unlocked_door_id is the door_id unlocked this call, or None.
        """
        if action == INTERACT:
            return False, self._process_interact(agent_id)

        agent_obj = self._agent_objs[agent_id]
        try:
            dr, dc = direction_for(action)
        except ValueError:
            return False, None  # unrecognized action -- no-op, not a collision

        target = (agent_obj.position[0] + dr, agent_obj.position[1] + dc)

        if target in self._obstacles:
            return True, None  # obstacles always block
        if target in self._doors and self._doors[target].blocks_movement:
            return True, None  # locked door blocks

        try:
            agent_obj.move_to(self.maze, *target)
            return False, None
        except (ValueError, IndexError):
            return True, None  # wall, out of bounds, or occupied by the other agent

    def _process_interact(self, agent_id: AgentID) -> Optional[str]:
        """
        Unlock at most one adjacent locked door per INTERACT, using a
        held key with a matching door_id. Returns the door_id unlocked,
        or None if nothing happened. See module docstring for why
        key/checkpoint pickup is automatic and INTERACT is reserved for
        this alone.
        """
        position = self._agent_objs[agent_id].position
        held = self._held_keys.setdefault(agent_id, {})
        for dr, dc in DIRECTIONS:
            neighbor = (position[0] + dr, position[1] + dc)
            door = self._doors.get(neighbor)
            if door is not None and door.is_locked and door.door_id in held:
                held[door.door_id].use(door)
                del held[door.door_id]
                return door.door_id
        return None

    def _process_auto_pickup(self, agent_id: AgentID) -> Tuple[Optional[str], Optional[str]]:
        """Returns (key_door_id_collected, checkpoint_id_reached), either possibly None."""
        position = self._agent_objs[agent_id].position
        key_collected: Optional[str] = None
        checkpoint_reached: Optional[str] = None

        key = self._keys.pop(position, None)
        if key is not None:
            key.collect(self.maze)
            self._held_keys.setdefault(agent_id, {})[key.door_id] = key
            key_collected = key.door_id

        cp = self._checkpoints.get(position)
        if cp is not None and not cp.is_reached:
            cp.mark_reached()
            checkpoint_reached = cp.checkpoint_id

        return key_collected, checkpoint_reached


if __name__ == "__main__":
    env = CoFedMazeParallelEnv(rows=9, cols=9, algorithm="recursive_backtracking")
    obs, infos = env.reset(seed=1)
    print("Agents:", env.agents)
    print("Observation shapes:", {a: o.shape for a, o in obs.items()})

    for step_num in range(10):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        print(f"step {step_num}: rewards={rewards} terminations={terminations} truncations={truncations}")
        if not env.agents:
            print("Episode ended.")
            break