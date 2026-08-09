"""
coalition_manager.py

Orchestrates the full coalition lifecycle for ONE node: singleton init
-> candidate check -> confirm-with-patience -> merge -> periodic health
check -> expel/dissolve. The single stateful object matching
state/nodeN/coalition_state.json (not yet built); every other file in
this package (merge.py, split.py, leave_one_out.py, dwell_timer.py,
reputation.py) implements one piece of logic that this class calls at
the right time, in the right order -- this file contains no threshold
math or evaluation logic of its own, only sequencing.

Max coalition size is hard-capped at 3, per the KG/Coalition
Implementation Strategy doc's hyperparameter table (2-3 range; 3 used
as the cap here since that's the upper/more-permissive end, and
leave_one_out.py already exists specifically to handle a size-3
coalition's health failures without a full dissolve).
"""

from typing import Dict, List, Optional, Set

from env.wrappers.pettingzoo_env import CoFedMazeParallelEnv
from federation.aggregation.fedavg import SharedState
from federation.aggregation.weighted import weighted_average
from knowledge_graph.directed_graph import DirectedKnowledgeGraph
from marl.models.vdn import VDNModel

from coalition.dwell_timer import DwellTimer
from coalition.leave_one_out import find_member_to_expel
from coalition.merge import DEFAULT_PATIENCE, MergeConfirmationTracker, pareto_check
from coalition.reputation import ReputationTracker
from coalition.split import health_check, should_dissolve

MAX_COALITION_SIZE = 3
DEFAULT_HEALTH_CHECK_INTERVAL = 5  # K, matches KnowledgeGraphUpdater's slow_loop_interval cadence


class CoalitionManager:
    """
    Owns one node's coalition membership and drives it through the
    full lifecycle each round via step().
    """

    def __init__(
        self,
        own_node_id: str,
        directed_graph: DirectedKnowledgeGraph,
        patience: int = DEFAULT_PATIENCE,
        dwell_episodes: int = 2,
        health_check_interval: int = DEFAULT_HEALTH_CHECK_INTERVAL,
    ) -> None:
        self.own_node_id = own_node_id
        self.directed_graph = directed_graph
        self.health_check_interval = health_check_interval

        # Every node starts as its own singleton coalition (Step 1).
        self.members: Set[str] = {own_node_id}

        self._merge_tracker = MergeConfirmationTracker(patience=patience)
        self._dwell_timer = DwellTimer(dwell_episodes=dwell_episodes)
        self._reputation = ReputationTracker()
        self._round = 0

    def is_singleton(self) -> bool:
        return len(self.members) == 1

    def step(
        self,
        coalition_model: VDNModel,
        member_shared_states: Dict[str, SharedState],
        member_weights: Dict[str, float],
        env: CoFedMazeParallelEnv,
        validation_seeds: List[int],
    ) -> None:
        """
        Run one round of the full coalition lifecycle.

        Args:
            coalition_model: This node's model, used as scratch space
                for Pareto/leave-one-out checks (always restored to its
                pre-call state by the functions this method calls).
            member_shared_states: {member_id: shared state}, covering
                EVERY current coalition member (including own_node_id)
                plus any candidates being considered this round.
            member_weights: {member_id: weight} (e.g. current KS-bar)
                for the same set of ids, used for weighted_average().
            validation_seeds: The shared small validation subset.
        """
        self._round += 1
        self._dwell_timer.tick()

        if len(self.members) < MAX_COALITION_SIZE:
            self._try_merge(coalition_model, member_shared_states, member_weights, env, validation_seeds)
        elif not self._dwell_timer.is_active() and self._round % self.health_check_interval == 0:
            self._check_health(coalition_model, member_shared_states, member_weights, env, validation_seeds)

    def _try_merge(
        self,
        coalition_model: VDNModel,
        member_shared_states: Dict[str, SharedState],
        member_weights: Dict[str, float],
        env: CoFedMazeParallelEnv,
        validation_seeds: List[int],
    ) -> None:
        candidates = [
            n for n in self.directed_graph.active_neighbors() if n not in self.members
        ]
        for candidate_id in candidates:
            if len(self.members) >= MAX_COALITION_SIZE:
                break  # cap hit mid-loop -- stop considering further candidates this round

            prospective_ids = list(self.members) + [candidate_id]
            if not all(m in member_shared_states for m in prospective_ids):
                # A current member's (or the candidate's) state isn't
                # cached yet -- defensively skip rather than KeyError.
                # Should be rare given scheduler.py now maintains a
                # persistent known-states cache, but this guards the
                # genuinely-first round after a merge, before this
                # node has heard from every member at least once.
                continue

            ks_bar = self.directed_graph.ks_bar(candidate_id)
            above_tau_form = ks_bar > self.directed_graph.tau_form

            prospective_states = [member_shared_states[m] for m in prospective_ids]
            prospective_weights = [member_weights.get(m, ks_bar if m == candidate_id else 1.0) for m in prospective_ids]
            prospective_aggregate = weighted_average(prospective_states, prospective_weights)

            self._merge_tracker.record_round(candidate_id, above_tau_form)
            if not self._merge_tracker.is_patience_satisfied(candidate_id):
                continue

            if pareto_check(coalition_model, prospective_aggregate, env, validation_seeds):
                self.members.add(candidate_id)
                self._merge_tracker.reset(candidate_id)
                self._dwell_timer.start()
                self._reputation.record(candidate_id, self._round, "merge", ks_bar)
            else:
                self._reputation.record(candidate_id, self._round, "confirm", ks_bar)

    def _check_health(
        self,
        coalition_model: VDNModel,
        member_shared_states: Dict[str, SharedState],
        member_weights: Dict[str, float],
        env: CoFedMazeParallelEnv,
        validation_seeds: List[int],
    ) -> None:
        healthy = health_check(self.directed_graph, self.members, self.own_node_id)

        if healthy:
            for member_id in self.members:
                if member_id != self.own_node_id:
                    self._reputation.record(member_id, self._round, "health_check_pass", self.directed_graph.ks_bar(member_id))
            return

        if should_dissolve(dwell_active=self._dwell_timer.is_active(), coalition_size=len(self.members), is_healthy=healthy):
            self._dissolve()
            return

        if len(self.members) == MAX_COALITION_SIZE:
            if not all(m in member_shared_states for m in self.members):
                return  # can't run leave-one-out without every member's state cached
            member_states = {m: member_shared_states[m] for m in self.members}
            weights = {m: member_weights.get(m, 1.0) for m in self.members}
            expelled = find_member_to_expel(coalition_model, member_states, weights, env, validation_seeds)
            if expelled != self.own_node_id:
                self.members.discard(expelled)
                self._merge_tracker.reset(expelled)
                self._reputation.record(expelled, self._round, "expel", self.directed_graph.ks_bar(expelled))

    def _dissolve(self) -> None:
        for member_id in list(self.members):
            if member_id != self.own_node_id:
                self._reputation.record(member_id, self._round, "dissolve", self.directed_graph.ks_bar(member_id))
        self.members = {self.own_node_id}


if __name__ == "__main__":
    from env.core.actions import NUM_ACTIONS
    from env.core.observations import NUM_CHANNELS
    from federation.topology.physical_graph import PhysicalGraph
    from federation.validation.transfer_validation import extract_shared_state

    graph = PhysicalGraph.from_yaml("configs/topology.yaml")
    dkg = DirectedKnowledgeGraph(own_node_id="N1", physical_graph=graph, tau_form=0.50, tau_break=0.30)

    manager = CoalitionManager(own_node_id="N1", directed_graph=dkg, patience=2, dwell_episodes=2)
    print("Initial state (singleton):", manager.members)
    assert manager.is_singleton()

    env = CoFedMazeParallelEnv(rows=9, cols=9, algorithm="recursive_backtracking", window_size=5, max_episode_steps=20)
    model = VDNModel(in_channels=NUM_CHANNELS, window_size=5, num_actions=NUM_ACTIONS, num_agents=2)

    n1_state = extract_shared_state(model)
    n2_model = VDNModel(in_channels=NUM_CHANNELS, window_size=5, num_actions=NUM_ACTIONS, num_agents=2)
    n2_state = extract_shared_state(n2_model)

    # Drive N2's edge above tau_form for `patience` rounds, then step the manager each round.
    for round_num in range(4):
        dkg.update_edge("N2", ks=0.9)
        manager.step(
            coalition_model=model,
            member_shared_states={"N1": n1_state, "N2": n2_state},
            member_weights={"N1": 1.0, "N2": dkg.ks_bar("N2")},
            env=env,
            validation_seeds=[1, 2],
        )
        print(f"Round {round_num + 1}: members={manager.members}")

    print("Final coalition membership:", manager.members)
    print("OK")
