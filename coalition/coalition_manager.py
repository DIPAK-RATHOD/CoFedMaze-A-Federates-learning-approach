"""
coalition_manager.py

Orchestrates the full coalition lifecycle for ONE node: singleton init
-> candidate check -> confirm-with-patience -> merge -> periodic health
check -> expel/dissolve.
"""

from typing import Any, Dict, List, Optional, Set

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
DEFAULT_HEALTH_CHECK_INTERVAL = 5


class CoalitionManager:
    """
    Owns one node's coalition membership and drives it through the
    full lifecycle each round via step() or evaluate_and_update().
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
        self._trainer = None

    def set_trainer(self, trainer: Any) -> None:
        """Attach trainer reference for automated Pareto validation in evaluate_and_update."""
        self._trainer = trainer

    def is_singleton(self) -> bool:
        return len(self.members) == 1

    def evaluate_and_update(
        self,
        current_round: int,
        known_states: Dict[str, SharedState],
        my_trajectory: Optional[Any] = None,
    ) -> List[str]:
        """
        High-level wrapper called by NodeScheduler.
        Evaluates potential merges/splits and returns the list of active coalition members.
        """
        trainer = getattr(self, "_trainer", None)
        if trainer is not None and len(known_states) > 1:
            weights = {}
            for nid in known_states:
                if nid == self.own_node_id:
                    weights[nid] = 1.0
                elif hasattr(self.directed_graph, "_trackers") and nid in self.directed_graph._trackers:
                    weights[nid] = self.directed_graph.ks_bar(nid)
                else:
                    weights[nid] = 0.5

            try:
                self.step(
                    coalition_model=trainer.online_model,
                    member_shared_states=known_states,
                    member_weights=weights,
                    env=trainer.env,
                    validation_seeds=[42, 43],
                )
            except Exception:
                pass
        return sorted(list(self.members))

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
                break

            prospective_ids = list(self.members) + [candidate_id]
            if not all(m in member_shared_states for m in prospective_ids):
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
                return
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
    print("evaluate_and_update test:", manager.evaluate_and_update(1, {"N1": {}, "N2": {}}))
    print("OK")
