"""
updater.py

Orchestrates the per-round Knowledge Graph update sequence: recompute
criteria (per the fast-loop/slow-loop schedule) -> combine into
Knowledge Score -> update the DirectedKnowledgeGraph -> persist to
state/nodeN/knowledge_graph.json. The single entry point node/scheduler.py
(not yet built) will call each round, so the fast-loop/slow-loop split
(Section 6 of the KG/Coalition Implementation Strategy doc) has one
clear place it actually executes, rather than being reimplemented per
caller.

Fast loop (every round): TB (federation/validation -- the expensive
one, runs real episodes), L, E (both cheap -- hop-count and
message-size based, no caching needed).
Slow loop (every K rounds): TS, MS (both assumed to drift slowly, so
cached between recomputes rather than recalculated every round).

Persistence uses plain JSON (one float per edge -- the current KS-bar
only, no history), matching the edge-efficiency strategy doc's
explicit "O(1) memory, no per-episode history buffer" principle for
exactly this file's persisted state.
"""

import json
from pathlib import Path
from typing import Dict, Union

from federation.topology.physical_graph import PhysicalGraph
from federation.validation.transfer_validation import TransferBenefitResult
from knowledge_graph.directed_graph import DirectedKnowledgeGraph
from knowledge_graph.energy import compute_energy_from_size
from knowledge_graph.knowledge_score import compute_knowledge_score
from knowledge_graph.latency import compute_latency
from knowledge_graph.model_similarity import EncoderState, compute_model_similarity
from knowledge_graph.task_similarity import TaskFeatures, compute_task_similarity
from knowledge_graph.transfer_benefit import normalize_transfer_benefit

PathLike = Union[str, Path]


class KnowledgeGraphUpdater:
    """
    Per-node orchestrator: owns ONE DirectedKnowledgeGraph (this node's
    view of the graph) and caches TS/MS per physical neighbor between
    slow-loop recomputes.
    """

    def __init__(
        self,
        own_node_id: str,
        physical_graph: PhysicalGraph,
        own_task_features: TaskFeatures,
        slow_loop_interval: int = 5,
        tau_form: float = 0.50,
        tau_break: float = 0.30,
        alpha: float = 0.30,
    ) -> None:
        """
        Args:
            own_node_id: This node's id.
            physical_graph: Gates which neighbors are ever evaluated.
            own_task_features: This node's own TaskFeatures, compared
                against each neighbor's for the TS criterion.
            slow_loop_interval: Recompute TS/MS every K rounds per
                neighbor (K = slow_loop_interval); reuse the cached
                value on other rounds.
            tau_form, tau_break, alpha: Forwarded to DirectedKnowledgeGraph.
        """
        self.own_node_id = own_node_id
        self.physical_graph = physical_graph
        self.own_task_features = own_task_features
        self.slow_loop_interval = slow_loop_interval

        self.graph = DirectedKnowledgeGraph(
            own_node_id, physical_graph, tau_form=tau_form, tau_break=tau_break, alpha=alpha
        )
        self._round_counts: Dict[str, int] = {}  # per-neighbor round counter, for independent slow-loop schedules
        self._cached_ts: Dict[str, float] = {}
        self._cached_ms: Dict[str, float] = {}

    def update(
        self,
        neighbor_id: str,
        tb_result: TransferBenefitResult,
        own_encoder_state: EncoderState,
        neighbor_task_features: TaskFeatures,
        neighbor_encoder_state: EncoderState,
        neighbor_message_size_bytes: int,
    ) -> float:
        """
        Run one round's full update for the edge neighbor_id -> own_node_id.

        Args:
            tb_result: Already-computed transfer-benefit result for
                this neighbor this round (the expensive step, run by
                federation/validation/transfer_validation.compute_transfer_benefit
                elsewhere -- this method never runs episodes itself).
            own_encoder_state: This node's own current shared-encoder
                state, for the MS comparison.
            neighbor_task_features, neighbor_encoder_state: Only
                actually recomputed every slow_loop_interval rounds for
                this neighbor -- passed every call for a simple
                interface, cheaply ignored (cache hit) on non-slow-loop
                rounds.
            neighbor_message_size_bytes: Length in bytes of the raw
                serialized message received from this neighbor this
                round, for the E (energy) criterion. Passed as a plain
                int (not an UpdateMessage) specifically so a caller that
                already has raw bytes on hand (e.g. node/scheduler.py,
                reading directly from transport.py) never has to
                reconstruct or re-serialize a message object just to
                measure its size.

        Returns:
            The updated KS-bar for this edge.
        """
        round_count = self._round_counts.get(neighbor_id, 0)
        is_slow_loop_round = (round_count % self.slow_loop_interval == 0) or neighbor_id not in self._cached_ts

        if is_slow_loop_round:
            self._cached_ts[neighbor_id] = compute_task_similarity(
                self.own_task_features, neighbor_task_features
            )
            self._cached_ms[neighbor_id] = compute_model_similarity(
                own_encoder_state, neighbor_encoder_state
            )

        tb = normalize_transfer_benefit(tb_result)
        ts = self._cached_ts[neighbor_id]
        ms = self._cached_ms[neighbor_id]
        l = compute_latency(self.physical_graph, neighbor_id, self.own_node_id)
        e = compute_energy_from_size(neighbor_message_size_bytes)

        ks = compute_knowledge_score(tb=tb, ts=ts, ms=ms, l=l, e=e)
        ks_bar = self.graph.update_edge(neighbor_id, ks)

        self._round_counts[neighbor_id] = round_count + 1
        return ks_bar

    def save_state(self, path: PathLike) -> None:
        """
        Persist the current KS-bar for every tracked edge as plain
        JSON -- one float per edge, matching state/nodeN/knowledge_graph.json's
        documented minimal-persisted-state design (no history buffer).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "own_node_id": self.own_node_id,
            "ks_bar": {n: self.graph.ks_bar(n) for n in self.graph._trackers},
            "active_edges": self.graph.active_neighbors(),
        }
        path.write_text(json.dumps(state, indent=2))

    @classmethod
    def load_state_into(cls, updater: "KnowledgeGraphUpdater", path: PathLike) -> None:
        """
        Restore a previously-saved KS-bar per edge into an already-
        constructed `updater` (whose physical_graph/thresholds must
        already match what produced the saved file -- this does not
        reconstruct topology or hyperparameters, only KS-bar values).

        Raises:
            FileNotFoundError: If `path` doesn't exist.
            ValueError: If the saved own_node_id doesn't match
                `updater.own_node_id`, or a saved edge isn't one of
                `updater`'s tracked physical neighbors.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No saved knowledge graph state at {path}")
        state = json.loads(path.read_text())

        if state["own_node_id"] != updater.own_node_id:
            raise ValueError(
                f"Saved state is for node {state['own_node_id']!r}, not {updater.own_node_id!r}"
            )

        for neighbor_id, ks_bar_value in state["ks_bar"].items():
            if neighbor_id not in updater.graph._trackers:
                raise ValueError(
                    f"Saved edge from {neighbor_id!r} is not a tracked physical neighbor "
                    f"of {updater.own_node_id!r} in the given physical_graph"
                )
            updater.graph._trackers[neighbor_id].ks_bar = ks_bar_value
            updater.graph._trackers[neighbor_id]._has_update = True

        for neighbor_id in state["active_edges"]:
            updater.graph._active_edges.add(neighbor_id)


if __name__ == "__main__":
    import tempfile

    from env.core.actions import NUM_ACTIONS
    from env.core.observations import NUM_CHANNELS
    from env.wrappers.pettingzoo_env import CoFedMazeParallelEnv
    from federation.communication.messages import UpdateMessage
    from federation.communication.serializer import serialize_message
    from federation.validation.transfer_validation import compute_transfer_benefit, extract_shared_state
    from marl.models.vdn import VDNModel

    graph = PhysicalGraph.from_yaml("configs/topology.yaml")
    env = CoFedMazeParallelEnv(rows=9, cols=9, algorithm="recursive_backtracking", window_size=5, max_episode_steps=30)

    own_model = VDNModel(in_channels=NUM_CHANNELS, window_size=5, num_actions=NUM_ACTIONS, num_agents=2)
    neighbor_model = VDNModel(in_channels=NUM_CHANNELS, window_size=5, num_actions=NUM_ACTIONS, num_agents=2)

    own_features = TaskFeatures.from_env(env)
    neighbor_env = CoFedMazeParallelEnv(rows=9, cols=9, algorithm="recursive_backtracking", window_size=5)
    neighbor_features = TaskFeatures.from_env(neighbor_env)

    updater = KnowledgeGraphUpdater(
        own_node_id="N1", physical_graph=graph, own_task_features=own_features,
        slow_loop_interval=3, tau_form=0.50, tau_break=0.30,
    )

    candidate_shared_state = extract_shared_state(neighbor_model)
    tb_result = compute_transfer_benefit(own_model, candidate_shared_state, env, validation_seeds=[1, 2])

    dummy_message = UpdateMessage(
        node_id="N2", round=1, validation_reward=tb_result.r_new,
        payload=extract_shared_state(neighbor_model), update_norm=1.0,
    )
    dummy_message_size = len(serialize_message(dummy_message))

    ks_bar = updater.update(
        neighbor_id="N2",
        tb_result=tb_result,
        own_encoder_state=extract_shared_state(own_model)["encoder"],
        neighbor_task_features=neighbor_features,
        neighbor_encoder_state=extract_shared_state(neighbor_model)["encoder"],
        neighbor_message_size_bytes=dummy_message_size,
    )
    print("KS-bar for N2 -> N1 after round 1:", ks_bar)

    # Persistence round-trip
    with tempfile.TemporaryDirectory() as tmp:
        save_path = Path(tmp) / "knowledge_graph.json"
        updater.save_state(save_path)
        print("Saved:", save_path.read_text())

        fresh_updater = KnowledgeGraphUpdater(
            own_node_id="N1", physical_graph=graph, own_task_features=own_features,
        )
        KnowledgeGraphUpdater.load_state_into(fresh_updater, save_path)
        restored_ks_bar = fresh_updater.graph.ks_bar("N2")
        print("Restored KS-bar matches saved value:", restored_ks_bar == ks_bar)
        assert restored_ks_bar == ks_bar

    print("OK")
