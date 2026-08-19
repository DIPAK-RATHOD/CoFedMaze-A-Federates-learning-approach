"""
transfer_validation.py

Temporarily loads a received (candidate) update, evaluates it on a
local validation subset, and computes the before/after reward delta --
Transfer Benefit (TB). Per the KG/Coalition Implementation Strategy
doc, this is the single most expensive step in the whole pipeline (it
requires running actual episodes), which is why the doc calls for a
small FIXED validation subset (5-10 mazes) for routine checks, with the
full validation set reserved for periodic confirmation only. This file
is subset-size-agnostic -- pass however many seeds you want; deciding
WHEN to use a small subset vs. the full set is node/scheduler.py's job
(not yet built), not this file's.

Only the SHARED components (encoder, memory) are swapped for testing --
never the private head -- matching the Shared vs. Private Model
Components glossary term, which VDNModel already implements as
separate encoder/memory/head attributes.

Interpretation note: the workplan (3.6) says "apply a small portion of
a neighbour update"; the KG/Coalition Implementation Strategy doc says
"apply the received shared update." Read together with the shared/
private split already established elsewhere in this project, "small
portion" is read here as "the shared component only" (as opposed to
the whole model including private heads) -- NOT a fractional/
interpolated blend of weights. Fractional blending is a distinct,
later step ("Partially mix it with the local model" -- workplan 3.10,
Safety Validation / Partial Acceptance / Rollback), not implemented
here.

TB formula (glossary): TB = (R_new - R_old) / |R_old|. R_old can be
zero or near-zero on a sparse-reward validation subset, which would
otherwise divide by (near) zero -- exactly the numerical-instability
concern raised when the reward policy was designed (see
env/wrappers/pettingzoo_env.py's module docstring). Guarded here with
an epsilon floor on the denominator.

Classification thresholds (helpful/neutral/harmful) are PLACEHOLDERS,
not specified anywhere in project docs -- the KG doc's tau_form=0.50
threshold applies to the composite Knowledge Score, not raw TB
directly. Override via classify_transfer_benefit's parameters once
real validation-reward variance has been measured, which the
KG/Coalition Implementation Strategy doc itself flags as a prerequisite
for finalizing any threshold.
"""

from typing import Any, Dict, List, NamedTuple

import torch

from env.core.constants import AGENT_A, AGENT_B
from env.wrappers.pettingzoo_env import CoFedMazeParallelEnv
from marl.agents.action_selector import EpsilonGreedySelector
from marl.agents.policy import Policy
from marl.agents.vdn_agent import VDNAgent
from marl.models.vdn import VDNModel

DEFAULT_EPSILON_FLOOR = 1e-3
DEFAULT_HELPFUL_THRESHOLD = 0.02
DEFAULT_HARMFUL_THRESHOLD = -0.02


class TransferBenefitResult(NamedTuple):
    r_old: float
    r_new: float
    transfer_benefit: float
    classification: str  # "helpful" | "neutral" | "harmful"


def extract_shared_state(model: VDNModel) -> Dict[str, Any]:
    """Snapshot ONLY the shared components (encoder, memory) -- never the private head."""
    return {
        "encoder": {k: v.clone() for k, v in model.encoder.state_dict().items()},
        "memory": {k: v.clone() for k, v in model.memory.state_dict().items()},
    }


def apply_shared_state(model: VDNModel, shared_state: Dict[str, Any]) -> None:
    model.encoder.load_state_dict(shared_state["encoder"])
    model.memory.load_state_dict(shared_state["memory"])


def blend_shared_state(model: VDNModel, candidate_shared_state: Dict[str, Any], alpha: float = 0.25) -> None:
    """
    Soft parameter blending for shared encoder and memory weights during federated updates:
    shared_local = (1 - alpha) * shared_local + alpha * candidate_shared
    """
    with torch.no_grad():
        for name, param in model.encoder.named_parameters():
            if name in candidate_shared_state["encoder"]:
                param.data.mul_(1.0 - alpha).add_(candidate_shared_state["encoder"][name].to(param.device), alpha=alpha)
        for name, param in model.memory.named_parameters():
            if name in candidate_shared_state["memory"]:
                param.data.mul_(1.0 - alpha).add_(candidate_shared_state["memory"][name].to(param.device), alpha=alpha)


def _run_greedy_episode(env: CoFedMazeParallelEnv, model: VDNModel, seed: int) -> float:
    """
    Play one full episode with GREEDY (non-exploratory) action
    selection, on a FIXED seed, and return the total shared team
    reward. Greedy, not epsilon-greedy: exploration noise would
    contaminate a comparison meant to isolate the effect of the
    candidate weights alone, not random action choice.

    Note: this duplicates a small amount of rollout-loop structure
    also present in marl/training/trainer.py's run_episode() (reset,
    step-until-done). That method also does epsilon-greedy action
    selection, epsilon scheduling, and Trajectory/replay bookkeeping
    that evaluation has no use for -- different enough that sharing one
    function didn't fit cleanly. If a THIRD copy of this loop turns out
    to be needed (e.g. evaluation/benchmark.py, coalition/leave_one_out.py
    -- neither built yet), that's the point to extract a shared
    rollout utility rather than duplicating a third time.
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
    while env.agents:
        actions = {}
        for agent_id in env.agents:
            obs_tensor = torch.from_numpy(obs[agent_id]).unsqueeze(0)
            action, _ = agents[agent_id].act_greedy(obs_tensor)
            actions[agent_id] = action
        obs, rewards, terminations, truncations, infos = env.step(actions)
        total_reward += rewards[AGENT_A]  # shared reward -- identical for both agents by construction

    return total_reward


def evaluate_model(env: CoFedMazeParallelEnv, model: VDNModel, seeds: List[int]) -> float:
    """
    Average total (shared team) reward across one greedy episode per
    seed in `seeds`.

    Raises:
        ValueError: If seeds is empty.
    """
    if not seeds:
        raise ValueError("seeds must not be empty")
    returns = [_run_greedy_episode(env, model, seed) for seed in seeds]
    return sum(returns) / len(returns)


def classify_transfer_benefit(
    tb: float,
    helpful_threshold: float = DEFAULT_HELPFUL_THRESHOLD,
    harmful_threshold: float = DEFAULT_HARMFUL_THRESHOLD,
) -> str:
    """
    See module docstring's note on these thresholds being placeholders.

    Raises:
        ValueError: If harmful_threshold >= helpful_threshold (would
            make "neutral" an empty or inverted range).
    """
    if harmful_threshold >= helpful_threshold:
        raise ValueError(
            f"harmful_threshold ({harmful_threshold}) must be < helpful_threshold "
            f"({helpful_threshold})"
        )
    if tb > helpful_threshold:
        return "helpful"
    if tb < harmful_threshold:
        return "harmful"
    return "neutral"


def compute_transfer_benefit(
    local_model: VDNModel,
    candidate_shared_state: Dict[str, Any],
    env: CoFedMazeParallelEnv,
    validation_seeds: List[int],
    epsilon_floor: float = DEFAULT_EPSILON_FLOOR,
    helpful_threshold: float = DEFAULT_HELPFUL_THRESHOLD,
    harmful_threshold: float = DEFAULT_HARMFUL_THRESHOLD,
) -> TransferBenefitResult:
    """
    Temporarily swap local_model's shared components (encoder, memory)
    for candidate_shared_state, evaluate both before and after on the
    SAME validation_seeds (isolating the weight-swap's effect from maze
    randomness), and ALWAYS revert local_model back to its original
    shared state -- even if evaluation raises partway through. A
    transfer-benefit TEST must never leave the local model corrupted;
    that's the entire reason this uses try/finally rather than a plain
    sequential swap-evaluate-revert.

    Args:
        local_model: This node's model. Its PRIVATE head is never
            touched. Its shared components (encoder, memory) are
            temporarily overwritten, then restored exactly.
        candidate_shared_state: A neighbor's shared state, in the same
            shape extract_shared_state() produces -- i.e.
            {"encoder": state_dict, "memory": state_dict}. Typically
            obtained by calling extract_shared_state() on the
            neighbor's own model (e.g. after
            utils.checkpoint.load_model_state() and constructing a
            temporary VDNModel from it).
        env: Environment to evaluate episodes against.
        validation_seeds: Fixed validation-maze seeds (glossary:
            Validation seeds -- "used for neighbour testing, coalition
            decisions, and model acceptance"; explicitly NOT test
            seeds).
        epsilon_floor: Minimum |R_old| used in TB's denominator, to
            avoid dividing by (near) zero on a sparse-reward validation
            subset.

    Returns:
        TransferBenefitResult(r_old, r_new, transfer_benefit, classification).

    Raises:
        ValueError: Propagated from evaluate_model() if validation_seeds
            is empty, or from classify_transfer_benefit() if the
            threshold arguments are inconsistent.
    """
    original_shared_state = extract_shared_state(local_model)
    try:
        r_old = evaluate_model(env, local_model, validation_seeds)
        apply_shared_state(local_model, candidate_shared_state)
        r_new = evaluate_model(env, local_model, validation_seeds)
    finally:
        apply_shared_state(local_model, original_shared_state)

    denominator = max(abs(r_old), epsilon_floor)
    tb = (r_new - r_old) / denominator
    classification = classify_transfer_benefit(tb, helpful_threshold, harmful_threshold)
    return TransferBenefitResult(r_old=r_old, r_new=r_new, transfer_benefit=tb, classification=classification)


if __name__ == "__main__":
    from env.core.actions import NUM_ACTIONS
    from env.core.observations import NUM_CHANNELS

    env = CoFedMazeParallelEnv(rows=9, cols=9, algorithm="recursive_backtracking", window_size=5, max_episode_steps=40)

    local_model = VDNModel(in_channels=NUM_CHANNELS, window_size=5, num_actions=NUM_ACTIONS, num_agents=2)
    neighbor_model = VDNModel(in_channels=NUM_CHANNELS, window_size=5, num_actions=NUM_ACTIONS, num_agents=2)

    candidate_shared_state = extract_shared_state(neighbor_model)
    validation_seeds = [1, 2, 3]

    original_snapshot = extract_shared_state(local_model)
    result = compute_transfer_benefit(local_model, candidate_shared_state, env, validation_seeds)
    print(result)

    reverted = all(
        torch.equal(original_snapshot["encoder"][k], local_model.encoder.state_dict()[k])
        for k in original_snapshot["encoder"]
    )
    print("Local model's shared weights exactly reverted after testing:", reverted)
    assert reverted
    print("OK")
