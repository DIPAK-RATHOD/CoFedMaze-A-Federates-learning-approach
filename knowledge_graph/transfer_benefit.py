"""
transfer_benefit.py

Normalizes a raw Transfer Benefit value -- already computed by
federation/validation/transfer_validation.compute_transfer_benefit(),
the actual expensive step, since it runs real episodes -- into [0,1]
for use in the Knowledge Score formula. This file does NOT run
episodes or recompute TB itself. Isolating normalization here, separate
from the expensive computation, is what lets a caller (updater.py, not
yet built) tell "cheap lookup" apart from "triggers a full episode
run," matching this file's stated purpose in the Directory Structure
Reference.
"""

from federation.validation.transfer_validation import TransferBenefitResult
from knowledge_graph.normalization import sigmoid_normalize

# PLACEHOLDER, not specified in any project doc: sigmoid steepness
# controlling how quickly TB saturates toward 0/1. 2.0 means a TB of
# about +-1.0 (a 100% reward swing) sits close to the normalized
# extremes; retune once real validation-reward variance is measured
# (which the KG/Coalition Implementation Strategy doc itself flags as
# a prerequisite for finalizing ANY threshold/steepness in this
# pipeline).
DEFAULT_MIDPOINT = 0.0
DEFAULT_STEEPNESS = 2.0


def normalize_transfer_benefit(
    tb_result: TransferBenefitResult,
    midpoint: float = DEFAULT_MIDPOINT,
    steepness: float = DEFAULT_STEEPNESS,
) -> float:
    """
    Sigmoid normalization (see normalization.py), not clip_normalize:
    raw TB has no natural fixed [min, max], and a sigmoid maps TB=0
    (no change from baseline) to exactly 0.5 -- a neutral score,
    correctly the midpoint of the [0,1] range the Knowledge Score
    formula expects every criterion to already be in.
    """
    return sigmoid_normalize(tb_result.transfer_benefit, midpoint=midpoint, steepness=steepness)


if __name__ == "__main__":
    from env.core.actions import NUM_ACTIONS
    from env.core.observations import NUM_CHANNELS
    from env.wrappers.pettingzoo_env import CoFedMazeParallelEnv
    from federation.validation.transfer_validation import compute_transfer_benefit, extract_shared_state
    from marl.models.vdn import VDNModel

    env = CoFedMazeParallelEnv(rows=9, cols=9, algorithm="recursive_backtracking", window_size=5, max_episode_steps=30)
    local_model = VDNModel(in_channels=NUM_CHANNELS, window_size=5, num_actions=NUM_ACTIONS, num_agents=2)
    neighbor_model = VDNModel(in_channels=NUM_CHANNELS, window_size=5, num_actions=NUM_ACTIONS, num_agents=2)
    candidate = extract_shared_state(neighbor_model)

    result = compute_transfer_benefit(local_model, candidate, env, validation_seeds=[1, 2, 3])
    print("Raw TB result:", result)

    normalized = normalize_transfer_benefit(result)
    print("Normalized TB (should be in [0,1]):", normalized)
    assert 0.0 <= normalized <= 1.0

    # TB=0 must map to exactly 0.5
    from federation.validation.transfer_validation import TransferBenefitResult
    neutral = TransferBenefitResult(r_old=5.0, r_new=5.0, transfer_benefit=0.0, classification="neutral")
    print("TB=0 normalizes to exactly 0.5:", normalize_transfer_benefit(neutral))
    assert normalize_transfer_benefit(neutral) == 0.5
    print("OK")
