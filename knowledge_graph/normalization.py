"""
normalization.py

Shared [0,1] normalization utility used by every Knowledge Score
criterion (transfer_benefit.py, and eventually task_similarity.py,
model_similarity.py, latency.py, energy.py) -- centralizing this
avoids five slightly-different scaling implementations drifting apart,
per Step 2 of Section 3.7.7's derivation ("Normalize every criterion...
into [0,1]").
"""

import math


def clip_normalize(value: float, min_value: float, max_value: float) -> float:
    """
    Linearly map `value` from [min_value, max_value] onto [0, 1],
    clamping values outside that range to the nearest bound rather than
    extrapolating -- an outlier beyond the expected range should
    saturate, not produce an out-of-[0,1] score that would silently
    corrupt the weighted KS sum.

    Raises:
        ValueError: If min_value >= max_value.
    """
    if min_value >= max_value:
        raise ValueError(f"min_value ({min_value}) must be < max_value ({max_value})")
    if value <= min_value:
        return 0.0
    if value >= max_value:
        return 1.0
    return (value - min_value) / (max_value - min_value)


def sigmoid_normalize(value: float, midpoint: float = 0.0, steepness: float = 1.0) -> float:
    """
    Smooth alternative to clip_normalize, for criteria with no natural
    hard [min, max] bound (e.g. Transfer Benefit, which can range
    arbitrarily -- a very large positive TB shouldn't need a
    hand-picked max to normalize sensibly). Maps (-inf, inf) onto
    (0, 1) via a logistic curve centered at `midpoint`; `steepness`
    controls how quickly it saturates toward the extremes.

    Raises:
        ValueError: If steepness is not positive.
    """
    if steepness <= 0:
        raise ValueError(f"steepness must be positive, got {steepness}")
    return 1.0 / (1.0 + math.exp(-steepness * (value - midpoint)))


if __name__ == "__main__":
    print("clip_normalize(5, 0, 10):", clip_normalize(5, 0, 10))
    assert clip_normalize(5, 0, 10) == 0.5
    print("clip_normalize(-5, 0, 10) clamps to 0:", clip_normalize(-5, 0, 10))
    assert clip_normalize(-5, 0, 10) == 0.0
    print("clip_normalize(50, 0, 10) clamps to 1:", clip_normalize(50, 0, 10))
    assert clip_normalize(50, 0, 10) == 1.0

    print("sigmoid_normalize(0):", sigmoid_normalize(0.0))
    assert sigmoid_normalize(0.0) == 0.5
    print("sigmoid_normalize(large positive) -> near 1:", sigmoid_normalize(100.0))
    print("sigmoid_normalize(large negative) -> near 0:", sigmoid_normalize(-100.0))
    assert sigmoid_normalize(100.0) > 0.99
    assert sigmoid_normalize(-100.0) < 0.01
    print("OK")
