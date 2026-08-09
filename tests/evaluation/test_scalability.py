import pytest

from evaluation.scalability import ScalePoint, default_scale_points


def test_default_scale_points_only_include_valid_coalition_sizes():
    points = default_scale_points()

    assert len(points) == 8
    assert all(1 <= point.max_coalition_size <= point.node_count for point in points)


def test_scale_point_rejects_invalid_coalition_size():
    with pytest.raises(ValueError):
        ScalePoint(node_count=2, max_coalition_size=3)
