import pytest

from evaluation.ablation import ablation_variants


def test_default_ablation_has_full_and_five_leave_one_out_variants():
    variants = ablation_variants()

    assert [variant.name for variant in variants] == [
        "full_ks", "without_tb", "without_ts", "without_ms", "without_l", "without_e",
    ]
    assert sum(variants[0].weights.values()) == pytest.approx(1.0)
    for variant in variants[1:]:
        assert variant.weights[variant.omitted_criterion] == 0.0
        assert sum(variant.weights.values()) == pytest.approx(1.0)
