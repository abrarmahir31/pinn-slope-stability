"""scripts/sig3_range.py pure helpers."""
import numpy as np
import pytest

from scripts import sig3_range as R


def test_weighted_percentile_equals_plain_with_unit_weights():
    v = np.arange(1.0, 101.0)
    assert R.weighted_percentile(v, np.ones(100), 50) == 50.0
    assert R.weighted_percentile(v, np.ones(100), 95) == 95.0


def test_weights_move_the_pooled_percentile():
    # Two strata: 30 low-stress points oversampled (weight 1/3 each) and 10
    # high-stress points (weight 3 each). Unweighted median is low-stress;
    # the area-weighted median is high-stress.
    v = np.r_[np.full(30, 50.0), np.full(10, 800.0)]
    w = np.r_[np.full(30, 1 / 3), np.full(10, 3.0)]
    assert R.summarise(v)["percentiles"]["50"] == 50.0
    assert R.summarise(v, w)["percentiles"]["50"] == 800.0


def test_summary_reports_tension_and_reference_coverage():
    r = R.summarise(np.array([-10.0, 100.0, 900e3, 1e3]), reference=850e3)
    assert r["frac_tension"] == pytest.approx(0.25)
    assert r["frac_below_reference"] == pytest.approx(0.75)


def test_summary_refuses_an_empty_zone():
    with pytest.raises(ValueError):
        R.summarise(np.array([]))


def test_select_by_tag_and_depth():
    tag = np.array(["Mk", "Mk_d", "Mk_d", "Tm"])
    dist = np.array([1.0, 2.0, 40.0, 5.0])
    assert R.select(tag, dist, ["Mk_d"], 10.0).tolist() == [False, True, False, False]
    assert R.select(tag, dist).all()
