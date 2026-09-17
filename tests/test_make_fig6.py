"""Masked recount helpers in scripts/make_fig6.py (Day 40).

Pure numpy, synthetic input; no checkpoint. Every "this is zero / collapses"
test is paired with a control that would fail if the helper were a constant.
"""
import math

import numpy as np
import pytest

from scripts import make_fig6 as f6
from src.step21_geometry import boundaries as bnd
from src.step21_geometry import geometry as g


# --- distance to the traction-free surface ---------------------------------

def test_points_on_the_surface_are_at_zero_distance():
    xs = np.array([10.0, 60.0, 110.0, 200.0])
    d, xn = f6.distance_to_free_surface(xs, g.z_ground(xs))
    assert np.all(d < 5e-3)
    assert np.allclose(xn, xs, atol=5e-3)


def test_distance_below_the_flat_bench_equals_the_vertical_offset():
    # The bench is 0.4 deg, so vertical offset and normal distance agree to
    # 1 - cos(0.4 deg) ~ 2.4e-5.
    x = 112.0
    z = float(g.z_ground(x)) - 4.0
    d, _ = f6.distance_to_free_surface([x], [z])
    assert d[0] == pytest.approx(4.0, rel=1e-3)


def test_distance_below_the_cut_face_is_normal_not_vertical():
    # Negative control for the flat case: on the 31 deg face a 10 m vertical
    # offset is ~8.57 m normal distance. A vertical-depth implementation
    # returns 10 and fails this.
    x = 50.0
    z = float(g.z_ground(x)) - 10.0
    d, _ = f6.distance_to_free_surface([x], [z])
    assert d[0] == pytest.approx(10.0 * math.cos(math.radians(g.CUT_ANGLE)),
                                 abs=0.15)
    assert d[0] < 9.5


def test_free_surface_extent_matches_the_boundary_sampler():
    # FREE_SURFACE_X_HI duplicates a literal in boundaries.sample_boundaries.
    pts = bnd.sample_boundaries(4000, seed=3)["natural_ground"]
    assert pts[:, 0].max() <= f6.FREE_SURFACE_X_HI
    assert pts[:, 0].max() > f6.FREE_SURFACE_X_HI - 1.0


def test_nearest_segment_labels():
    lab = f6.nearest_segment([5.0, g.X_CREST + 1.0, 200.0])
    assert list(lab) == ["cut_face", "bench", "natural_ground"]


# --- fraction_below ---------------------------------------------------------

def test_fraction_below_counts_points_without_weights():
    fs = np.array([0.5, 0.9, 1.5, 3.0])
    assert f6.fraction_below(fs) == pytest.approx(0.5)


def test_fraction_below_uses_importance_weights():
    # Two strata: the weak one is oversampled 3x and all below 1.
    fs = np.array([0.5, 0.5, 0.5, 2.0])
    w = np.array([1 / 3, 1 / 3, 1 / 3, 3.0])      # restores 25/75 area split
    assert f6.fraction_below(fs) == pytest.approx(0.75)
    assert f6.fraction_below(fs, w=w) == pytest.approx(0.25)


def test_mask_collapses_a_surface_band():
    rng = np.random.default_rng(0)
    dist = rng.uniform(0, 30, 2000)
    fs = np.where(dist < 3.0, 0.5, 2.0)            # failing only near surface
    assert f6.fraction_below(fs) > 0.05
    assert f6.fraction_below(fs, keep=dist >= 5.0) == 0.0


def test_mask_does_not_collapse_a_deep_band():
    # Control for the previous test: same machinery, sub-unity points deep.
    rng = np.random.default_rng(0)
    dist = rng.uniform(0, 30, 2000)
    fs = np.where(dist > 15.0, 0.5, 2.0)
    assert f6.fraction_below(fs, keep=dist >= 5.0) > f6.fraction_below(fs)


def test_fraction_below_is_nan_when_everything_is_masked():
    fs = np.array([0.5, 2.0])
    assert math.isnan(f6.fraction_below(fs, keep=np.array([False, False])))


# --- tension_census and the clip -------------------------------------------

class _Mat:
    sigma_ci, m_b, s, a = 4.29e6, 0.05902, 1.045e-4, 0.5081


def test_clip_credits_tension_with_the_rock_mass_ucs():
    # The corrected docstring's claim: at sigma_3 < 0 the clipped envelope
    # returns sigma_ci*s**a, which is ABOVE the true GHB value.
    m = _Mat()
    s3 = np.array([-0.5 * m.s * m.sigma_ci / m.m_b])
    clipped = f6.hoek_brown_sigma1(s3, m)[0]
    true = s3[0] + m.sigma_ci * (m.m_b * s3[0] / m.sigma_ci + m.s) ** m.a
    assert clipped == pytest.approx(m.sigma_ci * m.s ** m.a)
    assert clipped > true


def test_tension_census_counts_points_hidden_by_the_clip():
    st = 7.6e3
    s3 = np.array([1e5, -1e3, -2e4, -2e4])
    fs = np.array([0.8, 1.5, 1.2, 0.9])
    c = f6.tension_census(s3, np.full(4, st), fs)
    assert c["n_sigma3_tensile"] == 3
    assert c["n_beyond_tensile_cutoff"] == 2
    assert c["n_hidden_by_clip"] == 1
    assert c["n_fs_below_1_and_sigma3_tensile"] == 1


def test_tension_census_hidden_is_zero_when_nothing_is_beyond_cutoff():
    c = f6.tension_census(np.array([1e5, -1e3]), np.full(2, 7.6e3),
                          np.array([0.8, 1.5]))
    assert c["n_hidden_by_clip"] == 0