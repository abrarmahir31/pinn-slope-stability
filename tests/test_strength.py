"""Tests for `src/strength.py` and `src/failure_surface.py`.

Repo convention (step34_handoff.md): every claim gets a test INCLUDING a
negative control -- assert the thing is zero when it should be AND nonzero when
it should not be, or the first assertion is satisfied by `return 0.0`. Every
`!= pytest.approx(...)` carries an explicit `abs=` because quantities here live
near 1e-12 and the default makes a negative control silently vacuous.

These modules are the algebra layer and import no torch, so this file runs on
any box -- including one without a GPU, which is the point.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src import failure_surface as fs
from src import strength as st

# Section 5 design pair -- Phase-1 dataset, mc_strength/design_pair, flagged
# there as "THIS is the pair to reduce in the MC-SSR sweep (Step 5.1)".
MC_DESIGN = st.MCParams(c=4400.0, phi=math.radians(15.4))

# Sekkoy marl rock mass -- properties.py / Phase-1 strata/marl_sekkoy.
# D = 1 (blast-damaged), as stored for every stratum in the Phase-1 dataset.
# Using the D = 0 form here gives s = 3.87e-3 instead of 2.40e-4 and silently
# inflates the unconfined rock-mass strength by a factor of four.
GHB_MARL = st.ghb_from_gsi(sigma_ci=17.9e6, gsi=50, m_i=4, D=1.0)

SIG3_RANGE = (0.0, 8.5e5)   # 0-850 kPa, the span the Phase-1 MC equivalents use


# ---------------------------------------------------------------------------
# Mohr-Coulomb
# ---------------------------------------------------------------------------
def test_reduce_mc_divides_tan_phi_not_phi():
    r = st.reduce_mc(MC_DESIGN, 1.5)
    assert r.tan_phi == pytest.approx(MC_DESIGN.tan_phi / 1.5, rel=1e-14)
    assert r.c == pytest.approx(MC_DESIGN.c / 1.5, rel=1e-14)


def test_reduce_mc_is_not_angle_division():
    """Negative control for the quiet error the docstring warns about."""
    r = st.reduce_mc(MC_DESIGN, 1.5)
    wrong = MC_DESIGN.phi / 1.5
    assert r.phi != pytest.approx(wrong, abs=1e-6)


def test_reduce_mc_identity_at_srf_one():
    r = st.reduce_mc(MC_DESIGN, 1.0)
    assert r.c == pytest.approx(MC_DESIGN.c, rel=1e-15)
    assert r.phi == pytest.approx(MC_DESIGN.phi, rel=1e-15)


def test_reduce_mc_is_monotonically_weakening():
    prev = None
    for srf in (1.0, 1.2, 1.5, 2.0, 3.0):
        p = st.reduce_mc(MC_DESIGN, srf)
        tau = st.mc_sigma1(1.0e5, p)
        if prev is not None:
            assert tau < prev
        prev = tau


def test_mc_yield_is_zero_on_the_envelope():
    sig3 = np.array([0.0, 5.0e4, 2.0e5, 8.0e5])
    sig1 = st.mc_sigma1(sig3, MC_DESIGN)
    f = st.mc_yield(sig1, sig3, MC_DESIGN)
    assert np.allclose(f, 0.0, atol=1e-8)


def test_mc_yield_is_nonzero_off_the_envelope():
    """Negative control: the zero above must not be satisfied by returning 0."""
    sig3 = np.array([0.0, 5.0e4, 2.0e5, 8.0e5])
    sig1 = st.mc_sigma1(sig3, MC_DESIGN) * 1.10
    f = st.mc_yield(sig1, sig3, MC_DESIGN)
    assert np.all(np.abs(f) > 1e2)
    assert np.all(f > 0.0)          # overstressed is inadmissible, not elastic


def test_mc_strength_ratio_is_one_on_the_envelope():
    sig3 = np.array([1.0e4, 1.0e5, 5.0e5])
    sig1 = st.mc_sigma1(sig3, MC_DESIGN)
    assert np.allclose(st.mc_strength_ratio(sig1, sig3, MC_DESIGN), 1.0,
                       rtol=1e-10)


def test_mc_strength_ratio_exceeds_one_inside_the_envelope():
    sig3 = np.array([1.0e4, 1.0e5, 5.0e5])
    sig1 = 0.5 * (st.mc_sigma1(sig3, MC_DESIGN) + sig3)
    assert np.all(st.mc_strength_ratio(sig1, sig3, MC_DESIGN) > 1.0)


def test_reduce_mc_rejects_nonpositive_srf():
    with pytest.raises(ValueError):
        st.reduce_mc(MC_DESIGN, 0.0)


# ---------------------------------------------------------------------------
# GHB envelope
# ---------------------------------------------------------------------------
def test_ghb_from_gsi_reproduces_the_stored_tm_parameters():
    """Binds the generating formulae to properties.py TM, which stores the
    rounded values independently. Drift in either shows up here."""
    p = st.ghb_from_gsi(sigma_ci=70.0e6, gsi=60, m_i=12, D=1.0)
    assert p.m_b == pytest.approx(0.6892, rel=1e-3)
    assert p.s == pytest.approx(1.2726e-3, rel=1e-3)
    assert p.a == pytest.approx(0.5028, rel=1e-3)


def test_ghb_unconfined_strength_matches_closed_form():
    assert st.ghb_sigma1(0.0, GHB_MARL) == pytest.approx(
        GHB_MARL.ucs_rockmass, rel=1e-12)


def test_ghb_envelope_is_monotone_in_confinement():
    sig3 = np.linspace(0.0, 1.0e6, 50)
    sig1 = st.ghb_sigma1(sig3, GHB_MARL)
    assert np.all(np.diff(sig1) > 0.0)


def test_ghb_yield_is_zero_on_the_envelope_and_nonzero_off_it():
    sig3 = np.array([0.0, 1.0e5, 6.0e5])
    sig1 = st.ghb_sigma1(sig3, GHB_MARL)
    assert np.allclose(st.ghb_yield(sig1, sig3, GHB_MARL), 0.0, atol=1e-6)
    assert np.all(st.ghb_yield(1.05 * sig1, sig3, GHB_MARL) > 1e2)


def test_ghb_tensile_clamp_returns_finite_below_the_cutoff():
    """Below -s*sigma_ci/m_b the bracket is negative and a**0.5 is NaN."""
    cutoff = -GHB_MARL.s * GHB_MARL.sigma_ci / GHB_MARL.m_b
    sig3 = np.array([2.0 * cutoff, cutoff, 0.5 * cutoff])
    out = st.ghb_sigma1(sig3, GHB_MARL, clamp_tensile=True)
    assert np.all(np.isfinite(out))
    nan_out = st.ghb_sigma1(sig3, GHB_MARL, clamp_tensile=False)
    assert np.any(~np.isfinite(nan_out))     # negative control for the clamp


# ---------------------------------------------------------------------------
# Instantaneous Mohr-Coulomb
# ---------------------------------------------------------------------------
def test_ghb_instantaneous_mc_line_is_tangent_to_the_envelope():
    """The tangent line must touch the envelope and not cross it: the envelope
    is concave, so the MC line through the tangent point lies ABOVE it
    everywhere else."""
    sig3_t = 2.0e5
    c_i, phi_i = st.ghb_instantaneous_mc(sig3_t, GHB_MARL)
    line = st.MCParams(c=float(c_i), phi=float(phi_i))

    assert st.mc_sigma1(sig3_t, line) == pytest.approx(
        st.ghb_sigma1(sig3_t, GHB_MARL), rel=1e-9)

    sig3 = np.linspace(1.0e4, 8.0e5, 80)
    gap = st.mc_sigma1(sig3, line) - st.ghb_sigma1(sig3, GHB_MARL)
    assert np.all(gap > -1.0)                       # touches, never dips under
    assert np.max(gap) > 1e3                        # negative control: not flat


def test_ghb_instantaneous_mc_degenerates_to_pure_cohesion():
    """m_b -> 0 is a frictionless material with UCS = sigma_ci * s**a."""
    p = st.GHBParams(sigma_ci=17.9e6, m_b=1e-12, s=GHB_MARL.s, a=0.5)
    c_i, phi_i = st.ghb_instantaneous_mc(3.0e5, p)
    assert float(phi_i) == pytest.approx(0.0, abs=1e-9)
    assert float(c_i) == pytest.approx(0.5 * p.ucs_rockmass, rel=1e-6)


def test_ghb_instantaneous_friction_falls_with_confinement():
    """Curvature of the GHB envelope, restated as a physical claim."""
    sig3 = np.array([2.5e4, 1.0e5, 4.0e5, 8.5e5])
    _, phi = st.ghb_instantaneous_mc(sig3, GHB_MARL)
    assert np.all(np.diff(phi) < 0.0)


def test_stored_m_b_matches_the_generating_formula():
    """marl_sekkoy: GSI 50, D 1. Binds the dataset value to Hoek (2002)."""
    p = st.ghb_from_gsi(sigma_ci=17.9e6, gsi=50, m_i=4, D=1.0)
    assert p.m_b == pytest.approx(0.1125, rel=1e-3)
    assert p.s == pytest.approx(2.404e-4, rel=1e-3)
    assert p.a == pytest.approx(0.5057, rel=1e-3)


def test_tangent_friction_lies_below_the_phase1_best_fit():
    """The dataset's `hoek_brown_mc_equivalents` are Hoek's closed-form LINEAR
    fits over 0 <= sig3 <= sig3max. The GHB envelope is concave, so the secant
    fit over [0, S] is steeper than the tangent AT S -- always, and by more at
    low confinement where the curvature is worst. Getting this ordering
    backwards is how you convince yourself the parameters disagree when they
    do not.
    """
    for sig3max, phi_fit_deg in ((2.5e4, 43.19), (1.0e5, 36.57),
                                 (4.0e5, 27.03), (8.5e5, 21.72)):
        _, phi_t = st.ghb_instantaneous_mc(sig3max, GHB_MARL)
        assert math.degrees(float(phi_t)) < phi_fit_deg


def test_the_gap_to_the_best_fit_narrows_with_confinement():
    """Negative control on the ordering above: the gap must be a real
    curvature effect that shrinks as the envelope straightens out, not a
    constant offset that would satisfy the inequality for the wrong reason.

    Marl, D = 1: 7.95 deg at sig3max = 25 kPa down to 6.07 deg at 850 kPa.
    Both the ordering and this trend are sensitive to m_i and D -- an earlier
    draft of this file used m_i = 7 and the D = 0 form of s, and the trend came
    out reversed. If this test flips, suspect the parameters before the physics.
    """
    gaps = []
    for sig3max, phi_fit_deg in ((2.5e4, 43.19), (8.5e5, 21.72)):
        _, phi_t = st.ghb_instantaneous_mc(sig3max, GHB_MARL)
        gaps.append(phi_fit_deg - math.degrees(float(phi_t)))
    assert gaps[0] > gaps[1] + 1.0


# ---------------------------------------------------------------------------
# GHB strength reduction
# ---------------------------------------------------------------------------
def test_reduce_ghb_is_identity_at_srf_one():
    r = st.reduce_ghb(GHB_MARL, 1.0, sig3_range=SIG3_RANGE)
    assert r.m_b == pytest.approx(GHB_MARL.m_b, rel=1e-4)
    assert r.s == pytest.approx(GHB_MARL.s, rel=1e-3)
    assert r.a == GHB_MARL.a


def test_reduce_ghb_weakens_the_envelope_across_the_fit_range():
    r = st.reduce_ghb(GHB_MARL, 1.5, sig3_range=SIG3_RANGE)
    sig3 = np.linspace(*SIG3_RANGE, 40)
    base = st.ghb_sigma1(sig3, GHB_MARL)
    red = st.ghb_sigma1(sig3, r)
    assert np.all(red < base)
    assert np.max(base - red) > 1e4          # negative control: not a no-op


def test_reduce_ghb_is_monotone_in_srf():
    prev = None
    for srf in (1.0, 1.25, 1.5, 2.0):
        r = st.reduce_ghb(GHB_MARL, srf, sig3_range=SIG3_RANGE)
        strength = float(st.ghb_sigma1(3.0e5, r))
        if prev is not None:
            assert strength < prev
        prev = strength


def test_reduce_ghb_halves_the_instantaneous_shear_strength_at_srf_two():
    """The defining property: the REDUCED envelope's tangent pair should be the
    original tangent pair divided by SRF. Checked mid-range, where the fit is
    best conditioned -- this is the claim the whole method rests on."""
    srf = 2.0
    sig3 = 3.0e5
    r = st.reduce_ghb(GHB_MARL, srf, sig3_range=SIG3_RANGE)

    c0, phi0 = st.ghb_instantaneous_mc(sig3, GHB_MARL)
    cr, phir = st.ghb_instantaneous_mc(sig3, r)

    assert float(cr) == pytest.approx(float(c0) / srf, rel=0.10)
    assert math.tan(float(phir)) == pytest.approx(
        math.tan(float(phi0)) / srf, rel=0.10)


def test_reduce_ghb_rejects_a_degenerate_sig3_range():
    with pytest.raises(ValueError):
        st.reduce_ghb(GHB_MARL, 1.5, sig3_range=(5.0e5, 5.0e5))
    with pytest.raises(ValueError):
        st.reduce_ghb(GHB_MARL, 1.5, sig3_range=(-1.0, 5.0e5))


def test_reduce_ghb_result_depends_on_the_fit_range():
    """The range is a modelling choice, not a numerical detail. If this test
    ever passes trivially, the docstring's warning has stopped being true."""
    narrow = st.reduce_ghb(GHB_MARL, 1.5, sig3_range=(0.0, 1.0e5))
    wide = st.reduce_ghb(GHB_MARL, 1.5, sig3_range=(0.0, 2.0e6))
    assert narrow.m_b != pytest.approx(wide.m_b, abs=1e-6)


# ---------------------------------------------------------------------------
# Principal stresses and the sign flip
# ---------------------------------------------------------------------------
def test_principal_stresses_flip_tension_to_compression_positive():
    """A tension-positive column under gravity has szz negative; compression-
    positive output must be positive, with sig1 the vertical."""
    sxx = np.array([-0.5]);  szz = np.array([-1.0]);  sxz = np.array([0.0])
    sig1, sig3 = st.principal_stresses_from_cartesian(sxx, szz, sxz)
    assert sig1.item() == pytest.approx(1.0)
    assert sig3.item() == pytest.approx(0.5)


def test_principal_stresses_are_ordered():
    rng = np.random.default_rng(0)
    sxx, szz, sxz = rng.normal(size=(3, 200))
    sig1, sig3 = st.principal_stresses_from_cartesian(sxx, szz, sxz)
    assert np.all(sig1 >= sig3)


def test_principal_stresses_preserve_invariants():
    rng = np.random.default_rng(1)
    sxx, szz, sxz = rng.normal(size=(3, 100))
    sig1, sig3 = st.principal_stresses_from_cartesian(sxx, szz, sxz)
    assert np.allclose(sig1 + sig3, -(sxx + szz))            # trace, flipped
    assert np.allclose(sig1 - sig3,
                       np.sqrt((sxx - szz) ** 2 + 4.0 * sxz ** 2))


def test_pure_shear_maps_to_equal_and_opposite_principals():
    sig1, sig3 = st.principal_stresses_from_cartesian(
        np.array([0.0]), np.array([0.0]), np.array([0.3]))
    assert sig1.item() == pytest.approx(0.3)
    assert sig3.item() == pytest.approx(-0.3)


# ---------------------------------------------------------------------------
# Failure surface
# ---------------------------------------------------------------------------
def test_gamma_max_matches_the_analytic_value_in_pure_shear():
    """Tensor eps_xz = 0.1 is engineering gamma = 0.2, and in pure shear the
    principal difference IS the engineering shear strain."""
    assert fs.gamma_max(0.0, 0.0, 0.1) == pytest.approx(0.2, rel=1e-14)


def test_gamma_max_vanishes_under_rigid_rotation():
    """Negative control on frame-invariance. A rigid rotation has
    du/dz = -dv/dx, so the TENSOR shear strain is exactly zero -- and the
    normal strains are zero too."""
    assert fs.gamma_max(0.0, 0.0, 0.0) == pytest.approx(0.0, abs=1e-40)


def test_gamma_max_is_nonzero_for_simple_shear():
    """Negative control on the control: simple shear is rotation PLUS strain,
    and must not come out zero."""
    # du/dz = 0.2, dv/dx = 0 -> tensor eps_xz = 0.1
    assert fs.gamma_max(0.0, 0.0, 0.1) > 1e-3


def test_gamma_max_is_rotation_invariant():
    rng = np.random.default_rng(2)
    exx, ezz, exz = rng.normal(size=3)
    g0 = fs.gamma_max(exx, ezz, exz)
    for th in (0.3, 1.1, 2.7):
        c, s = math.cos(th), math.sin(th)
        exx_r = exx * c * c + ezz * s * s + 2 * exz * s * c
        ezz_r = exx * s * s + ezz * c * c - 2 * exz * s * c
        exz_r = (ezz - exx) * s * c + exz * (c * c - s * s)
        assert fs.gamma_max(exx_r, ezz_r, exz_r) == pytest.approx(g0, rel=1e-12)


def test_principal_strains_are_ordered_and_sum_to_the_trace():
    e1, e3 = fs.principal_strains(0.01, -0.004, 0.006)
    assert e1 >= e3
    assert e1 + e3 == pytest.approx(0.01 - 0.004, rel=1e-14)


def _synthetic_band(nx=120, nz=90, width=6.0):
    """A planted shear band along z = 40 + 0.35x, for ridge-tracing tests."""
    x = np.linspace(0.0, 268.0, nx)
    z = np.linspace(120.0, 200.0, nz)
    X, Z = np.meshgrid(x, z)
    Z_band = 130.0 + 0.22 * X
    G = np.exp(-0.5 * ((Z - Z_band) / width) ** 2)
    return X, Z, G, Z_band


def test_extract_ridge_recovers_a_planted_band():
    X, Z, G, Z_band = _synthetic_band()
    xr, zr = fs.extract_ridge(X, Z, G)
    assert xr.size > 100
    expected = 130.0 + 0.22 * xr
    dz = float(np.abs(Z[1, 0] - Z[0, 0]))
    assert np.max(np.abs(zr - expected)) < dz      # within one grid spacing


def test_extract_ridge_drops_quiet_columns():
    X, Z, G, _ = _synthetic_band()
    G[:, :30] *= 0.01                              # kill the band on the left
    xr, _ = fs.extract_ridge(X, Z, G, min_frac=0.25)
    assert xr.min() > X[0, 29]


def test_extract_ridge_rejects_a_dead_field():
    X, Z, G, _ = _synthetic_band()
    with pytest.raises(ValueError):
        fs.extract_ridge(X, Z, np.zeros_like(G))


def test_localisation_intensity_separates_banded_from_diffuse():
    X, Z, G, _ = _synthetic_band(width=3.0)
    diffuse = np.ones_like(G) + 0.01 * np.random.default_rng(3).normal(
        size=G.shape)
    assert fs.localisation_intensity(G) > 5.0 * fs.localisation_intensity(
        diffuse)


def test_band_width_tracks_the_planted_width():
    _, _, _, _ = _synthetic_band()
    w_narrow = fs.band_width(*_synthetic_band(width=3.0)[:3])
    w_wide = fs.band_width(*_synthetic_band(width=9.0)[:3])
    assert w_wide > 2.0 * w_narrow


def test_band_width_is_insensitive_to_post_processing_resolution():
    """The mesh-objectivity check, applied to itself: a band of fixed physical
    width must measure the same on a finer grid. If this fails, `band_width`
    cannot be used to diagnose resolution artefacts in the real field."""
    coarse = fs.band_width(*_synthetic_band(nx=120, nz=90, width=6.0)[:3])
    fine = fs.band_width(*_synthetic_band(nx=240, nz=180, width=6.0)[:3])
    assert fine == pytest.approx(coarse, rel=0.15)


# ---------------------------------------------------------------------------
# Yield constraint (src/plasticity.py)
# ---------------------------------------------------------------------------
from src import plasticity as pl                                    # noqa: E402

SIG_REF = 1.0e6


def test_yield_violation_is_zero_inside_the_envelope():
    sig3 = np.array([1.0e4, 1.0e5, 5.0e5])
    sig1 = 0.5 * (st.mc_sigma1(sig3, MC_DESIGN) + sig3)     # well inside
    v = pl.yield_violation(sig1, sig3, MC_DESIGN, "MC", sigma_scale=SIG_REF)
    assert np.all(v == 0.0)


def test_yield_violation_is_positive_outside_the_envelope():
    """Negative control: the zero above must not come from a dead relu."""
    sig3 = np.array([1.0e4, 1.0e5, 5.0e5])
    sig1 = 1.5 * st.mc_sigma1(sig3, MC_DESIGN)
    v = pl.yield_violation(sig1, sig3, MC_DESIGN, "MC", sigma_scale=SIG_REF)
    assert np.all(v > 1e-3)


def test_L_yield_grows_as_strength_is_reduced():
    """The mechanism the whole sweep depends on: shrink the surface, and a
    FIXED stress field violates it harder. If this is flat, SSR cannot work."""
    sig3 = np.linspace(1.0e4, 6.0e5, 50)
    sig1 = st.mc_sigma1(sig3, MC_DESIGN)                    # at yield at SRF 1
    prev = None
    for srf in (1.0, 1.5, 2.0, 3.0):
        p = st.reduce_mc(MC_DESIGN, srf)
        L = pl.L_yield(sig1, sig3, p, "MC", sigma_scale=SIG_REF)
        if prev is not None:
            assert L > prev
        prev = L


def test_L_yield_is_zero_at_srf_one_on_the_envelope():
    sig3 = np.linspace(1.0e4, 6.0e5, 50)
    sig1 = st.mc_sigma1(sig3, MC_DESIGN)
    L = pl.L_yield(sig1, sig3, MC_DESIGN, "MC", sigma_scale=SIG_REF)
    assert L == pytest.approx(0.0, abs=1e-30)


def test_admissible_fraction_moves_from_one_to_zero():
    sig3 = np.linspace(1.0e4, 6.0e5, 200)
    sig1 = 0.9 * st.mc_sigma1(sig3, MC_DESIGN)
    assert pl.admissible_fraction(sig1, sig3, MC_DESIGN, "MC",
                                  sigma_scale=SIG_REF) == 1.0
    weak = st.reduce_mc(MC_DESIGN, 5.0)
    assert pl.admissible_fraction(sig1, sig3, weak, "MC",
                                  sigma_scale=SIG_REF) == 0.0


def test_ghb_yield_violation_responds_to_reduction():
    sig3 = np.linspace(1.0e4, 6.0e5, 50)
    sig1 = st.ghb_sigma1(sig3, GHB_MARL)
    base = pl.L_yield(sig1, sig3, GHB_MARL, "GHB", sigma_scale=SIG_REF)
    red = st.reduce_ghb(GHB_MARL, 2.0, sig3_range=SIG3_RANGE)
    assert base == pytest.approx(0.0, abs=1e-20)
    assert pl.L_yield(sig1, sig3, red, "GHB", sigma_scale=SIG_REF) > 1e-6


def test_reduced_material_params_refuses_mc_without_the_design_pair():
    """The swap this guards against moves the FOS more than the MC-vs-GHB
    difference Step 5.2 exists to measure."""
    class _Mat:
        sigma_ci, m_b, s, a = 17.9e6, 0.1125, 2.404e-4, 0.5057
    with pytest.raises(ValueError):
        pl.reduced_material_params(_Mat(), "MC", 1.5)
    with pytest.raises(ValueError):
        pl.reduced_material_params(_Mat(), "GHB", 1.5)      # no sig3_range