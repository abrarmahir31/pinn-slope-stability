"""
tests/test_fe_gravity.py — Step 3.3a, the offline gravity solve.

The point of an offline sigma_0 is that it can be verified against closed-form
answers the PINN cannot supply. So most of this file is patch tests on
synthetic geometry, not assertions about Isikdere.

Sign convention is TENSION POSITIVE throughout (D-3.3.1).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.fe_gravity import (build_mesh, element_stress, plane_strain_D,
                            solve_gravity)


class FlatLayer:
    """Homogeneous layer, flat surface, no material contact. Closed form:
    sigma_v = rho g h and sigma_h/sigma_v = nu/(1-nu) exactly."""
    X_MIN, X_MK_DIVIDE, Z_BASE = 0.0, 50.0, 0.0
    W = 100.0
    H = 100.0
    z_ground = staticmethod(lambda x: np.full_like(np.asarray(x, float), 100.0))
    z_tm_top = staticmethod(lambda x: np.full_like(np.asarray(x, float), -1e9))
    x_f1 = staticmethod(lambda z: np.full_like(np.asarray(z, float), 100.0))
    material_tag = staticmethod(lambda x, z: np.full(np.shape(x), "M"))


E0, NU0, RHO0, G0 = 1.0e8, 0.30, 2000.0, 9.81


def _flat(nz=40, ncol=10):
    m = build_mesh(FlatLayer, ncol, ncol, nz // 4, nz - nz // 4)
    nd = m.nodes
    fixed = []
    base = np.flatnonzero(np.abs(nd[:, 1] - 0.0) < 1e-9)
    fixed += list(2 * base) + list(2 * base + 1)
    for xv in (0.0, 100.0):
        col = np.flatnonzero(np.abs(nd[:, 0] - xv) < 1e-9)
        fixed += list(2 * col)
    E = np.full(len(m.tris), E0)
    nu = np.full(len(m.tris), NU0)
    rho = np.full(len(m.tris), RHO0)
    u = solve_gravity(m, E, nu, rho, sorted(set(fixed)), G0)
    return m, element_stress(m, u, E, nu), u


# ---------------------------------------------------------------------------
# Mesh integrity
# ---------------------------------------------------------------------------
def test_no_degenerate_elements_when_the_contact_is_outside_the_column():
    """Regression. When z_tm_top falls below the base, the contact split
    collapsed the lower n_below+1 nodes onto one point and produced a strip of
    zero-area elements. The patch test still reported the right K0, because
    degenerate rows contribute nothing — so the answer looked correct while a
    quarter of the mesh did not exist."""
    m = build_mesh(FlatLayer, 10, 10, 10, 30)
    assert m.area.min() > 1e-9
    assert np.isfinite(m.area).all()


def test_every_element_has_positive_orientation():
    m = build_mesh(FlatLayer, 8, 8, 8, 20)
    p = m.nodes[m.tris]
    cross = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
             - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    assert (cross > 0).all()


# ---------------------------------------------------------------------------
# Elasticity
# ---------------------------------------------------------------------------
def test_plane_strain_D_is_symmetric_and_positive_definite():
    D = plane_strain_D(E0, NU0)
    assert np.allclose(D, D.T)
    assert (np.linalg.eigvalsh(D) > 0).all()


def test_incompressible_limit_raises_rather_than_returning_garbage():
    """nu = 0.5 is singular in plane strain. It raises at the source, which is
    where it can be understood — a NaN propagating into the solve would
    surface as an unexplained NaN loss several steps later. No stratum is near
    0.5 (max is Mk_d at 0.30), so this is a guard, not a limitation."""
    with pytest.raises(ZeroDivisionError):
        plane_strain_D(E0, 0.5)


def test_stiffness_grows_without_bound_approaching_the_limit():
    """Negative control for the test above: the raise is a genuine
    singularity, not an arbitrary rejected input."""
    d = [plane_strain_D(E0, nu)[0, 0] for nu in (0.30, 0.45, 0.499)]
    assert d[2] > d[1] > d[0]


# ---------------------------------------------------------------------------
# Patch tests against closed form
# ---------------------------------------------------------------------------
def test_flat_layer_reproduces_K0_equals_nu_over_one_minus_nu():
    m, sig, _ = _flat()
    c = m.centroids
    depth = FlatLayer.H - c[:, 1]
    sv, sh = -sig[:, 1], -sig[:, 0]
    ok = depth < 0.9 * FlatLayer.H          # exclude the fixed-base layer
    assert np.mean(sh[ok] / sv[ok]) == pytest.approx(NU0 / (1.0 - NU0), rel=1e-3)


def test_flat_layer_reproduces_rho_g_h():
    m, sig, _ = _flat()
    c = m.centroids
    depth = FlatLayer.H - c[:, 1]
    sv = -sig[:, 1]
    ok = depth < 0.9 * FlatLayer.H
    err = np.abs(sv[ok] - RHO0 * G0 * depth[ok]).max() / (RHO0 * G0 * FlatLayer.H)
    assert err < 0.02


def test_flat_layer_has_no_shear():
    """Negative control for the Isikdere result. A flat surface must give
    sigma_xz = 0; the ~97 kPa mean shear found on the real geometry is
    therefore the slope, not the solver."""
    m, sig, _ = _flat()
    sv_scale = RHO0 * G0 * FlatLayer.H
    assert np.abs(sig[:, 2]).max() / sv_scale < 0.01


def test_shear_converges_to_zero_under_refinement_on_a_flat_layer():
    """The residual shear above is discretisation, not physics, so it must
    shrink with h. If it plateaus, the mesh has a systematic bias and the
    Isikdere shear cannot be trusted either."""
    e = []
    for nz in (20, 40, 80):
        _, sig, _ = _flat(nz=nz)
        e.append(np.abs(sig[:, 2]).max())
    assert e[1] < 0.7 * e[0]
    assert e[2] < 0.7 * e[1]


def test_gravity_direction_is_downward():
    """Negative control on the load vector. Flipping the sign of the body
    force gives a state that still satisfies equilibrium and still looks like
    a stress field."""
    m, sig, u = _flat()
    assert u[1::2].min() < 0.0                  # settles
    assert (-sig[:, 1]).mean() > 0.0            # compression-positive sigma_v


def test_zero_density_gives_zero_stress():
    """The `return 0.0` control, inverted: with no weight there is nothing to
    balance, so every stress must vanish."""
    m = build_mesh(FlatLayer, 8, 8, 8, 20)
    nd = m.nodes
    fixed = []
    base = np.flatnonzero(np.abs(nd[:, 1]) < 1e-9)
    fixed += list(2 * base) + list(2 * base + 1)
    E = np.full(len(m.tris), E0)
    nu = np.full(len(m.tris), NU0)
    u = solve_gravity(m, E, nu, np.zeros(len(m.tris)), sorted(set(fixed)), G0)
    sig = element_stress(m, u, E, nu)
    assert np.abs(sig).max() == pytest.approx(0.0, abs=1e-9)


def test_stiffness_does_not_change_the_stress_in_a_homogeneous_layer():
    """Gravity turn-on in a homogeneous body is statically determinate in the
    vertical: sigma_v = rho g h regardless of E. If E leaks into the stress,
    the load vector is scaled wrong somewhere."""
    out = []
    for E in (1.0e7, 1.0e9):
        m = build_mesh(FlatLayer, 8, 8, 8, 20)
        nd = m.nodes
        fixed = []
        base = np.flatnonzero(np.abs(nd[:, 1]) < 1e-9)
        fixed += list(2 * base) + list(2 * base + 1)
        for xv in (0.0, 100.0):
            fixed += list(2 * np.flatnonzero(np.abs(nd[:, 0] - xv) < 1e-9))
        Ee = np.full(len(m.tris), E)
        nue = np.full(len(m.tris), NU0)
        u = solve_gravity(m, Ee, nue, np.full(len(m.tris), RHO0),
                          sorted(set(fixed)), G0)
        out.append(element_stress(m, u, Ee, nue)[:, 1])
    assert np.allclose(out[0], out[1], rtol=1e-9)


# ---------------------------------------------------------------------------
# Global equilibrium
# ---------------------------------------------------------------------------
def test_reactions_balance_the_total_weight():
    """The check that makes 'equilibrated' mean something. This is what the
    K0 column fails: it satisfies vertical equilibrium by construction and
    horizontal equilibrium not at all."""
    from src.fe_gravity import assemble
    m, _, u = _flat()
    E = np.full(len(m.tris), E0)
    nu = np.full(len(m.tris), NU0)
    rho = np.full(len(m.tris), RHO0)
    K, F = assemble(m, E, nu, rho, G0)
    R = K @ u - F
    W = (rho * m.area * G0).sum()
    assert R[1::2].sum() / W == pytest.approx(1.0, rel=1e-9)
    assert abs(R[0::2].sum()) / W == pytest.approx(0.0, abs=1e-10)


def test_per_element_array_and_callable_agree():
    """`_per_element` accepts both. A silent mismatch there would give the
    right stiffness with the wrong body force."""
    from src.fe_gravity import assemble
    m = build_mesh(FlatLayer, 8, 8, 8, 20)
    K1, F1 = assemble(m, lambda t: E0, lambda t: NU0, lambda t: RHO0, G0)
    K2, F2 = assemble(m, np.full(len(m.tris), E0), np.full(len(m.tris), NU0),
                      np.full(len(m.tris), RHO0), G0)
    assert np.allclose(F1, F2)
    assert abs(K1 - K2).max() == pytest.approx(0.0, abs=1e-6)


def test_wrong_length_per_element_array_is_rejected():
    from src.fe_gravity import assemble
    m = build_mesh(FlatLayer, 8, 8, 8, 20)
    with pytest.raises(ValueError):
        assemble(m, np.full(3, E0), lambda t: NU0, lambda t: RHO0, G0)


def test_column_endpoints_land_exactly_on_the_boundary():
    """Regression, and a portability one. pt(1.0) computes
    z_bot + 1.0*(z_top - z_bot), which IEEE does not guarantee returns exactly
    z_top. On one numpy build every node landed on the surface; on another,
    3 of 3483 landed 1 ulp outside and `inside_domain` called them "outside".
    Endpoints are now snapped, so the mesh is bit-reproducible across builds."""
    m = build_mesh(FlatLayer, 12, 12, 10, 20)
    top = m.nodes[m.nodes[:, 1] > 99.0]
    assert len(top) > 10
    assert (top[:, 1] == 100.0).all()
    base = m.nodes[m.nodes[:, 1] < 1.0]
    assert (base[:, 1] == 0.0).all()
