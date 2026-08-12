"""The storage term C* in coupling (Step 3.2, item 5).

C* has the worst bug history in this repo and has never run through the
residual path. Every test field so far has been hydrostatic, so dpsi*/dt* = 0
and C* is multiplied by zero: `test_richards_hydrostatic.py` reaching 1.1e-17
says nothing whatever about C*, and `test_materials.py` checks it standalone
but not in coupling.

The construction
----------------
Take psi* = f(z*) + k*t* and evaluate at t* = 0.

At t* = 0 the psi field is IDENTICAL for every k -- same suction, same K, same
gradient, same div* q*. Only dpsi*/dt* = k differs. So

    R(k) - R(0) == C*(psi*) * k        exactly, pointwise

with no cancellation error and without computing div* q* at all. That identity
is the whole file: it isolates C* inside the coupled residual and compares it
against the standalone value `test_materials.py` already guards.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from src.nondim import SCALES as S, Scales
from src.residuals import richards_residual
from src.sampling import sample_interior
from src.materials import load_materials, C_star, C_star
from src.step21_geometry.boundaries import Z_WT

# ===========================================================================
# ADAPTER -- fix these two if the signatures differ.
# ===========================================================================
MATS = load_materials()
N = 400
SEED = 20250812
DTYPE = torch.float64
ZW_STAR = Z_WT / S.L_ref


def call_residual(field, x, z, t, mat, s=S):
    """`richards_residual(fields, x, z, t, mat, s)`, as darcy_flux takes."""
    return richards_residual(field, x, z, t, mat, s)


def C_star_of(mat, psi_star, s=S):
    """`materials.C_star(psi, mat, s)` -- a module function, not a method."""
    return C_star(psi_star * s.H_ref, mat, s)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pts():
    """Interior points forced to t* = 0, one material at a time."""
    coll = sample_interior(N, SEED)
    out = {}
    for tag in np.unique(coll.tag):
        idx = torch.as_tensor(np.flatnonzero(coll.tag == tag), dtype=torch.long)
        x = coll.x[idx].detach().clone().requires_grad_(True)
        z = coll.z[idx].detach().clone().requires_grad_(True)
        t = torch.zeros_like(x).requires_grad_(True)
        out[str(tag)] = (x, z, t)
    return out


TAGS = ("Mk", "Mk_d", "Tm")


def transient(k: float = 0.0, a: float = 0.0):
    """psi* = -(z*-Z_WT*)/Pi_R_hz + a*(z*-Z_WT*) + k*t*.

    a = k = 0 is the hydrostatic anchor. k sets dpsi*/dt* = k everywhere,
    constant in space, so the storage term is C*(psi*) * k.
    """
    def field(x, z, t):
        dz = z - ZW_STAR
        return -dz / S.Pi_R_hz + a * dz + k * t + 0.0 * x
    return field


def wiggly(k: float = 1e-3, a: float = 0.0):
    """psi* with a NONLINEAR time dependence: + k*sin(t*).

    dpsi*/dt* = k*cos(t*), which is k at t* = 0 but has curvature, so a
    finite-difference time derivative and autograd only agree if the autograd
    path is right. `transient` alone cannot catch a derivative that is
    accidentally a difference quotient over the wrong interval.
    """
    def field(x, z, t):
        dz = z - ZW_STAR
        return -dz / S.Pi_R_hz + a * dz + k * torch.sin(t) + 0.0 * x
    return field


def _np(t):
    return t.detach().cpu().numpy().reshape(-1)


# ---------------------------------------------------------------------------
# The storage term exists at all
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tag", TAGS)
def test_hydrostatic_residual_is_still_zero(pts, tag):
    """Regression guard. The 1.1e-17 result must survive everything below."""
    x, z, t = pts[tag]
    R = call_residual(transient(k=0.0), x, z, t, MATS[tag])
    assert float(R.detach().abs().max()) < 1e-12


@pytest.mark.parametrize("tag", TAGS)
def test_a_transient_field_changes_the_residual(pts, tag):
    """Negative control. If C* were dropped from the residual entirely, this
    is the only test in the repo that would notice."""
    x, z, t = pts[tag]
    R0 = call_residual(transient(k=0.0), x, z, t, MATS[tag])
    Rk = call_residual(transient(k=1e-3), x, z, t, MATS[tag])
    assert float((Rk - R0).detach().abs().max()) > 0.0, (
        f"{tag}: dpsi*/dt* had no effect on the residual; the storage term is "
        f"not in the coupled path"
    )


# ---------------------------------------------------------------------------
# C* in coupling == C* standalone
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tag", TAGS)
@pytest.mark.parametrize("k", [1e-4, 1e-3, 1e-2])
def test_implied_C_star_matches_the_standalone_value(pts, tag, k):
    """The gap this file exists to close.

    At t* = 0, R(k) - R(0) = C*(psi*) * k pointwise. Divide out k and the
    result must equal what `test_materials.py` checks standalone. If the
    residual path applies a different C* -- a stale scale factor, a missing
    dtheta_ref -- the two disagree by exactly that factor.
    """
    x, z, t = pts[tag]
    mat = MATS[tag]

    R0 = call_residual(transient(k=0.0), x, z, t, mat)
    Rk = call_residual(transient(k=k), x, z, t, mat)
    implied = (Rk - R0) / k

    psi_star = transient(k=0.0)(x, z, t)
    expected = C_star_of(mat, psi_star)

    np.testing.assert_allclose(_np(implied), _np(expected), rtol=1e-9,
                               atol=1e-30)


@pytest.mark.parametrize("tag", TAGS)
def test_storage_term_is_linear_in_dpsi_dt(pts, tag):
    """C* multiplies dpsi*/dt* once, not twice and not under a nonlinearity."""
    x, z, t = pts[tag]
    mat = MATS[tag]

    R0 = call_residual(transient(k=0.0), x, z, t, mat)
    R1 = call_residual(transient(k=1e-3), x, z, t, mat)
    R3 = call_residual(transient(k=3e-3), x, z, t, mat)

    np.testing.assert_allclose(_np(R3 - R0), 3.0 * _np(R1 - R0),
                               rtol=1e-9, atol=1e-30)


@pytest.mark.parametrize("tag", TAGS)
def test_sign_of_the_storage_term(pts, tag):
    """C* > 0, so a wetting field (dpsi*/dt* > 0) must move the residual in the
    same direction everywhere. A sign slip here would look like a plausible
    drainage response in a trained model."""
    x, z, t = pts[tag]
    R0 = call_residual(transient(k=0.0), x, z, t, MATS[tag])
    Rk = call_residual(transient(k=1e-3), x, z, t, MATS[tag])
    d = _np(Rk - R0)
    assert (d > 0).all() or (d < 0).all(), (
        f"{tag}: the storage term changes sign across the domain, but C* is "
        f"positive definite and dpsi*/dt* is uniform"
    )


# ---------------------------------------------------------------------------
# The autograd time path
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tag", TAGS)
def test_nonlinear_time_dependence_gives_the_same_C_star(pts, tag):
    """psi* = ... + k*sin(t*) has dpsi*/dt* = k at t* = 0 but nonzero
    curvature. It must give the same implied C* as the linear field -- if it
    does not, the time derivative is not a derivative."""
    x, z, t = pts[tag]
    mat = MATS[tag]
    k = 1e-3

    R0 = call_residual(transient(k=0.0), x, z, t, mat)
    lin = (call_residual(transient(k=k), x, z, t, mat) - R0) / k
    sin = (call_residual(wiggly(k=k), x, z, t, mat) - R0) / k

    np.testing.assert_allclose(_np(sin), _np(lin), rtol=1e-9, atol=1e-30)


@pytest.mark.parametrize("tag", TAGS)
def test_storage_term_is_evaluated_at_the_sample_time(pts, tag):
    """At t* = pi/2 the sin field has dpsi*/dt* = k*cos(pi/2) = 0, so its
    storage contribution must vanish. Catches a time derivative frozen at
    t* = 0 -- which every previous test in this repo would have missed,
    because they all sample t* = 0.
    """
    x, z, _ = pts[tag]
    mat = MATS[tag]
    t = torch.full_like(x, float(np.pi / 2)).requires_grad_(True)

    R0 = call_residual(transient(k=0.0), x, z, t, mat)
    Rk = call_residual(wiggly(k=1e-3), x, z, t, mat)
    assert float((Rk - R0).abs().max()) < 1e-12, (
        f"{tag}: dpsi*/dt* is not zero at t* = pi/2; the time derivative may "
        f"be evaluated at the wrong point"
    )


# ---------------------------------------------------------------------------
# Scale propagation -- the ebf8f5c bug class
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tag", TAGS)
def test_C_star_respects_the_scales_argument(pts, tag):
    """`C_star` scale-propagation bug (ebf8f5c): a Scales instance passed in
    but not threaded through leaves the storage term on the module default.

    Doubling dtheta_ref must change the storage contribution. This asserts only
    that it CHANGES -- the direction is the constitutive model's business, and
    asserting a formula here would duplicate nondim.py.
    """
    x, z, t = pts[tag]
    mat = MATS[tag]
    other = dataclasses.replace(S, dtheta_ref=2.0 * S.dtheta_ref)

    a = call_residual(transient(k=1e-3), x, z, t, mat, S) \
        - call_residual(transient(k=0.0), x, z, t, mat, S)
    b = call_residual(transient(k=1e-3), x, z, t, mat, other) \
        - call_residual(transient(k=0.0), x, z, t, mat, other)

    assert not np.allclose(_np(a), _np(b), rtol=1e-6), (
        f"{tag}: doubling dtheta_ref did not change the storage term; the "
        f"Scales argument is not reaching C*"
    )