"""
tests/test_interface.py
=======================

Guards the interface flux-continuity term. Four assertions, each aimed at a
specific way this can be wrong while still looking plausible in a loss curve.

  1. The flux is CONSISTENT with the interior residual. div* q* + C* dpsi*/dt*
     must reproduce richards_residual to machine precision. If it does not,
     the interface loss and the interior loss are enforcing two different
     physics and no amount of training reconciles them.

  2. A hydrostatic field gives ZERO flux, and therefore zero jump even across
     the most extreme K contrast in the domain. This is the same test field
     that verified the Richards residual (vg.psi_initial), reused deliberately.

  3. The jump is NONZERO for a field that is not in equilibrium. Without this
     the previous test is vacuous -- a function returning 0.0 unconditionally
     would pass it.

  4. Pi_R_diff / Pi_R_grav == Pi_R_hz. This is the algebraic identity that
     makes test 2 work, and it is exactly the relation that broke when H_ref
     went missing from Pi_R_diff on 4 Aug. Asserting it directly means the
     next such regression is named rather than inferred.

Run:  pytest tests/test_interface.py -v
"""
import pytest
import torch

from src.derivatives import as_inputs, grad, divergence
from src.nondim import SCALES
from src.residuals import (
    richards_residual,
    darcy_flux,
    interface_flux_jump,
    swcc_from_material,
)
from src.materials import load_materials


TOL = 1e-12


@pytest.fixture(scope="module")
def mats():
    return load_materials()


def _points(n=64, seed=20260810):
    """Interior points, float64, differentiable leaves."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, generator=g, dtype=torch.float64) * 0.6
    z = torch.rand(n, 1, generator=g, dtype=torch.float64) * 0.4 + 1.17
    t = torch.rand(n, 1, generator=g, dtype=torch.float64) * 0.5
    return as_inputs(x, z, t)


def _hydrostatic(s=SCALES):
    """psi = -(z - Z_WT), non-dimensionalised. Zero total-head gradient.

    In starred variables: psi* = -(z* - Z_WT/L_ref) * L_ref / H_ref.

    Returns psi* ONLY, shape (N,1). residuals._psi_of accepts a bare tensor
    and treats it as psi -- so returning a (N,3) block here would silently
    compute the flux on three fields at once (psi, u=0, v=0) and every
    assertion below would be meaningless. That is not hypothetical: it is
    what the first version of this file did.
    """
    z_wt_star = 197.0 / s.L_ref

    def fields(x, z, t):
        return -(z - z_wt_star) * s.L_ref / s.H_ref

    return fields


def _tilted(s=SCALES):
    """NOT in hydrostatic equilibrium: a lateral ramp plus a vertical excess,
    so grad psi + e_z != 0 and the flux is real. Shape (N,1), as above.

    The vertical coefficient is deliberately large enough to lift psi well off
    the deep-suction tail where K_r underflows -- at 70 m suction K_r is order
    1e-6 and every flux, jump included, is numerically invisible.
    """
    base = _hydrostatic(s)

    def fields(x, z, t):
        return base(x, z, t) + 0.35 * x + 0.60 * z

    return fields


# ---------------------------------------------------------------------------
# 1. Consistency with the interior residual
# ---------------------------------------------------------------------------
def test_flux_divergence_reproduces_richards_residual(mats):
    """div* q* + C* dpsi*/dt*  ==  richards_residual, term for term.

    This is the assertion that ties the interface term to the interior term.
    It is checked on the TILTED field, not the hydrostatic one: on the
    hydrostatic field both sides are zero and the test would pass for a
    completely wrong flux.
    """
    mat = mats["Mk"]
    x, z, t = _points()
    fields = _tilted()

    qx, qz = darcy_flux(fields, x, z, t, mat)
    div_q = divergence([qx, qz], [x, z])

    psi = fields(x, z, t)
    C = swcc_from_material(mat, SCALES).C_star(psi)
    dpsi_dt = grad(psi, t)

    lhs = div_q + C * dpsi_dt
    rhs = richards_residual(fields, x, z, t, mat, normalise="none")

    scale = rhs.abs().max().item() + 1e-300
    err = (lhs - rhs).abs().max().item() / scale
    assert err < 1e-10, f"flux/residual inconsistency, max rel err {err:.3e}"


# ---------------------------------------------------------------------------
# 2. Hydrostatic field: zero flux, zero jump
# ---------------------------------------------------------------------------
def test_hydrostatic_flux_is_zero(mats):
    x, z, t = _points()
    for name in ("Mk", "Mk_d", "Tm"):
        qx, qz = darcy_flux(_hydrostatic(), x, z, t, mats[name])
        assert qx.abs().max().item() < TOL, f"{name}: q_x nonzero"
        assert qz.abs().max().item() < TOL, f"{name}: q_z nonzero"


def test_hydrostatic_jump_is_zero_across_worst_contrast(mats):
    """Zero flux on both sides means zero jump regardless of the K contrast."""
    x, z, t = _points()
    n = 1.0 / (2.0 ** 0.5)
    for a, b in (("Mk", "Mk_d"), ("Mk", "Tm"), ("Mk_d", "Tm")):
        jump = interface_flux_jump(_hydrostatic(), x, z, t,
                                   mats[a], mats[b], n, n)
        assert jump.abs().max().item() < TOL, f"{a}|{b}: spurious jump"


# ---------------------------------------------------------------------------
# 3. Negative control -- the jump must be able to fire
# ---------------------------------------------------------------------------
def test_jump_is_nonzero_off_equilibrium(mats):
    """Without this, test 2 is satisfied by `return 0.0`.

    The contrast pair is Mk|Tm, NOT Mk|Mk_d. Mk and Mk_d are hydraulically
    identical -- same K_s = 1e-9, and the weak zone was assigned the marl's
    alpha and n by analogy (Phase-1 vg_basis: "marl analog, identical
    alpha/n"). Their flux jump is exactly zero by construction, which is
    correct physics, not a broken term. Asserting a jump there is what the
    first version of this test did and it failed for the right reason.

    Also asserts the self-jump vanishes: a material compared against itself
    must give exactly zero, so anything else means the two sides are not being
    evaluated at the same point.
    """
    x, z, t = _points()
    fields = _tilted()
    n = 1.0 / (2.0 ** 0.5)

    cross = interface_flux_jump(fields, x, z, t,
                                mats["Mk"], mats["Tm"], n, n)
    same = interface_flux_jump(fields, x, z, t,
                               mats["Mk"], mats["Mk"], n, n)

    qa_x, qa_z = darcy_flux(fields, x, z, t, mats["Mk"])
    qb_x, qb_z = darcy_flux(fields, x, z, t, mats["Tm"])
    q_scale = max(qa_x.abs().max().item(), qa_z.abs().max().item(),
                  qb_x.abs().max().item(), qb_z.abs().max().item())

    # Printed, not just asserted: these magnitudes decide w_interface later.
    # If q* is order 1e-13 the interface term is numerically invisible next to
    # L_PDE and needs normalising, not just weighting.
    print(f"\n  |q*| max          = {q_scale:.3e}"
          f"\n  |[[q*.n]]| max    = {cross.abs().max().item():.3e}"
          f"\n  jump / flux       = {cross.abs().max().item() / (q_scale + 1e-300):.3f}")

    assert same.abs().max().item() < TOL, "self-jump must vanish identically"
    assert q_scale > 1e-30, (
        "the flux itself underflowed -- the test field sits too deep in "
        "suction to exercise anything. Raise the vertical coefficient in "
        "_tilted; this is a bad test, not a bad interface term."
    )
    assert cross.abs().max().item() > 1e-3 * q_scale, (
        "flux jump is under 0.1% of the flux magnitude across the Mk|Tm "
        "contact -- the interface term cannot be doing anything"
    )


def test_mk_and_mk_d_are_hydraulically_identical(mats):
    """Documents, rather than assumes, why Mk|Mk_d gets no interface term.

    Mk_d is a MECHANICAL weak zone (E 38 MPa vs 209 MPa). Hydraulically it was
    given the marl's K_s and van Genuchten parameters by analogy -- a Tier-B
    assignment, not a measurement. Consequence: the Mk|Mk_d contact carries no
    flux discontinuity, and sample_interfaces should not spend collocation
    points on it.

    If someone later gives Mk_d its own K_s -- physically defensible, since a
    sheared D=1 zone would normally be MORE permeable -- this test fails and
    tells them the interface sampling assumption has changed. That is the
    point of it.
    """
    x, z, t = _points()
    fields = _tilted()
    n = 1.0 / (2.0 ** 0.5)

    jump = interface_flux_jump(fields, x, z, t,
                               mats["Mk"], mats["Mk_d"], n, n)
    assert jump.abs().max().item() < TOL, (
        "Mk and Mk_d now differ hydraulically. The Mk|Mk_d contact needs an "
        "interface term and sample_interfaces must be updated to include it."
    )


# ---------------------------------------------------------------------------
# 4. The scaling identity the hydrostatic test depends on
# ---------------------------------------------------------------------------
def test_pi_group_identity():
    """Pi_R_diff / Pi_R_grav == Pi_R_hz == H_ref / L_ref.

    Named explicitly because this is the relation that failed on 4 Aug when
    Pi_R_diff was missing H_ref (2.99e-6 instead of 8.97e-5). That bug made
    gravity outweigh capillary diffusion by ~30x and would have produced a
    plausible-looking, wrong wetting front.
    """
    ratio = SCALES.Pi_R_diff / SCALES.Pi_R_grav
    assert ratio == pytest.approx(SCALES.Pi_R_hz, rel=1e-12)
    assert ratio == pytest.approx(SCALES.H_ref / SCALES.L_ref, rel=1e-12)