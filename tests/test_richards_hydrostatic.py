"""
tests/test_richards_hydrostatic.py
===================================

THE test for the Richards residual.

At t = 0 the seepage IC is psi = -(z - Z_WT) with Z_WT = 197 m a.s.l. That is
hydrostatic: total head h = psi + z = Z_WT is CONSTANT everywhere, so the
Darcy flux q = -K grad(psi + z) is identically zero, so div q = 0, so the
residual is exactly zero — for ANY K(psi), however violently it varies.

That last clause is what gives the test its power. K spans ~5 orders of
magnitude across this domain (n = 1.2 marl, suction -3 m at the floor to
-168 m at the crest), so the diffusion and gravity terms are individually
large and must cancel to machine precision. The test therefore catches:

  * a dropped gravity term        -> gravity term survives uncancelled
  * a sign error on gravity       -> terms add instead of cancelling
  * a wrong dK/dpsi chain rule    -> partial cancellation only
  * Pi_R_diff / Pi_R_grav mutually inconsistent -> partial cancellation
    (this is exactly the 30x error found on 4 Aug: Pi_R_diff was T*K/L^2
     instead of T*K*H/L^2, and this test would have caught it)

WHAT IT DOES NOT TEST: dpsi/dt is zero here, so C*(psi) is multiplied by zero
and never checked. The specific moisture capacity still needs its own test
against torch.autograd.grad(theta, psi) in tests/test_materials.py.

Run:  pytest tests/test_richards_hydrostatic.py -v
  or: python -m tests.test_richards_hydrostatic
"""

from __future__ import annotations

import torch

from src.derivatives import as_inputs, grad
from src.nondim import SCALES
from src.residuals import SWCC, richards_residual, richards_residual_expanded

torch.set_default_dtype(torch.float64)

Z_WT = 197.0        # m a.s.l., karstic table (Ulusay et al. 2014, sec. 5.3)
Z_BASE = 200.0      # m a.s.l., domain floor
Z_TOP = 364.87      # m a.s.l., highest natural ground
TOL = 1e-10         # relative to the largest surviving term


# ---------------------------------------------------------------------------
# An analytic SWCC. Deliberately NOT materials.py: this test must fail only
# when residuals.py is wrong, never because a JSON parameter changed.
# Mk-like: alpha = 0.5 1/m, n = 1.2, theta_s = 0.427, theta_r = 0.05, Ks = Kref
# ---------------------------------------------------------------------------
ALPHA, N_VG, TH_S, TH_R = 0.5, 1.2, 0.427, 0.05
M_VG = 1.0 - 1.0 / N_VG
KS_STAR = 1.0                       # K_s / K_ref, with K_s = K_ref = 1e-6 m/s


def _Se(psi_phys: torch.Tensor) -> torch.Tensor:
    S = ALPHA * (psi_phys.abs() + 1e-12)
    return (1.0 + S ** N_VG) ** (-M_VG)


def _K_star(psi_star: torch.Tensor) -> torch.Tensor:
    """Mualem-van Genuchten, dimensionless. Unsaturated branch only —
    every point in the Section 5 domain has psi < 0 at t = 0."""
    Se = _Se(psi_star * SCALES.H_ref).clamp(1e-12, 1.0)
    inner = 1.0 - (1.0 - Se ** (1.0 / M_VG)) ** M_VG
    return KS_STAR * Se ** 0.5 * inner ** 2


def _C_star(psi_star: torch.Tensor) -> torch.Tensor:
    """C * H_ref. Never exercised by this test (dpsi/dt = 0); present so the
    call signature is complete and so the term shows up in the diagnostics."""
    S = ALPHA * (psi_star * SCALES.H_ref).abs() + 1e-12
    C_phys = ((TH_S - TH_R) * ALPHA * M_VG * N_VG
              * S ** (N_VG - 1.0) * (1.0 + S ** N_VG) ** (-M_VG - 1.0))
    return C_phys * SCALES.H_ref


ANALYTIC_SWCC = SWCC(C_star=_C_star, K_star=_K_star, name="Mk-like analytic")


# ---------------------------------------------------------------------------
# The analytic field, in dimensionless coordinates
# ---------------------------------------------------------------------------
def hydrostatic(x, z, t):
    """psi* = (Z_WT - z) / H_ref, with z = z* L_ref. Independent of x and t —
    which is itself part of the test: autograd returns None for those and
    derivatives.grad must turn that into zeros, not crash."""
    return (Z_WT - z * SCALES.L_ref) / SCALES.H_ref


def tilted(x, z, t):
    """A field with a genuine flux: total head varies linearly in x.
    Used as the negative control — the residual must NOT vanish here."""
    return (Z_WT - z * SCALES.L_ref) / SCALES.H_ref - 0.02 * x


def _collocation(n_side: int = 24):
    zs = torch.linspace(Z_BASE + 0.5, Z_TOP, n_side) / SCALES.L_ref
    xs = torch.linspace(0.0, 266.0, n_side) / SCALES.L_ref
    X, Z = torch.meshgrid(xs, zs, indexing="ij")
    T = torch.full_like(X, 3.0)                      # t* = 3 days, arbitrary
    return as_inputs(X.reshape(-1), Z.reshape(-1), T.reshape(-1))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_hydrostatic_residual_is_zero():
    x, z, t = _collocation()
    R, terms = richards_residual(hydrostatic, x, z, t, ANALYTIC_SWCC,
                                 return_terms=True)

    scale = max(terms["diffusion"].abs().max().item(),
                terms["gravity"].abs().max().item())

    # (a) the test is not vacuous: the terms it cancels are actually there
    assert scale > 1e-12, "gravity/diffusion terms vanished — test proves nothing"
    Kmin = terms["K_star"].abs().min().item()
    Kmax = terms["K_star"].abs().max().item()
    assert Kmax / Kmin > 1e3, f"K* varies only {Kmax / Kmin:.1f}x — pick a wider domain"

    # (b) the actual assertion
    rel = (R.abs().max() / scale).item()
    assert rel < TOL, f"hydrostatic residual {rel:.3e} of term scale, want < {TOL}"


def test_gradient_of_total_head_is_zero():
    """The reason the residual vanishes, asserted directly. If this fails the
    IC itself is wrong and the residual test above is meaningless."""
    x, z, t = _collocation(8)
    psi = hydrostatic(x, z, t)
    dpsi_dz = grad(psi, z)
    # dh*/dz* where h* = psi* + z* L_ref/H_ref
    dh_dz = dpsi_dz + SCALES.L_ref / SCALES.H_ref
    assert dh_dz.abs().max().item() < 1e-12
    assert grad(psi, x).abs().max().item() == 0.0     # no x dependence at all


def test_divergence_and_expanded_forms_agree():
    """Cross-checks the manual product rule against autograd's. A discrepancy
    means dK/dpsi is wrong in the expanded form — or that the divergence form
    is not seeing the chain rule at all."""
    x, z, t = _collocation(12)
    R_div = richards_residual(tilted, x, z, t, ANALYTIC_SWCC)
    R_exp = richards_residual_expanded(tilted, x, z, t, ANALYTIC_SWCC)
    denom = R_div.abs().max().item()
    assert denom > 0
    assert (R_div - R_exp).abs().max().item() / denom < 1e-10


def test_negative_control_tilted_head_gives_nonzero_residual():
    """Guards against a residual that returns ~0 for everything."""
    x, z, t = _collocation(12)
    R = richards_residual(tilted, x, z, t, ANALYTIC_SWCC)
    assert R.abs().max().item() > 1e-12


def test_dropping_gravity_breaks_it():
    """Reproduces the bug the test exists to catch: without Pi_R_grav dK/dz
    the hydrostatic residual is large. If this passes with a small number,
    the gravity term was never contributing and the main test is a no-op."""
    x, z, t = _collocation(12)
    _, terms = richards_residual(hydrostatic, x, z, t, ANALYTIC_SWCC,
                                 return_terms=True)
    R_broken = terms["storage"] + terms["diffusion"]      # gravity omitted
    scale = terms["diffusion"].abs().max().item()
    assert R_broken.abs().max().item() / scale > 0.99


def test_normalisation_choice_does_not_move_the_zero():
    """The Step 1.6 scaling decision is still open; whichever way it goes,
    the hydrostatic state must stay a solution."""
    x, z, t = _collocation(12)
    for mode in ("none", "gravity", "storage"):
        R, terms = richards_residual(hydrostatic, x, z, t, ANALYTIC_SWCC,
                                     normalise=mode, return_terms=True)
        scale = max(terms["diffusion"].abs().max().item(),
                    terms["gravity"].abs().max().item())
        if mode == "gravity":
            scale /= SCALES.Pi_R_grav
        elif mode == "storage":
            scale /= terms["C_star"].abs().min().item()
        assert (R.abs().max().item() / scale) < TOL, mode


def test_float32_would_not_have_been_enough():
    """Documents why as_inputs defaults to float64. Not a correctness test —
    it records the precision headroom, and will start failing if someone
    switches the default and quietly loses eight digits."""
    x, z, t = _collocation(8)
    R, terms = richards_residual(hydrostatic, x, z, t, ANALYTIC_SWCC,
                                 return_terms=True)
    scale = terms["gravity"].abs().max().item()
    rel = R.abs().max().item() / scale
    assert rel < 1e-10, f"float64 headroom lost: {rel:.2e}"


if __name__ == "__main__":
    x, z, t = _collocation()
    R, terms = richards_residual(hydrostatic, x, z, t, ANALYTIC_SWCC,
                                 return_terms=True)
    print(f"{'term':<12}{'max |.|':>14}")
    for k in ("storage", "diffusion", "gravity"):
        print(f"{k:<12}{terms[k].abs().max().item():>14.6e}")
    print(f"{'RESIDUAL':<12}{R.abs().max().item():>14.6e}")
    print(f"\nK* range: {terms['K_star'].min().item():.3e} .. "
          f"{terms['K_star'].max().item():.3e}  "
          f"({terms['K_star'].max().item() / terms['K_star'].min().item():.3g}x)")
    print(f"Pi_R_diff = {SCALES.Pi_R_diff:.4e}   Pi_R_grav = {SCALES.Pi_R_grav:.4e}"
          f"   ratio L/H = {SCALES.Pi_R_grav / SCALES.Pi_R_diff:.4f}")
