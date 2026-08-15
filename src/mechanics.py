"""
src/mechanics.py — mechanical residual, autograd layer only.
============================================================

Same design rule as `residuals.py`: the algebra lives in `nondim.py`
(`mechanical_residual_nd`, `total_stress_nd`) and in `coupling.py`. This
module differentiates and hands over. It does not re-derive equilibrium, and
it does not reimplement the constitutive model — chi and theta come from
`materials.Se` / `materials.theta`.

SIGN CONVENTION — TENSION POSITIVE everywhere in this file.
-----------------------------------------------------------
sigma = D:eps with eps from displacement gradients is tension-positive by
construction; anything else means a minus sign inside the constitutive law,
and that minus gets forgotten exactly once. Two consequences at the module
boundaries:

  * `boundaries.sigma_v_geostatic` / `sigma_h_geostatic` are
    COMPRESSION-POSITIVE Pa. `sigma0_star_from_geostatic` below is the only
    place in the project that flips them.

  * The pore term is a SUBTRACTION and an INCREMENT:

        sigma_total = sigma_0 + D:eps - ( chi.psi - chi_0.psi_0 )

    sigma_0 from the FE gravity solve is a TOTAL stress (wet density, no pore
    term), so the initial pore state is already inside it and adding the
    absolute chi.psi counts it twice. Both facts are pinned by
    tests/test_initial_equilibrium.py; before D-3.3.3 the residual at the
    exact initial state was 0.604 in Mk rather than zero.

WHY THE STRESS SCALING IS CLEAN
-------------------------------
sigma* = sigma/sig_ref, eps_phys = (U_ref/L_ref)*eps*, and
U_ref = sig_ref*L_ref/E_ref, so

    sigma* = (E/sig_ref)*(U_ref/L_ref)*Dhat(nu):eps* = (E/E_ref)*Dhat(nu):eps*

with no leftover group. That cancellation is what the U_ref definition is
FOR, and it deserves a sentence in the methods section: the displacement
scale was chosen so the constitutive law is scale-free and every Pi group is
pushed into the body-force and coupling terms, where they can be compared
against each other.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from src import materials as _m
from src.step21_geometry.boundaries import Z_WT
from src.coupling import bishop_chi, bulk_density_ratio
from src.derivatives import grad
from src.nondim import SCALES, Scales, mechanical_residual_nd, total_stress_nd

__all__ = [
    "E_hat",
    "psi0_star",
    "lame_hat",
    "psi_of",
    "uv_of",
    "strain_star",
    "stress_increment_star",
    "sigma0_star_from_geostatic",
    "chi_of",
    "theta_of",
    "mechanical_residual",
]


# ---------------------------------------------------------------------------
# 1. Field slicing
# ---------------------------------------------------------------------------
# `loss.py` already defines psi_of/uv_of because `residuals._psi_of` returns a
# bare Tensor unchanged and would treat u*, v* as extra psi columns. Importing
# them from `loss.py` would invert the dependency — loss.py will shortly need
# mechanics.py. Duplicated here deliberately and temporarily; Fix 4 moves both
# to a shared module and deletes these.
def psi_of(out) -> Tensor:
    """Pressure head column, (N,1). Accepts a (N,3) tensor or a 3-tuple."""
    if isinstance(out, (tuple, list)):
        return out[0]
    return out[:, 0:1]


def uv_of(out) -> tuple[Tensor, Tensor]:
    """(u*, v*), each (N,1). Note `loss.uv_of` returns a single (N,2) tensor;
    this returns the pair, because every caller here immediately unpacks."""
    if isinstance(out, (tuple, list)):
        return out[1], out[2]
    return out[:, 1:2], out[:, 2:3]


# ---------------------------------------------------------------------------
# 2. Elastic constants
# ---------------------------------------------------------------------------
def E_hat(mat, s: Scales = SCALES) -> float:
    """E / E_ref. `mat.E` comes from properties.py, which is the single source
    of truth. NOTE: `boundaries.E_RM` carries DIFFERENT values for Mk and Mk_d
    (4.78e8 / 8.65e7 against properties' 2.09e8 / 3.81e7). See Fix 3."""
    return float(mat.E) / s.E_ref


def lame_hat(mat, s: Scales = SCALES) -> tuple[float, float]:
    """(lambda_hat, 2*mu_hat), plane strain, dimensionless."""
    nu = float(mat.nu)
    Eh = E_hat(mat, s)
    lam = Eh * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    two_mu = Eh / (1.0 + nu)
    return lam, two_mu


# ---------------------------------------------------------------------------
# 3. Kinematics
# ---------------------------------------------------------------------------
def strain_star(u: Tensor, v: Tensor, x: Tensor, z: Tensor, *,
                physical: bool = False, s: Scales = SCALES):
    """Small-strain tensor from the dimensionless displacement field.

    Returns (eps_xx, eps_zz, eps_xz, eps_v). eps_xz is the TENSOR component,
    0.5*(du/dz + dv/dx), not engineering gamma_xz. The constitutive law uses
    2*mu*eps_xz, which gives the same tau either way — but a reader checking
    against a gamma-based textbook will see a factor of two, and should find
    this sentence before they find a bug that is not there.

    physical=True multiplies by U_ref/L_ref to give true strain. Kozeny-Carman
    needs that; the constitutive law does not. The factor is 0.01 and the
    error is silent.
    """
    eps_xx = grad(u, x)
    eps_zz = grad(v, z)
    eps_xz = 0.5 * (grad(u, z) + grad(v, x))
    eps_v = eps_xx + eps_zz

    if physical:
        k = s.U_ref / s.L_ref
        return k * eps_xx, k * eps_zz, k * eps_xz, k * eps_v
    return eps_xx, eps_zz, eps_xz, eps_v


# ---------------------------------------------------------------------------
# 4. Constitutive
# ---------------------------------------------------------------------------
def stress_increment_star(eps_xx, eps_zz, eps_xz, mat, s: Scales = SCALES):
    """Plane-strain Hooke, tension positive. Returns (sxx*, szz*, sxz*).

        sigma_xx = lam*eps_v + 2mu*eps_xx
        sigma_zz = lam*eps_v + 2mu*eps_zz
        sigma_xz =             2mu*eps_xz

    Takes eps* (NOT physical strain) — the scale group cancels, see header.
    """
    eps_v = eps_xx + eps_zz
    lam, two_mu = lame_hat(mat, s)
    return (lam * eps_v + two_mu * eps_xx,
            lam * eps_v + two_mu * eps_zz,
            two_mu * eps_xz)


def sigma0_star_from_geostatic(sig_v_pa, sig_h_pa, sig_xz_pa=None,
                               s: Scales = SCALES):
    """COMPRESSION-POSITIVE geostatic column (Pa) -> tension-positive sigma0*.

    The only sign flip in the project. Call it at SAMPLE time and carry the
    result on the Collocation, not inside the residual: the geometry
    round-trip x -> x/L_ref -> x moves boundary points by ~1e-16, enough for
    `material_tag` to return "outside" on curved segments.
    """
    sxx = -torch.as_tensor(sig_h_pa, dtype=torch.float64).reshape(-1, 1) / s.sig_ref
    szz = -torch.as_tensor(sig_v_pa, dtype=torch.float64).reshape(-1, 1) / s.sig_ref
    if sig_xz_pa is None:
        sxz = torch.zeros_like(szz)
    else:
        sxz = -torch.as_tensor(sig_xz_pa, dtype=torch.float64).reshape(-1, 1) / s.sig_ref
    return sxx, szz, sxz


# ---------------------------------------------------------------------------
# 5. Constitutive lookups — psi* in, physical units handled here
# ---------------------------------------------------------------------------
def psi0_star(z_star: Tensor, s: Scales = SCALES) -> Tensor:
    """Initial suction head, dimensionless, DIFFERENTIABLE in z*.

        psi_0 = Z_WT - z          (initial.psi_initial, hydrostatic)

    Restated analytically here rather than looked up, because the pore term
    enters the residual through `div`: what the equilibrium equation sees is
    grad(chi_0 psi_0), and a cached or detached psi_0 has no gradient. Doing
    that produces a residual identical to the absolute-pore-term bug it was
    meant to fix, which is how one debugging round was lost.

    `test_psi0_star_matches_the_initial_condition_module` binds this to
    `initial.psi_initial` so the two cannot drift. Z_WT = 197 < Z_BASE = 200,
    so psi_0 < 0 everywhere -- Section 5 is entirely unsaturated at t = 0 and
    chi_0 is well below 1, which is why the increment matters at all.
    """
    return (Z_WT - z_star * s.L_ref) / s.H_ref


def chi_of(psi_star: Tensor, mat, s: Scales = SCALES) -> Tensor:
    """Bishop chi = Se(psi). psi* is multiplied back up by H_ref because the
    van Genuchten alpha is in 1/m — the same single-line convention as
    `residuals.swcc_from_material`."""
    return bishop_chi(_m.Se(psi_star * s.H_ref, mat))


def theta_of(psi_star: Tensor, mat, s: Scales = SCALES) -> Tensor:
    """Volumetric water content theta(psi). NOT Se."""
    return _m.theta(psi_star * s.H_ref, mat)


# ---------------------------------------------------------------------------
# 6. The residual
# ---------------------------------------------------------------------------
def mechanical_residual(fields, x: Tensor, z: Tensor, t: Tensor, mat, *,
                        sigma0_star=None,
                        div_sigma0_star=None,
                        rho0_ratio: Tensor | float = 1.0,
                        bishop: bool = True,
                        body_force: bool = True,
                        s: Scales = SCALES,
                        return_terms: bool = False):
    """Quasi-static equilibrium residual, per component, dimensionless.

        div* sigma_eff*  +  Pi_M_body * (rho_b/rho_b_ref) * e_z  =  0

    with sigma_eff* = sigma0* + dsigma*(u*,v*) + Bishop pore term on the
    NORMAL components only (the shear component takes no pore term).

    THE sigma0 QUESTION, stated plainly
    -----------------------------------
    In incremental form div sigma_total = div sigma0 + div dsigma. If sigma0
    is genuinely EQUILIBRATED then div sigma0 + Pi_M_body*rho0_ratio*e_z = 0
    pointwise, those terms cancel, and the residual collapses to

        div* dsigma*  +  Pi_M_body * (rho_b - rho_0)/rho_b_ref * e_z  =  0

    needing no spatial derivative of sigma0 at all. That is the default path
    (`div_sigma0_star=None`), and it is cheap.

    It is also a claim the raw K0 column does not satisfy — the handoff
    records a 0.20-0.26 rho*g horizontal imbalance on the 31.06 deg face. Two
    honest responses:

      (a) equilibrate sigma0 first, verify the imbalance is below a number
          written into DECISIONS.md, then use the default path;
      (b) pass `div_sigma0_star` and carry the imbalance, which requires
          sigma0 differentiable in (x,z) — a small pretrained sigma0 net, or
          an interpolant with an analytic derivative.

    Do not take the default without doing (a). It will train, the loss will
    fall, and it will converge to an equilibrium in which a ~25% body-force
    imbalance is physics.

    Parameters
    ----------
    bishop : False drops the pore term. The one-way mechanical ablation.
    body_force : False drops gravity. Diagnostic only — use it to check that
        a term you expect to dominate actually does.
    """
    out = fields(x, z, t)
    psi = psi_of(out)
    u, v = uv_of(out)

    eps_xx, eps_zz, eps_xz, _ = strain_star(u, v, x, z, s=s)
    dsxx, dszz, dsxz = stress_increment_star(eps_xx, eps_zz, eps_xz, mat, s)

    if sigma0_star is not None:
        s0xx, s0zz, s0xz = sigma0_star
        sxx, szz, sxz = dsxx + s0xx, dszz + s0zz, dsxz + s0xz
    else:
        sxx, szz, sxz = dsxx, dszz, dsxz

    if bishop:
        # Increment, not absolute: sigma_0 already carries chi_0*psi_0.
        chi = chi_of(psi, mat, s)
        psi0 = psi0_star(z, s)
        pore = chi * psi - chi_of(psi0, mat, s) * psi0
        sxx = total_stress_nd(sxx, pore, torch.ones_like(pore), s)
        szz = total_stress_nd(szz, pore, torch.ones_like(pore), s)
    else:
        chi = None

    div_x = grad(sxx, x) + grad(sxz, z)
    div_z = grad(sxz, x) + grad(szz, z)

    if body_force:
        rho_ratio = (bulk_density_ratio(theta_of(psi, mat, s),
                                        float(mat.rho_dry), s.rho_b_ref))
    else:
        rho_ratio = torch.zeros_like(psi)

    if div_sigma0_star is not None:
        div_x = div_x + div_sigma0_star[0]
        div_z = div_z + div_sigma0_star[1]
    elif body_force:
        # sigma0 assumed equilibrated: only the CHANGE in body force survives.
        rho_ratio = rho_ratio - rho0_ratio

    res_x, res_z = mechanical_residual_nd((div_x, div_z), rho_ratio,
                                          e_z=(0.0, -1.0), s=s)

    if not return_terms:
        return res_x, res_z

    terms = {
        "sigma_xx": sxx, "sigma_zz": szz, "sigma_xz": sxz,
        "eps_xx": eps_xx, "eps_zz": eps_zz, "eps_xz": eps_xz,
        "div_x": div_x, "div_z": div_z,
        "rho_ratio": rho_ratio,
        "chi": chi,
        "bishop_on": bishop,
        "sigma0_differentiated": div_sigma0_star is not None,
    }
    return (res_x, res_z), terms