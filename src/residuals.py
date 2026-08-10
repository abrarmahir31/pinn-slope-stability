from __future__ import annotations
"""
src/residuals.py — PDE residuals, autograd layer only.
======================================================

DESIGN RULE (settled 4 Aug): the residual *algebra* lives in `src/nondim.py`
(`richards_residual_nd`, `mechanical_residual_nd`, `effective_stress_nd`),
where the Pi groups are applied. This module's only job is to produce the
already-differentiated dimensionless building blocks those functions expect,
and hand them over. Nothing here re-derives an equation.

Everything is dimensionless. Coordinates are (x*, z*, t*) = (x/L_ref,
z/L_ref, t/T_ref) and the field is psi* = psi/H_ref. The one place physical
units appear is inside `swcc_from_material`, which multiplies psi* back up by
H_ref before calling `materials.K` / `materials.C`, because the van Genuchten
alpha is in 1/m. That conversion is a single line, on purpose.

`fields` is a CALLABLE, not a `PINN`. Anything with the signature
`fields(x, z, t) -> psi*` (or `-> (psi*, u*, v*)`, or a dict) works, so an
analytic field can be pushed through exactly the same code path as the network.
That is what makes the hydrostatic test in tests/test_richards_hydrostatic.py
a test of the residual rather than a test of the network.
"""
from src.nondim import SCALES, Scales, richards_residual_nd, darcy_flux_nd

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence
import torch 
from torch import Tensor 

from src.derivatives import divergence, grad
from src.nondim import SCALES, Scales, richards_residual_nd

__all__ = [
    "SWCC",
    "swcc_from_material",
    "richards_residual",
    "richards_residual_expanded",
    "darcy_flux",
    "interface_flux_jump",
]


# ---------------------------------------------------------------------------
# 1. What this module needs from the constitutive layer
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SWCC:
    """The whole constitutive contract, in dimensionless terms.

    C_star(psi*) -> C * H_ref            [-]   specific moisture capacity
    K_star(psi*) -> K(psi) / K_ref       [-]   unsaturated conductivity

    Both must be torch-differentiable in psi*. Deliberately minimal: the
    residual does not need theta, Se or chi, so it does not ask for them.
    Keeping the surface this small is what lets the tests substitute an
    analytic SWCC without importing materials.py at all.
    """
    C_star: Callable[[Tensor], Tensor]
    K_star: Callable[[Tensor], Tensor]
    name: str = "unnamed"


def swcc_from_material(mat, s: Scales = SCALES) -> SWCC:
    """Adapt a `materials.Material` to the dimensionless SWCC contract.

    Two conventions are asserted here, and this is the ONLY place either is
    applied. If `materials.py` ever changes, this function changes and nothing
    else does.

      * `materials.C(psi_phys, mat)` already returns the DIMENSIONLESS
        capacity (the settled `C * H_ref` convention), so it is passed
        through untouched.
      * `materials.K(psi_phys, mat)` returns PHYSICAL m/s, so it is divided
        by K_ref here.
    """
    from src import materials as _m  # local import: keeps tests free of it

    def C_star(psi_star: Tensor) -> Tensor:
        return _m.C_star(psi_star * s.H_ref, mat, s=s)      # 8 spaces

    def K_star(psi_star: Tensor) -> Tensor:
        return _m.K(psi_star * s.H_ref, mat) / s.K_ref      # 8 spaces
    
    
    return SWCC(C_star=C_star, K_star=K_star,
                name=getattr(mat, "name", "material"))


def _as_swcc(mat, s: Scales) -> SWCC:
    return mat if isinstance(mat, SWCC) else swcc_from_material(mat, s)


def _psi_of(out):
    """Accept psi, (psi, u, v), [psi, ...] or {'psi': ...} from `fields`."""
    if isinstance(out, Tensor):
        return out
    if isinstance(out, Mapping):
        return out["psi"]
    if isinstance(out, Sequence):
        return out[0]
    raise TypeError(f"fields() returned {type(out)!r}; expected Tensor, "
                    "tuple/list (psi first) or mapping with key 'psi'")


# ---------------------------------------------------------------------------
# 2. The Richards residual
# ---------------------------------------------------------------------------
def richards_residual(fields, x: Tensor, z: Tensor, t: Tensor, mat,
                      s: Scales = SCALES, normalise: str = "none",
                      return_terms: bool = False):
    """Dimensionless mixed-form Richards residual at the given points.

        R = C* dpsi*/dt*  -  Pi_R_diff div*[K* grad* psi*]  -  Pi_R_grav dK*/dz*

    Parameters
    ----------
    fields : callable (x, z, t) -> psi* (or a tuple/dict containing it)
    x, z, t : leaf tensors, requires_grad=True, same shape (N, 1)
    mat : a `materials.Material` OR an `SWCC` (tests pass the latter)
    normalise : how to rescale the whole equation. The Step 1.6 scaling
        decision is still open, so it is a flag, not a hard-coded constant:
          "none"    — as written above; the diffusion term is O(1e-6)
          "gravity" — divide through by Pi_R_grav; gravity term becomes O(1)
                      and diffusion O(H_ref/L_ref) = 0.176
          "storage" — divide by the pointwise |C*|, if storage dominates
        Dividing by a positive constant cannot change where R = 0, so every
        test below passes under all three. It only changes conditioning.
    return_terms : also return the three terms separately, undivided. Use this
        to produce the Pi-group magnitude table for the methods section —
        it reports what the optimiser actually sees.

    Notes
    -----
    The diffusion term is built as div[K grad psi] on the ASSEMBLED flux
    rather than expanded into K laplacian(psi) + dK/dpsi |grad psi|^2. Both
    are correct; autograd derives the chain rule for the first, which removes
    the most error-prone line in the file. `richards_residual_expanded`
    implements the second, and the two are compared in the tests.
    """
    psi = _psi_of(fields(x, z, t))
    swcc = _as_swcc(mat, s)

    K = swcc.K_star(psi)
    C = swcc.C_star(psi)

    dpsi_dt = grad(psi, t)
    dpsi_dx = grad(psi, x)
    dpsi_dz = grad(psi, z)

    # div*[ K* grad* psi* ] — chain rule through K*(psi*(x,z)) done by autograd
    div_Kgrad = divergence([K * dpsi_dx, K * dpsi_dz], [x, z])

    # dK*/dz* — the gravity drainage term. Also a chain rule through psi*.
    dK_dz = grad(K, z)

    R = richards_residual_nd(C, dpsi_dt, div_Kgrad, dK_dz, s)

    if normalise == "gravity":
        R = R / s.Pi_R_grav
    elif normalise == "storage":
        R = R / (C.abs() + 1e-12)
    elif normalise != "none":
        raise ValueError(f"normalise={normalise!r} not in "
                         "{'none', 'gravity', 'storage'}")

    if not return_terms:
        return R

    terms = {
        "storage": C * dpsi_dt,
        "diffusion": -s.Pi_R_diff * div_Kgrad,
        "gravity": -s.Pi_R_grav * dK_dz,
        "K_star": K,
        "C_star": C,
        "dpsi_dz": dpsi_dz,
    }
    return R, terms


def richards_residual_expanded(fields, x: Tensor, z: Tensor, t: Tensor, mat,
                               s: Scales = SCALES):
    """Same residual, product rule written out. Cross-check only.

        div[K grad psi] = K (d2psi/dx2 + d2psi/dz2)
                          + dK/dpsi (dpsi/dx^2 + dpsi/dz^2)

    This is the form in the roadmap. It is kept because agreement with
    `richards_residual` to machine precision is a direct test of dK/dpsi, and
    because a reviewer asking "did you expand it correctly?" gets an answer.
    Do not use it in training — it is one extra graph traversal for nothing.
    """
    psi = _psi_of(fields(x, z, t))
    swcc = _as_swcc(mat, s)

    K = swcc.K_star(psi)
    C = swcc.C_star(psi)

    dpsi_dt = grad(psi, t)
    dpsi_dx = grad(psi, x)
    dpsi_dz = grad(psi, z)
    d2psi_dx2 = grad(dpsi_dx, x)
    d2psi_dz2 = grad(dpsi_dz, z)

    dK_dpsi = grad(K, psi)          # psi is a non-leaf graph node; this is fine

    div_Kgrad = (K * (d2psi_dx2 + d2psi_dz2)
                 + dK_dpsi * (dpsi_dx ** 2 + dpsi_dz ** 2))
    dK_dz = dK_dpsi * dpsi_dz

    return richards_residual_nd(C, dpsi_dt, div_Kgrad, dK_dz, s)

def darcy_flux(fields, x, z, t, mat, s: Scales = SCALES):
    """Dimensionless Darcy flux (q*_x, q*_z) at the given points.

    Same convention as richards_residual: `fields` is a CALLABLE, not a PINN,
    so analytic test fields go through the identical path.

    `mat` selects which stratum's SWCC is used. At an interface that is the
    whole point -- the two sides disagree.
    """
    psi = _psi_of(fields(x, z, t))
    swcc = _as_swcc(mat, s)
    K = swcc.K_star(psi)
    return darcy_flux_nd(K, grad(psi, x), grad(psi, z), s)


def interface_flux_jump(fields, x, z, t, mat_a, mat_b, nx, nz,
                        s: Scales = SCALES):
    """Normal-flux discontinuity across a material contact: [[q*.n]].

    A single network gives psi continuity across the contact for free -- there
    is only one psi field. What it does NOT give is mass conservation ACROSS
    the contact. K_s spans ~3 orders between Mk, Mk_d and Tm, so
    q = -K(psi)(grad psi + e_z) is discontinuous for any psi the optimiser can
    reach unless grad psi compensates by exactly the K ratio.

    Nothing currently makes it. The cheapest solution available to the network
    is to let mass leak across the boundary, and the interior residual cannot
    see it: both sides are individually satisfied while the pair is not. The
    symptom in a trained model is a wetting front that stalls or accelerates
    at a stratum contact for no physical reason.

    Both fluxes are evaluated at the SAME (x, z, t) with the SAME psi, so the
    jump is purely the constitutive difference. (nx, nz) is the unit normal to
    the contact; its sign is irrelevant because the loss squares this.
    """
    qa_x, qa_z = darcy_flux(fields, x, z, t, mat_a, s)
    qb_x, qb_z = darcy_flux(fields, x, z, t, mat_b, s)
    return (qa_x - qb_x) * nx + (qa_z - qb_z) * nz


# ---------------------------------------------------------------------------
# 3. Mechanical residual — BLOCKED
# ---------------------------------------------------------------------------
def mechanical_residual(*args, **kwargs):
    """Not yet written. Blocked on the gravity warm-up (Step 3.4).

    The incremental convention (sigma_total = sigma_0(lookup) + dsigma(u,v))
    means this function needs the EQUILIBRATED sigma_0, which does not exist
    until the warm-up has been run. Writing it against the raw K0 column would
    bake in the 0.20-0.26 rho*g horizontal imbalance on the 31.06 deg face.
    """
    raise NotImplementedError(
        "mechanical_residual is blocked on the gravity warm-up producing an "
        "equilibrated sigma_0 lookup. See docs/open_items.md."
    )
