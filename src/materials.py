"""Section 5 material set — thin adapter over properties.py and vg.py.

properties.py is the SINGLE SOURCE OF TRUTH for parameters (see its docstring).
vg.py is the single implementation of the constitutive model. This module adds
only what Phase 3 needs on top: a per-unit record, and the C* scaling that the
Richards residual expects.

Sign convention: psi < 0 unsaturated (suction), psi >= 0 saturated.
"""
from dataclasses import dataclass

import torch

from src import properties as P
from src import vg
from src.nondim import SCALES


@dataclass(frozen=True)
class Material:
    tag: str
    K_s: float        # m/s
    E: float          # Pa
    nu: float
    K0: float
    n0: float
    theta_s: float    # = 0.9 * n0 (porosity-tie rule, see properties.py)
    theta_r: float
    alpha: float      # 1/m
    n: float
    m: float
    rho_dry: float    # kg/m3
    rho_sat: float    # kg/m3
    sigma_ci: float   # Pa
    m_b: float
    s: float
    a: float
    gsi: float

    @property
    def dtheta(self) -> float:
        return self.theta_s - self.theta_r


def load_materials() -> dict:
    return {
        u: Material(
            tag=u,
            K_s=P.K_S[u], E=P.E[u], nu=P.NU[u], K0=P.K0[u],
            n0=P.N0[u], theta_s=P.THETA_S[u], theta_r=P.THETA_R[u],
            alpha=P.ALPHA[u], n=P.VG_N[u], m=P.VG_M[u],
            rho_dry=P.RHO_DRY[u], rho_sat=P.RHO_SAT[u],
            sigma_ci=P.SIG_CI[u], m_b=P.M_B[u], s=P.HB_S[u],
            a=P.HB_A[u], gsi=P.GSI[u],
        )
        for u in P.UNITS
    }


# --- constitutive model: delegate to vg.py, never reimplement ---

def theta(psi, mat):
    """Volumetric water content [-]."""
    return vg.theta(psi, mat.theta_r, mat.theta_s, mat.alpha, mat.n)


def Se(psi, mat):
    """Effective saturation [-]; also the Bishop chi in Step 3.3."""
    return vg.Se(psi, mat.alpha, mat.n)


def K(psi, mat):
    """Unsaturated hydraulic conductivity [m/s]. vg.K_r is RELATIVE."""
    return mat.K_s * vg.K_r(psi, mat.alpha, mat.n)


def C_star(psi, mat, s=None):
    """Dimensionless specific moisture capacity for the Richards residual.

    vg.C returns dtheta/dpsi in 1/m; H_ref makes it dimensionless and
    dtheta_ref applies the storage rescaling (see nondim.Pi_R_diff).
    """
    s = SCALES if s is None else s
    return vg.C(psi, mat.theta_r, mat.theta_s, mat.alpha, mat.n) \
        * s.H_ref / s.dtheta_ref
