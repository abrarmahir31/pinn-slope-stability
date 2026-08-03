"""
vg.py -- van Genuchten-Mualem retention and conductivity. ONE definition.

theta(psi) feeds the mechanical body force and the geostatic column.
Se(psi)    feeds the Bishop effective-stress coupling (chi = Se).
C(psi)     feeds the Richards storage term.
K_r(psi)   feeds the Richards conductivity.

All four come from the same (theta_r, theta_s, alpha, n) set, so they must come
from the same code. Two copies drift apart; this is the copy.

Works on floats, numpy arrays and torch tensors. Autograd flows through
everything here -- psi is a network output in Phase 3, and the body force now
depends on it, so these are part of the HM coupling, not preprocessing.

Sign convention: psi < 0 is suction (unsaturated). psi >= 0 is saturated.

WARNING on the parameters (Mk, Mk_d): alpha = 0.5 1/m, n = 1.2 are
Carsel & Parrish silty-clay SOIL values used as an analogue for a rock mass,
tier B, no uncertainty band. They give K_r = 4.3e-7 at psi = -168 m, i.e.
K = 4.3e-16 m/s, against a rainfall flux of 5.56e-6 m/s. Matrix infiltration is
impossible by ~10 orders of magnitude. See the SWCC decision note before
interpreting any rainfall result.
"""

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:                                    # pragma: no cover
    _HAS_TORCH = False


def _lib(a):
    if _HAS_TORCH and isinstance(a, torch.Tensor):
        return torch
    return np


def _where(cond, a, b):
    if _HAS_TORCH and isinstance(cond, torch.Tensor):
        return torch.where(cond, a, b)
    return np.where(cond, a, b)


# --------------------------------------------------------------------------
def Se(psi, alpha, n, eps=1e-12):
    """Effective saturation, 0..1. Bishop's chi = Se."""
    L = _lib(psi)
    m = 1.0 - 1.0 / n
    ap = alpha * L.abs(psi) if L is torch else alpha * np.abs(psi)
    unsat = (1.0 + ap ** n) ** (-m)
    return _where(psi < 0.0, unsat, (1.0 if not hasattr(psi, "shape")
                                     else 0.0 * psi + 1.0))


def theta(psi, theta_r, theta_s, alpha, n):
    """Volumetric water content."""
    return theta_r + (theta_s - theta_r) * Se(psi, alpha, n)


def C(psi, theta_r, theta_s, alpha, n, eps=1e-12):
    """Specific moisture capacity, dtheta/dpsi. Zero when saturated."""
    L = _lib(psi)
    m = 1.0 - 1.0 / n
    ap = alpha * (L.abs(psi) if L is torch else np.abs(psi))
    ap = ap + eps
    num = alpha * m * n * (theta_s - theta_r) * ap ** (n - 1.0)
    den = (1.0 + ap ** n) ** (m + 1.0)
    val = num / den
    return _where(psi < 0.0, val, 0.0 * val)


def K_r(psi, alpha, n, eps=1e-12):
    """Mualem relative conductivity, 0..1.

    NOTE the stiffness: with n = 1.2 (m = 0.1667) this spans ~5 orders of
    magnitude over the domain's psi range. Expect the Richards residual to be
    numerically brutal; this is where per-term normalisation earns its keep.
    """
    s = Se(psi, alpha, n)
    s = _lib(s).clip(s, eps, 1.0) if _lib(s) is np else s.clamp(eps, 1.0)
    m = 1.0 - 1.0 / n
    inner = 1.0 - (1.0 - s ** (1.0 / m)) ** m
    return s ** 0.5 * inner ** 2


# --------------------------------------------------------------------------
def psi_initial(z, z_wt=197.0):
    """Step 2.3 seepage IC: suction hydrostatic above the karstic table.

    z_wt = 197.0 m a.s.l. is the UPPER bound of the karstic head range
    (113-197). Z_BASE = 200, so the whole domain is unsaturated and this is
    negative everywhere in it.
    """
    return -(z - z_wt)


def theta_initial(z, unit, props, z_wt=197.0):
    """theta at t = 0 for a unit, from psi_initial."""
    return theta(psi_initial(z, z_wt),
                 props["THETA_R"][unit], props["THETA_S"][unit],
                 props["ALPHA"][unit], props["VG_N"][unit])


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import properties as P
    print("psi(m)   " + "".join(f"{u:>22s}" for u in P.UNITS))
    for p in (-1, -10, -60, -168):
        row = f"{p:6d}   "
        for u in P.UNITS:
            th = theta(float(p), P.THETA_R[u], P.THETA_S[u],
                       P.ALPHA[u], P.VG_N[u])
            kr = K_r(float(p), P.ALPHA[u], P.VG_N[u])
            row += f"  th={th:.3f} Kr={kr:.2e}"
        print(row)
    print()
    for u in P.UNITS:
        th0 = theta(-100.0, P.THETA_R[u], P.THETA_S[u], P.ALPHA[u], P.VG_N[u])
        print(f"{u:5s} rho_dry={P.RHO_DRY[u]:7.1f}  "
              f"rho_b(psi=-100)={P.RHO_DRY[u] + th0 * 1000:7.1f}  "
              f"rho_sat={P.RHO_SAT[u]:7.1f}  "
              f"(+{100 * th0 * 1000 / P.RHO_DRY[u]:.1f}% wet)")
