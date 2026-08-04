"""
nondim.py  —  Step 1.6: Non-dimensionalisation of the coupled HM PINN PDEs
============================================================================
Işıkdere lignite open-pit mine  (Ulusay et al., 2014, Engineering Geology)

Central source of truth for every characteristic scale and every
dimensionless group used in the PINN loss. Import this module everywhere so
that the network, the collocation sampler, and the residual functions all
agree on one consistent scaling.

Coordinate / field mapping used throughout the project
------------------------------------------------------
    x, z   [m]      spatial coordinates      -> x*, z*   (divide by L_ref)
    t      [s]      time                     -> t*       (divide by T_ref)
    psi    [m]      pressure head (Richards) -> psi*      (divide by H_ref)
    u, w   [m]      displacements            -> u*, w*    (divide by U_ref)
    sigma  [Pa]     stress                   -> sigma*    (divide by sig_ref)

All residuals are algebraically rearranged so the dominant term carries a
coefficient of 1.0 and the remaining terms carry the dimensionless groups
defined below. See `report_scales()` for the numerical values.

Author: <thesis>  |  Phase 1, Step 1.6
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import math


# ---------------------------------------------------------------------------
# 1. Physical constants
# ---------------------------------------------------------------------------
G_ACCEL = 9.81          # m/s^2, consistent with project roadmap
RHO_W   = 1000.0        # kg/m^3, density of water


# ---------------------------------------------------------------------------
# 2. Characteristic scales  (chosen from the Işıkdere domain geometry)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Scales:
    """Characteristic reference values. Each has a physical justification so
    the resulting dimensionless groups remain interpretable for reviewers."""

    # --- primary scales ---
    L_ref:   float = 170.0        # m     pit depth (characteristic length)
    T_ref:   float = 86_400.0     # s     one day (transient seepage timescale)
    H_ref:   float = 30.0         # m     max piezometric head above coal seam
    sig_ref: float = 1.0e6        # Pa    1 MPa (typical overburden stress)
    K_ref:   float = 1.0e-6       # m/s   representative sat. conductivity
    E_ref:   float = 1.0e9        # Pa    representative rock-mass modulus (~1 GPa)

    # --- representative bulk properties (for body-force group only) ---
    rho_b_ref: float = 1800.0     # kg/m^3 saturated bulk density (mid-range)

    # --- derived scales (computed in __post_init__ via object.__setattr__) ---
    U_ref: float = field(default=0.0)   # m  displacement scale = sig_ref*L_ref/E_ref

    def __post_init__(self):
        object.__setattr__(self, "U_ref", self.sig_ref * self.L_ref / self.E_ref)

    # -- dimensionless groups appearing in the residuals ------------------
    @property
    def Pi_R_diff(self) -> float:
        """Richards diffusion group:  T_ref * K_ref / L_ref^2 .
        Multiplies the spatial-diffusion term when time is the leading term."""
        return self.T_ref * self.K_ref * self.H_ref / self.L_ref**2

    @property
    def Pi_R_grav(self) -> float:
        """Richards gravity/drainage group:  K_ref * T_ref / L_ref ."""
        return self.K_ref * self.T_ref / self.L_ref

    @property
    def Pi_R_hz(self) -> float:
        """Head-vs-length ratio H_ref / L_ref (appears in the (psi + z) term)."""
        return self.H_ref / self.L_ref

    @property
    def Pi_M_body(self) -> float:
        """Mechanical body-force group:  rho_b_ref * g * L_ref / sig_ref .
        Order 1 by construction -> body force and stress-divergence balance."""
        return self.rho_b_ref * G_ACCEL * self.L_ref / self.sig_ref

    @property
    def Pi_M_couple(self) -> float:
        """Pore-pressure -> effective-stress coupling group:
        rho_w * g * H_ref / sig_ref .  Maps psi* into the effective-stress term."""
        return RHO_W * G_ACCEL * self.H_ref / self.sig_ref


SCALES = Scales()   # module-level singleton; import this everywhere


# ---------------------------------------------------------------------------
# 3. Scaling helpers (raw <-> dimensionless)
# ---------------------------------------------------------------------------
def nd_x(x, s: Scales = SCALES):     return x / s.L_ref
def nd_t(t, s: Scales = SCALES):     return t / s.T_ref
def nd_psi(psi, s: Scales = SCALES): return psi / s.H_ref
def nd_u(u, s: Scales = SCALES):     return u / s.U_ref
def nd_sig(sig, s: Scales = SCALES): return sig / s.sig_ref

def redim_x(xs, s: Scales = SCALES):     return xs * s.L_ref
def redim_t(ts, s: Scales = SCALES):     return ts * s.T_ref
def redim_psi(psis, s: Scales = SCALES): return psis * s.H_ref
def redim_u(us, s: Scales = SCALES):     return us * s.U_ref
def redim_sig(sigs, s: Scales = SCALES): return sigs * s.sig_ref


# ---------------------------------------------------------------------------
# 4. Dimensionless PDE residuals (framework-agnostic; pass in derivatives)
# ---------------------------------------------------------------------------
# These functions take ALREADY-DIFFERENTIATED dimensionless quantities. In the
# PINN they are called with tensors whose derivatives come from autograd on the
# dimensionless coordinates (x*, z*, t*). They return residuals at O(1).

def richards_residual_nd(C_star, dpsi_dt_star,
                         div_Kgrad_star, dK_dz_star,
                         s: Scales = SCALES):
    """Dimensionless mixed-form Richards residual.

        C*(psi*) * dpsi*/dt*
          - Pi_R_diff * div_x*[ K*(psi*) grad_x* psi* ]
          - Pi_R_grav * dK*/dz*                              = 0

    Parameters are the dimensionless building blocks:
      C_star          : specific moisture capacity C*(psi*)  (dimensionless)
      dpsi_dt_star    : d psi* / d t*
      div_Kgrad_star  : div_x*[ K*(psi*) grad_x* psi* ]      (the diffusion term)
      dK_dz_star      : d K* / d z*   (gravity drainage term)
    """
    return (C_star * dpsi_dt_star
            - s.Pi_R_diff * div_Kgrad_star
            - s.Pi_R_grav * dK_dz_star)


def mechanical_residual_nd(div_sigma_eff_star, rho_ratio, e_z=(0.0, -1.0),
                           s: Scales = SCALES):
    """Dimensionless quasi-static equilibrium residual (per component).

        div* sigma*_eff  +  Pi_M_body * (rho_b/rho_b_ref) * e_z  = 0

    div_sigma_eff_star : divergence of dimensionless effective stress (vector)
    rho_ratio          : rho_b / rho_b_ref at the collocation point
    e_z                : unit gravity direction (x, z); default gravity in -z
    Returns a 2-tuple (res_x, res_z).
    """
    bx, bz = e_z
    body = s.Pi_M_body * rho_ratio
    res_x = div_sigma_eff_star[0] + body * bx
    res_z = div_sigma_eff_star[1] + body * bz
    return res_x, res_z


def effective_stress_nd(sigma_star, psi_star, chi, s: Scales = SCALES):
    """Dimensionless Bishop effective stress (isotropic pore term).

        sigma*_eff = sigma* - Pi_M_couple * chi * psi*   (on normal components)

    chi ~ effective saturation Theta (Bishop parameter). psi* < 0 in suction.
    """
    return sigma_star - s.Pi_M_couple * chi * psi_star


# ---------------------------------------------------------------------------
# 5. Reporting
# ---------------------------------------------------------------------------
def report_scales(s: Scales = SCALES) -> str:
    lines = []
    lines.append("CHARACTERISTIC SCALES (Işıkdere)")
    lines.append(f"  L_ref   = {s.L_ref:>12.4g} m     (pit depth)")
    lines.append(f"  T_ref   = {s.T_ref:>12.4g} s     (1 day)")
    lines.append(f"  H_ref   = {s.H_ref:>12.4g} m     (head above coal seam)")
    lines.append(f"  sig_ref = {s.sig_ref:>12.4g} Pa    (1 MPa overburden)")
    lines.append(f"  K_ref   = {s.K_ref:>12.4g} m/s   (sat. conductivity)")
    lines.append(f"  E_ref   = {s.E_ref:>12.4g} Pa    (rock-mass modulus)")
    lines.append(f"  U_ref   = {s.U_ref:>12.4g} m     (= sig_ref*L_ref/E_ref)")
    lines.append("")
    lines.append("DIMENSIONLESS GROUPS")
    lines.append(f"  Pi_R_diff  = T*K/L^2      = {s.Pi_R_diff:.4e}")
    lines.append(f"  Pi_R_grav  = K*T/L        = {s.Pi_R_grav:.4e}")
    lines.append(f"  Pi_R_hz    = H/L          = {s.Pi_R_hz:.4e}")
    lines.append(f"  Pi_M_body  = rho*g*L/sig  = {s.Pi_M_body:.4e}")
    lines.append(f"  Pi_M_couple= rho_w*g*H/sig= {s.Pi_M_couple:.4e}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report_scales())
