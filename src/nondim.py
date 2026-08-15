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

from src import properties as P


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

    # H_ref: hydrostatic head range over the Section 5 domain (psi_0 = Z_WT - z,
    # spanning ~0-161 m). REVISED from 30.0, which was the Section 7 confined
    # head above the coal seam -- a unit that does not exist in Section 5. That
    # value predated the Day-13 stratigraphic correction. With H_ref ~ L_ref the
    # ratio Pi_R_grav / Pi_R_diff = L_ref / H_ref -> ~1, so both Richards terms
    # sit at O(1) with T_ref = 1 day retained and no division-through needed.
    H_ref:   float = 165.0        # m     hydrostatic head range, Section 5

    sig_ref: float = 1.0e6        # Pa    1 MPa (typical overburden stress)
    K_ref:   float = 1.0e-6       # m/s   representative sat. conductivity
    E_ref: float = 1.0e8          # Pa    marl-scale reference; Tm (4.264e9)
                                  #       excluded as rigid basement. Adopted
                                  #       marl moduli 3.81e7-2.09e8. D-S.1.
                                  
    # --- representative bulk properties (for body-force group only) ---
    rho_b_ref: float = 1800.0     # kg/m^3, arbitrary O(1) reference (Mk sat = 1946)

    # dtheta_ref: theta_s - theta_r for Mk, derived rather than hard-coded so it
    # cannot drift from the constitutive layer. properties.py carries the
    # literature theta_s = 0.38 for Mk (NOT the porosity-tie 0.9*n0 = 0.384).
    # Sits in the denominator of BOTH Richards groups, so it is not cosmetic.
    dtheta_ref: float = field(
        default_factory=lambda: P.THETA_S["Mk"] - P.THETA_R["Mk"]
    )

    # --- derived scales (computed in __post_init__ via object.__setattr__) ---
    U_ref: float = field(default=0.0)   # m  displacement scale = sig_ref*L_ref/E_ref

    def __post_init__(self):
        object.__setattr__(self, "U_ref", self.sig_ref * self.L_ref / self.E_ref)

    # -- dimensionless groups appearing in the residuals ------------------
    @property
    def Pi_R_diff(self) -> float:
        """Richards diffusion group:  T_ref * K_ref * H_ref / (L_ref^2 * dtheta_ref).
        Multiplies the spatial-diffusion term when time is the leading term."""
        return self.T_ref * self.K_ref * self.H_ref / self.L_ref**2 / self.dtheta_ref

    @property
    def Pi_R_grav(self) -> float:
        """Richards gravity/drainage group:  K_ref * T_ref / (L_ref * dtheta_ref)."""
        return self.K_ref * self.T_ref / self.L_ref / self.dtheta_ref

    @property
    def Pi_R_hz(self) -> float:
        """Head-vs-length ratio H_ref / L_ref (appears in the (psi + z) term)."""
        return self.H_ref / self.L_ref
    @property
    def Q_ref(self) -> float:
        """Flux scale, m/s.  q* = q / Q_ref = q * T_ref / (L_ref * dtheta_ref).

        Derived from the residual, not chosen. The gravity term of
        darcy_flux_nd is

            q*_z,grav = -Pi_R_grav * K*
                      = -(K_ref T_ref / (L_ref dtheta_ref)) (K / K_ref)
                      = -K * T_ref / (L_ref dtheta_ref)

        against a physical q_grav = -K, so q* = q * T_ref/(L_ref dtheta_ref).
        The diffusion term reduces to the same factor -- that agreement is the
        check that this is the flux scale implied by the non-dimensionalisation
        rather than an independent guess. tests/test_loss_bc.py asserts it.

        Note this is NOT K_ref. A prescribed flux divided by K_ref would be
        wrong by T_ref K_ref / (L_ref dtheta_ref) = Pi_R_grav, i.e. by orders
        of magnitude, and would train to a smooth, confidently wrong answer.
        """
        return self.L_ref * self.dtheta_ref / self.T_ref

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


def total_stress_nd(sigma_eff_star, pore_star, chi, s: Scales = SCALES):
    """Total stress from Bishop effective stress. TENSION POSITIVE (D-3.3.1).

        sigma*_total = sigma*_eff  -  Pi_M_couple * chi * pore*   (normal comps)

    DIRECTION MATTERS AND IS THE WHOLE POINT OF THE RENAME. Bishop in the
    usual compression-positive form is sigma'_c = sigma_c - chi p. Converting
    (sigma_t = -sigma_c) gives sigma'_t = sigma_t + chi p, i.e.

        effective = total + chi p        total = effective - chi p

    The constitutive law produces EFFECTIVE stress (sigma' = D:eps -- the
    skeleton carries it), while equilibrium acts on TOTAL stress
    (div sigma_total + rho_b g = 0). So the residual needs this direction, and
    the sign is a subtraction.

    D-3.3.1 briefly had this as an addition under the name
    `effective_stress_nd`. That formula is the correct total->effective
    converter; it was simply being called in the opposite direction, and
    nothing caught it because at the time the function had no call sites at
    all. The name now states the direction so the mistake is not re-typable.

    `pore_star` is an INCREMENT, chi*psi* - chi_0*psi_0*, not the absolute
    suction -- see mechanics.mechanical_residual. This function does not know
    that; it only applies the sign. Passing chi and pore separately is kept so
    the Bishop parameter stays visible at the call site.
    """
    return sigma_eff_star - s.Pi_M_couple * chi * pore_star


def effective_stress_nd(*args, **kwargs):
    """REMOVED in D-3.3.3. Use `total_stress_nd`.

    Not an alias: the sign flipped and the argument is now an increment, so a
    silent forward would produce plausible numbers that are wrong twice over.
    """
    raise NotImplementedError(
        "effective_stress_nd was replaced by total_stress_nd in D-3.3.3. The "
        "sign and the argument both changed; see DECISIONS.md D-3.3.3."
    )


def darcy_flux_nd(K_star, dpsi_dx_star, dpsi_dz_star, s: Scales = SCALES):
    """Dimensionless Darcy flux, consistent with richards_residual_nd.

        q* = -( Pi_R_diff * K* grad* psi*  +  Pi_R_grav * K* e_z )

    Built so that

        div* q*  +  C* dpsi*/dt*   ==   richards_residual_nd(...)

    exactly, term for term. That identity is the first assertion in
    tests/test_interface.py, and it is what stops the interface loss and the
    interior loss from enforcing two slightly different physics.

    The z-component groups as

        q*_z = -Pi_R_grav * K* * ( Pi_R_hz * dpsi*/dz*  +  1 )

    so a hydrostatic field (dpsi/dz = -1, i.e. dpsi*/dz* = -1/Pi_R_hz) gives
    exactly zero flux. This relies on Pi_R_diff / Pi_R_grav == Pi_R_hz. If
    H_ref is ever dropped from Pi_R_diff again, the identity breaks and the
    hydrostatic test fails loudly instead of quietly rescaling gravity.

    Returns (q*_x, q*_z).
    """
    qx = -s.Pi_R_diff * K_star * dpsi_dx_star
    qz = -(s.Pi_R_diff * K_star * dpsi_dz_star + s.Pi_R_grav * K_star)
    return qx, qz


# ---------------------------------------------------------------------------
# 5. Reporting
# ---------------------------------------------------------------------------
def report_scales(s: Scales = SCALES) -> str:
    lines = []
    lines.append("CHARACTERISTIC SCALES (Işıkdere, Section 5)")
    lines.append(f"  L_ref      = {s.L_ref:>12.4g} m     (pit depth)")
    lines.append(f"  T_ref      = {s.T_ref:>12.4g} s     (1 day)")
    lines.append(f"  H_ref      = {s.H_ref:>12.4g} m     (hydrostatic head range)")
    lines.append(f"  sig_ref    = {s.sig_ref:>12.4g} Pa    (1 MPa overburden)")
    lines.append(f"  K_ref      = {s.K_ref:>12.4g} m/s   (sat. conductivity)")
    lines.append(f"  E_ref      = {s.E_ref:>12.4g} Pa    (rock-mass modulus)")
    lines.append(f"  U_ref      = {s.U_ref:>12.4g} m     (= sig_ref*L_ref/E_ref)")
    lines.append(f"  rho_b_ref  = {s.rho_b_ref:>12.4g} kg/m3")
    lines.append(f"  dtheta_ref = {s.dtheta_ref:>12.4g} -     "
                 f"(Mk: theta_s {P.THETA_S['Mk']:.4g} - theta_r {P.THETA_R['Mk']:.4g})")
    lines.append("")
    lines.append("DIMENSIONLESS GROUPS")
    lines.append(f"  Pi_R_diff  = T*K*H/(L^2*dth) = {s.Pi_R_diff:.4e}")
    lines.append(f"  Pi_R_grav  = K*T/(L*dth)     = {s.Pi_R_grav:.4e}")
    lines.append(f"  ratio grav/diff = L/H        = {s.Pi_R_grav / s.Pi_R_diff:.4f}")
    lines.append(f"  Pi_R_hz    = H/L             = {s.Pi_R_hz:.4e}")
    lines.append(f"  Pi_M_body  = rho*g*L/sig     = {s.Pi_M_body:.4e}")
    lines.append(f"  Pi_M_couple= rho_w*g*H/sig   = {s.Pi_M_couple:.4e}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report_scales())