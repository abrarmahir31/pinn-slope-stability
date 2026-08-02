"""
isikdere_phase1.py — single import point for the Phase-1 reference state
=======================================================================
Isikdere lignite open-pit mine · PINN coupled hydro-mechanical slope stability
Primary source: Ulusay, Ekmekci, Tuncay & Hasancebi (2014), Eng. Geol. 181, 261-280

Loads isikdere_phase1_dataset.json and exposes it as typed objects plus the
constitutive functions the PDE residuals need. Every constitutive function is
written with plain arithmetic and abs() only, so the same code runs on floats,
numpy arrays and torch tensors — autograd flows through unchanged.

    from isikdere_phase1 import DATA, STRATA, SCALES, vg_theta, vg_K, vg_C

    marl = STRATA["marl_sekkoy"]
    K = vg_K(psi, marl)              # works on a torch tensor
    r  = richards_residual_nd(...)   # dimensionless, O(1)

This file SUPERSEDES nondim.py, phase1_reference_state.json and
step_1_2_1_3_adopted_parameters.json. Ten source conflicts were resolved to
build it — read DATA["conflicts_resolved"] before you trust any number, and
call print_conflicts() to see them.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT = os.path.join(_HERE, "isikdere_phase1_dataset.json")


def load(path: str = _DEFAULT) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


DATA = load()

G = DATA["schema"]["constants"]["g_mps2"]            # 9.81 m/s^2
RHO_W = DATA["schema"]["constants"]["rho_w_kgpm3"]   # 1000 kg/m^3
GAMMA_W = DATA["schema"]["constants"]["gamma_w_Npm3"]  # 9810 N/m^3


# --------------------------------------------------------------------------
# Stratum
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Stratum:
    """One geological unit. All SI. Fields are None where the paper reports
    nothing and no defensible correlation exists (e.g. Hoek-Brown on a soil)."""
    key: str
    label: str
    kind: str                       # "rock" | "soil"
    role: str
    # density / phase
    rho_dry: float                  # kg/m^3
    rho_sat: float                  # kg/m^3
    n0: float                       # initial porosity
    Gs: float
    # hydraulic
    K_s: Optional[float]            # m/s
    K_s_range: Optional[list]
    # van Genuchten
    theta_r: float
    theta_s: float
    alpha: float                    # 1/m  (head form)
    alpha_Pa: float                 # 1/Pa (pressure form)
    n: float
    m: float
    vg_tier: str                    # "A" computed from site data | "B" literature analog
    # elastic
    E: float                        # Pa
    E_lo: float
    E_hi: float
    nu: float
    G_mod: float                    # shear modulus, Pa
    lam: float                      # Lame lambda, Pa
    K_bulk: float                   # Pa
    K0: float                       # nu/(1-nu), at-rest earth pressure coefficient
    # Hoek-Brown
    sigma_ci: Optional[float] = None
    m_i: Optional[float] = None
    GSI: Optional[float] = None
    D: Optional[float] = None
    m_b: Optional[float] = None
    s: Optional[float] = None
    a: Optional[float] = None
    raw: Optional[dict] = None      # full JSON record, including provenance notes

    @property
    def is_rock(self) -> bool:
        return self.kind == "rock"

    def rho_bulk(self, Sw):
        """Bulk density at water saturation Sw in [0,1]. This is the body-force
        density in the mechanical residual — using rho_dry or rho_sat as a
        constant is a modelling choice you would have to defend."""
        return self.rho_dry + self.n0 * Sw * RHO_W


def _mk(key: str, d: dict) -> Stratum:
    return Stratum(
        key=key, label=d["label"], kind=d["kind"], role=d["role"],
        rho_dry=d["rho_dry_kgpm3"], rho_sat=d["rho_sat_kgpm3"],
        n0=d["n0"], Gs=d["Gs"],
        K_s=d["K_s_mps"], K_s_range=d["K_s_range_mps"],
        theta_r=d["vg_theta_r"], theta_s=d["vg_theta_s"],
        alpha=d["vg_alpha_1pm"], alpha_Pa=d["vg_alpha_1pPa"],
        n=d["vg_n"], m=d["vg_m"], vg_tier=d["vg_tier"],
        E=d["E_Pa"], E_lo=d["E_lo_Pa"], E_hi=d["E_hi_Pa"], nu=d["nu"],
        G_mod=d["G_Pa"], lam=d["lambda_Pa"], K_bulk=d["K_bulk_Pa"], K0=d["K0_at_rest"],
        sigma_ci=d["sigma_ci_Pa"], m_i=d["m_i"], GSI=d["GSI"], D=d["D"],
        m_b=d["m_b"], s=d["s"], a=d["a"], raw=d,
    )


STRATA: Dict[str, Stratum] = {k: _mk(k, v) for k, v in DATA["strata"].items()}
ORDER = list(STRATA.keys())                       # stable index order for one-hot / masks
INDEX = {k: i for i, k in enumerate(ORDER)}


# --------------------------------------------------------------------------
# van Genuchten - Mualem
# --------------------------------------------------------------------------
# psi is PRESSURE HEAD in metres. psi < 0 is suction, psi >= 0 is saturated.
#
# The saturated branch is handled by clipping to the negative part rather than
# by an if-statement: _neg(psi) = (psi - |psi|)/2 = min(psi, 0). That uses only
# arithmetic and abs(), so it is one expression on floats, numpy arrays and
# torch tensors alike, and autograd flows through it. Below the water table
# (psi > 0) every function then returns its saturated value exactly:
# Se = 1, theta = theta_s, K = K_s, C = 0.

def _neg(x):
    """min(x, 0), branch-free."""
    return 0.5 * (x - abs(x))


def vg_Se(psi, st: Stratum):
    """Effective saturation Se in (0, 1]. Se = [1 + |alpha*psi|^n]^(-m), psi < 0;
    Se = 1 for psi >= 0."""
    ah = abs(st.alpha * _neg(psi))
    return (1.0 + ah ** st.n) ** (-st.m)


def vg_theta(psi, st: Stratum):
    """Volumetric water content."""
    return st.theta_r + (st.theta_s - st.theta_r) * vg_Se(psi, st)


def vg_K(psi, st: Stratum, K_s: Optional[float] = None):
    """Unsaturated hydraulic conductivity, Mualem (1976) with l = 0.5.

        K = K_s * Se^0.5 * [1 - (1 - Se^(1/m))^m]^2

    Pass K_s explicitly to feed the Kozeny-Carman-updated value from the
    coupled loop instead of the reference value."""
    Ks = st.K_s if K_s is None else K_s
    Se = vg_Se(psi, st)
    inner = 1.0 - (1.0 - Se ** (1.0 / st.m)) ** st.m
    return Ks * Se ** 0.5 * inner ** 2


def vg_C(psi, st: Stratum):
    """Specific moisture capacity C = dtheta/dpsi [1/m], analytic derivative.

        C = alpha*n*m*(theta_s-theta_r)*|alpha*psi|^(n-1) * [1+|alpha*psi|^n]^(-m-1)

    Written from the closed-form derivative rather than Se^(1/m) algebra so it
    stays finite as psi -> 0 (where Se -> 1 and the Se^(1/m) form goes 0/0).
    C = 0 for psi >= 0: a saturated element stores no water by desaturation.
    The kink at psi = 0 is physical and standard; if your optimiser struggles
    with it, smooth it there rather than removing the branch."""
    ah = abs(st.alpha * _neg(psi))
    return (st.alpha * st.n * st.m * (st.theta_s - st.theta_r)
            * ah ** (st.n - 1.0) * (1.0 + ah ** st.n) ** (-st.m - 1.0))


def water_saturation(psi, st: Stratum):
    """Sw = theta/n0, the quantity the bulk-density body force needs."""
    return vg_theta(psi, st) / st.n0


# --------------------------------------------------------------------------
# Coupling: porosity update and Kozeny-Carman
# --------------------------------------------------------------------------
def porosity(eps_v, st: Stratum):
    """n = n0 + eps_v, with eps_v = du/dx + dw/dz (positive in extension)."""
    return st.n0 + eps_v


def kozeny_carman(n, st: Stratum):
    """K_s(n) = K_s0 * (n/n0)^3 * ((1-n0)/(1-n))^2.

    This is the two-way coupling: mechanical volumetric strain changes porosity,
    porosity changes conductivity, conductivity changes the pressure field, and
    pressure feeds back into effective stress. Evaluated on the same collocation
    points, so there is no outer iteration."""
    return st.K_s * (n / st.n0) ** 3 * ((1.0 - st.n0) / (1.0 - n)) ** 2


# --------------------------------------------------------------------------
# Hoek-Brown
# --------------------------------------------------------------------------
def hoek_brown_params(GSI: float, m_i: float, D: float = 1.0):
    m_b = m_i * math.exp((GSI - 100.0) / (28.0 - 14.0 * D))
    s = math.exp((GSI - 100.0) / (9.0 - 3.0 * D))
    a = 0.5 + (1.0 / 6.0) * (math.exp(-GSI / 15.0) - math.exp(-20.0 / 3.0))
    return m_b, s, a


def hoek_brown_sigma1(sigma3, st: Stratum):
    """Major principal effective stress at failure (compression positive).
    sigma_1' = sigma_3' + sigma_ci*(m_b*sigma_3'/sigma_ci + s)^a"""
    if not st.is_rock:
        raise ValueError(f"{st.key} is a soil unit — use Mohr-Coulomb, not Hoek-Brown")
    return sigma3 + st.sigma_ci * (st.m_b * sigma3 / st.sigma_ci + st.s) ** st.a


def hoek_brown_ssr(sigma3, st: Stratum, srf: float):
    """Hoek-Brown envelope with strength reduced by factor srf, for the GHB-SSR
    sweep of Step 5.2. Reduction applied to m_b and s per the standard scheme."""
    m_b, s, a = st.m_b / srf, st.s / srf, st.a
    return sigma3 + st.sigma_ci * (m_b * sigma3 / st.sigma_ci + s) ** a


def mohr_coulomb_equivalent(st: Stratum, sig3max: float):
    """Hoek (2002) closed-form MC linearisation over 0 <= sig3' <= sig3max.
    Returns (c [Pa], phi [deg]).

    WARNING: c and phi from this fit are NOT material constants. They depend
    entirely on sig3max — see DATA['hoek_brown_mc_equivalents']. Near a free
    face this systematically over-predicts strength relative to the curved
    GHB envelope."""
    m_b, s, a = st.m_b, st.s, st.a
    s3n = sig3max / st.sigma_ci
    term = (s + m_b * s3n) ** (a - 1.0)
    phi = math.asin((6 * a * m_b * term) /
                    (2 * (1 + a) * (2 + a) + 6 * a * m_b * term))
    c = (st.sigma_ci * ((1 + 2 * a) * s + (1 - a) * m_b * s3n) * term) / \
        ((1 + a) * (2 + a) * math.sqrt(1 + (6 * a * m_b * term) / ((1 + a) * (2 + a))))
    return c, math.degrees(phi)


# --------------------------------------------------------------------------
# Non-dimensionalisation  (supersedes nondim.py)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Scales:
    L_ref: float
    T_ref: float
    H_ref: float
    sig_ref: float
    K_ref: float
    E_ref: float
    rho_b_ref: float
    U_ref: float

    # --- dimensionless groups ---
    @property
    def Pi_R_diff(self) -> float:
        """Richards diffusion group T*K/L^2. NOT O(1) with T_ref = 1 day — see
        DATA['nondimensionalisation']['caveat']. That is physically correct,
        and it is the term you must normalise explicitly in the loss."""
        return self.T_ref * self.K_ref / self.L_ref ** 2

    @property
    def Pi_R_grav(self) -> float:
        return self.K_ref * self.T_ref / self.L_ref

    @property
    def Pi_R_hz(self) -> float:
        return self.H_ref / self.L_ref

    @property
    def Pi_M_body(self) -> float:
        return self.rho_b_ref * G * self.L_ref / self.sig_ref

    @property
    def Pi_M_couple(self) -> float:
        return RHO_W * G * self.H_ref / self.sig_ref

    @property
    def T_diffusive(self) -> float:
        """L^2/K — the timescale on which pressure actually crosses the pit."""
        return self.L_ref ** 2 / self.K_ref


_s = DATA["nondimensionalisation"]["scales"]
SCALES = Scales(L_ref=_s["L_ref_m"], T_ref=_s["T_ref_s"], H_ref=_s["H_ref_m"],
                sig_ref=_s["sig_ref_Pa"], K_ref=_s["K_ref_mps"], E_ref=_s["E_ref_Pa"],
                rho_b_ref=_s["rho_b_ref_kgpm3"], U_ref=_s["U_ref_m"])


def nd_x(x, s: Scales = SCALES): return x / s.L_ref
def nd_t(t, s: Scales = SCALES): return t / s.T_ref
def nd_psi(p, s: Scales = SCALES): return p / s.H_ref
def nd_u(u, s: Scales = SCALES): return u / s.U_ref
def nd_sig(g, s: Scales = SCALES): return g / s.sig_ref

def redim_x(x, s: Scales = SCALES): return x * s.L_ref
def redim_t(t, s: Scales = SCALES): return t * s.T_ref
def redim_psi(p, s: Scales = SCALES): return p * s.H_ref
def redim_u(u, s: Scales = SCALES): return u * s.U_ref
def redim_sig(g, s: Scales = SCALES): return g * s.sig_ref


def richards_residual_nd(C_star, dpsi_dt_star, div_Kgrad_star, dK_dz_star,
                         s: Scales = SCALES):
    """C* dpsi*/dt* - Pi_R_diff*div*[K* grad* psi*] - Pi_R_grav*dK*/dz* = 0"""
    return (C_star * dpsi_dt_star
            - s.Pi_R_diff * div_Kgrad_star
            - s.Pi_R_grav * dK_dz_star)


def mechanical_residual_nd(div_sigma_eff_star, rho_ratio, e_z=(0.0, -1.0),
                           s: Scales = SCALES):
    """div* sigma*_eff + Pi_M_body*(rho_b/rho_b_ref)*e_z = 0. Returns (res_x, res_z)."""
    bx, bz = e_z
    body = s.Pi_M_body * rho_ratio
    return div_sigma_eff_star[0] + body * bx, div_sigma_eff_star[1] + body * bz


def effective_stress_nd(sigma_star, psi_star, chi, s: Scales = SCALES):
    """Bishop effective stress, dimensionless: sigma*_eff = sigma* - Pi_M_couple*chi*psi*.
    chi = Se in the unsaturated zone, 1 when saturated."""
    return sigma_star - s.Pi_M_couple * chi * psi_star


# --------------------------------------------------------------------------
# Boundary conditions and validation targets
# --------------------------------------------------------------------------
BC = DATA["boundary_conditions"]
VALIDATION = DATA["validation"]
GEOMETRY = DATA["geometry"]


def seepage_flux(t_days):
    """Neumann flux on the excavation face, m^3/s per metre of excavation length.

        q(t) = q1 / sqrt(t),  q1 = 0.56 m^3/day/m

    The 0.056 printed in the paper's text is a factor-of-ten typo; the plotted
    curve and the published 30-day cumulative both give 0.56. See conflict C2."""
    q1 = BC["seepage_face_flux"]["q1_m3_day_per_m"]
    return q1 / (t_days ** 0.5) / 86400.0


def drawdown_head(x_m, t_days):
    """Analytical piezometric head h(x,t) [m], Nguyen & Raudkivi (1982):
        h = h0 - s_w*erfc( x / (2*sqrt(w*t)) )
    Phase-5 VALIDATION TARGET — do not train on it. See conflict C10 for the
    T/S inconsistency this solution carries."""
    d = BC["drawdown_analytic"]
    w = d["w_m2_per_day"]
    return d["h0_m"] - d["s_w_m"] * math.erfc(x_m / (2.0 * math.sqrt(w * t_days)))


def fos_targets(section: Optional[str] = None):
    """LEM factors of safety to benchmark PINN-SSR against (Grade A)."""
    rows = VALIDATION["lem_factors_of_safety"]
    return [r for r in rows if section is None or r["section"] == section]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def print_conflicts():
    print("SOURCE CONFLICTS RESOLVED IN THIS DATASET")
    print("=" * 78)
    for c in DATA["conflicts_resolved"]:
        print(f"\n[{c['id']}] {c['severity'].upper():6s}  {c['field']}")
        for src in c["sources_disagree"]:
            print(f"    - {src}")
        print(f"  -> {c['resolution']}")
        if "reason" in c:
            print(f"     {c['reason']}")


def summary():
    print(f"{DATA['schema']['name']} v{DATA['schema']['version']}  ({DATA['schema']['generated']})")
    print("=" * 104)
    hdr = (f"{'stratum':16} {'kind':5} {'rho_dry':>8} {'rho_sat':>8} {'n0':>6} "
           f"{'K_s (m/s)':>10} {'th_s':>6} {'alpha':>6} {'n':>6} {'E (Pa)':>10} "
           f"{'nu':>5} {'tier':>4}")
    print(hdr)
    print("-" * 104)
    for k in ORDER:
        st = STRATA[k]
        ks = "n/a" if st.K_s is None else f"{st.K_s:.2e}"
        print(f"{k:16} {st.kind:5} {st.rho_dry:8.0f} {st.rho_sat:8.0f} {st.n0:6.3f} "
              f"{ks:>10} {st.theta_s:6.3f} {st.alpha:6.3f} {st.n:6.3f} "
              f"{st.E:10.3e} {st.nu:5.2f} {st.vg_tier:>4}")
    print("\nHOEK-BROWN (rock strata, D = 1)")
    for k in ORDER:
        st = STRATA[k]
        if st.is_rock:
            print(f"  {k:16} GSI={st.GSI:3.0f}  sig_ci={st.sigma_ci:9.3e} Pa  "
                  f"m_i={st.m_i:3.0f}  m_b={st.m_b:.4f}  s={st.s:.3e}  a={st.a:.4f}")
    print("\nDIMENSIONLESS GROUPS")
    for name, val in [("Pi_R_diff  T*K/L^2", SCALES.Pi_R_diff),
                      ("Pi_R_grav  K*T/L", SCALES.Pi_R_grav),
                      ("Pi_R_hz    H/L", SCALES.Pi_R_hz),
                      ("Pi_M_body  rho*g*L/sig", SCALES.Pi_M_body),
                      ("Pi_M_couple rho_w*g*H/sig", SCALES.Pi_M_couple)]:
        flag = "O(1) ok" if 1e-2 <= abs(val) <= 1e2 else "OFF-SCALE -> normalise explicitly"
        print(f"  {name:28} = {val: .4e}   {flag}")
    print(f"  U_ref = {SCALES.U_ref:.3f} m   T_diffusive = {SCALES.T_diffusive/86400/365:.0f} yr")
    print(f"\n{len(DATA['conflicts_resolved'])} source conflicts resolved "
          f"— call print_conflicts()")
    print(f"{len(VALIDATION['lem_factors_of_safety'])} LEM benchmarks, "
          f"{len(VALIDATION['monitoring_wells'])} monitoring wells, "
          f"{len(BC['seepage_face_flux']['series'])}-day inflow series")
    blocking = [o for o in DATA["open_items"] if "BLOCK" in o["status"].upper()]
    if blocking:
        print("\nBLOCKING OPEN ITEMS:")
        for o in blocking:
            print(f"  ! {o['item']} — {o['why']}")


if __name__ == "__main__":
    summary()
