"""
Boundary-condition samplers for Section 5, Isikdere pit.

Stratified by segment: each physical condition gets its own point set, so the
loss can weight them independently and each can be verified separately.

Units SI throughout: fluxes m/s, heads and elevations m, pressures Pa.

HYDROGEOLOGICAL CONTEXT
    Karstic (Tm) head is 113-197 m a.s.l. (Ulusay et al. 2014, Sec. 5.3, six
    wells screened in the karstic bedrock). Z_BASE = 200 m a.s.l. The water
    table therefore lies BELOW the domain: every point is unsaturated, and
    this is a rainfall-infiltration problem, not a confined-aquifer one.
"""
import numpy as np
try:
    import geometry as g
    from src import properties as P, vg
except ModuleNotFoundError:
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from src import properties as P, vg
    from src.step21_geometry import geometry as g

RAIN_FLUX = 5.56e-6          # m/s, 20 mm/hr triggered case
Z_WT      = 197.0            # m a.s.l., top of the karstic head range
X_F1_BASE = float(g.x_f1(g.Z_BASE))
Z_F1_TOP  = 358.07
X_NAT_MIN = 123.0            # bench ends / natural ground begins

SEEPAGE_FACE_MODE = "noflow"   # "noflow" | "fischer_burmeister"


def outward_normal(f, x, h=0.05):
    """Unit outward normal on an upper surface z = f(x).

    Neumann conditions are on q.n, not q_z. On the 31 deg cut face these
    differ by 14 per cent.
    """
    x = np.asarray(x, float)
    dz = (f(x + h) - f(x - h)) / (2 * h)
    nx, nz = -dz, np.ones_like(dz)
    m = np.hypot(nx, nz)
    return np.column_stack([nx / m, nz / m])


def _arclength_sample(f, x0, x1, n, rng):
    """n points on z = f(x), spaced evenly in ARC LENGTH.

    Uniform sampling in x would under-resolve the steep cut face relative to
    the flat bench by about 17 per cent.
    """
    xs = np.linspace(x0, x1, 400)
    zs = f(xs)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(zs)))])
    xq = np.interp(rng.uniform(0, s[-1], n), s, xs)
    return np.column_stack([xq, f(xq)])


def sample_boundaries(n=2000, seed=0):
    """Returns {segment: (N,2) array of (x, z)}."""
    r = np.random.default_rng(seed)
    return {
        "natural_ground": _arclength_sample(g.z_ground, X_NAT_MIN, 266.0, n, r),
        "bench":          _arclength_sample(g.z_ground, g.X_CREST,
                                            g.X_BENCH_END, n // 4, r),
        "cut_face":       _arclength_sample(g.z_ground, 0.0, g.X_CREST, n, r),
        "pit_floor":      np.column_stack([np.zeros(n // 3),
                                           r.uniform(g.Z_BASE, g.Z_TOE, n // 3)]),
        "base":           np.column_stack([r.uniform(0.0, X_F1_BASE, n // 2),
                                           np.full(n // 2, g.Z_BASE)]),
        "far_field_f1":   (lambda z: np.column_stack([g.x_f1(z), z]))(
                              r.uniform(g.Z_BASE, Z_F1_TOP, n)),
    }


def psi_far_field(z):
    """Suction profile on F1, hydrostatic w.r.t. a water table at Z_WT.

    NOT a positive-pressure hydrostatic condition -- the table is below the
    domain, so psi is negative everywhere: -3 m at the base, -161 m at the top.
    """
    return -(np.asarray(z, float) - Z_WT)


# Saturated conductivity per stratum, m/s. Imported rather than restated --
# see properties.py / materials.py for provenance.
#   Mk    1.00e-09   0.004 mm/hr
#   Mk_d  1.00e-09   0.004 mm/hr   (marl analog; see open_items.md)
#   Tm    3.13e-06  11.268 mm/hr
K_SAT = {"Mk": 1.0e-9, "Mk_d": 1.0e-9, "Tm": 3.13e-6}

INFILTRATION_MODE = "capacity_limited"   # "raw" | "capacity_limited"


def infiltration_capacity(x, z):
    """Maximum normal flux the ground can accept at (x, z), m/s.

    Darcy caps infiltration at K(psi) <= K_s. Prescribing more than this makes
    the boundary condition unsatisfiable: no psi field exists that conducts
    5560x the saturated conductivity of marl, so L_BC would plateau at a floor
    set by an unphysical target rather than by convergence.

    K_s is the ceiling (reached only at psi = 0). Using K_s rather than
    K(psi) keeps this a fixed target instead of a solution-dependent one --
    the psi-dependent version is the complementarity BC below, deferred with
    the seepage face for the same reason.
    """
    tag = g.material_tag(np.asarray(x, float), np.asarray(z, float))
    return np.array([K_SAT[str(t)] for t in np.atleast_1d(tag)])


def flux_bc(segment, pts):
    """Prescribed normal flux q.n (m/s). POSITIVE = into the domain.

    On rainfall segments the applied flux is min(rain . n, K_s). The rest is
    runoff.

    This is NOT a numerical convenience. At 20 mm/hr against marl at
    0.004 mm/hr, 99.98 per cent of the rain runs off; the uncapped condition
    asks the marl surface to conduct 5560x its saturated conductivity. On Tm
    the ratio is 1.8x, so roughly 56 per cent infiltrates. Capping changes
    which stratum drives the infiltration response, and that belongs in the
    methods section, not in a comment.

    Set INFILTRATION_MODE = "raw" to recover the old behaviour for comparison.
    """
    if segment in ("natural_ground", "bench", "cut_face_rain"):
        nrm = outward_normal(g.z_ground, pts[:, 0])
        q_rain = RAIN_FLUX * nrm[:, 1]
        if INFILTRATION_MODE == "raw":
            return q_rain
        return np.minimum(q_rain, infiltration_capacity(pts[:, 0], pts[:, 1]))
    if segment in ("pit_floor", "base"):
        return np.zeros(len(pts))
    if segment == "cut_face" and SEEPAGE_FACE_MODE == "noflow":
        return np.zeros(len(pts))
    raise ValueError(segment + " is not a flux boundary in mode " +
                     SEEPAGE_FACE_MODE)

def fischer_burmeister(psi, qn, eps=1e-6):
    """Smooth complementarity residual for the seepage face.

    phi(a,b) = a + b - sqrt(a^2 + b^2 + eps),  a = -psi, b = q.n
    Zero iff psi <= 0, q.n >= 0, psi*(q.n) = 0.
    Enable only after a working "noflow" run shows the wetting front reaching
    the face -- debugging complementarity and PINN convergence at once is a
    bad place to be.
    """
    a, b = -psi, qn
    return a + b - np.sqrt(a * a + b * b + eps)


def mechanical_bc(segment):
    """Displacement constraints per segment. None = traction-free."""
    return {"base":         ("fixed",  dict(u=0.0, v=0.0)),
            "far_field_f1": ("roller", dict(u=0.0)),
            "pit_floor":    ("roller", dict(u=0.0))}.get(segment)



# ============================ MECHANICAL BOUNDARY DATA =======================
# Elastic constants, Hoek-Diederichs (2006):
#   E_rm = E_i (0.02 + (1 - D/2)/(1 + exp((60 + 15D - GSI)/11))),  E_i = MR*sig_ci
#   Mk    MR 175, sig_ci 17.9 MPa, GSI 50  ->  209 MPa    (Step 1.3)
#   Mk_d  MR 175, sig_ci 4.29 MPa, GSI 45  ->  38.1 MPa   (Step 1.3)
#   Tm    MR 500, sig_ci 70.0 MPa, GSI 60  ->  4264 MPa   DERIVED HERE --
#         Step 1.3 predates the identification of Tm as basement limestone.
#         MR 500 is Hoek's crystalline-limestone value; sig_ci 70 MPa is the
#         same literature estimate as geometry.SIG_CI. Sensitivity, not fact.

# Elastic constants come from properties.py (single source of truth), which
# reads the Phase-1 dataset: Hoek-Diederichs with MR_adopted = 175 for both
# marls, MR 500 for Tm. The MR 400 figures previously hardcoded here were
# never in the derivation pipeline -- 4.78e8 vs the dataset's 2.089e8 for Mk,
# a 2.29x discrepancy that existed only in this file's comment block.

try:
    from src.properties import E as E_RM, NU, RHO_DRY as RHO
except ModuleNotFoundError:
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from src.properties import E as E_RM, NU, RHO_DRY as RHO

K0 = {k: v / (1.0 - v) for k, v in NU.items()}

G_ACC = 9.81
RHO_W = 1000.0

FIXED         = ("base",)
ROLLER_X      = ("far_field_f1", "pit_floor")
TRACTION_FREE = ("natural_ground", "bench", "cut_face")


def mechanical_bc(segment):
    """(kind, constraints) for a segment. None means traction-free."""
    if segment in FIXED:
        return ("fixed", dict(u=0.0, v=0.0))
    if segment in ROLLER_X:
        return ("roller_x", dict(u=0.0))
    if segment in TRACTION_FREE:
        return None
    raise ValueError("unknown segment: " + segment)


def body_force(x, z, psi=None):
    """(N,2) body force rho_b(x,z,psi)*g, downward. Zero outside the domain.

    rho_b = rho_dry + theta(psi) * rho_w.

    psi=None reproduces the old DRY behaviour and is kept only so existing
    callers do not silently break -- it is NOT the physical case. Pass the
    network's psi in Phase 3; the body force is part of the HM coupling and
    autograd must flow through it.
    """
    tag = g.material_tag(x, z)
    rho = np.zeros(np.shape(tag), float)
    for k, v in P.RHO_DRY.items():
        rho[tag == k] = v
    if psi is not None:
        th = np.zeros(np.shape(tag), float)
        for k in P.UNITS:
            m = tag == k
            if m.any():
                th[m] = vg.theta(np.asarray(psi, float)[m],
                                 P.THETA_R[k], P.THETA_S[k],
                                 P.ALPHA[k], P.VG_N[k])
        rho = rho + th * RHO_W
    return np.stack([np.zeros_like(rho), -rho * G_ACC], axis=-1)


def traction_free_normals(segment, pts):
    """Outward unit normals where sigma.n = 0 will be enforced."""
    if segment not in TRACTION_FREE:
        raise ValueError(segment + " is not traction-free")
    return outward_normal(g.z_ground, pts[:, 0])


# cumulative water-content integral, built once per unit.
# theta depends on z only (psi_0 is linear in z), so INT theta dz is a 1-D
# function per material. Tabulate it and interpolate: same speed as the closed
# form, and exact to the grid.
_ZQ = np.linspace(190.0, 380.0, 20001)


def _water_cumint(unit, z_wt=197.0):
    th = vg.theta(vg.psi_initial(_ZQ, z_wt),
                  P.THETA_R[unit], P.THETA_S[unit],
                  P.ALPHA[unit], P.VG_N[unit])
    F = np.concatenate([[0.0], np.cumsum(0.5 * (th[1:] + th[:-1]) * np.diff(_ZQ))])
    return F


_WCUM = {u: _water_cumint(u) for u in P.UNITS}


def _wint(unit, a, b):
    """INT_a^b theta(z) dz, elementwise, clipped at a<=b."""
    F = _WCUM[unit]
    return np.clip(np.interp(b, _ZQ, F) - np.interp(a, _ZQ, F), 0.0, None)


def sigma_v_geostatic(x, z, n_layers=None, wet=True, z_wt=197.0):
    """Vertical geostatic stress (Pa, compression positive).

    Closed form for the dry skeleton -- every column is at most two layers,
    because the Mk/Mk_d divide is vertical, so a column is Mk-over-Tm or
    Mk_d-over-Tm, never both.

    The water weight is added as a tabulated 1-D integral of theta(z) under the
    t = 0 suction profile psi_0 = -(z - z_wt). It has no closed form for
    van Genuchten n = 1.2.

    wet=False reproduces the old dry column, for comparison only.
    n_layers is accepted and ignored, for signature compatibility.
    """
    x = np.asarray(x, float)
    z = np.asarray(z, float)
    top = g.z_ground(x)
    # the contact pinches out at the toe; digitisation puts it ~0.14 m above
    # ground at x = 0, which would otherwise integrate a negative thickness
    ztm = np.minimum(g.z_tm_top(x), top)
    is_mk = x < g.X_MK_DIVIDE
    up = np.where(is_mk, "Mk", "Mk_d")
    rho_up = np.where(is_mk, P.RHO_DRY["Mk"], P.RHO_DRY["Mk_d"])
    z_up = np.maximum(z, ztm)
    h_up = np.clip(top - z_up, 0.0, None)      # marl above the point
    h_tm = np.clip(ztm - z, 0.0, None)         # Tm above the point
    dry = rho_up * h_up + P.RHO_DRY["Tm"] * h_tm
    if not wet:
        return G_ACC * dry
    w_up = np.where(is_mk,
                    _wint("Mk", z_up, top),
                    _wint("Mk_d", z_up, top))
    w_tm = _wint("Tm", z, ztm)
    return G_ACC * (dry + RHO_W * (w_up + w_tm))


K0_SMOOTH_W = 2.0   # m, half-width of the K0 transition across the contact


def k0_field(x, z):
    """Per-unit K0, linearly blended over +/-K0_SMOOTH_W about the Tm contact.

    K0 is genuinely discontinuous there (sigma_v is continuous, sigma_h is not).
    A tanh network cannot represent a jump, so the contact is smoothed over a
    stated width. This is a numerical device applied to a quantity that is
    already an idealisation -- record K0_SMOOTH_W in the decision log.
    """
    x = np.asarray(x, float); z = np.asarray(z, float)
    top = g.z_ground(x)
    ztm = np.minimum(g.z_tm_top(x), top)
    k_up = np.where(x < g.X_MK_DIVIDE, K0["Mk"], K0["Mk_d"])
    k_dn = K0["Tm"]
    t = np.clip((z - (ztm - K0_SMOOTH_W)) / (2.0 * K0_SMOOTH_W), 0.0, 1.0)
    return k_dn + (k_up - k_dn) * t


def sigma_h_geostatic(x, z):
    """Horizontal geostatic stress, K0*sigma_v, with K0 smoothed across contacts."""
    return k0_field(x, z) * sigma_v_geostatic(x, z)


if __name__ == "__main__":
    for k, v in sample_boundaries().items():
        print("  %-16s %5d pts   x %6.1f-%6.1f   z %6.1f-%6.1f"
              % (k, len(v), v[:, 0].min(), v[:, 0].max(),
                 v[:, 1].min(), v[:, 1].max()))
