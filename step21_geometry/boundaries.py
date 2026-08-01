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
import geometry as g

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


def flux_bc(segment, pts):
    """Prescribed normal flux q.n (m/s). POSITIVE = into the domain."""
    if segment in ("natural_ground", "bench"):
        nrm = outward_normal(g.z_ground, pts[:, 0])
        return RAIN_FLUX * nrm[:, 1]
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
#   Mk    MR 400, sig_ci 17.9 MPa, GSI 50  ->  478 MPa    (Step 1.3)
#   Mk_d  MR 400, sig_ci 4.29 MPa, GSI 45  ->  86.5 MPa   (Step 1.3)
#   Tm    MR 500, sig_ci 70.0 MPa, GSI 60  ->  4264 MPa   DERIVED HERE --
#         Step 1.3 predates the identification of Tm as basement limestone.
#         MR 500 is Hoek's crystalline-limestone value; sig_ci 70 MPa is the
#         same literature estimate as geometry.SIG_CI. Sensitivity, not fact.

E_RM = {"Mk": 4.78e8, "Mk_d": 8.65e7, "Tm": 4.264e9}   # Pa
NU   = {"Mk": 0.28,   "Mk_d": 0.30,   "Tm": 0.25}

# K0 = nu/(1-nu). The roadmap's single 0.39 is the MARL value; Tm is 87% of
# the domain by area and takes 0.333.
K0   = {k: v / (1.0 - v) for k, v in NU.items()}

G_ACC = 9.81

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


def body_force(x, z):
    """(N,2) body force rho(x,z)*g, downward. Zero outside the domain.
    rho comes from material_tag, so the 1519/1723/2650 density contrast
    enters the momentum residual automatically."""
    tag = g.material_tag(x, z)
    rho = np.zeros(np.shape(tag), float)
    for k, v in g.RHO.items():
        rho[tag == k] = v
    return np.stack([np.zeros_like(rho), -rho * G_ACC], axis=-1)


def traction_free_normals(segment, pts):
    """Outward unit normals where sigma.n = 0 will be enforced."""
    if segment not in TRACTION_FREE:
        raise ValueError(segment + " is not traction-free")
    return outward_normal(g.z_ground, pts[:, 0])


def sigma_v_geostatic(x, z, n_layers=400):
    """Vertical geostatic stress (Pa, compression positive), integrating
    rho*g from the ground surface down to z."""
    x = np.asarray(x, float); z = np.asarray(z, float)
    top = g.z_ground(x)
    t = np.linspace(0.0, 1.0, n_layers)[:, None]
    zz = top[None, :] - t * (top - z)[None, :]
    tag = g.material_tag(np.broadcast_to(x, zz.shape), zz)
    rho = np.zeros(zz.shape)
    for k, v in g.RHO.items():
        rho[tag == k] = v
    dz = (top - z) / (n_layers - 1)
    return np.trapz(rho, dx=1.0, axis=0) * dz * G_ACC


def sigma_h_geostatic(x, z):
    """Horizontal geostatic stress, K0*sigma_v, K0 taken per unit."""
    tag = g.material_tag(x, z)
    k0 = np.zeros(np.shape(tag), float)
    for k, v in K0.items():
        k0[tag == k] = v
    return k0 * sigma_v_geostatic(x, z)


if __name__ == "__main__":
    for k, v in sample_boundaries().items():
        print("  %-16s %5d pts   x %6.1f-%6.1f   z %6.1f-%6.1f"
              % (k, len(v), v[:, 0].min(), v[:, 0].max(),
                 v[:, 1].min(), v[:, 1].max()))
