#!/usr/bin/env bash
# Run this from step21_geometry.  It writes boundaries.py, 11_bc_checks.py,
# and 12_bc_plot.py, then runs the checks and the plot.
set -e

cat > boundaries.py << 'PYEOF'
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


if __name__ == "__main__":
    for k, v in sample_boundaries().items():
        print("  %-16s %5d pts   x %6.1f-%6.1f   z %6.1f-%6.1f"
              % (k, len(v), v[:, 0].min(), v[:, 0].max(),
                 v[:, 1].min(), v[:, 1].max()))
PYEOF

cat > 11_bc_checks.py << 'PYEOF'
"""Verify BC samplers land on their segments. Gate before Step 2.3."""
import numpy as np
import geometry as g
import boundaries as b

bc = b.sample_boundaries(2000)
ok = True

def rep(name, passed, detail=""):
    global ok
    print("[%s] %-34s %s" % ("PASS" if passed else "FAIL", name, detail))
    ok &= bool(passed)

# 1 -- surface segments lie ON the ground surface
for s in ("natural_ground", "bench", "cut_face"):
    p = bc[s]
    d = np.abs(p[:, 1] - g.z_ground(p[:, 0]))
    rep(s + " on surface", d.max() < 0.01, "max dev %.2e m" % d.max())

# 2 -- segments occupy their intended x-ranges
r = {s: (bc[s][:, 0].min(), bc[s][:, 0].max())
     for s in ("cut_face", "bench", "natural_ground")}
rep("cut face x 0-102", r["cut_face"][1] <= g.X_CREST + 1,
    "%.1f-%.1f" % r["cut_face"])
rep("bench x 102-123", r["bench"][0] >= g.X_CREST - 1
    and r["bench"][1] <= g.X_BENCH_END + 1, "%.1f-%.1f" % r["bench"])
rep("natural ground x >123", r["natural_ground"][0] >= b.X_NAT_MIN - 1,
    "%.1f-%.1f" % r["natural_ground"])

# 3 -- F1 points lie on the fault trace
p = bc["far_field_f1"]
rep("F1 pts on fault", np.abs(p[:, 0] - g.x_f1(p[:, 1])).max() < 0.01)

# 4 -- boundary points sit on the domain edge
for s, p in bc.items():
    ins = g.inside_domain(p[:, 0], p[:, 1] - 0.5)
    rep(s + " on domain edge", ins.mean() > 0.95,
        "%.1f%% inboard-adjacent" % (100 * ins.mean()))

# 5 -- normals outward, and consistent with a 31 deg face
n = b.outward_normal(g.z_ground, bc["cut_face"][:, 0])
rep("cut-face normals outward", bool((n[:, 1] > 0).all()))
c = n[:, 1].mean()
rep("cut-face normal ~cos(31 deg)", 0.83 < c < 0.88, "%.3f" % c)

# 6 -- far-field suction negative everywhere
psi = b.psi_far_field(bc["far_field_f1"][:, 1])
rep("far-field psi negative", bool((psi < 0).all()),
    "%.1f to %.1f m" % (psi.min(), psi.max()))

print("\n" + ("ALL BC CHECKS PASSED -- proceed to Step 2.3"
              if ok else "BC CHECKS FAILED"))
PYEOF

cat > 12_bc_plot.py << 'PYEOF'
"""Boundary points coloured by segment. Manuscript Fig. 4."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import geometry as g
import boundaries as b

bc = b.sample_boundaries(600)
fig, ax = plt.subplots(figsize=(12, 6))

x = np.linspace(0, 266, 500)
ax.plot(x, g.z_ground(x), "k-", lw=0.8, alpha=0.4)
ax.plot(x, g.z_tm_top(x), "k--", lw=0.8, alpha=0.4)

cols = {"natural_ground": "#2ca02c", "bench": "#8c564b",
        "cut_face": "#d62728", "pit_floor": "#1f77b4",
        "base": "#7f7f7f", "far_field_f1": "#9467bd"}
for s, p in bc.items():
    ax.scatter(p[:, 0], p[:, 1], s=6, c=cols[s],
               label="%s (%d)" % (s, len(p)))

p = bc["cut_face"][::20]
n = b.outward_normal(g.z_ground, p[:, 0])
ax.quiver(p[:, 0], p[:, 1], n[:, 0], n[:, 1], color="k",
          scale=25, width=0.002, alpha=0.6)

ax.axhline(b.Z_WT, ls=":", c="b", lw=1.2,
           label="karstic water table %.0f m (below domain)" % b.Z_WT)
ax.set_xlabel("x (m from toe)")
ax.set_ylabel("z (m a.s.l.)")
ax.legend(fontsize=8, ncol=2)
ax.set_title("Section 5 boundary sampling")
plt.tight_layout()
plt.savefig("output/section5_boundaries.png", dpi=200)
print("written: output/section5_boundaries.png")
PYEOF

echo "--- files written ---"
python boundaries.py
echo
python 11_bc_checks.py
echo
python 12_bc_plot.py
