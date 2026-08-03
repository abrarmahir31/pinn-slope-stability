#!/usr/bin/env bash
# Step 2.2 Day 12 -- mechanical boundary conditions.
# Run from step21_geometry. Appends to boundaries.py, writes 13_mech_checks.py.
set -e

python - << 'PATCH'
s = open('boundaries.py').read()

# replace the stub mechanical_bc with the full block
old = s[s.index('# ---------------------------------------------------------------- mechanical'):
        s.index('if __name__ == "__main__":')]

new = '''# ============================ MECHANICAL BOUNDARY DATA =======================
# Elastic constants, Hoek-Diederichs (2006):
#   E_rm = E_i (0.02 + (1 - D/2) / (1 + exp((60 + 15D - GSI)/11))),  E_i = MR*sig_ci
#
#   Mk    MR 400, sig_ci 17.9 MPa, GSI 50  ->  E_rm  478 MPa   (Step 1.3)
#   Mk_d  MR 400, sig_ci 4.29 MPa, GSI 45  ->  E_rm 86.5 MPa   (Step 1.3)
#   Tm    MR 500, sig_ci 70.0 MPa, GSI 60  ->  E_rm 4264 MPa   DERIVED HERE --
#         Step 1.3 predates the identification of Tm as basement limestone and
#         omits it. MR 500 is Hoek's crystalline-limestone value; sig_ci 70 MPa
#         is the same literature estimate used in geometry.SIG_CI. Both feed
#         the sensitivity sweep, not a measured value.

E_RM = {"Mk": 4.78e8, "Mk_d": 8.65e7, "Tm": 4.264e9}   # Pa
NU   = {"Mk": 0.28,   "Mk_d": 0.30,   "Tm": 0.25}      # -

# K0 = nu / (1 - nu). The roadmap's single value of 0.39 is the MARL figure;
# Tm is 87 per cent of the domain by area and takes 0.333.
K0   = {k: v / (1.0 - v) for k, v in NU.items()}

G_ACC = 9.81   # m/s2

# Boundary partition. Every segment appears exactly once.
FIXED       = ("base",)                             # u = v = 0
ROLLER_X    = ("far_field_f1", "pit_floor")         # u = 0, v free
TRACTION_FREE = ("natural_ground", "bench", "cut_face")


def mechanical_bc(segment):
    """(kind, constraint dict) for a segment. None means traction-free."""
    if segment in FIXED:
        return ("fixed", dict(u=0.0, v=0.0))
    if segment in ROLLER_X:
        return ("roller_x", dict(u=0.0))
    if segment in TRACTION_FREE:
        return None
    raise ValueError("unknown segment: " + segment)


def body_force(x, z):
    """(N,2) body force rho(x,z) * g, downward. Zero outside the domain.

    rho comes from material_tag, so the three-unit density contrast
    (Mk 1519, Mk_d 1723, Tm 2650 kg/m3) enters the momentum residual
    automatically.
    """
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
    """Vertical geostatic stress (Pa, compression positive) by integrating
    rho*g from the ground surface down to z. Used as the Step 2.3 mechanical
    initial condition and as a check on the trained stress field."""
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
    """Horizontal geostatic stress, K0 * sigma_v, with K0 taken per unit."""
    tag = g.material_tag(x, z)
    k0 = np.zeros(np.shape(tag), float)
    for k, v in K0.items():
        k0[tag == k] = v
    return k0 * sigma_v_geostatic(x, z)


'''
open('boundaries.py', 'w').write(s.replace(old, new))
print("boundaries.py extended")
PATCH

cat > 13_mech_checks.py << 'PYEOF'
"""Mechanical BC verification. Gate before Step 2.3."""
import numpy as np
import geometry as g
import boundaries as b

ok = True
def rep(name, passed, detail=""):
    global ok
    print("[%s] %-36s %s" % ("PASS" if passed else "FAIL", name, detail))
    ok &= bool(passed)

bc = b.sample_boundaries(2000)

# 1 -- every segment has exactly one mechanical condition
seen = set(b.FIXED) | set(b.ROLLER_X) | set(b.TRACTION_FREE)
rep("boundary partition complete", seen == set(bc),
    "%d segments" % len(seen))
rep("no segment double-assigned",
    len(b.FIXED) + len(b.ROLLER_X) + len(b.TRACTION_FREE) == len(seen))

# 2 -- K0 per unit, and NOT the roadmap's single marl value
rep("K0 = nu/(1-nu) per unit",
    all(abs(b.K0[k] - b.NU[k] / (1 - b.NU[k])) < 1e-9 for k in b.NU),
    "Mk %.3f  Mk_d %.3f  Tm %.3f" % (b.K0["Mk"], b.K0["Mk_d"], b.K0["Tm"]))

# 3 -- body force magnitude matches rho*g per unit
for k in ("Mk", "Mk_d", "Tm"):
    pts = {"Mk": (50.0, 290.0), "Mk_d": (200.0, 340.0), "Tm": (150.0, 260.0)}[k]
    f = b.body_force(np.array([pts[0]]), np.array([pts[1]]))[0]
    want = -g.RHO[k] * b.G_ACC
    rep("body force in " + k, abs(f[1] - want) < 1.0,
        "%.0f N/m3 (rho %.0f)" % (f[1], g.RHO[k]))

# 4 -- body force is zero outside the domain
f = b.body_force(np.array([50.0]), np.array([400.0]))[0]
rep("body force zero outside", abs(f[1]) < 1e-9)

# 5 -- geostatic stress: analytic cross-check at the crest, all Mk_d/Mk column
xq, zq = 200.0, 335.0
sv = float(b.sigma_v_geostatic(np.array([xq]), np.array([zq]))[0])
h = float(g.z_ground(xq) - zq)
approx = g.RHO["Mk_d"] * b.G_ACC * h
rep("sigma_v at crest ~ rho*g*h", abs(sv - approx) / max(approx, 1) < 0.05,
    "%.0f Pa vs %.0f Pa over %.1f m" % (sv, approx, h))

# 6 -- sigma_v increases monotonically with depth
z = np.linspace(340.0, 210.0, 60)
sv = b.sigma_v_geostatic(np.full_like(z, 200.0), z)
rep("sigma_v monotonic with depth", bool(np.all(np.diff(sv) > 0)),
    "%.2f MPa at z=210" % (sv[-1] / 1e6))

# 7 -- sigma_h < sigma_v everywhere (K0 < 1)
sh = b.sigma_h_geostatic(np.full_like(z, 200.0), z)
rep("sigma_h < sigma_v", bool(np.all(sh < sv)),
    "ratio %.3f-%.3f" % ((sh / sv).min(), (sh / sv).max()))

# 8 -- traction-free normals defined and outward on all three surfaces
for s in b.TRACTION_FREE:
    n = b.traction_free_normals(s, bc[s])
    rep(s + " normals outward", bool((n[:, 1] > 0).all()),
        "mean nz %.3f" % n[:, 1].mean())

# 9 -- total domain weight, order-of-magnitude sanity
r = np.random.default_rng(0)
px = r.uniform(0, 268, 200000); pz = r.uniform(g.Z_BASE, 366, 200000)
ins = g.inside_domain(px, pz)
area = ins.mean() * 268 * (366 - g.Z_BASE)
w = -b.body_force(px[ins], pz[ins])[:, 1].mean() * area
rep("domain weight plausible", 5e8 < w < 5e9,
    "%.2f GN/m, area %.0f m2" % (w / 1e9, area))

print("\n" + ("ALL MECHANICAL CHECKS PASSED -- Step 2.2 complete"
              if ok else "MECHANICAL CHECKS FAILED"))
PYEOF

echo "--- running ---"
python 13_mech_checks.py
