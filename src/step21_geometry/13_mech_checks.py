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
rep("boundary partition complete", seen == set(bc), "%d segments" % len(seen))
rep("no segment double-assigned",
    len(b.FIXED) + len(b.ROLLER_X) + len(b.TRACTION_FREE) == len(seen))

# 2 -- K0 per unit
rep("K0 = nu/(1-nu) per unit",
    all(abs(b.K0[k] - b.NU[k]/(1-b.NU[k])) < 1e-9 for k in b.NU),
    "Mk %.3f  Mk_d %.3f  Tm %.3f" % (b.K0["Mk"], b.K0["Mk_d"], b.K0["Tm"]))

# 3 -- body force magnitude per unit
for k, p in [("Mk", (80.0, 308.9)), ("Mk_d", (200.0, 340.0)),
             ("Tm", (150.0, 260.0))]:
    tag = str(g.material_tag(np.array([p[0]]), np.array([p[1]]))[0])
    f = b.body_force(np.array([p[0]]), np.array([p[1]]))[0]
    want = -g.RHO[k]*b.G_ACC
    rep("body force in "+k, tag == k and abs(f[1]-want) < 1.0,
        "%.0f N/m3 (tag %s)" % (f[1], tag))

# 4 -- zero outside the domain
f = b.body_force(np.array([50.0]), np.array([400.0]))[0]
rep("body force zero outside", abs(f[1]) < 1e-9)

# 5 -- geostatic sigma_v vs rho*g*h by hand
# The hand calculation is a DRY skeleton column. sigma_v_geostatic(wet=True)
# adds rho_w * int(theta dz) under the t=0 suction profile, +9.6% here.
# Compare like with like, then assert the water weight separately so it
# cannot silently vanish.
xq, zq = 200.0, 335.0
sv_dry = float(b.sigma_v_geostatic(np.array([xq]), np.array([zq]), wet=False)[0])
sv_wet = float(b.sigma_v_geostatic(np.array([xq]), np.array([zq]))[0])
h = float(g.z_ground(xq) - zq)
approx = g.RHO["Mk_d"] * b.G_ACC * h
rep("sigma_v (dry) ~ rho*g*h", abs(sv_dry - approx) / max(approx, 1) < 0.01,
    "%.0f vs %.0f Pa over %.1f m" % (sv_dry, approx, h))
rep("water weight adds 5-15%", 1.05 < sv_wet / sv_dry < 1.15,
    "wet/dry = %.4f" % (sv_wet / sv_dry))

# 6 -- monotonic with depth
z = np.linspace(340.0, 210.0, 60)
sv = b.sigma_v_geostatic(np.full_like(z, 200.0), z)
rep("sigma_v monotonic with depth", bool(np.all(np.diff(sv) > 0)),
    "%.2f MPa at z=210" % (sv[-1]/1e6))

# 7 -- sigma_h < sigma_v
sh = b.sigma_h_geostatic(np.full_like(z, 200.0), z)
rep("sigma_h < sigma_v", bool(np.all(sh < sv)),
    "ratio %.3f-%.3f" % ((sh/sv).min(), (sh/sv).max()))

# 8 -- traction-free normals
for s in b.TRACTION_FREE:
    n = b.traction_free_normals(s, bc[s])
    rep(s+" normals outward", bool((n[:,1] > 0).all()),
        "mean nz %.3f" % n[:,1].mean())

# 9 -- total weight sanity
r = np.random.default_rng(0)
px = r.uniform(0, 268, 200000); pz = r.uniform(g.Z_BASE, 366, 200000)
ins = g.inside_domain(px, pz)
area = ins.mean()*268*(366-g.Z_BASE)
w = -b.body_force(px[ins], pz[ins])[:,1].mean()*area
rep("domain weight plausible", 5e8 < w < 5e9,
    "%.2f GN/m over %.0f m2" % (w/1e9, area))

print("\n" + ("ALL MECHANICAL CHECKS PASSED -- Step 2.2 complete"
              if ok else "MECHANICAL CHECKS FAILED"))
