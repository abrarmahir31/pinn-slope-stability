"""Gate before Step 2.2. All checks must pass."""
import numpy as np
import geometry as g

x = np.linspace(g.X_MIN, 266.0, 2001)
ok = True

def rep(name, passed, detail=""):
    global ok
    print(f"[{'PASS' if passed else 'FAIL'}] {name:<32} {detail}")
    ok &= bool(passed)

# 1 -- stacking order
order = g.z_ground(x) >= g.z_tm_top(x) - 0.25
rep("ground above top-of-Tm", order.all(), f"{(~order).sum()} violations")

# 2 -- cut face angle
s = x <= 98
m = np.polyfit(x[s], g.z_ground(x[s]), 1)[0]
a = np.degrees(np.arctan(m))
rep("cut face 31 deg", 30 <= a <= 32, f"{a:.2f} deg")

# 3 -- toe elevation
rep("toe at 271 m asl", abs(g.z_ground(0.0) - g.Z_TOE) < 1.0,
    f"{float(g.z_ground(0.0)):.2f} m")

# 4 -- marl band thickness plausible
t = g.z_ground(x) - g.z_tm_top(x)
rep("marl band 0-40 m", t.min() >= -0.5 and t.max() <= 40,
    f"{t.min():.1f} to {t.max():.1f} m")

# 5 -- each flattening ray daylights once, then stays below ground.
# Skip x < 5 m: all rays converge on the toe within digitisation noise there,
# so the sign of (ground - ray) is meaningless and would give a false crossing.
for nm, f in [("alt20", g.z_alt20), ("alt22", g.z_alt22), ("alt25", g.z_alt25)]:
    xs = x[(x >= 5) & (x <= 250)]
    d = g.z_ground(xs) - f(xs)
    cross = np.where(np.diff(np.sign(d)))[0]
    xd = xs[cross[0]] if len(cross) else np.inf
    inner = xs < xd
    rep(f"{nm} below ground to daylight",
        inner.sum() > 10 and (d[inner] >= -0.5).all(),
        f"daylights at x = {xd:.0f} m, {inner.sum()} pts tested")

# 6 -- every interior point tagged, properties defined
X, Z = np.meshgrid(x[::8], np.linspace(g.Z_BASE, 370, 400))
tag = g.material_tag(X, Z)
rep("all interior points tagged",
    not np.any((tag == "outside") & g.inside_domain(X, Z)))
present = set(np.unique(tag)) - {"outside"}
rep("properties defined", present <= set(g.K_S) & set(g.RHO), f"{sorted(present)}")

print("\n" + ("ALL CHECKS PASSED -- proceed to Step 2.2"
              if ok else "CHECKS FAILED"))
