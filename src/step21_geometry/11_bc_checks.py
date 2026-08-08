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

# 4 -- boundary points sit on the domain edge.
# Probe along each segment's OWN inward direction: down for upper surfaces,
# up for the base, inboard in x for the pit floor and the F1 fault.
probe = {"natural_ground": (0.0, -0.5), "bench": (0.0, -0.5),
         "cut_face": (0.0, -0.5), "pit_floor": (+0.5, 0.0),
         "base": (0.0, +0.5),     "far_field_f1": (-0.5, 0.0)}
for s, p in bc.items():
    dx, dz = probe[s]
    ins = g.inside_domain(p[:, 0] + dx, p[:, 1] + dz)
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
import sys
sys.exit(0 if ok else 1)