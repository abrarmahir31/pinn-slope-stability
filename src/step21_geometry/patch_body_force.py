"""
patch_body_force.py -- add saturation weight to the mechanical body force
and the geostatic column.

Run from step21_geometry/ AFTER vg.py and properties.py are in place:

    python patch_body_force.py
    python 13_mech_checks.py          # or whatever your mech check is called

WHY. rho in geometry.RHO is DRY (Mk 1518.9 = gamma 14.9 kN/m3). Section 5's
mechanism is rainfall infiltration, which has two limbs: suction is lost
(Bishop coupling, already present) AND the wetted material gets heavier
(absent). Mk dry -> saturated is 1519 -> 1946 kg/m3, +28%.

It also matters at t = 0. psi_0 = -(z - 197) reaches -168 m, but n = 1.2 makes
the retention curve so flat that Se only falls to ~0.41 there: the initial
state is NOT dry, theta_0 ~ 0.19-0.24 in the marl. The stored geostatic column
has therefore been ~12% light through the marl band -- which is where Mk_d,
the failing unit, sits.

CONSEQUENCE: sigma_0 is the reference for the incremental convention
sigma_total = sigma_0 + dsigma(u,v), it feeds the Bishop coupling and every SRF
evaluation. REGENERATE ic_cache.npz after this patch.
"""

import re
import pathlib

SRC = pathlib.Path("boundaries.py")
s = SRC.read_text()
orig = s

# ---------------------------------------------------------------- body_force
BODY = '''def body_force(x, z, psi=None):
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
'''

# ------------------------------------------------------- sigma_v_geostatic
SIGMA = '''# cumulative water-content integral, built once per unit.
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
'''


def swap(pattern, new, label):
    global s
    m = re.search(pattern, s, re.S)
    if not m:
        raise SystemExit(f"anchor not found for {label}")
    s = s[:m.start()] + new + s[m.end():]
    print(f"  replaced {label}")


swap(r"def body_force\(x, z\):.*?(?=\n\ndef |\n\n# )", BODY, "body_force")
swap(r"def sigma_v_geostatic\(x, z, n_layers=None\):.*?(?=\n\nK0_SMOOTH_W)",
     SIGMA, "sigma_v_geostatic")

# imports
if "import vg" not in s:
    s = re.sub(r"(import geometry as g\n)", r"\1import properties as P\nimport vg\n", s, count=1)
    print("  added imports")
if "RHO_W" not in s.split("def ")[0]:
    s = re.sub(r"(\nG_ACC\s*=\s*[\d.]+)", r"\1\nRHO_W = 1000.0", s, count=1)
    print("  added RHO_W")

SRC.write_text(s)
print(f"patched boundaries.py ({len(orig)} -> {len(s)} chars)")

# ---------------------------------------------------------------- verify
print("\n--- dry vs wet geostatic column ---")
import importlib
import numpy as np
import geometry as g
import boundaries as B
importlib.reload(B)

xs = np.full(6, 150.0)
zs = np.array([360.0, 340.0, 320.0, 300.0, 250.0, 200.0])
d = B.sigma_v_geostatic(xs, zs, wet=False)
w = B.sigma_v_geostatic(xs, zs, wet=True)
print(f"{'z':>7} {'dry kPa':>10} {'wet kPa':>10} {'diff':>8}")
for zz, dd, ww in zip(zs, d, w):
    pct = 100 * (ww - dd) / dd if dd > 0 else 0.0
    print(f"{zz:7.1f} {dd/1e3:10.1f} {ww/1e3:10.1f} {pct:7.1f}%")
print("\nREGENERATE ic_cache.npz -- sigma_0 has changed.")
