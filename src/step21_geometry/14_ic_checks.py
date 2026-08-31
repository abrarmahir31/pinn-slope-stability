# Windows: torch must load before numpy/MKL or shm.dll fails to resolve.
import torch as _torch  # noqa: F401

"""Step 2.3 acceptance checks. Every check guards against passing on an empty array."""
import hashlib, json, pathlib
import numpy as np
import geometry as g
import boundaries as b
from initial import psi_initial

N_FAIL = 0


def check(label, cond, detail=""):
    global N_FAIL
    ok = bool(cond)
    if not ok:
        N_FAIL += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {label:44s} {detail}")


def geometry_hash():
    return hashlib.sha256(pathlib.Path("geometry.py").read_bytes()).hexdigest()[:12]


def build_ic_cache(n_ic=4000, seed=20260813, path="ic_cache.npz"):
    """Freeze the IC point set; precompute everything static on it.

    Stratified by material so the Mk wedge (~3% of area) is not swamped by Tm.
    Per-point weights keep L_IC an unbiased estimate of the domain integral.
    """
    rng = np.random.default_rng(seed)
    want = {"Mk": 0.25, "Mk_d": 0.35, "Tm": 0.40}      # sampling shares
    got = {k: [] for k in want}
    while any(len(got[k]) < want[k] * n_ic for k in want):
        xr = rng.uniform(g.X_MIN, g.X_MAX, 20000)
        zr = rng.uniform(g.Z_BASE, 366.0, 20000)
        m = g.inside_domain(xr, zr)
        xr, zr = xr[m], zr[m]
        tg = g.material_tag(xr, zr)
        for k in want:
            need = int(want[k] * n_ic) - len(got[k])
            if need > 0:
                sel = np.where(tg == k)[0][:need]
                got[k].extend(zip(xr[sel], zr[sel]))
    pts = np.array([p for k in want for p in got[k][: int(want[k] * n_ic)]])
    x, z = pts[:, 0], pts[:, 1]
    tag = g.material_tag(x, z)

    # area fractions, for the importance weights
    xa = rng.uniform(g.X_MIN, g.X_MAX, 400000)
    za = rng.uniform(g.Z_BASE, 366.0, 400000)
    ma = g.inside_domain(xa, za)
    ta = g.material_tag(xa[ma], za[ma])
    frac = {k: float((ta == k).mean()) for k in want}
    share = {k: float((tag == k).mean()) for k in want}
    w = np.array([frac[t] / share[t] for t in tag])
    w /= w.mean()

    psi0 = psi_initial(x, z)
    sv = b.sigma_v_geostatic(x, z)
    sh = b.sigma_h_geostatic(x, z)
    meta = dict(seed=seed, n_ic=len(x), Z_WT=b.Z_WT, Z_BASE=g.Z_BASE,
                X_MK_DIVIDE=g.X_MK_DIVIDE, K0=b.K0, K0_SMOOTH_W=b.K0_SMOOTH_W,
                area_fraction=frac, sample_share=share, geometry_sha=geometry_hash())
    np.savez_compressed(path, x=x, z=z, psi0=psi0, sig_v=sv, sig_h=sh,
                        w=w, tag=tag.astype("U4"), meta=json.dumps(meta))
    return path, meta


print("=== Step 2.3 initial conditions ===\n")

# --- seepage IC -----------------------------------------------------------
check("psi = 0 at the water table", abs(float(psi_initial(0.0, b.Z_WT))) < 1e-12,
      f"Z_WT = {b.Z_WT}")

path, meta = build_ic_cache()
d = np.load(path, allow_pickle=False)
x, z, psi0, sv, sh, w, tag = (d[k] for k in ("x", "z", "psi0", "sig_v", "sig_h", "w", "tag"))
check("cache non-empty", x.size > 100, f"n = {x.size}")

n_sat = int((psi0 > 0).sum())
check("saturated branch empty", n_sat == 0, f"{n_sat} saturated points (expected 0)")
# domain extremes, computed rather than assumed: the top boundary is F1, not
# the end of the digitised ground curve
_xg, _zg = np.meshgrid(np.linspace(g.X_MIN, g.X_MAX, 1200),
                       np.linspace(g.Z_BASE, 366.0, 1200))
_in = g.inside_domain(_xg.ravel(), _zg.ravel())
Z_TOP_DOMAIN = float(_zg.ravel()[_in].max())
PSI_MIN, PSI_MAX = b.Z_WT - Z_TOP_DOMAIN, b.Z_WT - g.Z_BASE
check("psi0 within domain bounds",
      psi0.min() >= PSI_MIN - 1e-9 and psi0.max() <= PSI_MAX + 1e-9,
      f"{psi0.min():.2f} .. {psi0.max():.2f} m  (bounds {PSI_MIN:.2f} .. {PSI_MAX:.2f})")
check("psi0 spans most of the range", psi0.min() < PSI_MIN + 5.0,
      f"z_top_domain = {Z_TOP_DOMAIN:.2f} m a.s.l.")

# IC must equal the F1 Dirichlet at t = 0
zf = np.linspace(g.Z_BASE, 358.0, 500)
check("IC == F1 Dirichlet at t=0",
      np.abs(psi_initial(g.x_f1(zf), zf) - b.psi_far_field(zf)).max() < 1e-10,
      f"max diff {np.abs(psi_initial(g.x_f1(zf), zf) - b.psi_far_field(zf)).max():.2e} m")

# total head uniform => Darcy flux identically zero => both no-flow BCs exact
h = psi0 + z
check("total head uniform (zero Darcy flux)", np.ptp(h) < 1e-9, f"ptp {np.ptp(h):.2e} m")

# --- mechanical IC --------------------------------------------------------
check("sigma_v >= 0", sv.min() >= 0.0, f"min {sv.min():.3f} Pa")
check("sigma_h < sigma_v", bool((sh < sv).all()), f"ratio {(sh/np.maximum(sv,1)).max():.4f}")

top = g.z_ground(x); ztm = np.minimum(g.z_tm_top(x), top)
far = np.abs(z - ztm) > b.K0_SMOOTH_W
bad = 0
for u in ("Mk", "Mk_d", "Tm"):
    k = (tag == u) & far & (sv > 1e3)
    if k.sum() > 10:
        bad += int((np.abs(sh[k]/sv[k] - b.K0[u]) > 1e-9).sum())
check("K0 exact outside smoothing band", bad == 0 and far.sum() > 100,
      f"{far.sum()} pts outside band, {bad} deviations")

# --- stratification and weights ------------------------------------------
check("all three units sampled", all((tag == u).sum() > 200 for u in ("Mk", "Mk_d", "Tm")),
      "  ".join(f"{u}:{(tag==u).sum()}" for u in ("Mk", "Mk_d", "Tm")))
check("weights normalised", abs(w.mean() - 1.0) < 1e-9, f"mean {w.mean():.9f}")

# --- provenance -----------------------------------------------------------
check("geometry hash recorded", meta["geometry_sha"] == geometry_hash(), meta["geometry_sha"])

print(f"\narea fractions  {meta['area_fraction']}")
print(f"sample shares   {meta['sample_share']}")
print(f"\n{'ALL IC CHECKS PASSED -- Step 2.3 seepage+mechanical IC complete' if N_FAIL == 0 else f'{N_FAIL} CHECK(S) FAILED'}")

import sys
sys.exit(0 if N_FAIL == 0 else 1)
