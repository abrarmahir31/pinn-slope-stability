"""
src/sigma0.py — the equilibrated initial stress, loaded and attached.
=====================================================================

Step 3.3a, second half. `fe_gravity.py` solves for sigma_0 and
`17_sigma0_solve.py` caches it; this module is the only thing that knows the
cache's file format, in the same way `sampling.load_initial` is the only thing
that knows `ic_cache.npz`.

WHY THIS ATTACHES AT SAMPLE TIME
--------------------------------
sigma_0 is a geometry query, and geometry queries happen ONCE, when the
Collocation is built, and then travel with it. The x -> x/L_ref -> x round
trip moves points by ~1e-16, which is enough to push a boundary point outside
the mesh and get NaN back. Doing the lookup inside the residual would put that
round trip in the training loop, several thousand times per epoch.

TWO FIELDS, NOT ONE
-------------------
`mechanical_residual` in its collapsed (equilibrated-sigma_0) form needs BOTH
sigma_0 and the density that sigma_0 was equilibrated against:

    div* dsigma*  +  Pi_M_body * (rho_b - rho_0)/rho_b_ref * e_z  =  0

If rho_0 defaults to 1.0 while the FE solve used the wet profile
rho_dry + theta(psi_0)*rho_w, the two do not cancel and the residual carries a
spurious body force of order (rho_0 - rho_b_ref)/rho_b_ref -- up to 0.48 for
Tm. So `attach_sigma0` sets `coll.rho0` as well, and it recomputes it from
psi_initial rather than interpolating it, because that is exact.

Sign convention: TENSION POSITIVE (D-3.3.1), inherited from the cache.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch

from src.nondim import SCALES, Scales

__all__ = ["CACHE_PATH", "load_sigma0_cache", "sigma0_field", "rho0_field",
           "attach_sigma0"]

CACHE_PATH = "src/step21_geometry/sigma0_cache.npz"


def load_sigma0_cache(path: str = CACHE_PATH):
    """Raw cache contents. Regenerate with 17_sigma0_solve.py.

    Gitignored and deterministic, like ic_cache.npz — regenerating is the
    intended path rather than shipping the binary.
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Regenerate it:\n"
            "  cd src/step21_geometry && python 17_sigma0_solve.py"
        )
    d = np.load(p, allow_pickle=True)
    meta = [str(m) for m in d["meta"]]
    if "tension_positive" not in meta:
        raise ValueError(
            f"{p} does not declare the tension-positive convention "
            f"(meta={meta}). Refusing to guess — see D-3.3.1."
        )
    return d


def sigma0_field(cache=None, path: str = CACHE_PATH):
    """Callable (x_phys, z_phys) -> (N,3) [sxx, szz, sxz] in Pa.

    Linear interpolation over the FE triangulation. Points that fall outside
    the mesh fall back to the nearest element's constant stress, and the
    number of such points is returned by `attach_sigma0` rather than being
    swallowed: a handful at the boundary is the ~1e-14 rounding this project
    already knows about, while hundreds means the mesh and the sampler
    disagree about where the domain is.
    """
    from matplotlib.tri import LinearTriInterpolator, Triangulation
    from scipy.spatial import cKDTree

    d = cache if cache is not None else load_sigma0_cache(path)
    nodes, tris = d["nodes"], d["tris"]
    sig_n, sig_e = d["sigma_n"], d["sigma_e"]

    tri = Triangulation(nodes[:, 0], nodes[:, 1], tris)
    interps = [LinearTriInterpolator(tri, sig_n[:, k]) for k in range(3)]
    centroids = nodes[tris].mean(axis=1)
    kd = cKDTree(centroids)

    def field(x, z):
        x = np.asarray(x, float).ravel()
        z = np.asarray(z, float).ravel()
        out = np.empty((x.size, 3))
        miss = np.zeros(x.size, dtype=bool)
        for k, f in enumerate(interps):
            v = f(x, z)
            m = np.ma.getmaskarray(v)
            out[:, k] = np.ma.filled(v, 0.0)
            miss |= m
        if miss.any():
            _, j = kd.query(np.column_stack([x[miss], z[miss]]))
            out[miss] = sig_e[j]
        return out, int(miss.sum())

    return field


def rho0_field(path_geom: str | None = None):
    """Callable (x_phys, z_phys) -> (N,) kg/m3, the wet bulk density the FE
    solve was equilibrated against. Recomputed, not interpolated."""
    import sys
    root = pathlib.Path(__file__).resolve().parents[1]
    for p in (str(root), str(root / "src" / "step21_geometry")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import geometry as g
    from initial import psi_initial
    from src import materials as mt

    mats = mt.load_materials()

    def field(x, z):
        x = np.asarray(x, float).ravel()
        z = np.asarray(z, float).ravel()
        psi = np.asarray(psi_initial(x, z), float).ravel()
        tag = np.asarray(g.material_tag(x, z)).astype(str)
        rho = np.full(x.size, np.nan)
        for k, m in mats.items():
            sel = tag == k
            if sel.any():
                rho[sel] = m.rho_dry + mt.theta(psi[sel], m) * 1000.0

        # `material_tag` returns "outside" for a handful of boundary points
        # even after the inward nudge -- e.g. sample_boundary puts pit_floor's
        # topmost point marginally above z_ground. Nearest in-domain neighbour
        # rather than a raise: rho_0 is a smooth field and these points are
        # centimetres from a valid one, so refusing would block the mechanical
        # BCs over a sampler rounding artefact. Anything beyond a few percent
        # is a different problem and does raise.
        bad = ~np.isfinite(rho)
        if bad.any():
            if bad.mean() > 0.05:
                raise ValueError(
                    f"rho0_field: {bad.sum()}/{bad.size} points have no "
                    f"material ({sorted(set(np.unique(tag)) - set(mats))}). "
                    f"More than 5% means the sampler and the geometry "
                    f"disagree, not that a few points rounded out."
                )
            from scipy.spatial import cKDTree
            good = ~bad
            if not good.any():
                raise ValueError("rho0_field: no point has a material tag")
            _, j = cKDTree(np.column_stack([x[good], z[good]])).query(
                np.column_stack([x[bad], z[bad]]))
            rho[bad] = rho[good][j]
        return rho

    return field


def attach_sigma0(coll, sig_fn=None, rho_fn=None, s: Scales = SCALES,
                  max_miss_frac: float = 0.01, nudge_m: float = 0.25):
    """Return `coll` with `sigma0` (N,3) and `rho0` (N,1) set, dimensionless.

    `coll.x, coll.z` are dimensionless; this multiplies back up by L_ref for
    the lookup, which is the single place that conversion happens, exactly as
    `swcc_from_material` is the single place psi* goes back to metres.

    BOUNDARY SETS ARE NUDGED INWARD by `nudge_m` along -n before the lookup.
    Boundary points lie exactly ON the FE mesh edge, and the point-in-triangle
    test rejects them: 47.8% of boundary points miss. That is NOT chord sag --
    refining the mesh 4x moves it to 47.4% -- it is the edge case itself. A
    0.25 m offset takes the miss rate to 0.08%.

    The cost is that sigma_0 is read 0.25 m inside the surface rather than on
    it. sigma_0 is continuous, so the error is about rho*g*0.25 = 5 kPa
    against a sigma_v that reaches 5.9 MPa -- 0.08%. It is largest in relative
    terms at the crest, where sigma_v itself goes to zero; that is also where
    the traction-free condition is least load-bearing.

    Applied only when the Collocation carries normals, so interior sets are
    untouched and their numbers do not move.
    """
    sig_fn = sig_fn if sig_fn is not None else sigma0_field()
    rho_fn = rho_fn if rho_fn is not None else rho0_field()

    xp = coll.x.detach().cpu().numpy().ravel() * s.L_ref
    zp = coll.z.detach().cpu().numpy().ravel() * s.L_ref

    if coll.nx is not None and coll.nz is not None and nudge_m:
        xp = xp - nudge_m * coll.nx.detach().cpu().numpy().ravel()
        zp = zp - nudge_m * coll.nz.detach().cpu().numpy().ravel()

    sig_pa, n_miss = sig_fn(xp, zp)
    if n_miss > max_miss_frac * len(xp):
        raise ValueError(
            f"attach_sigma0: {n_miss}/{len(xp)} points fell outside the FE "
            f"mesh and used a nearest-element fallback. That is more than "
            f"{max_miss_frac:.0%} and means the sampler and the mesh disagree "
            f"about the domain, not that a few boundary points rounded out."
        )

    kw = dict(dtype=coll.x.dtype, device=coll.x.device)
    coll.sigma0 = torch.as_tensor(sig_pa / s.sig_ref, **kw)
    coll.rho0 = torch.as_tensor(
        rho_fn(xp, zp) / s.rho_b_ref, **kw).reshape(-1, 1)
    coll.sigma0_misses = n_miss
    return coll