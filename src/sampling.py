"""
src/sampling.py — collocation point generation.
================================================

Produces the point sets the loss function integrates over: interior points for
L_PDE, boundary points for L_BC, and the stored t = 0 state for L_IC.

THE UNIT BOUNDARY. `geometry.py` and `boundaries.py` work in PHYSICAL metres
and metres a.s.l. The network and `residuals.py` work in DIMENSIONLESS
coordinates. This module is where the conversion happens: sample in physical
space (because `inside_domain` and `material_tag` are physical), convert once
on the way out. Nothing downstream of here sees metres.

Deliberately NOT sourced from `config.BOUNDS`. That dict carries
`z_min = 0.0, z_max = 1.18`, i.e. elevation 0-200 m — the region BELOW the
domain floor. Its own TODO says to source these from geometry once Z_BASE was
settled; it is settled, so this module reads geometry directly and the stale
constants are left alone rather than propagated.

STRATIFICATION. Mk_d is 9.7% of the domain by area but is the unit that
fails (sigma_ci 4.29 vs 17.9 MPa). Uniform sampling would put ~10% of the
collocation budget on the physics that decides the answer. So points are
stratified by material tag on the same 0.25/0.35/0.40 shares as `ic_cache`,
and each point carries a weight w = area_fraction / sample_share so that
`(w * residual**2).mean()` remains an unbiased estimate of the domain
integral. Use the weights or the oversampling silently biases the loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from src.derivatives import as_inputs
from src.nondim import SCALES, Scales

try:
    import geometry as g
    import boundaries as bnd
except ModuleNotFoundError:
    from src.step21_geometry import boundaries as bnd
    from src.step21_geometry import geometry as g

__all__ = [
    "Collocation",
    "SAMPLE_SHARES",
    "domain_bbox",
    "sample_interior",
    "sample_boundary",
    "load_initial",
]

# Matches ic_cache.npz (seed 20260813). Keep in step or L_IC and L_PDE
# integrate the domain with different weightings.
SAMPLE_SHARES = {"Mk": 0.25, "Mk_d": 0.35, "Tm": 0.40}

T_MAX_DAYS = 30.0       # t* upper bound; the rainfall window, still contested


@dataclass
class Collocation:
    """A point set in DIMENSIONLESS coordinates, ready for residuals.py.

    x, z, t : leaf tensors (N, 1), requires_grad=True
    w       : per-point weight (N, 1), mean 1.0, no grad
    tag     : numpy array of material tags, for diagnostics and per-unit loss
    """
    x: Tensor
    z: Tensor
    t: Tensor
    w: Tensor
    tag: np.ndarray

    def __len__(self) -> int:
        return self.x.shape[0]

    def counts(self) -> dict[str, int]:
        u, c = np.unique(self.tag, return_counts=True)
        return {str(k): int(v) for k, v in zip(u, c)}


def _lhs(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Latin hypercube on [0,1]^d. Five lines; avoids a scipy dependency.

    Note: rejection sampling downstream destroys the strict stratification.
    LHS still helps — it prevents the clumping plain uniform sampling gives
    at these counts — but do not claim strict LHS coverage in the methods
    section for the interior set. The boundary sets, which are not rejected,
    keep it.
    """
    cut = np.linspace(0.0, 1.0, n + 1)
    out = np.empty((n, d))
    for j in range(d):
        u = rng.uniform(cut[:n], cut[1:])
        out[:, j] = rng.permutation(u)
    return out


def domain_bbox(n_scan: int = 400) -> tuple[float, float, float, float]:
    """(x_min, x_max, z_min, z_max) of the domain, found by scanning.

    Derived rather than hard-coded: the F1 far-field boundary is inclined
    (x_f1(z)), so there is no single x_max constant to import, and the ground
    surface is an interpolant. Scanning `inside_domain` is slower than reading
    a constant but cannot go stale when the geometry is revised.
    """
    xs = np.linspace(-50.0, 400.0, n_scan)
    zs = np.linspace(g.Z_BASE - 5.0, g.Z_BASE + 250.0, n_scan)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    inside = np.asarray(g.inside_domain(X.ravel(), Z.ravel())).reshape(X.shape)
    if not inside.any():
        raise RuntimeError("domain_bbox(): inside_domain() is False everywhere "
                           "over the scan window. Check the scan range.")
    xi, zi = np.where(inside)
    pad_x = (xs[1] - xs[0])
    pad_z = (zs[1] - zs[0])
    return (float(xs[xi.min()] - pad_x), float(xs[xi.max()] + pad_x),
            float(zs[zi.min()] - pad_z), float(zs[zi.max()] + pad_z))


def _rejection_sample(n_target: int, bbox, rng, max_rounds: int = 40):
    """Uniform-in-domain (x, z) by LHS over the bbox plus rejection."""
    x0, x1, z0, z1 = bbox
    xs, zs = [], []
    got = 0
    for _ in range(max_rounds):
        n_draw = max(1024, int((n_target - got) * 3))
        u = _lhs(n_draw, 2, rng)
        x = x0 + u[:, 0] * (x1 - x0)
        z = z0 + u[:, 1] * (z1 - z0)
        keep = np.asarray(g.inside_domain(x, z), dtype=bool)
        xs.append(x[keep])
        zs.append(z[keep])
        got += int(keep.sum())
        if got >= n_target:
            break
    else:
        raise RuntimeError(f"_rejection_sample(): only {got}/{n_target} points "
                           "after max_rounds. Is the bbox far too large?")
    return np.concatenate(xs)[:n_target], np.concatenate(zs)[:n_target]


def sample_interior(n: int = 20000, seed: int = 20260808,
                    stratify: bool = True, t_max: float = T_MAX_DAYS,
                    s: Scales = SCALES) -> Collocation:
    """Interior collocation points for L_PDE, dimensionless, weighted.

    Returns points whose material composition follows SAMPLE_SHARES, with
    weights restoring the true area fractions. Set stratify=False for plain
    area-proportional sampling (weights all 1.0) — useful as a check that
    the weighting is doing what it claims.
    """
    rng = np.random.default_rng(seed)
    bbox = domain_bbox()

    # Oversample once; use it both to estimate area fractions and to draw from.
    pool_x, pool_z = _rejection_sample(max(20 * n, 40000), bbox, rng)
    pool_tag = np.asarray(g.material_tag(pool_x, pool_z)).astype(str)

    tags = sorted(SAMPLE_SHARES)
    area_frac = {k: float((pool_tag == k).mean()) for k in tags}
    unknown = set(np.unique(pool_tag)) - set(tags)
    if unknown:
        raise ValueError(f"material_tag returned unexpected tags: {unknown}")

    xs, zs, ws, tg = [], [], [], []
    for k in tags:
        share = SAMPLE_SHARES[k] if stratify else area_frac[k]
        n_k = int(round(n * share))
        if n_k == 0:
            continue
        idx = np.where(pool_tag == k)[0]
        if len(idx) < n_k:
            raise RuntimeError(f"pool has {len(idx)} '{k}' points, need {n_k}. "
                               "Raise the pool multiplier in sample_interior.")
        pick = rng.choice(idx, size=n_k, replace=False)
        xs.append(pool_x[pick])
        zs.append(pool_z[pick])
        ws.append(np.full(n_k, area_frac[k] / share))
        tg.append(np.full(n_k, k))

    x = np.concatenate(xs)
    z = np.concatenate(zs)
    w = np.concatenate(ws)
    tag = np.concatenate(tg)

    order = rng.permutation(len(x))          # shuffle; some optimisers batch
    x, z, w, tag = x[order], z[order], w[order], tag[order]

    t = rng.uniform(0.0, t_max, size=len(x))  # t is ALREADY dimensionless (days)
    w = w / w.mean()                          # mean 1, so loss magnitudes stay readable

    xt, zt, tt = as_inputs(x / s.L_ref, z / s.L_ref, t)
    wt = torch.as_tensor(w, dtype=xt.dtype).reshape(-1, 1)
    return Collocation(x=xt, z=zt, t=tt, w=wt, tag=tag)


def sample_boundary(n: int = 2000, seed: int = 0, t_max: float = T_MAX_DAYS,
                    s: Scales = SCALES) -> dict[str, Collocation]:
    """Boundary points per segment, from boundaries.sample_boundaries().

    Delegates rather than reimplements: `sample_boundaries` already does
    arclength-proportional sampling on the six segments and is covered by
    11_bc_checks.py. This only converts units and attaches a time coordinate.
    """
    raw = bnd.sample_boundaries(n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    out: dict[str, Collocation] = {}
    for seg, pts in raw.items():
        pts = np.asarray(pts, dtype=float)
        t = rng.uniform(0.0, t_max, size=len(pts))
        xt, zt, tt = as_inputs(pts[:, 0] / s.L_ref, pts[:, 1] / s.L_ref, t)
        wt = torch.ones_like(xt)
        tag = np.asarray(g.material_tag(pts[:, 0], pts[:, 1])).astype(str)
        out[seg] = Collocation(x=xt, z=zt, t=tt, w=wt, tag=tag)
    return out


def load_initial(path: str = "src/step21_geometry/ic_cache.npz",
                 s: Scales = SCALES):
    """The stored t = 0 state for L_IC. Regenerate with 14_ic_checks.py.

    The cache is gitignored, so a fresh clone will not have it. It is
    deterministic (seed 20260813, geometry hash recorded in `meta`), so
    regenerating is the intended path rather than shipping the binary.

    Returns (Collocation at t*=0, dict of the stored target fields).
    """
    import pathlib
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Regenerate it:\n"
            "  cd src/step21_geometry && python 14_ic_checks.py")

    d = np.load(p, allow_pickle=True)
    x, z = np.asarray(d["x"], float), np.asarray(d["z"], float)
    w = np.asarray(d["w"], float) if "w" in d else np.ones(len(x))
    tag = np.asarray(d["tag"]).astype(str) if "tag" in d else \
        np.asarray(g.material_tag(x, z)).astype(str)

    xt, zt, tt = as_inputs(x / s.L_ref, z / s.L_ref, np.zeros(len(x)))
    wt = torch.as_tensor(w / w.mean(), dtype=xt.dtype).reshape(-1, 1)
    targets = {k: np.asarray(d[k]) for k in d.files
               if k not in ("x", "z", "w", "tag", "meta")}
    return Collocation(x=xt, z=zt, t=tt, w=wt, tag=tag), targets


if __name__ == "__main__":
    print("bbox (x0, x1, z0, z1):", tuple(round(v, 2) for v in domain_bbox()))
    c = sample_interior(5000)
    print(f"interior n = {len(c)}  counts = {c.counts()}")
    print(f"weights: mean {c.w.mean():.6f}  min {c.w.min():.4f}  "
          f"max {c.w.max():.4f}")
    print(f"x* range {c.x.min():.4f}..{c.x.max():.4f}   "
          f"z* range {c.z.min():.4f}..{c.z.max():.4f}   "
          f"t* range {c.t.min():.2f}..{c.t.max():.2f}")
    b = sample_boundary(600)
    print("boundary segments:", {k: len(v) for k, v in b.items()})