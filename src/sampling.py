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
    x: torch.Tensor
    z: torch.Tensor
    t: torch.Tensor
    w: torch.Tensor
    tag: np.ndarray
    nx: torch.Tensor | None = None      # outward unit normal, boundary sets only
    nz: torch.Tensor | None = None

    # Equilibrated initial state, attached by sigma0.attach_sigma0. Both are
    # dimensionless and TENSION POSITIVE (D-3.3.1). They are optional because
    # the hydraulic loss does not need them, and None is the honest default:
    # a zero sigma0 would look like a stress-free domain rather than an
    # unattached field.
    sigma0: torch.Tensor | None = None   # (N,3) [sxx*, szz*, sxz*]
    rho0: torch.Tensor | None = None     # (N,1) rho_0 / rho_b_ref
    sigma0_misses: int = 0               # points that used a nearest fallback

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
                    s: Scales = SCALES,
                    device: str = "cpu", dtype=torch.float64) -> Collocation:
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

    xt, zt, tt = as_inputs(x / s.L_ref, z / s.L_ref, t, device=device)
    wt = torch.as_tensor(w, dtype=xt.dtype, device=xt.device).reshape(-1, 1) 
    return Collocation(x=xt, z=zt, t=tt, w=wt, tag=tag)

def _segment_normal(seg: str, pts: np.ndarray) -> np.ndarray:
    """(N,2) outward unit normals, from PHYSICAL coordinates.

    Three segments follow the ground surface and need the differentiated
    normal; three are straight and have constant normals. Written out per
    segment rather than inferred, because a sign error here is invisible in
    the loss -- it turns an inflow condition into an outflow one and trains
    perfectly well.

    Unit vectors are scale-invariant, so no non-dimensionalisation is needed.
    But outward_normal differentiates z_ground with respect to x, so it must
    be handed physical x, not x/L_ref.
    """
    n = len(pts)
    if seg in ("natural_ground", "bench", "cut_face"):
        return bnd.outward_normal(g.z_ground, pts[:, 0])
    if seg == "base":            # domain floor, outward is -z
        return np.column_stack([np.zeros(n), -np.ones(n)])
    if seg == "pit_floor":       # vertical pit wall at x = 0, outward is -x
        return np.column_stack([-np.ones(n), np.zeros(n)])
    if seg == "far_field_f1":    # F1 plane, outward is +x
        return np.column_stack([np.ones(n), np.zeros(n)])
    raise ValueError(f"no outward normal defined for segment {seg!r}")

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
        nrm = _segment_normal(seg, pts)
        out[seg] = Collocation(
            x=xt, z=zt, t=tt, w=wt, tag=tag,
            nx=torch.as_tensor(nrm[:, 0], dtype=xt.dtype).reshape(-1, 1),
            nz=torch.as_tensor(nrm[:, 1], dtype=xt.dtype).reshape(-1, 1),
        )
    return out


def load_initial(path: str = "src/step21_geometry/ic_cache.npz",
                 s: Scales = SCALES, device: str = "cpu"):
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

    xt, zt, tt = as_inputs(x / s.L_ref, z / s.L_ref, np.zeros(len(x)), device=device)
    wt = torch.as_tensor(w / w.mean(), dtype=xt.dtype, device=xt.device).reshape(-1, 1)
    targets = {k: np.asarray(d[k]) for k in d.files
               if k not in ("x", "z", "w", "tag", "meta")}
    return Collocation(x=xt, z=zt, t=tt, w=wt, tag=tag), targets


# ===========================================================================
# Interface sampling -- the Mk|Tm and Mk_d|Tm contacts along z_tm_top(x).
#
# Mk|Mk_d is NOT sampled: it carries no flux discontinuity while the two marls
# share K_s, alpha and n (Phase-1 `vg_basis: "marl analog"`).
# `test_mk_and_mk_d_are_hydraulically_identical` guards that assumption; if it
# ever fails, a third contact belongs here.
#
# Tm is always the LOWER material at this contact, so the dict key carries both
# sides -- "mk_tm" means Mk above, Tm below -- and `Collocation.tag` holds the
# UPPER material only. Nothing downstream should infer the lower side from
# geometry; it is Tm by construction of this sampler.
# ===========================================================================

INTERFACE_EPS_M = 0.10      # m, offset used to identify the material each side
_TRACE_DX_M = 0.01          # m, central-difference step for the trace slope

CONTACTS = {"mk_tm": "Mk", "mkd_tm": "Mk_d"}



# --- ADAPTER -------------------------------------------------------------
# `sample_boundary` already builds Collocation tensors somehow. If it has
# equivalents of these two, DELETE these and call those instead -- two ways of
# making a leaf tensor in one module is how dtype drift starts.
def _iface_leaf(a):
    """(N,) float array -> (N,1) leaf tensor requiring grad."""
    return torch.as_tensor(np.asarray(a, float), dtype=torch.float64
                           ).reshape(-1, 1).requires_grad_(True)


def _iface_col(a):
    """(N,) float array -> (N,1) tensor, no grad."""
    return torch.as_tensor(np.asarray(a, float), dtype=torch.float64).reshape(-1, 1)


def _tm_top_slope(x_phys):
    """dz/dx of the top-of-Tm trace, central difference (physical metres).

    The trace is piecewise-linear between digitised points, so this is exact
    away from the nodes and averages the two limbs at them. It is discontinuous
    at x = 90/91 where `d_mk_tm` and `d_mkd_base` tile (finding 4); that seam
    sits inside the Mk|Tm contact, ~0.7 m from X_MK_DIVIDE.
    """
    x = np.asarray(x_phys, float)
    h = _TRACE_DX_M
    return (g.z_tm_top(x + h) - g.z_tm_top(x - h)) / (2.0 * h)


def _tm_top_normal(x_phys):
    """Unit normal to z_tm_top, pointing UP -- out of Tm, into the marl.

    Returns (nx, nz). Sign convention matters: `interface_flux_jump` compares
    q.n evaluated with the SAME normal on both sides, so flipping this flips
    the sign of the jump.
    """
    dz = _tm_top_slope(x_phys)
    norm = np.hypot(dz, 1.0)
    return -dz / norm, np.ones_like(dz) / norm


def _arclength_x(f, x0, x1, n, rng, m=2000):
    """Sample x so points are uniform along ARC LENGTH of z = f(x).

    Uniform-in-x would under-resolve the steep limbs of the contact. Same
    approach as `boundaries._arclength_sample`, but returns x only.
    """
    xs = np.linspace(x0, x1, m)
    zs = f(xs)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(zs)))])
    return np.interp(rng.uniform(0.0, s[-1], n), s, xs)


def _contact_x_range(upper, margin=1.0):
    """[x0, x1] over which this contact exists inside the domain.

    Excludes x where the contact has risen to within `margin` of the ground
    surface (it does not outcrop in Section 5, but the digitised traces
    approach each other near the pit) and where it has passed beyond F1.
    """
    lo = g.X_MIN if upper == "Mk" else g.X_MK_DIVIDE
    hi = g.X_MK_DIVIDE if upper == "Mk" else g.X_MAX

    xs = np.linspace(lo, hi, 4000)
    zt = g.z_tm_top(xs)
    ok = (zt < g.z_ground(xs) - margin) & (zt >= g.Z_BASE) & (xs <= g.x_f1(zt))
    if not ok.any():
        raise ValueError(f"no valid {upper}|Tm contact in x = [{lo}, {hi}]")
    return float(xs[ok].min()), float(xs[ok].max())


def sample_interfaces(n: int = 500, seed: int = 0, t_max: float = T_MAX_DAYS,
                      s: Scales = SCALES) -> dict[str, Collocation]:
    """Collocation sets on the Mk|Tm and Mk_d|Tm contacts.

    Returns {"mk_tm": Collocation, "mkd_tm": Collocation}, keyed by contact to
    match `sample_boundary`, so `L_interface` can report per-contact parts.

    `n` is per contact. Geometry is queried ONCE, here: tags and normals travel
    with the Collocation and must never be recomputed in the torch layer, where
    the x -> x/L_ref -> x round trip moves points by ~1e-16 and `material_tag`
    starts returning "outside" on curved traces.
    """
    rng = np.random.default_rng(seed)
    out = {}

    for key, upper in CONTACTS.items():
        x0, x1 = _contact_x_range(upper)
        x = _arclength_x(g.z_tm_top, x0, x1, n, rng)
        z = g.z_tm_top(x)

        # Identify the material each side. This is the check that the trace and
        # `material_tag` agree; disagreement means the tiling seam or
        # X_MK_DIVIDE has moved, not that a point is merely awkward.
        eps = INTERFACE_EPS_M
        tag_up = g.material_tag(x, z + eps)
        tag_dn = g.material_tag(x, z - eps)
        keep = (tag_up == upper) & (tag_dn == "Tm")
        if keep.mean() < 0.98:
            raise ValueError(
                f"{key}: only {100 * keep.mean():.1f}% of sampled points have "
                f"{upper} above and Tm below. The trace and material_tag "
                f"disagree -- check X_MK_DIVIDE against the d_mk_tm/d_mkd_base "
                f"tiling at x = 90/91."
            )
        x, z = x[keep], z[keep]

        nx, nz = _tm_top_normal(x)
        t = rng.uniform(0.0, t_max, len(x))
        w = np.ones(len(x))          # arc-length sampling already sets density

        # t_max is in days and T_ref = 86400 s, so t* = t_days directly.
        out[key] = Collocation(
            x=_iface_leaf(x / s.L_ref),
            z=_iface_leaf(z / s.L_ref),
            t=_iface_leaf(t),
            w=_iface_col(w),
            tag=np.full(len(x), upper),
            nx=_iface_col(nx),
            nz=_iface_col(nz),
        )

    return out


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

    
def to_device(coll, device):
    """Move a Collocation, preserving leaf status on x/z/t.

    .to() on a leaf with requires_grad returns a NON-leaf, and
    torch.autograd.grad(psi, x) then resolves to the wrong node -- silently,
    with no exception. Rebuild the leaves instead. `tag` stays numpy.
    """
    import dataclasses
    def leaf(v):
        return None if v is None else \
            v.detach().to(device).requires_grad_(v.requires_grad)
    def plain(v):
        return None if v is None else v.detach().to(device)
    return dataclasses.replace(
        coll, x=leaf(coll.x), z=leaf(coll.z), t=leaf(coll.t),
        w=plain(coll.w), nx=plain(coll.nx), nz=plain(coll.nz),
        sigma0=plain(coll.sigma0), rho0=plain(coll.rho0))    
