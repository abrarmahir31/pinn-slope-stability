"""src/sampling_front.py — collocation stratified by the variable that decides
whether the Richards residual carries information.

WHY THIS EXISTS. `sampling.sample_interior` stratifies by MATERIAL TAG, on
shares 0.25 / 0.35 / 0.40 chosen because Mk_d is the unit that fails
mechanically (sigma_ci 4.29 vs 17.9 MPa). That is the right stratification for
`L_PDE_mech` and it is close to the worst one for `L_PDE`: `scripts/
inert_fraction.py` shows 94.7% of a 10,000-point production draw lands where
neither capillary diffusion nor gravity drainage can move information 0.1 m in
the whole 30-day window, so the Richards residual there is C* dpsi*/dt* = 0 and
any frozen field satisfies it exactly. Oversampling Mk_d by 3.6x spends more of
the budget on the deadest part of the hydraulic problem.

The two residuals want opposite point sets and currently share one
(`scripts/train.py:92` makes a single `coll` and passes it to both
`L_PDE` and `L_PDE_mech`). This module provides the hydraulic one.

THE STRATIFYING VARIABLE. K* varies by six orders across the domain and is a
function of psi alone; on the initial condition psi_0 = -(z - Z_WT), so the
initial suction is a pure function of elevation. Every quantity that decides
liveness -- K*, C*, the diffusivity Pi_R_diff K*/C*, the celerity
Pi_R_grav (dK*/dpsi*)/C* -- is monotone in it. Log-spaced bands in
(z - Z_WT) therefore give roughly equal K*-decades per band, which is the
stratification the problem actually has. Linear-in-z bands would put four
bands inside one decade of K* near the floor and none across the six decades
above it.

A SECOND BAND AT THE GROUND SURFACE is needed and does not follow from
elevation. The rain BC applies q = min(rain.n, K_s) (capacity-limited,
`boundaries.flux_bc`), and for the matrix to conduct K_s the surface must go to
psi ~ 0 whatever its elevation. That boundary layer is the only live region in
Mk and Mk_d -- their front advances 8-10 mm in 30 days -- and elevation bands
cannot see it because the ground surface spans z = 271 to 358 m.

WEIGHTS AND WHAT THEY MEAN. Each point carries w = area_fraction / sample_share
exactly as `sample_interior` does, so `(w * R**2).mean()` remains an unbiased
estimate of the same domain integral and the loss VALUE is comparable across
samplers. This is importance sampling: same objective, lower-variance
estimator. It is NOT a reweighting of the physics, and `--no-restore-weights`
(w = 1) is a genuinely different objective -- the residual integrated against
the sampling density instead of against area. That second option is defensible
for a PINN (the minimiser is the same at R = 0, and finite capacity has to be
spent somewhere) but it is a modelling choice and must not be made by accident.
Default is the unbiased one.

WHAT THIS DOES NOT DO. It does not follow the Tm wetting front, which descends
at q/dtheta = 18.1 m/day and crosses the whole domain in ~9 days, so the live
region is time-dependent and this sampler is static in t. Doing that properly
needs the adaptive half of the design, and the adaptive criterion must NOT be
|R|: in the inert zone |R| is the storage term and is LARGER (6e-4 in Mk) than
in live Tm (1.9e-6), so residual-based refinement would refine into the dead
region. The criterion is the flux fraction max(|diffusion|, |gravity|) /
|storage| -- the same quantity that fixes `inspect_richards_terms.py`'s
verdict labels.
"""
from __future__ import annotations

import numpy as np
import torch

from src.derivatives import as_inputs
from src.nondim import SCALES, Scales
from src.sampling import Collocation, T_MAX_DAYS, _rejection_sample, domain_bbox

try:
    import geometry as g
except ModuleNotFoundError:
    from src.step21_geometry import geometry as g

__all__ = ["Z_WT", "SUCTION_EDGES", "SURFACE_BAND_M",
           "band_of", "sample_interior_front"]

Z_WT = 197.0            # vg.psi_initial default; 3 m BELOW the domain floor

# Edges in (z - Z_WT) metres, i.e. in initial suction. The domain spans
# z = 200..358, so this is 3..161 m. Log-spaced: each band is about half a
# decade of suction and, through van Genuchten, one to two decades of K*.
SUCTION_EDGES = (3.0, 6.0, 12.0, 30.0, 75.0, 165.0)

# Depth below the ground surface, metres, within which a point is assigned to
# the surface band regardless of elevation. 2.0 m is ~6x the 0.31 m saturated
# 30-day diffusion length in the marls, so it brackets the infiltration
# boundary layer with margin rather than resolving it exactly.
SURFACE_BAND_M = 2.0


def band_of(x_phys, z_phys):
    """Band index per point: 0 = surface layer, 1..len(SUCTION_EDGES)-1 = suction bands.

    The surface band takes precedence: a point 1 m below the crest is in the
    infiltration boundary layer even though its initial suction is 161 m.
    """
    x = np.asarray(x_phys, float)
    z = np.asarray(z_phys, float)
    depth = g.z_ground(x) - z
    suction = np.clip(z - Z_WT, SUCTION_EDGES[0], SUCTION_EDGES[-1] - 1e-9)
    b = np.digitize(suction, np.asarray(SUCTION_EDGES[1:-1]), right=False) + 1
    return np.where(depth <= SURFACE_BAND_M, 0, b)


N_BANDS = len(SUCTION_EDGES)          # 1 surface band + (len(EDGES) - 1) suction bands


def sample_interior_front(n: int = 10000, seed: int = 20260808,
                          shares: np.ndarray | None = None,
                          t_max: float = T_MAX_DAYS,
                          s: Scales = SCALES,
                          restore_weights: bool = True,
                          device: str = "cpu") -> Collocation:
    """Interior points stratified by initial suction, plus a surface band.

    `shares` is a length-N_BANDS vector summing to 1: [surface, then one per
    suction band, shallowest first]. Default is uniform across bands, which
    already lifts the live fraction by more than an order of magnitude; tune it
    once the adaptive half exists rather than by hand now.

    Weights restore the true area fractions unless `restore_weights=False`.
    See the module docstring -- that flag changes the objective, not the
    estimator.
    """
    rng = np.random.default_rng(seed)
    bbox = domain_bbox()

    if shares is None:
        shares = np.full(N_BANDS, 1.0 / N_BANDS)
    shares = np.asarray(shares, float)
    if shares.shape != (N_BANDS,):
        raise ValueError(f"shares must have length {N_BANDS}, got {shares.shape}")
    if not np.isclose(shares.sum(), 1.0):
        raise ValueError(f"shares must sum to 1, got {shares.sum()}")

    # One large uniform-in-area pool: it both estimates the band area fractions
    # and supplies the draws, exactly as sample_interior does. The pool must be
    # big enough that the RAREST band still has n * share members -- the
    # surface band is ~1% of the area, so the multiplier is larger here.
    pool_x, pool_z = _rejection_sample(max(60 * n, 200_000), bbox, rng)
    pool_band = band_of(pool_x, pool_z)
    pool_tag = np.asarray(g.material_tag(pool_x, pool_z)).astype(str)

    area_frac = np.array([(pool_band == b).mean() for b in range(N_BANDS)])

    xs, zs, ws, tg = [], [], [], []
    for b in range(N_BANDS):
        if shares[b] <= 0 or area_frac[b] == 0.0:
            continue
        n_b = int(round(n * shares[b]))
        idx = np.where(pool_band == b)[0]
        if len(idx) < n_b:
            raise RuntimeError(
                f"band {b} has {len(idx)} pool points, need {n_b}. Raise the "
                "pool multiplier in sample_interior_front, or lower its share.")
        pick = rng.choice(idx, size=n_b, replace=False)
        xs.append(pool_x[pick])
        zs.append(pool_z[pick])
        ws.append(np.full(n_b, area_frac[b] / shares[b] if restore_weights
                          else 1.0))
        tg.append(pool_tag[pick])

    x = np.concatenate(xs)
    z = np.concatenate(zs)
    w = np.concatenate(ws)
    tag = np.concatenate(tg)

    order = rng.permutation(len(x))
    x, z, w, tag = x[order], z[order], w[order], tag[order]

    t = rng.uniform(0.0, t_max, size=len(x))
    w = w / w.mean()

    xt, zt, tt = as_inputs(x / s.L_ref, z / s.L_ref, t, device=device)
    wt = torch.as_tensor(w, dtype=xt.dtype, device=xt.device).reshape(-1, 1)
    return Collocation(x=xt, z=zt, t=tt, w=wt, tag=tag)


if __name__ == "__main__":
    from src.sampling import sample_interior

    for name, c in (("uniform", sample_interior(5000, 20260808)),
                    ("front",   sample_interior_front(5000, 20260808))):
        zp = c.z.detach().reshape(-1).numpy() * SCALES.L_ref
        xp = c.x.detach().reshape(-1).numpy() * SCALES.L_ref
        b = band_of(xp, zp)
        print(f"{name:>8}: counts {c.counts()}  "
              f"w in [{c.w.min():.3g}, {c.w.max():.3g}]")
        print(f"          bands  "
              + "  ".join(f"{i}:{int((b == i).sum())}" for i in range(N_BANDS)))