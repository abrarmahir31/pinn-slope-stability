"""Failure-surface extraction from the converged critical state (Step 5.3).

The premise of SSR failure-surface extraction: at the critical SRF the
displacement field localises, and the locus of peak maximum-shear-strain IS the
slip surface. No trial circles, no search over candidate surfaces. That is the
main practical advantage of the method over limit equilibrium and it is worth
stating in the thesis that way.

This module is forward-pass only -- it evaluates a converged network on a fixed
post-processing grid and does image-processing on the result. It is cheap, it
runs on a laptop, and it never needs a gradient step.

Shear strain convention, since `mechanics.strain_star` warns about exactly this
factor of two: `eps_xz` there is the TENSOR component, 0.5*(du/dz + dv/dx).
The maximum ENGINEERING shear strain is the difference of principal strains,

    gamma_max = eps_1 - eps_3 = sqrt((eps_xx - eps_zz)**2 + 4 * eps_xz**2)

so the 4 belongs there when `eps_xz` is the tensor component and would be a 1
if it were engineering gamma_xz. A reader checking against a gamma-based
textbook will see the discrepancy; this sentence is here so they find it before
they find a bug that is not there.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1. Strain invariant
# ---------------------------------------------------------------------------
def gamma_max(eps_xx, eps_zz, eps_xz):
    """Maximum engineering shear strain from TENSOR strain components.

    Frame-invariant by construction: it is the diameter of the strain Mohr
    circle, so a rigid-body rotation gives exactly zero.
    `test_gamma_max_vanishes_under_rigid_rotation` is the negative control.
    """
    return np.sqrt((eps_xx - eps_zz) ** 2 + 4.0 * eps_xz ** 2)


def principal_strains(eps_xx, eps_zz, eps_xz):
    """`(eps_1, eps_3)` with eps_1 >= eps_3, extension-positive."""
    mean = 0.5 * (eps_xx + eps_zz)
    rad = 0.5 * gamma_max(eps_xx, eps_zz, eps_xz)
    return mean + rad, mean - rad


# ---------------------------------------------------------------------------
# 2. Ridge extraction
# ---------------------------------------------------------------------------
def extract_ridge(X, Z, G, *, mask=None, min_frac=0.25):
    """Trace the locus of peak `G` down the slope: one point per column.

    `X`, `Z`, `G` are (nz, nx) arrays as produced by `np.meshgrid(x, z)` and a
    grid evaluation of `gamma_max`. Returns `(x_ridge, z_ridge)`, both 1-D,
    with columns dropped where the peak is weak or masked out.

    The column-wise argmax is the honest simple choice for a slope failure
    surface, which is single-valued in x for the compound basal-plus-exit
    mechanism you are expecting. It will mis-trace a surface that overhangs or
    doubles back -- if the MC and GHB surfaces disagree in a way that looks
    like a vertical jump rather than a shift, suspect this function before you
    suspect the physics, and plot the raw `G` field to check.

    `min_frac` drops columns whose peak is below that fraction of the global
    peak. Those are columns the mechanism does not pass through; keeping them
    draws a ridge across quiet elastic material and makes the surface look like
    it daylights in the wrong place.
    """
    G = np.asarray(G, dtype=float)
    if mask is not None:
        G = np.where(mask, G, np.nan)

    if np.all(np.isnan(G)):
        raise ValueError("gamma_max field is entirely masked or NaN")

    gpeak = np.nanmax(G)
    if not np.isfinite(gpeak) or gpeak <= 0:
        raise ValueError(f"non-positive peak shear strain: {gpeak}")

    xs, zs = [], []
    for j in range(G.shape[1]):
        col = G[:, j]
        if np.all(np.isnan(col)):
            continue
        i = int(np.nanargmax(col))
        if col[i] < min_frac * gpeak:
            continue
        xs.append(X[i, j])
        zs.append(Z[i, j])

    return np.asarray(xs), np.asarray(zs)


def localisation_intensity(G, mask=None):
    """Peak-to-median ratio of the shear-strain field.

    A blunt but useful scalar: in an elastic, non-localised state it sits at a
    few; once a band forms it climbs by an order of magnitude or more. Track it
    across the sweep. It is a far better convergence tell than the loss alone,
    because `L_PDE` can flatten for optimiser reasons that have nothing to do
    with the slope failing.
    """
    G = np.asarray(G, dtype=float)
    if mask is not None:
        G = np.where(mask, G, np.nan)
    med = np.nanmedian(G)
    if med <= 0:
        return float("inf")
    return float(np.nanmax(G) / med)


def band_width(X, Z, G, *, mask=None, frac=0.5):
    """Area of the region above `frac` of peak strain, divided by ridge length.

    A mesh-objectivity check with teeth. A genuine localisation band has a
    width set by the regularisation; a PINN's is set by network capacity and
    collocation density. If this changes materially when you refine the
    post-processing grid, the band is a resolution artefact and the FOS that
    came with it is not converged. Report the number next to the failure
    surface figure.
    """
    G = np.asarray(G, dtype=float)
    if mask is not None:
        G = np.where(mask, G, np.nan)

    gpeak = np.nanmax(G)
    hot = G >= frac * gpeak

    dx = float(np.abs(X[0, 1] - X[0, 0]))
    dz = float(np.abs(Z[1, 0] - Z[0, 0]))
    area = float(np.count_nonzero(hot)) * dx * dz

    xr, zr = extract_ridge(X, Z, G, mask=mask, min_frac=frac)
    if xr.size < 2:
        return float("nan")
    length = float(np.sum(np.hypot(np.diff(xr), np.diff(zr))))
    if length <= 0:
        return float("nan")
    return area / length