"""Yield-constraint loss term — the piece Step 5.1 is missing.

WHY THIS FILE EXISTS
--------------------
As of the day-28 snapshot, `mechanics.total_stress_star` assembles

    sigma = sigma_0 + D:eps - (chi psi - chi_0 psi_0)

which is LINEAR ELASTIC plus a Bishop pore term, and `mechanical_residual`
takes its divergence. There is no yield surface anywhere in `src/`. Strength
parameters -- c, phi, m_b, s -- do not appear in the governing equations at
all.

The consequence for the SSR sweep: **reducing strength changes nothing.** The
loss at SRF = 3 is bit-for-bit the loss at SRF = 1, the network converges
identically at every step of the sweep, and the "critical SRF where L_PDE
flatlines high" never arrives because nothing was ever constrained by strength.
A sweep run against the elastic formulation will return a smooth, converged,
meaningless curve. It will not crash, which is the dangerous part.

So SSR needs a constitutive model in which strength binds. This module supplies
the cheapest defensible one: a soft yield constraint.

THE SOFT-CONSTRAINT APPROACH, AND ITS PRICE
-------------------------------------------
Add a loss term penalising stress states outside the yield surface:

    L_yield = mean( relu(f(sigma; params))**2 )   over the collocation set

with f the yield function from `strength.py`, normalised. Drive it alongside
the equilibrium residual. At low SRF the network satisfies both: the elastic
solution is admissible. As SRF rises the surface shrinks, the two objectives
compete, and past the critical SRF no field satisfies both -- equilibrium
residual and yield penalty cannot fall together, the total loss plateaus well
above its baseline, and displacements grow. That plateau IS the failure
signature, and it is the same qualitative signature as non-convergence in an
FE SSR analysis.

What this is NOT: a return-mapping plasticity algorithm. A proper elasto-plastic
PINN enforces admissibility exactly by projecting trial stress onto the yield
surface and tracking plastic strain as an internal variable with a consistency
condition. That gives a correct stress field and a correct plastic flow
direction; this gives an approximately admissible one, with the approximation
controlled by a penalty weight you have to choose.

Three things follow, and all three belong in the thesis rather than buried here:

  1. **The FOS depends on the penalty weight.** Too soft and the network buys
     equilibrium by violating yield, pushing the critical SRF up; too stiff and
     it stalls early, pushing it down. You MUST report FOS against at least
     three weights and show the plateau. If it moves more than a few percent,
     it is not converged and the number is not reportable.
  2. **No plastic flow rule, so no dilatancy and no correct post-peak
     kinematics.** The failure SURFACE from `failure_surface.py` is still
     meaningful -- localisation shows up in the elastic-strain field once yield
     binds -- but the displacement MAGNITUDES past the critical SRF are not.
     Do not quote them.
  3. **Associated vs non-associated does not arise**, because there is no flow
     rule to associate. That is a simplification to declare, not to hide. For
     a frictional material the two give materially different factors of safety
     and reviewers know it.

If the supervisor wants a defensible FOS rather than a demonstration of method,
the honest scope is return mapping, and that is weeks, not days.

Layer role: this module does algebra and reductions only. The autograd-facing
assembly lives in `loss.py`. Backend-agnostic, like `strength.py`.
"""
from __future__ import annotations

import numpy as np

from src import strength as st

try:
    import torch as _torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _torch = None
    _HAS_TORCH = False


def _relu(v):
    if _HAS_TORCH and isinstance(v, _torch.Tensor):
        return _torch.clamp(v, min=0.0)
    return np.maximum(v, 0.0)


def _mean(v):
    if _HAS_TORCH and isinstance(v, _torch.Tensor):
        return _torch.mean(v)
    return np.mean(v)


# ---------------------------------------------------------------------------
def yield_violation(sig1, sig3, params, criterion: str, *,
                    sigma_scale: float):
    """Normalised, one-sided yield violation. Zero inside, positive outside.

    `sig1`, `sig3` are COMPRESSION-POSITIVE principal EFFECTIVE stresses, in
    the same units as `sigma_scale`. Get them from
    `strength.principal_stresses_from_cartesian`, which does the tension-to-
    compression flip from the `mechanics.py` convention.

    `sigma_scale` non-dimensionalises f so the penalty is comparable across
    strata whose strengths differ by three orders of magnitude (marl at
    sigma_ci 17.9 MPa against Tm at 70 MPa). Use `s.sig_ref`, consistently,
    everywhere -- NOT each material's own sigma_ci, or the balancer will see a
    term whose scale changes with the tag partition and will chase it.
    """
    if criterion == "MC":
        f = st.mc_yield(sig1, sig3, params)
    elif criterion == "GHB":
        f = st.ghb_yield(sig1, sig3, params)
    else:
        raise ValueError(f"unknown criterion {criterion!r}")
    return _relu(f / sigma_scale)


def L_yield(sig1, sig3, params, criterion: str, *, sigma_scale: float):
    """Mean squared yield violation -- the loss term itself.

    Squared, not absolute: the gradient must vanish smoothly at the surface,
    or every point sitting exactly at yield receives a constant kick and the
    optimiser jitters along the envelope instead of settling on it.
    """
    v = yield_violation(sig1, sig3, params, criterion,
                        sigma_scale=sigma_scale)
    return _mean(v ** 2)


def admissible_fraction(sig1, sig3, params, criterion: str, *,
                        sigma_scale: float, tol: float = 1e-9):
    """Fraction of points inside the yield surface. The sweep's health check.

    Read it alongside the loss at every SRF. The two failure modes look
    identical in the loss and completely different here:

      - a genuine critical state: the fraction falls steadily, a connected
        band of yielded points forms, and localisation intensity climbs;
      - a training failure: the fraction collapses everywhere at once, or
        stays near 1 while the loss plateaus anyway.

    Only the first is a factor of safety. Log this at every SRF; it is the
    cheapest protection against reporting an optimiser artefact as geotechnics.
    """
    v = yield_violation(sig1, sig3, params, criterion,
                        sigma_scale=sigma_scale)
    if _HAS_TORCH and isinstance(v, _torch.Tensor):
        return float((v <= tol).double().mean().item())
    return float(np.mean(np.asarray(v) <= tol))


def reduced_material_params(mat, criterion: str, srf: float, *,
                            mc_base: st.MCParams | None = None,
                            sig3_range: tuple[float, float] | None = None):
    """Strength parameters for one material at one SRF.

    MC: reduces `mc_base`, which is NOT read off the Material -- the Phase-1
    dataset flags `mc_strength.design_pair` (c = 4400 Pa, phi = 15.4 deg,
    smooth bedding residual) as the pair to reduce for Step 5.1, and it is a
    DISCONTINUITY strength, not a rock-mass one. The Material record carries
    the rock-mass GHB set instead. Passing the wrong one silently swaps a
    bedding-controlled failure for an intact-rock one and moves the FOS by
    more than the MC-vs-GHB difference you are trying to measure.

    GHB: reduces the material's own (sigma_ci, m_b, s, a) via
    `strength.reduce_ghb`, which needs `sig3_range`. See that docstring --
    the range is a modelling choice you must set from the baseline stress
    field and report.
    """
    if criterion == "MC":
        if mc_base is None:
            raise ValueError(
                "MC sweep needs the design pair explicitly; see docstring")
        return st.reduce_mc(mc_base, srf)

    if criterion == "GHB":
        if sig3_range is None:
            raise ValueError("GHB reduction needs sig3_range; see docstring")
        base = st.GHBParams(sigma_ci=mat.sigma_ci, m_b=mat.m_b,
                            s=mat.s, a=mat.a)
        return st.reduce_ghb(base, srf, sig3_range=sig3_range)

    raise ValueError(f"unknown criterion {criterion!r}")