"""Failure criteria and shear-strength reduction — ALGEBRA ONLY, no autograd.

Layer role mirrors `nondim.py`: this module holds the constitutive algebra for
the Mohr-Coulomb and Generalized Hoek-Brown criteria and the strength-reduction
operators used by the SSR sweep (Steps 5.1 / 5.2). It never calls autograd and
never touches a network. The autograd-facing wrapper lives in `residuals.py` /
the sweep driver; keep it that way.

Two conventions that bite, stated once:

**Sign.** `mechanics.py` is TENSION-POSITIVE (see `stress_increment_star`).
Rock-mechanics failure criteria are written compression-positive. Every public
function here takes and returns COMPRESSION-POSITIVE effective stresses, and
the caller converts. `principal_stresses_from_cartesian` does the flip for you
and is the intended entry point from the PINN side. Doing the flip in one
place, loudly, is deliberate: a silent sign error in a yield function does not
crash, it just returns a plausible and wrong factor of safety.

**Effective stress.** These criteria take EFFECTIVE stress. Section 5 is
unsaturated at t = 0 (`psi0_star` < 0 everywhere), so the Bishop coupling in
`mechanics.chi_of` matters and the caller must apply it before arriving here.
This module cannot check that you did.

Backends: every function works on numpy arrays, python floats, and torch
tensors. The dispatch is duck-typed in `_ops`; no torch import is required to
use the numpy path, which is what lets the test suite run without a GPU box.

References
----------
Hoek, Carranza-Torres & Corkum (2002) — GHB envelope and the m_b, s, a
    reductions from GSI, m_i, D. Restated in the Phase-1 dataset under
    `constitutive_models/hoek_brown`.
Hammah, Yacoub, Corkum & Curran (2004) — "The shear strength reduction method
    for the generalized Hoek-Brown criterion". The instantaneous-Mohr-Coulomb
    route implemented in `reduce_ghb`. There is no defensible way to divide
    m_b and s by the SRF directly; see that function's docstring.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:  # torch is optional for the algebra layer
    import torch as _torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover - exercised only on boxes without torch
    _torch = None
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# 0. Backend dispatch
# ---------------------------------------------------------------------------
class _NumpyOps:
    sqrt = staticmethod(np.sqrt)
    sin = staticmethod(np.sin)
    cos = staticmethod(np.cos)
    tan = staticmethod(np.tan)
    asin = staticmethod(np.arcsin)
    atan = staticmethod(np.arctan)
    where = staticmethod(np.where)
    maximum = staticmethod(np.maximum)
    minimum = staticmethod(np.minimum)

    @staticmethod
    def power(b, e):
        return np.power(b, e)


class _TorchOps:  # pragma: no cover - needs torch
    sqrt = staticmethod(lambda v: _torch.sqrt(v))
    sin = staticmethod(lambda v: _torch.sin(v))
    cos = staticmethod(lambda v: _torch.cos(v))
    tan = staticmethod(lambda v: _torch.tan(v))
    asin = staticmethod(lambda v: _torch.asin(v))
    atan = staticmethod(lambda v: _torch.atan(v))
    where = staticmethod(lambda c, a, b: _torch.where(c, a, b))
    maximum = staticmethod(lambda a, b: _torch.clamp(a, min=b) if isinstance(b, float)
                           else _torch.maximum(a, b))
    minimum = staticmethod(lambda a, b: _torch.clamp(a, max=b) if isinstance(b, float)
                           else _torch.minimum(a, b))

    @staticmethod
    def power(b, e):
        return _torch.pow(b, e)


def _ops(v):
    if _HAS_TORCH and isinstance(v, _torch.Tensor):
        return _TorchOps
    return _NumpyOps


def _asarray_like(v, ref):
    """Broadcast a python scalar to the backend of `ref`."""
    if _HAS_TORCH and isinstance(ref, _torch.Tensor):
        if isinstance(v, _torch.Tensor):
            return v
        return _torch.as_tensor(v, dtype=ref.dtype, device=ref.device)
    return v


# ---------------------------------------------------------------------------
# 1. Parameter records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MCParams:
    """Mohr-Coulomb effective-stress strength pair."""
    c: float        # Pa
    phi: float      # RADIANS

    @property
    def phi_deg(self) -> float:
        return math.degrees(self.phi)

    @property
    def tan_phi(self) -> float:
        return math.tan(self.phi)


@dataclass(frozen=True)
class GHBParams:
    """Generalized Hoek-Brown rock-mass parameters."""
    sigma_ci: float   # Pa, intact UCS
    m_b: float
    s: float
    a: float

    @property
    def ucs_rockmass(self) -> float:
        """Unconfined rock-mass strength, sigma_ci * s**a. Useful sanity peg."""
        return self.sigma_ci * self.s ** self.a


def ghb_from_gsi(sigma_ci: float, gsi: float, m_i: float, D: float) -> GHBParams:
    """Hoek et al. (2002) reductions. Mirrors the Phase-1 dataset strings.

    Present so a test can bind the stored m_b, s, a back to their generating
    formula; `properties.py` remains the single source of truth for the values
    actually used.
    """
    m_b = m_i * math.exp((gsi - 100.0) / (28.0 - 14.0 * D))
    s = math.exp((gsi - 100.0) / (9.0 - 3.0 * D))
    a = 0.5 + (1.0 / 6.0) * (math.exp(-gsi / 15.0) - math.exp(-20.0 / 3.0))
    return GHBParams(sigma_ci=sigma_ci, m_b=m_b, s=s, a=a)


# ---------------------------------------------------------------------------
# 2. Principal stresses  (tension-positive in  ->  compression-positive out)
# ---------------------------------------------------------------------------
def principal_stresses_from_cartesian(sxx, szz, sxz, *, tension_positive=True):
    """Plane-strain principal stresses, returned COMPRESSION-POSITIVE.

    `sxx, szz, sxz` come from `mechanics.total_stress_star` and are
    tension-positive by that module's convention; they must already carry the
    Bishop effective-stress correction. Returns `(sig1, sig3)` with
    sig1 >= sig3, both compression-positive, in whatever units went in
    (dimensionless sigma* in, dimensionless out -- multiply by `s.sig_ref` for
    Pa, and do it before comparing against c in Pa).

    Set `tension_positive=False` if you are passing an already-flipped field.
    """
    xp = _ops(sxx)
    if tension_positive:
        sxx, szz, sxz = -sxx, -szz, -sxz

    mean = 0.5 * (sxx + szz)
    rad = xp.sqrt((0.5 * (sxx - szz)) ** 2 + sxz ** 2)
    return mean + rad, mean - rad


# ---------------------------------------------------------------------------
# 3. Mohr-Coulomb
# ---------------------------------------------------------------------------
def mc_sigma1(sig3, p: MCParams):
    """MC envelope: major principal stress at failure, compression-positive."""
    sp, cp = math.sin(p.phi), math.cos(p.phi)
    return (2.0 * p.c * cp + sig3 * (1.0 + sp)) / (1.0 - sp)


def mc_yield(sig1, sig3, p: MCParams):
    """Yield function f, compression-positive principal stresses.

    f = (sig1 - sig3) - [2c cos(phi) + (sig1 + sig3) sin(phi)]

    f < 0 elastic, f = 0 at failure, f > 0 inadmissible. Returned in stress
    units, NOT normalised -- do not read its magnitude as a margin.
    """
    sp, cp = math.sin(p.phi), math.cos(p.phi)
    return (sig1 - sig3) - (2.0 * p.c * cp + (sig1 + sig3) * sp)


def mc_strength_ratio(sig1, sig3, p: MCParams, eps: float = 1e-30):
    """Local mobilised-strength ratio: available / mobilised shear.

    Equals 1 at failure, > 1 elastic. This is the pointwise quantity whose
    minimum over the domain is sometimes reported as a "local FOS"; it is NOT
    the SSR factor of safety and the two should never be conflated in the
    write-up. Kept because it is a cheap pre-sweep sanity check: if this is
    already below 1 across a wide band at SRF = 1, the baseline is not
    equilibrated and the sweep will waste a day.
    """
    xp = _ops(sig1)
    sp, cp = math.sin(p.phi), math.cos(p.phi)
    mobilised = sig1 - sig3
    available = 2.0 * p.c * cp + (sig1 + sig3) * sp
    return available / xp.maximum(mobilised, eps)


def reduce_mc(p: MCParams, srf: float) -> MCParams:
    """Classic SSR reduction: c/SRF and arctan(tan(phi)/SRF).

    Note it is TAN(phi) that is divided, not phi. Dividing the angle is a
    common and quiet error -- at phi = 15.4 deg and SRF = 1.5 the two differ by
    about 0.6 deg, small enough to look like noise and large enough to move the
    third significant figure of the FOS.
    """
    if srf <= 0:
        raise ValueError(f"srf must be positive, got {srf}")
    return MCParams(c=p.c / srf, phi=math.atan(math.tan(p.phi) / srf))


# ---------------------------------------------------------------------------
# 4. Generalized Hoek-Brown
# ---------------------------------------------------------------------------
def ghb_sigma1(sig3, p: GHBParams, clamp_tensile: bool = True):
    """GHB envelope, compression-positive:

        sig1 = sig3 + sigma_ci * (m_b * sig3 / sigma_ci + s) ** a

    The bracket goes negative for sig3 below the tensile cutoff
    -s*sigma_ci/m_b, where a fractional power is undefined. `clamp_tensile`
    floors it at zero, which caps strength at the tensile limit rather than
    returning NaN mid-sweep. Leave it on unless you are deliberately probing
    the tensile corner.
    """
    xp = _ops(sig3)
    base = p.m_b * sig3 / p.sigma_ci + p.s
    if clamp_tensile:
        base = xp.maximum(base, _asarray_like(0.0, sig3))
    return sig3 + p.sigma_ci * xp.power(base, p.a)


def ghb_yield(sig1, sig3, p: GHBParams):
    """Yield function f = sig1_applied - sig1_envelope. f > 0 inadmissible."""
    return sig1 - ghb_sigma1(sig3, p)


def ghb_instantaneous_mc(sig3, p: GHBParams, clamp_tensile: bool = True):
    """Tangent Mohr-Coulomb pair at a given confining stress.

    Returns `(c_i, phi_i)` with phi_i in radians -- the MC line tangent to the
    GHB envelope at sig3. From the envelope slope n = d sig1 / d sig3:

        n       = 1 + a * m_b * (m_b sig3 / sigma_ci + s) ** (a - 1)
        sin phi = (n - 1) / (n + 1)
        c       = (sig1 - n sig3) * (1 - sin phi) / (2 cos phi)

    Degenerate limit worth knowing: as m_b -> 0 the slope goes to 1, phi -> 0,
    and c -> sigma_ci * s**a / 2, i.e. half the unconfined rock-mass strength.
    `test_ghb_instantaneous_mc_degenerates_to_pure_cohesion` pins that.

    Arrays in, arrays out; this is the per-point route. For the sweep you want
    `reduce_ghb`, which collapses the whole envelope to one reduced parameter
    set so the PINN sees a single material.
    """
    xp = _ops(sig3)
    base = p.m_b * sig3 / p.sigma_ci + p.s
    if clamp_tensile:
        base = xp.maximum(base, _asarray_like(1e-300, sig3))

    n = 1.0 + p.a * p.m_b * xp.power(base, p.a - 1.0)
    sin_phi = (n - 1.0) / (n + 1.0)
    phi = xp.asin(sin_phi)
    cos_phi = xp.cos(phi)

    sig1 = ghb_sigma1(sig3, p, clamp_tensile=clamp_tensile)
    c = (sig1 - n * sig3) * (1.0 - sin_phi) / (2.0 * cos_phi)
    return c, phi


def reduce_ghb(p: GHBParams, srf: float, *,
               sig3_range: tuple[float, float],
               n_fit: int = 64,
               hold_a: bool = True) -> GHBParams:
    """Strength-reduce a GHB parameter set by SRF (Hammah et al. 2004).

    There is no direct analogue of "divide c and tan(phi)" for GHB: m_b and s
    are not strengths, they are envelope shape parameters, and dividing them by
    the SRF reduces strength by an amount that varies with confinement in a way
    nobody can defend in a viva. The accepted route, and the one implemented
    here, is:

      1. over a grid of confining stresses spanning `sig3_range`, take the
         instantaneous MC tangent (c_i, phi_i) of the ORIGINAL envelope;
      2. reduce each tangent pair the classic way, c_i/SRF and
         arctan(tan(phi_i)/SRF);
      3. rebuild the implied major principal stress at each sig3 from the
         reduced tangent pair;
      4. least-squares fit a new (m_b', s') -- with `a` held fixed by default --
         through those reduced points.

    **`sig3_range` is a modelling choice and it is yours to make and defend.**
    The fit is only as good as the confining range it covers, and the answer
    moves if you pick a range the failure surface does not actually sample. Set
    it from the converged baseline stress field along the candidate failure
    zone -- roughly `(0, sig3_p95)` over the elements the mechanism runs
    through -- not from the full domain, whose deep Tm points sit at
    confinements the slip surface never sees. Record the range you used next to
    every FOS_GHB you report.

    Returns a fresh `GHBParams`. `srf = 1` returns the input to fitting
    tolerance (pinned by `test_reduce_ghb_is_identity_at_srf_one`).
    """
    from scipy.optimize import least_squares

    if srf <= 0:
        raise ValueError(f"srf must be positive, got {srf}")

    lo, hi = sig3_range
    if not (hi > lo >= 0.0):
        raise ValueError(
            f"sig3_range must satisfy hi > lo >= 0, got {sig3_range}")

    sig3 = np.linspace(lo, hi, n_fit)

    c_i, phi_i = ghb_instantaneous_mc(sig3, p)
    c_r = c_i / srf
    phi_r = np.arctan(np.tan(phi_i) / srf)

    sp, cp = np.sin(phi_r), np.cos(phi_r)
    sig1_reduced = (2.0 * c_r * cp + sig3 * (1.0 + sp)) / (1.0 - sp)

    scale = max(abs(float(np.max(sig1_reduced))), 1.0)

    def residual(theta):
        m_b, s_ = np.exp(theta[0]), np.exp(theta[1])
        trial = GHBParams(p.sigma_ci, m_b, s_, p.a)
        return (ghb_sigma1(sig3, trial) - sig1_reduced) / scale

    theta0 = np.array([math.log(max(p.m_b, 1e-12)),
                       math.log(max(p.s, 1e-12))])
    sol = least_squares(residual, theta0, method="lm", xtol=1e-14, ftol=1e-14)

    if not sol.success:
        raise RuntimeError(
            f"GHB reduction fit failed at srf={srf}: {sol.message}")

    m_b_r, s_r = float(np.exp(sol.x[0])), float(np.exp(sol.x[1]))

    if not hold_a:
        raise NotImplementedError(
            "Fitting `a` as well is possible but under-determined over a "
            "narrow sig3 range; Hammah et al. hold it fixed. If you need it, "
            "widen the range and add a test that the three-parameter fit is "
            "actually better conditioned.")

    return GHBParams(sigma_ci=p.sigma_ci, m_b=m_b_r, s=s_r, a=p.a)


# ---------------------------------------------------------------------------
# 5. Convenience: the two criteria behind one call
# ---------------------------------------------------------------------------
def reduced_params(criterion: str, base, srf: float, **kw):
    """Dispatch used by the sweep driver so the loop body is criterion-free."""
    if criterion == "MC":
        return reduce_mc(base, srf)
    if criterion == "GHB":
        return reduce_ghb(base, srf, **kw)
    raise ValueError(f"unknown criterion {criterion!r}, expected 'MC' or 'GHB'")

# ---------------------------------------------------------------------------
# 6. Rock-mass MC equivalents (Day 41, D-5.2 option a)
# ---------------------------------------------------------------------------
def ghb_equivalent_mc(p: GHBParams, sig3max: float) -> MCParams:
    """Hoek, Carranza-Torres & Corkum (2002) equivalent c, phi of the GHB
    envelope `p` over 0 <= sig3 <= sig3max (Pa). `p` already carries D through
    m_b and s. Reproduces the Phase-1 dataset's `hoek_brown_mc_equivalents`
    to 0.03% (tests/test_decision_evidence.py).

    These are ROCK-MASS strengths, far stronger than the bedding residual
    pair; which one the MC sweep reduces is D-5.2, not this function.
    """
    if not sig3max > 0:
        raise ValueError("sig3max must be > 0 Pa")
    s3n = sig3max / p.sigma_ci
    a, mb, s = p.a, p.m_b, p.s
    k = 6.0 * a * mb * (s + mb * s3n) ** (a - 1.0)
    d = (1.0 + a) * (2.0 + a)
    phi = math.asin(k / (2.0 * d + k))
    c = (p.sigma_ci * ((1.0 + 2.0 * a) * s + (1.0 - a) * mb * s3n)
         * (s + mb * s3n) ** (a - 1.0) / (d * math.sqrt(1.0 + k / d)))
    return MCParams(c=c, phi=phi)
