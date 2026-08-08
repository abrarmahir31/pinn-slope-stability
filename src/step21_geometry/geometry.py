"""
Section 5 (5-5' N) slope geometry, Isikdere lignite open pit.
Ulusay, Ekmekci, Tuncay & Hasancebi (2014), Eng. Geol. 181, 261-280.

COORDINATES
    x : m horizontal, 0 at the pit toe, increasing upslope (NE)
    z : m above sea level
    Plane strain, matching the 2-D LEM of the source paper.

SOURCE AND CALIBRATION
    Fig. 19d, embedded JPEG extracted at its native 300 dpi (qtable mean 8.4,
    blockiness 1.006 -- effectively lossless), cropped and upscaled 4x,
    digitised in WebPlotDigitizer.
    Panel V:H = 1.007, i.e. no vertical exaggeration.
    Four independent calibration checks:
      - the 20/22/25 deg dashed alternatives fit to 20.63/22.50/25.07 deg
      - all three extrapolate to a common toe at 270.80-270.97 m asl
      - the F1 fault top lands on the ground surface to 0.03 m
      - the top-of-Tm contact, traced twice, agreed to 0.18 m mean absolute
    Estimated digitisation precision: +/- 0.2 m.

VALIDATION TARGETS (values printed on Fig. 19d)
    initial cut, 31 deg   F = 0.94   <- primary
    flattened to 25 deg   F = 1.059
    flattened to 22 deg   F = 1.342
    flattened to 20 deg   F = 1.820
"""
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from pathlib import Path

DIG = Path(__file__).parent / "digitised"


def _load(name):
    d = pd.read_csv(DIG / f"{name}.csv", header=None, names=["x", "z"])
    d = d.apply(pd.to_numeric, errors="coerce").dropna()
    d = d.sort_values("x").drop_duplicates("x")
    return d["x"].to_numpy(), d["z"].to_numpy()


def _interp(x, z):
    # PCHIP, not CubicSpline. A spline overshoots between digitised points and
    # can lift the top-of-Tm above the ground surface -- an impossible geometry
    # that corrupts material tags with no error raised.
    return PchipInterpolator(x, z, extrapolate=True)


z_ground = _interp(*_load("d_ground"))

# Top of Tm: two traces tiling ONE contact. d_mk_tm covers x 0-90 (the marl
# band is labelled Mk there); d_mkd_base covers x 91-266 (labelled Mk_d).
_a, _b = _load("d_mk_tm"), _load("d_mkd_base")
z_tm_top = _interp(np.concatenate([_a[0], _b[0]]),
                   np.concatenate([_a[1], _b[1]]))

# F1 is near-vertical (81.4 deg), so fit x as a function of z.
_fx, _fz = _load("d_f1")
_fm, _fc = np.polyfit(_fz, _fx, 1)


def x_f1(z):
    """Horizontal position of the F1 fault trace at elevation z."""
    return _fm * np.asarray(z, float) + _fc


z_alt20, z_alt22, z_alt25 = (_interp(*_load(n))
                             for n in ("d_alt20", "d_alt22", "d_alt25"))

# --------------------------------------------------------------- constants
Z_TOE      = 271.13    # m asl, intercept of the 0-98 m cut-face fit
X_CREST    = 102.1     # m, crest of the excavated face
Z_CREST    = 332.70    # m asl
CUT_ANGLE  = 31.06     # deg, fitted x 0-98 m, RMS 0.149 m, n=16
CUT_HEIGHT = 61.6      # m
X_BENCH_END = 122.9    # m, bench runs X_CREST -> here at 0.4 deg

X_MK_DIVIDE = 90.7     # ASSUMPTION. The Mk/Mk_d contact is vertical; placed at
                       # the midpoint of the two traces. The figure's tick may
                       # sit nearer x = 105 -- re-digitise to confirm.

Z_BASE = 200.0         # MODELLING CHOICE, not a measurement. ~1 cut height
                       # below the toe so the fixed bottom BC does not perturb
                       # the failure zone. Assumes Tm continues this deep --
                       # verify against Fig. 5b.

X_MIN, X_MAX = 0.0, 268.0

F_TARGETS = {"initial_31deg": 0.94, "flat_25deg": 1.059,
             "flat_22deg": 1.342, "flat_20deg": 1.820}


# -------------------------------------------------------------- predicates
def inside_domain(x, z):
    """Solid ground: below topography, above the base, inboard of F1."""
    x = np.asarray(x, float)
    z = np.asarray(z, float)
    return ((z <= z_ground(x)) & (z >= Z_BASE)
            & (x <= x_f1(z)) & (x >= X_MIN))


def material_tag(x, z):
    """Stratum at (x, z). Vectorised over arrays.

    Section 5 stack: Tm (crystallized limestone, Turgut Fm.) below the
    top-of-Tm contact; a marl band above it, divided LATERALLY -- Mk toward
    the toe, Mk_d upslope. That contact is vertical, so the test is on x.
    """
    x = np.asarray(x, float)
    z = np.asarray(z, float)
    band = z > z_tm_top(x)
    return np.select(
        [~inside_domain(x, z),
         band & (x <= X_MK_DIVIDE),
         band],
        ["outside", "Mk", "Mk_d"],
        default="Tm",
    )


# ============================== MATERIAL PROPERTIES ==========================
# Unit identities are fixed by the Fig. 19 caption in Ulusay et al. (2014):
#   Mk   = marl of the Sekkoy formation           (aquiclude)
#   Mk_d = WEAK ZONE in the Sekkoy formation      (aquiclude) -- crushed,
#          paleokarstic material along Fault 1, above +280 m a.s.l.
#   Tm   = crystallized BASEMENT limestone        (karstic aquifer)
#
# Note the inverted hydrogeology: the aquifer (Tm) lies BENEATH two aquicludes.
# K jumps three orders of magnitude across the top-of-Tm contact, which runs
# through the middle of the domain.

try:
    from src.properties import RHO_DRY as RHO   # dry density, kg/m3
except ModuleNotFoundError:
    import sys, pathlib as _pl
    sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
    from src.properties import RHO_DRY as RHO
K_S    = {"Mk": 1.0e-9, "Mk_d": 1.0e-9, "Tm": 3.13e-6}  # m/s
SIG_CI = {"Mk": 1.79e7, "Mk_d": 4.29e6, "Tm": 7.0e7}    # Pa
GSI    = {"Mk": 50,     "Mk_d": 45,     "Tm": 60}
M_I    = {"Mk": 4,      "Mk_d": 3,      "Tm": 12}
D_DIST = {"Mk": 1,      "Mk_d": 1,      "Tm": 1}

# Residual Mohr-Coulomb, design-governing pair (Table 3a, validated by the
# paper's own back-analysis at F = 1).
C_RES   = {"Mk": 4.4e3, "Mk_d": 4.4e3, "Tm": 4.4e3}     # Pa
PHI_RES = {"Mk": 15.4,  "Mk_d": 15.4,  "Tm": 15.4}      # deg

# Mk and Mk_d: Ulusay Tables 2, 3a, 3b, via the Step 1.1 SI sheet.
# Tm: LITERATURE ESTIMATES, NOT FROM THE PAPER --
#     rho 2650 kg/m3 (gamma ~26 kN/m3, crystalline limestone)
#     sig_ci 70 MPa (mid-range), GSI 60, m_i 12 (Hoek, limestone/marble)
#     C_RES/PHI_RES carry the marl values as a conservative placeholder.
# Ulusay et al. report no Tm strength because no critical surface passes
# through the basement. Tm's density (geostatic stress) and permeability
# (karstic aquifer) matter; its strength should not govern. VERIFY by
# sweeping sig_ci 40-100 MPa and showing F is insensitive.

MK_D_IS_WEAK_ZONE = True   # sig_ci 4.29 vs 17.9 MPa for Mk -- a four-fold
                           # contrast, and Mk_d is ~76% of the marl band by
                           # area. Likely governs F = 0.94.
