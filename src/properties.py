"""
properties.py -- constitutive parameters for the Section 5 domain.

SINGLE SOURCE OF TRUTH. Everything mechanical/hydraulic that Phase 3 needs is
imported from Phase 1/isikdere_phase1_dataset.json, EXCEPT the Tm entry, which
postdates that file (Tm was identified as basement crystallized limestone during
the Step 2.1 stratigraphic correction, 2026-08-01).

Import this, not the JSON, and not the comment blocks in boundaries.py.

    from properties import E, NU, N0, THETA_S, THETA_R, ALPHA, VG_N, VG_M, K_S

Design rules:
  * Loop over UNIT_MAP, never over the dataset's strata. Four of its six strata
    (coal, tuffite, underclay, slope_debris) are SECTION 7 units and are not in
    this domain.
  * Assert on missing keys. A silent .get(k, default) for Poisson's ratio trains
    fine and gives a wrong answer in Phase 5.
  * Do NOT import the JSON's 'geometry', 'boundary_conditions' or 'validation'
    blocks. They were written 2026-07-29, before the Fig. 5b -> Fig. 19d
    correction, and describe a 400x170 m domain with a coal seam that does not
    exist here. geometry.py / boundaries.py / G.F_TARGETS are the authorities.
"""

import json
from pathlib import Path

# --------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_CANDIDATES = [
    _HERE.parent / "Phase 1" / "isikdere_phase1_dataset.json",
    _HERE.parent / "step21_geometry" / "Phase 1" / "isikdere_phase1_dataset.json",
    _HERE / "Phase 1" / "isikdere_phase1_dataset.json",
]
_PATH = next((p for p in _CANDIDATES if p.exists()), None)
if _PATH is None:
    raise FileNotFoundError(
        "isikdere_phase1_dataset.json not found. Looked in:\n  "
        + "\n  ".join(str(p) for p in _CANDIDATES))

_D = json.loads(_PATH.read_text())
_S = _D["strata"]

# geometry.py material_tag -> Phase-1 dataset stratum key
UNIT_MAP = {"Mk": "marl_sekkoy", "Mk_d": "marl_weakzone"}
UNITS = ["Mk", "Mk_d", "Tm"]

# --------------------------------------------------------------------------
# TM ADDENDUM
# --------------------------------------------------------------------------
# Not in the Phase-1 dataset. Derived 2026-08-03, same methods as the other
# strata so the treatment is uniform and defensible in the methods section.
#
#   E_rm   Hoek-Diederichs (2006), E_i = MR*sig_ci = 500 * 70 MPa = 35.0 GPa,
#          GSI 60, D 1  ->  4263.9 MPa
#   m_b,s,a  standard GHB reduction, GSI 60, m_i 12, D 1
#   nu     0.25, typical crystalline limestone
#   K0     nu/(1-nu) = 0.3333
#   n0     ** SEE WARNING **  1 - rho_dry/(rho_w*Gs), Gs = 2.71 (calcite)
#   theta_s  0.9*n0, the porosity-tie rule adopted for all rock analogues
#            in conflict C-? of the Phase-1 dataset
#
# WARNING -- Tm density.  geometry.RHO['Tm'] = 2650 kg/m3 is the standard
# limestone GRAIN density. Against the dataset's assumed Gs = 2.65 it gives
# n0 = 0.0000 exactly, i.e. a pore-free solid, which cannot carry
# K_s = 3.13e-6 m/s. Gs = 2.71 (calcite) is adopted here so that n0 = 0.0221.
# This is a CHOICE, not a measurement. If a site dry bulk density for Tm ever
# surfaces, replace it. Flag in the methods section.
#
# WARNING -- Tm SWCC is TIER C.  No grain-size data exists for a crystalline
# limestone and no site retention curve was measured. ALPHA and VG_N below are
# placeholders for a fractured-limestone literature set and MUST be replaced
# with a cited source before publication, and carried as a sensitivity variable.
# A 2.2%-porosity rock at K = 3e-6 m/s is fracture-dominated: any matrix SWCC
# describes the pathway the water does not use.
TM = {
    "E_Pa":          4.2639e9,
    "nu":            0.25,
    "K0_at_rest":    0.25 / 0.75,
    "Gs":            2.71,
    "Gs_source":     "calcite, adopted 2026-08-03 (see warning)",
    "rho_dry_kgpm3": 2650.0,
    "n0":            1.0 - 2650.0 / (1000.0 * 2.71),
    "K_s_mps":       3.13e-6,
    "sigma_ci_Pa":   70.0e6,
    "GSI":           60,
    "m_i":           12,
    "D":             1.0,
    "m_b":           0.6892,
    "s":             1.2726e-3,
    "a":             0.5028,
    "vg_theta_r":    0.005,      # PLACEHOLDER -- cite before publication
    "vg_alpha_1pm":  2.0,        # PLACEHOLDER -- cite before publication
    "vg_n":          2.0,        # PLACEHOLDER -- cite before publication
    "vg_tier":       "C",
}
TM["vg_theta_s"] = 0.9 * TM["n0"]
TM["rho_sat_kgpm3"] = TM["rho_dry_kgpm3"] + TM["n0"] * 1000.0
TM["vg_m"] = 1.0 - 1.0 / TM["vg_n"]

# --------------------------------------------------------------------------
_FIELDS = {
    "E":        "E_Pa",
    "NU":       "nu",
    "K0":       "K0_at_rest",
    "N0":       "n0",
    "K_S":      "K_s_mps",
    "RHO_DRY":  "rho_dry_kgpm3",
    "RHO_SAT":  "rho_sat_kgpm3",
    "THETA_R":  "vg_theta_r",
    "THETA_S":  "vg_theta_s",
    "ALPHA":    "vg_alpha_1pm",
    "VG_N":     "vg_n",
    "VG_M":     "vg_m",
    "SIG_CI":   "sigma_ci_Pa",
    "GSI":      "GSI",
    "M_I":      "m_i",
    "M_B":      "m_b",
    "HB_S":     "s",
    "HB_A":     "a",
    "GS":       "Gs",
}


def _record(unit):
    if unit == "Tm":
        return TM
    key = UNIT_MAP[unit]
    if key not in _S:
        raise KeyError(f"stratum '{key}' (for unit '{unit}') not in dataset; "
                       f"available: {sorted(_S)}")
    return _S[key]


_TABLES = {}
_missing = []
for _name, _field in _FIELDS.items():
    _t = {}
    for _u in UNITS:
        _r = _record(_u)
        if _field not in _r:
            _missing.append(f"{_u}.{_field}")
            continue
        _t[_u] = _r[_field]
    _TABLES[_name] = _t

if _missing:
    raise KeyError(
        "properties.py: missing fields, refusing to load with defaults:\n  "
        + "\n  ".join(_missing)
        + "\n(fix the dataset or the TM addendum; do not add .get() fallbacks)")

globals().update(_TABLES)

RHO_W = 1000.0
G_ACC = 9.81


def rho_bulk(unit, theta):
    """Bulk density at volumetric water content theta.

    The mechanical body force must use THIS, not RHO_DRY. Rainfall infiltration
    has two limbs: suction is lost (Bishop coupling) AND the wetted material
    gets heavier. For Mk, dry -> saturated is 1518.9 -> 1945.7 kg/m3, a 28%
    increase in driving weight. Omitting it models a rainfall-triggered failure
    with a body force that cannot respond to rainfall.
    """
    return RHO_DRY[unit] + theta * RHO_W                      # noqa: F821


def _selfcheck():
    """Cross-checks that must hold. Run: python properties.py"""
    ok = True
    for u in UNITS:
        k0 = NU[u] / (1.0 - NU[u])                            # noqa: F821
        if abs(k0 - K0[u]) > 1e-4:                            # noqa: F821
            print(f"[FAIL] {u}: K0_at_rest {K0[u]:.4f} != nu/(1-nu) {k0:.4f}")
            ok = False
        ts = 0.9 * N0[u]                                      # noqa: F821
        if abs(ts - THETA_S[u]) > 0.01:                       # noqa: F821
            print(f"[WARN] {u}: theta_s {THETA_S[u]:.4f} != 0.9*n0 {ts:.4f}")
        rs = RHO_DRY[u] + N0[u] * RHO_W                       # noqa: F821
        if abs(rs - RHO_SAT[u]) > 5.0:                        # noqa: F821
            print(f"[WARN] {u}: rho_sat {RHO_SAT[u]:.1f} != dry+n0*rho_w {rs:.1f}")
        m = 1.0 - 1.0 / VG_N[u]                               # noqa: F821
        if abs(m - VG_M[u]) > 1e-4:                           # noqa: F821
            print(f"[FAIL] {u}: vg_m {VG_M[u]:.4f} != 1-1/n {m:.4f}")
            ok = False
    print(f"[INFO] E contrast Mk/Mk_d = {E['Mk']/E['Mk_d']:.1f}x, "      # noqa: F821
          f"Tm/Mk_d = {E['Tm']/E['Mk_d']:.0f}x")                          # noqa: F821
    print(f"[INFO] K_s contrast Tm/Mk = {K_S['Tm']/K_S['Mk']:.0f}x")      # noqa: F821
    print("[INFO] Tm SWCC is TIER C placeholder -- cite before publication")
    print("ALL CROSS-CHECKS PASSED" if ok else "CROSS-CHECKS FAILED")
    return ok


if __name__ == "__main__":
    print(f"loaded {_PATH}")
    for u in UNITS:
        print(f"  {u:5s} E={E[u]/1e6:8.1f} MPa  nu={NU[u]:.2f}  "        # noqa: F821
              f"n0={N0[u]:.4f}  theta_s={THETA_S[u]:.4f}  "              # noqa: F821
              f"K_s={K_S[u]:.2e}  rho_dry={RHO_DRY[u]:.1f}")             # noqa: F821
    print()
    _selfcheck()
