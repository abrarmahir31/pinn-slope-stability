"""Section 5 sensitivity arms and the solver hooks they need (Day 40).

WHY THIS FILE EXISTS. `src/sensitivity.py` derives parameter sets correctly
(its GSI chain reproduces `load_materials()` for Mk, Mk_d and Tm to four
figures via `properties.UNIT_MAP`), but two things in it never reached the
solver:

  * `oat_arms()` perturbs **coal** K_s. Section 5 has no coal seam (the Day 1
    stratigraphic correction); the arm would move a parameter nothing reads.
  * its override keys (`ks_multiplier`, `rain_mm_hr`, `stratum_overrides`)
    are not consumed anywhere. An arm launched with them would re-run the
    baseline and report a zero-width tornado bar.

`sensitivity.py` is left untouched so its 27 tests keep their meaning. This
module supplies the Section 5 arm list and the hooks, and `ssr_sweep.py`
applies them.

WHERE EACH INPUT ACTUALLY ENTERS -- all three must be patched together:

  * `Material` records (`loss_fn["mats"]`): K_s in the Darcy flux, E/nu in the
    constitutive law, m_b/s/a in the GHB yield term.
  * `boundaries.K_SAT`: a DUPLICATE of K_s used for the infiltration-capacity
    cap in `flux_bc`. Patching only `Material.K_s` leaves the cap at baseline
    and the K_s arm half-applied, silently.
  * `boundaries.RAIN_FLUX`: module global read at call time.

WHAT THIS CANNOT FIX. `sigma_0` comes from an offline FE gravity solve
(`sigma0_cache.npz`) with the BASELINE moduli. An arm that moves E (the GSI
arm does, deliberately) runs against a sigma_0 that is no longer equilibrated
for its stiffness contrast. `sigma0_consistent()` reports this and the sweep
writes it to result.json. Whether to re-solve sigma_0 per arm is a decision.

WHAT IS A DECISION, NOT A DEFAULT. Which stratum's K_s replaces the plan's
coal-seam arm, and which marl carries the GSI arm, are required arguments of
`section5_oat_arms`. Nothing here picks them.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import os

from src import sensitivity as sens
from src.coupling import KC_DEFAULT, KC_OFF, KC_ON
from src.materials import Material
from src.properties import UNIT_MAP
from src.step21_geometry import boundaries as bnd

SECTION5_TAGS = ("Mk", "Mk_d", "Tm")
MM_HR_TO_M_S = 1.0 / 3.6e6            # 20 mm/hr -> 5.556e-6 m/s
_MATERIAL_FIELDS = {f.name for f in dataclasses.fields(Material)} - {"tag"}
_KNOWN_KEYS = {"materials", "ks_multiplier", "rain_flux", "kc"}
_KC = {"default": KC_DEFAULT, "all": KC_ON, "off": KC_OFF}
# Fields whose change invalidates the stored sigma_0 (FE gravity solve inputs).
_SIGMA0_FIELDS = {"E", "nu", "K0", "rho_dry", "rho_sat"}


def validate(overrides: dict) -> dict:
    """Reject anything the hooks would not apply. Returns `overrides`."""
    unknown = set(overrides) - _KNOWN_KEYS
    if unknown:
        raise KeyError(f"unknown override key(s) {sorted(unknown)}; "
                       f"known: {sorted(_KNOWN_KEYS)}")
    for key in ("materials", "ks_multiplier"):
        for tag, val in overrides.get(key, {}).items():
            if tag not in SECTION5_TAGS:
                raise KeyError(f"{key}: {tag!r} is not a Section 5 stratum "
                               f"{SECTION5_TAGS}")
            if key == "materials":
                bad = set(val) - _MATERIAL_FIELDS
                if bad:
                    raise KeyError(f"materials[{tag}]: unknown field(s) "
                                   f"{sorted(bad)}")
            elif not val > 0:
                raise ValueError(f"ks_multiplier[{tag}] must be > 0")
    if "materials" in overrides and "ks_multiplier" in overrides:
        both = {t for t, v in overrides["materials"].items() if "K_s" in v}
        both &= set(overrides["ks_multiplier"])
        if both:
            raise ValueError(f"K_s set twice for {sorted(both)}")
    if "rain_flux" in overrides and not overrides["rain_flux"] > 0:
        raise ValueError("rain_flux must be > 0 (m/s)")
    if "kc" in overrides and overrides["kc"] not in _KC:
        raise KeyError(f"kc must be one of {sorted(_KC)}")
    return overrides


def apply_material_overrides(mats: dict, overrides: dict) -> dict:
    """New materials dict with overrides applied. Input is not mutated."""
    validate(overrides)
    out = dict(mats)
    for tag, fields in overrides.get("materials", {}).items():
        out[tag] = dataclasses.replace(out[tag], **fields)
    for tag, f in overrides.get("ks_multiplier", {}).items():
        out[tag] = dataclasses.replace(out[tag], K_s=out[tag].K_s * f)
    return out


@contextlib.contextmanager
def boundary_overrides(mats: dict, overrides: dict):
    """Patch `boundaries.K_SAT` to match `mats` and set `RAIN_FLUX`; restore
    on exit, including on exceptions. `mats` must already carry the overrides
    (from `apply_material_overrides`), so the two K_s copies cannot diverge."""
    validate(overrides)
    saved_k = dict(bnd.K_SAT)
    saved_rain = bnd.RAIN_FLUX
    try:
        for tag in bnd.K_SAT:
            if tag in mats:
                bnd.K_SAT[tag] = float(mats[tag].K_s)
        if "rain_flux" in overrides:
            bnd.RAIN_FLUX = float(overrides["rain_flux"])
        yield
    finally:
        bnd.K_SAT.clear()
        bnd.K_SAT.update(saved_k)
        bnd.RAIN_FLUX = saved_rain


def kc_config(name: str):
    if name not in _KC:
        raise KeyError(f"kc must be one of {sorted(_KC)}")
    return _KC[name]


def sigma0_consistent(overrides: dict) -> bool:
    """False if the arm moves an input of the offline sigma_0 solve."""
    touched = set()
    for fields in overrides.get("materials", {}).values():
        touched |= set(fields)
    return not (touched & _SIGMA0_FIELDS)


# ---------------------------------------------------------------------------
# Arm list
# ---------------------------------------------------------------------------
def section5_oat_arms(*, ks_stratum: str, gsi_stratum: str,
                      ks_frac: float = 0.20,
                      rain_mm_hr=(10.0, 20.0, 40.0),
                      gsi_levels=(40, 50, 60),
                      base_rain_mm_hr: float = 20.0) -> list[sens.Arm]:
    """OAT arms with overrides the solver consumes.

    `ks_stratum` replaces the plan's coal-seam K_s arm and `gsi_stratum` names
    the marl whose GSI moves. Both are REQUIRED: they are modelling choices.
    The GSI arm re-derives m_b, s, a AND E together through
    `sensitivity.derive_stratum`, and records `gsi` so the Material stays
    self-describing.
    """
    for name, tag in (("ks_stratum", ks_stratum), ("gsi_stratum", gsi_stratum)):
        if tag not in SECTION5_TAGS:
            raise KeyError(f"{name}={tag!r} is not in {SECTION5_TAGS}")
    arms = [sens.Arm("baseline", "base", "baseline", {})]

    for lvl, f in (("low", 1.0 - ks_frac), ("high", 1.0 + ks_frac)):
        arms.append(sens.Arm(f"{ks_stratum}_K_s", lvl,
                             f"{ks_stratum} K_s x{f:.2f}",
                             {"ks_multiplier": {ks_stratum: f}}))

    for r in rain_mm_hr:
        if r == base_rain_mm_hr:
            continue
        lvl = "low" if r < base_rain_mm_hr else "high"
        arms.append(sens.Arm("rainfall", lvl, f"rain {r:g} mm/hr",
                             {"rain_flux": r * MM_HR_TO_M_S}))

    key = UNIT_MAP.get(gsi_stratum, gsi_stratum)
    base = sens.BASELINE[key]
    for g in gsi_levels:
        if g == base["gsi"]:
            continue
        lvl = "low" if g < base["gsi"] else "high"
        p = sens.derive_stratum(gsi_stratum, **{**base, "gsi": g})
        arms.append(sens.Arm(f"{gsi_stratum}_GSI", lvl,
                             f"{gsi_stratum} GSI {g}",
                             {"materials": {gsi_stratum: {
                                 "m_b": p.m_b, "s": p.s, "a": p.a,
                                 "E": p.E_rm, "gsi": float(g)}}}))
    return arms


def section5_coupling_arms() -> list[sens.Arm]:
    """The three KC arms of D-3.3.2, with the key the sweep consumes."""
    return [sens.Arm(a.factor, a.level, a.label, {"kc": a.overrides["kc"]})
            for a in sens.coupling_arms()]


def arm_slug(arm: sens.Arm) -> str:
    return f"{arm.factor}_{arm.level}".replace(" ", "_")


def write_arm_files(arms, outdir: str) -> list[str]:
    """One JSON per arm: {factor, level, label, overrides}. Returns paths."""
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for arm in arms:
        validate(arm.overrides)
        p = os.path.join(outdir, f"{arm_slug(arm)}.json")
        with open(p, "w") as f:
            json.dump(dataclasses.asdict(arm), f, indent=1)
        paths.append(p)
    return paths


def load_arm_file(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    validate(d.get("overrides", {}))
    return d
