"""Scale consistency: nondim.py against properties.py and the Phase-1 dataset.

`properties.py` treats `Phase 1/isikdere_phase1_dataset.json` as the single
source of truth for material parameters, but NOTHING binds the JSON's
`nondimensionalisation.scales` block to `nondim.py`. They have drifted in
opposite directions:

    H_ref   nondim.py 165, JSON 30    -- nondim.py is AHEAD (deliberate)
    E_ref   nondim.py 1e9, JSON 1e8   -- nondim.py is BEHIND (stale)

Neither is a simple stale-value story, which is the point: an unbound pair
drifts both ways and nothing notices. `test_nondim_matches_the_phase1_dataset`
is the binding. Every difference must be listed in KNOWN_DIVERGENCES with a
reason, so a NEW divergence fails while documented ones pass.

The scales themselves are checked against what they are FOR: E_ref must make
u* order one across the adopted moduli, H_ref must make psi* order one across
the initial condition. A reference scale that does not do that is not a
reference scale.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.nondim import SCALES as S
from src.materials import load_materials
from src.loss import ic_targets

MATS = load_materials()

# E_ref is a scale for the strata that actually deform. Tm is the crystallised
# limestone basement; see test_E_ref_is_representative_of_the_adopted_moduli.
DEFORMING_STRATA = ("Mk", "Mk_d")

DATASET = (Path(__file__).resolve().parents[1]
           / "Phase 1" / "isikdere_phase1_dataset.json")

# Divergences between nondim.py and the Phase-1 dataset that are DELIBERATE.
# Each needs a reason. Anything not listed here is drift and fails.
KNOWN_DIVERGENCES: dict[str, str] = {}


@pytest.fixture(scope="module")
def dataset_scales():
    if not DATASET.exists():
        pytest.skip(f"Phase-1 dataset not found at {DATASET}")
    with open(DATASET) as f:
        return json.load(f)["nondimensionalisation"]["scales"]


# ---------------------------------------------------------------------------
# Internal consistency
# ---------------------------------------------------------------------------
def test_U_ref_is_derived_not_stored():
    """U_ref = sig_ref * L_ref / E_ref. If E_ref is edited and U_ref is not
    recomputed, every displacement is silently rescaled."""
    assert S.U_ref == pytest.approx(S.sig_ref * S.L_ref / S.E_ref, rel=1e-12)


def test_U_ref_tracks_a_changed_E_ref():
    """Negative control: __post_init__ must recompute, not use a stored value."""
    from src.nondim import Scales
    doubled = Scales(E_ref=2.0 * S.E_ref)
    assert doubled.U_ref == pytest.approx(0.5 * S.U_ref, rel=1e-12)


# ---------------------------------------------------------------------------
# The scales do what they are for
# ---------------------------------------------------------------------------
def test_E_ref_is_representative_of_the_adopted_moduli():
    """u* = u / U_ref = E_ref / E_actual, so E_ref must sit within an order of
    magnitude of every stratum's modulus or u* is not order one somewhere.

    Tm is excluded: at E = 4.26 GPa the crystallised limestone basement is
    112x stiffer than Mk_d, so no single E_ref puts all three within an order.
    Small u* in the basement is physically right -- it is effectively rigid and
    the failure mechanism runs through the marls. E_ref is a MARL scale, and
    the geometric mean of Mk and Mk_d is 8.9e7, i.e. the adopted 1e8.

    (The Phase-1 dataset's revision note says the moduli "span 8e6 to 2.09e8".
    That is wrong twice: it omits Tm at 4.26e9, and no stratum has E = 8e6.
    The conclusion it reaches is still correct.)
    """
    ratios = {tag: S.E_ref / m.E for tag, m in MATS.items()
              if tag in DEFORMING_STRATA}
    bad = {t: r for t, r in ratios.items() if not (0.1 <= r <= 10.0)}
    assert not bad, (
        f"E_ref = {S.E_ref:.3g} Pa gives u* = E_ref/E outside [0.1, 10] for "
        f"{ {t: round(r, 2) for t, r in bad.items()} }. Adopted moduli: "
        f"{ {t: f'{m.E:.3g}' for t, m in MATS.items()} }"
    )


def test_H_ref_makes_psi_star_order_one():
    """psi* over the initial condition should fill roughly [-1, 0].

    Too large and the field is compressed into a sliver of the network's
    output range; too small and psi* runs to -5 and the tanh saturates.
    """
    _, psi0_star = ic_targets()
    peak = float(np.abs(psi0_star.detach().cpu().numpy()).max())
    assert 0.3 <= peak <= 1.5, (
        f"max |psi*| = {peak:.3f} over the IC with H_ref = {S.H_ref} m. "
        f"H_ref should be close to the largest suction head in the domain."
    )


def test_pi_r_hz_follows_H_ref():
    assert S.Pi_R_hz == pytest.approx(S.H_ref / S.L_ref, rel=1e-12)


# ---------------------------------------------------------------------------
# The binding that was missing
# ---------------------------------------------------------------------------
def test_nondim_matches_the_phase1_dataset(dataset_scales):
    """Every scale in nondim.py must equal the Phase-1 dataset's value, or be
    listed in KNOWN_DIVERGENCES with a written reason.

    This is the test whose absence let H_ref and E_ref drift in opposite
    directions without anything noticing.
    """
    mapping = {
        "L_ref_m": S.L_ref,
        "T_ref_s": S.T_ref,
        "H_ref_m": S.H_ref,
        "sig_ref_Pa": S.sig_ref,
        "K_ref_mps": S.K_ref,
        "E_ref_Pa": S.E_ref,
        "rho_b_ref_kgpm3": S.rho_b_ref,
        "U_ref_m": S.U_ref,
    }

    drifted = {}
    for key, live in mapping.items():
        if key not in dataset_scales:
            continue
        recorded = float(dataset_scales[key])
        if live == pytest.approx(recorded, rel=1e-6):
            continue
        if key in KNOWN_DIVERGENCES:
            continue
        drifted[key] = (recorded, live)

    assert not drifted, (
        "nondim.py has drifted from the Phase-1 dataset:\n"
        + "\n".join(f"  {k}: dataset {d:.4g} vs nondim.py {l:.4g}"
                    for k, (d, l) in drifted.items())
        + "\nEither update the value, or add it to KNOWN_DIVERGENCES with a "
          "reason."
    )


def test_known_divergences_are_still_divergent(dataset_scales):
    """Housekeeping: once a divergence is resolved, its entry must be removed,
    or the allowlist quietly grows into a place where real drift can hide."""
    mapping = {"H_ref_m": S.H_ref, "E_ref_Pa": S.E_ref, "U_ref_m": S.U_ref}
    stale = [
        k for k in KNOWN_DIVERGENCES
        if k in mapping and k in dataset_scales
        and mapping[k] == pytest.approx(float(dataset_scales[k]), rel=1e-6)
    ]
    assert not stale, (
        f"{stale} no longer diverge from the dataset; remove them from "
        f"KNOWN_DIVERGENCES"
    )


def test_every_divergence_has_a_reason():
    for key, reason in KNOWN_DIVERGENCES.items():
        assert isinstance(reason, str) and len(reason) > 40, (
            f"KNOWN_DIVERGENCES[{key!r}] needs a real explanation"
        )

def test_tm_is_stiff_enough_to_justify_excluding_it():
    """Negative control for the exclusion above. If Tm ever gets a compliant
    modulus -- a fractured or weathered basement -- it rejoins the average and
    E_ref has to be rechosen."""
    tm = MATS["Tm"].E
    softest = min(MATS[t].E for t in DEFORMING_STRATA)
    assert tm > 20.0 * softest, (
        f"Tm at {tm:.3g} Pa is no longer far stiffer than the marls "
        f"({softest:.3g} Pa); it should not be excluded from E_ref"
    )
