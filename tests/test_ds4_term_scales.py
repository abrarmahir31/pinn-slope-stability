"""Pins the evidence D-S.4 REVISED rests on.

DECISIONS.md D-S.4 REVISED quotes a D*/G* table verbatim and draws three
conclusions from it. The table is pure parameter arithmetic -- no network, no
sampling, no randomness -- so unlike `test_the_measured_pde_seeds_are_not_
reproducible_across_draws`, which can only read its artifact, these tests
RECOMPUTE from `properties.py` and `nondim.py` and compare. That is the point:
the failure mode being guarded against is a silent parameter drift that leaves
the decision reading as measured when it is no longer true.

Regenerate the artifact with
    PYTHONPATH=. python scripts/ds4_term_scales.py \\
        --json docs/ds4_term_scales.json

If one of these fails, do not update the number. The decision's reasoning has
changed and the entry in DECISIONS.md has to change with it.
"""
import json
import pathlib

import pytest

from scripts.ds4_term_scales import PSI_M, ratios
from src.materials import load_materials
from src.nondim import SCALES

try:
    import geometry as g
except ModuleNotFoundError:
    from src.step21_geometry import geometry as g

Z_WT = 197.0                       # vg.psi_initial default
PSI_FLOOR = -(g.Z_BASE - Z_WT)     # -3.0 m: the WETTEST point in Section 5
UNITS = ("Mk", "Mk_d", "Tm")


@pytest.fixture(scope="module")
def artifact():
    p = (pathlib.Path(__file__).resolve().parents[1]
         / "docs" / "ds4_term_scales.json")
    if not p.exists():
        pytest.skip(f"{p.name} not present; regenerate with "
                    "scripts/ds4_term_scales.py --json")
    return json.loads(p.read_text())


@pytest.fixture(scope="module")
def live():
    mats = load_materials()
    return {(u, psi): ratios(psi, mats[u], SCALES)
            for u in UNITS for psi in PSI_M}


def test_the_artifact_matches_a_live_recomputation(artifact, live):
    """The drift guard. Reading the artifact alone would not catch this.

    A change to K_S, ALPHA, VG_N, THETA_S/THETA_R in `properties.py`, or to
    H_ref / L_ref / K_ref / T_ref / dtheta_ref in `nondim.py`, moves these
    numbers. The recorded table would stay put and D-S.4 REVISED would go on
    quoting figures nothing produces.
    """
    assert len(artifact["rows"]) == len(UNITS) * len(PSI_M)
    for row in artifact["rows"]:
        got = live[(row["unit"], row["psi_m"])]
        for key in ("K_star", "C_star", "D_over_S", "G_over_S", "D_over_G"):
            assert got[key] == pytest.approx(row[key], rel=1e-9), (
                f"{row['unit']} at psi = {row['psi_m']} m: {key} recomputes as "
                f"{got[key]:.6e} against a recorded {row[key]:.6e}. A material "
                "or scale parameter has changed; D-S.4 REVISED quotes the old "
                "value.")


def test_the_pi_group_ratio_itself_is_still_1_03(artifact):
    """What D-S.4 got RIGHT, and what the revision does not touch.

    The original decision's arithmetic is correct: Pi_R_grav / Pi_R_diff is
    exactly L_ref/H_ref. The revision rejects the INFERENCE from it, not the
    number, and that distinction matters -- a reader who concludes the scaling
    work was wrong has misread the entry.
    """
    assert artifact["Pi_ratio"] == pytest.approx(SCALES.L_ref / SCALES.H_ref)
    assert artifact["Pi_ratio"] == pytest.approx(1.030, abs=5e-4)


def test_neither_flux_term_is_o1_anywhere_in_section_5(live):
    """The core of the revision.

    Section 5 is unsaturated everywhere -- Z_WT = 197 m sits below
    Z_BASE = 200 m -- so psi_0 = -3 m at the floor is the wettest the initial
    condition ever gets. Evaluated there, both flux-to-storage ratios are
    orders below 1 in every unit. "Both Richards terms are O(1)" is false on
    the IC, which is the assumption `normalise="none"` was closed on.
    """
    assert PSI_FLOOR == pytest.approx(-3.0), (
        "Z_WT or Z_BASE has moved; every liveness argument in D-S.4 REVISED, "
        "sampling_front.py and inert_fraction.py is keyed to psi_0 = -3 m")

    for u in UNITS:
        r = live[(u, -3.0)]
        assert r["D_over_S"] < 0.1, f"{u}: diffusion/storage is now O(1) at the floor"
        assert r["G_over_S"] < 0.1, f"{u}: gravity/storage is now O(1) at the floor"


def test_the_marl_flux_terms_are_never_o1_at_any_saturation(live):
    """The stronger half of the claim, and the one worth a viva question.

    In Mk and Mk_d the flux terms do not reach 0.1 of storage even as psi -> 0.
    This is not "O(1) only near saturation" -- it is O(1) nowhere, at any
    water content, so no amount of wetting during the run makes the marl
    Richards residual a balance rather than a storage identity.
    """
    wettest = max(PSI_M)           # -0.01 m, the wet end of the tabulated range
    for u in ("Mk", "Mk_d"):
        r = live[(u, wettest)]
        assert r["D_over_S"] < 0.1, (
            f"{u}: diffusion/storage reaches {r['D_over_S']:.2e} at "
            f"psi = {wettest} m. The marls now carry a real diffusion term.")
        assert r["G_over_S"] < 0.1, (
            f"{u}: gravity/storage reaches {r['G_over_S']:.2e} at "
            f"psi = {wettest} m.")

    # Negative control: Tm DOES get there, so the assertions above are testing
    # a property of the marls and not of the threshold.
    assert live[("Tm", wettest)]["G_over_S"] > 1.0, (
        "Tm's gravity term no longer becomes O(1) near saturation; the "
        "marl-only claim above is no longer a distinction")


def test_diffusion_and_gravity_are_not_balanced_to_three_percent(live):
    """The struck sentence, pinned so it cannot creep back in.

    D-S.4 read "balanced to 3%" off the Pi ratio. The terms are not: D*/G*
    spans three orders across the domain and is nowhere near 1.03.
    """
    dg = [live[(u, psi)]["D_over_G"] for u in UNITS for psi in PSI_M]
    assert max(dg) < 0.5, (
        f"D*/G* now reaches {max(dg):.3g}; the two flux terms may be "
        "comparable after all and D-S.4 REVISED needs rereading")
    assert max(dg) / min(dg) > 100, (
        "D*/G* no longer spans orders across the domain, so 'the ratio of the "
        "terms is not the ratio of the coefficients' has lost its evidence")


def test_tm_only_becomes_flux_dominated_wetter_than_the_domain_floor(artifact):
    """Why Tm is the one unit that could go live, and only after wetting.

    The bisected crossings sit at psi = -1.6 mm (diffusion) and -0.39 m
    (gravity), both WETTER than psi_0 = -3 m. So on the initial condition Tm is
    storage-only like the marls; it is the rain BC driving psi up toward 0 that
    could change that, which is the same mechanism as the 18.1 m/day front in
    D-3.3.2's Day 27 addendum.
    """
    crit = artifact["psi_crit"]
    for u in ("Mk", "Mk_d"):
        for term in ("diffusion", "gravity"):
            assert crit[f"{u}_{term}"] is None, (
                f"{u} {term} now has a 0.1 crossing at "
                f"{crit[f'{u}_{term}']}; the marls are no longer O(1)-nowhere")

    for term in ("diffusion", "gravity"):
        psi_c = crit[f"Tm_{term}"]
        assert psi_c is not None
        assert psi_c > PSI_FLOOR, (
            f"Tm {term} crosses 0.1 at psi = {psi_c:.4f} m, which is DRIER "
            f"than the domain floor's {PSI_FLOOR} m. Tm would then be "
            "flux-dominated on the initial condition and D-S.4 REVISED's "
            "'neither is O(1) anywhere in Section 5' is wrong.")