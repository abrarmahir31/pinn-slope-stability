"""Tests for `src/sensitivity.py` — Step 6.2 support (Days 49–52).

Same convention as `test_strength.py`: every "is zero / is equal" claim is
paired with a control that would fail on a stub implementation.
"""
from __future__ import annotations

import math

import pytest

from src import sensitivity as sn


# ---------------------------------------------------------------------------
# Modulus derivation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tag,expected_E_rm", [
    ("marl_sekkoy",   2.08950e8),
    ("marl_weakzone", 3.8057e7),
    ("coal",          9.3431e7),
    ("tuffite",       3.2350e7),
    ("Tm",            4.2639e9),
])
def test_E_rm_reproduces_the_stored_moduli(tag, expected_E_rm):
    """Binds Hoek-Diederichs to every stored E in the Phase-1 dataset plus the
    Tm addendum. If `properties.py` is ever re-derived, this fails first."""
    p = sn.baseline_stratum(tag)
    assert p.E_rm == pytest.approx(expected_E_rm, rel=1e-3)


def test_E_rm_is_not_simply_the_intact_modulus():
    """Negative control: the formula must actually reduce E_i, not pass it
    through. At D = 1 the reduction is an order of magnitude."""
    p = sn.baseline_stratum("marl_sekkoy")
    assert p.E_rm < 0.1 * p.E_i


def test_E_rm_rises_with_gsi():
    prev = None
    for gsi in (30, 40, 50, 60, 70):
        E = sn.E_rm_hoek_diederichs(3.1325e9, gsi, 1.0)
        if prev is not None:
            assert E > prev
        prev = E


def test_E_rm_falls_with_disturbance():
    """D is blast damage; more damage cannot stiffen the mass."""
    assert (sn.E_rm_hoek_diederichs(3.1325e9, 50, 1.0)
            < sn.E_rm_hoek_diederichs(3.1325e9, 50, 0.0))


def test_intact_modulus_is_invariant_to_gsi():
    """E_i = MR * sigma_ci is a property of the intact rock. Only E_rm moves."""
    lo = sn.derive_stratum("marl_sekkoy",
                           **{**sn.BASELINE["marl_sekkoy"], "gsi": 40})
    hi = sn.derive_stratum("marl_sekkoy",
                           **{**sn.BASELINE["marl_sekkoy"], "gsi": 60})
    assert lo.E_i == pytest.approx(hi.E_i, rel=1e-15)
    assert lo.E_rm != pytest.approx(hi.E_rm, abs=1.0)


def test_gsi_arm_moves_modulus_and_strength_together():
    """The guard named in the module docstring. Perturbing GSI must move all
    four derived quantities; holding E_rm fixed suppresses the Kozeny-Carman
    feedback that Day 51 exists to measure."""
    lo = sn.derive_stratum("marl_sekkoy",
                           **{**sn.BASELINE["marl_sekkoy"], "gsi": 40})
    hi = sn.derive_stratum("marl_sekkoy",
                           **{**sn.BASELINE["marl_sekkoy"], "gsi": 60})
    assert hi.m_b > lo.m_b
    assert hi.s > lo.s
    assert hi.a < lo.a              # `a` falls as GSI rises
    assert hi.E_rm > lo.E_rm


def test_the_gsi_arm_spans_a_factor_of_three_in_modulus():
    """Magnitude check on the claim in the docstring. If this shrinks, the
    warning about holding E_rm fixed has stopped being load-bearing."""
    lo = sn.derive_stratum("marl_sekkoy",
                           **{**sn.BASELINE["marl_sekkoy"], "gsi": 40})
    hi = sn.derive_stratum("marl_sekkoy",
                           **{**sn.BASELINE["marl_sekkoy"], "gsi": 60})
    assert hi.E_rm / lo.E_rm > 2.5


def test_derived_ghb_matches_strength_module():
    p = sn.baseline_stratum("marl_sekkoy")
    g = p.ghb()
    assert g.m_b == pytest.approx(0.1125, rel=1e-3)
    assert g.s == pytest.approx(2.404e-4, rel=1e-3)


# ---------------------------------------------------------------------------
# OAT arm construction
# ---------------------------------------------------------------------------
def test_oat_arms_contain_exactly_one_baseline_first():
    arms = sn.oat_arms()
    assert arms[0].is_base
    assert sum(a.is_base for a in arms) == 1


def test_oat_arms_do_not_resolve_the_base_level_twice():
    """Rainfall 20 mm/hr and GSI 50 ARE the baseline; re-solving them wastes a
    lab-PC day each."""
    arms = sn.oat_arms()
    labels = [a.label for a in arms]
    assert "rain 20 mm/hr" not in labels
    assert "marl_sekkoy GSI 50" not in labels


def test_oat_ks_arms_are_symmetric_about_one():
    arms = [a for a in sn.oat_arms() if a.factor == "coal_K_s"]
    facs = sorted(a.overrides["ks_multiplier"]["coal"] for a in arms)
    assert facs[0] == pytest.approx(0.8)
    assert facs[1] == pytest.approx(1.2)


def test_gsi_arms_carry_all_four_derived_overrides():
    arms = [a for a in sn.oat_arms() if a.factor == "marl_GSI"]
    assert arms
    for a in arms:
        ov = a.overrides["stratum_overrides"]["marl_sekkoy"]
        assert set(ov) == {"m_b", "s", "a", "E"}


def test_coupling_arms_are_three_not_two():
    """DECISIONS.md specifies KC-everywhere, KC-default and KC-off."""
    arms = sn.coupling_arms()
    assert len(arms) == 3
    assert sum(a.is_base for a in arms) == 1
    assert {a.overrides["kc"] for a in arms} == {"default", "all", "off"}


# ---------------------------------------------------------------------------
# Tornado assembly
# ---------------------------------------------------------------------------
def _demo_results():
    return {
        ("baseline", "base"): 1.30,
        ("coal_K_s", "low"): 1.28,   ("coal_K_s", "high"): 1.33,
        ("rainfall", "low"): 1.30,   ("rainfall", "high"): 1.30,
        ("marl_GSI", "low"): 1.05,   ("marl_GSI", "high"): 1.52,
    }


def test_tornado_orders_by_span_widest_first():
    bars = sn.tornado(_demo_results())
    assert [b.factor for b in bars] == ["marl_GSI", "coal_K_s", "rainfall"]
    assert bars[0].span > bars[-1].span


def test_tornado_percentages_are_signed_relative_to_base():
    bars = {b.factor: b for b in sn.tornado(_demo_results())}
    g = bars["marl_GSI"]
    assert g.low_pct < 0 < g.high_pct
    assert g.high_pct == pytest.approx(100.0 * (1.52 - 1.30) / 1.30)


def test_tornado_represents_an_insensitive_factor_as_a_flat_bar():
    """Finding 1 predicts the rainfall factor comes out flat. It must survive
    the assembly as a zero-span bar, not get dropped."""
    bars = {b.factor: b for b in sn.tornado(_demo_results())}
    assert "rainfall" in bars
    assert bars["rainfall"].span == pytest.approx(0.0, abs=1e-12)


def test_tornado_requires_a_baseline():
    r = _demo_results()
    del r[("baseline", "base")]
    with pytest.raises(ValueError):
        sn.tornado(r)


def test_tornado_handles_a_one_sided_factor():
    r = _demo_results()
    del r[("coal_K_s", "low")]
    bars = {b.factor: b for b in sn.tornado(r)}
    assert bars["coal_K_s"].low == bars["coal_K_s"].high


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def test_seed_statistics_uses_the_sample_sd():
    """n-1, not n. At five seeds the two differ by 12%, the same order as the
    effects the tornado plot ranks."""
    vals = [1.30, 1.32, 1.28, 1.31, 1.29]
    s = sn.seed_statistics(vals)
    assert s.n == 5
    assert s.mean == pytest.approx(1.30)
    assert s.sd == pytest.approx(0.0158113883, rel=1e-6)


def test_seed_statistics_is_not_the_population_sd():
    """Negative control on the above."""
    vals = [1.30, 1.32, 1.28, 1.31, 1.29]
    import numpy as np
    assert sn.seed_statistics(vals).sd != pytest.approx(
        float(np.std(vals)), abs=1e-6)


def test_seed_statistics_rejects_a_single_run():
    with pytest.raises(ValueError):
        sn.seed_statistics([1.30])


def test_convergence_check_accepts_a_settled_sequence():
    ok, rel = sn.convergence_check([5000, 10000, 20000, 50000],
                                   [1.42, 1.34, 1.31, 1.307])
    assert ok
    assert len(rel) == 3


def test_convergence_check_rejects_late_drift():
    """An early plateau followed by drift is not convergence. Averaging the
    relative changes would call this converged; using the last one does not."""
    ok, rel = sn.convergence_check([5000, 10000, 20000, 50000],
                                   [1.30, 1.301, 1.302, 1.44])
    assert not ok
    assert rel[-1] > rel[0]


def test_convergence_check_requires_increasing_refinement():
    with pytest.raises(ValueError):
        sn.convergence_check([50000, 10000], [1.3, 1.3])


# ---------------------------------------------------------------------------
# Coupling effect
# ---------------------------------------------------------------------------
def test_coupling_effect_is_positive_when_one_way_is_unconservative():
    assert sn.coupling_effect(fos_two_way=1.30, fos_one_way=1.58) == \
        pytest.approx(21.538, rel=1e-3)


def test_coupling_effect_is_zero_when_the_arms_agree():
    assert sn.coupling_effect(1.30, 1.30) == pytest.approx(0.0, abs=1e-30)


def test_coupling_effect_signs_the_other_direction():
    """Negative control, and the case the docstring warns not to report as a
    magnitude: feedback stiffening the response."""
    assert sn.coupling_effect(fos_two_way=1.30, fos_one_way=1.10) < 0.0