"""src/arms.py -- Section 5 arms and the solver hooks they need."""
import dataclasses

import numpy as np
import pytest

from src import arms
from src import sensitivity as sens
from src.materials import load_materials
from src.step21_geometry import boundaries as bnd


@pytest.fixture
def mats():
    return load_materials()


def test_validate_rejects_unknown_keys_and_the_coal_seam():
    with pytest.raises(KeyError):
        arms.validate({"stratum_overrides": {}})
    with pytest.raises(KeyError):
        arms.validate({"ks_multiplier": {"coal": 1.2}})
    with pytest.raises(KeyError):
        arms.validate({"materials": {"Mk": {"not_a_field": 1.0}}})


def test_validate_rejects_k_s_set_twice():
    with pytest.raises(ValueError):
        arms.validate({"materials": {"Tm": {"K_s": 1e-6}},
                       "ks_multiplier": {"Tm": 1.2}})


def test_material_overrides_do_not_mutate_the_input(mats):
    out = arms.apply_material_overrides(mats, {"ks_multiplier": {"Tm": 1.2}})
    assert out["Tm"].K_s == pytest.approx(1.2 * mats["Tm"].K_s)
    assert mats["Tm"].K_s == load_materials()["Tm"].K_s
    assert out["Mk"] is mats["Mk"]


def test_boundary_overrides_align_the_flux_cap_with_the_material(mats):
    ov = {"ks_multiplier": {"Tm": 0.8}}
    m2 = arms.apply_material_overrides(mats, ov)
    base_cap = bnd.K_SAT["Tm"]
    with arms.boundary_overrides(m2, ov):
        assert bnd.K_SAT["Tm"] == pytest.approx(0.8 * base_cap)
        assert bnd.K_SAT["Tm"] == m2["Tm"].K_s
    assert bnd.K_SAT["Tm"] == base_cap


def test_boundary_overrides_restore_after_an_exception(mats):
    ov = {"rain_flux": 1.0e-5}
    base = bnd.RAIN_FLUX
    with pytest.raises(RuntimeError):
        with arms.boundary_overrides(mats, ov):
            assert bnd.RAIN_FLUX == 1.0e-5
            raise RuntimeError
    assert bnd.RAIN_FLUX == base


def _rain_segments():
    out = {}
    for seg in ("natural_ground", "bench"):
        pts = np.asarray(bnd.sample_boundaries(2000, seed=1)[seg])
        tag = np.asarray(bnd.g.material_tag(pts[:, 0], pts[:, 1])).astype(str)
        out[seg] = (pts, tag)
    return out


def test_rain_override_reaches_flux_bc_below_the_cap(mats):
    # The hook works: below K_s * n_z the applied flux follows the rain.
    for seg, (pts, tag) in _rain_segments().items():
        with arms.boundary_overrides(mats, {"rain_flux": 1.0e-10}):
            q_lo = bnd.flux_bc(seg, pts, tag=tag)
        with arms.boundary_overrides(mats, {"rain_flux": 2.0e-10}):
            q_hi = bnd.flux_bc(seg, pts, tag=tag)
        assert np.all(q_hi > q_lo)


def test_rainfall_arms_are_currently_identical_boundary_conditions(mats):
    """FINDING (Day 40), not a property to preserve. Both rainfall segments
    are entirely Mk_d at K_s = 1e-9 m/s, so the capacity cap binds everywhere
    and 10 / 20 / 40 mm/hr give bit-identical fluxes. The rainfall OAT arm
    cannot move FOS under INFILTRATION_MODE = "capacity_limited". This test
    FAILS the day that changes -- then the rainfall arm is worth running."""
    for seg, (pts, tag) in _rain_segments().items():
        assert set(tag) == {"Mk_d"}
        qs = []
        for r in (10.0, 20.0, 40.0):
            with arms.boundary_overrides(
                    mats, {"rain_flux": r * arms.MM_HR_TO_M_S}):
                qs.append(bnd.flux_bc(seg, pts, tag=tag))
        assert np.array_equal(qs[0], qs[1]) and np.array_equal(qs[1], qs[2])


def test_baseline_rain_is_twenty_mm_per_hour():
    assert 20.0 * arms.MM_HR_TO_M_S == pytest.approx(bnd.RAIN_FLUX, rel=1e-3)


def test_section5_arms_require_explicit_strata():
    with pytest.raises(TypeError):
        arms.section5_oat_arms()
    with pytest.raises(KeyError):
        arms.section5_oat_arms(ks_stratum="coal", gsi_stratum="Mk")


def test_section5_arms_have_one_baseline_first_and_no_coal():
    a = arms.section5_oat_arms(ks_stratum="Tm", gsi_stratum="Mk")
    assert a[0].is_base and sum(x.is_base for x in a) == 1
    assert not any("coal" in x.factor for x in a)
    for x in a:
        arms.validate(x.overrides)


def test_gsi_arm_matches_the_derivation_chain():
    a = arms.section5_oat_arms(ks_stratum="Tm", gsi_stratum="Mk")
    lo = next(x for x in a if x.factor == "Mk_GSI" and x.level == "low")
    p = sens.derive_stratum("Mk", **{**sens.BASELINE["marl_sekkoy"], "gsi": 40})
    f = lo.overrides["materials"]["Mk"]
    assert f["E"] == pytest.approx(p.E_rm) and f["m_b"] == pytest.approx(p.m_b)
    assert f["gsi"] == 40.0


def test_gsi_base_level_reproduces_the_solver_material(mats):
    # Binds sensitivity.BASELINE to properties via UNIT_MAP.
    p = sens.derive_stratum("Mk", **sens.BASELINE["marl_sekkoy"])
    assert p.E_rm == pytest.approx(mats["Mk"].E, rel=1e-4)
    assert p.m_b == pytest.approx(mats["Mk"].m_b, rel=1e-3)


def test_sigma0_consistency_flags_modulus_arms_only():
    a = arms.section5_oat_arms(ks_stratum="Tm", gsi_stratum="Mk_d")
    for x in a:
        moves_E = "materials" in x.overrides
        assert arms.sigma0_consistent(x.overrides) is (not moves_E)


def test_coupling_arms_are_three_and_consumable():
    a = arms.section5_coupling_arms()
    assert sorted(x.overrides["kc"] for x in a) == ["all", "default", "off"]
    assert arms.kc_config("off").enabled is False


def test_arm_files_round_trip(tmp_path):
    a = arms.section5_oat_arms(ks_stratum="Tm", gsi_stratum="Mk")
    paths = arms.write_arm_files(a, str(tmp_path))
    assert len(paths) == len(set(paths)) == len(a)
    d = arms.load_arm_file(paths[-1])
    assert d["overrides"] == a[-1].overrides
