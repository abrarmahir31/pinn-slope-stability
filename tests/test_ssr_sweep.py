"""scripts/ssr_sweep.py -- Day 40.

`_has_failed` decides every FOS in the thesis, so it gets a full truth table
with a negative control per condition. `_diagnostics` is pinned to the
penalty's own stress (the Day-39 version tested the network increment alone).
One tiny end-to-end run guards the wiring bugs that only show up when the
pieces meet: rebuilding the net from the checkpoint, the criterion reaching
the loss, and the objective matching training.
"""
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import ssr_sweep as S
from scripts import train
from scripts.grad_norm_table import losses_at
from src import strength as st
from src.coupling import KC_DEFAULT, KC_OFF
from src.loss import L_yield
from src.materials import load_materials
from src.weighting import TERMS

REQ = ["--w-yield", "1", "--yield-norm", "raw", "--admissible-metric", "area",
       "--admissible-drop", "0.1"]


def _args(**kw):
    base = dict(criterion="GHB", sig3_lo=0.0, sig3_hi=8.5e5,
                mc_strength=None, mc_sig3max=None,
                plateau_factor=20.0, disp_factor=10.0, admissible_drop=0.10,
                admissible_metric="area")
    base.update(kw)
    return SimpleNamespace(**base)


# --- parse: decisions have no defaults -------------------------------------

def test_parse_requires_the_decision_flags():
    with pytest.raises(SystemExit):
        S.parse(["--criterion", "MC", "--baseline", "x", "--out", "o"])


def test_parse_refuses_ghb_without_a_sig3_range():
    with pytest.raises(SystemExit):
        S.parse(["--criterion", "GHB", "--baseline", "x", "--out", "o", *REQ])


def test_parse_refuses_a_sig3_range_for_mc():
    with pytest.raises(SystemExit):
        S.parse(["--criterion", "MC", "--baseline", "x", "--out", "o", *REQ,
                 "--mc-strength", "rockmass", "--mc-sig3max", "1.79e5",
                 "--sig3-lo", "0", "--sig3-hi", "1e5"])


def test_parse_refuses_mc_without_a_strength_choice():
    with pytest.raises(SystemExit):
        S.parse(["--criterion", "MC", "--baseline", "x", "--out", "o", *REQ])


def test_parse_refuses_rockmass_mc_without_sig3max():
    with pytest.raises(SystemExit):
        S.parse(["--criterion", "MC", "--baseline", "x", "--out", "o", *REQ,
                 "--mc-strength", "rockmass"])


def test_parse_accepts_a_complete_rockmass_mc_command():
    a = S.parse(["--criterion", "MC", "--baseline", "x", "--out", "o", *REQ,
                 "--mc-strength", "rockmass", "--mc-sig3max", "1.79e5"])
    assert a.mc_sig3max == 1.79e5


def test_yield_params_rockmass_mc_are_per_material_hoek2002():
    mats = load_materials()
    a = _args(criterion="MC", mc_strength="rockmass", mc_sig3max=1.79e5)
    yp = S._yield_params(a, mats, 1.0)
    for t, m in mats.items():
        eq = st.ghb_equivalent_mc(st.GHBParams(sigma_ci=m.sigma_ci, m_b=m.m_b,
                                               s=m.s, a=m.a), 1.79e5)
        assert yp[t].c == pytest.approx(eq.c)
        assert yp[t].phi == pytest.approx(eq.phi)
    assert yp["Tm"].c > yp["Mk_d"].c            # negative control: not one pair
    assert yp["Mk_d"].c > S.MC_DESIGN.c


def test_yield_params_mc_refuses_an_unset_strength():
    with pytest.raises(ValueError):
        S._yield_params(_args(criterion="MC"), load_materials(), 1.0)


def test_parse_accepts_a_complete_ghb_command():
    a = S.parse(["--criterion", "GHB", "--baseline", "x", "--out", "o", *REQ,
                 "--sig3-lo", "0", "--sig3-hi", "8.5e5"])
    assert S._sig3_range(a) == (0.0, 8.5e5)


# --- _sig3_range / _yield_params ---------------------------------------------

def test_sig3_range_is_none_for_mc_and_refuses_missing_for_ghb():
    assert S._sig3_range(_args(criterion="MC")) is None
    with pytest.raises(ValueError):
        S._sig3_range(_args(sig3_hi=None))


def test_yield_params_mc_reduces_the_design_pair_for_every_tag():
    mats = load_materials()
    yp = S._yield_params(_args(criterion="MC", mc_strength="bedding"), mats, 1.5)
    assert set(yp) == set(mats)
    exp = st.reduce_mc(S.MC_DESIGN, 1.5)
    for p in yp.values():
        assert p.c == pytest.approx(exp.c) and p.phi == pytest.approx(exp.phi)
    assert exp.c == pytest.approx(4400.0 / 1.5)


def test_yield_params_ghb_are_per_material_and_move_with_srf():
    mats = load_materials()
    a = _args()
    y1, y2 = S._yield_params(a, mats, 1.0), S._yield_params(a, mats, 1.5)
    assert y1["Mk_d"].sigma_ci != y1["Tm"].sigma_ci
    for t in mats:                               # negative control
        assert (y2[t].m_b, y2[t].s) != (y1[t].m_b, y1[t].s)


def test_yield_params_ghb_depend_on_the_fit_range():
    mats = load_materials()
    a = S._yield_params(_args(), mats, 1.5)["Mk_d"]
    b = S._yield_params(_args(sig3_hi=2.0e5), mats, 1.5)["Mk_d"]
    assert (a.m_b, a.s) != (b.m_b, b.s)


# --- admissible metric --------------------------------------------------------

def test_admissible_metric_variants():
    adm = {"Mk": 1.0, "Mk_d": 0.45, "Tm": 1.0}
    sh = {"Mk": 0.03, "Mk_d": 0.10, "Tm": 0.87}
    assert S._admissible_metric(adm, sh, "area") == pytest.approx(0.945)
    assert S._admissible_metric(adm, sh, "min_tag") == 0.45
    assert S._admissible_metric(adm, sh, "tag:Mk_d") == 0.45
    with pytest.raises(ValueError):
        S._admissible_metric(adm, sh, "tag:coal")


# --- _has_failed: truth table ---------------------------------------------------

REF = {"total": 1.0, "disp": 1.0, "admissible": 0.45}
FAIL = {"total": 30.0, "disp": 20.0, "admissible": 0.30}


def test_has_failed_when_all_three_conditions_hold():
    assert S._has_failed(FAIL, REF, _args()) is True


@pytest.mark.parametrize("key,value", [("total", 19.0), ("disp", 9.0),
                                       ("admissible", 0.40)])
def test_has_failed_is_false_if_any_single_condition_is_missing(key, value):
    assert S._has_failed({**FAIL, key: value}, REF, _args()) is False


def test_has_failed_is_false_at_the_reference_state():
    assert S._has_failed(REF, REF, _args()) is False


def test_has_failed_boundaries_are_strict():
    edge = {"total": 20.0, "disp": 10.0, "admissible": 0.35}
    assert S._has_failed(edge, REF, _args()) is False


def test_admissible_condition_is_relative_to_the_reference():
    # Day 40: Mk_d starts ~45% admissible. The Day-39 absolute floor (0.80)
    # would have called this unchanged state a failure at SRF = 1.
    same = {**FAIL, "admissible": REF["admissible"]}
    assert S._has_failed(same, REF, _args()) is False
    assert REF["admissible"] < 0.80


# --- objective and diagnostics on real collocation ------------------------------

@pytest.fixture(scope="module")
def tiny():
    t = train.parse(["--layers", "2", "--width", "16", "--n-pde", "60",
                     "--n-bc", "400", "--n-iface", "20", "--ansatz", "exp",
                     "--seed", "3"])
    net, loss_fn, coll = train.setup(t)
    return t, net, loss_fn, coll


def test_losses_match_training_assembly(tiny):
    _, net, loss_fn, coll = tiny
    for kc, fb in ((KC_DEFAULT, True), (KC_OFF, False)):
        a = S._losses(net, loss_fn, coll, kc)
        b = losses_at(net, loss_fn, coll, feedback=fb)
        assert set(a) == set(b) == set(TERMS)
        for k in a:
            assert float(a[k].detach()) == pytest.approx(float(b[k].detach()), rel=1e-12, abs=0)


def test_frozen_total_is_the_weighted_sum_and_refuses_missing_terms():
    L = {k: torch.tensor(float(i + 1)) for i, k in enumerate(TERMS)}
    w = {k: 2.0 for k in TERMS}
    assert float(S._frozen_total(L, w)) == pytest.approx(2.0 * sum(
        range(1, len(TERMS) + 1)))
    with pytest.raises(KeyError):
        S._frozen_total({k: v for k, v in L.items() if k != "ic_disp"}, w)


def test_diagnostics_are_computed_on_the_penalty_stress(tiny):
    t, net, loss_fn, coll = tiny
    mats = loss_fn["mats"]
    yp = S._yield_params(_args(), mats, 1.0)
    d = S._diagnostics(net, coll, mats, yp, "GHB", "area")
    _, parts = L_yield(net, coll, mats, yp, criterion="GHB", per_tag=True)
    for tag, v in d["admissible_by_tag"].items():
        assert v == pytest.approx(float(parts[f"admissible_{tag}"]))
    assert sum(d["area_shares"].values()) == pytest.approx(1.0)


def test_diagnostics_change_when_sigma0_is_removed(tiny):
    # Negative control for the Day-39 bug: zeroing sigma_0 must change what
    # is measured. If it did not, the check would not be seeing sigma_0.
    t, net, loss_fn, coll = tiny
    mats = loss_fn["mats"]
    yp = S._yield_params(_args(), mats, 1.0)
    with_s0 = S._diagnostics(net, coll, mats, yp, "GHB", "area")
    saved = coll.sigma0
    try:
        coll.sigma0 = torch.zeros_like(saved)
        no_s0 = S._diagnostics(net, coll, mats, yp, "GHB", "area")
    finally:
        coll.sigma0 = saved
    assert with_s0["pde_yield"] != pytest.approx(no_s0["pde_yield"])


# --- rebuilding the network from the checkpoint -----------------------------------

def test_train_args_follow_the_checkpoint_not_train_defaults():
    cfg = {"layers": 8, "width": 64, "ansatz": "exp", "eps_psi": 0.3,
           "cap_k": 100.0, "n_pde": 10000, "n_bc": 3000, "n_iface": 500,
           "seed": 7, "t_max": 30.0}
    a = SimpleNamespace(n_pde=None, n_bc=None, n_iface=None,
                        sampling_seed=None, device="cpu")
    t = S._train_args(cfg, a)
    assert (t.layers, t.ansatz, t.n_pde, t.seed) == (8, "exp", 10000, 7)
    a.n_pde = 300
    assert S._train_args(cfg, a).n_pde == 300


def test_train_args_refuse_a_checkpoint_without_its_ansatz():
    a = SimpleNamespace(n_pde=None, n_bc=None, n_iface=None,
                        sampling_seed=None, device="cpu")
    with pytest.raises(KeyError):
        S._train_args({"layers": 8, "width": 64, "eps_psi": 0.3}, a)


def test_kc_follows_the_checkpoint_unless_an_arm_overrides_it():
    assert S._kc_name({"no_feedback": False}, {}) == "default"
    assert S._kc_name({"no_feedback": True}, {}) == "off"
    assert S._kc_name({"no_feedback": False}, {"kc": "all"}) == "all"


# --- end to end, tiny ------------------------------------------------------------------

@pytest.mark.parametrize("criterion", ["MC", "GHB"])
def test_smoke_run_end_to_end(tmp_path, tiny, criterion):
    t, net, _, _ = tiny
    ck = tmp_path / "ckpt.pt"
    torch.save({"net": net.state_dict(), "bal": {"w": {k: 1.0 for k in TERMS}},
                "cfg": {**vars(t), "no_feedback": False}, "step": 0}, ck)
    out = tmp_path / f"ssr_{criterion}"
    argv = ["--criterion", criterion, "--baseline", str(ck), "--out", str(out),
            "--device", "cpu", "--smoke", *REQ]
    if criterion == "GHB":
        argv += ["--sig3-lo", "0", "--sig3-hi", "8.5e5"]
    else:
        argv += ["--mc-strength", "rockmass", "--mc-sig3max", "1.79e5"]
    res = S.run(S.parse(argv))

    recs = [json.loads(l) for l in open(out / "sweep.jsonl")]
    assert [r["srf"] for r in recs] == pytest.approx([1.0, 1.05, 1.10, 1.15])
    assert all(r["finite"] and 0.0 <= r["admissible"] <= 1.0 for r in recs)
    assert len(list((out / "states").glob("srf_*.pt"))) == 4
    assert res["criterion"] == criterion and res["kc"] == "default"
    assert res["sig3_range"] == ([0.0, 8.5e5] if criterion == "GHB" else None) \
        or res["sig3_range"] == ((0.0, 8.5e5) if criterion == "GHB" else None)
    saved = json.load(open(out / "result.json"))
    for k in ("admissible_metric", "admissible_drop", "w_yield", "yield_norm",
              "plateau_factor", "disp_factor", "status",
              "admissible_secondary", "admissible_secondary_ref",
              "baseline_sha256", "baseline_step"):
        assert k in saved
    assert saved["admissible_metric_role"].startswith("primary")
    assert "not used" in saved["admissible_secondary_role"]
    assert all("admissible_secondary" in r for r in recs)
    # Resume: a second invocation re-uses every point and adds none.
    S.run(S.parse(argv))
    assert len(open(out / "sweep.jsonl").readlines()) == 4


def test_reachable_drop_exposes_an_area_threshold_that_cannot_fire():
    # baseline-v1 shares and GHB admissibility (decision_evidence.py, D = 1)
    adm = {"Mk": 0.999, "Mk_d": 0.389, "Tm": 1.0}
    sh = {"Mk": 0.0302, "Mk_d": 0.0980, "Tm": 0.8719}
    r = S.reachable_drop(adm, sh, "area")
    assert r["dominant_tag"] == "Tm"
    assert r["reachable_without_dominant"] == pytest.approx(
        0.0302 * 0.999 + 0.0980 * 0.389, rel=1e-9)
    assert r["reachable_without_dominant"] < 0.10


def test_reachable_drop_for_a_single_marl_tag_is_its_full_value():
    # Control: tag:Mk_d does not depend on Tm, so everything is reachable.
    adm = {"Mk": 0.999, "Mk_d": 0.389, "Tm": 1.0}
    sh = {"Mk": 0.0302, "Mk_d": 0.0980, "Tm": 0.8719}
    assert S.reachable_drop(adm, sh, "tag:Mk_d")["reachable_without_dominant"] \
        == pytest.approx(0.389)


# --- D-5.5: primary and secondary metric ---------------------------------------

def test_parse_defaults_follow_d55():
    a = S.parse(["--criterion", "GHB", "--baseline", "x", "--out", "o",
                 "--w-yield", "1", "--yield-norm", "ref",
                 "--admissible-drop", "0.1", "--sig3-lo", "0", "--sig3-hi", "1.79e5"])
    assert a.admissible_metric == "min_tag"
    assert a.admissible_secondary == "tag:Mk_d"


def test_parse_rejects_an_invalid_secondary_and_accepts_none():
    base = ["--criterion", "GHB", "--baseline", "x", "--out", "o", *REQ,
            "--sig3-lo", "0", "--sig3-hi", "1.79e5"]
    with pytest.raises(SystemExit):
        S.parse(base + ["--admissible-secondary", "tag:coal"])
    assert S.parse(base + ["--admissible-secondary", "none"]).admissible_secondary == "none"


def test_diagnostics_record_the_secondary_without_changing_the_primary(tiny):
    t, net, loss_fn, coll = tiny
    mats = loss_fn["mats"]
    yp = S._yield_params(_args(), mats, 1.0)
    d1 = S._diagnostics(net, coll, mats, yp, "GHB", "min_tag", secondary="tag:Mk_d")
    d0 = S._diagnostics(net, coll, mats, yp, "GHB", "min_tag", secondary=None)
    assert d1["admissible"] == d0["admissible"]
    assert d1["admissible_secondary"] == pytest.approx(d1["admissible_by_tag"]["Mk_d"])
    assert d0["admissible_secondary"] is None


def test_has_failed_ignores_the_secondary_metric():
    rec = {**FAIL, "admissible": REF["admissible"], "admissible_secondary": 0.0}
    assert S._has_failed(rec, {**REF, "admissible_secondary": 1.0}, _args()) is False
