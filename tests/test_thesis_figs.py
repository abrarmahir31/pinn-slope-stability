"""Figs 2 and 3, check_baseline and progress (Day 42)."""
import json
import os

import pytest
import torch

from scripts import check_baseline as CB
from scripts import make_fig2 as F2
from scripts import make_fig3 as F3
from scripts import progress as P
from scripts import train
from src.weighting import TERMS


def test_fig3_segment_kinds_come_from_the_live_bc_function():
    k = F3.segment_kinds()
    assert k["base"] == "fixed"
    assert k["far_field_f1"] == "roller_x" and k["pit_floor"] == "roller_x"
    assert {k[s] for s in ("natural_ground", "bench", "cut_face")} == {"traction-free"}


def test_fig3_writes_and_reports_stratified_shares(tmp_path):
    out = tmp_path / "fig3.png"
    assert F3.main(["--n-interior", "600", "--n-boundary", "120",
                    "--out", str(out)]) == 0
    meta = json.load(open(tmp_path / "fig3.json"))
    sh = meta["shares"]
    assert sh["Mk_d"]["points"] > sh["Mk_d"]["area"]      # oversampled
    assert sum(v["area"] for v in sh.values()) == pytest.approx(1.0)
    assert out.stat().st_size > 20000


def test_fig2_text_comes_from_cfg_and_terms():
    cfg = {"layers": 8, "width": 64, "ansatz": "cap", "eps_psi": 0.3,
           "n_pde": 10000, "n_bc": 3000, "epochs": 100000, "lbfgs_epochs": 500,
           "every": 25, "lam": 0.3}
    t = F2.describe(cfg)
    assert "8 x 64" in t["net"] and "cap" in t["ansatz"]
    assert "100,000" in t["optim"] and "L-BFGS 500" in t["optim"]
    assert all(term in t["losses"] for term in TERMS)


def test_fig2_refuses_a_cfg_missing_the_architecture():
    with pytest.raises(KeyError):
        F2.describe({"layers": 8})


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    t = train.parse(["--layers", "2", "--width", "16", "--ansatz", "exp"])
    net, _, _ = train.setup(t)
    d = tmp_path_factory.mktemp("run")
    torch.save({"net": net.state_dict(), "cfg": vars(t), "step": 3},
               d / "ckpt_final.pt")
    rec = {"step": 3, "L": {k: 1e-3 for k in TERMS},
           "L0": {k: 1.0 for k in TERMS}, "psi_max_m": -2.0,
           "psi_sat_frac": 0.0, "sec_per_epoch": 0.1}
    (d / "log.jsonl").write_text(json.dumps(rec) + "\n")
    return str(d)


def test_check_baseline_reports_drift_and_ratios(tiny_run):
    r = CB.report(tiny_run, 30.0)
    assert set(r["drift"]) == {"t0", "t30"}
    assert r["loss_ratios"]["bc_mech"] == pytest.approx(1e-3)
    assert CB.facts(r) == []


def test_check_baseline_flags_the_rejected_ansatz():
    r = {"ansatz": "unbounded", "loss_ratios": {}}
    assert any("D-A.1" in b for b in CB.facts(r))
    assert CB.facts({"ansatz": None, "loss_ratios": {}})


def test_progress_handles_a_run_with_nothing_finished(tmp_path):
    (tmp_path / "r").mkdir()
    s = P.summarise(str(tmp_path / "r"))
    assert s["done"] == 0 and not s["finished"]


def test_progress_reads_records_in_srf_order(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    with open(d / "sweep.jsonl", "w") as f:
        for srf in (1.5, 1.0):
            f.write(json.dumps({"srf": srf, "total": 1.0, "disp": 1.0,
                                "sec": 60, "admissible": 0.9}) + "\n")
    s = P.summarise(str(d))
    assert [r["srf"] for r in s["records"]] == [1.0, 1.5]
    assert P.main([str(d)]) == 0
