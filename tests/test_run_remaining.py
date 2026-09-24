"""scripts/run_remaining.py and scripts/recompute_criterion.py -- no GPU."""
import json
import os
import sys

import pytest

from scripts import recompute_criterion as RC
from scripts import run_remaining as R


# --- the plan ---------------------------------------------------------------

def _flag(cmd, name):
    return cmd[cmd.index(name) + 1]


def test_plan_order_and_names():
    names = [j["name"] for j in R.plan()]
    assert names[:2] == ["ssr_ghb_n5000_wscaled", "ssr_ghb_n20000_wscaled"]
    assert names[2:4] == ["ssr_arm_coupling_low", "ssr_arm_coupling_high"]
    assert names[-4:] == [f"ssr_ghb_draw{s}" for s in (11, 12, 13, 14)]
    assert len(names) == len(set(names)) == 12


def test_npde_runs_use_the_d517_scaled_weight_and_match_the_hand_command():
    # The 24 Sep hand command: --n-pde 5000 --w-yield 2 / --n-pde 20000
    # --w-yield 0.5. Resume only works if every fingerprinted flag agrees.
    jobs = {j["name"]: R.command(j) for j in R.plan()}
    c5, c20 = jobs["ssr_ghb_n5000_wscaled"], jobs["ssr_ghb_n20000_wscaled"]
    assert (_flag(c5, "--n-pde"), float(_flag(c5, "--w-yield"))) == ("5000", 2.0)
    assert (_flag(c20, "--n-pde"), float(_flag(c20, "--w-yield"))) == ("20000", 0.5)
    for c in (c5, c20):
        assert float(_flag(c, "--admissible-floor")) == 0.5
        assert float(_flag(c, "--plateau-factor")) == 5
        assert float(_flag(c, "--disp-factor")) == 5
        assert _flag(c, "--yield-norm") == "raw"
        assert _flag(c, "--admissible-mode") == "floor"
        assert float(_flag(c, "--srf-step")) == 0.25
        assert float(_flag(c, "--srf-max")) == 4.0
        assert "--yield-reduction" not in c          # default legacy, as run


def test_every_other_run_is_at_the_reported_weight_and_10k():
    for j in R.plan():
        c = R.command(j)
        assert _flag(c, "--baseline") == R.BASELINE
        if j["group"] != "npde":
            assert _flag(c, "--w-yield") == R.W_REPORT
            assert "--n-pde" not in c                 # checkpoint's 10,000


def test_arm_files_named_in_the_plan_are_the_ones_make_arms_writes(tmp_path):
    from scripts import make_arms
    args = list(R.MAKE_ARMS)
    args[args.index("--out") + 1] = str(tmp_path)
    make_arms.main(args)
    written = {os.path.splitext(n)[0] for n in os.listdir(tmp_path)}
    wanted = {os.path.splitext(os.path.basename(_flag(R.command(j), "--overrides")))[0]
              for j in R.plan() if j["group"] in ("coupling", "ks", "gsi")}
    assert wanted == written


def test_gpu_parse_only_counts_python():
    assert R.parse_compute_apps("") == []
    assert R.parse_compute_apps("1234, C:\\x\\python.exe\n") == ["1234, C:\\x\\python.exe"]
    assert R.parse_compute_apps("99, chrome.exe\n") == []


def test_dry_run_runs_nothing(capsys, monkeypatch):
    called = []
    monkeypatch.setattr(R, "run_streamed", lambda *a, **k: called.append(a))
    assert R.main(["--dry-run"]) == 0
    assert called == []
    assert "ssr_arm_Mk_d_GSI_high" in capsys.readouterr().out


def test_run_streamed_tees_and_returns_exit_code(tmp_path):
    log = R.Log(str(tmp_path / "main.log"))
    rc = R.run_streamed([sys.executable, "-c", "print('hello'); raise SystemExit(3)"],
                        log, str(tmp_path / "c" / "console.log"))
    assert rc == 3
    assert "hello" in open(tmp_path / "main.log").read()
    assert "hello" in open(tmp_path / "c" / "console.log").read()


def test_copy_results_skips_smoke_and_aborted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name, smoke in (("ssr_a", False), ("ssr_smoke", True),
                        ("ssr_x_ABORTED_v1", False)):
        os.makedirs(f"runs/{name}")
        json.dump({"smoke": smoke}, open(f"runs/{name}/result.json", "w"))
        open(f"runs/{name}/sweep.jsonl", "w").write("{}\n")
    R.copy_results(R.Log(None), dest="out")
    assert sorted(os.listdir("out")) == ["ssr_a_result.json", "ssr_a_sweep.jsonl"]


# --- recompute_criterion ------------------------------------------------------

S = {"plateau_factor": 5, "disp_factor": 5, "admissible_mode": "floor",
     "admissible_floor": 0.5, "admissible_drop": None, "srf_start": 1.0,
     "bisect_tol": 0.01}


def _rec(srf, total, disp, adm):
    return {"srf": srf, "total": total, "disp": disp, "admissible": adm,
            "finite": True}


def test_loss_floor_can_fire_before_three_when_displacement_lags():
    recs = [_rec(1.0, 0.2, 1e-6, 0.74),
            _rec(1.25, 0.6, 2e-6, 0.60),
            _rec(1.5, 1.1, 4e-6, 0.45),      # loss + floor, disp only 4x
            _rec(1.75, 1.5, 9e-6, 0.40)]     # all three
    three = RC.bracket(recs, S, "three")
    lf = RC.bracket(recs, S, "loss_floor")
    assert three["bracket"] == [1.5, 1.75]
    assert lf["bracket"] == [1.25, 1.5] and lf["width"] == 0.25
    assert lf["fired_at_failure"] == {"loss": True, "disp": False,
                                      "admissible": True}


def test_non_monotone_points_are_flagged_not_used():
    recs = [_rec(1.0, 0.2, 1e-6, 0.74), _rec(1.25, 1.2, 1e-4, 0.45),
            _rec(1.5, 0.9, 1e-4, 0.55)]
    b = RC.bracket(recs, S, "loss_floor")
    assert b["bracket"] == [1.0, 1.25] and b["non_monotone"] == [1.5]


def test_no_failure_and_missing_reference():
    recs = [_rec(1.0, 0.2, 1e-6, 0.74), _rec(1.25, 0.3, 1e-6, 0.70)]
    assert RC.bracket(recs, S, "three")["status"] == "no_failure"
    assert RC.bracket(recs[1:], S, "three")["status"] == "no_reference_record"


def test_analyse_reproduces_the_committed_calibrated_probe(tmp_path):
    d = tmp_path / "ssr_probe"
    d.mkdir()
    root = os.path.join(os.path.dirname(__file__), "..", "docs", "results")
    for src, dst in (("probe_calibrated_result.json", "result.json"),
                     ("probe_calibrated_sweep.jsonl", "sweep.jsonl")):
        p = os.path.join(root, src)
        if not os.path.exists(p):
            pytest.skip(f"{src} not in docs/results")
        open(d / dst, "wb").write(open(p, "rb").read())
    r = RC.analyse(str(d))
    assert r["three_reproduces_result_json"]
    assert abs(r["three"]["fos"] - 1.488) < 1e-3
    assert abs(r["loss_floor"]["fos"] - 1.488) < 1e-3


def test_wait_needs_consecutive_idle_polls(monkeypatch):
    # busy, idle, busy (the gap between the chained 5k and 20k runs), then idle x3
    seq = iter([True, False, True, False, False, False])
    calls = []
    monkeypatch.setattr(R, "gpu_busy", lambda: calls.append(1) or next(seq))
    monkeypatch.setattr(R, "POLL_S", 0)
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    assert R.wait_for_gpu(R.Log(None), R.plan(), max_hours=1) is True
    assert len(calls) == 6


def test_wait_gives_up_after_max_hours(monkeypatch):
    monkeypatch.setattr(R, "gpu_busy", lambda: True)
    monkeypatch.setattr(R, "POLL_S", 0)
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    assert R.wait_for_gpu(R.Log(None), R.plan(), max_hours=0) is False


def test_without_nvidia_smi_a_fresh_sweep_log_counts_as_busy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("runs/ssr_ghb_n20000_wscaled")
    open("runs/ssr_ghb_n20000_wscaled/sweep.jsonl", "w").write("{}\n")
    assert R.recently_written(["runs/ssr_ghb_n20000_wscaled/sweep.jsonl"], 90)
    assert not R.recently_written(["runs/nothing/sweep.jsonl"], 90)
