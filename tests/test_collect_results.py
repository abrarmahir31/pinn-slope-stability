"""scripts/collect_results.py and scripts/make_arms.py, on synthetic runs."""
import json
import os

import pytest

from scripts import collect_results as C
from scripts import make_arms


def _run(root, name, **kw):
    r = {"criterion": "GHB", "fos": 1.2, "bracket": [1.19, 1.21],
         "w_yield": 1.0, "yield_norm": "ref", "admissible_metric": "area",
         "admissible_drop": 0.1, "plateau_factor": 20.0, "disp_factor": 10.0,
         "epochs": 1500, "lr": 3e-4, "sig3_range": [0.0, 1.5e5],
         "mc_design": None, "cold_start": False, "kc": "default",
         "n_pde": 10000, "baseline": "b.pt", "sampling_seed": 7,
         "status": "fos", "smoke": False, "arm": None}
    r.update(kw)
    d = os.path.join(root, name)
    os.makedirs(d)
    with open(os.path.join(d, "result.json"), "w") as f:
        json.dump(r, f)


def test_smoke_runs_are_excluded(tmp_path):
    _run(tmp_path, "ssr_a")
    _run(tmp_path, "ssr_smoke", smoke=True)
    assert len(C.load_runs(str(tmp_path / "ssr*"))) == 1


def test_plateau_groups_by_w_yield_and_reports_spread(tmp_path):
    for w, f in ((0.3, 1.10), (1.0, 1.20), (3.0, 1.30)):
        _run(tmp_path, f"ssr_w{w}", w_yield=w, fos=f)
    p = C.plateau(C.load_runs(str(tmp_path / "ssr*")))
    assert len(p) == 1
    assert [x[0] for x in p[0]["points"]] == [0.3, 1.0, 3.0]
    assert p[0]["rel_spread"] == pytest.approx(0.2 / 1.2)


def test_plateau_never_mixes_sig3_ranges(tmp_path):
    _run(tmp_path, "ssr_a", w_yield=0.3)
    _run(tmp_path, "ssr_b", w_yield=3.0, sig3_range=[0.0, 8.5e5])
    assert C.plateau(C.load_runs(str(tmp_path / "ssr*"))) == []


def test_cold_start_pairs(tmp_path):
    _run(tmp_path, "ssr_warm", fos=1.20)
    _run(tmp_path, "ssr_cold", fos=1.25, cold_start=True)
    c = C.cold_start(C.load_runs(str(tmp_path / "ssr*")))
    assert c[0]["diff"] == pytest.approx(0.05)


def test_tornado_needs_a_matching_baseline(tmp_path):
    _run(tmp_path, "ssr_base", fos=1.20)
    for lvl, f in (("low", 1.10), ("high", 1.35)):
        _run(tmp_path, f"ssr_ks_{lvl}", fos=f,
             arm={"factor": "Tm_K_s", "level": lvl})
    t = C.tornado(C.load_runs(str(tmp_path / "ssr*")))
    assert t[0]["bars"][0]["span"] == pytest.approx(0.25)


def test_tornado_is_empty_without_a_baseline(tmp_path):
    _run(tmp_path, "ssr_ks_low", arm={"factor": "Tm_K_s", "level": "low"})
    assert C.tornado(C.load_runs(str(tmp_path / "ssr*"))) == []


def test_coupling_effect_is_signed(tmp_path):
    _run(tmp_path, "ssr_base", fos=1.20)
    _run(tmp_path, "ssr_off", fos=1.08, kc="off",
         arm={"factor": "coupling", "level": "low"})
    c = C.coupling(C.load_runs(str(tmp_path / "ssr*")))
    assert c[0]["one_way_overestimate_pct"] < 0     # one-way LOWER here


def test_replicates_and_n_pde(tmp_path):
    _run(tmp_path, "ssr_s1", sampling_seed=1, fos=1.20)
    _run(tmp_path, "ssr_s2", sampling_seed=2, fos=1.24)
    _run(tmp_path, "ssr_n5k", sampling_seed=1, n_pde=5000, fos=1.18)
    rows = C.load_runs(str(tmp_path / "ssr*"))
    assert C.replicates(rows)[0]["n"] == 2
    assert C.n_pde(rows)[0]["n_pde"] == [5000, 10000]


def test_main_refuses_when_there_is_nothing(tmp_path):
    assert C.main(["--runs", str(tmp_path / "ssr*"),
                   "--out", str(tmp_path / "o")]) == 1


def test_make_arms_requires_strata_and_skips_rainfall_by_default(tmp_path):
    with pytest.raises(SystemExit):
        make_arms.main(["--out", str(tmp_path)])
    make_arms.main(["--ks-stratum", "Tm", "--gsi-stratum", "Mk",
                    "--out", str(tmp_path)])
    names = sorted(os.listdir(tmp_path))
    assert not any("rainfall" in n or "baseline" in n for n in names)
    assert "coupling_low.json" in names


# --- D-5.6: validation-only baselines -------------------------------------------

def _manifest(root, tag, sha):
    d = os.path.join(root, tag)
    os.makedirs(d)
    with open(os.path.join(d, "MANIFEST.json"), "w") as f:
        json.dump({"files": {"run/ckpt_final.pt": {"sha256": sha}}}, f)


def test_headline_status_marks_validation_baselines(tmp_path):
    _manifest(str(tmp_path / "b"), "baseline-v1", "abc")
    shas = C.validation_only_shas(str(tmp_path / "b"))
    assert C.headline_status({"baseline_sha256": "abc"}, shas) == "validation_only:baseline-v1"
    assert C.headline_status({"baseline_sha256": "def"}, shas) == "eligible"
    assert C.headline_status({}, shas) == "unknown"


def test_load_runs_attaches_headline_status(tmp_path):
    _manifest(str(tmp_path / "b"), "baseline-v1", "abc")
    _run(tmp_path, "ssr_v1", baseline_sha256="abc")
    _run(tmp_path, "ssr_prod", baseline_sha256="fff")
    rows = {os.path.basename(r["dir"]): r["headline"]
            for r in C.load_runs(str(tmp_path / "ssr*"), str(tmp_path / "b"))}
    assert rows == {"ssr_v1": "validation_only:baseline-v1", "ssr_prod": "eligible"}


def test_the_repo_manifest_identifies_baseline_v1():
    shas = C.validation_only_shas("baselines")
    assert "0a193d83f3d87da53ac606742a952b041bcf27a0d94408f2314326180e67a53f" in shas
