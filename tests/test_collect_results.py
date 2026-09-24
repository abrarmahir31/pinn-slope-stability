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
    # D-5.17: a legacy 5k run matches the 10k w_yield 1 run at w_yield 2.
    _run(tmp_path, "ssr_s1", sampling_seed=1, fos=1.20)
    _run(tmp_path, "ssr_s2", sampling_seed=2, fos=1.24)
    _run(tmp_path, "ssr_n5k", sampling_seed=1, n_pde=5000, w_yield=2.0,
         fos=1.18)
    rows = C.load_runs(str(tmp_path / "ssr*"))
    assert C.replicates(rows)[0]["n"] == 2
    g = C.n_pde(rows)[0]
    assert g["n_pde"] == [5000, 10000] and g["w_eff"] == 1.0


def test_n_pde_never_groups_unscaled_legacy_runs(tmp_path):
    # Same w_yield at 5k/10k/20k is three DIFFERENT penalty strengths (D-5.17).
    _run(tmp_path, "ssr_n10k", sampling_seed=1, fos=1.49)
    _run(tmp_path, "ssr_n5k", sampling_seed=1, n_pde=5000, fos=1.49)
    _run(tmp_path, "ssr_n20k", sampling_seed=1, n_pde=20000, fos=1.62)
    assert C.n_pde(C.load_runs(str(tmp_path / "ssr*"))) == []


def test_n_pde_mean_reduction_needs_no_scaling(tmp_path):
    _run(tmp_path, "ssr_n10k", sampling_seed=1, yield_reduction="mean")
    _run(tmp_path, "ssr_n5k", sampling_seed=1, n_pde=5000,
         yield_reduction="mean")
    assert C.n_pde(C.load_runs(str(tmp_path / "ssr*")))[0]["w_eff"] == 1.0


def test_missing_yield_reduction_is_legacy_and_never_mixes_with_mean(tmp_path):
    _run(tmp_path, "ssr_a", w_yield=0.3)
    _run(tmp_path, "ssr_b", w_yield=3.0, yield_reduction="mean")
    rows = C.load_runs(str(tmp_path / "ssr*"))
    assert {r["yield_reduction"] for r in rows} == {"legacy", "mean"}
    assert C.plateau(rows) == []


def test_make_arms_ks_factors_replace_the_symmetric_pair(tmp_path):
    make_arms.main(["--ks-stratum", "Mk_d", "--gsi-stratum", "Mk_d",
                    "--gsi-levels", "35,45,55", "--ks-factors", "0.1,10",
                    "--out", str(tmp_path)])
    lo = json.load(open(tmp_path / "Mk_d_K_s_low.json"))
    hi = json.load(open(tmp_path / "Mk_d_K_s_high.json"))
    assert lo["overrides"] == {"ks_multiplier": {"Mk_d": 0.1}}
    assert hi["overrides"] == {"ks_multiplier": {"Mk_d": 10.0}}
    gsi = json.load(open(tmp_path / "Mk_d_GSI_low.json"))
    assert gsi["overrides"]["materials"]["Mk_d"]["gsi"] == 35.0
    assert len(os.listdir(tmp_path)) == 6       # 2 K_s, 2 GSI, 2 coupling
    with pytest.raises(SystemExit):
        make_arms.main(["--ks-stratum", "Mk_d", "--gsi-stratum", "Mk_d",
                        "--ks-factors", "1.2,0.8", "--out", str(tmp_path)])


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
    assert C.headline_status({"baseline_sha256": "def"}, shas) == "unfrozen_baseline"
    assert C.headline_status({}, shas) == "unknown"


def test_load_runs_attaches_headline_status(tmp_path):
    _manifest(str(tmp_path / "b"), "baseline-v1", "abc")
    _run(tmp_path, "ssr_v1", baseline_sha256="abc")
    _manifest(str(tmp_path / "b"), "baseline-v2", "fff")
    _run(tmp_path, "ssr_prod", baseline_sha256="fff")
    _run(tmp_path, "ssr_edited", baseline_sha256="999")
    rows = {os.path.basename(r["dir"]): r["headline"]
            for r in C.load_runs(str(tmp_path / "ssr*"), str(tmp_path / "b"))}
    assert rows == {"ssr_v1": "validation_only:baseline-v1",
                    "ssr_prod": "eligible",
                    "ssr_edited": "unfrozen_baseline"}


def test_the_repo_manifest_identifies_baseline_v1():
    shas = C.validation_only_shas("baselines")
    assert "0a193d83f3d87da53ac606742a952b041bcf27a0d94408f2314326180e67a53f" in shas


def test_an_overwritten_checkpoint_is_not_headline_eligible(tmp_path):
    """Day 42 incident: a checkpoint edited in place matches no manifest."""
    _manifest(str(tmp_path / "b"), "baseline-v1", "abc")
    shas = C.frozen_shas(str(tmp_path / "b"))
    assert C.headline_status({"baseline_sha256": "edited"}, shas) == "unfrozen_baseline"
