"""Figure scripts: they write when there is data and refuse when there is not."""
import json
import os

import pytest
import torch

from scripts import make_fig9 as F9
from scripts import make_ssr_figs as F
from scripts import collect_results as C
from scripts import train
from src.weighting import TERMS


def _run(root, name, recs=True, **kw):
    r = {"criterion": "GHB", "fos": 1.2, "bracket": [1.15, 1.2], "w_yield": 1.0,
         "yield_norm": "ref", "admissible_metric": "area",
         "admissible_drop": 0.1, "plateau_factor": 20.0, "disp_factor": 10.0,
         "epochs": 1500, "lr": 3e-4, "sig3_range": [0.0, 1.5e5],
         "mc_design": None, "cold_start": False, "kc": "default",
         "n_pde": 10000, "baseline": "b.pt", "sampling_seed": 7,
         "status": "fos", "smoke": False, "arm": None}
    r.update(kw)
    d = os.path.join(root, name)
    os.makedirs(d)
    json.dump(r, open(os.path.join(d, "result.json"), "w"))
    if recs:
        with open(os.path.join(d, "sweep.jsonl"), "w") as f:
            for s, L, adm in ((1.0, 1.0, 0.95), (1.1, 2.0, 0.90), (1.2, 50.0, 0.7)):
                f.write(json.dumps({"srf": s, "total": L, "admissible": adm,
                                    "finite": True}) + "\n")
    return d


def test_every_figure_skips_rather_than_drawing_empty(tmp_path):
    for name, fn in F.FIGS.items():
        with pytest.raises(F.NoData):
            fn([], str(tmp_path / f"{name}.png"))
        assert not (tmp_path / f"{name}.png").exists()


def test_require_turns_a_skip_into_failure(tmp_path):
    assert F.main(["--runs", str(tmp_path / "ssr*"), "--out", str(tmp_path),
                   "--require"]) == 1


def test_figures_write_with_data(tmp_path):
    for w, f in ((0.3, 1.1), (1.0, 1.2), (3.0, 1.25)):
        _run(tmp_path, f"ssr_w{w}", w_yield=w, fos=f)
    for w, f in ((0.3, 1.0), (1.0, 1.05), (3.0, 1.08)):
        _run(tmp_path, f"ssr_mc_w{w}", criterion="MC", w_yield=w, fos=f,
             mc_strength="rockmass", mc_sig3max=4e5, sig3_range=None)
    _run(tmp_path, "ssr_off", fos=1.3, kc="off",
         arm={"factor": "coupling", "level": "low"})
    for lvl, f in (("low", 1.1), ("high", 1.3)):
        _run(tmp_path, f"ssr_ks_{lvl}", fos=f,
             arm={"factor": "Tm_K_s", "level": lvl})
    rows = C.load_runs(str(tmp_path / "ssr*"))
    for name, fn in F.FIGS.items():
        p = tmp_path / f"{name}.png"
        assert fn(rows, str(p)) >= 1
        assert p.stat().st_size > 5000


def test_fig8_skips_runs_without_finite_records(tmp_path):
    _run(tmp_path, "ssr_norecs", recs=False)
    with pytest.raises(F.NoData):
        F.fig8(C.load_runs(str(tmp_path / "ssr*")), str(tmp_path / "f.png"))


@pytest.fixture(scope="module")
def tiny_state(tmp_path_factory):
    t = train.parse(["--layers", "2", "--width", "16", "--ansatz", "exp"])
    net, _, _ = train.setup(t)
    root = tmp_path_factory.mktemp("f9")
    d = _run(str(root), "ssr_tiny", bracket=[1.1, 1.15])
    os.makedirs(os.path.join(d, "states"))
    for srf in (1.1, 1.15):
        torch.save({"net": net.state_dict(), "cfg": vars(t),
                    "sweep": {"criterion": "GHB"}},
                   os.path.join(d, "states", f"srf_{srf:.6f}.pt"))
    return d


def test_state_path_picks_the_requested_end_of_the_bracket(tiny_state):
    assert F9.state_path(tiny_state, "failed")[1] == 1.15
    assert F9.state_path(tiny_state, "stable")[1] == 1.1


def test_state_path_refuses_a_run_without_failure(tmp_path):
    d = _run(str(tmp_path), "ssr_nofail", bracket=[3.0, None], fos=None)
    with pytest.raises(ValueError):
        F9.state_path(d, "failed")


def test_fig9_end_to_end_on_a_tiny_state(tiny_state, tmp_path):
    out = tmp_path / "fig9.png"
    assert F9.main(["--run", tiny_state, "--state", "failed", "--nx", "24",
                    "--nz", "16", "--out", str(out)]) == 0
    s = json.load(open(tmp_path / "fig9.json"))
    assert s[0]["srf"] == 1.15 and len(s[0]["ridge_x"]) > 0
    assert out.stat().st_size > 5000


def test_build_net_refuses_a_state_without_its_ansatz():
    with pytest.raises(KeyError):
        F9.build_net({"layers": 2, "width": 16, "eps_psi": 0.3})


def test_fig9_reports_localisation_relative_to_the_sweep_start(tiny_state, tmp_path):
    # Tiny state: start and failed nets are identical, so the ratio is 1 --
    # the value that must read as "no band formed".
    F9.main(["--run", tiny_state, "--state", "failed", "--nx", "24",
             "--nz", "16", "--out", str(tmp_path / "f.png")])
    s = json.load(open(tmp_path / "f.json"))
    assert s[0]["li_ratio_vs_start"] == pytest.approx(1.0)


def test_figures_from_non_headline_runs_are_stamped(tmp_path):
    p = tmp_path / "f.png"
    for name, rows, expected in (("val", [{"headline": "validation_only:baseline-v1"}], True),
                                 ("ok", [{"headline": "eligible"}], False)):
        import matplotlib.pyplot as plt
        fig = plt.figure(); plt.plot([0, 1]); fig.savefig(p); plt.close(fig)
        assert F.stamp_if_not_headline(str(p), rows) is expected, name


def test_fig10_bars_need_the_reported_w_yield(tmp_path):
    for w, f in ((0.3, 1.1), (3.0, 1.25)):
        _run(tmp_path, f"ssr_w{w}", w_yield=w, fos=f)
    rows = C.load_runs(str(tmp_path / "ssr*"))
    with pytest.raises(F.NoData):
        F.fig10(rows, str(tmp_path / "f.png"), w_report=1.0)
    assert F.fig10(rows, str(tmp_path / "f.png"), w_report=0.3) == 1
