"""House figure style (src/plot_style.py) and the adopted-criterion results
figures (scripts/make_results_figs.py)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src import plot_style as ps
from scripts import make_results_figs as R

RESULTS = os.path.join(os.path.dirname(__file__), "..", "docs", "results")
HAVE_RESULTS = os.path.exists(os.path.join(RESULTS, "final_results.json"))


def test_apply_sets_times_12pt_and_stix_everywhere():
    ps.apply()
    rc = plt.rcParams
    assert rc["font.serif"][0] == "Times New Roman"
    assert rc["mathtext.fontset"] == "stix"
    for k in ("font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize",
              "ytick.labelsize", "legend.fontsize", "figure.titlesize"):
        assert float(rc[k]) == 12.0, k
    assert rc["savefig.dpi"] >= 300


def test_save_writes_png_and_pdf_at_the_text_width(tmp_path):
    ps.apply()
    fig, ax = plt.subplots(figsize=(4.0, 3.0), layout="constrained")
    ax.plot([0, 1], [0, 1])
    ax.set(xlabel="a long x label", ylabel="y")
    paths = ps.save(fig, str(tmp_path / "f.png"))
    assert sorted(os.path.basename(p) for p in paths) == ["f.pdf", "f.png"]
    w, _ = ps.png_size_in(str(tmp_path / "f.png"))
    assert 6.2 <= w <= 6.3


def test_criterion_colours_are_distinct_and_fixed():
    assert ps.CRITERION["GHB"] != ps.CRITERION["MC"]
    assert set(ps.STRATA) == {"Mk", "Mk_d", "Tm"}
    # no boundary-segment colour may equal a stratum colour (Fig 3 legend)
    assert not set(ps.SEGMENT.values()) & set(ps.STRATA.values())


def test_binding_names_the_condition_not_yet_met_at_the_stable_end():
    cfg = {"bracket": [1.1, 1.2], "loss_reference": 1.0}
    recs = [{"srf": 1.1, "total": 6.0, "admissible": 0.6},
            {"srf": 1.2, "total": 7.0, "admissible": 0.4}]
    assert R.binding(cfg, recs) == "floor"
    recs[0].update(total=4.0, admissible=0.4)
    assert R.binding(cfg, recs) == "loss"
    recs[0].update(total=4.0, admissible=0.6)
    assert R.binding(cfg, recs) == "both"


def test_grid_versus_bisection_points():
    assert R.on_grid(1.0) and R.on_grid(1.25) and R.on_grid(3.5)
    assert not R.on_grid(1.4921875) and not R.on_grid(1.375 + 0.0625)


@pytest.mark.skipif(not HAVE_RESULTS, reason="docs/results not present")
def test_figS2_readout_reproduces_the_D513_tables():
    recs = R.load_sweep(RESULTS, "probe_v2_disp5")
    s = np.array([r["srf"] for r in recs])
    L = np.array([r["total"] for r in recs])
    A = np.array([r["admissible"] for r in recs])
    Lref = L[np.argmin(np.abs(s - 1.0))]
    got = [R.first_met(s, L, k * Lref, True) for k in (5, 10, 15, 20)]
    assert got == pytest.approx([1.5, 2.0, 2.75, 3.453], abs=1e-3)
    assert np.isnan(R.first_met(s, L, 25 * Lref, True))
    got = [R.first_met(s, A, f, False) for f in (0.6, 0.5, 0.4, 0.3, 0.2, 0.15)]
    assert got == pytest.approx([1.25, 1.5, 1.75, 2.25, 3.0, 3.469], abs=1e-3)


@pytest.mark.skipif(not HAVE_RESULTS, reason="docs/results not present")
def test_results_figures_write_at_text_width(tmp_path):
    assert R.main(["--results", RESULTS, "--out", str(tmp_path),
                   "--require"]) == 0
    for name in R.FIGS:
        png = tmp_path / f"{name}.png"
        assert png.stat().st_size > 20000, name
        assert (tmp_path / f"{name}.pdf").exists(), name
        w, _ = ps.png_size_in(str(png))
        assert 6.2 <= w <= 6.3, (name, w)


@pytest.mark.skipif(not HAVE_RESULTS, reason="docs/results not present")
def test_results_figures_use_adopted_not_as_run_values():
    fin = R.load_final(RESULTS)
    c = fin["configurations"]["ssr_arm_Mk_d_GSI_low"]
    # as run 3.215 (three-condition); adopted 1.043 (D-5.18 / D-5.19)
    assert c["as_run"]["fos"] > 3.0 and c["value"] < 1.1
    low = [t for t in fin["summary"]["tornado"] if t["factor"] == "Mk_d GSI"]
    assert low and low[0]["low"] == pytest.approx(c["value"])
