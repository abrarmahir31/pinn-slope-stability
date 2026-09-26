r"""Thesis Figs 8, 10, 11, 12 and S2 under the ADOPTED criterion. NO GPU.

    python scripts/make_results_figs.py                      # all, docs/figs
    python scripts/make_results_figs.py --only fig10,fig12 --out docs/figs

Reads only what git holds: `docs/results/final_results.json` (written by
`final_results.py`) for every reported number, and
`docs/results/<run>_sweep.jsonl` for the per-SRF curves. Nothing is typed in.

Why a second script next to `make_ssr_figs.py`: that one draws from `runs/`
and plots each run's AS-RUN FOS (three-condition detector). The thesis
reports the D-5.18 / D-5.19 values, which differ for several runs (e.g. Mk_d
GSI 35: 3.215 as run, 1.043 adopted). Plotting the as-run numbers would put
figures and FINAL_RESULTS.md in disagreement.

  Fig 8   L / L_ref and admissible fraction (Mk_d, Mk) vs SRF, GHB and MC at
          w_yield 1; thresholds drawn (5 x L_ref, floor 0.50); 0.25-grid
          points filled, bisection points open and not joined (revision 3).
  Fig 10  (a) GHB vs MC at w_yield 1, hatched, "calibrated (D-5.13)", replicate
          SD as error bars, 1.5 target line, no LEM bar (O-5, O-15);
          (b) FOS vs w_yield, ticks 0.3/1/3, marker = binding condition
          (absorbs Fig S1, O-5).
  Fig 11  coupling arms, point +/- replicate SD against the baseline band.
  Fig 12  tornado, seed-7 arms against the seed-7 baseline, baseline band.
  Fig S2  criterion dependence from probe_v2_disp5: SRF at which each
          condition ALONE is first met, vs its threshold (D-5.13 tables);
          the calibrated setting marked (revision 1).

Colour rule (src/plot_style.py): criterion = colour (GHB cobalt, MC
vermilion); w_yield = marker; sensitivity level low/high = purple/amber.
All text 12 pt Times; 6.25 in wide; PNG (300 dpi) + PDF.

CAPTIONS MUST SAY: every FOS is calibrated (D-5.13); the error bars / bands
are the GHB collocation-draw replicate SD (n = 5), applied to MC as well.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullLocator
import numpy as np

from src import plot_style as ps

PLATEAU = 5.0          # D-5.13 / D-5.18
FLOOR = 0.50
TARGET = 1.5           # D-5.13 calibration target
GRID0, GRID_STEP = 1.0, 0.25   # D-5.16 SRF grid; anything else is bisection


class NoData(Exception):
    pass


# --------------------------------------------------------------------------- io
def load_final(results: str) -> dict:
    p = os.path.join(results, "final_results.json")
    if not os.path.exists(p):
        raise NoData(f"{p} missing: run scripts/final_results.py first")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_sweep(results: str, name: str) -> list:
    p = os.path.join(results, f"{name}_sweep.jsonl")
    if not os.path.exists(p):
        raise NoData(f"{p} missing")
    by = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("finite", True) and "admissible" in r:
                    by[round(float(r["srf"]), 6)] = r
    return [by[k] for k in sorted(by)]


def on_grid(srf: float) -> bool:
    k = (srf - GRID0) / GRID_STEP
    return abs(k - round(k)) < 1e-6


def binding(cfg: dict, recs: list) -> str:
    """Which condition decided the bracket: the one NOT yet met at the stable
    end. 'loss' / 'floor' / 'both' (neither met at the stable end)."""
    lo = (cfg.get("bracket") or [None])[0]
    if lo is None:
        return "unknown"
    r = min(recs, key=lambda x: abs(x["srf"] - lo))
    if abs(r["srf"] - lo) > 1e-5:
        return "unknown"
    loss = r["total"] > PLATEAU * cfg["loss_reference"]
    floor = r["admissible"] < FLOOR
    return {(True, False): "floor", (False, True): "loss",
            (False, False): "both"}.get((loss, floor), "unknown")


def _cfg(final, name):
    c = final["configurations"].get(name)
    if c is None or c.get("value") is None:
        raise NoData(f"{name}: no adopted value in final_results.json")
    return c


def _sd(final) -> float:
    return float(final["summary"]["replicates"]["sd"])


# --------------------------------------------------------------------------- Fig 8
def fig8(final, results, out):
    runs = [("GHB", "ssr_ghb_w1"), ("MC", "ssr_mc_w1")]
    fig = plt.figure(figsize=(ps.WIDTH, 6.3), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
    a0 = fig.add_subplot(gs[0, :])
    a1 = fig.add_subplot(gs[1, 0], sharex=a0)
    a2 = fig.add_subplot(gs[1, 1], sharex=a0, sharey=a1)
    for crit, name in runs:
        c = _cfg(final, name)
        recs = load_sweep(results, c["source"])
        col = ps.CRITERION[crit]
        s = np.array([r["srf"] for r in recs])
        L = np.array([r["total"] for r in recs]) / c["loss_reference"]
        adm = {t: np.array([(r.get("admissible_by_tag") or {}).get(t, np.nan)
                            for r in recs]) for t in ("Mk_d", "Mk")}
        g = np.array([on_grid(v) for v in s])
        for ax, y in ((a0, L), (a1, adm["Mk_d"]), (a2, adm["Mk"])):
            ax.plot(s[g], y[g], "-", color=col, lw=1.5, marker="o", ms=5,
                    zorder=3)
            ax.plot(s[~g], y[~g], ls="none", marker="o", ms=5, mfc="white",
                    mec=col, mew=1.2, zorder=4)
            ax.axvline(c["value"], color=col, lw=1.0, ls=":", zorder=2)
    a0.set_yscale("log")
    a0.yaxis.set_major_locator(FixedLocator([1, 2, 3, 4, 5, 6, 8]))
    a0.yaxis.set_minor_locator(NullLocator())
    a0.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    a0.set_xlabel("SRF")
    a0.axhline(PLATEAU, color=ps.INK, lw=1.0, ls="--")
    a0.text(0.99, PLATEAU, r"$5\,\mathcal{L}_\mathrm{ref}$", ha="right",
            va="bottom", transform=a0.get_yaxis_transform())
    a0.set(ylabel=r"$\mathcal{L}\,/\,\mathcal{L}_\mathrm{ref}$")
    for ax, t in ((a1, "Mk_d"), (a2, "Mk")):
        ax.axhline(FLOOR, color=ps.INK, lw=1.0, ls="--")
        ax.set_xlabel("SRF")
        ax.text(0.97, 0.05, ps.STRATA_LABEL[t], transform=ax.transAxes,
                ha="right", va="bottom", fontweight="bold",
                color=ps.STRATA[t])
    a1.text(0.03, FLOOR, "floor 0.50", transform=a1.get_yaxis_transform(),
            ha="left", va="bottom")
    a1.set_ylabel("Admissible fraction")
    a2.tick_params(labelleft=False)
    a1.set_ylim(0.0, 1.14)
    for ax, k in ((a0, "a"), (a1, "b"), (a2, "c")):
        ps.panel_label(ax, k)
    h = [Line2D([], [], color=ps.CRITERION[c_], lw=1.5, label=f"{c_}")
         for c_, _ in runs]
    h += [Line2D([], [], ls="none", marker="o", color=ps.GREY, label="0.25 grid"),
          Line2D([], [], ls="none", marker="o", mfc="white", mec=ps.GREY,
                 label="bisection"),
          Line2D([], [], color=ps.GREY, ls=":", label="FOS")]
    fig.legend(handles=h, loc="outside upper center", ncol=5, frameon=False,
               handlelength=1.6, columnspacing=0.9)
    ps.save(fig, out)
    return len(runs)


# --------------------------------------------------------------------------- Fig 10
BIND_MARKER = {"loss": "s", "floor": "D", "both": "o", "unknown": "X"}
BIND_LABEL = {"loss": r"loss binds ($5\,\mathcal{L}_\mathrm{ref}$)",
              "floor": "floor binds (0.50)", "both": "both at once",
              "unknown": "not resolved"}


def fig10(final, results, out):
    sd = _sd(final)
    fig, (a, b) = plt.subplots(1, 2, figsize=(ps.WIDTH, 3.6),
                               layout="constrained",
                               gridspec_kw={"width_ratios": [0.85, 1.15]})
    # (a) bars at the reported weight.
    names = [("GHB", "ssr_ghb_w1"), ("MC", "ssr_mc_w1")]
    vals = [_cfg(final, n)["value"] for _, n in names]
    for i, ((crit, _), v) in enumerate(zip(names, vals)):
        a.bar(i, v, width=0.62, color=ps.CRITERION[crit], alpha=0.30,
              edgecolor=ps.CRITERION[crit], hatch="///", lw=1.3, zorder=2)
        a.errorbar(i, v, yerr=sd, fmt="none", ecolor=ps.INK, elinewidth=1.2,
                   capsize=5, zorder=3)
        a.text(i, v + sd + 0.015, f"{v:.3f}", ha="center", va="bottom")
    a.axhline(TARGET, color=ps.INK, lw=1.0, ls="--", zorder=1)
    a.set_xticks([0, 1], [f"{c}-PINN" for c, _ in names])
    a.set_xlim(-0.55, 1.55)
    a.set_ylim(0.0, 1.85)
    a.set_ylabel("FOS (calibrated)")
    a.legend(handles=[Patch(fc="white", ec=ps.GREY, hatch="///",
                            label="calibrated (D-5.13)"),
                      Line2D([], [], color=ps.INK, ls="--", label="target 1.5"),
                      Line2D([], [], color=ps.INK, marker="_", ms=10, ls="-",
                             lw=1.2, label=r"$\pm$1 replicate SD")],
             loc="lower center", framealpha=0.95, handlelength=1.6)
    a.grid(axis="x", visible=False)
    ps.panel_label(a, "a")

    # (b) FOS vs w_yield, marker = binding condition.
    used = set()
    for crit in ("GHB", "MC"):
        pts = []
        for w in (0.3, 1.0, 3.0):
            n = f"ssr_{crit.lower()}_w{w:g}"
            c = _cfg(final, n)
            bnd = binding(c, load_sweep(results, c["source"]))
            pts.append((w, c["value"], bnd))
        w_, f_, bd = zip(*pts)
        col = ps.CRITERION[crit]
        b.plot(w_, f_, "-", color=col, lw=1.4, zorder=2)
        for w, f, bn in pts:
            b.errorbar(w, f, yerr=sd, fmt=BIND_MARKER[bn], ms=6.5, color=col,
                       mfc=col, mec="white", mew=0.8, ecolor=col,
                       elinewidth=1.0, capsize=3, zorder=3)
            used.add(bn)
    b.axhline(TARGET, color=ps.INK, lw=1.0, ls="--", zorder=1)
    b.set_xscale("log")
    b.xaxis.set_major_locator(FixedLocator([0.3, 1.0, 3.0]))
    b.xaxis.set_minor_locator(NullLocator())
    b.set_xticklabels(["0.3", "1", "3"])
    b.set_xlim(0.22, 4.1)
    b.set_xlabel(r"$w_\mathrm{yield}$")
    b.set_ylabel("FOS (calibrated)")
    ps.panel_label(b, "b")
    h = [Line2D([], [], color=ps.CRITERION[c_], lw=1.4, label=c_)
         for c_ in ("GHB", "MC")]
    h += [Line2D([], [], ls="none", marker=BIND_MARKER[k], color=ps.GREY,
                 label=BIND_LABEL[k]) for k in ("loss", "floor", "both",
                                                "unknown") if k in used]
    b.legend(handles=h, loc="upper left", bbox_to_anchor=(0.0, 0.90),
             frameon=False, handlelength=1.3, borderaxespad=0.2)
    lo = min(min(vals), *[_cfg(final, f"ssr_{c.lower()}_w{w:g}")["value"]
                          for c in ("GHB", "MC") for w in (0.3, 1, 3)])
    hi = max(*[_cfg(final, f"ssr_{c.lower()}_w{w:g}")["value"]
               for c in ("GHB", "MC") for w in (0.3, 1, 3)])
    b.set_ylim(lo - 0.08, hi + 0.22)
    ps.save(fig, out)
    return 2


# --------------------------------------------------------------------------- Fig 11
def fig11(final, results, out):
    sd = _sd(final)
    order = [("ssr_arm_coupling_low", "KC off\n(one-way)"),
             ("ssr_ghb_w1", "KC marls\n(default)"),
             ("ssr_arm_coupling_high", "KC\neverywhere")]
    vals = [_cfg(final, n)["value"] for n, _ in order]
    base = vals[1]
    fig, ax = plt.subplots(figsize=(ps.WIDTH, 3.2), layout="constrained")
    ax.axhspan(base - sd, base + sd, color=ps.CRITERION["GHB"], alpha=0.10,
               lw=0, zorder=0)
    ax.axhline(base, color=ps.CRITERION["GHB"], lw=0.9, ls="-", alpha=0.6)
    for i, v in enumerate(vals):
        ax.errorbar(i, v, yerr=sd, fmt="o", ms=7, color=ps.CRITERION["GHB"],
                    mec="white", mew=0.8, capsize=5, elinewidth=1.2, zorder=3)
        d = "" if i == 1 else f"  ({100*(v-base)/base:+.1f}%)"
        ax.text(i + 0.07, v, f"{v:.3f}{d}", ha="left", va="center")
    ax.set_xticks(range(3), [l for _, l in order])
    ax.set_xlim(-0.4, 2.9)
    ax.set_ylim(base - 0.12, base + 0.12)
    ax.set_ylabel("FOS (GHB, calibrated)")
    ax.grid(axis="x", visible=False)
    ax.text(2.85, base + sd, r"baseline $\pm$1 SD", ha="right", va="bottom",
            color=ps.CRITERION["GHB"])
    ps.save(fig, out)
    return 3


# --------------------------------------------------------------------------- Fig 12
def fig12(final, results, out):
    sd = _sd(final)
    base = float(final["summary"]["baseline"])
    tor = [t for t in final["summary"].get("tornado", []) if
           t.get("low") is not None and t.get("high") is not None]
    if not tor:
        raise NoData("fig12: no tornado rows in final_results.json")
    tor = sorted(tor, key=lambda t: t["swing"])          # biggest on top
    fig, ax = plt.subplots(figsize=(ps.WIDTH, 0.95 * len(tor) + 1.5),
                           layout="constrained")
    ax.axvspan(base - sd, base + sd, color=ps.GREY, alpha=0.15, lw=0, zorder=0)
    for i, t in enumerate(tor):
        for lvl in ("low", "high"):
            v = t[lvl]
            ax.barh(i, v - base, left=base, height=0.52, color=ps.LEVEL[lvl],
                    edgecolor=ps.INK, lw=0.6, zorder=2)
            right = v >= base
            ax.text(v + (0.015 if right else -0.015), i,
                    f"{t[lvl + '_label']}: {v:.3f}", ha="left" if right
                    else "right", va="center")
    ax.axvline(base, color=ps.INK, lw=1.0, zorder=3)
    ax.set_yticks(range(len(tor)),
                  [t["factor"].replace("Mk_d", r"Mk$_\mathrm{d}$")
                   .replace("K_s", r"$K_s$") for t in tor])
    ax.set_ylim(-0.6, len(tor) - 0.4)
    lo = min(min(t["low"], t["high"]) for t in tor)
    hi = max(max(t["low"], t["high"]) for t in tor)
    ax.set_xlim(lo - 0.45, hi + 0.45)
    ax.set_xlabel("FOS (GHB, calibrated)")
    ax.grid(axis="y", visible=False)
    h = [Patch(fc=ps.LEVEL["low"], ec=ps.INK, lw=0.6, label="low level"),
         Patch(fc=ps.LEVEL["high"], ec=ps.INK, lw=0.6, label="high level"),
         Line2D([], [], color=ps.INK, lw=1.0, label=f"baseline {base:.3f}"),
         Patch(fc=ps.GREY, alpha=0.15, lw=0, label=r"$\pm$1 replicate SD")]
    fig.legend(handles=h, loc="outside upper center", ncol=4, frameon=False,
               handlelength=1.3, columnspacing=0.9)
    ps.save(fig, out)
    return len(tor)


# --------------------------------------------------------------------------- Fig S2
def first_met(s, y, thr, above: bool):
    """Lowest evaluated SRF at which y crosses `thr` (the D-5.13 readout),
    or nan if never within the sweep."""
    m = (y > thr) if above else (y < thr)
    return float(s[np.argmax(m)]) if m.any() else np.nan


def figS2(final, results, out, probe="probe_v2_disp5"):
    recs = load_sweep(results, probe)
    s = np.array([r["srf"] for r in recs])
    L = np.array([r["total"] for r in recs])
    A = np.array([r["admissible"] for r in recs])
    Lref = L[np.argmin(np.abs(s - 1.0))]
    fig, (a, b) = plt.subplots(1, 2, figsize=(ps.WIDTH, 3.3), sharey=True,
                               layout="constrained")
    col = ps.CRITERION["GHB"]
    k = np.linspace(1.5, 30, 400)
    # Dense threshold sweep of a piecewise-constant readout: drawn as a line,
    # the jumps land exactly where the threshold crosses an evaluated SRF.
    a.plot(k, [first_met(s, L, kk * Lref, True) for kk in k], color=col,
           lw=1.5)
    kt = np.array([5, 10, 15, 20])
    a.plot(kt, [first_met(s, L, kk * Lref, True) for kk in kt], "o", ms=5,
           color=col, mec="white", mew=0.8)
    a.plot([PLATEAU], [first_met(s, L, PLATEAU * Lref, True)], "*", ms=15,
           color=ps.VERMILION, mec=ps.INK, mew=0.6, zorder=5)
    a.set(xlabel=r"Loss factor $k$ ($\mathcal{L} > k\,\mathcal{L}_\mathrm{ref}$)",
          ylabel="SRF at which condition\nis first met", xlim=(0, 30.5))
    never = k[np.isnan([first_met(s, L, kk * Lref, True) for kk in k])]
    if never.size:
        a.axvspan(never.min(), 30.5, color=ps.GREY, alpha=0.12, lw=0)
        a.text(0.97, 0.06, f"not met by\nSRF {s.max():g}", transform=a.transAxes,
               ha="right", va="bottom")
    f = np.linspace(0.08, 0.72, 400)
    b.plot(f, [first_met(s, A, ff, False) for ff in f], color=col, lw=1.5)
    ft = np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.15])
    b.plot(ft, [first_met(s, A, ff, False) for ff in ft], "o", ms=5,
           color=col, mec="white", mew=0.8)
    b.plot([FLOOR], [first_met(s, A, FLOOR, False)], "*", ms=15,
           color=ps.VERMILION, mec=ps.INK, mew=0.6, zorder=5)
    b.set(xlabel="Admissible floor", xlim=(0.72, 0.08))
    for ax, l in ((a, "a"), (b, "b")):
        ps.panel_label(ax, l)
    h = [Line2D([], [], ls="none", marker="*", ms=12, color=ps.VERMILION,
                mec=ps.INK, label="calibrated (D-5.13)"),
         Line2D([], [], ls="none", marker="o", color=col,
                label="D-5.13 table")]
    a.legend(handles=h, loc="upper left", bbox_to_anchor=(0.0, 0.90),
             frameon=True, handlelength=1.0)
    ps.save(fig, out)
    return 2


FIGS = {"fig8": fig8, "fig10": fig10, "fig11": fig11, "fig12": fig12,
        "figS2": figS2}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="docs/results")
    ap.add_argument("--out", default="docs/figs")
    ap.add_argument("--only", default=",".join(FIGS))
    ap.add_argument("--require", action="store_true")
    a = ap.parse_args(argv)
    ps.apply()
    final = load_final(a.results)
    os.makedirs(a.out, exist_ok=True)
    skipped = 0
    for name in a.only.split(","):
        path = os.path.join(a.out, f"{name}.png")
        try:
            n = FIGS[name](final, a.results, path)
            w, h = ps.png_size_in(path)
            print(f"wrote {path} (+ .pdf)  {w:.2f} x {h:.2f} in  ({n} series)")
        except NoData as e:
            skipped += 1
            print(f"SKIPPED {e}")
    return 1 if (a.require and skipped) else 0


if __name__ == "__main__":
    sys.exit(main())
