r"""Fig. 4 -- per-term training loss on semi-log vs epoch, Adam -> L-BFGS marked.

THE SPEC AS WRITTEN IS NOT DRAWABLE. Three corrections, all forced by decisions
already taken; do not "fix" this script back toward the original wording.

  1. `L_data` DOES NOT EXIST (D-3.2.1). There is no data-misfit term and there
     will not be one. Plotting it would require inventing it.

  2. THE TERM LIST IS SIX, NOT FOUR: bc_mech, pde_mech, pde_richards, bc,
     ic_head, ic_disp (`interface` is a seventh but freezes -- see below).
     `L_PDE` is not one curve: the hydraulic and mechanical residuals behave
     oppositely, so summing them hides the only thing the figure is for.

  3. DO NOT PLOT THE WEIGHTED TOTAL. It steps discontinuously at every
     balancer update (`bal.w` changes, so the objective changes), and a healthy
     run looks divergent. What is plotted is each term's UNWEIGHTED loss as a
     ratio to its own step-0 value, which is what the log already stores as
     `L` and `L0`.

`interface` is drawn dashed and greyed: it freezes below `activity_floor`
early (D-W.4) and its later values are a held weight, not training progress.

    PYTHONPATH=. python scripts/make_fig4.py runs/polish_exp_seed7/log.jsonl
    PYTHONPATH=. python scripts/make_fig4.py LOG --out docs/fig4.png
    PYTHONPATH=. python scripts/make_fig4.py LOG1 LOG2 --labels adam polish

Style: `src/plot_style.py` (Times New Roman 12 pt, 6.25 in wide, PNG + PDF).
The legend sits in a strip above the axes so no trajectory is covered.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from src import plot_style as ps

#: Drawn in this order; bc_mech first because it is the anchor (D-W.2).
TERMS = ("bc_mech", "pde_mech", "pde_richards", "bc", "ic_head", "ic_disp")
FROZEN_TERMS = ("interface",)

COLOURS = ps.LOSS
STYLES = ps.LOSS_STYLE
MARKERS = {"bc_mech": "s", "pde_mech": "o", "pde_richards": "D", "bc": "^",
           "ic_head": "v", "ic_disp": "P", "interface": None}

#: Legend text. The code names are kept (they match the loss module and
#: Fig 2) but typeset as symbols, as in the Methodology.
LABELS = {"bc_mech": r"$\mathcal{L}_{\mathrm{BC,mech}}$",
          "pde_mech": r"$\mathcal{L}_{\mathrm{PDE,mech}}$",
          "pde_richards": r"$\mathcal{L}_{\mathrm{PDE,Richards}}$",
          "bc": r"$\mathcal{L}_{\mathrm{BC,hyd}}$",
          "ic_head": r"$\mathcal{L}_{\mathrm{IC},\psi}$",
          "ic_disp": r"$\mathcal{L}_{\mathrm{IC},u}$",
          "interface": r"$\mathcal{L}_{\mathrm{interface}}$"}


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"{path}: no records")
    return rows


def series(rows, term):
    """(steps, L/L0) for one term, skipping records where it is absent or
    non-positive. A zero cannot be drawn on a log axis and silently dropping
    it is better than a gap the reader cannot see."""
    xs, ys = [], []
    for r in rows:
        L = (r.get("L") or {}).get(term)
        L0 = (r.get("L0") or {}).get(term)
        if L is None or not L0 or L <= 0:
            continue
        xs.append(r["step"])
        ys.append(L / L0)
    return xs, ys


def _sci(v: float) -> str:
    m, e = f"{v:.0e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def _roughness(ys) -> float:
    """Median |step-to-step change| in decades, over the second half of the
    run. Above ~0.5 a curve is noise-dominated at the plotted resolution."""
    import numpy as np
    y = np.log10(np.asarray(ys[len(ys) // 2:], float))
    return float(np.median(np.abs(np.diff(y)))) if y.size > 2 else 0.0


def handover_step(rows):
    """Where Adam ends. Prefers the explicit `phase` key; falls back to the
    largest step gap, which is what a handover looks like when the phase key
    is absent (the ansatz-era logs do not carry one)."""
    phases = [(r["step"], r.get("phase")) for r in rows if r.get("phase")]
    if phases:
        for i in range(1, len(phases)):
            if phases[i][1] != phases[i - 1][1]:
                return phases[i][0], "phase key"
        return None, "single phase"
    return None, "no phase key"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="one or more log.jsonl")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default="docs/figs/fig4.png",
                    help="PNG path; a PDF is written next to it")
    ap.add_argument("--title", default=None)
    ap.add_argument("--no-interface", action="store_true")
    ap.add_argument("--ymin", type=float, default=1e-16,
                    help="lower y limit; the frozen interface term (~1e-26) "
                         "is annotated rather than allowed to compress the "
                         "axis by ten decades")
    ap.add_argument("--code-labels", action="store_true",
                    help="legend with code names (bc_mech, ...) not symbols")
    a = ap.parse_args(argv)
    ps.apply()

    rows = []
    for p in a.logs:
        rows.extend(load(p))
    rows.sort(key=lambda r: r["step"])

    fig, ax = plt.subplots(figsize=(ps.WIDTH, 4.4), layout="constrained")
    lab = (lambda t: t) if a.code_labels else (lambda t: LABELS.get(t, t))

    drawn = []
    xmax = rows[-1]["step"]
    for term in TERMS:
        xs, ys = series(rows, term)
        if not xs:
            continue
        # The noisiest trajectories go underneath and thinner, so they do not
        # hide the smooth ones; the data are drawn unsmoothed.
        noisy = _roughness(ys) > 0.5
        ax.semilogy(xs, ys, label=lab(term), color=COLOURS[term],
                    ls=STYLES[term], lw=0.9 if noisy else 1.5,
                    zorder=2 if noisy else 3, marker=MARKERS[term], ms=4.5,
                    markevery=max(1, len(xs) // 9), mec="white", mew=0.6)
        drawn.append(term)
    below = []
    if not a.no_interface:
        for term in FROZEN_TERMS:
            xs, ys = series(rows, term)
            if xs:
                ax.semilogy(xs, ys, label=lab(term) + " (frozen)",
                            color=COLOURS[term], lw=1.2, ls=STYLES[term],
                            zorder=1)
                drawn.append(term)
                if min(ys) < a.ymin:
                    below.append((term, min(ys)))

    if not drawn:
        raise SystemExit("nothing drawable: no term had a positive L and L0")

    ax.axhline(1.0, color="#9CA3AF", lw=0.8, ls="-", zorder=1)

    step, how = handover_step(rows)
    if step is not None:
        ax.axvline(step, color=ps.INK, lw=1.0, ls="-.")
        ax.annotate("Adam → L-BFGS", xy=(step, 1.0), xytext=(4, 4),
                    textcoords="offset points", rotation=90, va="bottom")

    ax.set_xlim(0, xmax)
    ax.set_ylim(bottom=a.ymin)
    for term, lo in below:
        # Stated on the figure, not silently cut: the frozen term leaves the
        # axis, and its floor is printed so the reader knows where it went.
        ax.text(0.985, 0.03, f"{LABELS.get(term, term) if not a.code_labels else term}"
                f" continues below the axis (min {_sci(lo)})", ha="right",
                va="bottom", transform=ax.transAxes, color=ps.GREY)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v/1000:g}k" if v else "0"))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10),
                                          numticks=12))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\mathcal{L}_i\,/\,\mathcal{L}_i(0)$ (unweighted)")
    if a.title:
        ax.set_title(a.title)
    ax.grid(True, which="major")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center",
               ncol=4, handlelength=2.2, columnspacing=0.9, frameon=False)
    for p in ps.save(fig, a.out):
        print(f"wrote {p}")
    print(f"records {len(rows)}, steps {rows[0]['step']}-{rows[-1]['step']}, "
          f"terms drawn: {', '.join(drawn)}")
    print(f"handover marker: {how}"
          + (f" at step {step}" if step is not None else " -- NOT MARKED"))

    # The diagnosis Day 28 actually asks for: which terms never descend.
    print("\nfinal L/L0 per term (a term at or above 1.0 never descended):")
    for term in drawn:
        xs, ys = series(rows, term)
        print(f"   {term:<14}{ys[-1]:>12.3e}   min {min(ys):.3e} "
              f"at epoch {xs[ys.index(min(ys))]}")
    print("\nNOTE: this figure is only meaningful for runs under a FIXED\n"
          "activity floor. `regime_floor` ships at 0.0 (D-A.2), so any run\n"
          "made before that is consistent; if you enable the regime gate,\n"
          "do not mix runs from either side of the change on one axis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())