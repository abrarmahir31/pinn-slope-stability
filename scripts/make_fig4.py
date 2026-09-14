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
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#: Drawn in this order; bc_mech first because it is the anchor (D-W.2).
TERMS = ("bc_mech", "pde_mech", "pde_richards", "bc", "ic_head", "ic_disp")
FROZEN_TERMS = ("interface",)

COLOURS = {
    "bc_mech": "#000000",
    "pde_mech": "#d62728",
    "pde_richards": "#1f77b4",
    "bc": "#2ca02c",
    "ic_head": "#ff7f0e",
    "ic_disp": "#9467bd",
    "interface": "#999999",
}


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
    ap.add_argument("--out", default="docs/fig4.png")
    ap.add_argument("--title", default=None)
    ap.add_argument("--no-interface", action="store_true")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args(argv)

    rows = []
    for p in a.logs:
        rows.extend(load(p))
    rows.sort(key=lambda r: r["step"])

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    drawn = []
    for term in TERMS:
        xs, ys = series(rows, term)
        if not xs:
            continue
        ax.semilogy(xs, ys, label=term, color=COLOURS[term], lw=1.6)
        drawn.append(term)
    if not a.no_interface:
        for term in FROZEN_TERMS:
            xs, ys = series(rows, term)
            if xs:
                ax.semilogy(xs, ys, label=f"{term} (frozen)",
                            color=COLOURS[term], lw=1.1, ls="--", alpha=0.7)
                drawn.append(term)

    if not drawn:
        raise SystemExit("nothing drawable: no term had a positive L and L0")

    ax.axhline(1.0, color="#666666", lw=0.8, ls=":", zorder=0)

    step, how = handover_step(rows)
    if step is not None:
        ax.axvline(step, color="#333333", lw=1.0, ls="-.")
        ax.annotate("Adam → L-BFGS", xy=(step, ax.get_ylim()[1]),
                    xytext=(4, -12), textcoords="offset points",
                    fontsize=8, rotation=90, va="top")

    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$L_i / L_i(0)$  (unweighted)")
    ax.set_title(a.title or "Per-term training loss, relative to initialisation")
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, ncol=2, framealpha=0.9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi)
    print(f"wrote {a.out}")
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