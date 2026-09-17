r"""Figs 8, 10, 11, 12 from SSR runs. A figure with no data is SKIPPED with a
message, never written empty (--require turns a skip into an error).

    python scripts/make_ssr_figs.py --runs "runs/ssr*" --out docs/figs

  Fig 8   loss and admissible metric vs SRF, one line per run (baseline arm)
  Fig 10  FOS vs w_yield per criterion -- the soft-constraint plateau --
          with the LEM value from geometry.F_TARGETS for reference
  Fig 11  coupling arms (KC default / everywhere / off)
  Fig 12  tornado, from collect_results.tornado

The LEM line is Ulusay et al.'s F = 0.94 from residual bedding strength in a
limit-equilibrium analysis. It is a reference, not a target the PINN FOS is
tuned toward; the caption must say which strength each number used.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts import collect_results as C
from src.step21_geometry import geometry as g


class NoData(Exception):
    pass


def load_sweep(run_dir):
    p = os.path.join(run_dir, "sweep.jsonl")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        recs = [json.loads(l) for l in f if l.strip()]
    return sorted(recs, key=lambda r: r["srf"])


def fig8(rows, out):
    base = [r for r in rows if r["factor"] is None]
    if not base:
        raise NoData("fig8: no baseline-arm runs")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    n = 0
    for r in base:
        recs = [x for x in load_sweep(r["dir"]) if x.get("finite", True)]
        if not recs:
            continue
        s = [x["srf"] for x in recs]
        ls = "-" if r["criterion"] == "GHB" else "--"
        lab = (f"{r['criterion']} w={r['w_yield']:g}"
               f"{' cold' if r['cold_start'] else ''}")
        l, = a1.semilogy(s, [x["total"] for x in recs], ls, marker="o", ms=3,
                         label=lab)
        a2.plot(s, [x["admissible"] for x in recs], ls, marker="o", ms=3,
                color=l.get_color(), label=lab)
        if r.get("fos") is not None:
            for ax in (a1, a2):
                ax.axvline(r["fos"], color=l.get_color(), lw=0.8, ls=":")
        n += 1
    if n == 0:
        plt.close(fig)
        raise NoData("fig8: runs found but no finite sweep records")
    a1.set(xlabel="SRF", ylabel="total loss", title="loss vs SRF")
    a2.set(xlabel="SRF", ylabel=f"admissible ({base[0]['admissible_metric']})",
           title="admissible fraction vs SRF")
    a2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return n


def fig10(rows, out):
    groups = C.plateau(rows)
    if not groups:
        raise NoData("fig10: no w_yield plateau groups (need >= 2 w_yield "
                     "with FOS at matching settings)")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for grp in groups:
        w, f = zip(*grp["points"])
        ax.semilogx(w, f, "o-", label=f"{grp['criterion']}"
                    f"{' cold' if grp['cold_start'] else ''} "
                    f"(spread {100*grp['rel_spread']:.1f}%)")
    ax.axhline(g.F_TARGETS["initial_31deg"], color="k", ls="--", lw=0.8,
               label="LEM, Ulusay et al. (2014)")
    ax.set(xlabel="$w_{yield}$", ylabel="FOS", title="FOS vs penalty weight")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return len(groups)


def fig11(rows, out):
    groups = C.coupling(rows)
    if not groups:
        raise NoData("fig11: no KC default + KC off pair")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    order = ["off", "default", "all"]
    names = {"off": "KC off\n(one-way)", "default": "KC marls\n(record)",
             "all": "KC all"}
    width = 0.8 / len(groups)
    for i, grp in enumerate(groups):
        kcs = [k for k in order if k in grp["fos"]]
        xs = [order.index(k) + i * width for k in kcs]
        ax.bar(xs, [grp["fos"][k] for k in kcs], width,
               label=f"{grp['criterion']} w={grp['w_yield']:g} "
                     f"({grp['one_way_overestimate_pct']:+.1f}%)")
    ax.set_xticks(range(3), [names[k] for k in order])
    ax.set(ylabel="FOS", title="coupling arms")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return len(groups)


def fig12(rows, out):
    groups = [t for t in C.tornado(rows) if t["bars"]]
    if not groups:
        raise NoData("fig12: no arms with a matching baseline run")
    fig, axes = plt.subplots(1, len(groups), figsize=(5.5 * len(groups), 3.8),
                             squeeze=False)
    for ax, grp in zip(axes[0], groups):
        bars = list(reversed(grp["bars"]))
        for i, b in enumerate(bars):
            ax.barh(i, b["low"] - grp["base"], left=grp["base"], color="C0")
            ax.barh(i, b["high"] - grp["base"], left=grp["base"], color="C3")
        ax.set_yticks(range(len(bars)), [b["factor"] for b in bars])
        ax.axvline(grp["base"], color="k", lw=0.8)
        ax.set(xlabel="FOS", title=f"{grp['criterion']} w={grp['w_yield']:g}")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return len(groups)


FIGS = {"fig8": fig8, "fig10": fig10, "fig11": fig11, "fig12": fig12}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs/ssr*")
    ap.add_argument("--out", default="docs/figs")
    ap.add_argument("--only", default=",".join(FIGS))
    ap.add_argument("--require", action="store_true")
    a = ap.parse_args(argv)
    rows = C.load_runs(a.runs)
    os.makedirs(a.out, exist_ok=True)
    skipped = 0
    for name in a.only.split(","):
        path = os.path.join(a.out, f"{name}.png")
        try:
            n = FIGS[name](rows, path)
            print(f"wrote {path}  ({n} series/groups)")
        except NoData as e:
            skipped += 1
            print(f"SKIPPED {e}")
    return 1 if (a.require and skipped) else 0


if __name__ == "__main__":
    sys.exit(main())
