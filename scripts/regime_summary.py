r"""One line per (cell, time): does ANY stratum transport water?

Reads the artifacts `inspect_richards_terms.py --json` writes and collapses
them to the question D-A.1 actually turns on. The per-cell tables are three
strata x four times each; across nine ablation cells that is 108 rows, and the
thing being looked for is a single `solving`.

    PYTHONPATH=. python scripts/regime_summary.py docs/richards_terms_*.json
    PYTHONPATH=. python scripts/regime_summary.py            # all in docs/

`reach` is p / the material's attainable ceiling (D-A.2), so it IS comparable
across strata and across cells; raw p is not. `reach_max` below is the best
any stratum manages at that time.

A cell whose `reach_max` never leaves 0.000 is not solving Richards. It may
still post a small `pde_richards`, because a degenerate equation is trivially
satisfiABLE -- that is the whole reason this script exists rather than reading
the loss.
"""
import glob
import json
import sys
from collections import Counter

import paths


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    files = []
    for a in argv:
        files.extend(sorted(glob.glob(a)))
    if not files:
        files = sorted(str(p) for p in paths.DOCS.glob("richards_terms_*.json"))
    if not files:
        print("no richards_terms_*.json found; generate with\n"
              "  inspect_richards_terms.py CKPT --json docs/richards_terms_NAME.json")
        return 1

    print(f"{'cell':<26}{'t':>5}{'reach_max':>11}{'K/K_ic min':>12}"
          f"{'K/K_ic max':>12}{'psi>=0':>8}  verdicts")
    print("-" * 100)
    overall = Counter()
    best_reach = {}

    for f in files:
        try:
            d = json.load(open(f))
        except (OSError, ValueError) as e:
            print(f"{f}: unreadable ({e})")
            continue
        name = f.replace("\\", "/").rsplit("/", 1)[-1]
        name = name.replace("richards_terms_", "").replace(".json", "")
        rows = d.get("rows", [])
        if not rows or "reach" not in rows[0]:
            print(f"{name:<26}  STALE ARTIFACT -- no 'reach' field. Regenerate.")
            continue
        times = sorted({r["t_days"] for r in rows})
        for t in times:
            at = [r for r in rows if r["t_days"] == t]
            rmax = max(r["reach"] for r in at)
            kmin = min(r["k_over_k_ic"] for r in at)
            kmax = max(r["k_over_k_ic"] for r in at)
            sat = max(r.get("psi_ge0_frac", 0.0) for r in at)
            vs = Counter(r["verdict"] for r in at)
            overall.update(vs)
            best_reach[name] = max(best_reach.get(name, 0.0), rmax)
            label = " ".join(f"{k}x{v}" for k, v in sorted(vs.items()))
            print(f"{name:<26}{t:>5.0f}{rmax:>11.3f}{kmin:>12.3g}"
                  f"{kmax:>12.3g}{sat:>8.3f}  {label}")
        print()

    print("verdict totals across every cell and time:")
    for k, v in sorted(overall.items(), key=lambda kv: -kv[1]):
        print(f"   {k:<20}{v:>5}")

    print("\nbest reach any stratum achieved, per cell:")
    for k, v in sorted(best_reach.items(), key=lambda kv: -kv[1]):
        print(f"   {k:<26}{v:>9.3f}")

    if overall and not overall.get("solving"):
        print("\nNO CELL SOLVES. Not one stratum at one time reaches 10% of\n"
              "what its own material permits. The ablation is then choosing\n"
              "between three ways of not transporting water, and D-A.1's\n"
              "ranking is a ranking of boundary/IC fit only -- which is worth\n"
              "recording, but is not the claim the arm names suggest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())