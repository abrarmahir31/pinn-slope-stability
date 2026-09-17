r"""Collect every SSR result into one table. REPORTS; DOES NOT JUDGE.

    python scripts/collect_results.py --runs "runs/ssr*" --out docs/results

Writes fos_table.json and fos_table.md. Smoke runs are excluded. Groups are
formed only among runs whose settings match on everything except the varied
quantity, so a w_yield plateau never mixes sig3 ranges or failure thresholds:

  plateau      FOS vs w_yield       (baseline arm, per criterion + settings)
  cold_start   warm vs cold FOS     (same settings and w_yield)
  tornado      sensitivity.tornado  (arms with a matching baseline run)
  coupling     sensitivity.coupling_effect, KC default vs KC off
  replicates   sensitivity.seed_statistics over distinct baselines/seeds
  n_pde        sensitivity.convergence_check over N_PDE levels

Whether a spread is small enough, or which w_yield goes in the thesis, is not
decided here.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

from src import sensitivity as sens

# D-5.6: runs on these frozen baselines validate code; they are not headline
# results. Identified by checkpoint SHA-256 from the baseline's MANIFEST.json,
# so a copied or renamed checkpoint is still recognised.
VALIDATION_ONLY_BASELINES = ("baseline-v1",)


def validation_only_shas(root: str = "baselines") -> dict:
    out = {}
    for tag in VALIDATION_ONLY_BASELINES:
        mp = os.path.join(root, tag, "MANIFEST.json")
        if not os.path.exists(mp):
            continue
        with open(mp) as f:
            files = json.load(f).get("files", {})
        for name, meta in files.items():
            if name.endswith(".pt") and meta.get("sha256"):
                out[meta["sha256"]] = tag
    return out


def headline_status(run: dict, shas: dict) -> str:
    """'eligible', 'validation_only:<tag>', or 'unknown' (no sha recorded)."""
    sha = run.get("baseline_sha256")
    if not sha:
        return "unknown"
    if sha in shas:
        return f"validation_only:{shas[sha]}"
    return "eligible"


_SETTINGS = ("criterion", "yield_norm", "admissible_metric", "admissible_drop",
             "plateau_factor", "disp_factor", "epochs", "lr", "sig3_range",
             "mc_design")


def load_runs(pattern: str, baselines_root: str = "baselines") -> list[dict]:
    shas = validation_only_shas(baselines_root)
    rows = []
    for rj in sorted(glob.glob(os.path.join(pattern, "result.json"))):
        with open(rj) as f:
            r = json.load(f)
        if r.get("smoke"):
            continue
        arm = r.get("arm") or {}
        r["dir"] = os.path.dirname(rj)
        r["factor"] = arm.get("factor")
        r["level"] = arm.get("level")
        r["headline"] = headline_status(r, shas)
        rows.append(r)
    return rows


def _key(r, *extra):
    base = tuple(json.dumps(r.get(k), sort_keys=True) for k in _SETTINGS)
    return base + tuple(json.dumps(r.get(k), sort_keys=True) for k in extra)


def _with_fos(rows):
    return [r for r in rows if r.get("fos") is not None]


def plateau(rows):
    groups = defaultdict(list)
    for r in _with_fos(rows):
        if r["factor"] is None:
            groups[_key(r, "cold_start", "kc", "n_pde", "baseline",
                        "sampling_seed")].append(r)
    out = []
    for g in groups.values():
        if len({r["w_yield"] for r in g}) < 2:
            continue
        pts = sorted((r["w_yield"], r["fos"]) for r in g)
        f = [p[1] for p in pts]
        mean = sum(f) / len(f)
        out.append({"criterion": g[0]["criterion"],
                    "cold_start": g[0]["cold_start"],
                    "points": pts, "min": min(f), "max": max(f),
                    "rel_spread": (max(f) - min(f)) / mean,
                    "dirs": [r["dir"] for r in g]})
    return out


def cold_start(rows):
    groups = defaultdict(dict)
    for r in _with_fos(rows):
        if r["factor"] is None:
            k = _key(r, "w_yield", "kc", "n_pde", "baseline", "sampling_seed")
            groups[k][bool(r["cold_start"])] = r
    return [{"criterion": g[False]["criterion"], "w_yield": g[False]["w_yield"],
             "warm": g[False]["fos"], "cold": g[True]["fos"],
             "diff": g[True]["fos"] - g[False]["fos"]}
            for g in groups.values() if True in g and False in g]


def tornado(rows):
    base = {}
    for r in _with_fos(rows):
        if r["factor"] is None and not r["cold_start"] and r["kc"] == "default":
            base[_key(r, "w_yield", "n_pde", "baseline", "sampling_seed")] = r
    arms_ = defaultdict(dict)
    for r in _with_fos(rows):
        if r["factor"] not in (None, "coupling") and not r["cold_start"]:
            k = _key(r, "w_yield", "n_pde", "baseline", "sampling_seed")
            arms_[k][(r["factor"], r["level"])] = r["fos"]
    out = []
    for k, res in arms_.items():
        if k not in base:
            continue
        bars = sens.tornado({("baseline", "base"): base[k]["fos"], **res})
        out.append({"criterion": base[k]["criterion"],
                    "w_yield": base[k]["w_yield"], "base": base[k]["fos"],
                    "bars": [{"factor": b.factor, "low": b.low, "high": b.high,
                              "low_pct": b.low_pct, "high_pct": b.high_pct,
                              "span": b.span} for b in bars]})
    return out


def coupling(rows):
    groups = defaultdict(dict)
    for r in _with_fos(rows):
        if r["factor"] in (None, "coupling") and not r["cold_start"]:
            k = _key(r, "w_yield", "n_pde", "baseline", "sampling_seed")
            groups[k][r["kc"]] = r
    out = []
    for g in groups.values():
        if "default" in g and "off" in g:
            out.append({"criterion": g["default"]["criterion"],
                        "w_yield": g["default"]["w_yield"],
                        "fos": {kc: g[kc]["fos"] for kc in g},
                        "one_way_overestimate_pct": sens.coupling_effect(
                            g["default"]["fos"], g["off"]["fos"])})
    return out


def replicates(rows):
    groups = defaultdict(dict)
    for r in _with_fos(rows):
        if r["factor"] is None and not r["cold_start"] and r["kc"] == "default":
            k = _key(r, "w_yield", "n_pde")
            groups[k][(r["baseline"], r["sampling_seed"])] = r
    out = []
    for g in groups.values():
        if len(g) >= 2:
            s = sens.seed_statistics([r["fos"] for r in g.values()])
            out.append({"criterion": next(iter(g.values()))["criterion"],
                        "n": s.n, "mean": s.mean, "sd": s.sd,
                        "members": sorted(map(str, g))})
    return out


def n_pde(rows):
    groups = defaultdict(dict)
    for r in _with_fos(rows):
        if r["factor"] is None and not r["cold_start"] and r["kc"] == "default":
            groups[_key(r, "w_yield", "baseline", "sampling_seed")][r["n_pde"]] = r
    out = []
    for g in groups.values():
        if len(g) >= 2:
            ns = sorted(g)
            conv, rel = sens.convergence_check(ns, [g[n]["fos"] for n in ns])
            out.append({"criterion": g[ns[0]]["criterion"], "n_pde": ns,
                        "fos": [g[n]["fos"] for n in ns],
                        "rel_changes": rel, "last_change_within_1pct": conv})
    return out


def to_markdown(rows, summary) -> str:
    L = ["# SSR results (generated by scripts/collect_results.py)", "",
         "Runs marked `validation_only` used a D-5.6 validation baseline and "
         "must not be reported as headline results.", "",
         "| run | criterion | w_yield | norm | cold | kc | arm | FOS | bracket | status | headline |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        fos = "—" if r.get("fos") is None else f"{r['fos']:.3f}"
        arm = "—" if r["factor"] is None else f"{r['factor']}:{r['level']}"
        L.append(f"| {os.path.basename(r['dir'])} | {r['criterion']} | "
                 f"{r['w_yield']} | {r['yield_norm']} | {r['cold_start']} | "
                 f"{r['kc']} | {arm} | {fos} | {r.get('bracket')} | "
                 f"{r.get('status')} | {r['headline']} |")
    for name, items in summary.items():
        L += ["", f"## {name}", "", "```", json.dumps(items, indent=1), "```"]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs/ssr*")
    ap.add_argument("--out", default="docs/results")
    ap.add_argument("--baselines", default="baselines")
    a = ap.parse_args(argv)
    rows = load_runs(a.runs, a.baselines)
    if not rows:
        print(f"no non-smoke result.json under {a.runs}", file=sys.stderr)
        return 1
    summary = {"plateau": plateau(rows), "cold_start": cold_start(rows),
               "tornado": tornado(rows), "coupling": coupling(rows),
               "replicates": replicates(rows), "n_pde": n_pde(rows)}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "fos_table.json"), "w") as f:
        json.dump({"runs": rows, **summary}, f, indent=1)
    with open(os.path.join(a.out, "fos_table.md"), "w") as f:
        f.write(to_markdown(rows, summary))
    print(f"{len(rows)} runs -> {a.out}/fos_table.json, fos_table.md")
    n_val = sum(r["headline"] != "eligible" for r in rows)
    if n_val:
        print(f"  {n_val} run(s) are NOT headline-eligible (validation baseline "
              f"or no baseline hash) -- see the headline column")
    for name, items in summary.items():
        print(f"  {name}: {len(items)} group(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
