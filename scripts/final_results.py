r"""The reportable Phase 5/6 numbers under the ADOPTED criterion. NO GPU.

    python scripts/final_results.py --results docs/results

Reads only docs/results/<run>_result.json and <run>_sweep.jsonl (what git
holds; runs/ is gitignored) and writes FINAL_RESULTS.md + final_results.json.

Adopted criterion (supervisor, 27 Sep):
  D-5.18  failure = loss > 5 x reference AND admissible (min_tag) < 0.50;
          the displacement condition is dropped (O-22).
  D-5.19  the loss reference is L(SRF 1) of the same configuration WITHOUT
          the arm's change: the baseline run for physics arms (K_s, GSI,
          coupling), the run itself for everything else (w_yield plateau,
          MC, N_PDE, collocation draws).

For each configuration the value reported is, in order of preference:
  1. a rerun made under the adopted criterion (`<name>_lossfloor` or
     `<name>_common`, disp_factor 0), whose own bisection resolved it; else
  2. the original sweep re-read under the adopted criterion, accepted only
     if the bracket it gives is no wider than 2 x bisect-tol; else
  3. UNRESOLVED, with the bracket.

Every FOS here is CALIBRATED (D-5.13): the criterion was chosen to put the
baseline near 1.5. Only relative statements are predictive.
"""
import argparse
import glob
import json
import math
import os
import statistics
import sys

BASELINE = "ssr_ghb_w1"
PLATEAU = 5.0
FLOOR = 0.50
TOL = 0.01
RERUN_SUFFIXES = ("_lossfloor", "_common")
NOT_REPORTED = ("ssr_ghb_n5000", "ssr_ghb_n20000")   # unscaled, D-5.17: evidence only


# --------------------------------------------------------------------------- io
def load(results_dir: str) -> dict:
    runs = {}
    for p in sorted(glob.glob(os.path.join(results_dir, "*_result.json"))):
        name = os.path.basename(p)[:-len("_result.json")]
        if not name.startswith("ssr_"):
            continue
        sp = os.path.join(results_dir, name + "_sweep.jsonl")
        with open(p, encoding="utf-8") as f:
            res = json.load(f)
        if res.get("smoke") or not os.path.exists(sp):
            continue
        by = {}
        with open(sp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if r.get("finite", True) and "admissible" in r:
                        by[round(float(r["srf"]), 6)] = r
        runs[name] = {"result": res, "records": [by[k] for k in sorted(by)]}
    return runs


# --------------------------------------------------------------------------- criterion
def reread(records: list, loss_threshold: float) -> dict:
    """Bracket under loss + floor from the SRF points the sweep evaluated."""
    def failed(r):
        return r["total"] > loss_threshold and r["admissible"] < FLOOR
    fails = [r["srf"] for r in records if failed(r)]
    if not fails:
        return {"status": "no_failure", "fos": None, "bracket": None}
    hi = min(fails)
    stable = [r["srf"] for r in records if not failed(r) and r["srf"] < hi]
    lo = max(stable) if stable else None
    if lo is None:
        return {"status": "fails_at_start", "fos": None, "bracket": [None, hi]}
    return {"status": "fos", "fos": 0.5 * (lo + hi), "bracket": [lo, hi],
            "width": hi - lo}


def is_arm(res: dict) -> bool:
    return bool(res.get("arm"))


def adopted(runs: dict) -> dict:
    """name (without rerun suffix) -> reportable entry."""
    base_L = runs[BASELINE]["records"][0]["total"]
    out = {}
    names = sorted(n for n in runs if not n.endswith(RERUN_SUFFIXES))
    for n in names:
        res, recs = runs[n]["result"], runs[n]["records"]
        ref = base_L if is_arm(res) else recs[0]["total"]
        rr = reread(recs, PLATEAU * ref)
        e = {"name": n, "as_run": {"fos": res.get("fos"),
                                   "bracket": res.get("bracket")},
             "reread": rr, "loss_reference": ref,
             "reference_rule": "baseline (D-5.19)" if is_arm(res) else "own",
             "w_yield": res.get("w_yield"), "n_pde": res.get("n_pde"),
             "criterion": res.get("criterion"), "kc": res.get("kc"),
             "sampling_seed": res.get("sampling_seed"),
             "arm": (res.get("arm") or {}).get("label"),
             "cold_start": res.get("cold_start"),
             "sigma0_consistent": res.get("sigma0_consistent", True)}
        rerun = next((n + s for s in RERUN_SUFFIXES if n + s in runs), None)
        if rerun:
            r2 = runs[rerun]["result"]
            e.update(value=r2["fos"], bracket=r2["bracket"], source=rerun,
                     how="rerun under the adopted criterion")
        elif rr["status"] == "fos" and rr["width"] <= 2 * TOL + 1e-9:
            e.update(value=rr["fos"], bracket=rr["bracket"], source=n,
                     how="as-run sweep re-read (bracket resolved)")
        else:
            e.update(value=None, bracket=rr.get("bracket"), source=n,
                     how="UNRESOLVED: rerun needed")
        e["reported"] = n not in NOT_REPORTED
        out[n] = e
    return out


# --------------------------------------------------------------------------- tables
def v(tab, n):
    return tab[n]["value"] if n in tab else None


def summary(tab: dict) -> dict:
    base = v(tab, BASELINE)
    reps = [v(tab, n) for n in [BASELINE] + [f"ssr_ghb_draw{s}" for s in (11, 12, 13, 14)]
            if v(tab, n) is not None]
    rep = {"n": len(reps), "values": reps,
           "mean": statistics.mean(reps) if reps else None,
           "sd": statistics.stdev(reps) if len(reps) > 1 else None}
    ghb = {w: v(tab, f"ssr_ghb_w{w}") for w in ("0.3", "1", "3")}
    mc = {w: v(tab, f"ssr_mc_w{w}") for w in ("0.3", "1", "3")}
    npde = {5000: v(tab, "ssr_ghb_n5000_wscaled"), 10000: base,
            20000: v(tab, "ssr_ghb_n20000_wscaled")}

    def pct(x):
        return None if x is None or base is None else 100 * (x - base) / base
    tornado = [{"factor": f, "low_label": ll, "high_label": hl,
                "low": v(tab, lo), "high": v(tab, hi),
                "low_pct": pct(v(tab, lo)), "high_pct": pct(v(tab, hi)),
                "swing": (None if None in (v(tab, lo), v(tab, hi))
                          else abs(v(tab, hi) - v(tab, lo)))}
               for f, lo, hi, ll, hl in (
                   ("Mk_d GSI", "ssr_arm_Mk_d_GSI_low", "ssr_arm_Mk_d_GSI_high", "GSI 35", "GSI 55"),
                   ("Mk_d K_s", "ssr_arm_Mk_d_K_s_low", "ssr_arm_Mk_d_K_s_high", "x0.1", "x10"))]
    tornado.sort(key=lambda b: -(b["swing"] or 0))
    coupling = {"off (one-way)": v(tab, "ssr_arm_coupling_low"),
                "marls only (default)": base,
                "everywhere": v(tab, "ssr_arm_coupling_high")}
    return {"baseline": base, "replicates": rep,
            "plateau": {"GHB": ghb, "MC": mc,
                        "GHB_spread": _spread(ghb), "MC_spread": _spread(mc)},
            "mc_minus_ghb": {w: (None if None in (mc[w], ghb[w]) else mc[w] - ghb[w])
                             for w in ghb},
            "n_pde": npde,
            "n_pde_rel_change": {n: (None if x is None or base is None
                                     else (x - base) / base) for n, x in npde.items()},
            "tornado": tornado, "coupling": coupling,
            "coupling_one_way_minus_default": (
                None if None in (coupling["off (one-way)"], base)
                else coupling["off (one-way)"] - base)}


def _spread(d):
    xs = [x for x in d.values() if x is not None]
    return (max(xs) - min(xs)) if len(xs) > 1 else None


def pc(x):
    return "—" if x is None else f"{x:+.1f}"


def sgn(x):
    return "—" if x is None else f"{x:+.3f}"


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def fb(b):
    if not b or b[0] is None:
        return "—"
    return f"{b[0]:.3f}–{b[1]:.3f}"


def to_markdown(tab: dict, s: dict) -> str:
    rep = s["replicates"]
    sd = rep["sd"]
    L = ["# Final Phase 5/6 results (scripts/final_results.py)", "",
         "Criterion: loss > 5 × reference AND admissible fraction (least-"
         "admissible stratum) < 0.50 — D-5.18 (displacement test dropped) and "
         "D-5.19 (physics arms use the baseline's loss reference). Baseline-v2, "
         "GHB σ₃ 0–179 kPa, raw yield norm, legacy yield reduction (D-5.17), "
         "SRF step 0.25 bisected to 0.01.", "",
         "**Every FOS is calibrated (D-5.13).** The criterion was chosen to put "
         "the baseline near 1.5; the absolute values are not predictions. The "
         "relative results below are.", "",
         "## 1. Collocation-draw replicates (the noise yardstick)", "",
         "| run | sampling seed | FOS |", "|---|---|---|"]
    for n in [BASELINE] + [f"ssr_ghb_draw{k}" for k in (11, 12, 13, 14)]:
        if n in tab:
            L.append(f"| {n} | {tab[n]['sampling_seed']} | {f3(tab[n]['value'])} |")
    if sd:
        top = max(rep["values"]) == s["baseline"]
        L += ["", f"**Mean {f3(rep['mean'])}, SD {f3(sd)} (n = {rep['n']}, CV "
              f"{100 * sd / rep['mean']:.1f}%).**"
              + (" The baseline seed (7) is the highest; the calibration target "
                 "was met on it." if top else "")
              + f" Use SD = {f3(sd)} as the resolution of a single run; the SD "
              f"of a difference between two single runs is √2 × that ≈ "
              f"{f3(sd * math.sqrt(2))}."]
    else:
        L += ["", f"Only {rep['n']} replicate(s): no SD, so no resolution "
              f"statement can be made."]
    L += ["",
          "## 2. w_yield plateau and MC vs GHB (Fig 10)", "",
          "| w_yield | FOS_GHB | FOS_MC | MC − GHB |", "|---|---|---|---|"]
    for w in ("0.3", "1", "3"):
        L.append(f"| {w} | {f3(s['plateau']['GHB'][w])} | {f3(s['plateau']['MC'][w])} "
                 f"| {sgn(s['mc_minus_ghb'][w])} |")
    d1 = s["mc_minus_ghb"]["1"]
    L += ["", f"Spread over w_yield 0.3–3: GHB {f3(s['plateau']['GHB_spread'])}, "
          f"MC {f3(s['plateau']['MC_spread'])}."]
    if d1 is not None and sd:
        dsd = sd * math.sqrt(2)
        L[-1] += (f" At w_yield 1, MC − GHB = {d1:+.3f} against a difference-SD "
                  f"of ≈ {dsd:.3f} ({abs(d1) / dsd:.1f} SD): "
                  + ("**not resolvable by FOS**." if abs(d1) < 2 * dsd
                     else "**resolved (> 2 SD)**.")
                  + " (Replicate SD measured for GHB only.)")
    L += ["",
          "## 3. Sensitivity (Fig 12) — seed-7 arms against the seed-7 baseline "
          f"{f3(s['baseline'])}", "",
          "| factor | low | FOS | Δ% | high | FOS | Δ% | swing |",
          "|---|---|---|---|---|---|---|---|"]
    for b in s["tornado"]:
        L.append(f"| {b['factor']} | {b['low_label']} | {f3(b['low'])} | "
                 f"{pc(b['low_pct'])} | {b['high_label']} | {f3(b['high'])} | "
                 f"{pc(b['high_pct'])} | {f3(b['swing'])} |")
    L += ["", "GSI arms run against the baseline σ₀ (O-11, stated as a "
          "limitation). K_s ×10 and GSI 35 are the D-5.19 reruns.", "",
          "## 4. Coupling (Fig 11)", "", "| Kozeny–Carman feedback | FOS |",
          "|---|---|"]
    for k, x in s["coupling"].items():
        L.append(f"| {k} | {f3(x)} |")
    cd = s["coupling_one_way_minus_default"]
    L += ["", "—" if cd is None else
          f"One-way minus default: {cd:+.3f}"
          + (f" ({abs(cd) / sd:.1f} replicate SD)." if sd else ".")]
    L += ["",
          "## 5. Collocation convergence (N_PDE, matched effective w_yield, D-5.17)",
          "", "| N_PDE | w_yield | FOS | change vs 10k |", "|---|---|---|---|"]
    for n, w in ((5000, 2.0), (10000, 1.0), (20000, 0.5)):
        rc = s["n_pde_rel_change"][n]
        L.append(f"| {n:,} | {w:g} | {f3(s['n_pde'][n])} | "
                 f"{'—' if n == 10000 or rc is None else f'{100 * rc:+.1f}%'} |")
    diffs = [abs(x - s["baseline"]) for n, x in s["n_pde"].items()
             if n != 10000 and x is not None]
    worst = max(diffs) if diffs else None
    L += ["", ("—" if worst is None else
               f"Largest change from 10k: {worst:.3f}"
               + (f" = {worst / sd:.1f} replicate SD." if sd else ".")), "",
          "## 6. Every configuration: as run vs adopted criterion", "",
          "| configuration | loss reference | FOS as run | FOS adopted | bracket | how |",
          "|---|---|---|---|---|---|"]
    for n, e in sorted(tab.items()):
        tag = "" if e["reported"] else " *(evidence only)*"
        L.append(f"| {n}{tag} | {e['reference_rule']} | {f3(e['as_run']['fos'])} | "
                 f"**{f3(e['value'])}** | {fb(e['bracket'])} | {e['how']} |")
    L += ["", "The cold-start check under the adopted criterion is bracketed "
          "only to 0.25 (it contains the warm value); the 5,000-epoch cold "
          "step of 22 Sep remains the evidence that warm-starting is unbiased.",
          ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join("docs", "results"))
    a = ap.parse_args(argv)
    runs = load(a.results)
    if BASELINE not in runs:
        print(f"{BASELINE} not found in {a.results}", file=sys.stderr)
        return 1
    tab = adopted(runs)
    s = summary(tab)
    with open(os.path.join(a.results, "final_results.json"), "w", encoding="utf-8") as f:
        json.dump({"configurations": tab, "summary": s}, f, indent=1)
    with open(os.path.join(a.results, "FINAL_RESULTS.md"), "w", encoding="utf-8") as f:
        f.write(to_markdown(tab, s))
    bad = [n for n, e in tab.items() if e["value"] is None and e["reported"]
           and not e["cold_start"]]
    print(f"{len(tab)} configurations -> {a.results}/FINAL_RESULTS.md, final_results.json")
    if bad:
        print(f"  UNRESOLVED (rerun needed): {bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
