r"""Re-read every SSR sweep under both failure criteria. OFFLINE, NO GPU.

    python scripts/recompute_criterion.py --runs "runs/ssr*" --out docs/results

O-22 asks whether to drop the displacement condition from the calibrated
criterion (D-5.13). Every per-SRF record is in `sweep.jsonl`, so the answer
can be computed for every run without training anything:

  three       loss plateau AND displacement jump AND admissible condition
              (what ssr_sweep.py used; must reproduce result.json)
  loss_floor  loss plateau AND admissible condition (O-22 option "drop")

The bracket is read from the points the sweep actually evaluated. Those points
were chosen by bisection under `three`, so a `loss_floor` bracket can be wider
than --bisect-tol; the `width` column says so, and a wide bracket needs a
rerun under the new criterion before it is reported. Non-monotone points
(stable above the first failure) are ignored for the bracket and flagged.

REPORTS; DOES NOT DECIDE. Which criterion the thesis uses is O-22.
"""
import argparse
import glob
import json
import os
import sys

CRITERIA = ("three", "loss_floor")
TOL = 1e-6


def load_records(path: str) -> list[dict]:
    """sweep.jsonl -> finite records, one per SRF (the last one wins)."""
    by_srf = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("finite", True) and "admissible" in r:
                by_srf[round(float(r["srf"]), 6)] = r
    return [by_srf[k] for k in sorted(by_srf)]


def conditions(rec: dict, ref: dict, s: dict) -> dict:
    """The three conditions of ssr_sweep._has_failed, separately."""
    if s.get("admissible_mode", "drop") == "floor":
        adm = rec["admissible"] < s["admissible_floor"]
    else:
        adm = (ref["admissible"] - rec["admissible"]) > s["admissible_drop"]
    return {"loss": rec["total"] > s["plateau_factor"] * ref["total"],
            "disp": rec["disp"] > s["disp_factor"] * ref["disp"],
            "admissible": bool(adm)}


def failed(c: dict, criterion: str) -> bool:
    if criterion == "three":
        return c["loss"] and c["disp"] and c["admissible"]
    if criterion == "loss_floor":
        return c["loss"] and c["admissible"]
    raise KeyError(criterion)


def bracket(records: list[dict], settings: dict, criterion: str) -> dict:
    srf0 = float(settings.get("srf_start", 1.0))
    ref = next((r for r in records if abs(r["srf"] - srf0) < TOL), None)
    if ref is None:
        return {"status": "no_reference_record"}
    flags = [(r["srf"], failed(conditions(r, ref, settings), criterion), r)
             for r in records]
    fails = [s for s, f, _ in flags if f]
    if not fails:
        return {"status": "no_failure", "fos": None,
                "max_srf": max(s for s, _, _ in flags)}
    f_min = min(fails)
    stable = [s for s, f, _ in flags if not f and s < f_min]
    lo = max(stable) if stable else None
    rec_f = next(r for s, _, r in flags if s == f_min)
    out = {"status": "fos", "bracket": [lo, f_min],
           "fos": None if lo is None else 0.5 * (lo + f_min),
           "width": None if lo is None else f_min - lo,
           "non_monotone": [s for s, f, _ in flags if not f and s > f_min],
           "fired_at_failure": conditions(rec_f, ref, settings)}
    return out


def settings_of(result: dict) -> dict:
    fp = result.get("fingerprint") or {}
    s = {k: result.get(k) for k in ("plateau_factor", "disp_factor",
                                     "admissible_mode", "admissible_floor",
                                     "admissible_drop")}
    s["srf_start"] = fp.get("srf_start", 1.0)
    s["bisect_tol"] = 0.01
    return s


def analyse(run_dir: str) -> dict | None:
    rj = os.path.join(run_dir, "result.json")
    sj = os.path.join(run_dir, "sweep.jsonl")
    if not (os.path.exists(rj) and os.path.exists(sj)):
        return None
    with open(rj, encoding="utf-8") as f:
        result = json.load(f)
    if result.get("smoke"):
        return None
    s = settings_of(result)
    recs = load_records(sj)
    out = {"run": os.path.basename(os.path.normpath(run_dir)),
           "criterion": result.get("criterion"),
           "w_yield": result.get("w_yield"), "n_pde": result.get("n_pde"),
           "arm": (result.get("arm") or {}).get("label"),
           "cold_start": result.get("cold_start"),
           "result_json": {"fos": result.get("fos"),
                           "bracket": result.get("bracket"),
                           "status": result.get("status")},
           "n_records": len(recs)}
    for c in CRITERIA:
        out[c] = bracket(recs, s, c)
    rb, tb = result.get("bracket"), out["three"].get("bracket")
    out["three_reproduces_result_json"] = bool(
        result.get("status") == "fos" and rb and tb and None not in tb
        and abs(rb[0] - tb[0]) < TOL and abs(rb[1] - tb[1]) < TOL) or (
        result.get("status") == "no_failure"
        and out["three"]["status"] == "no_failure")
    w = out["loss_floor"].get("width")
    out["loss_floor_needs_rerun"] = bool(
        out["loss_floor"]["status"] == "fos"
        and (w is None or w > 2 * s["bisect_tol"] + TOL))
    return out


def _fmt(b: dict) -> str:
    if b.get("status") != "fos":
        return b.get("status", "?")
    lo, hi = b["bracket"]
    if lo is None:
        return f"< {hi:.3f}"
    return f"{b['fos']:.3f} ({lo:.3f}–{hi:.3f})"


def to_markdown(rows: list[dict]) -> str:
    L = ["# Failure criterion re-read (scripts/recompute_criterion.py)", "",
         "O-22: `three` = loss AND displacement AND admissible (the sweeps as "
         "run); `loss_floor` = loss AND admissible. Brackets come from the "
         "SRF points each sweep evaluated. **rerun?** = the loss_floor "
         "bracket is wider than 2 × bisect-tol and must be resolved by a "
         "sweep under the new criterion before it is reported.", "",
         "| run | arm | w_yield | N_PDE | FOS three | = result.json | "
         "FOS loss_floor | disp fired at loss_floor failure | rerun? |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lf = r["loss_floor"]
        disp = (lf.get("fired_at_failure") or {}).get("disp")
        L.append(f"| {r['run']} | {r['arm'] or '—'} | {r['w_yield']} | "
                 f"{r['n_pde']} | {_fmt(r['three'])} | "
                 f"{'yes' if r['three_reproduces_result_json'] else '**NO**'} | "
                 f"{_fmt(lf)} | {'—' if disp is None else disp} | "
                 f"{'yes' if r['loss_floor_needs_rerun'] else 'no'} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs/ssr*")
    ap.add_argument("--out", default="docs/results")
    a = ap.parse_args(argv)
    rows = [r for d in sorted(glob.glob(a.runs)) if os.path.isdir(d)
            for r in [analyse(d)] if r is not None]
    if not rows:
        print(f"no run with result.json + sweep.jsonl under {a.runs}",
              file=sys.stderr)
        return 1
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "criterion_o22.json"), "w",
              encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    with open(os.path.join(a.out, "criterion_o22.md"), "w",
              encoding="utf-8") as f:
        f.write(to_markdown(rows))
    bad = [r["run"] for r in rows if not r["three_reproduces_result_json"]]
    print(f"{len(rows)} runs -> {a.out}/criterion_o22.json, criterion_o22.md")
    if bad:
        print(f"  WARNING: the three-condition re-read does NOT reproduce "
              f"result.json for {bad} -- do not trust the loss_floor column "
              f"for these until the difference is understood.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
