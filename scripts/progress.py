r"""Progress of one or more sweep runs, safe to call before anything finished.

    python scripts/progress.py runs/probe_ghb_ref runs/probe_ghb_raw
    python scripts/progress.py "runs/ssr*"
"""
import glob
import json
import os
import sys


def summarise(run_dir) -> dict:
    p = os.path.join(run_dir, "sweep.jsonl")
    if not os.path.exists(p):
        return {"run": run_dir, "done": 0, "records": [],
                "finished": os.path.exists(os.path.join(run_dir, "result.json"))}
    with open(p) as f:
        recs = [json.loads(l) for l in f if l.strip()]
    res_p = os.path.join(run_dir, "result.json")
    res = json.load(open(res_p)) if os.path.exists(res_p) else None
    return {"run": run_dir, "done": len(recs),
            "records": sorted(recs, key=lambda r: r["srf"]),
            "finished": res is not None,
            "status": res.get("status") if res else "running",
            "fos": res.get("fos") if res else None}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    dirs = sorted({d for pat in argv for d in glob.glob(pat) if os.path.isdir(d)})
    if not dirs:
        print("no run directories match", argv)
        return 1
    for d in dirs:
        s = summarise(d)
        head = f"{d}: {s['done']} SRF done"
        if s["finished"]:
            head += f", FINISHED ({s.get('status')}, FOS {s.get('fos')})"
        elif s["done"] == 0:
            head += " - first SRF still training"
        print(head)
        for r in s["records"]:
            print(f"   SRF {r['srf']:.3f}  adm {r.get('admissible', float('nan')):.3f}"
                  f"  L {r['total']:.3e}  |u| {r['disp']:.3e}  {r['sec']/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
