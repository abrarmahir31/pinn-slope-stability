"""Last-record summary for every runs/* directory with a log."""
import glob
import json
import os

for d in sorted(glob.glob("runs/*")):
    p = os.path.join(d, "log.jsonl")
    if not os.path.exists(p):
        continue
    rows = [json.loads(l) for l in open(p)]
    r = rows[-1]
    cfg = json.load(open(os.path.join(d, "config.json")))
    sp = r.get("spread")
    print(f"{os.path.basename(d):16s} "
          f"warm={cfg.get('warmup'):>4} sched={cfg.get('lr_schedule','n/a'):>6} "
          f"ep={r['step']:>5}  spread={sp if sp else float('nan'):7.2f}  "
          f"clipped={len(r.get('clipped') or [])} frozen={len(r.get('frozen') or [])}")
    L, L0 = r["L"], r["L0"]
    print("     " + "  ".join(f"{t}={L[t]/L0[t]:.3g}"
                              for t in ("bc_mech", "pde_mech", "pde_richards",
                                        "bc", "ic_head", "ic_disp")))