r"""Report the quantities D-5.6 and D-5.7 turn on, for a trained baseline, next
to the same quantities for baseline-v1. REPORTS NUMBERS; NO PASS/FAIL
THRESHOLDS except two facts that are not judgement calls (the ansatz is
recorded, and it is not the rejected `unbounded`).

    PYTHONPATH=. python scripts/check_baseline.py runs/prod_baseline_v2
    PYTHONPATH=. python scripts/check_baseline.py runs/prod_baseline_v2 --compare runs/ansatz/exp_seed7

  base drift     max |u|, |v| on the nominally fixed base z = Z_BASE, m, at
                 t = 0 and t_max (D-5.6: 1.3-1.4 mm on baseline-v1)
  psi            psi_max_m, psi_sat_frac from the last log record; cap at
                 20250812 pinned psi_max at the cap (D-A.1, D-5.7)
  loss ratios    L / L0 per term at the last record; pde_richards ~0.4 was the
                 D-A.1 cap failure signature, ~1e-5 healthy
  training       steps, sec_per_epoch, ansatz, seed, L-BFGS length

Whether the production drift is "small enough" is open (DECISIONS.md open
issues register); this script supplies the number.
"""
import argparse
import dataclasses
import json
import math
import os
import sys

import numpy as np
import torch

from src.config import BOUNDS, tiny
from src.model import PINN, NearPhysical
from src.nondim import SCALES
from src.step21_geometry import geometry as g


def last_record(log_path):
    last = None
    with open(log_path) as f:
        for line in f:
            if line.strip():
                last = json.loads(line)
    if last is None:
        raise ValueError(f"{log_path} is empty")
    return last


def loss_ratios(rec) -> dict:
    return {k: rec["L"][k] / rec["L0"][k] for k in rec["L"]
            if rec["L0"].get(k)}


def base_drift(ckpt_path, t_days, n=200) -> dict:
    d = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = d["cfg"]
    pc = dataclasses.replace(tiny(), n_layers=cfg["layers"],
                             n_neurons=cfg["width"], device="cpu")
    uv = {} if cfg.get("eps_uv") is None else {"eps_uv": float(cfg["eps_uv"])}
    net = NearPhysical(PINN(pc, BOUNDS), eps_psi=cfg["eps_psi"],
                       mode=cfg["ansatz"], cap_k=cfg.get("cap_k", 100.0), **uv)
    net.load_state_dict(d["net"])
    net.eval()
    x_hi = float(g.x_f1(g.Z_BASE)) - 0.5
    xs = np.linspace(0.5, x_hi, n)
    out = {}
    with torch.no_grad():
        for t in t_days:
            X = torch.tensor(xs / SCALES.L_ref).reshape(-1, 1)
            Z = torch.full_like(X, g.Z_BASE / SCALES.L_ref)
            T = torch.full_like(X, t)
            o = net(X, Z, T).numpy() * SCALES.U_ref
            out[f"t{t:g}"] = {"u_max_m": float(np.abs(o[:, 1]).max()),
                              "v_max_m": float(np.abs(o[:, 2]).max())}
    return {"cfg": cfg, "step": d.get("step"), "drift": out}


def report(run_dir, t_max):
    ck = os.path.join(run_dir, "ckpt_final.pt")
    lg = os.path.join(run_dir, "log.jsonl")
    bd = base_drift(ck, (0.0, t_max))
    rec = last_record(lg) if os.path.exists(lg) else None
    return {"run": run_dir, "step": bd["step"], "drift": bd["drift"],
            "loss_raw": dict(rec["L"]) if rec else None,
            "ansatz": bd["cfg"].get("ansatz"), "seed": bd["cfg"].get("seed"),
            "epochs": bd["cfg"].get("epochs"),
            "lbfgs_epochs": bd["cfg"].get("lbfgs_epochs"),
            "psi_max_m": rec.get("psi_max_m") if rec else None,
            "psi_sat_frac": rec.get("psi_sat_frac") if rec else None,
            "sec_per_epoch": rec.get("sec_per_epoch") if rec else None,
            "loss_ratios": loss_ratios(rec) if rec else None}


def facts(r) -> list:
    """The only non-judgement checks."""
    bad = []
    if r["ansatz"] is None:
        bad.append("checkpoint cfg has no ansatz (D-A.3)")
    elif r["ansatz"] == "unbounded":
        bad.append("ansatz is 'unbounded', rejected by D-A.1")
    for k, v in (r["loss_ratios"] or {}).items():
        if not math.isfinite(v):
            bad.append(f"non-finite loss ratio for {k}")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--compare", default="runs/ansatz/exp_seed7")
    ap.add_argument("--t-max", type=float, default=30.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    rows = [report(a.run, a.t_max)]
    if a.compare and os.path.exists(os.path.join(a.compare, "ckpt_final.pt")):
        rows.append(report(a.compare, a.t_max))

    print(f"{'':28}" + "".join(f"{os.path.basename(r['run']):>22}" for r in rows))
    def line(name, f):
        print(f"{name:28}" + "".join(f"{f(r):>22}" for r in rows))
    line("step", lambda r: str(r["step"]))
    line("ansatz / seed", lambda r: f"{r['ansatz']} / {r['seed']}")
    line("L-BFGS epochs", lambda r: str(r["lbfgs_epochs"]))
    for t in ("t0", f"t{a.t_max:g}"):
        line(f"base max|u| {t} (mm)", lambda r, t=t: f"{1e3*r['drift'][t]['u_max_m']:.3f}")
        line(f"base max|v| {t} (mm)", lambda r, t=t: f"{1e3*r['drift'][t]['v_max_m']:.3f}")
    line("psi_max (m)", lambda r: "-" if r["psi_max_m"] is None else f"{r['psi_max_m']:.3g}")
    line("psi_sat_frac", lambda r: "-" if r["psi_sat_frac"] is None else f"{r['psi_sat_frac']:.3g}")
    line("sec/epoch", lambda r: "-" if r["sec_per_epoch"] is None else f"{r['sec_per_epoch']:.3f}")
    for k in (rows[0]["loss_ratios"] or {}):
        line(f"L/L0 {k}", lambda r, k=k: "-" if not r["loss_ratios"] else f"{r['loss_ratios'].get(k, float('nan')):.2e}")
    print("  (L/L0 is normalised by EACH run's own first-step loss; the "
          "ratios are not comparable between runs. Absolute values are:)")
    for k in (rows[0]["loss_raw"] or {}):
        line(f"L {k}", lambda r, k=k: "-" if not r["loss_raw"] else f"{r['loss_raw'].get(k, float('nan')):.2e}")
    bad = facts(rows[0])
    print("\nFACTS: " + ("none failing" if not bad else "; ".join(bad)))
    print("Judgement calls (drift small enough? psi healthy?) are open issues; "
          "record them in DECISIONS.md.")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
