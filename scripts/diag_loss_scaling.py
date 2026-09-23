"""Does any loss term scale with N_PDE the way L_yield does?

Read-only, CPU. Builds the collocation set exactly as ssr_sweep.py does at
N_PDE = 5k, 10k, 20k (same seeds, so the same network each time) and evaluates
every loss term plus L_yield. A term that is a proper mean over points stays
near 1x from 5k to 20k; a term with the (n,1) x (n,) broadcasting bug grows
about 4x. Writes nothing.

Run from the repo root:  python scripts\\diag_loss_scaling.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import ssr_sweep as s  # noqa: E402
import train  # noqa: E402
from src.loss import L_yield  # noqa: E402

BASE = r"runs\ansatz_epsuv1e-2\ckpt_final.pt"
ckpt = torch.load(BASE, map_location="cpu", weights_only=False)
cfg = ckpt.get("cfg") or {}

results = {}
for n in (5000, 10000, 20000):
    a = s.parse([
        "--criterion", "GHB", "--baseline", BASE, "--out", "runs/_diag_tmp",
        "--w-yield", "1", "--yield-norm", "raw",
        "--sig3-lo", "0", "--sig3-hi", "179000",
        "--admissible-mode", "floor", "--admissible-floor", "0.5",
        "--n-pde", str(n), "--device", "cpu",
    ])
    net, loss_fn, coll = train.setup(s._train_args(cfg, a))
    kc = s.arms.kc_config(s._kc_name(cfg, {}))
    terms = s._losses(net, loss_fn, coll, kc)
    row = {k: float(v.detach()) for k, v in terms.items()}
    yp = s._yield_params(a, loss_fn["mats"], 1.0)
    row["yield"] = float(L_yield(net, coll, loss_fn["mats"], yp,
                                 criterion="GHB")[0].detach())
    row["yield_mean"] = float(L_yield(net, coll, loss_fn["mats"], yp, criterion="GHB",
                                      reduction="mean")[0].detach())
    results[n] = row
    print(f"N_PDE={n} done ({len(row)} terms)", flush=True)

print(f"\n{'term':22s} {'5k':>12s} {'10k':>12s} {'20k':>12s} {'20k/5k':>8s}  verdict")
for k in results[5000]:
    v5, v10, v20 = (results[n][k] for n in (5000, 10000, 20000))
    ratio = v20 / v5 if v5 else float("nan")
    verdict = "SCALES WITH N" if ratio > 2.5 else "ok (mean)"
    print(f"{k:22s} {v5:12.4e} {v10:12.4e} {v20:12.4e} {ratio:8.2f}  {verdict}")