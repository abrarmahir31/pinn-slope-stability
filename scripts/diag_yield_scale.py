"""Why does L_yield(baseline, SRF 1) double with N_PDE?

Read-only diagnostic. Builds the collocation set exactly as ssr_sweep.py does
(parse -> _train_args -> train.setup) at N_PDE = 5k, 10k, 20k on the CPU, and
prints point counts, weights and the yield term per stratum. Writes nothing.

Run from the repo root:  python scripts\\diag_yield_scale.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ssr_sweep as s  # noqa: E402
import train  # noqa: E402
from src.loss import L_yield  # noqa: E402

BASE = r"runs\ansatz_epsuv1e-2\ckpt_final.pt"

ckpt = torch.load(BASE, map_location="cpu", weights_only=False)
cfg = ckpt.get("cfg") or {}

for n in (5000, 10000, 20000):
    a = s.parse([
        "--criterion", "GHB", "--baseline", BASE, "--out", "runs/_diag_tmp",
        "--w-yield", "1", "--yield-norm", "raw",
        "--sig3-lo", "0", "--sig3-hi", "179000",
        "--admissible-mode", "floor", "--admissible-floor", "0.5",
        "--n-pde", str(n), "--device", "cpu",
    ])
    net, loss_fn, coll = train.setup(s._train_args(cfg, a))
    mats = loss_fn["mats"]
    yp = s._yield_params(a, mats, 1.0)
    y, parts = L_yield(net, coll, mats, yp, criterion="GHB", per_tag=True)

    w = coll.w.detach().cpu().reshape(-1)
    tags = coll.tag
    print(f"\nN_PDE={n}  points={len(w)}  sum(w)={float(w.sum()):.4g}  "
          f"L_yield={float(y):.4e}")
    for t in sorted(set(tags.tolist())):
        m = torch.as_tensor(tags == t)
        print(f"  {t:5s} n={int(m.sum()):6d}  sum(w)={float(w[m].sum()):.4g}  "
              f"mean w={float(w[m].mean()):.3g}  "
              f"pde_yield={float(parts['pde_yield_' + t]):.4e}  "
              f"adm={float(parts['admissible_' + t]):.3f}")