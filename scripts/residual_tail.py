r"""Where does L_PDE_richards actually come from? Top-1% concentration by sampler.

`scripts/seed_variance_decoupled.py` gave a result that contradicts the Day 26
plan. Concentrating collocation in the live band (`src/sampling_front.py`) made
`w^[pde_richards]` spread WORSE, not better: 469.9x -> 15989.5x on the
draw-only arm at N = 1200. The Day 26 note assumed the 1264x spread came from
the live band being under-sampled. This script tests the competing explanation.

    Day 26 hypothesis  the estimator is heavy-tailed because ~87% of points sit
                       in the inert zone and the gradient is carried by the few
                       that land near the live band. Fix: sample the live band.

    Competing          the estimator is heavy-tailed because van Genuchten K_r
                       with n = 1.2 is near-singular in psi close to zero. The
                       tail IS the live band. Sampling it harder loads more
                       mass onto the tail. Fix: not a sampler.

The two make opposite predictions for tail concentration -- the fraction of
sum(w R^2) contributed by the top 1% of points. Under the first, the front
sampler should SPREAD the loss over more points and lower concentration. Under
the second, it should RAISE it.

Reported on both an untrained network (which is what the w^ seed is measured
on) and a checkpoint, because the two need not agree.

RESULT (Day 27), and it is not a clean win for either. Read the two nets
separately; the framing above is too coarse for what came out.

  UNTRAINED, the net the w^ seed is actually measured on. Both samplers are
  catastrophic and the sampler barely matters: 2-3 points out of 4000 carry 90%
  of the loss, top-1% share 99.3-99.95% under uniform and 98.8-99.1% under
  front in two of three seeds. Every top contributor sits at psi = -0.06 to
  -0.47 m where psi_0 should be -30 to -77 m. That is the van Genuchten
  singularity, reached because `NearPhysical` is unbounded above and at
  eps_psi = 0.3 puts 15-35% of points at psi >= 0. Neither hypothesis as
  stated: the tail is not the live band, it is a region the SOLUTION never
  occupies.

  CHECKPOINT. The front sampler simultaneously RAISES top-1% share (31.6% ->
  65.7%) and LOWERS median |R| by two orders (1.24e-03 -> 1.12e-05). Both, and
  they are the same fact. Uniform sampling's bulk sits in the inert zone, where
  the residual is the storage term and is LARGER than in the live band -- so
  most of uniform's loss is psi drifting in time where nothing should happen,
  which is the 150 m IC violation showing up in the PDE term. The front sampler
  removes that mass. What is left is concentrated because it is genuinely
  concentrated.

  CONSEQUENCE for adaptive refinement: do NOT refine on |R|. It is larger in
  the dead zone. The criterion has to be the flux fraction
  max(|diffusion|, |gravity|) / |storage|.

    PYTHONPATH=. python scripts/residual_tail.py
    PYTHONPATH=. python scripts/residual_tail.py --ckpt runs/prod02/ckpt_002000.pt
"""
import argparse
import dataclasses
import json

import numpy as np
import torch

from src.config import BOUNDS, bounds_from_geometry, full, tiny
from src.loss import psi_of
from src.materials import load_materials
from src.model import PINN, NearPhysical
from src.nondim import SCALES
from src.residuals import richards_residual
from src.sampling import sample_interior
from src.sampling_front import sample_interior_front


def per_point(net, coll, mats, s=SCALES):
    """w * R^2 per collocation point, and |R| per point."""
    contrib = np.zeros(len(coll))
    absR = np.zeros(len(coll))
    w = coll.w.detach().reshape(-1).numpy()

    def fields(x, z, t):
        return psi_of(net(x, z, t))

    for tag in sorted(set(coll.tag.tolist())):
        m = coll.tag == tag
        mt = torch.as_tensor(m)
        x = coll.x[mt].reshape(-1, 1).clone().requires_grad_(True)
        z = coll.z[mt].reshape(-1, 1).clone().requires_grad_(True)
        t = coll.t[mt].reshape(-1, 1).clone().requires_grad_(True)
        R = richards_residual(fields, x, z, t, mats[tag], s=s)
        r = R.detach().reshape(-1).numpy()
        absR[m] = np.abs(r)
        contrib[m] = w[m] * r ** 2
    return contrib, absR


def summarise(contrib, absR, label):
    n = len(contrib)
    tot = contrib.sum()
    order = np.argsort(contrib)[::-1]
    top1 = contrib[order[:max(1, n // 100)]].sum() / tot
    top01 = contrib[order[:max(1, n // 1000)]].sum() / tot
    # Points needed to reach 90% of the loss.
    c = np.cumsum(contrib[order]) / tot
    n90 = int(np.searchsorted(c, 0.90) + 1)
    row = {"label": label, "n": n, "mean_loss": float(tot / n),
           "top1pct_share": float(top1), "top0.1pct_share": float(top01),
           "n_for_90pct": n90, "frac_for_90pct": float(n90 / n),
           "absR_p50": float(np.median(absR)),
           "absR_p99": float(np.quantile(absR, 0.99)),
           "absR_max": float(absR.max())}
    print(f"  {label:<28}{tot / n:>12.3e}{100 * top1:>10.2f}%"
          f"{100 * top01:>10.2f}%{n90:>9d}{100 * n90 / n:>8.2f}%"
          f"{np.median(absR):>12.3e}{absR.max():>12.3e}")
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[20250812, 7, 1234])
    ap.add_argument("--eps-psi", type=float, default=0.3)
    ap.add_argument("--json", metavar="PATH")
    a = ap.parse_args(argv)

    mats = load_materials()
    out = {"n": a.n, "seeds": a.seeds, "ckpt": a.ckpt, "rows": []}

    nets = {}
    cfg = dataclasses.replace(tiny(), n_layers=8, n_neurons=64, seed=a.seeds[0])
    nets["untrained"] = NearPhysical(PINN(cfg, BOUNDS), eps_psi=a.eps_psi)
    if a.ckpt:
        B = bounds_from_geometry(t_max=30.0, s=SCALES)
        c2 = full()
        c2.device, c2.dtype = "cpu", "float64"
        d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        nt = NearPhysical(PINN(c2, B), eps_psi=a.eps_psi)
        nt.load_state_dict(d["net"])
        nt.eval()
        nets[f"ckpt@{d.get('step')}"] = nt

    hdr = (f"  {'sampler / net / seed':<28}{'mean w R^2':>12}{'top 1%':>11}"
           f"{'top 0.1%':>10}{'n for 90%':>9}{'':>8}{'|R| p50':>12}"
           f"{'|R| max':>12}")
    for net_name, net in nets.items():
        print(f"\n{net_name}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for sname, mk in (("uniform", sample_interior),
                          ("front", sample_interior_front)):
            for sd in a.seeds:
                coll = mk(a.n, sd)
                contrib, absR = per_point(net, coll, mats)
                out["rows"].append(
                    summarise(contrib, absR, f"{sname}  seed {sd}")
                    | {"net": net_name, "sampler": sname, "seed": sd})
            print()

    print("Read the two nets separately -- they answer different questions.\n"
          "  UNTRAINED is the net w^ is seeded from. If 'n for 90%' is single "
          "digits under\n  BOTH samplers, the tail is the ansatz reaching "
          "psi = 0, not the collocation:\n  cross-check the psi >= 0 fraction "
          "before blaming the sampler.\n"
          "  CKPT: 'front' raising top-1% while dropping |R| p50 by orders is "
          "the expected\n  signature, not a contradiction. Uniform's bulk is "
          "inert-zone storage residual,\n  which is LARGER than the live "
          "band's -- so refining on |R| refines into the\n  dead region. Use "
          "max(flux)/storage as the adaptive criterion instead.")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())