r"""Task 2 input: the baseline sigma_3 distribution per stratum, for choosing
the GHB fit range `--sig3-lo/--sig3-hi`. PRINTS DATA; DOES NOT CHOOSE.

Why a script and not docs/figs/fig6.json: that JSON stores sigma_1 max and FS
statistics only. It holds no sigma_3 values, so the range cannot be read from
it.

sigma_3 here is COMPRESSION-POSITIVE, Pa, on the TOTAL stress sigma_0 +
D:eps with the Bishop increment -- the same quantity `loss.L_yield` checks
and `make_fig6.py` evaluates. Negative values are tension.

    python scripts/sig3_range.py runs/ansatz/exp_seed7/ckpt_final.pt
    python scripts/sig3_range.py CKPT --tags Mk_d --max-depth-m 30 --t 30

What to report next to every FOS_GHB: the tags and depth band the range was
read from, the percentiles used as lo/hi, and this script's JSON.
"""
import argparse
import json
import os

import numpy as np
import torch

from scripts.make_fig6 import distance_to_free_surface, net_from_ckpt
from src import strength as st
from src.config import bounds_from_geometry, full
from src.materials import load_materials
from src.mechanics import total_stress_star
from src.nondim import SCALES
from src.sampling import sample_interior
from src.sigma0 import attach_sigma0

PCTS = (1, 5, 25, 50, 75, 95, 99)
PHASE1_DESIGN_SIG3MAX = 8.5e5


def weighted_percentile(v, w, q):
    """Percentile q (0-100) of v under importance weights w (inverted CDF)."""
    v, w = np.asarray(v, float), np.asarray(w, float)
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return float(v[o][min(np.searchsorted(cw, q / 100.0 * cw[-1]), v.size - 1)])


def summarise(s3_pa, w=None, reference=PHASE1_DESIGN_SIG3MAX) -> dict:
    """Area-weighted percentiles (Pa), tension fraction, and where `reference`
    falls. Pooling strata without `w` weights them by the sampler's shares
    (Mk_d 35% of the draw, ~10% of the area) -- pass coll.w."""
    s3 = np.asarray(s3_pa, float)
    if s3.size == 0:
        raise ValueError("no points in the selected zone")
    w = np.ones_like(s3) if w is None else np.asarray(w, float)
    return {"n": int(s3.size),
            "min": float(s3.min()), "max": float(s3.max()),
            "percentiles": {str(p): weighted_percentile(s3, w, p) for p in PCTS},
            "frac_tension": float(w[s3 < 0.0].sum() / w.sum()),
            "frac_below_reference": float(w[s3 <= reference].sum() / w.sum()),
            "reference_pa": float(reference)}


def select(tag, dist_m, tags=None, max_depth_m=None):
    keep = np.ones(len(tag), bool)
    if tags:
        keep &= np.isin(tag, list(tags))
    if max_depth_m is not None:
        keep &= np.asarray(dist_m) <= max_depth_m
    return keep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--t", type=float, default=30.0)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--sampling-seed", type=int, default=20260808)
    ap.add_argument("--tags", default=None, help="comma list, e.g. Mk_d")
    ap.add_argument("--max-depth-m", type=float, default=None,
                    help="keep points within this distance of the free surface")
    ap.add_argument("--no-bishop", action="store_true")
    ap.add_argument("--json", default="docs/results/sig3_range.json")
    a = ap.parse_args(argv)

    cfg = full()
    cfg.device, cfg.dtype = "cpu", "float64"
    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = net_from_ckpt(d, cfg, bounds_from_geometry(t_max=30.0, s=SCALES))
    net.eval()
    coll = attach_sigma0(sample_interior(a.n, a.sampling_seed))
    mats = load_materials()

    S3, TAG, X, Z, W = [], [], [], [], []
    for tag in sorted(set(coll.tag.tolist())):
        m = torch.as_tensor(coll.tag == tag)
        x = coll.x[m].reshape(-1, 1).clone().requires_grad_(True)
        z = coll.z[m].reshape(-1, 1).clone().requires_grad_(True)
        t = torch.full_like(x, a.t).requires_grad_(True)
        s0 = coll.sigma0[m]
        (sxx, szz, sxz), _, _ = total_stress_star(
            net, x, z, t, mats[tag],
            sigma0_star=(s0[:, 0:1], s0[:, 1:2], s0[:, 2:3]),
            bishop=not a.no_bishop)
        _, s3 = st.principal_stresses_from_cartesian(
            sxx.detach(), szz.detach(), sxz.detach())
        S3.append(s3.numpy().ravel() * SCALES.sig_ref)
        TAG += [tag] * int(m.sum())
        W.append(coll.w[m].detach().numpy().ravel())
        X.append(x.detach().numpy().ravel() * SCALES.L_ref)
        Z.append(z.detach().numpy().ravel() * SCALES.L_ref)
    S3, TAG, W = np.concatenate(S3), np.asarray(TAG), np.concatenate(W)
    DIST, _ = distance_to_free_surface(np.concatenate(X), np.concatenate(Z))

    tags = a.tags.split(",") if a.tags else None
    out = {"ckpt": a.ckpt, "t_days": a.t, "n": a.n, "bishop": not a.no_bishop,
           "zone": {"tags": tags, "max_depth_m": a.max_depth_m}, "per_tag": {}}
    print(f"sigma_3 (compression +, kPa) at t = {a.t:g} d"
          f"{'   [BISHOP OFF]' if a.no_bishop else ''}")
    print(f"{'zone':>14}{'n':>6}" + "".join(f"{'p'+str(p):>9}" for p in PCTS)
          + f"{'tension':>9}{'<=850kPa':>10}")

    def row(name, keep):
        r = summarise(S3[keep], W[keep])
        print(f"{name:>14}{r['n']:>6}" + "".join(
            f"{r['percentiles'][str(p)]/1e3:>9.1f}" for p in PCTS)
            + f"{100*r['frac_tension']:>8.1f}%{100*r['frac_below_reference']:>9.1f}%")
        return r

    for tag in sorted(set(TAG.tolist())):
        keep = select(TAG, DIST, [tag], a.max_depth_m)
        if keep.any():
            out["per_tag"][tag] = row(tag, keep)
    keep = select(TAG, DIST, tags, a.max_depth_m)
    out["selected_zone"] = row("SELECTED", keep)
    print("\nThis is the input to a decision, not the decision. Record which "
          "zone and percentiles set --sig3-lo/--sig3-hi.")
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    with open(a.json, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
