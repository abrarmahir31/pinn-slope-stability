r"""Fig. 3 -- computational domain, strata, boundary segments and a
representative collocation cloud. Everything drawn comes from the live code:
`geometry.material_tag` / `inside_domain` for the strata,
`boundaries.sample_boundaries` for the segments, `boundaries.mechanical_bc`
for the mechanical boundary kinds, and `sampling.sample_interior` for the
cloud. Nothing is hand-typed, so the figure cannot disagree with the model.

    PYTHONPATH=. python scripts/make_fig3.py --out docs/figs/fig3.png

Panel (a): strata and boundary segments, labelled with their mechanical kind.
Panel (b): interior collocation points (stratified draw, as trained) and
boundary points. The caption must say the cloud is STRATIFIED (D-Samp.1):
Mk_d is ~10% of the area but 35% of the points.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.nondim import SCALES
from src.sampling import sample_interior
from src.step21_geometry import boundaries as bnd
from src.step21_geometry import geometry as g

TAG_COLOURS = {"Mk": "#c9b27c", "Mk_d": "#d9774b", "Tm": "#8fa9c7"}
SEG_STYLE = {"natural_ground": "C2", "bench": "C8", "cut_face": "C3",
             "pit_floor": "C4", "base": "k", "far_field_f1": "C0"}


def strata_grid(nx=400, nz=260):
    xs = np.linspace(0.0, 270.0, nx)
    zs = np.linspace(g.Z_BASE, 362.0, nz)
    X, Z = np.meshgrid(xs, zs)
    inside = np.asarray(g.inside_domain(X, Z), bool)
    tag = np.asarray(g.material_tag(X, Z)).astype(str)
    return X, Z, inside, tag


def segment_kinds() -> dict:
    """Mechanical kind per segment from the live `mechanical_bc`."""
    out = {}
    for seg in bnd.sample_boundaries(10, seed=0):
        k = bnd.mechanical_bc(seg)
        out[seg] = "traction-free" if k is None else k[0]
    return out


def area_and_sample_shares(coll) -> dict:
    w = coll.w.detach().numpy().ravel()
    tags = sorted(set(coll.tag.tolist()))
    return {t: {"area": float(w[coll.tag == t].sum() / w.sum()),
                "points": float(np.mean(coll.tag == t))} for t in tags}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-interior", type=int, default=3000)
    ap.add_argument("--n-boundary", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20260808)
    ap.add_argument("--out", default="docs/figs/fig3.png")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args(argv)

    X, Z, inside, tag = strata_grid()
    kinds = segment_kinds()
    segs = bnd.sample_boundaries(a.n_boundary, seed=a.seed)
    coll = sample_interior(a.n_interior, a.seed)
    xi = coll.x.detach().numpy().ravel() * SCALES.L_ref
    zi = coll.z.detach().numpy().ravel() * SCALES.L_ref
    shares = area_and_sample_shares(coll)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 8.5), sharex=True)
    for t, c in TAG_COLOURS.items():
        m = inside & (tag == t)
        a1.scatter(X[m], Z[m], s=1.2, c=c, marker="s", linewidths=0,
                   label=f"{t} ({100*shares[t]['area']:.1f}% area)")
    for seg, pts in segs.items():
        o = np.argsort(pts[:, 0] if seg not in ("pit_floor", "far_field_f1")
                       else pts[:, 1])
        a1.plot(pts[o, 0], pts[o, 1], "-", lw=2.2, color=SEG_STYLE.get(seg, "k"),
                label=f"{seg}: {kinds[seg]}")
    a1.set(ylabel="z (m a.s.l.)", title="(a) strata and boundary segments",
           aspect="equal")
    a1.legend(fontsize=7, ncol=2, loc="lower right")

    for t, c in TAG_COLOURS.items():
        m = coll.tag == t
        a2.scatter(xi[m], zi[m], s=2, c=c, linewidths=0,
                   label=f"{t}: {100*shares[t]['points']:.0f}% of points")
    for seg, pts in segs.items():
        a2.scatter(pts[:, 0], pts[:, 1], s=3, c=SEG_STYLE.get(seg, "k"),
                   linewidths=0)
    a2.set(xlabel="x (m from toe)", ylabel="z (m a.s.l.)", aspect="equal",
           title=f"(b) collocation, stratified draw (N = {a.n_interior} "
                 f"interior shown)")
    a2.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi)
    plt.close(fig)
    meta = {"segment_kinds": kinds, "shares": shares,
            "n_interior": a.n_interior, "seed": a.seed,
            "cut_angle_deg": g.CUT_ANGLE, "x_crest_m": g.X_CREST,
            "z_base_m": g.Z_BASE}
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {a.out}")
    for t, v in shares.items():
        print(f"  {t:5s} area {100*v['area']:5.1f}%   points {100*v['points']:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
