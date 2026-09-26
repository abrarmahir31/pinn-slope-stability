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

Style: `src/plot_style.py` -- 6.25 in wide, 12 pt Times, equal aspect, one
shared legend under the panels (strata: area and point shares; segments:
mechanical kind). Writes PNG (300 dpi) and PDF.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from src import plot_style as ps
from src.nondim import SCALES
from src.sampling import sample_interior
from src.step21_geometry import boundaries as bnd
from src.step21_geometry import geometry as g

TAG_COLOURS = ps.STRATA
TAG_FILL = ps.STRATA_FILL
SEG_STYLE = ps.SEGMENT
SEG_NAME = {"natural_ground": "natural ground", "bench": "bench",
            "cut_face": "cut face", "pit_floor": "pit floor", "base": "base",
            "far_field_f1": "far field (F1)"}
KIND_NAME = {"traction-free": "traction-free", "fixed": "fixed",
             "roller_x": r"roller, $u = 0$"}


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
    ap.add_argument("--out", default="docs/figs/fig3.png",
                    help="PNG path; a PDF is written next to it")
    a = ap.parse_args(argv)

    X, Z, inside, tag = strata_grid()
    kinds = segment_kinds()
    segs = bnd.sample_boundaries(a.n_boundary, seed=a.seed)
    coll = sample_interior(a.n_interior, a.seed)
    xi = coll.x.detach().numpy().ravel() * SCALES.L_ref
    zi = coll.z.detach().numpy().ravel() * SCALES.L_ref
    shares = area_and_sample_shares(coll)

    ps.apply()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(ps.WIDTH, 8.35), sharex=True,
                                 layout="constrained")
    for ax in (a1, a2):
        ps.field_axes(ax)
        ax.set_aspect("equal")

    # (a) strata as flat tints, boundary segments as heavy coloured lines.
    ps.draw_strata(a1, fill=True, lines=True, lw=0.8, color=ps.GREY)
    for seg, pts in segs.items():
        o = np.argsort(pts[:, 0] if seg not in ("pit_floor", "far_field_f1")
                       else pts[:, 1])
        a1.plot(pts[o, 0], pts[o, 1], "-", lw=2.6, color=SEG_STYLE.get(seg, "k"),
                solid_capstyle="butt", zorder=8)
    a1.set_ylabel("Elevation (m a.s.l.)")
    ps.panel_label(a1, "a")

    # (b) the stratified cloud, coloured by stratum, and the boundary draw.
    ps.ground_line(a2, lw=0.8, color=ps.GREY)
    for t, c in TAG_COLOURS.items():
        m = coll.tag == t
        a2.scatter(xi[m], zi[m], s=2.2, c=c, linewidths=0, rasterized=True,
                   zorder=3)
    for seg, pts in segs.items():
        a2.scatter(pts[:, 0], pts[:, 1], s=5, c=SEG_STYLE.get(seg, "k"),
                   linewidths=0, rasterized=True, zorder=4)
    a2.set(xlabel="Distance from toe, $x$ (m)", ylabel="Elevation (m a.s.l.)")
    ps.panel_label(a2, "b")
    for ax in (a1, a2):
        ax.set_xlim(0.0, 270.0)
        ax.set_ylim(g.Z_BASE - 2.0, 364.0)

    # One legend for both panels.
    h_strata = [Patch(fc=TAG_FILL[t], ec=TAG_COLOURS[t], lw=1.2,
                      label=f"{ps.STRATA_LABEL[t]}: {100*shares[t]['area']:.1f}% "
                            f"area, {100*shares[t]['points']:.0f}% points")
                for t in TAG_COLOURS if t in shares]
    h_seg = [Line2D([], [], color=SEG_STYLE[sg], lw=2.6,
                    label=f"{SEG_NAME.get(sg, sg)} "
                          f"({KIND_NAME.get(kinds[sg], kinds[sg])})")
             for sg in SEG_NAME if sg in segs]
    # Strata key inside (a), over the uniform Tm tint; segment key below.
    a1.legend(handles=h_strata, loc="lower center", frameon=True,
              bbox_to_anchor=(0.5, 0.03))
    fig.legend(handles=h_seg, loc="outside lower center", ncol=2,
               frameon=False, handlelength=1.6, columnspacing=1.2)
    for p in ps.save(fig, a.out):
        print(f"wrote {p}")
    meta = {"segment_kinds": kinds, "shares": shares,
            "n_interior": a.n_interior, "seed": a.seed,
            "cut_angle_deg": g.CUT_ANGLE, "x_crest_m": g.X_CREST,
            "z_base_m": g.Z_BASE}
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    for t, v in shares.items():
        print(f"  {t:5s} area {100*v['area']:5.1f}%   points {100*v['points']:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
