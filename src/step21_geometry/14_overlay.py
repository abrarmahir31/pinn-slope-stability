#!/usr/bin/env python
"""
14_overlay.py -- Day 14 verification: digitised geometry against the source raster.

This is the only check in Phase 2 that is NOT self-referential. Every test in
09_checks.py re-derives quantities FROM the CSVs, so they agree with each other
by construction. This one compares the CSVs against Ulusay et al.'s printed
figure. If a contact was traced onto the wrong line, only this will show it.

Run from step21_geometry/:

    python 14_overlay.py --discover      # find CSVs and the raster, print sizes
    python 14_overlay.py --grid          # overlay + coordinate grid, to fix calibration
    python 14_overlay.py                 # final figure -> output/fig03_domain_overlay.png

Calibration: edit CALIB below. You need two points whose pixel position in the
raster AND (x, z) position in your coordinate system you know. The pit toe and
the crest are good choices -- both are already in geometry.py as constants.
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import geometry as G
import boundaries as B

# --------------------------------------------------------------------------
# CALIBRATION -- edit these four lines, then run with --grid to verify
# --------------------------------------------------------------------------
# Two reference points. px/py are pixel coords in the raster (origin top-left,
# as reported by any image viewer). x/z are the same points in metres.
CALIB = {
    "p1": dict(px=None, py=None, x=0.0,   z=G.Z_TOE),    # pit toe
    "p2": dict(px=None, py=None, x=G.X_CREST, z=G.Z_CREST),  # crest
}
RASTER = "figures/fig19d_section5.png"
OUTDIR = "output"

COLORS = {"Mk": "#c98a3a", "Mk_d": "#a3453b", "Tm": "#3f6d8f"}
CONTACT_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231",
                  "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]


# --------------------------------------------------------------------------
def find_csvs():
    """Locate digitised contact CSVs wherever they ended up."""
    pats = ["data/digitised/*.csv", "data/*.csv", "digitised/*.csv",
            "csv/*.csv", "*.csv"]
    for p in pats:
        hits = sorted(glob.glob(p))
        if hits:
            return p, hits
    return None, []


def load_contact(path):
    """Load an (x, z) CSV, tolerating a header row and comma or whitespace."""
    for kw in (dict(delimiter=",", skiprows=1), dict(delimiter=","), dict()):
        try:
            a = np.loadtxt(path, **kw)
            if a.ndim == 2 and a.shape[1] >= 2:
                return a[:, :2]
        except Exception:
            continue
    return None


def discover():
    pat, hits = find_csvs()
    print(f"CSV pattern matched: {pat}")
    for h in hits:
        a = load_contact(h)
        if a is None:
            print(f"  {h:45s}  UNREADABLE")
        else:
            print(f"  {os.path.basename(h):45s}  {len(a):5d} pts   "
                  f"x {a[:,0].min():7.1f}-{a[:,0].max():7.1f}   "
                  f"z {a[:,1].min():7.1f}-{a[:,1].max():7.1f}")
    print()
    if os.path.exists(RASTER):
        img = plt.imread(RASTER)
        print(f"Raster {RASTER}: {img.shape[1]} x {img.shape[0]} px")
        print("Open it in an image viewer, hover over the pit toe and the crest,")
        print("and put those pixel coordinates into CALIB.")
    else:
        print(f"Raster NOT FOUND at {RASTER}")
        for c in sorted(glob.glob("figures/*.png")):
            print("   candidate:", c)


def extent_from_calib():
    """Axis-aligned affine from two reference points -> imshow extent."""
    p1, p2 = CALIB["p1"], CALIB["p2"]
    if p1["px"] is None or p2["px"] is None:
        return None
    img = plt.imread(RASTER)
    H, W = img.shape[0], img.shape[1]

    sx = (p2["x"] - p1["x"]) / (p2["px"] - p1["px"])          # m per px
    sz = (p2["z"] - p1["z"]) / (p2["py"] - p1["py"])          # m per px (negative)

    x_left = p1["x"] - p1["px"] * sx
    x_right = x_left + W * sx
    z_top = p1["z"] - p1["py"] * sz
    z_bot = z_top + H * sz
    return [x_left, x_right, z_bot, z_top]


# --------------------------------------------------------------------------
def marl_band_diagnostic():
    """Locate where the marl band thickness goes negative.

    Top-of-Tm is traced in TWO pieces: d_mk_tm.csv (Mk on Tm, x 3.8-90.2) and
    d_mkd_base.csv (Mk_d on Tm, x 95.1-266.1). There is a 4.9 m gap between
    them and X_MK_DIVIDE = 90.7 sits inside it.
    """
    print("\n--- marl band thickness scan ---")
    pat, hits = find_csvs()
    names = {os.path.basename(h).lower(): h for h in hits}
    left   = next((v for k, v in names.items() if "mk_tm"    in k), None)
    right  = next((v for k, v in names.items() if "mkd_base" in k), None)
    ground = next((v for k, v in names.items() if "ground"   in k), None)
    if not (left and right and ground):
        print("  contacts not identified; skipping.")
        return None
    L, R = load_contact(left), load_contact(right)
    a = np.vstack([L, R])
    a = a[np.argsort(a[:, 0])]
    print(f"  joined top-of-Tm: {len(a)} pts, "
          f"gap {L[:, 0].max():.1f} -> {R[:, 0].min():.1f} m "
          f"(X_MK_DIVIDE = {G.X_MK_DIVIDE})")
    xs = np.linspace(a[:, 0].min(), a[:, 0].max(), 2000)
    ztm = np.interp(xs, a[:, 0], a[:, 1])
    b = load_contact(ground)
    zg = np.interp(xs, b[:, 0], b[:, 1])
    t = zg - ztm
    i = int(np.argmin(t))
    print(f"  min thickness {t.min():+.3f} m at x = {xs[i]:.1f} m")
    if t.min() < 0:
        bad = t < 0
        print(f"  NEGATIVE from x = {xs[bad].min():.1f} to {xs[bad].max():.1f} m")
        inside = L[:, 0].max() <= xs[i] <= R[:, 0].min()
        print("  -> IN THE TRACE GAP" if inside else "  -> outside the gap: real")
        for cand in (90.7, 92.65, 95.1):
            zc = np.interp(cand, a[:, 0], a[:, 1])
            print(f"     X_MK_DIVIDE = {cand:6.2f} -> top-of-Tm z = {zc:.2f} m")
        return float(xs[i])
    return None

# --------------------------------------------------------------------------
def build(show_grid=False, zoom_x=None):
    os.makedirs(OUTDIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=200)

    # (1) raster background
    ext = extent_from_calib()
    if ext and os.path.exists(RASTER):
        ax.imshow(plt.imread(RASTER), extent=ext, aspect="equal",
                  alpha=0.55, zorder=0, interpolation="bilinear")
    else:
        print("!! No calibration -- background omitted. This is the whole point of")
        print("!! the script; fill in CALIB before trusting the figure.")

    # (2) collocation cloud, tinted by material_tag
    rng = np.random.default_rng(20260814)
    N = 60000
    px = rng.uniform(G.X_MIN, G.X_MAX, N)
    pz = rng.uniform(G.Z_BASE, 370.0, N)
    keep = np.array([G.inside_domain(a, b) for a, b in zip(px, pz)])
    px, pz = px[keep][:2500], pz[keep][:2500]
    tags = np.array([str(G.material_tag(a, b)) for a, b in zip(px, pz)])
    for t, c in COLORS.items():
        m = tags == t
        if m.any():
            ax.scatter(px[m], pz[m], s=1.6, c=c, alpha=0.45, lw=0,
                       zorder=2, rasterized=True)
    print("\ncollocation tag fractions:",
          {t: round(float((tags == t).mean()), 4) for t in COLORS})

    # (3) digitised contacts
    _, hits = find_csvs()
    for i, h in enumerate(hits):
        a = load_contact(h)
        if a is None or len(a) < 2:
            continue
        ax.plot(a[:, 0], a[:, 1], lw=1.3, zorder=3,
                color=CONTACT_COLORS[i % len(CONTACT_COLORS)],
                label=os.path.splitext(os.path.basename(h))[0])

    # (4) boundary samples + outward normals on the cut face
    try:
        segs = B.sample_boundaries(n=1200, seed=0)
        for name, val in (segs.items() if isinstance(segs, dict) else []):
            pts = val["pts"] if isinstance(val, dict) and "pts" in val else val
            pts = np.asarray(pts)
            if pts.ndim != 2 or pts.shape[1] < 2:
                continue
            ax.scatter(pts[:, 0], pts[:, 1], s=5, marker="s", zorder=4,
                       edgecolors="none", label=f"BC: {name}")
            if "cut" in str(name).lower() or "face" in str(name).lower():
                sub = pts[:: max(1, len(pts) // 12)]
                try:
                    nrm = np.asarray(B.traction_free_normals(name, sub))
                    ax.quiver(sub[:, 0], sub[:, 1], nrm[:, 0], nrm[:, 1],
                              color="k", width=0.0028, scale=22, zorder=5)
                except Exception as e:
                    print("  (normals skipped:", e, ")")
    except Exception as e:
        print("  (boundary overlay skipped:", e, ")")

    # (5) annotations
    ax.axvline(G.X_MK_DIVIDE, color="k", ls=":", lw=0.9, zorder=6)
    ax.text(G.X_MK_DIVIDE + 2, G.Z_BASE + 8, "X_MK_DIVIDE\n(assumed)",
            fontsize=6.5, va="bottom")
    ax.axhline(G.Z_BASE, color="k", ls="--", lw=0.9, zorder=6)
    ax.axhline(197.0, color="#1f6fb4", ls=":", lw=1.1, zorder=6)
    ax.text(G.X_MAX - 4, 197.5, "karstic water table 197 m a.s.l.",
            fontsize=6.5, ha="right", color="#1f6fb4")

    if show_grid:
        ax.set_xticks(np.arange(0, G.X_MAX + 1, 20))
        ax.set_yticks(np.arange(G.Z_BASE, 371, 10))
        ax.grid(True, lw=0.3, alpha=0.6, zorder=1)

    if zoom_x:
        ax.set_xlim(zoom_x[0], zoom_x[1])
    else:
        ax.set_xlim(G.X_MIN - 5, G.X_MAX + 5)
    ax.set_ylim(G.Z_BASE - 5, 372)
    ax.set_aspect("equal")
    ax.set_xlabel("x from pit toe (m)")
    ax.set_ylabel("elevation (m a.s.l.)")
    ax.set_title("Section 5-5′ — digitised domain over Ulusay et al. (2014) Fig. 19d",
                 fontsize=10)

    h, l = ax.get_legend_handles_labels()
    h += [Line2D([], [], marker="o", ls="", color=c, label=f"{t} collocation")
          for t, c in COLORS.items()]
    ax.legend(handles=h, fontsize=5.6, ncol=3, loc="upper left", framealpha=0.9)

    out = os.path.join(OUTDIR, "fig03_domain_overlay"
                       + ("_grid" if show_grid else "") + ".png")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"\nwrote {out}")
    return out


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--zoom", nargs=2, type=float, default=None)
    args = ap.parse_args()

    if args.discover:
        discover()
        sys.exit(0)

    bad_x = marl_band_diagnostic()
    build(show_grid=args.grid, zoom_x=args.zoom)
    if bad_x is not None:
        build(show_grid=True, zoom_x=(bad_x - 15, bad_x + 15))
