r"""Fig 9 -- gamma_max and the extracted failure surface from a critical state.

    python scripts/make_fig9.py --run runs/ssr_ghb_w1 --run runs/ssr_mc_w1 \
        --state failed --out docs/figs/fig9.png

`--state` has no default. `failed` is the first SRF that met the failure
signature, `stable` the last that did not; which one is "the critical state"
is a reporting choice and goes in the caption. Forward pass only (autograd
for strain), laptop-friendly.

`extract_ridge` draws a line through ANY field, including an elastic one with
no band. So each run also reports LI relative to the sweep's own SRF-start
state (`li_ratio_vs_start`); near 1 means no localisation formed and the red
line is not a slip surface. The ratio is reported, not thresholded.

Reports per run: localisation intensity (peak/median gamma_max) and band
width. If the band width changes materially with --nx/--nz the band is a
resolution artefact (failure_surface.band_width docstring) -- run twice.

FIGURE (revision 4, thesis layout): one full-width panel per run at true
scale, ALL on one shared log10(gamma_max) colour scale (one colour bar), the
strata contacts overlaid, the extracted ridge in white-edged black, and a
crest inset with unit displacement-direction arrows. The old
"extracted surfaces" summary panel is gone: the ridge is on each panel.
Style: `src/plot_style.py`; PNG (300 dpi) + PDF; JSON unchanged.

CAPTION MUST SAY (D-5.15): the state shown is `failed` (first SRF at which
the calibrated criterion fired) and why; LI/LI(start) per run; that a
continuum without bedding planes cannot represent the bedding-plane slides
Ulusay et al. (2014) report.
"""
import argparse
import dataclasses
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch

from src import plot_style as ps
from src import failure_surface as fsurf
from src.config import BOUNDS, tiny
from src.mechanics import strain_star, uv_of
from src.model import PINN, NearPhysical
from src.nondim import SCALES
from src.step21_geometry import geometry as g


def build_net(cfg: dict):
    """Rebuild the network exactly as train.setup does (D-A.3)."""
    for k in ("layers", "width", "ansatz", "eps_psi"):
        if k not in cfg:
            raise KeyError(f"state cfg lacks {k!r}")
    pc = dataclasses.replace(tiny(), n_layers=cfg["layers"],
                             n_neurons=cfg["width"], seed=cfg.get("seed", 0),
                             device="cpu")
    uv = {} if cfg.get("eps_uv") is None else {"eps_uv": float(cfg["eps_uv"])}
    return NearPhysical(PINN(pc, BOUNDS), eps_psi=cfg["eps_psi"],
                        mode=cfg["ansatz"], cap_k=cfg.get("cap_k", 100.0), **uv)


def state_path(run_dir: str, state: str) -> tuple[str, float]:
    with open(os.path.join(run_dir, "result.json")) as f:
        res = json.load(f)
    lo, hi = res["bracket"]
    srf = hi if state == "failed" else lo
    if srf is None:
        raise ValueError(f"{run_dir}: no failed SRF (status {res.get('status')})")
    p = os.path.join(run_dir, "states", f"srf_{round(srf, 6):.6f}.pt")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p, srf


def start_state_path(run_dir: str) -> str:
    """The lowest-SRF state the sweep saved: its own elastic reference."""
    sd = os.path.join(run_dir, "states")
    files = sorted((float(f[4:-3]), f) for f in os.listdir(sd)
                   if f.startswith("srf_") and f.endswith(".pt"))
    if not files:
        raise FileNotFoundError(f"no states in {sd}")
    return os.path.join(sd, files[0][1])


def _load_net(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    net = build_net(d["cfg"])
    net.load_state_dict(d["net"])
    net.eval()
    return net, d


def gamma_grid(net, t_days, nx, nz, chunk=4096, with_uv=False):
    xs = np.linspace(g.X_MIN, g.X_MAX, nx)
    zs = np.linspace(g.Z_BASE, float(g.z_ground(xs).max()), nz)
    X, Z = np.meshgrid(xs, zs)
    inside = np.asarray(g.inside_domain(X, Z), bool)
    G = np.full(X.shape, np.nan)
    U = np.full(X.shape, np.nan)
    V = np.full(X.shape, np.nan)
    px, pz = X[inside], Z[inside]
    vals, us, vs = [], [], []
    for i in range(0, px.size, chunk):
        x = torch.tensor(px[i:i + chunk] / SCALES.L_ref).reshape(-1, 1).requires_grad_(True)
        z = torch.tensor(pz[i:i + chunk] / SCALES.L_ref).reshape(-1, 1).requires_grad_(True)
        t = torch.full_like(x, t_days)
        u, v = uv_of(net(x, z, t))
        exx, ezz, exz, _ = strain_star(u, v, x, z, physical=True)
        vals.append(fsurf.gamma_max(exx.detach().numpy().ravel(),
                                    ezz.detach().numpy().ravel(),
                                    exz.detach().numpy().ravel()))
        us.append(u.detach().numpy().ravel())
        vs.append(v.detach().numpy().ravel())
    G[inside] = np.concatenate(vals)
    if with_uv:
        U[inside] = np.concatenate(us)
        V[inside] = np.concatenate(vs)
        return X, Z, G, inside, U, V
    return X, Z, G, inside


#: Crest window for the inset (m): the cut-face top, bench and the Mk / Mk_d
#: contact, where the Phase 5 strain concentrates.
CREST_WIN = (80.0, 135.0, 300.0, 342.0)


def _crest_inset(ax, X, Z, LG, U, V, xr, zr, norm):
    """Zoom on the crest with unit displacement-direction arrows."""
    xa, xb, za, zb = CREST_WIN
    ins = ax.inset_axes([0.58, 0.035, 0.41, 0.54])
    ins.set_facecolor("white")        # nothing of the main panel shows through
    ins.set_anchor("SE")
    ins.set_zorder(12)                # above the main panel's ridge line
    ins.pcolormesh(X, Z, LG, shading="auto", cmap=ps.CMAP_MAGNITUDE,
                   norm=norm, rasterized=True)
    ps.draw_strata(ins, fill=False, lines=True, lw=0.8, color="white",
                   x=(xa, xb), z=(za, zb), nx=200, nz=160)
    ps.domain_outline(ins, lw=0.9)
    ins.plot(xr, zr, "-", color="black", lw=2.0)
    ins.plot(xr, zr, "-", color="white", lw=0.8)
    m = ((X >= xa) & (X <= xb) & (Z >= za) & (Z <= zb) & np.isfinite(U)
         & np.isfinite(V))
    step = max(1, int(np.ceil(np.sqrt(m.sum() / 60.0))))
    sub = np.zeros_like(m)
    sub[::step, ::step] = True
    k = m & sub
    mag = np.hypot(U[k], V[k])
    ok = mag > 0
    if ok.any():   # nothing drawn for --inset-arrows none (all nan)
        ins.quiver(X[k][ok], Z[k][ok], (U[k] / mag)[ok], (V[k] / mag)[ok],
                   angles="xy", pivot="mid", color="white", edgecolor=ps.INK,
                   linewidth=0.4, scale=16, width=0.011, headwidth=3.5,
                   headlength=4, headaxislength=3.6, zorder=7)
    ins.set(xlim=(xa, xb), ylim=(za, zb))
    ins.set_aspect("equal")
    ps.field_axes(ins)
    ins.set_xticks([])
    ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_edgecolor(ps.INK)
        sp.set_linewidth(1.1)
    ax.indicate_inset_zoom(ins, edgecolor=ps.INK, alpha=1.0, lw=0.9)
    return ins


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--state", choices=("failed", "stable"), required=True)
    ap.add_argument("--t", type=float, default=30.0)
    ap.add_argument("--nx", type=int, default=200)
    ap.add_argument("--nz", type=int, default=200)
    ap.add_argument("--min-frac", type=float, default=0.25)
    ap.add_argument("--out", default="docs/figs/fig9.png")
    ap.add_argument("--mask-f1-m", type=float, default=0.0,
                    help="exclude points within this horizontal distance (m) of "
                         "the F1 far-field boundary from the ridge, LI and band "
                         "width. The ground/F1 corner (traction-free surface "
                         "meeting a roller) is a strain singularity; 0 keeps "
                         "the original statistics. A REPORTING CHOICE: state it.")
    ap.add_argument("--inset-arrows", choices=("total", "increment", "none"),
                    default="total",
                    help="crest-inset arrows: total displacement direction, the "
                         "increment since the sweep's SRF-start state, or none. "
                         "|d| is micrometre-scale (D-5.13 note 4, D-5.18).")
    a = ap.parse_args(argv)
    if a.mask_f1_m < 0:
        ap.error("--mask-f1-m must be >= 0")

    ps.apply()
    n = len(a.run)
    fields, summary = [], []
    for i, run in enumerate(a.run):
        p, srf = state_path(run, a.state)
        net, d = _load_net(p)
        X, Z, G, inside, U, V = gamma_grid(net, a.t, a.nx, a.nz, with_uv=True)
        net0, _ = _load_net(start_state_path(run))
        _, _, G0, _, U0, V0 = gamma_grid(net0, a.t, a.nx, a.nz, with_uv=True)
        if a.inset_arrows == "increment":
            U, V = U - U0, V - V0
        elif a.inset_arrows == "none":
            U, V = np.full_like(U, np.nan), np.full_like(V, np.nan)
        # Statistics region: the domain, optionally minus a strip along F1.
        stat = inside & (X <= g.x_f1(Z) - a.mask_f1_m)
        li0 = fsurf.localisation_intensity(G0, stat)
        if not np.isfinite(G).any() or np.nanmax(G) <= 0:
            raise ValueError(f"{run}: gamma_max is empty or zero -- not plotting "
                             f"a blank field")
        xr, zr = fsurf.extract_ridge(X, Z, G, mask=stat, min_frac=a.min_frac)
        crit = d.get("sweep", {}).get("criterion", os.path.basename(run))
        rec = {"run": run, "criterion": crit, "state": a.state, "srf": srf,
               "localisation_intensity": fsurf.localisation_intensity(G, stat),
               "localisation_intensity_srf_start": li0,
               "band_width_m": fsurf.band_width(X, Z, G, mask=stat),
               "grid": [a.nx, a.nz], "ridge_x": xr.tolist(), "ridge_z": zr.tolist(),
               "mask_f1_m": a.mask_f1_m, "inset_arrows": a.inset_arrows}
        rec["li_ratio_vs_start"] = rec["localisation_intensity"] / li0
        # Where the ridge sits relative to F1: a ridge hugging the ground/F1
        # corner is the boundary singularity, not a slip surface.
        d_f1 = (g.x_f1(zr) - xr) if xr.size else np.array([np.nan])
        rec["ridge_max_dist_to_f1_m"] = float(np.nanmax(d_f1))
        if xr.size and rec["ridge_max_dist_to_f1_m"] < 25.0:
            print(f"  WARNING: every ridge point is within "
                  f"{rec['ridge_max_dist_to_f1_m']:.1f} m of the F1 boundary -- "
                  f"the ridge is the ground/F1 corner, not a slip surface. "
                  f"Try --mask-f1-m 20.")
        summary.append(rec)
        fields.append((X, Z, G, U, V, xr, zr, crit, srf, rec))
        print(f"{run}: SRF {srf:.3f}  ridge points {xr.size}  "
              f"LI {rec['localisation_intensity']:.2f}  "
              f"band width {rec['band_width_m']:.2f} m  "
              f"LI/LI(start) {rec['li_ratio_vs_start']:.2f}")
        print("  A ridge is only a failure surface if a band formed: read "
              "LI/LI(start) before the red line.")

    # ONE colour scale for every panel (revision 4): the p1-p99.9 range of
    # log10(gamma_max) over all runs, so equal colours mean equal strain.
    allg = np.concatenate([np.log10(f[2][np.isfinite(f[2]) & (f[2] > 0)])
                           for f in fields])
    norm = Normalize(*np.percentile(allg, [1.0, 99.9]))
    x0, x1, z0, z1 = ps.domain_extent()
    fig, axes = plt.subplots(n, 1, figsize=(ps.WIDTH, 3.05 * n + 0.95),
                             sharex=True, squeeze=False, layout="constrained")
    axes = axes[:, 0]
    for i, (ax, (X, Z, G, U, V, xr, zr, crit, srf, rec)) in enumerate(
            zip(axes, fields)):
        with np.errstate(divide="ignore", invalid="ignore"):
            LG = np.log10(G)
        im = ax.pcolormesh(X, Z, LG, shading="auto", cmap=ps.CMAP_MAGNITUDE,
                           norm=norm, rasterized=True)
        ps.field_axes(ax)
        ax.set_aspect("equal")
        ps.draw_strata(ax, fill=False, lines=True, lw=0.8, color="white")
        ps.domain_outline(ax, lw=0.9)
        ax.plot(xr, zr, "-", color="black", lw=2.4, zorder=8)
        ax.plot(xr, zr, "-", color="white", lw=1.0, zorder=9)
        ax.set(xlim=(x0, x1), ylim=(z0, z1), ylabel="$z$ (m)")
        ax.text(0.015, 0.975, f"({'abcdefgh'[i]})  {crit}, SRF {srf:.3f} "
                f"({a.state}); LI/LI$_0$ = {rec['li_ratio_vs_start']:.0f}",
                transform=ax.transAxes, ha="left", va="top", zorder=20)
        if a.mask_f1_m > 0:
            zz = np.linspace(z0, z1, 50)
            ax.fill_betweenx(zz, g.x_f1(zz) - a.mask_f1_m, g.x_f1(zz),
                             facecolor="none", edgecolor="white", hatch="//",
                             lw=0.0, zorder=7)
        _crest_inset(ax, X, Z, LG, U, V, xr, zr, norm)
    axes[-1].set_xlabel("$x$ (m)")
    ps.colorbar(fig, im, axes.tolist(), r"$\log_{10}\gamma_{\max}$",
                location="bottom", fraction=0.04, pad=0.01, aspect=35)
    ps.save(fig, a.out)
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
