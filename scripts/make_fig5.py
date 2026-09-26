r"""Fig. 5 -- pore-pressure field at t = 0, 1, 7, 30 days, with the analytic IC
as a control row.

Two panels per time: the field psi (m of head) and its DEPARTURE FROM THE
INITIAL CONDITION, psi - psi_0. The second is the one that carries information.
Section 5 is unsaturated everywhere (Z_WT = 197 m is 3 m below Z_BASE = 200 m),
psi runs -3 to -161 m across the domain, and on that range a raw psi panel at
t = 30 is visually identical to one at t = 0 whatever the model did. The
difference panel is where a wetting front would appear if there were one.

READ IT AGAINST D-A.4. Under the capacity-limited rain boundary the marls
accept 1e-9 m/s, which is 2.59 mm of water over the window and a front of
8-10 mm in a 158 m domain. The expected and CORRECT appearance of the
difference panels is therefore near-zero everywhere. A large excursion is a
finding about the model, not about the hydrology -- check `psi>=0 frac` in the
header before believing any structure you see.

The colour scale is shared across the time panels so they are comparable; the
difference panels get their own symmetric scale about zero.

    PYTHONPATH=. python scripts/make_fig5.py runs/ansatz/exp_seed7/ckpt_final.pt
    PYTHONPATH=. python scripts/make_fig5.py CKPT --out docs/figs/fig5.png --n 220

LAYOUT (thesis, A4 portrait): one ROW per time, two columns -- psi | psi -
psi_0 -- at true scale (equal aspect), 6.25 in wide. Each column has ONE
horizontal colour bar shared by all its rows, so the times are directly
comparable. Style: `src/plot_style.py`. Writes PNG (300 dpi) and PDF.

CAPTION MUST SAY (O-21): the hydraulic field is essentially static over the
30-day window; the departure column is the evidence, not a plotting fault.
"""
import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from src import plot_style as ps
from src.config import bounds_from_geometry, full
from src.loss import psi_of
from src.model import PINN, NearPhysical
from src.nondim import SCALES
from src.step21_geometry import geometry as g

Z_WT = 197.0


def net_from_ckpt(d, cfg, BOUNDS, mode=None, eps_psi=None):
    """Rebuild the ansatz the checkpoint was trained with (D-A.3). `mode`
    changes the forward pass, so a default rebuild evaluates a cap/exp run as
    unbounded and yields a different field from the same weights."""
    saved = d.get("cfg") or {}
    kw = {}
    eps = eps_psi if eps_psi is not None else saved.get("eps_psi")
    md = mode if mode is not None else saved.get("ansatz")
    ck = saved.get("cap_k")
    if eps is not None:
        kw["eps_psi"] = float(eps)
    if md is not None:
        kw["mode"] = md
    if ck is not None:
        kw["cap_k"] = float(ck)
    uv = saved.get("eps_uv")   # Day 42: rebuild at the
    if uv is not None:         # checkpoint's eps_uv, not the default
        kw["eps_uv"] = float(uv)
    if md is None:
        print("WARNING: checkpoint carries no 'ansatz' key; using the default. "
              "If this was a cap/exp run the field below is WRONG.")
    net = NearPhysical(PINN(cfg, BOUNDS), **kw)
    net.load_state_dict(d["net"])
    return net


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--t", type=float, nargs="+", default=[0.0, 1.0, 7.0, 30.0])
    ap.add_argument("--n", type=int, default=320, help="grid points per axis")
    ap.add_argument("--dpct", type=float, default=99.5,
                    help="departure colour limit = this percentile of "
                         "|psi - psi_0| over all panels; values beyond it "
                         "saturate (bar arrows). 100 = the plain maximum.")
    ap.add_argument("--mode", default=None, choices=("unbounded", "cap", "exp"))
    ap.add_argument("--eps-psi", type=float, default=None)
    ap.add_argument("--out", default="docs/figs/fig5.png",
                    help="PNG path; a PDF is written next to it")
    a = ap.parse_args(argv)
    ps.apply()

    BOUNDS = bounds_from_geometry(t_max=30.0, s=SCALES)
    cfg = full()
    cfg.device, cfg.dtype = "cpu", "float64"
    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = net_from_ckpt(d, cfg, BOUNDS, mode=a.mode, eps_psi=a.eps_psi)
    net.eval()
    print(f"checkpoint: {a.ckpt}   step: {d.get('step')}")
    print(f"ansatz:     {net.summary()}")

    # Grid over the geometry bbox, masked to the domain.
    xs = np.linspace(g.X_MIN, g.X_MAX, a.n)
    # Up to the highest ground point, not Z_CREST: the natural ground behind
    # the crest rises to ~358 m, and a Z_CREST ceiling silently cut the upper
    # Mk_d out of every panel.
    zs = np.linspace(g.Z_BASE, float(np.max(g.z_ground(xs))), a.n)
    X, Z = np.meshgrid(xs, zs)
    inside = np.asarray(g.inside_domain(X.ravel(), Z.ravel())).reshape(X.shape)
    print(f"grid {a.n}x{a.n}, {inside.sum()} points inside the domain "
          f"({100*inside.mean():.1f}%)")

    L, H = SCALES.L_ref, SCALES.H_ref
    xf = torch.tensor(X.ravel() / L, dtype=torch.float64).reshape(-1, 1)
    zf = torch.tensor(Z.ravel() / L, dtype=torch.float64).reshape(-1, 1)

    # Analytic hydrostatic IC in metres of head, the control.
    psi0 = -(Z - Z_WT)

    fields, sat = [], []
    for t_days in a.t:
        tf = torch.full_like(xf, t_days)
        with torch.no_grad():
            psi = psi_of(net(xf, zf, tf)).reshape(X.shape).numpy() * H
        psi = np.where(inside, psi, np.nan)
        fields.append(psi)
        frac = float(np.nanmean((psi >= 0).astype(float)))
        sat.append(frac)
        print(f"  t={t_days:>5.1f} d   psi median {np.nanmedian(psi):>9.2f} m   "
              f"min {np.nanmin(psi):>9.2f}   max {np.nanmax(psi):>8.2f}   "
              f"psi>=0 frac {frac:.4f}")

    if any(s > 0 for s in sat):
        print("\nWARNING: psi >= 0 present. The domain is unsaturated "
              "everywhere by construction\n(Z_WT = 197 m is below Z_BASE = "
              "200 m), so those points are unphysical.\nC* is identically zero "
              "there and the equation has changed type (D-A.2).")

    psi0m = np.where(inside, psi0, np.nan)
    diffs = [f - psi0m for f in fields]

    vmin = float(np.nanmin(fields))
    vmax = float(np.nanmax(fields))
    dabs = np.abs(np.asarray(diffs))
    dtrue = float(np.nanmax(dabs)) or 1.0
    dmax = float(np.nanpercentile(dabs, a.dpct)) or dtrue
    print(f"departure colour limit +/-{dmax:.3g} m (p{a.dpct:g}); true max "
          f"{dtrue:.3g} m" + ("  -- beyond-limit values saturate"
                              if dtrue > dmax * 1.001 else ""))

    nt = len(a.t)
    fig, axes = plt.subplots(nt, 2, figsize=(ps.WIDTH, 1.55 * nt + 1.55),
                             sharex=True, sharey=True, squeeze=False,
                             layout="constrained")
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.02,
                                hspace=0.02)
    xs_top = np.linspace(g.X_MIN, g.X_MAX, 600)

    norm_d = TwoSlopeNorm(vcenter=0.0, vmin=-dmax, vmax=dmax)
    for j, t_days in enumerate(a.t):
        a0, a1 = axes[j]
        im = a0.pcolormesh(X, Z, fields[j], cmap=ps.CMAP_FIELD, vmin=vmin,
                           vmax=vmax, shading="auto", rasterized=True)
        im2 = a1.pcolormesh(X, Z, diffs[j], cmap=ps.CMAP_DIVERGING,
                            norm=norm_d, shading="auto", rasterized=True)
        for ax in (a0, a1):
            ps.field_axes(ax)
            ax.set_aspect("equal")
            ps.domain_outline(ax, lw=0.8)
        # Letter and time share the empty sky above the cut face.
        for k, ax in enumerate((a0, a1)):
            ax.text(0.015, 0.975, f"({'abcdefghijklmnop'[2 * j + k]})  "
                    f"$t$ = {t_days:g} d", transform=ax.transAxes, ha="left",
                    va="top", zorder=20)
        a0.set_ylabel("$z$ (m)")
    axes[0][0].set_title(r"Pressure head $\psi$")
    axes[0][1].set_title(r"Departure from IC, $\psi - \psi_0$")
    for ax in axes[-1]:
        ax.set_xlabel("$x$ (m)")
    x0, x1, z0, z1 = ps.domain_extent()
    axes[0][0].set_xlim(x0, x1)
    axes[0][0].set_ylim(z0, z1)

    ps.colorbar(fig, im, axes[:, 0].tolist(), r"$\psi$ (m of head)",
                location="bottom", fraction=0.035, pad=0.01, aspect=30)
    cb = ps.colorbar(fig, im2, axes[:, 1].tolist(),
                     r"$\psi - \psi_0$ (m)", location="bottom",
                     fraction=0.035, pad=0.01, aspect=30,
                     extend="both" if dtrue > dmax * 1.001 else "neither")
    cb.formatter.set_powerlimits((-2, 3))

    for p in ps.save(fig, a.out):
        print(f"wrote {p}")

    span = float(np.nanmax(np.abs(diffs[-1])))
    print(f"\nlargest |psi - psi_0| at t = {a.t[-1]:g} d: {span:.3f} m")
    print("Expected under the capacity-limited rain BC: millimetres "
          "(D-A.4).\nA metres-scale excursion is a finding about the model, "
          "not the hydrology.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())