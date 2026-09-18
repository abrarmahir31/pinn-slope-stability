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
    PYTHONPATH=. python scripts/make_fig5.py CKPT --out docs/fig5.png --n 220
"""
import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    ap.add_argument("--n", type=int, default=200, help="grid points per axis")
    ap.add_argument("--mode", default=None, choices=("unbounded", "cap", "exp"))
    ap.add_argument("--eps-psi", type=float, default=None)
    ap.add_argument("--out", default="docs/fig5.png")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args(argv)

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
    zs = np.linspace(g.Z_BASE, g.Z_CREST, a.n)
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
    dmax = float(np.nanmax(np.abs(diffs))) or 1.0

    nt = len(a.t)
    fig, axes = plt.subplots(2, nt, figsize=(3.5 * nt, 6.4), squeeze=False)
    ext = [g.X_MIN, g.X_MAX, g.Z_BASE, g.Z_CREST]

    for j, t_days in enumerate(a.t):
        im = axes[0][j].imshow(fields[j], origin="lower", extent=ext,
                               aspect="auto", cmap="viridis",
                               vmin=vmin, vmax=vmax)
        axes[0][j].set_title(f"t = {t_days:g} d")
        if j == 0:
            axes[0][j].set_ylabel(r"$\psi$  (m head)" + "\nelevation (m)")
        im2 = axes[1][j].imshow(diffs[j], origin="lower", extent=ext,
                                aspect="auto", cmap="RdBu_r",
                                vmin=-dmax, vmax=dmax)
        axes[1][j].set_xlabel("x (m)")
        if j == 0:
            axes[1][j].set_ylabel(r"$\psi - \psi_0$  (m)" + "\nelevation (m)")

    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.85, label=r"$\psi$ (m)")
    fig.colorbar(im2, ax=axes[1].tolist(), shrink=0.85,
                 label=r"$\psi-\psi_0$ (m)")
    fig.suptitle("Pore-pressure head field and departure from the initial "
                 "condition", y=0.99)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
    print(f"\nwrote {a.out}")

    span = float(np.nanmax(np.abs(diffs[-1])))
    print(f"\nlargest |psi - psi_0| at t = {a.t[-1]:g} d: {span:.3f} m")
    print("Expected under the capacity-limited rain BC: millimetres "
          "(D-A.4).\nA metres-scale excursion is a finding about the model, "
          "not the hydrology.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())