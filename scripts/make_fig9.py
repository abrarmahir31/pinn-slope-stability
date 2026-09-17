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
"""
import argparse
import dataclasses
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

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
    return NearPhysical(PINN(pc, BOUNDS), eps_psi=cfg["eps_psi"],
                        mode=cfg["ansatz"], cap_k=cfg.get("cap_k", 100.0))


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


def gamma_grid(net, t_days, nx, nz, chunk=4096):
    xs = np.linspace(g.X_MIN, g.X_MAX, nx)
    zs = np.linspace(g.Z_BASE, float(g.z_ground(xs).max()), nz)
    X, Z = np.meshgrid(xs, zs)
    inside = np.asarray(g.inside_domain(X, Z), bool)
    G = np.full(X.shape, np.nan)
    px, pz = X[inside], Z[inside]
    vals = []
    for i in range(0, px.size, chunk):
        x = torch.tensor(px[i:i + chunk] / SCALES.L_ref).reshape(-1, 1).requires_grad_(True)
        z = torch.tensor(pz[i:i + chunk] / SCALES.L_ref).reshape(-1, 1).requires_grad_(True)
        t = torch.full_like(x, t_days)
        u, v = uv_of(net(x, z, t))
        exx, ezz, exz, _ = strain_star(u, v, x, z, physical=True)
        vals.append(fsurf.gamma_max(exx.detach().numpy().ravel(),
                                    ezz.detach().numpy().ravel(),
                                    exz.detach().numpy().ravel()))
    G[inside] = np.concatenate(vals)
    return X, Z, G, inside


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
    a = ap.parse_args(argv)

    n = len(a.run)
    fig, axes = plt.subplots(n + 1, 1, figsize=(8, 3.2 * (n + 1)))
    summary, xg = [], np.linspace(g.X_MIN, g.X_MAX, 400)
    for i, run in enumerate(a.run):
        p, srf = state_path(run, a.state)
        net, d = _load_net(p)
        X, Z, G, inside = gamma_grid(net, a.t, a.nx, a.nz)
        net0, _ = _load_net(start_state_path(run))
        _, _, G0, _ = gamma_grid(net0, a.t, a.nx, a.nz)
        li0 = fsurf.localisation_intensity(G0, inside)
        if not np.isfinite(G).any() or np.nanmax(G) <= 0:
            raise ValueError(f"{run}: gamma_max is empty or zero -- not plotting "
                             f"a blank field")
        xr, zr = fsurf.extract_ridge(X, Z, G, mask=inside, min_frac=a.min_frac)
        crit = d.get("sweep", {}).get("criterion", os.path.basename(run))
        rec = {"run": run, "criterion": crit, "state": a.state, "srf": srf,
               "localisation_intensity": fsurf.localisation_intensity(G, inside),
               "localisation_intensity_srf_start": li0,
               "band_width_m": fsurf.band_width(X, Z, G, mask=inside),
               "grid": [a.nx, a.nz], "ridge_x": xr.tolist(), "ridge_z": zr.tolist()}
        rec["li_ratio_vs_start"] = rec["localisation_intensity"] / li0
        summary.append(rec)
        ax = axes[i]
        im = ax.pcolormesh(X, Z, np.log10(G), shading="auto", cmap="viridis")
        fig.colorbar(im, ax=ax, label=r"$\log_{10}\gamma_{max}$")
        ax.plot(xr, zr, "r-", lw=1.2)
        ax.plot(xg, g.z_ground(xg), "k-", lw=0.8)
        ax.set(title=f"{crit}  SRF {srf:.3f} ({a.state})  "
                     f"LI {rec['localisation_intensity']:.1f} "
                     f"({rec['li_ratio_vs_start']:.1f}x SRF-start)", ylabel="z (m)")
        axes[n].plot(xr, zr, "-", lw=1.5, label=f"{crit} SRF {srf:.3f}")
        print(f"{run}: SRF {srf:.3f}  ridge points {xr.size}  "
              f"LI {rec['localisation_intensity']:.2f}  "
              f"band width {rec['band_width_m']:.2f} m  "
              f"LI/LI(start) {rec['li_ratio_vs_start']:.2f}")
        print("  A ridge is only a failure surface if a band formed: read "
              "LI/LI(start) before the red line.")
    axes[n].plot(xg, g.z_ground(xg), "k-", lw=0.8)
    axes[n].set(title="extracted surfaces", xlabel="x (m)", ylabel="z (m)")
    axes[n].legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200)
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
