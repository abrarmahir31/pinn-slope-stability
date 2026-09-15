r"""Fig. 6 -- baseline displacement field at SRF = 1, with a Hoek-Brown check
that the baseline is ELASTIC and not already failing.

SRF = 1 MEANS NO STRENGTH REDUCTION. There is no strength-reduction loop in
`train.py`; `hoek_brown_ssr` lives in Phase 1 only. So every checkpoint this
project has produced is already an SRF = 1 field, and this script inspects one
rather than computing anything new. Say that in the caption: the figure is the
unreduced baseline, and the SSR sweep that would follow is a separate exercise.

THE CHECK IS THE POINT, NOT THE PICTURE. Days 34-35 ask to confirm the
baseline is elastic and not already failing. A displacement field alone cannot
answer that -- it is small everywhere by construction, because `NearPhysical`
multiplies the raw network by `eps_uv` = 1e-3 (open item: that makes the
`bc_mech` traction essentially sigma0.n). What answers it is the stress state
against the generalised Hoek-Brown envelope

    sigma_1 = sigma_3 + sigma_ci ( m_b sigma_3 / sigma_ci + s )^a

evaluated per stratum on the TOTAL stress, sigma_0 + dsigma(u,v), which is the
quantity D-3.3.3 made assemblable. Points where sigma_1 exceeds the envelope
are already at or past failure at SRF = 1, and if there are many the SSR sweep
has nothing to reduce.

SIGN CONVENTION. The residual layer is tension-positive (D-3.3.1). Hoek-Brown
is written compression-positive, so the principal stresses are negated on
entry here. That flip happens in exactly one place -- here -- for the same
reason `mechanics.sigma0_star_from_geostatic` is the only other place it
happens. Getting it wrong produces a plausible-looking field in which the
slope is stabilised by the thing that should destabilise it.

    PYTHONPATH=. python scripts/make_fig6.py runs/ansatz/exp_seed7/ckpt_final.pt
    PYTHONPATH=. python scripts/make_fig6.py CKPT --t 30 --out docs/fig6.png
"""
import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import bounds_from_geometry, full
from src.materials import load_materials
from src.mechanics import total_stress_star
from src.model import PINN, NearPhysical
from src.nondim import SCALES
from src.sampling import sample_interior
from src.sigma0 import attach_sigma0
from src.step21_geometry import geometry as g


def net_from_ckpt(d, cfg, BOUNDS, mode=None, eps_psi=None):
    """D-A.3: the ansatz comes from the checkpoint, not from a default."""
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
    if md is None:
        print("WARNING: checkpoint has no 'ansatz' key; using the default. "
              "A cap/exp run read this way gives the WRONG field.")
    net = NearPhysical(PINN(cfg, BOUNDS), **kw)
    net.load_state_dict(d["net"])
    return net


def hoek_brown_sigma1(sig3_pa, mat):
    """Envelope strength, COMPRESSION-POSITIVE, Pa. Negative sigma_3 (net
    tension) is clipped to zero: GHB is not valid in tension and extrapolating
    it there would manufacture strength that does not exist."""
    s3 = np.maximum(sig3_pa, 0.0)
    inner = mat.m_b * s3 / mat.sigma_ci + mat.s
    inner = np.maximum(inner, 0.0)
    return s3 + mat.sigma_ci * inner ** mat.a


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--t", type=float, default=30.0, help="day to inspect")
    ap.add_argument("--n", type=int, default=6000, help="collocation points")
    ap.add_argument("--sampling-seed", type=int, default=20260808)
    ap.add_argument("--mode", default=None, choices=("unbounded", "cap", "exp"))
    ap.add_argument("--eps-psi", type=float, default=None)
    ap.add_argument("--out", default="docs/fig6.png")
    ap.add_argument("--json", default="docs/fig6_elastic_check.json")
    ap.add_argument("--no-bishop", action="store_true",
                    help="evaluate FS on sigma_0 + D:eps WITHOUT the Bishop "
                         "pore term. Diagnostic: separates 'the equilibrated "
                         "baseline stress is already at failure' from 'the "
                         "psi drift pushed it over'. Run both and compare.")
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
    print(f"t = {a.t:g} d,  SRF = 1 (no strength reduction applied)"
          + ("   [BISHOP OFF -- diagnostic]" if a.no_bishop else "") + "\n")

    coll = sample_interior(a.n, a.sampling_seed)
    coll = attach_sigma0(coll)
    mats = load_materials()

    U_ref = SCALES.U_ref
    L_ref = SCALES.L_ref   # coll.x/.z are x*/L_ref; the axes are in metres
    sig_ref = SCALES.sig_ref
    rows, X, Zc, Ud, Vd, FS, TAG = [], [], [], [], [], [], []

    for tag in sorted(set(coll.tag.tolist())):
        m = torch.as_tensor(coll.tag == tag)
        if not m.any():
            continue
        x = coll.x[m].reshape(-1, 1).clone().requires_grad_(True)
        z = coll.z[m].reshape(-1, 1).clone().requires_grad_(True)
        t = torch.full_like(x, a.t).requires_grad_(True)
        s0 = coll.sigma0[m]

        out = net(x, z, t)
        u = out[:, 1:2] * U_ref
        v = out[:, 2:3] * U_ref

        # sigma0 is stored (N,3); total_stress_star wants three columns, the
        # same unpacking loss.L_BC_mech does at loss.py:399.
        (sxx, szz, sxz), _chi, _eps = total_stress_star(
            lambda X_, Z_, T_: net(X_, Z_, T_), x, z, t, mats[tag],
            sigma0_star=(s0[:, 0:1], s0[:, 1:2], s0[:, 2:3]),
            bishop=not a.no_bishop)
        sxx = sxx.detach().numpy().ravel() * sig_ref
        szz = szz.detach().numpy().ravel() * sig_ref
        sxz = sxz.detach().numpy().ravel() * sig_ref

        # Principal stresses, tension-positive as the residual layer uses.
        c = 0.5 * (sxx + szz)
        r = np.sqrt(0.25 * (sxx - szz) ** 2 + sxz ** 2)
        p_hi, p_lo = c + r, c - r
        # Flip to compression-positive for Hoek-Brown (D-3.3.1).
        s1 = -p_lo          # most compressive
        s3 = -p_hi          # least compressive
        s1_env = hoek_brown_sigma1(s3, mats[tag])
        with np.errstate(divide="ignore", invalid="ignore"):
            fs = np.where(s1 > 0, s1_env / s1, np.inf)

        # Dimensionalise: without this every point lands outside the
        # metre-scale axis limits below and all three panels render blank
        # while the colourbars still populate from the data.
        X.append(coll.x[m].detach().numpy().ravel() * L_ref)
        Zc.append(coll.z[m].detach().numpy().ravel() * L_ref)
        Ud.append(u.detach().numpy().ravel())
        Vd.append(v.detach().numpy().ravel())
        FS.append(fs)
        TAG += [tag] * int(m.sum())

        mag = np.hypot(u.detach().numpy().ravel(), v.detach().numpy().ravel())
        yielded = float(np.mean(fs < 1.0))
        rows.append({"tag": tag, "n": int(m.sum()),
                     "u_max_m": float(np.abs(Ud[-1]).max()),
                     "v_max_m": float(np.abs(Vd[-1]).max()),
                     "disp_max_m": float(mag.max()),
                     "disp_p50_m": float(np.median(mag)),
                     "sigma1_max_Pa": float(s1.max()),
                     "fs_min": float(np.nanmin(fs)),
                     "fs_p05": float(np.nanpercentile(fs, 5)),
                     "frac_yielded": yielded})

    X = np.concatenate(X); Zc = np.concatenate(Zc)
    Ud = np.concatenate(Ud); Vd = np.concatenate(Vd); FS = np.concatenate(FS)
    MAG = np.hypot(Ud, Vd)

    print(f"{'unit':>6}{'n':>7}{'|u| max':>11}{'|v| max':>11}"
          f"{'|d| p50':>11}{'FS min':>10}{'FS p05':>10}{'yielded':>9}")
    for r_ in rows:
        print(f"{r_['tag']:>6}{r_['n']:>7}{r_['u_max_m']:>11.3e}"
              f"{r_['v_max_m']:>11.3e}{r_['disp_p50_m']:>11.3e}"
              f"{r_['fs_min']:>10.3f}{r_['fs_p05']:>10.3f}"
              f"{r_['frac_yielded']:>9.4f}")

    frac_all = float(np.mean(FS < 1.0))
    print(f"\nDOMAIN-WIDE: {100*frac_all:.3f}% of points have FS < 1 at SRF = 1")
    if frac_all == 0.0:
        print("VERDICT: baseline is ELASTIC everywhere. An SSR sweep has "
              "something to reduce.")
    elif frac_all < 0.01:
        print("VERDICT: essentially elastic; isolated points below 1 are "
              "probably the free surface,\nwhere sigma_3 goes into tension and "
              "GHB is not valid. Check where they are before\ntreating them as "
              "failure.")
    else:
        print("VERDICT: THE BASELINE IS ALREADY FAILING at SRF = 1. An SSR "
              "sweep cannot start\nfrom here -- there is nothing left to "
              "reduce. Resolve this before Days 36-37.")

    # -- figure -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    sc0 = axes[0].scatter(X, Zc, c=MAG, s=3, cmap="magma")
    axes[0].set_title(f"displacement magnitude, t = {a.t:g} d")
    fig.colorbar(sc0, ax=axes[0], label="|d| (m)")

    step = max(1, len(X) // 700)
    axes[1].quiver(X[::step], Zc[::step], Ud[::step], Vd[::step],
                   MAG[::step], cmap="magma", scale_units="xy", angles="xy")
    axes[1].set_title("displacement direction")

    fsc = np.clip(FS, 0, 3)
    sc2 = axes[2].scatter(X, Zc, c=fsc, s=3, cmap="RdYlGn", vmin=0, vmax=3)
    axes[2].set_title(f"Hoek–Brown FS at SRF = 1  "
                      f"({100*frac_all:.2f}% below 1)")
    fig.colorbar(sc2, ax=axes[2], label="$\\sigma_{1,env}/\\sigma_1$")

    for ax in axes:
        ax.set_xlabel("x (m)")
        ax.set_xlim(g.X_MIN, g.X_MAX)
        # Data reaches the natural ground above the crest, so Z_CREST as a
        # ceiling silently clips the upper slope.
        ax.set_ylim(g.Z_BASE, max(g.Z_CREST, float(Zc.max())) + 2.0)
    axes[0].set_ylabel("elevation (m)")
    fig.tight_layout()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi)
    print(f"\nwrote {a.out}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump({"ckpt": a.ckpt, "step": d.get("step"), "t_days": a.t,
                       "n": a.n, "srf": 1.0, "bishop": not a.no_bishop,
                       "frac_yielded_domain": frac_all,
                       "per_unit": rows}, f, indent=1)
        print(f"wrote {a.json}")

    print("\nNOTE: `eps_uv` = 1e-3 caps the network's displacement "
          "contribution, so |d| being\nsmall is partly the ansatz and not "
          "only the physics (open item, Days 24-26). The\nFS column does not "
          "share that caveat -- it is dominated by sigma_0, which is the\n"
          "equilibrated FE field, not the network.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())