r"""Fig. 7 -- PINN drawdown against the Nguyen & Raudkivi (1982) analytical
solution. SOLVER VERIFICATION, deliberately independent of the case study.

WHAT THIS IS AND IS NOT. This does NOT validate the Section 5 hydro-mechanical
model, and it does not depend on it -- no geometry, no Richards, no coal seam
in the domain, nothing from D-A.5. It is the standard verification a PINN
thesis needs: take a problem with a known closed-form solution, solve it with
the same machinery, and show the machinery is right. It answers "is the solver
correct?", which is a different question from "is the model of Isikdere
right?", and the two must be reported separately.

THE PROBLEM. A semi-infinite confined aquifer, drawdown imposed at the
excavation face at t = 0:

    ds/dt = w d2s/dx2 ,   w = T/S        (T transmissivity, S storativity)
    s(x, 0) = 0
    s(0, t) = s_w        for t > 0
    s(x -> inf, t) = 0

with the solution quoted as Eq. 1 of Ulusay et al. (2014):

    s(x, t) = s_w erfc[ x / (2 sqrt(w t)) ]

This is LINEAR. Richards is not, so a good score here does not transfer to the
hydraulic solver in `train.py`. What it does establish is that the collocation
scheme, the derivative chain through `torch.autograd`, the loss assembly and
the optimiser recover a known parabolic solution to several digits -- which is
exactly the thing a reviewer asks about when the main result is a NEGATIVE one
(D-A.4). If this figure is clean, "the model shows no transient" cannot be
dismissed as "the solver does not work".

NON-DIMENSIONALISATION, following Step 1.6's practice rather than repeating
it: X = x/L, tau = t/t_max, S* = s/s_w gives

    dS*/dtau = Pi d2S*/dX2 ,   Pi = w t_max / L^2

and Pi is reported at startup. If Pi is far from O(1) the scaling is wrong and
the result should not be trusted, the same criterion as everywhere else in
this project.

PARAMETERS, AND AN HONEST TENSION IN THE SOURCE. The paper gives the coal
seam K = 0.1-0.5 m/day from Packer tests and S = 0.00003 (Lohman 1972), which
at b = 30 m gives w = T/S = 1e5 to 5e5 m^2/day. But it also states that
significant drawdown occurs within about 100 m of the face in 24 h, which
implies 2 sqrt(w * 1 day) ~ 100-200 m, i.e. w ~ 2.5e3 to 1e4. Those two
readings differ by two orders. The default below follows the OBSERVED
behaviour (`--w 5000`) because that is what Fig. 10a of the paper depicts;
`--w` overrides. For VERIFICATION purposes the choice is immaterial -- the
PINN and the analytical solution are compared at the same w -- but the number
must not be quoted as a site parameter without resolving the tension.

    PYTHONPATH=. python scripts/verify_nguyen_raudkivi.py
    PYTHONPATH=. python scripts/verify_nguyen_raudkivi.py --epochs 6000 --w 3e5
    PYTHONPATH=. python scripts/verify_nguyen_raudkivi.py --out docs/fig7.png
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def analytic(x, t, w, s_w):
    """s(x,t) = s_w erfc[x / (2 sqrt(w t))]. At t = 0 the limit is 0 for
    x > 0 and s_w at x = 0; returned as 0 so the IC is single-valued away
    from the corner."""
    x = np.asarray(x, float)
    t = np.asarray(t, float)
    out = np.zeros(np.broadcast(x, t).shape, float)
    live = t > 0
    arg = np.zeros_like(out)
    arg[live] = x[live] / (2.0 * np.sqrt(w * t[live]))
    from scipy.special import erfc
    out[live] = s_w * erfc(arg[live])
    return out


class Net(nn.Module):
    """Small tanh MLP on (X, tau) in [0,1]^2, mapped to [-1,1] on entry for
    the same reason src.model.PINN does it: tanh saturates past |x| ~ 3 and a
    raw coordinate kills the derivative pathway."""

    def __init__(self, width=48, depth=4):
        super().__init__()
        layers, d = [], 2
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, X, tau):
        z = torch.cat([2.0 * X - 1.0, 2.0 * tau - 1.0], dim=1)
        return self.net(z)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--w", type=float, default=5000.0,
                    help="T/S in m^2/day (default 5000; see docstring)")
    ap.add_argument("--s-w", type=float, default=20.0,
                    help="drawdown at the face, m (paper: ~20 m in ~1 week)")
    ap.add_argument("--h0", type=float, default=30.0,
                    help="initial piezometric level, m (Fig. 10a)")
    ap.add_argument("--L", type=float, default=2500.0,
                    help="SOLVE domain length, m. Must be well beyond the "
                         "diffusion length 2*sqrt(w*t_max), or the far-field "
                         "condition s=0 at X=1 is FALSE and contaminates the "
                         "interior -- that error dominated the first draft "
                         "(R^2 0.94 at L=600, where 2 sqrt(w t_max) = 775 m). "
                         "Plotting is clipped by --x-plot.")
    ap.add_argument("--x-plot", type=float, default=600.0,
                    help="x range shown in the figure, m (paper Fig. 10a)")
    ap.add_argument("--t-max", type=float, default=30.0, help="days")
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--n-pde", type=int, default=4000)
    ap.add_argument("--n-bc", type=int, default=600)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="docs/fig7.png")
    ap.add_argument("--json", default="docs/verify_nguyen_raudkivi.json")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dt = torch.float64
    torch.set_default_dtype(dt)

    Pi = a.w * a.t_max / (a.L ** 2)
    diff_len_1d = 2.0 * math.sqrt(a.w * 1.0)
    diff_len_max = 2.0 * math.sqrt(a.w * a.t_max)
    print(f"w = {a.w:.4g} m^2/day   s_w = {a.s_w:g} m   h0 = {a.h0:g} m")
    print(f"L = {a.L:g} m   t_max = {a.t_max:g} d")
    print(f"Pi = w t_max / L^2 = {Pi:.4g}   "
          f"(O(1) is the target; far from it means the scaling is wrong)")
    print(f"diffusion length 2*sqrt(w t) at t = 1 d: {diff_len_1d:.1f} m "
          f"(paper describes ~100 m)")
    print(f"                            at t = t_max: {diff_len_max:.1f} m "
          f"-- L/that = {a.L/diff_len_max:.2f} (want >= 2.5)")
    if a.L < 2.5 * diff_len_max:
        print("   WARNING: L is too short. The far-field condition s=0 at X=1 "
              "is FALSE\n   at this L and will contaminate the interior. "
              "Raise --L.")
    print(f"implied T = w*S at S = 3e-5: {a.w * 3e-5:.4g} m^2/day  "
          f"-> K = {a.w * 3e-5 / 30.0:.4g} m/day at b = 30 m "
          f"(Packer gave 0.1-0.5)\n")

    net = Net(a.width, a.depth)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    # Collocation, resampled once up front (fixed draw: this is a verification,
    # and a moving draw would confound the convergence history).
    Xc = torch.rand(a.n_pde, 1, requires_grad=True)
    Tc = torch.rand(a.n_pde, 1, requires_grad=True)
    # IC: tau = 0, s = 0 for all X > 0.
    Xi = torch.rand(a.n_bc, 1)
    Ti = torch.zeros_like(Xi)
    # Face BC: X = 0, s = s_w, for tau > 0.
    Tf = torch.rand(a.n_bc, 1)
    Xf = torch.zeros_like(Tf)
    # Far field: X = 1, s ~ 0 (exact only as L -> inf; the residual there is
    # reported so the truncation is visible rather than assumed).
    Tinf = torch.rand(a.n_bc, 1)
    Xinf = torch.ones_like(Tinf)

    hist = []
    for ep in range(a.epochs + 1):
        S = net(Xc, Tc)
        S_t, = torch.autograd.grad(S.sum(), Tc, create_graph=True)
        S_x, = torch.autograd.grad(S.sum(), Xc, create_graph=True)
        S_xx, = torch.autograd.grad(S_x.sum(), Xc, create_graph=True)
        r = S_t - Pi * S_xx
        L_pde = r.pow(2).mean()
        L_ic = net(Xi, Ti).pow(2).mean()
        L_bc = (net(Xf, Tf) - 1.0).pow(2).mean()
        L_inf = net(Xinf, Tinf).pow(2).mean()
        loss = L_pde + 10.0 * (L_ic + L_bc) + L_inf

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if ep % 200 == 0:
            hist.append({"epoch": ep, "pde": float(L_pde), "ic": float(L_ic),
                         "bc": float(L_bc), "inf": float(L_inf),
                         "total": float(loss)})
            print(f"  ep {ep:>5}  pde {float(L_pde):.3e}  ic {float(L_ic):.3e}"
                  f"  bc {float(L_bc):.3e}  far {float(L_inf):.3e}")

    # -- comparison ---------------------------------------------------------
    net.eval()
    DIST = [50.0, 150.0, 300.0]          # three distances
    TIMES = [1.0, 7.0, 30.0]             # three times
    pairs = []
    for xd in DIST:
        for td in TIMES:
            Xq = torch.tensor([[xd / a.L]])
            Tq = torch.tensor([[td / a.t_max]])
            with torch.no_grad():
                s_pinn = float(net(Xq, Tq)) * a.s_w
            s_true = float(analytic(np.array([xd]), np.array([td]),
                                    a.w, a.s_w)[0])
            pairs.append({"x_m": xd, "t_d": td,
                          "s_pinn": s_pinn, "s_analytic": s_true,
                          "h_pinn": a.h0 - s_pinn,
                          "h_analytic": a.h0 - s_true})

    # Dense grid error, which is the number that should be quoted.
    xg = np.linspace(0.0, a.x_plot, 160)
    tg = np.linspace(0.05, a.t_max, 60)
    XX, TT = np.meshgrid(xg, tg)
    with torch.no_grad():
        Sg = net(torch.tensor(XX.ravel() / a.L).reshape(-1, 1),
                 torch.tensor(TT.ravel() / a.t_max).reshape(-1, 1)
                 ).numpy().reshape(XX.shape) * a.s_w
    Sa = analytic(XX, TT, a.w, a.s_w)
    err = Sg - Sa
    rmse = float(np.sqrt((err ** 2).mean()))
    maxe = float(np.abs(err).max())
    denom = float(((Sa - Sa.mean()) ** 2).sum())
    r2 = 1.0 - float((err ** 2).sum()) / denom if denom > 0 else float("nan")

    # The corner (x -> 0, t -> 0) is a genuine DISCONTINUITY of the problem:
    # s jumps 0 -> s_w there. Every gridded method smears it and a PINN is no
    # exception. Reporting a max error dominated by that corner says nothing
    # about the solver, so both numbers are given and the excluded region is
    # stated rather than quietly trimmed.
    corner = (XX < 0.02 * a.L) & (TT < 0.05 * a.t_max)
    err_in = err[~corner]
    rmse_in = float(np.sqrt((err_in ** 2).mean()))
    maxe_in = float(np.abs(err_in).max())

    print(f"\nover the dense grid ({Sg.size} points):")
    print(f"   RMSE      {rmse:.4e} m   ({100*rmse/a.s_w:.3f}% of s_w)")
    print(f"   max error {maxe:.4e} m   ({100*maxe/a.s_w:.3f}% of s_w)")
    print(f"   R^2       {r2:.6f}")
    print(f"excluding the (x<{0.02*a.L:.0f} m, t<{0.05*a.t_max:.1f} d) corner, "
          f"where s jumps 0 -> s_w\nand the solution is genuinely "
          f"discontinuous ({100*corner.mean():.1f}% of points):")
    print(f"   RMSE      {rmse_in:.4e} m   ({100*rmse_in/a.s_w:.3f}% of s_w)")
    print(f"   max error {maxe_in:.4e} m   ({100*maxe_in/a.s_w:.3f}% of s_w)")
    print("\nat the nine Fig. 7 points:")
    print(f"{'x (m)':>8}{'t (d)':>8}{'s PINN':>11}{'s exact':>11}{'err':>11}")
    for p in pairs:
        print(f"{p['x_m']:>8.0f}{p['t_d']:>8.0f}{p['s_pinn']:>11.4f}"
              f"{p['s_analytic']:>11.4f}{p['s_pinn']-p['s_analytic']:>11.2e}")

    # -- figure -------------------------------------------------------------
    fig = plt.figure(figsize=(11.5, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.30)

    ax0 = fig.add_subplot(gs[0, 0])
    for td, c in zip(TIMES, ("#1f77b4", "#ff7f0e", "#2ca02c")):
        sa = analytic(xg, np.full_like(xg, td), a.w, a.s_w)
        with torch.no_grad():
            sp = net(torch.tensor(xg / a.L).reshape(-1, 1),
                     torch.full((len(xg), 1), td / a.t_max)).numpy().ravel() * a.s_w
        ax0.plot(xg, a.h0 - sa, color=c, lw=1.8, label=f"exact, t={td:g} d")
        ax0.plot(xg, a.h0 - sp, color=c, lw=0, marker="o", ms=2.6,
                 markevery=7, label=f"PINN, t={td:g} d")
    ax0.set_xlabel("distance from excavation face (m)")
    ax0.set_ylabel("piezometric level (m)")
    ax0.set_title("Drawdown profiles")
    ax0.grid(alpha=0.25, lw=0.5)
    ax0.legend(fontsize=7, ncol=2)

    ax1 = fig.add_subplot(gs[0, 1])
    xs = [p["h_analytic"] for p in pairs]
    ys = [p["h_pinn"] for p in pairs]
    mk = {1.0: "o", 7.0: "s", 30.0: "^"}
    for p in pairs:
        ax1.scatter(p["h_analytic"], p["h_pinn"], s=42,
                    marker=mk[p["t_d"]], facecolor="none",
                    edgecolor="#1f77b4", lw=1.3)
    lo = min(min(xs), min(ys))
    hi = max(max(xs), max(ys))
    pad = 0.05 * (hi - lo or 1.0)
    ax1.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k-", lw=1.0)
    ax1.set_xlim(lo - pad, hi + pad)
    ax1.set_ylim(lo - pad, hi + pad)
    ax1.set_aspect("equal")
    ax1.set_xlabel("analytical head (m)")
    ax1.set_ylabel("PINN head (m)")
    ax1.set_title(f"1:1,  $R^2$ = {r2:.5f}")
    ax1.grid(alpha=0.25, lw=0.5)
    for t_d, m in mk.items():
        ax1.scatter([], [], marker=m, facecolor="none", edgecolor="#1f77b4",
                    label=f"t = {t_d:g} d")
    ax1.legend(fontsize=7, loc="upper left")

    ax2 = fig.add_subplot(gs[0, 2])
    im = ax2.pcolormesh(xg, tg, err, cmap="RdBu_r",
                        vmin=-maxe, vmax=maxe, shading="auto")
    ax2.set_xlabel("distance (m)")
    ax2.set_ylabel("time (d)")
    ax2.set_title(f"error,  RMSE = {rmse:.2e} m")
    ax2.axvline(0.02 * a.L, color="k", lw=0.7, ls=":")
    fig.colorbar(im, ax=ax2, label="PINN − exact (m)")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
    print(f"\nwrote {a.out}")

    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w") as f:
            json.dump({"w": a.w, "s_w": a.s_w, "h0": a.h0, "L": a.L,
                       "t_max": a.t_max, "Pi": Pi, "epochs": a.epochs,
                       "seed": a.seed, "width": a.width, "depth": a.depth,
                       "rmse_m": rmse, "max_err_m": maxe, "r2": r2,
                       "rmse_m_excl_corner": rmse_in,
                       "max_err_m_excl_corner": maxe_in,
                       "pairs": pairs, "history": hist}, f, indent=1)
        print(f"wrote {a.json}")

    print("\nThis verifies the SOLVER on a linear parabolic problem with a\n"
          "known solution. It says nothing about the Section 5 model, and the\n"
          "two claims must be reported separately -- but it is what lets the\n"
          "negative result in D-A.4 stand as physics rather than as a\n"
          "suspected implementation failure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())