"""Manuscript Fig. 3 -- domain, contacts, cut face, collocation cloud, IC fields."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import geometry as g
import boundaries as b

d = np.load("ic_cache.npz", allow_pickle=False)
x, z, psi0, sv, sh, tag = (d[k] for k in ("x", "z", "psi0", "sig_v", "sig_h", "tag"))
meta = json.loads(str(d["meta"]))

xs = np.linspace(g.X_MIN, g.X_MAX, 800)
zg, ztm = g.z_ground(xs), np.minimum(g.z_tm_top(xs), g.z_ground(xs))
zf = np.linspace(g.Z_BASE, b.Z_F1_TOP, 200)

# field grid, masked outside the domain
gx, gz = np.meshgrid(np.linspace(g.X_MIN, g.X_MAX, 500),
                     np.linspace(g.Z_BASE, 366.0, 500))
inside = g.inside_domain(gx.ravel(), gz.ravel()).reshape(gx.shape)


def frame(ax):
    ax.plot(xs, zg, "k-", lw=1.4, zorder=5)
    ax.plot(xs, ztm, "-", color="#b00", lw=1.2, zorder=5)
    ax.plot(g.x_f1(zf), zf, "k--", lw=1.1, zorder=5)
    ax.axhline(g.Z_BASE, color="k", lw=1.1, zorder=5)
    ax.axvline(g.X_MK_DIVIDE, color="#666", ls=":", lw=1.0, zorder=4)
    ax.set_xlim(g.X_MIN - 5, g.X_MAX + 5)
    ax.set_ylim(g.Z_BASE - 5, 370)
    ax.set_aspect("equal")
    ax.set_xlabel("x from toe (m)")
    ax.set_ylabel("elevation (m a.s.l.)")


def masked(f):
    v = np.full(gx.shape, np.nan)
    v[inside] = f(gx[inside], gz[inside])
    return v


fig, axes = plt.subplots(3, 1, figsize=(7.2, 13.5), dpi=200)

# (a) domain + stratified collocation cloud
ax = axes[0]
frame(ax)
for u, c in (("Mk", "#1f77b4"), ("Mk_d", "#d62728"), ("Tm", "#7f7f7f")):
    k = tag == u
    ax.scatter(x[k], z[k], s=1.2, c=c, alpha=0.55, lw=0,
               label=f"{u}  n={k.sum()}  ({100*meta['area_fraction'][u]:.1f}% area)")
ax.plot([0, 102.1], [271.13, 332.70], "-", color="#0a0", lw=2.0,
        label="cut face 31.06$^\\circ$", zorder=6)
ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
ax.set_title("(a) domain, contacts and stratified IC collocation cloud", fontsize=9)

# (b) psi_0
ax = axes[1]
cf = ax.contourf(gx, gz, masked(lambda a, c: b.Z_WT - c), levels=24, cmap="viridis")
frame(ax)
ax.axhline(b.Z_WT, color="c", ls="--", lw=1.2)
ax.text(5, b.Z_WT + 1.5, f"water table {b.Z_WT:.0f} m a.s.l. (below domain floor)",
        fontsize=6.5, color="c")
plt.colorbar(cf, ax=ax, label="$\\psi_0$ (m)", fraction=0.03)
ax.set_title("(b) initial pressure head -- unsaturated throughout", fontsize=9)

# (c) sigma_h, discontinuity should be visible
ax = axes[2]
cf = ax.contourf(gx, gz, masked(b.sigma_h_geostatic) / 1e6, levels=24, cmap="magma")
frame(ax)
plt.colorbar(cf, ax=ax, label="$\\sigma_h$ (MPa)", fraction=0.03)
ax.set_title(f"(c) horizontal geostatic stress, $K_0$ smoothed over "
             f"$\\pm${b.K0_SMOOTH_W:.0f} m", fontsize=9)

plt.tight_layout()
plt.savefig("figures/fig3_domain_ic.png", dpi=300, bbox_inches="tight")
print("written: figures/fig3_domain_ic.png")
print(f"z_top_domain {b.Z_F1_TOP:.2f}   psi0 {psi0.min():.2f}..{psi0.max():.2f} m")
