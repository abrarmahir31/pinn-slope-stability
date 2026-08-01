"""Boundary points coloured by segment. Manuscript Fig. 4."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import geometry as g
import boundaries as b

bc = b.sample_boundaries(600)
fig, ax = plt.subplots(figsize=(12, 6))

x = np.linspace(0, 266, 500)
ax.plot(x, g.z_ground(x), "k-", lw=0.8, alpha=0.4)
ax.plot(x, g.z_tm_top(x), "k--", lw=0.8, alpha=0.4)

cols = {"natural_ground": "#2ca02c", "bench": "#8c564b",
        "cut_face": "#d62728", "pit_floor": "#1f77b4",
        "base": "#7f7f7f", "far_field_f1": "#9467bd"}
for s, p in bc.items():
    ax.scatter(p[:, 0], p[:, 1], s=6, c=cols[s],
               label="%s (%d)" % (s, len(p)))

p = bc["cut_face"][::20]
n = b.outward_normal(g.z_ground, p[:, 0])
ax.quiver(p[:, 0], p[:, 1], n[:, 0], n[:, 1], color="k",
          scale=25, width=0.002, alpha=0.6)

ax.axhline(b.Z_WT, ls=":", c="b", lw=1.2,
           label="karstic water table %.0f m (below domain)" % b.Z_WT)
ax.set_xlabel("x (m from toe)")
ax.set_ylabel("z (m a.s.l.)")
ax.legend(fontsize=8, ncol=2)
ax.set_title("Section 5 boundary sampling")
plt.tight_layout()
plt.savefig("output/section5_boundaries.png", dpi=200)
print("written: output/section5_boundaries.png")
