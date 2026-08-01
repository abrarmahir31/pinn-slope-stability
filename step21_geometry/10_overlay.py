"""Digitised geometry over the source figure. Manuscript Fig. 3."""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from PIL import Image
import geometry as g

# panel (d) pixel calibration, upscaled crop
X0PX, Y400PX, Y300PX, X200PX = 3900., 638., 1866., 1460.
mx = (X0PX - X200PX) / 200.
mz = (Y300PX - Y400PX) / 100.
to_px = lambda x, z: (X0PX - x * mx, Y400PX + (400 - z) * mz)

img = Image.open("figures/fig19d_section5.png")
x = np.linspace(0, 266, 600)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 11))

a1.imshow(img)
for f, lab, c in [(g.z_ground, "ground / cut face", "#d62728"),
                  (g.z_tm_top, "top of Tm", "#1f77b4")]:
    px, py = to_px(x, f(x)); a1.plot(px, py, lw=1.3, c=c, label=lab)

# Clip each flattening ray at its daylight point -- the printed dashes stop
# where the ray emerges through the existing ground surface.
for f, lab, c in [(g.z_alt20, "20 deg (F=1.820)", "#2ca02c"),
                  (g.z_alt22, "22 deg (F=1.342)", "#9467bd"),
                  (g.z_alt25, "25 deg (F=1.059)", "#ff7f0e")]:
    xs = x[x >= 5]
    d = g.z_ground(xs) - f(xs)
    cr = np.where(np.diff(np.sign(d)))[0]
    xd = xs[cr[0]] if len(cr) else x.max()
    xc = x[x <= xd]
    px, py = to_px(xc, f(xc)); a1.plot(px, py, lw=1.3, c=c, label=lab)
zf = np.linspace(272, 358, 20)
a1.plot(*to_px(g.x_f1(zf), zf), lw=1.3, c="k", label="F1")
a1.set_xlim(0, img.width); a1.set_ylim(img.height, 0); a1.axis("off")
a1.legend(fontsize=8, loc="upper left")
a1.set_title("Digitised contacts overlaid on Fig. 19d")

rng = np.random.default_rng(0)
px_ = rng.uniform(0, 268, 80000); pz_ = rng.uniform(g.Z_BASE, 370, 80000)
tag = g.material_tag(px_, pz_)
for n, c in [("Tm", "#c8763c"), ("Mk", "#f2e394"), ("Mk_d", "#e8d16a")]:
    m = tag == n
    if m.any(): a2.scatter(px_[m], pz_[m], s=1.2, c=c, label=f"{n} (n={m.sum()})")
a2.plot(x, g.z_ground(x), "k-", lw=1.2)
a2.plot(x, g.z_tm_top(x), "k--", lw=1.0)
a2.axhline(g.Z_BASE, c="0.5", lw=0.9)
a2.axvline(g.X_MK_DIVIDE, c="r", ls=":", lw=1.0, label="X_MK_DIVIDE (assumed)")
a2.set_xlabel("x (m from toe)"); a2.set_ylabel("z (m a.s.l.)")
a2.legend(fontsize=8, markerscale=6); a2.set_title("Material tags")

plt.tight_layout(); plt.savefig("output/section5_geometry.png", dpi=200)
print("written: output/section5_geometry.png")
